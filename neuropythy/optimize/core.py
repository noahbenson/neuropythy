####################################################################################################
# neuropythy/optimize/core.py
# Code for optimizing functions; made with the intention of optimizing functions on the cortical
# surface using gradient descent or a gradient-aware method.
# By Noah C. Benson

import os, gzip, types, six, abc, pimms
import numpy                 as np
import numpy.linalg          as npla
import scipy                 as sp
import scipy.sparse          as sps
import scipy.optimize        as spopt
import pyrsistent            as pyr
from   functools         import reduce
from   ..                import geometry as geo
from   ..                import mri      as mri
from   ..util            import (numel, rows, part, hstack, vstack, repmat, flatter, flattest,
                                 times, plus, minus, zdivide, zinv, power, ctimes, cpower,
                                 sine, cosine, tangent, cosecant, secant, cotangent)
from   ..geometry        import (triangle_area)

# Helper Functions #################################################################################
def fapply(f, x, tz=False):
    '''
    fapply(f,x) yields the result of applying f either to x, if x is a normal value or array, or to
      x.data if x is a sparse matrix. Does not modify x (unless f modifiex x).

    The optional argument tz (default: False) may be set to True to specify that, if x is a sparse 
    matrix that contains at least 1 element that is a sparse-zero, then f(0) should replace all the
    sparse-zeros in x (unless f(0) == 0).
    '''
    if sps.issparse(x):
        y = x.copy()
        y.data = f(x.data)
        if tz and y.getnnz() < np.prod(y.shape):
            z = f(np.array(0))
            if z != 0:
                y = y.toarray()
                y[y == 0] = z
        return y
    else: return f(x)
def finto(x, ii, n, null=0):
    '''
    finto(x,ii,n) yields a vector u of length n such that u[ii] = x.

    Notes:
      * The ii index may be a tuple (as can be passed to numpy arrays' getitem method) in order to
        specify that the specific elements of a multidimensional output be set. In this case, the
        argument n should be a tuple of sizes (a single integer is taken to be a square/cube/etc).
      * x may be a sparse-array, but in it will be reified by this function.

    The following optional arguments are allowed:
      * null (defaut: 0) specifies the value that should appear in the elements of u that are not
        set.
    '''
    x  = x.toarray() if sps.issparse(x) else np.asarray(x)
    shx = x.shape
    if isinstance(ii, tuple):
        if not pimms.is_vector(n): n = tuple([n for _ in ii])
        if len(n) != len(ii): raise ValueError('%d-dim index but %d-dim output' % (len(ii),len(n)))
        sh = n + shx[1:]
    elif pimms.is_int(ii): sh = (n,) + shx
    else:                  sh = (n,) + shx[1:]
    u = np.zeros(sh, dtype=x.dtype) if null == 0 else np.full(sh, null, dtype=x.dtype)
    u[ii] = x
    return u
def fdot(a,b,s=None):
    '''
    fdot(a,b) yields the dot product of a and b, doing so in a fashion that respects sparse matrices
      when encountered. This does not error check for bad dimensionality.
    fdot(a,b,shape) yields the dot product of a and b, interpreting vectors as either rows or
      columns in such a way that the shape of the resulting output is equal to shape. If this cannot
      be done, an exception is raised. 
    '''
    if s is None:
        if sps.issparse(a): return a.dot(b)
        elif sps.issparse(b): return b.T.dot(a.T).T
        else: return np.dot(a,b)
    else:
        a = a if sps.issparse(a) else np.squeeze(a)
        b = b if sps.issparse(b) else np.squeeze(b)
        sa = a.shape
        sb = b.shape
        (la,lb,ls) = [len(x) for x in (sa,sb,s)]
        if la == 0 or lb == 0:   z = fdot(a,b,None)
        elif la == 2 or lb == 2: z = fdot(a,b,None)
        elif la != 1 or lb != 1: raise ValueError('dot only works with tensor rank <= 2')
        elif ls == 0:            return np.dot(a,b)
        elif ls == 2:            z = fdot(np.expand_dims(a,-1), np.expand_dims(a,0))
        else: raise ValueError('dot: cannot turn %s * %s into %s' % (sa,sb,s))
        if z.shape == s: return z
        elif ls == 0 and z.shape == (1,1): return z[0,0]
        elif ls == 1 and s[0] == numel(z): return (z.toarray() if sps.issparse(z) else z).reshape(s)
        else: raise ValueError('dot: cannot turn %s * %s into %s' % (sa,sb,s))

# Potential Functions ##############################################################################
@six.add_metaclass(abc.ABCMeta)
class PotentialFunction(object):
    '''
    The PotentialFunction class is intended as the base-class for all potential functions that can
    be minimized by neuropythy. PotentialFunction is effectively an abstract class that requires its
    subclasses to implement the method __call__(), which must take one argument: a numpy vector of
    parameters. The method must return a tuple of (z,dz) where z is  the potential value for the
    given paramters and dz is the Jacobian of the function at the parameters. Note that if the
    potential z returned is a scalar, then dz must be a vector of length len(params); if z is a
    vector, then dz must be a matrix of size (len(z) x len(params))
    '''
    # The value() function should return the potential alone
    @abc.abstractmethod
    def value(self, params):
        '''
        pf.value(params) yields the potential function value at the given parameters params. Values
          must always be vectors; if the function returns a scalar, it must return a 1-element
          vector instead.
        '''
        raise RuntimeError('The value() method was not overloaded for object %s' % self)
    @abc.abstractmethod
    def jacobian(self, params, into=None):
        '''
        pf.jacobian(params) yields the potential function jacobian matrix at the given parameters
          params. The jacobian must always be a (possibly sparse) matrix; if there is a scalar
          output value for the potential function, then Jacobian should be (1 x n); if there is a
          scalar input parameter, the Jacobian should be (n x 1).

        If the optional matrix into is provided then the returned jacobian may optionally be added
        directly into this matrix and returned.
        '''
        raise RuntimeError('The gradient() method was not overloaded for object %s' % self)
    # The __call__ function is how one generally calls a potential function
    def __call__(self, params):
        '''
        pf(params) yields the tuple (z, dz) where z is the potential value at the given parameters
          vector, params, and dz is the vector of the potential gradient.
        '''
        z  = self.value(params)
        dz = self.jacobian(params)
        if sps.issparse(dz): dz = dz.toarray()
        z  = np.squeeze(z)
        dz = np.squeeze(dz)
        return (z,dz)
    def fun(self):
        '''
        pf.fun() yields a value calculation function for the given potential function pf that is
          appropriate for passing to a minimizer.
        '''
        return lambda x: np.squeeze(self.value(x))
    def jac(self):
        '''
        pf.jac() yields a jacobian calculation function for the given potential function pf that is
          appropritate for passing to a minimizer.
        '''
        def _jacobian(x):
            dz = self.jacobian(x)
            if sps.issparse(dz): dz = dz.toarray()
            dz = np.asarray(dz)
            return np.squeeze(dz)
        return _jacobian
    def minimize(self, x0, **kwargs):
        '''
        pf.minimize(x0) minimizes the given potential function starting at the given point x0; any
          additional options are passed along to scipy.optimize.minimize.
        '''
        x0 = np.asarray(x0)
        kwargs = pimms.merge({'jac':self.jac(), 'method':'BFGS'}, kwargs)
        res = spopt.minimize(self.fun(), x0.flatten(), **kwargs)
        res.x = np.reshape(res.x, x0.shape)
        return res
    def argmin(self, x0, **kwargs):
        '''
        pf.argmin(x0) is equivalent to pf.minimize(x0).x.
        '''
        return self.minimize(x0, **kwargs).x
    def maximize(self, x0, **kwargs):
        '''
        pf.maximize(x0) is equivalent to (-pf).minimize(x0).
        '''
        return (-self).minimize(x0, **kwargs)
    def argmax(self, x0, **kwargs):
        '''
        pf.argmax(x0) is equivalent to pf.maximize(x0).x.
        '''
        return self.maximize(x0, **kwargs).x
    # Arithmetic Operators #########################################################################
    def __getitem__(self, ii):
        return part(self, ii)
    def __neg__(self):
        if is_const_potential(self): return const_potential(-self.c)
        else: return PotentialTimesConstant(self, -1)
    def __add__(self, x):
        x = to_potential(x)
        if is_const_potential(x):
            if   np.isclose(x.c, 0).all(): return self
            elif is_const_potential(self): return const_potential(self.c + x.c)
            else:                          return PotentialPlusConstant(self, x.c)
        elif is_const_potential(self):     return PotentialPlusConstant(x.c, self)
        else:                              return PotentialPlusPotential(self, x)
    def __radd__(self, x):
        return self.__add__(x)
    def __sub__(self, x):
        return self.__add__(-x)
    def __rsub__(self, x):
        return PotentialPlusConstant(-self, x)
    def __mul__(self, x):
        x = to_potential(x)
        if is_const_potential(x):
            if   np.isclose(x.c, 1).all(): return self
            elif np.isclose(x.c, 0).all(): return x
            elif is_const_potential(self): return const_potential(self.c * x.c)
            else:                          return PotentialTimesConstant(self, x.c)
        elif is_const_potential(self):     return PotentialTimesConstant(x.c, self)
        else:                              return PotentialTimesPotential(self, x)
    def __rmul__(self, x):
        return (self * x)
    def __div__(self, x):
        x = to_potential(x)
        if is_const_potential(x):
            if   np.isclose(x.c, 1).all(): return self
            elif is_const_potential(self): return const_potential(self.c / x.c)
            else:                          return PotentialTimesConstant(self, 1.0/x.c)
        else:                              return PotentialTimesPotential(self, 1/x)
    def __rdiv__(self, x):
        x = to_potential(x)
        if is_const_potential(self): return x / self.c
        else:                        return x * PotentialPowerConstant(self, -1)
    def __truediv__(self, x):
        return self.__div__(x)
    def __rtruediv__(self, x):
        return self.__rdiv__(x)
    def __pow__(self, x):
        x = to_potential(x)
        if is_const_potential(x):
            if   is_const_potential(self): return const_potential(power(self.c, x.c))
            else:                          return PotentialPowerConstant(self, x.c)
        elif is_const_potential(self):     return ConstantPowerPotential(self, x)
        else:                              return PotentialPowerPotential(self, x)
    def __rpow__(self, x):
        return to_potential(x).__pow__(self)
    def compose(self, f):
        return compose(self, f)
def is_potential(f):
    '''
    is_potential(f) yields True if f is a potential function and False otherwise.
    '''
    return isinstance(f, PotentialFunction)
@pimms.immutable
class PotentialIdentity(PotentialFunction):
    '''
    PotentialIdentity is a potential function that represents the arguments given to it as outputs.
    '''
    def __init__(self): pass
    def value(self, params): return np.asarray(params)
    def jacobian(self, params, into=None):
        if into is None: into =  sps.eye(numel(params))
        else:            into += sps.eye(numel(params))
        return into
identity = PotentialIdentity()
@pimms.immutable
class PotentialLambda(PotentialFunction):
    '''
    PotentialLambda is a PotentialFunction type that takes as initialization arguments two functions
    g and h which must return the value and the jacobian, respectively.
    '''
    def __init__(self, g, h):
        self.valfn = g
        self.jacfn = h
    @pimms.param
    def valfn(v): return v
    @pimms.param
    def jacfn(j): return j
    def value(self, x): return self.valfn(x)
    def jacobian(self, x, into=None):
        try:    return self.jacfn(x, into)
        except: return self.jacfn(x)
@pimms.immutable
class PotentialConstant(PotentialFunction):
    def __init__(self, c):
        self.c = c
    @pimms.param
    def c(c0):
        c0 = c0.toarray() if sps.issparse(c0) else np.array(c0)
        if len(c0.shape) > 1: c0 = c0.flatten()
        c0.setflags(write=False)
        return c0
    def value(self, params):
        return self.c
    def jacobian(self, params, into=None):
        c = self.c
        d = 1 if len(c.shape) == 0 else c.shape[0]
        return sps.csr_matrix(([], [[],[]]), shape=(d, len(params))) if into is None else into
def is_const_potential(f):
    '''
    is_const_potential(f) yields True if f is a constant potential function and False otherwise.
    '''
    return isinstance(f, PotentialConstant)
def const_potential(f):
    '''
    const_potential(f) yields f if f is a constant potential function; if f is a constant, yields
      a potential function that always yields f; otherwise raises an error.
    '''
    if is_const_potential(f): return f
    elif pimms.is_array(f, 'number'): return PotentialConstant(f)
    else: raise ValueError('Could not convert given value to potential constant: %s' % f)
def const(f):
    '''
    const(f) yields f.c if f is a constant potential function; if f is a constant, it yields f or
      the equivalent numpy array object; if f is a potential function that is not const, or is not
      a valid potential function constant, yields None.
    '''
    if is_const_potential(f): return f.c
    elif pimms.is_array(f, 'number'):
        if sps.issparse(f): return f
        else: return np.asarray(f)
    else: return None
def to_potential(f):
    '''
    to_potential(f) yields f if f is a potential function; if f is not, but f can be converted to
      a potential function, that conversion is performed then the result is yielded.
    to_potential(Ellipsis) yields a potential function whose output is simply its input (i.e., the
      identity function).
    to_potential(None) is equivalent to to_potential(0).

    The following can be converted into potential functions:
      * Anything for which pimms.is_array(x, 'number') yields True (i.e., arrays of constants).
      * Any tuple (g, h) where g(x) yields a potential value and h(x) yields a jacobian matrix for
        the parameter vector x.
    '''
    if   is_potential(f): return f
    elif f is Ellipsis:   return identity
    elif pimms.is_array(f, 'number'): return const_potential(f)
    elif isinstance(f, tuple) and len(f) == 2: return PotentialLambda(f[0], f[1])
    else: raise ValueError('Could not convert object to potential function')
@pimms.immutable
class PotentialComposition(PotentialFunction):
    def __init__(self, g, h):
        self.g = g
        self.h = h
    @pimms.param
    def g(g0): return g0
    @pimms.param
    def h(h0): return h0
    def value(self, params):
        return self.g.value(self.h.value(params))
    def jacobian(self, params, into=None):
        zh  = self.h.value(params)
        dzh = self.h.jacobian(params)
        zg  = self.g.value(zh)
        dzg = self.g.jacobian(zh)
        if into is None: into =  fdot(dzg, dzh)
        else:            into += fdot(dzg, dzh)
        return into
def compose(*args):
    '''
    compose(g, h...) yields a potential function f that is the result of composing together all the
      arguments g, h, etc. after calling to_potential() on each. The result is defined such that
      f(x) is equivalent to g(h(...(x))).
    '''
    return reduce(lambda h,g: PotentialComposition(g,h), reversed(list(map(to_potential, args))))
@pimms.immutable
class PotentialPart(PotentialFunction):
    def __init__(self, ii, input_len=None):
        self.output_indices = ii
        self.input_len = input_len
    @pimms.param
    def output_indices(ii):
        ii = flattest(ii)
        if (np.issubdtype(ii.dtype, np.dtype('bool').type) or
            np.logical_or(ii == True, ii == False).all()):
            ii = np.where(ii)[0]
        return pimms.imm_array(ii)
    @pimms.param
    def input_len(m):
        if m is None: return m
        assert(pimms.is_int(m) and m > 0)
        return int(m)
    @pimms.value
    def jacobian_matrix(output_indices, input_len):
        m = np.max(output_indices) + 1 if input_len is None else input_len
        n = len(output_indices)
        return sps.csr_matrix((np.ones(n), (np.arange(n), output_indices)), shape=(n,m))
    def value(self, params):
        ii = self.output_indices
        return flattest(params)[ii]
    def jacobian(self, params, into=None):
        params = flattest(params)
        jm = self.jacobian_matrix
        if jm.shape[1] != len(params):
            jm = jm.copy()
            jm.reshape((j.shape[0], len(params)))
        if into is None: into =  jm
        else:            into += jm
        return into
def part(f, ii):
    '''
    part(u, ii) for constant or constant potential u yields a constant-potential form of u[ii].
    part(f, ii) for potential function f yields a potential function g(x) that is equivalent to
      f(x[ii]).
    '''
    f = to_potential(f)
    if is_const_potential(f): return ConstantPotential(f.c[ii])
    else:                     return compose(PotentialPart(ii), to_potential(f))
@pimms.immutable
class PotentialPlusPotential(PotentialFunction):
    def __init__(self, g, h):
        self.g = g
        self.h = h
    @pimms.param
    def g(g0): return g0
    @pimms.param
    def h(h0): return h0
    def value(self, params):
        return self.g.value(params) + self.h.value(params)
    def jacobian(self, params, into=None):
        dg = self.g.jacobian(params, into=into)
        if into is None: into = dg
        dh = self.h.jacobian(params, into=into)
        if   dg is into and dh is into: pass
        elif dg is into: into += dh
        elif dh is into: into += dg
        else:
            into = dg
            into += dh
        return into
@pimms.immutable
class PotentialPlusConstant(PotentialFunction):
    def __init__(self, f, c):
        self.f = f
        self.c = c
    @pimms.param
    def f(f0): return f0
    @pimms.param
    def c(c0): return c0
    def value(self,params):
        return self.f.value(params) + self.c
    def jacobian(self, params, into=None):
        return self.f.jacobian(params, into=into)
@pimms.immutable
class PotentialTimesPotential(PotentialFunction):
    def __init__(self, g, h):
        self.g = g
        self.h = h
    @pimms.param
    def g(g0): return g0
    @pimms.param
    def h(h0): return h0
    def value(self, params):
        g = self.g.value(params)
        h = self.h.value(params)
        return g * h
    def jacobian(self, params, into=None):
        g  = self.g.value(params)
        dg = self.g.jacobian(params)
        h  = self.h.value(params)
        dh = self.h.jacobian(params)
        if into is None:
            into = cplus(times(dg, h), times(dh, g))
        else:
            into += times(dg, h)
            into += times(dh, g)
        return into
    def __call__(self, params):
        g  = self.g.value(params)
        dg = self.g.jacobian(params)
        h  = self.h.value(params)
        dh = self.h.jacobian(params)
        return (g*h, cplus(times(dg, h), times(dh, g)))
@pimms.immutable
class PotentialTimesConstant(PotentialFunction):
    def __init__(self, f, c):
        self.f = f
        self.c = c
    @pimms.param
    def f(f0): return f0
    @pimms.param
    def c(c0): return c0
    def value(self, params):
        z = self.f.value(params)
        return z * self.c
    def jacobian(self, params, into=None):
        dz = self.f.jacobian(params)
        if into is None: into =  times(dz, self.c)
        else:            into += times(dz, self.c)
        return into
@pimms.immutable
class PotentialPowerConstant(PotentialFunction):
    def __init__(self, f, c):
        self.f = f
        self.c = c
    @pimms.param
    def f(f0): return f0
    @pimms.param
    def c(c0): return c0
    def value(self, params):
        z = self.f.value(params)
        return z**self.c
    def jacobian(self, params, into=None):
        z  = self.f.value(params)
        dz = self.f.jacobian(params)
        c  = self.c
        if into is None: into =  times(dz, c * z**(c-1))
        else:            into += times(dz, c * z**(c-1))
        return into
    def __call__(self, params):
        z  = self.f.value(params)
        dz = self.f.jacobian(params)
        c  = self.c
        dz = times(dz, c * z**(c-1))
        return (z, dz)
@pimms.immutable
class ConstantPowerPotential(PotentialFunction):
    def __init__(self, c, f):
        self.f = f
        self.c = c
    @pimms.param
    def f(f0): return f0
    @pimms.param
    def c(c0):
        if np.isclose(c0, 0).any(): raise ValueError('in c**f(x) c cannot be zero')
        return c0
    @pimms.value
    def log_c(c): return np.log(c)
    def value(self, params):
        z = self.f.value(params)
        return self.c**z
    def jacobian(self, params, into=None):
        z = self.value(params)
        dz = self.f.jacobian(params)
        if into is None: into =  times(dz, self.log_c * z)
        else:            into += times(dz, self.log_c * z)
        return into
def exp(x):
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(np.exp(x.c))
    else:                     return ConstantPowerPotential(np.e, x)
def exp2(x):
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(np.exp2(x.c))
    else:                     return ConstantPowerPotential(2.0, x)
@pimms.immutable
class PotentialPowerPotential(PotentialFunction):
    def __init__(self, g, h):
        self.g = g
        self.h = h
    @pimms.param
    def g(g0): return g0
    @pimms.param
    def h(h0): return h0
    def value(self, params):
        zg = self.g.value(params)
        zh = self.h.value(params)
        return zg ** zh
    def jacobian(self, params, into=None):
        zg  = self.g.value(params)
        dzg = self.g.jacobian(params)
        zh  = self.h.value(params)
        dzh = self.h.jacobian(params)
        z   = zg ** zh
        if into is None: into =  times(plus(times(dzg, zh, inv(zg)), times(dzh, np.log(zg))), z)
        else:            into += times(plus(times(dzg, zh, inv(zg)), times(dzh, np.log(zg))), z)
        return into
def power(x,y):
    x = to_potential(x)
    y = to_potential(y)
    xc = is_const_potential(x)
    yc = is_const_potential(y)
    if xc and yc: return PotentialPowerPotential(x,   y)
    elif xc:      return ConstantPowerPotential( x.c, y)
    elif yc:      return PotentialPowerConstant( x,   y.c)
    else:         return ConstantPotential(power(x.c, y.c))
def sqrt(x): return power(x, 0.5)
def cbrt(x): return power(x, 1.0/3.0)
@pimms.immutable
class PotentialLog(PotentialFunction):
    def __init__(self, f, base=None):
        self.f = f
        self.base = base
    @pimms.param
    def f(f0): return to_potential(f0)
    @pimms.param
    def base(b): return None if b is None else to_potential(b)
    def value(self, params):
        z = self.f.value(params)
        if self.base is None: return np.log(z)
        b = self.base.value(params)
        return np.log(z, b)
    def jacobian(self, params, into=None):
        z  = self.f.value(params)
        dz = self.f.jacobian(params)
        if base is None:
            dz = divide(dz, z)
        else:
            b = self.base.value(params)
            db = self.base.jacobian(params)
            logb = np.log(b)
            dz = dz / logb - times(np.log(z), db) / (b * logb * logb)
        if into is None: into =  dz
        else:            into += dz
        return into
def log(x, base=None):
    x = to_potential(x)
    xc = is_const_potential(x)
    if base is None:
        if xc: return ConstantPotential(np.log(x.c))
        else:  return PotentialLog(x)
    base = to_potential(base)
    bc = is_const_potential(base)
    if xc and bc: return ConstantPotential(np.log(x.c, b.c))
    else:         return PotentialLog(x, b)
def log2(x):  return log(x, 2)
def log10(x): return log(x, 10)
@pimms.immutable
class PotentialSum(PotentialFunction):
    def __init__(self, f, weights=None):
        self.f = f
        self.weights = None
    @pimms.param
    def f(f0): return to_potential(f0)
    @pimms.param
    def weights(w): return None if w is None else pimms.imm_array(w)
    def value(self, params):
        z = self.f.value(params)
        w = self.weights
        if w is None: return np.sum(z)
        else: return np.dot(z, w)
    def jacobian(self, params, into=None):
        dz = self.f.jacobian(params)
        w = self.weights
        if w is None: q = dz.sum(axis=0)
        else:         q = times(dz, w).sum(axis=0)
        if into is None: into =  q
        else:            into += q
        return into
def sum(x, weights=None):
    '''
    sum(x) yields either a potential-sum object if x is a potential function or the sum of x if x
      is not. If x is not a potential-field then it must be a vector.
    sum(x, weights=w) uses the given weights to produce a weighted sum.
    '''
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(np.sum(x.c))
    else:                     return PotentialSum(x, weights=weights)
@pimms.immutable
class CosPotential(PotentialFunction):
    '''
    CosPotential is a potential function that represents cos(x).
    '''
    def __init__(self): pass
    def value(self, x): return cosine(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        z = sps.diags(-sine(x))
        if into is None: into =  z
        else:            into += z
        return into
@pimms.immutable
class SinPotential(PotentialFunction):
    '''
    SinPotential is a potential function that represents sin(x).
    '''
    def __init__(self): pass
    def value(self, x): return sine(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        z = sps.diags(cosine(x))
        if into is None: into =  z
        else:            into += z
        return into
@pimms.immutable
class TanPotential(PotentialFunction):
    '''
    TanPotential is a potential function that represents tan(x).
    '''
    def __init__(self): pass
    def value(self, x): return tangent(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        z = sps.diags(secant(x)**2)
        if into is None: into =  z
        else:            into += z
        return into
@pimms.immutable
class SecPotential(PotentialFunction):
    '''
    SecPotential is a potential function that represents sec(x).
    '''
    def __init__(self): pass
    def value(self, x): return secant(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        z = sps.diags(secant(x)*tangent(x))
        if into is None: into =  z
        else:            into += z
        return into
@pimms.immutable
class CscPotential(PotentialFunction):
    '''
    CscPotential is a potential function that represents csc(x).
    '''
    def __init__(self): pass
    def value(self, x): return cosecant(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        z = sps.diags(-cosecant(x)*cotangent(x))
        if into is None: into =  z
        else:            into += z
        return into
@pimms.immutable
class CotPotential(PotentialFunction):
    '''
    CotPotential is a potential function that represents cot(x).
    '''
    def __init__(self): pass
    def value(self, x): return cotangent(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        if into is None: into =  -cosecant(x)**2
        else:            into += -cosecant(x)**2
        return into
def cos(x):
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(cosine(x.c))
    elif x is identity:       return CosPotential()
    else:                     return compose(CosPotential(), x)
def sin(x):
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(sine(x.c))
    elif x is identity:       return SinPotential()
    else:                     return compose(SinPotential(), x)
def tan(x):
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(tangent(x.c))
    elif x is identity:       return TanPotential()
    else:                     return compose(TanPotential(), x)
def sec(x):
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(secant(x.c))
    elif x is identity:       return SecPotential()
    else:                     return compose(SecPotential(), x)
def csc(x):
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(cosecant(x.c))
    elif x is identity:       return CscPotential()
    else:                     return compose(CscPotential(), x)
def cot(x):
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(cotangent(x.c))
    elif x is identity:       return CotPotential()
    else:                     return compose(CotPotential(), x)
@pimms.immutable
class ArcSinPotential(PotentialFunction):
    '''
    ArcSinPotential is a potential function that represents asin(x).
    '''
    def __init__(self): pass
    def value(self, x): return arcsine(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        z = 1.0 / np.sqrt(1.0 - x**2)
        z = sps.diags(z)
        if into is None: into =  z
        else:            into += z
        return into
@pimms.immutable
class ArcCosPotential(PotentialFunction):
    '''
    ArcCosPotential is a potential function that represents acos(x).
    '''
    def __init__(self): pass
    def value(self, x): return arccosine(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        z = -1.0 / np.sqrt(1.0 - x**2)
        z = sps.diags(z)
        if into is None: into =  z
        else:            into += z
        return into
@pimms.immutable
class ArcTanPotential(PotentialFunction):
    '''
    ArcTanPotential is a potential function that represents atan(x).
    '''
    def __init__(self): pass
    def value(self, x): return arctangent(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        z = 1.0 / (1.0 + x**2)
        z = sps.diags(z)
        if into is None: into =  z
        else:            into += z
        return into
@pimms.immutable
class ArcTan2Potential(PotentialFunction):
    '''
    ArcTan2Potential is a potential function that represents atan2(y,x).
    '''
    def __init__(self, y, x):
        self.y = y
        self.x = x
    @pimms.param
    def y(y0): return to_potential(y0)
    @pimms.param
    def x(x0): return to_potential(x0)
    def value(self, params):
        y = self.y.value(params)
        x = self.x.value(params)
        return arctangent(y, x)
    def jacobian(self, x, into=None):
        y  = self.y.value(params)
        x  = self.x.value(params)
        dy = self.y.jacobian(params)
        dx = self.x.jacobian(params)
        if   dy.shape[0] == 1 and dx.shape[0] > 1: dy = repmat(dy, dx.shape[0], 1)
        elif dx.shape[0] == 1 and dy.shape[0] > 1: dx = repmat(dx, dy.shape[0], 1)
        dz = divide(times(dy, x) - times(dx, y), np.sqrt(x**2 + y**2))
        if into is None: into =  z
        else:            into += z
        return into
def asin(x):
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(arcsine(x.c))
    elif x is identity:       return ArcSinPotential()
    else:                     return compose(ArcSinPotential(), x)
def acos(x):
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(arccosine(x.c))
    elif x is identity:       return ArcCosPotential()
    else:                     return compose(ArcCosPotential(), x)
def atan(x):
    x = to_potential(x)
    if is_const_potential(x): return ConstantPotential(arctangent(x.c))
    elif x is identity:       return ArcTanPotential()
    else:                     return compose(ArcTanPotential(), x)
def atan2(y,x): return ArcTan2Potential(y, x)
def row_norms(ii, f=Ellipsis, squared=False):
    '''
    row_norms(ii) yields a potential function h(x) that calculates the vector norms of the rows of
      the matrix formed by [x[i] for i in ii] (ii is a matrix of parameter indices).
    row_norms(ii, f) yield a potential function h(x) equivalent to compose(row_norms(ii), f).
    '''
    try:
        (n,m) = ii
        # matrix shape given
        ii = np.reshape(np.arange(n*m), (n,m))
    except: ii = np.asarray(ii)
    f = to_potential(f)
    if is_const_potential(f):
        q = flattest(f.c)
        q = np.sum([q[i]**2 for i in ii.T], axis=0)
        return ConstantPotential(q if squared else np.sqrt(q))
    F = reduce(lambda a,b: a + b, [part(..., col)**2 for col in ii.T])
    F = compose(F, f)
    if not squared: F = sqrt(F)
    return F
def col_norms(ii, f=Ellipsis, squared=False):
    '''
    col_norms(ii) yields a potential function h(x) that calculates the vector norms of the columns
      of the matrix formed by [x[i] for i in ii] (ii is a matrix of parameter indices).
    col_norms(ii, f) yield a potential function h(x) equivalent to compose(col_norms(ii), f).
    '''
    try:
        (n,m) = ii
        # matrix shape given
        ii = np.reshape(np.arange(n*m), (n,m))
    except: ii = np.asarray(ii)
    f = to_potential(f)
    if is_const_potential(f):
        q = flattest(f.c)
        q = np.sum([q[i]**2 for i in ii], axis=0)
        return ConstantPotential(q if squared else np.sqrt(q))
    F = reduce(lambda a,b: a + b, [part(..., col)**2 for col in ii])
    F = compose(F, f)
    if not squared: F = sqrt(F)
    return F
def distances(a, b, shape, squared=False, axis=1):
    '''
    distances(a, b, (n,d)) yields a potential function whose output is equivalent to the row-norms
      of reshape(a(x), (n,d)) - reshape(b(x), (n,d)).
    
    The shape argument (n,m) may alternately be a matrix of parameter indices, as can be passed to
    row_norms and col_norms.

    The following optional arguments are accepted:
      * squared (default: False) specifies whether the output should be the square distance or the
        distance.
      * axis (default: 1) specifies whether the rows (axis = 1) or columns (axis = 0) are treated
        as the vectors between which the distances should be calculated.
    '''
    a = to_potential(a)
    b = to_potential(b)
    if axis == 1: return row_norms(shape, a - b, squared=squared)
    else:         return col_norms(shape, a - b, squared=squared)
@pimms.immutable
class PotentialPiecewise(PotentialFunction):
    def __init__(self, dflt, *args):
        self.default = dflt
        self.pieces = args
    @pimms.param
    def default(d): return to_potential(d)
    @pimms.param
    def pieces(ps):
        r = []
        for p in ps:
            try:    ((mn, mx), f) = p
            except: (mn,  mx,  f) = p
            if mx < mn: raise ValueError('given piece has mn > mx: %s' % p)
            f = to_potential(f)
            r.append(((mn,mx),f))
        r = sorted(r, key=lambda x:x[0])
        for ((lmn,lmx),(umn,umx)) in zip(r[:-1],r[1:]):
            if lmx > umn: raise ValueError('pieces contain overlapping ranges')
        return tuple(r)
    @pimms.value
    def pieces_with_default(pieces, default):
        ps = list(pieces)
        ps.append(((-np.inf,np.inf), default))
        return tuple(ps)
    def value(self, params):
        params = flattest(params)
        n = len(params)
        ii = np.arange(n)
        res = np.zeros(n)
        for ((mn,mx), f) in self.pices_with_default:
            if len(ii) == 0: break
            k = np.where((x >= mn) & (x <= mx))[0]
            if len(k) == 0: continue
            kk = ii[k]
            res[kk] = f.value(params[kk])
            ii = np.delete(ii, k)
        return res
    def jacobian(self, params, into=None):
        params = flattest(params)
        n = len(params)
        ii = np.arange(n)
        (rs,cs,zs) = ([],[],[])
        for ((mn,mx), f) in self.pieces_with_default:
            if len(ii) == 0: break
            k = np.where((x >= mn) & (x <= mx))[0]
            if len(k) == 0: continue
            kk = ii[k]
            j = f.jacobian(params[kk])
            if j.shape[0] == 1 and j.shape[1] > 1: j = repmat(j, j.shape[1], 1)
            for (us,ju) in zip([rs,cs,zs], sps.find(j)): us.append(ju)
            ii = np.delete(ii, k)
        (rs,cs,zs) = [np.concatenate(us) for us in (rs,cs,zs)]
        dz = sps.csr_matrix((xs, (rs,cs)), shape=(n,n))
        if into is None: into =  dz
        else:            into += dz
        return into
@pimms.immutable
class TriangleSignedArea2DPotential(PotentialFunction):
    '''
    TriangleSignedArea2DPotential(n) yields a potential function that tracks the signed area of
    the given face count n embedded in 2 dimensions.
    The signed area is positive if the triangle is counter-clockwise and negative if the triangle is
    clockwise.
    '''
    def value(self, p):
        # transpose to be 3 x 2 x n
        p = np.transpose(np.reshape(p, (-1, 3, 2)), (1,2,0))
        # First, get the two legs...
        (dx_ab, dy_ab) = p[1] - p[0]
        (dx_ac, dy_ac) = p[2] - p[0]
        (dx_bc, dx_bc) = p[2] - p[1]
        # now, the area is half the z-value of the cross-product...
        sarea = 0.5 * (dx_ab*dy_ac - dx_ac*dy_ab)
        return sarea
    def jacobian(self, p, into=None):
        p = np.transpose(np.reshape(p, (-1, 3, 2)), (1,2,0))
        (dx_ab, dy_ab) = p[1] - p[0]
        (dx_ac, dy_ac) = p[2] - p[0]
        (dx_bc, dy_bc) = p[2] - p[1]
        z = 0.5 * np.transpose([[-dy_bc,dx_bc], [dy_ac,-dx_ac], [-dy_ab,dx_ab]], (2,0,1))
        m = numel(p)
        n = p.shape[2]
        ii = (np.arange(n) * np.ones([6, n])).T.flatten()
        z = sps.csr_matrix((z.flatten(), (ii, np.arange(len(ii)))), shape=(n, m))
        if into is None: into =  z
        else:            intp += z
        return into
def signed_face_areas(faces, axis=1):
    '''
    signed_face_areas(faces) yields a potential function f(x) that calculates the signed area of
      each face represented by the simplices matrix faces.

    If faces is None, then the parameters must arrive in the form of a flattened (n x 3 x 2) matrix
    where n is the number of triangles. Otherwise, the faces matrix must be either (n x 3) or (n x 3
    x s); if the former, each row must list the vertex indices for the faces where the vertex matrix
    is presumed to be shaped (V x 2). Alternately, faces may be a full (n x 3 x 2) simplex array of
    the indices into the parameters.

    The optional argument axis (default: 1) may be set to 0 if the faces argument is a matrix but
    the coordinate matrix will be (2 x V) instead of (V x 2).
    '''
    faces = np.asarray(faces)
    if len(faces.shape) == 2:
        if faces.shape[1] != 3: faces = faces.T
        n = 2 * (np.max(faces) + 1)
        if axis == 0: tmp = np.reshape(np.arange(n), (2,-1)).T
        else:         tmp = np.reshape(np.arange(n), (-1,2))
        faces = np.reshape(tmp[faces.flat], (-1,3,2))
    faces = faces.flatten()
    return compose(TriangleSignedArea2DPotential(), part(..., faces))
@pimms.immutable
class TriangleArea2DPotential(PotentialFunction):
    '''
    TriangleArea2DPotential(n) yields a potential function that tracks the unsigned area of the
    given number of faces.
    '''
    def value(self, p):
        # transpose to be 3 x 2 x n
        p = np.transpose(np.reshape(p, (-1, 3, 2)), (1,2,0))
        # First, get the two legs...
        (dx_ab, dy_ab) = p[1] - p[0]
        (dx_ac, dy_ac) = p[2] - p[0]
        (dx_bc, dy_bc) = p[2] - p[1]
        # now, the area is half the z-value of the cross-product...
        sarea0 = 0.5 * (dx_ab*dy_ac - dx_ac*dy_ab)
        # but we want to abs it
        return np.abs(sarea0)
    def jacobian(self, p, into=None):
        # transpose to be 3 x 2 x n
        p = np.transpose(np.reshape(p, (-1, 3, 2)), (1,2,0))
        # First, get the two legs...
        (dx_ab, dy_ab) = p[1] - p[0]
        (dx_ac, dy_ac) = p[2] - p[0]
        (dx_bc, dy_bc) = p[2] - p[1]
        # now, the area is half the z-value of the cross-product...
        sarea0 = 0.5 * (dx_ab*dy_ac - dx_ac*dy_ab)
        # but we want to abs it
        dsarea0 = np.sign(sarea0)
        z = np.transpose([[-dy_bc,dx_bc], [dy_ac,-dx_ac], [-dy_ab,dx_ab]], (2,0,1))
        z = times(0.5*dsarea0, z)
        m = numel(p)
        n = p.shape[2]
        ii = (np.arange(n) * np.ones([6, n])).T.flatten()
        z = sps.csr_matrix((z.flatten(), (ii, np.arange(len(ii)))), shape=(n, m))
        if into is None: into =  z
        else:            intp += z
        return into
def face_areas(faces, input_len=None):
    '''
    face_areas(faces) yields a potential function f(x) that calculates the unsigned area of each
      faces represented by the simplices matrix faces.

    If faces is None, then the parameters must arrive in the form of a flattened (n x 3 x 2) matrix
    where n is the number of triangles. Otherwise, the faces matrix must be either (n x 3) or (n x 3
    x s); if the former, each row must list the vertex indices for the faces where the vertex matrix
    is presumed to be shaped (V x 2). Alternately, faces may be a full (n x 3 x 2) simplex array of
    the indices into the parameters.

    The optional argument axis (default: 1) may be set to 0 if the faces argument is a matrix but
    the coordinate matrix will be (2 x V) instead of (V x 2).
    '''
    faces = np.asarray(faces)
    if len(faces.shape) == 2:
        if faces.shape[1] != 3: faces = faces.T
        n = 2 * (np.max(faces) + 1)
        if axis == 0: tmp = np.reshape(np.arange(n), (2,-1)).T
        else:         tmp = np.reshape(np.arange(n), (-1,2))
        faces = np.reshape(tmp[faces.flat], (-1,3,2))
    faces = faces.flatten()
    return compose(TriangleArea2DPotential(), part(..., faces))
