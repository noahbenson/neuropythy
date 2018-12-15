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
from   ..                import geometry as geo
from   ..                import mri      as mri
from   ..util            import (numel, rows, part, hstack, vstack, repmat,
                                 times, plus, minus, zdivide, zinv, power, ctimes, cpower)
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
def dot(a,b,s=None):
    '''
    dot(a,b) yields the dot product of a and b, doing so in a fashion that respects sparse matrices
      when encountered. This does not error check for bad dimensionality.
    dot(a,b,shape) yields the dot product of a and b, interpreting vectors as either rows or
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
        if la == 0 or lb == 0:   z = dot(a,b,None)
        elif la == 2 or lb == 2: z = dot(a,b,None)
        elif la != 1 or lb != 1: raise ValueError('dot only works with tensor rank <= 2')
        elif ls == 0:            return np.dot(a,b)
        elif ls == 2:            z = dot(np.expand_dims(a,-1), np.expand_dims(a,0))
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
        return PotentialSubselection(self, ii)
    def __neg__(self):
        return PotentialTimesConstant(self, -1)
    def __add__(self, x):
        if isinstance(x, PotentialFunction): return PotentialPlusPotential(self, x)
        elif np.isclose(x, 0).all():         return self
        else:                                return PotentialPlusConstant(self, x)
    def __radd__(self, x):
        if np.isclose(x, 0): return self
        else:                return PotentialPlusConstant(self, x)
    def __sub__(self, x):
        return self.__add__(-x)
    def __rsub__(self, x):
        return PotentialPlusConstant(PotentialTimesConstant(self, -1), x)
    def __mul__(self, x):
        if isinstance(x, PotentialFunction): return PotentialTimesPotential(self, x)
        elif np.isclose(x, 1).all():         return self
        else:                                return PotentialTimesConstant(self, x)
    def __rmul__(self, x):
        if np.isclose(x, 1): return self
        else:                return PotentialTimesConstant(self, x)
    def __div__(self, x):
        return self.__mul__(1/x)
    def __rdiv__(self, x):
        return PotentialPowerConstant(self, -1) * x
    def __truediv__(self, x):
        return self.__mul__(1/x)
    def __rtruediv__(self, x):
        return PotentialPowerConstant(self, -1) * x
    def __pow__(self, x):
        if isinstance(x, PotentialFunction): return PotentialPowerPotential(self, x)
        else:                                return PotentialPowerConstant(self, x)
    def __rpow__(self, x):
        return ConstantPowerPotential(x, self)
    def log(self, base=None):
        if base is None:
            return PotentialLog(self)
        elif isinstance(base, PotentialFunction):
            return PotentialTimesPotential(PotentialLog(self), 1.0/PotentialLog(base))
        else:
            return PotentialTimesConstant(PotentialLog(self), 1.0/np.log(base))
    def sqrt(self):
        return PotentialPowerConstant(self, 0.5)
    def compose(self, f):
        return PotentialComposition(self, f)
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
def to_potential(f):
    '''
    to_potential(f) yields f if f is a potential function; if f is not, but f can be converted to
      a potential function, that conversion is performed then the result is yielded.

    The following can be converted into potential functions:
      * Anything for which pimms.is_array(x, 'number') yields True (i.e., arrays of constants).
      * Any tuple (g, h) where g(x) yields a potential value and h(x) yields a jacobian matrix for
        the parameter vector x.
    '''
    if   is_potential(f): return f
    elif pimms.is_array(f, 'number'): return PotentialConstant(f)
    elif isinstance(f, tuple) and len(f) == 2: return PotentialLambda(f[0], f[1])
    else: raise ValueError('Could not convert object to potential function')
@pimms.immutable
class PotentialSubselection(PotentialFunction):
    def __init__(self, f, ii, jj=None, output_len=None):
        self.f = f
        self.input_indices  = ii
        self.output_indices = jj
        self.output_length  = output_len
    @pimms.param
    def f(f0): return f0
    @pimms.param
    def input_indices(ii):
        ii = np.asarray(ii)
        if (np.issubdtype(ii.dtype, np.dtype('bool').type) or
            (len(ii) > 2 and np.logical_or(ii == True, ii == False).all())):
            ii = np.where(ii)[0]
        return pimms.imm_array(ii)
    @pimms.param
    def output_indices(jj):
        if jj is None: return None
        ii = np.asarray(jj)
        if (np.issubdtype(ii.dtype, np.dtype('bool').type) or
            (len(ii) > 2 and np.logical_or(ii == True, ii == False).all())):
            ii = np.where(ii)[0]
        return pimms.imm_array(ii)
    @pimms.param
    def output_length(ol):
        if ol is None: return None
        elif ol < 1: raise ValueError('output_len must be > 0')
        return int(ol)
    def value(self, params):
        params = np.asarray(params)
        ii = self.input_indices
        if ii is not None: params = params[ii]
        z = self.f.value(params)
        if self.output_indices is None: return z
        else: return finto(z, self.output_indices, self.output_length)
    def jacobian(self, params, into=None):
        params = np.asarray(params)
        ii = self.input_indices
        if ii is not None: params = params[ii]
        if into is None: into = np.zeros([self.output_length, len(params)])
        sh = np.shape(into)
        if sh[1] != len(params) or sh[0] != self.output_length:
            raise ValueError('bad jacobian into size given to subselection potential')
        subinto = into[self.output_indices, self.input_indices]
        tmp = self.f.jacobian(params, into=subinto)
        if tmp is not subinto: subinto += tmp
        return into
def subpotential(f, ii, jj=None, output_len=None):
    '''
    subpotential(u, ii), for vector u, yields u.
    subpotential(f, ii), for potential function f, yields a potential function like f, but that 
      operates only over the subset ii of its parameters.
    subpotential(u, ii, jj, n), for vector u, yields a vector v of length n such that v[jj] == u.
    subpotential(f, ii, jj, n), for potential function f, yields a potential function g that is like
      f but that, when given the argument x, operates only over x[ii] as its inputs and yields a
      vector v of length n such that v[jj] == f(x[ii]).
    '''
    if is_potential(f): return PotentialSubselection(f, ii, jj=jj, output_len=output_len)
    elif jj is None:    return f
    else:               return finto(f, jj, output_len)
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
        dh = self.h.jacobian(params, into=into)
        if into is None:                into = cplus(dg, dh)
        elif dg is into and dh is into: pass
        elif dh is into:                into += dg
        elif dg is into:                into += dh
        else:                           into = cplus(dg, dh)
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
    def __call__(self, params):
        z = self.value(params)
        dz = self.f.jacobian(params)
        dz = times(dz, self.log_c * z)
        return (z, dz)
def exp(x):
    if is_potential(x): return ConstantPowerPotential(np.e, x)
    else: return np.exp(x)
def exp2(x):
    if is_potential(x): return ConstantPowerPotential(2, x)
    else: return np.exp2(x)
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
    def __call__(self, params):
        zg  = self.g.value(params)
        dzg = self.g.jacobian(params)
        zh  = self.h.value(params)
        dzh = self.h.jacobian(params)
        z  = zg ** zh
        dz = times(plus(times(dzg, zh, inv(zg)), times(dzh, np.log(zg))), z)
        return (z,dz)
def power(x,y):
    if is_potential(x):
        if is_potential(y): return PotentialPowerPotential(x, y)
        else:               return PotentialPowerConstant(x, y)
    elif is_potential(y):   return ConstantPowerPotential(x, y)
    else:                   return np.pow(x,y)
def sqrt(x): return power(x, 0.5)
@pimms.immutable
class PotentialLog(PotentialFunction):
    def __init__(self, f):
        self.f = f
    @pimms.param
    def f(f0): return f0
    def value(self, params):
        return np.log(self.f.value(params))
    def jacobian(self, params, into=None):
        z  = self.f.value(params)
        dz = self.f.jacobian(params)
        if into is None: into =  times(dz, inv(z))
        else:            into += times(dz, inv(z))
        return into
    def __call__(self, params):
        z  = self.f.value(params)
        dz = self.f.jacobian(params)
        return (z, times(dz, inv(z)))
def log(x, base=None):
    if is_potential(x): return PotentialLog(x, base=base)
    else:               return np.log(x, base)
def log2(x):  return log(x,2)
def log10(x): return log(x,10)
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
        if into is None: into =  dzg.dot(dzh)
        else:            into += dzg.dot(dzh)
        return into
    def __call__(self, params):
        zh  = self.h.value(params)
        dzh = self.h.jacobian(params)
        zg  = self.g.value(zh)
        dzg = self.g.jacobian(zh)
        return (zg, dzg.dot(dzh))
def compose(*args):
    '''
    compose(g, h...) yields a potential function f that is the result of composing together all the
      arguments g, h, etc. after calling to_potential() on each. The result is defined such that
      f(x) is equivalent to g(h(...(x))).
    '''
    return reduce(lambda h,g: PotentialComposition(g,h), reversed(map(to_potential, args)))
@pimms.immutable
class PotentialSum(PotentialFunction):
    def __init__(self, f, weights=None):
        self.f = f
        self.weights = None
    @pimms.param
    def f(f0): return f0
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
    if is_potential(x):   return PotentialSum(x, weights=weights)
    elif weights is None: return np.sum(x)
    else:                 return np.dot(x, weights) / np.sum(weights)
#here #TODO -- add trig functions (sin, cos, arctan, etc)
@pimms.immutable
class SimplexPotential(PotentialFunction):
    '''
    SimplexPotential is the base-class for potentials that use simplices (i.e., subsets of the
    parameter space) to define themselves and don't want to have to deal with monitoring the
    specifics of the indexing.

    To create a SimplexPotential, the constructor must be given a matrix of simplex indices; each
    row corresponds to one simplex. Note that while the first dimension must correspond to simplex,
    the remaining dimensions may be any shape. When the new potential function object is called, it
    internally sorts the parameters into a matrix the same shape as the simplex index and then calls
    the simplex_potential() function. This function should return an array of potential values, one
    per simplex, and a gradient array the same shape as the simplex index. This gradient array is
    then sorted into a sparse Jacobian matrix, which is returned along with the vector potential
    values. To create a scalar potential, one generally uses PotentialSum() on such an object.
    '''
    def __init__(self, simplices, f, input_len=None):
        self.simplices = simplices
        self.f = f
        self.input_len = input_len
    @pimms.param
    def simplices(s):
        s = np.array(s)
        if   len(s.shape) == 0: s = np.asarray([[s]])
        elif len(s.shape) == 1: s = np.reshape(s, (s.shape[0], 1))
        s.setflags(write=False)
        return s
    @pimms.param
    def f(f0): return to_potential(f0)
    @pimms.param
    def input_len(i):
        if   i is None:       return None
        elif pimms.is_int(i): return int(i)
        else: raise ValueError('input_len must be None or an integer')
    @pimms.value
    def simplex_dimensions(simplices):
        return np.prod(simplices.shape[1:], dtype=np.int)
    @pimms.value
    def flat_simplices(simplices):
        fs = simplices.flatten()
        fs.setflags(write=False)
        return fs
    @pimms.value
    def jacobian_indices(flat_simplices, simplex_dimensions):
        (s,d) = (int(len(flat_simplices)/simplex_dimensions), simplex_dimensions)
        ii = np.tile(np.reshape(np.arange(s), (s,1)), (1,d)).flatten()
        ii.setflags(write=False)
        return (ii, flat_simplices)
    @pimms.value
    def jacobian_transform(flat_simplices, simplex_dimensions, input_len):
        n = len(flat_simplices)
        N = np.max(flat_simplices)+1 if input_len is None else input_len
        (s,d) = (int(n/simplex_dimensions), simplex_dimensions)
        return sps.csc_matrix((np.ones(n), (np.arange(n), flat_simplices)), shape=(n, N))
    def value(self, params):
        params = np.asarray(params)
        s = np.reshape(params.flat[self.flat_simplices], self.simplices.shape)
        return self.f.value(s)
    def jacobian_fix(self, dz, params):
        # two options: (1) they return an array the same shape as s of each jacobian value (but not
        #                  arranged into a proper jacobian matrix
        #              (2) they returned a proper jacobian matrix that needs to be compressed back
        #                  into the original parameter shape
        s = self.simplices
        if dz.shape == s.shape:
            dz = sps.csr_matrix((dz.flatten(), self.jacobian_indices), shape=(len(s), len(params)))
        elif pimms.is_matrix(dz):
            jtr = self.jacobian_transform
            n = numel(params)
            if jtr.shape[1] != n:
                jtr = jtr.copy()
                jtr.resize([jtr.shape[0], n])
            dz = dz.dot(jtr)
        else: raise ValueError('Could not understand jacobian of simplex potential')
        return dz
    def jacobian(self, params, into=None):
        params = np.asarray(params)
        s = np.reshape(params.flat[self.flat_simplices], self.simplices.shape)
        dz = self.f.jacobian(s)
        dz = self.jacobian_fix(dz, params)
        if into is None: into =  dz
        else:            into += dz
        return into
    def __call__(self, params):
        params = np.asarray(params)
        s = np.reshape(params.flat[self.flat_simplices], self.simplices.shape)
        z  = self.f.value(s)
        dz = self.f.jacobian(s)
        dz = self.jacobian_fix(dz, params)
        return (z, dz)
def with_simplices(simplices, f, input_len=None):
    '''
    with_simplices(simplices, f) yields a potential function that operates on the reordered
      parameters according to the simplices array.

    Essentially, if g = with_simplices(s, f), then, given a parameter vector x, g.value(x) takes x
    and rearranges it into an array of simplices s where:
      s = reshape(x[simplices.flatten()], simplices.shape);
    g then calls and returns f.value(s). When g.jacobian(x) is called, g performs a similar
    operation in which it first obtains the jacobian of f: j = f.jacobian(s), then converts the 
    matrix j from an (n x m) matrix where n is the numel(g.value(s)) and m is numel(s) to an (n x k)
    matrix where k is numel(x). 

    The simplices argument must be an array of simplex indices, each row of which row corresponds to
    one simplex. Note that while the first dimension must correspond to simplex, the remaining
    dimensions may be any shape. The jacobian matrix returned by f.jacobian(s) must jave the same
    number of columns, in the same order, as simplices.flatten(), and must have the same number of
    rows as numel(f.value(s)).

    The optional argument input_len (default: None) may be passed to specify that the function g
    should expect the given input length; otherwise, the input_len is assumed to be max(simplices);
    though note that this argument does not prevent the transformation from working--it merely saves
    time when the transformation from f.jacobian to g.jacobian can be pre-cached.
    '''
    return SimplexPotential(simplices, f, input_len=input_len)
@pimms.immutable
class SimplexReadyPotential(PotentialFunction):
    '''
    SimplexReadyPotential is an abstract potential function class that can keep track of how its
    potential value/gradient should be converted into a jacobian matrix.
    '''
    def __init__(self, sh):
        self.simplex_shape = sh
    @pimms.param
    def simplex_shape(dims):
        dims = dims if sps.issparse(dims) else np.asarray(dims)
        if   pimms.is_vector(dims, 'number'): return tuple(dims)
        elif pimms.is_array(dims,  'number'): return dims.shape
        else: raise ValueError('Cannot understand simplex_shape argument')
    @pimms.value
    def simplex_count(simplex_shape): return simplex_shape[0] if len(simplex_shape) > 0 else 1
    @pimms.value
    def simplex_size(simplex_shape): return np.prod(simplex_shape[1:], dtype=np.int)
    @pimms.value
    def simplex_sparse_indices(simplex_count, simplex_size):
        (r,c) = (simplex_count, simplex_size)
        ii = repmat(np.reshape(np.arange(r), (r,1)), 1,c)
        ii = ii.flatten()
        jj = np.arange(r*c)
        ii.setflags(write=False)
        jj.setflags(write=False)
        return (ii, jj)
    def jacobian_to_matrix(self, jac):
        (r,c) = (self.simplex_count, self.simplex_size)
        return sps.csr_matrix((jac.flatten(), self.simplex_sparse_indices), shape=(r,r*c))
@pimms.immutable
class FixedDistancePotential(SimplexReadyPotential):
    '''
    FixedDistancePotential(x0) represents the potential function that is the displacement of the
    parameter(s) from the reference value(s) x0. If x0 is a vector, then the call is equivalent to
    DistancePotential([x0]) (i.e., a 1 x n matrix). If x0 is a single value, then [[x0]] is used.
    For a matrix x0 with dimensions (n x m), then x0 is assumed to represent a set of n vectors with
    m dimensions each.

    For parameter matrix x and reference matrix x0 the potential is:
      np.sqrt(np.sum((np.reshape(x, x0.shape) - x0)**2, axis=1))
    And the gradient is is the direction of greatest increase (i.e. the row-vector away from x0).
    '''
    def __init__(self, x0):
        self.reference = x0
    @pimms.param
    def reference(x0):
        x0 = x0.toarray() if sps.issparse(x0) else np.array(x0)
        if   len(x0.shape) == 0: x0 = np.asarray([[x0]])
        elif len(x0.shape) == 1: x0 = np.reshape(s, (x0.shape[0], 1))
        else:                    x0 = np.reshape(x0, (x0.shape[0], np.prod(x0.shape[1:])))
        x0.setflags(write=False)
        return x0
    @pimms.value
    def simplex_shape(reference): return reference.shape
    def value(self, params):
        ref = self.reference
        d = np.reshape(params, ref.shape) - ref
        d = np.sqrt(np.sum(d**2, axis=1))
        return dist
    def jacobian(self, params, into=None):
        ref = self.reference
        r = numel(params)
        c = ref.shape[0]
        delta = np.reshape(params, ref.shape) - self.reference
        dist = np.sqrt(np.sum(delta**2, axis=1))
        z = zdivide(delta, dist)
        z = self.jacobian_to_matrix(z)
        if into is None: into =  z
        else:            intp += z
        return into
def fixed_distance(d0, transpose=False, input_len=None):
    '''
    fixed_distance(d0) yields a potential function f such that f(x) yields the vector y such
      that y[i] == np.linalg.norm(np.reshape(x, d0.shape)[i] - d0[i]).
    
    The reference d0 must be a matrix of points where each column is a dimension and each row is a
    point.

    The jacobian matrix of a fixed_distance potential function is an (s x n) matrix where s is equal
    to len(d0) and n is equal to numel(x).

    The optional argument transpose (default: False) specifies whether the parameter vector arrives
    in the form of a flattened row-points (False) or a flattened array of column-points (True).
    Alternately, transpose may be set to a simplex array the same shape as d0.
    '''
    d0 = d0 if sps.issparse(d0) else np.asarray(d0)
    (s,d) = d0.shape
    n = s*d
    if   transpose is False: return FixedDistancePotential(d0)
    elif transpose is True:  (simplices,N) = (np.reshape(np.arange(n), (d,s)).T, n)
    else:                    (simplices,N) = (np.asarray(transpose),             input_len)
    return SimplexPotential(simplices, FixedDistancePotential(d0), input_len=N)
@pimms.immutable
@six.add_metaclass(abc.ABCMeta)
class PairedSimplexPotential(SimplexReadyPotential):
    '''
    PairedSimplexPotential(n, d) represents a potential function that operates over a set of n pairs
    of d-dimensional simplices. It defines its own value and jacobian functions, so overloading
    classes should instead define spotential() and sjacobian().
    '''
    def __init__(self, n, d):
        SimplexReadyPotential.__init(self, (n, 2, d))
    @abc.abstractmethod
    def svalue(self, ii, jj): raise NotImplementedError('svalue is abstract')
    @abc.abstractmethod
    def sjacobian(self, ii, jj): raise NotImplementedError('sjacobian is abstract')
    def value(self, params):
        params = np.reshape(params, (self.simplex_shape))
        return svalue(params[:,0], params[:,1])
    def jacobian(self, params, into=None):
        ss = self.simplex_shape
        params = params.reshape(params, ss)
        z = self.sjacobian(params[:,0], params[:,1])
        if isinstance(z, tuple) and len(z) == 2: z = np.transpose(x, (1,0,2))
        z = self.jacobian_to_matrix(z)
        if into is None: into =  z
        else:            into += z
        return into
@pimms.immutable
class PairedDifferencePotential(PairedSimplexPotential):
    '''
    PairedDifferencePotential(n) represents the potential function that is the difference
    between two sets of n values; i.e., params should be reshaped to (n x 2), an array of n
    parameter pairs, then params[:,0] - params[:,1] is returned. The gradient is is the direction
    of greatest increase of the difference between the points.
    '''
    def __init__(self, n):
        PairedSimplexPotential.__init(self, n, 1)
    def svalue(self, ii, jj):
        return (ii - jj).flatten()
    def sjacobian(self, ii, jj):
        oo = np.ones(numel(ii))
        return (oo,-oo)
def paired_difference(ii, jj, input_len=None):
    '''
    paired_difference(ii, jj) yields a potential function that calculates the difference between the
      parameters in ii and those in the jj, both of which should be vectors. In other words, for
      paired difference potential function f and parameters x, y = f(x) is equivalent to:
      y = [x[i] - x[j] for (i,j) in zip(ii,jj)].
    '''
    (ii,jj) = [np.squeeze(u) for u in (ii,jj)]
    if ii.shape != jj.shape: raise ValueError('simplex-sets must be the same shape')
    if len(ii.shape) > 1: raise ValueError('pair indices must be vectors')
    pd = PairedDifferencePotential(ii.shape[0])
    return SimplexPotential(np.transpose([ii,jj]), pd)
@pimms.immutable
class PairedMeanDifferencePotential(PairedSimplexPotential):
    '''
    PairedMeanDifferencePotential(n, d) represents the potential function that is the difference
    between the means of the parameter comprising the paired simplices of dimension d in its input
    parameters (i.e., params should be reshaped to (n x 2 x d), an array of n d-dimensional simplex
    pairs).

    For parameter matrix x and dimensionality d, the calculation is like:
      x = np.reshape(x, (-1,2,d))
      (mu1,mu2) = [np.mean(s, axis=1) for s in (x[:,0], x[:,1])]
      mu1 - mu2
    And the gradient is is the direction of greatest increase of the distance between the points.
    '''
    def __init__(self, n, d):
        PairedSimplexPotential.__init(self, n, d)
    def svalue(self, ii, jj):
        mu0 = np.mean(ii, axis=1)
        mu1 = np.mean(jj, axis=1)
        return mu0 - mu1
    def sjacobian(self, ii, jj):
        sh = ii.shape
        q = np.full(sh, 1.0 / sh[1])
        return (q,-q)
def paired_mean_difference(ii, jj, input_len=None):
    '''
    paired_mean_difference(ii, jj) yields a potential function that calculates the difference
      between the parameters in the rows of ii and those in the rows of jj. In other words, for
      paired mean difference potential function f and parameters x, y = f(x) is equivalent to:
      y = [mean(x[i]) - mean(x[j]) for (i,j) in zip(ii,jj)].
    '''
    (ii,jj) = [u if sps.issparse(u) else np.asarray(u) for u in (ii,jj)]
    if ii.shape != jj.shape: raise ValueError('simplex-sets must be the same shape')
    if len(ii.shape) > 2: (ii,jj) = [np.reshape(u, (u.shape[0],-1)) for u in (ii,jj)]
    pmd = PairedMeanDifferencePotential(ii.shape[0], ii.shape[1])
    return SimplexPotential(np.transpose([ii,jj], (1,0,2)), pmd)
@pimms.immutable
class PairedDistancePotential(PairedSimplexPotential):
    '''
    PairedDistancePotential(ii, jj) represents the potential function that is the displacement of
    the parameter(s) ii from the parameter(s) jj. Both ii and jj should be identically-shaped
    matrices of simplices.

    For parameter matrix x and flattened simplex matrices ii and jj, the potential is like:
      np.sqrt(np.sum((x[ii] - x0[jj])**2, axis=1))
    And the gradient is is the direction of greatest increase of the distance between the points.
    '''
    def __init__(self, n, d):
        PairedSimplexPotential.__init(self, n, d)
    def svalue(self, ii, jj):
        return np.sqrt(np.sum((ii - jj)**2, axis=1))
    def sjacobian(self, ii, jj):
        delta = ii - jj
        dist = np.sqrt(np.sum(delta**2, axis=1))
        dz = zdivide(delta, dist)
        return (dz, -dz)
@pimms.immutable
class TriangleSignedArea2DPotential(SimplexReadyPotential):
    '''
    TriangleSignedArea2DPotential(n) yields a potential function that tracks the signed area of
    the given face count n embedded in 2 dimensions.
    The signed area is positive if the triangle is counter-clockwise and negative if the triangle is
    clockwise.
    '''
    def __init__(self, n, ):
        SimplexReadyPotential.__init__(self, (n, 3, 2))
    def value(self, p):
        # transpose to be 3 x 2 x n
        p = np.transpose(p, (1,2,0))
        # First, get the two legs...
        (dx_ab, dy_ab) = p[1] - p[0]
        (dx_ac, dy_ac) = p[2] - p[0]
        (dx_bc, dx_bc) = p[2] - p[1]
        # now, the area is half the z-value of the cross-product...
        sarea = 0.5 * (dx_ab*dy_ac - dx_ac*dy_ab)
        return sarea
    def jacobian(self, p, into=None):
        p = np.transpose(p, (1,2,0))
        (dx_ab, dy_ab) = p[1] - p[0]
        (dx_ac, dy_ac) = p[2] - p[0]
        (dx_bc, dy_bc) = p[2] - p[1]
        z = 0.5 * np.transpose([[-dy_bc,dx_bc], [dy_ac,-dx_ac], [-dy_ab,dx_ab]], (2,0,1))
        z = self.jacobian_to_matrix(z)
        if into is None: into =  z
        else:            intp += z
        return into
def signed_face_area(faces, input_len=None):
    '''
    signed_face_area(faces) yields a potential function f(x) that calculates the signed area of each
      faces represented by the simplices matrix faces.

    The faces matrix must be (n x 3); each row must list the vertex indices for the faces where the
    vertex matrix is presumed to be shaped (V x 2). Alternately, faces may be a full (n x 3 x 2)
    simplex matrix.

    The optional argument input_len may be given to specify the expected number of vertices (not
    parameters--parameters is 2 times vertices since vertices have an x and y coordinates) of the
    expected input; if None (the default), uses max(faces)*2.
    '''
    faces = np.asarray(faces)
    if len(faces.shape) == 2:
        if faces.shape[1] != 3: faces = faces.T
        n = np.max(faces)+1 if input_len is None else input_len
        N = 2*n
        tmp = np.reshape(np.arange(N), (n,2))
        faces = np.reshape(tmp[faces.flat], (faces.shape[0],3,2))
    elif len(faces.shape) == 3 and faces.shape[1:] == (3,2):
        N = np.max(faces)+1 if input_len is None else input_len*2
        if N % 2 == 1: N += 1
        n = N/2
    else: raise ValueError('faces is not a valid 2D face simplex matrix/array')
    sa = TriangleSignedArea2DPotential(faces.shape[0])
    return SimplexPotential(faces, sa, input_len=N)
@pimms.immutable
class TriangleArea2DPotential(SimplexReadyPotential):
    '''
    TriangleArea2DPotential(n) yields a potential function that tracks the unsigned area of the
    given number of faces.
    '''
    def __init__(self, n):
        SimplexReadyPotential.__init__(self, (n,3,2))
    def value(self, p):
        # transpose to be 3 x 2 x n
        p = np.transpose(p, (1,2,0))
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
        p = np.transpose(p, (1,2,0))
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
        z = self.jacobian_to_matrix(z)
        if into is None: into =  z
        else:            intp += z
        return into
def face_area(faces, input_len=None):
    '''
    face_area(faces) yields a potential function f(x) that calculates the unsigned area of each
      faces represented by the simplices matrix faces.

    The faces matrix must be (n x 3); each row must list the vertex indices for the faces where the
    vertex matrix is presumed to be shaped (V x 2). Alternately, faces may be a full (n x 3 x 2)
    simplex matrix.

    The optional argument input_len may be given to specify the expected number of vertices (not
    parameters--parameters is 2 times vertices since vertices have an x and y coordinates) of the
    expected input; if None (the default), uses max(faces)*2.
    '''
    faces = np.asarray(faces)
    if len(faces.shape) == 2:
        if faces.shape[1] != 3: faces = faces.T
        n = np.max(faces)+1 if input_len is None else input_len
        N = 2*n
        tmp = np.reshape(np.arange(N), (n,2))
        faces = np.reshape(tmp[faces.flat], (n,3,2))
    elif len(faces.shape) == 3 and faces.shape[1:] == (3,2):
        N = np.max(faces)+1 if input_len is None else input_len*2
        if N % 2 == 1: N += 1
        n = N/2
    else: raise ValueError('faces is not a valid 2D face simplex matrix/array')
    sa = TriangleArea2DPotential(n)
    return SimplexPotential(faces, sa, input_len=N)
