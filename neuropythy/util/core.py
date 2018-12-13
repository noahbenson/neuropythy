####################################################################################################
# neuropythy/util/core.py
# This file implements the command-line tools that are available as part of neuropythy as well as
# a number of other random utilities.

import types, inspect, atexit, shutil, tempfile, pimms, os, six
import numpy                             as np
import scipy.sparse                      as sps
import pyrsistent                        as pyr
import nibabel                           as nib
import nibabel.freesurfer.mghformat      as fsmgh
from   functools                    import reduce

if six.PY2: (_tuple_type, _list_type) = (types.TupleType, types.ListType)
else:       (_tuple_type, _list_type) = (tuple, list)

def curry(f, *args0, **kwargs0):
    '''
    curry(f, ...) yields a function equivalent to f with all following arguments and keyword
      arguments passed. This is much like the partial function, but yields a function instead of
      a partial object and thus is suitable for use with pimms lazy maps.
    '''
    def curried_f(*args, **kwargs):
        return f(*(args0 + args), **pimms.merge(kwargs0, kwargs))
    return curried_f

@pimms.immutable
class ObjectWithMetaData(object):
    '''
    ObjectWithMetaData is a class that stores a few useful utilities and the param meta_data, all of
    which assist in tracking a persistent map of meta-data with an object.
    '''
    def __init__(self, meta_data=None):
        if meta_data is None:
            self.meta_data = pyr.m()
        else:
            self.meta_data = meta_data
    @pimms.option(pyr.m())
    def meta_data(md):
        '''
        obj.meta_data is a persistent map of meta-data provided to the given object, obj.
        '''
        if md is None: return pyr.m()
        return md if pimms.is_pmap(md) else pyr.pmap(md)
    def meta(self, name):
        '''
        obj.meta(x) is equivalent to obj.meta_data.get(name, None).
        '''
        return self.meta_data.get(name, None)
    def with_meta(self, *args, **kwargs):
        '''
        obj.with_meta(...) collapses the given arguments with pimms.merge into the object's current
        meta_data map and yields a new object with the new meta-data.
        '''
        md = pimms.merge(self.meta_data, *(args + (kwargs,)))
        if md is self.meta_data: return self
        else: return self.copy(meta_data=md)
    def wout_meta(self, *args, **kwargs):
        '''
        obj.wout_meta(...) removes the given arguments (keys) from the object's current meta_data
        map and yields a new object with the new meta-data.
        '''
        md = self.meta_data
        for a in args:
            if pimms.is_vector(a):
                for u in a:
                    md = md.discard(u)
            else:
                md = md.discard(a)
        return self if md is self.meta_data else self.copy(meta_data=md)

def to_affine(aff, dims=None):
    '''
    to_affine(None) yields None.
    to_affine(data) yields an affine transformation matrix equivalent to that given in data. Such a
      matrix may be specified either as (matrix, offset_vector), as an (n+1)x(n+1) matrix, or, as an
      n x (n+1) matrix.
    to_affine(data, dims) additionally requires that the dimensionality of the data be dims; meaning
      that the returned matrix will be of size (dims+1) x (dims+1).
    '''
    if aff is None: return None
    if isinstance(aff, _tuple_type):
        # allowed to be (mtx, offset)
        if (len(aff) != 2                       or
            not pimms.is_matrix(aff[0], 'real') or
            not pimms.is_vector(aff[1], 'real')):
            raise ValueError('affine transforms must be matrices or (mtx,offset) tuples')
        mtx = np.asarray(aff[0])
        off = np.asarray(aff[1])
        if dims is not None:
            if mtx.shape[0] != dims or mtx.shape[1] != dims:
                raise ValueError('%dD affine matrix must be %d x %d' % (dims,dims,dims))
            if off.shape[0] != dims:
                raise ValueError('%dD affine offset must have length %d' % (dims,dims))
        else:
            dims = off.shape[0]
            if mtx.shape[0] != dims or mtx.shape[1] != dims:
                raise ValueError('with offset size=%d, matrix must be %d x %d' % (dims,dims,dims))
        aff = np.zeros((dims+1,dims+1), dtype=np.float)
        aff[dims,dims] = 1
        aff[0:dims,0:dims] = mtx
        aff[0:dims,dims] = off
        return pimms.imm_array(aff)
    if not pimms.is_matrix(aff, 'real'):
        raise ValueError('affine transforms must be matrices or (mtx, offset) tuples')
    aff = np.asarray(aff)
    if dims is None:
        dims = aff.shape[1] - 1
    if aff.shape[0] == dims:
        lastrow = np.zeros((1,dims+1))
        lastrow[0,-1] = 1
        aff = np.concatenate((aff, lastrow))
    if aff.shape[1] != dims+1 or aff.shape[0] != dims+1:
        arg = (dims, dims,dims+1, dims+1,dims+1)
        raise ValueError('%dD affine matrix must be %dx%d or %dx%d' % args)
    return aff

class AutoDict(dict):
    '''
    AutoDict is a handy kind of dictionary that automatically fills vivifies itself when a miss
    occurs. By default, the new value returned on miss is an AutoDict, but this may be changed by
    setting the object's on_miss() function to be something like lambda:[] (to return an empty
    list).
    '''
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.on_miss = lambda:type(self)()
    def __missing__(self, key):
        value = self.on_miss()
        self[key] = value
        return value

def simplex_summation_matrix(simplices, weight=None, inverse=False):
    '''
    simplex_summation_matrix(mtx) yields a scipy sparse array matrix that, when dotted with a
      column vector of length m (where m is the number of simplices described in the simplex matrix,
      mtx), yields a vector of length n (where n is the number of vertices in the simplex mesh); the
      returned vetor is the sum over each vertex, of the faces to which it belongs.

    The matrix mtx must be oriented such that the first dimension (rows) corresponds to the vertices
    of the simplices and the second dimension (columns) corresponds to simplices themselves.

    The optional argument weight may specify a weight for each face, in which case the summation is
    a weighted sum instead of a flat sum.

    The optional argument inverse=True may be given to indicate that the inverse summation matrix
    (summation of the vertices onto the simplices) should be returned.
    '''
    simplices = np.asarray(simplices)
    n = np.max(simplices) + 1
    (d,m) = simplices.shape
    rng = range(m)
    if inverse:
        if weight is None: f = sps.csr_matrix
        else:
            nrng = range(n)
            ww = sps.csr_matrix((weight, (nrng, nrng)), shape=(n,n), dtype=np.float)
            f = lambda *args,**kwargs: ww.dot(sps.csc_matrix(*args,**kwargs))
        s = f((np.ones(d*m, dtype=np.int),
               (np.concatenate([rng for _ in range(d)]), np.concatenate(simplices))),
              shape=(m,n),
              dtype=np.int)
    else:
        s = sps.csr_matrix(
            (np.ones(d*m, dtype=np.int),
             (np.concatenate(simplices), np.concatenate([rng for _ in range(d)]))),
            shape=(n,m),
            dtype=np.int)
        if weight is not None:
            s = s.dot(sps.csc_matrix((weight, (rng, rng)), shape=(m,m), dtype=np.float))
    return s
def simplex_averaging_matrix(simplices, weight=None, inverse=False):
    '''
    Simplex_averaging_matrix(mtx) is equivalent to simplex_simmation_matrix, except that each row of
      the matrix is subsequently normalized such that all rows sum to 1.
    
    The optional argument inverse=True may be passed to indicate that the inverse averaging matrix
    (of vertices onto simplices) should be returned.
    '''
    m = simplex_summation_matrix(simplices, weight=weight, inverse=inverse)
    rs = np.asarray(m.sum(axis=1), dtype=np.float)[:,0]
    invrs = zinv(rs)
    rng = range(m.shape[0])
    diag = sps.csr_matrix((invrs, (rng, rng)), dtype=np.float)
    return diag.dot(sps.csc_matrix(m, dtype=np.float))

def is_image(image):
    '''
    is_image(img) yields True if img is an instance if nibabe.analuze.SpatialImage and false
      otherwise.
    '''
    return isinstance(image, nib.analyze.SpatialImage)

def is_address(data):
    '''
    is_address(addr) yields True if addr is a valid address dict for addressing positions on a mesh
      or in a cortical sheet and False otherwise.
    '''
    return (isinstance(data, dict) and 'faces' in data and 'coordinates' in data)

def address_data(data, dims=None, surface=0.5, strict=True):
    '''
    address_data(addr) yields the tuple (faces, coords) of the address data where both faces and
      coords are guaranteed to be numpy arrays with sizes (3 x n) and (d x n); this will coerce
      the data found in addr if necessary to do this. If the data is not valid, then an error is
      raised. If the address is empty, this yields (None, None).

    The following options may be given:
       * dims (default None) specifies the dimensions requested for the coordinates. If 2, then
         the final dimension is dropped from 3D coordinates; if 3 then will add the optional
         surface argument as the final dimension of 2D coordinates.
       * surface (default: 0.5) specifies the surface to use for 2D addresses when a 3D address;
         is requested. If None, then an error will be raised when this condition is encountered.
         This should be either 'white', 'pial', 'midgray', or a real number in the range [0,1]
         where 0 is the white surface and 1 is the pial surface.
       * strict (default: True) specifies whether an error should be raised when there are
         non-finite values found in the faces or the coordinates matrices. These values are usually
         indicative of an attempt to address a point that was not inside the mesh/cortex.
    '''
    if data is None: return (None, None)
    if not is_address(data): raise ValueError('argument is not an address')
    faces = np.asarray(data['faces'])
    coords = np.asarray(data['coordinates'])
    if len(faces.shape) > 2 or len(coords.shape) > 2:
        raise ValueError('address data contained high-dimensional arrays')
    elif len(faces.shape) != len(coords.shape):
        raise ValueError('address data faces and coordinates are different shapes')
    elif len(faces) == 0: return (None, None)
    if len(faces.shape) == 2 and faces.shape[0] != 3: faces = faces.T
    if faces.shape[0] != 3: raise ValueError('address contained bad face matrix')
    if len(coords.shape) == 2 and coords.shape[0] not in (2,3): coords = coords.T
    if coords.shape[0] not in (2,3): raise ValueError('address coords are neither 2D nor 3D')
    if dims is None: dims = coords.shape[0]
    elif coords.shape[0] != dims:
        if dims == 2: coords = coords[:2]
        else:
            if surface is None: raise ValueError('address data must be 3D')
            elif pimms.is_str(surface):
                surface = surface.lower()
                if surface == 'pial': surface = 1
                elif surface == 'white': surface = 0
                elif surface in ('midgray', 'mid', 'middle'): surface = 0.5
                else: raise ValueError('unrecognized surface name: %s' % surface)
            if not pimms.is_real(surface) or surface < 0 or surface > 1:
                raise ValueError('surface must be a real number in [0,1]')
            coords = np.vstack((coords, np.full((1, coords.shape[1]), surface)))
    if strict:
        if np.sum(np.logical_not(np.isfinite(coords))) > 0:
            w = np.where(np.logical_not(np.isfinite(coords)))
            if len(w[0]) > 10:
                raise ValueError('address contains %d non-finite coords' % len(w[0]))
            else:
                raise ValueError('address contains %d non-finite coords (%s)' % (len(w),w))
        if np.sum(np.logical_not(np.isfinite(faces))) > 0:
            w = np.where(np.logical_not(np.isfinite(faces)))
            if len(w[0]) > 10:
                raise ValueError('address contains %d non-finite faces' % len(w[0]))
            else:
                raise ValueError('address contains %d non-finite faces (%s)' % (len(w[0]),w))
    return (faces, coords)

def unbroadcast(a, b):
    '''
    unbroadcast(a, b) yields a tuple (aa, bb) that is equivalent to (a, b) except that aa and bb
      have been reshaped such that arithmetic numpy operations such as aa * bb will result in
      row-wise operation instead of column-wise broadcasting.
    '''
    # they could be sparse:
    spa = sps.issparse(a)
    spb = sps.issparse(b)
    if   spa and spb: return (a,b)
    elif spa or  spb:
        def fix(sp,nm):
            nm = np.asarray(nm)
            dnm = len(nm.shape)
            dsp = len(sp.shape)
            # if we have (sparse matrix) * (high-dim array), unbroadcast the dense array
            if   dnm >  dsp: return unbroadcast(sp.toarray(), nm)
            elif dnm == dsp: return (sp, nm)
            else:            return (sp, np.reshape(nm, (dsp[0], 1)))
        return fix(a, b) if spa else tuple(reversed(fix(b, a)))
    # okay, no sparse matrices found:
    a = np.asarray(a)
    b = np.asarray(b)
    da = len(a.shape)
    db = len(b.shape)
    if   da > db: return (a, np.reshape(b, b.shape + tuple(np.ones(da-db, dtype=np.int))))
    elif da < db: return (np.reshape(a, a.shape + tuple(np.ones(db-da, dtype=np.int))), b)
    else:         return (a, b)
def plus(*args):
    '''
    plus(a, b...) returns the sum of all the values as a numpy array object. Unlike numpy's add
      function or a+b syntax, plus will thread over the earliest dimension possible; thus if a.shape
      a.shape is (4,2) and b.shape is 4, plus(a,b) is a equivalent to
      [ai+bi for (ai,bi) in zip(a,b)].
    '''
    n = len(args)
    if   n == 0: return np.asarray(0)
    elif n == 1: return np.asarray(args[0])
    elif n >  2: return reduce(plus, args)
    (a,b) = unbroadcast(*args)
    return a + b
def cplus(*args):
    '''
    cplus(a, b...) returns the sum of all the values as a numpy array object. Like numpy's add
      function or a+b syntax, plus will thread over the latest dimension possible.

    Additionally, cplus works correctly with sparse arrays.
    '''
    n = len(args)
    if   n == 0: return np.asarray(0)
    elif n == 1: return np.asarray(args[0])
    elif n >  2: return reduce(plus, args)
    (a,b) = args
    if np.issparse(a):
        if not np.issparse(b):
            b = np.asarray(b)
            if len(b.shape) == 0: b = np.reshape(b, (1,1))
    elif np.issparse(b):
        a = np.asarray(a)
        if len(a.shape) == 0: a = np.reshape(a, (1,1))
    else:
        a = np.asarray(a)
        b = np.asarray(b)
    return a + b
def minus(a, b):
    '''
    minus(a, b) returns the difference a - b as a numpy array object. Unlike numpy's subtract
      function or a - b syntax, minus will thread over the earliest dimension possible; this if
      a.shape is (4,2) and b.shape is 4, a - b is a equivalent to [ai-bi for (ai,bi) in zip(a,b)].
    '''
    n = len(args)
    if   n == 0: return np.asarray(0)
    elif n == 1: return np.asarray(args[0])
    elif n >  2: return reduce(plus, args)
    (a,b) = unbroadcast(*args)
    return a - b
def times(*args):
    '''
    times(a, b...) returns the product of all the values as a numpy array object. Unlike numpy's
      multiply function or a*b syntax, times will thread over the earliest dimension possible; thus
      if a.shape is (4,2) and b.shape is 4, times(a,b) is a equivalent to
      [ai*bi for (ai,bi) in zip(a,b)].
    '''
    n = len(args)
    if   n == 0: return np.asarray(0)
    elif n == 1: return np.asarray(args[0])
    elif n >  2: return reduce(plus, args)
    (a,b) = unbroadcast(*args)
    if   sps.issparse(a): return a.multiply(b)
    elif sps.issparse(b): return b.multiply(a)
    else:                 return a * b
def ctimes(*args):
    '''
    ctimes(a, b...) returns the product of all the values as a numpy array object. Like numpy's
      multiply function or a*b syntax, times will thread over the latest dimension possible; thus
      if a.shape is (4,2) and b.shape is 2, times(a,b) is a equivalent to a * b.

    Unlike numpy's multiply function, ctimes works with sparse matrices and will reify them.
    '''
    n = len(args)
    if   n == 0: return np.asarray(0)
    elif n == 1: return np.asarray(args[0])
    elif n >  2: return reduce(plus, args)
    (a,b) = args
    if   sps.issparse(a): return a.multiply(b)
    elif sps.issparse(b): return b.multiply(a)
    else:                 return np.asarray(a) * b
def inv(x):
    '''
    inv(x) yields the inverse of x, 1/x.

    Note that inv supports sparse matrices, but it is forced to reify them. Additionally, because
    inv raises an error on divide-by-zero, they are unlikely to work. For better sparse-matrix
    support, see zinv.
    '''
    if sps.issparse(x): return 1.0 / x.toarray()        
    else:               return 1.0 / np.asarray(x)
def zinv(x, null=0):
    '''
    zinv(x) yields 1/x if x is not close to 0 and 0 otherwise. Automatically threads over arrays and
      supports sparse-arrays.

    The optional argument null (default: 0) may be given to specify that zeros in the arary x should
    instead be replaced with the given value. Note that if this value is not equal to 0, then any
    sparse array passed to zinv must be reified.

    The zinv function never raises an error due to divide-by-zero; if you desire this behavior, use
    the inv function instead.
    '''
    if sps.issparse(x):
        if null != 0: return zinv(x.toarray(), null=null)
        x = x.copy()
        x.data = zinv(x.data)
        try: x.eliminate_zeros()
        except: pass
        return x
    else:
        x = np.asarray(x)
        z = np.isclose(x, 0)
        r = np.logical_not(z) / (x + z)
        if null == 0: return r
        r[z] = null
        return r
def divide(a, b):
    '''
    divide(a, b) returns the quotient a / b as a numpy array object. Unlike numpy's divide function
      or a/b syntax, divide will thread over the earliest dimension possible; thus if a.shape is
      (4,2) and b.shape is 4, divide(a,b) is a equivalent to [ai/bi for (ai,bi) in zip(a,b)].

    Note that divide(a,b) supports sparse array arguments, but if b is a sparse matrix, then it will
    be reified. Additionally, errors are raised by this function when divide-by-zero occurs, so it
    is usually not useful to use divide() with sparse matrices--see zdivide instead.
    '''
    (a,b) = unbroadcast(a,b)
    if   sps.issparse(a): return a.multiply(inv(b))
    elif sps.issparse(b): return a / b.toarray()
    else:                 return a / b
def zdivide(a, b, null=0):
    '''
    zdivide(a, b) returns the quotient a / b as a numpy array object. Unlike numpy's divide function
      or a/b syntax, zdivide will thread over the earliest dimension possible; thus if a.shape is
      (4,2) and b.shape is 4, zdivide(a,b) is a equivalent to [ai*zinv(bi) for (ai,bi) in zip(a,b)].

    The optional argument null (default: 0) may be given to specify that zeros in the arary b should
    instead be replaced with the given value in the result. Note that if this value is not equal to
    0, then any sparse array passed as argument b must be reified.

    The zdivide function never raises an error due to divide-by-zero; if you desire this behavior,
    use the divide function instead.
    '''
    (a,b) = unbroadcast(a,b)
    b = zinv(b, null=null)
    if   sps.issparse(a): return a.multiply(b)
    elif sps.issparse(b): return b.multiply(a)
    else:                 return a * b
def power(a,b):
    '''
    power(a,b) is equivalent to a**b except that, like the neuropythy.util.times function, it
      threads over the earliest dimension possible rather than the latest, as numpy's power function
      and ** syntax do. The power() function also works with sparse arrays; though it must reify
      them during the process.
    '''
    if sps.issparse(a): a = a.toarray()
    if sps.issparse(b): b = b.toarray()
    (a,b) = unbroadcast(a,b)
    return a ** b
def cpower(a,b):
    '''
    cpower(a,b) is equivalent to a**b except that it also operates over sparse arrays; though it
    must reify them to do so.
    '''
    if sps.issparse(a): a = a.toarray()
    if sps.issparse(b): b = b.toarray()
    return a ** b

_default_rtol = inspect.getargspec(np.isclose)[3][0]
_default_atol = inspect.getargspec(np.isclose)[3][1]
def replace_close(x, xhat, rtol=_default_rtol, atol=_default_atol, copy=True):
    '''
    replace_close(x, xhat) yields x if x is not close to xhat and xhat otherwise. Closeness is
      determined by numpy's isclose(), and the atol and rtol options are passed along.

    The x and xhat arguments may be lists/arrays.

    The optional argument copy may also be set to False to chop x in-place.
    '''
    x = np.array(x) if copy else np.asarray(x)
    w = np.isclose(x, xhat, rtol=rtol, atol=atol)
    x[w] = np.asarray(xhat)[w]
    return x
def chop(x, rtol=_default_rtol, atol=_default_atol, copy=True):
    '''
    chop(x) yields x if x is not close to round(x) and round(x) otherwise. Closeness is determined
      by numpy's isclose(), and the atol and rtol options are passed along.

    The x and xhat arguments may be lists/arrays.

    The optional argument copy may also be set to False to chop x in-place.
    '''
    return replace_close(x, np.round(x), rtol=rtol, atol=atol, copy=copy)

def library_path():
    '''
    library_path() yields the path of the neuropythy library.
    '''
    return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

@pimms.immutable
class CurveSpline(ObjectWithMetaData):
    '''
    CurveSpline is an immutable class for tracking curve objects produced using scipy.interpolate's
    spl* functions. Removes a lot of the confusion around these functions and manages data/function
    calls for the curves. CurveSpline is a pimms immutable class, but should generally be created
    via the curve_spline() function.
    '''
    def __init__(self, x, y=None,
                 order=3, weights=None, smoothing=None, periodic=False,
                 distances=None,
                 meta_data=None):
        ObjectWithMetaData.__init__(self, meta_data=meta_data)
        x = np.asarray(x)
        if y is not None: x = np.asarray([x,y])
        self.coordinates = x
        self.order = order
        self.weights = weights
        self.smoothing = smoothing
        self.periodic = periodic
        self.distances = distances
    @pimms.param
    def coordinates(x):
        'curve.coordinates is the seed coordinate matrix for the given curve.'
        x = np.asarray(x)
        assert(len(x.shape) == 2)
        if x.shape[0] != 2: x = x.T
        assert(x.shape[0] == 2)
        return pimms.imm_array(x)
    @pimms.param
    def order(o):
        'curve.degree is the degree of the interpolating splines for the given curv.'
        assert(pimms.is_int(o) and o >= 0)
        return o
    @pimms.param
    def smoothing(s):
        'curve.smoothing is the amount of smoothing passed to splrep for the given curve.'
        if s is None: return None
        assert(pimms.is_number(s) and s >= 0)
        return s
    @pimms.param
    def weights(w):
        'curve.weights are the weights passed to splrep for a given curve.'
        if w is None: return None
        w = pimms.imm_array(w)
        assert(pimms.is_vector(w, 'number'))
        return w
    @pimms.param
    def periodic(p):
        'curve.periodic is True if the given curve is a periodic curve and False otherwise.'
        assert(p is True or p is False)
        return p
    @pimms.param
    def distances(ds):
        'curve.distances is the specified curve-distances between points in the given curve.'
        if ds is None: return None
        ds = pimms.imm_array(ds)
        assert(pimms.is_vector(ds, 'number'))
        assert((ds > 0).all())
        return ds
    @pimms.require
    def check_distances(distances, coordinates):
        if distances is not None and len(distances) != coordinates.shape[1] - 1:
            raise ValueError('Distances must be diffs of coordinates')
        return True
    @pimms.value
    def t(distances,coordinates):
        n = coordinates.shape[1]
        if distances is None: distances = np.ones(n - 1)
        t = np.cumsum(np.pad(distances, (1,0), 'constant'))
        t.setflags(write=False)
        return t
    @pimms.value
    def splrep(coordinates, t, order, weights, smoothing, periodic):
        from scipy import interpolate
        (x,y) = coordinates
        xtck = interpolate.splrep(t, x, k=order, s=smoothing,
                                  w=weights, per=periodic)
        ytck = interpolate.splrep(t, y, k=order, s=smoothing,
                                  w=weights, per=periodic)
        return tuple([tuple([pimms.imm_array(u) for u in tck])
                      for tck in (xtck,ytck)])
    def __repr__(self):
        return 'CurveSpline(<%d points>, order=%d, %f <= t <= %f)' % (
            self.coordinates.shape[1],
            self.order, self.t[0], self.t[-1])
    def __call__(self, t, derivative=0):
        from scipy import interpolate
        xint = interpolate.splev(t, self.splrep[0], der=derivative)
        yint = interpolate.splev(t, self.splrep[1], der=derivative)
        return np.asarray([xint,yint])
    def curve_length(self, start=None, end=None, precision=0.01):
        '''
        Calculates the length of the curve by dividing the curve up
        into pieces of parameterized-length <precision>.
        '''
        if start is None: start = self.t[0]
        if end is None: end = self.t[-1]
        from scipy import interpolate
        if self.order == 1:
            # we just want to add up along the steps...
            ii = [ii for (ii,t) in enumerate(self.t) if start < t and t < end]
            ts = np.concatenate([[start], self.t[ii], [end]])
            xy = np.vstack([[self(start)], self.coordinates[:,ii].T, [self(end)]])
            return np.sum(np.sqrt(np.sum((xy[1:] - xy[:-1])**2, axis=1)))
        else:
            t = np.linspace(start, end, int(np.ceil((end-start)/precision)))
            dt = t[1] - t[0]
            dx = interpolate.splev(t, self.splrep[0], der=1)
            dy = interpolate.splev(t, self.splrep[1], der=1)
            return np.sum(np.sqrt(dx**2 + dy**2)) * dt
    def linspace(self, n=100, derivative=0):
        '''
        curv.linspace(n) yields n evenly-spaced points along the curve.
        '''
        ts = np.linspace(self.t[0], self.t[-1], n)
        return self(ts, derivative=derivative)
    def even_out(self, precision=0.001):
        '''
        Yields an equivalent curve but where the parametric value t
        is equivalent to x/y distance (up to the given precision).
        '''
        dists = [self.curve_length(s, e, precision=precision)
                 for (s,e) in zip(self.t[:-1], self.t[1:])]
        return CurveSpline(self.coordinates,
                           order=self.order,
                           weights=self.weights,
                           smoothing=self.smoothing,
                           periodic=self.periodic,
                           distances=dists,
                           meta_data=self.meta_data)
    def reverse(self):
        '''
        curve.reverse() yields the inverted spline-curve equivalent to curve.
        '''
        return CurveSpline(
            np.flip(self.coordinates, axis=1),
            distances=(None if self.distances is None else np.flip(self.distances, axis=0)),
            order=self.order, weights=self.weights, smoothing=self.smoothing,
            periodic=self.periodic, meta_data=self.meta_data)
    def subcurve(self, t0, t1):
        '''
        curve.subcurve(t0, t1) yields a curve-spline object that is equivalent to the given
          curve but that extends from curve(t0) to curve(t1) only.
        '''
        # if t1 is less than t0, then we want to actually do this in reverse...
        if t1 == t0: raise ValueError('Cannot take subcurve of a point')
        if t1 < t0:
            tt = self.curve_length()
            return self.reverse().subcurve(tt - t0, tt - t1)
        idx = [ii for (ii,t) in enumerate(self.t) if t0 < t and t < t1]
        pt0 = self(t0)
        pt1 = self(t1)
        coords = np.vstack([[pt0], self.coordinates.T[idx], [pt1]])
        ts = np.concatenate([[t0], self.t[idx], [t1]])
        dists  = None if self.distances is None else np.diff(ts)
        return CurveSpline(
            coords.T,
            order=self.order,
            smoothing=self.smoothing,
            periodic=False,
            distances=dists,
            meta_data=self.meta_data)

def curve_spline(x, y=None, weights=None, order=3, even_out=False,
                 smoothing=None, periodic=False, meta_data=None):
    '''
    curve_spline(coords) yields a bicubic spline function through
      the points in the given coordinate matrix.
    curve_spline(x, y) uses the coordinate matrix [x,y].

    The function returned by curve_spline() is f(t), defined on the
    interval from 0 to n-1 where n is the number of points in the
    coordinate matrix provided.
    
    The following options are accepted:
      * weights (None) the weights to use in smoothing.
      * smoothing (None) the amount to smooth the points.
      * order (3) the order of the polynomial used in the splines.
      * periodic (False) whether the points are periodic or not.
      * even_out (False) whether to even out the distances along
        the curve.
      * meta_data (None) an optional map of meta-data to give the
        spline representation.
    '''
    curv = CurveSpline(x,y, 
                       weights=weights, order=order,
                       smoothing=smoothing, periodic=periodic,
                       meta_data=meta_data)
    if even_out: curv = curv.even_out()
    return curv
def curve_intersection(c1, c2, grid=16):
    '''
    curve_intersect(c1, c2) yields the parametric distances (t1, t2)
      such that c1(t1) == c2(t2).
      
    The optional parameter grid may specify the number of grid-points
    to use in the initial search for a start-point (default: 20).
    '''
    from scipy.optimize import minimize
    (ts1,ts2) = [np.linspace(c.t[0], c.t[-1], grid) for c in (c1,c2)]
    (pts1,pts2) = [c(ts) for (c,ts) in zip([c1,c2],[ts1,ts2])]
    ds = np.sqrt([np.sum((pts2.T - pp)**2, axis=1) for pp in pts1.T])
    (ii,jj) = np.unravel_index(np.argmin(ds), ds.shape)
    (t01,t02) = (ts1[ii], ts2[jj])
    def f(t): return np.sum((c1(t[0]) - c2(t[1]))**2)
    (t1,t2) = minimize(f, (t01, t02)).x
    return (t1,t2)

class DataStruct(object):
    '''
    A DataStruct object is an immutable map-like object that accepts any number of kw-args on input
    and assigns all of them as members which are then immutable.
    '''
    def __init__(self, **kw):    self.__dict__.update(kw)
    def __setattr__(self, k, v): raise ValueError('DataStruct objects are immutable')
    def __delattr__(self, k):    raise ValueError('DataStruct objects are immutable')
def data_struct(*args, **kw):
    '''
    data_struct(args...) collapses all arguments (which must be maps) and keyword arguments
      right-to-left into a single mapping and uses this mapping to create a DataStruct object.
    '''
    m = pimms.merge(*args, **kw)
    return DataStruct(**m)

def tmpdir(prefix='npythy_tempdir_', delete=True):
    '''
    tmpdir() creates a temporary directory and yields its path. At python exit, the directory and
      all of its contents are recursively deleted (so long as the the normal python exit process is
      allowed to call the atexit handlers).
    tmpdir(prefix) uses the given prefix in the tempfile.mkdtemp() call.
    
    The option delete may be set to False to specify that the tempdir should not be deleted on exit.
    '''
    path = tempfile.mkdtemp(prefix=prefix)
    if not os.path.isdir(path): raise ValueError('Could not find or create temp directory')
    if delete: atexit.register(shutil.rmtree, path)
    return path
