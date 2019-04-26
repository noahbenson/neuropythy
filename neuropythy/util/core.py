####################################################################################################
# neuropythy/util/core.py
# This file implements the command-line tools that are available as part of neuropythy as well as
# a number of other random utilities.

import types, inspect, atexit, shutil, tempfile, importlib, pimms, os, six
import collections                       as colls
import numpy                             as np
import scipy.sparse                      as sps
import pyrsistent                        as pyr
import nibabel                           as nib
import nibabel.freesurfer.mghformat      as fsmgh
from   functools                    import reduce

if six.PY2: (_tuple_type, _list_type) = (types.TupleType, types.ListType)
else:       (_tuple_type, _list_type) = (tuple, list)

# Used by functions that pass arguments on to the isclose and related functions
default_rtol = inspect.getargspec(np.isclose)[3][0]
default_atol = inspect.getargspec(np.isclose)[3][1]

# A few functions were moved into pimms; they still appear here for compatibility
from pimms import (is_tuple, is_list, is_set, curry)

def to_hemi_str(s):
    '''
    to_hemi_str(s) yields either 'lh', 'rh', or 'lr' depending on the input s.

    The match rules for s are as follows:
      * if s is None or Ellipsis, returns 'lr'
      * if s is not a string, error; otherwise s becomes s.lower()
      * if s is in ('lh','rh','lr'), returns s
      * if s is in ('left', 'l', 'sh'), returns 'lh'
      * if s is in ('right', 'r', 'dh'), returns 'rh'
      * if s in in ('both', 'all', 'xh'), returns 'lr'
      * otherwise, raises an error
    '''
    if s is None or s is Ellipsis: return 'lr'
    if not pimms.is_str(s): raise ValueError('to_hemi_str(%s): not a string or ... or None' % s)
    s = s.lower()
    if   s in ('lh',    'rh',  'lr'): return s
    elif s in ('left',  'l',   'sh'): return 'lh'
    elif s in ('right', 'r',   'dh'): return 'rh'
    elif s in ('both',  'all', 'xh'): return 'lr'
    else: raise ValueError('Could not understand to_hemi_str argument: %s' % s)

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
    def meta(self, name, missing=None):
        '''
        obj.meta(x) is equivalent to obj.meta_data.get(name, None).
        obj.meta(x, nf) is equivalent to obj.meta_data.get(name, nf)
        '''
        return self.meta_data.get(name, missing)
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
    def normalize(self):
        '''
        obj.normalize() yields a JSON-friendly Python native data-structure (i.e., dicts, lists,
          strings, numbers) that represents the given object obj. If obj contains data that cannot
          be represented in a normalized format, raises an error.

        Note that if the object's meta_data cannot be encoded, then any part of the meta_data that
        fails is simply excluded from the normalized representation.

        This function generally shouldn't be called directly unless you plan to call
        <class>.denormalize(data) directly as well--rather, use normalize(obj) and
        denormalize(data). These latter calls ensure that the type information necessary to deduce
        the proper class's denormalize function is embedded in the data.
        '''
        params = pimms.imm_params(self)
        if 'meta_data' in params:
            md = dict(params['meta_data'])
            del params['meta_data']
            params = normalize(params)
            for k in list(md.keys()):
                if not pimms.is_str(k):
                    del md[k]
                    continue
                try: md[k] = normalize(md[k])
                except Exception: del md[k]
            params['meta_data'] = md
        else: params = normalize(params)
        return params
    @classmethod
    def denormalize(self, params):
        '''
        ObjectWithMetaData.denormalize(params) is used to denormalize an object given a mapping of
          normalized JSON parameters, as produced via obj.normalize() or normalize(obj).

        This function should generally be called by the denormalize() function rather than being
        called directly unless the data you have was produced by a call to obj.normalize() rather
        than normalize(obj).
        '''
        return self(**params)
normalize_type_key = '__type__'
def normalize(data):
    '''
    normalize(obj) yields a JSON-friendly normalized description of the given object. If the data
      cannot be normalized an error is raised.

    Any object that implements a normalize() function can be normalized, so long as the mapping 
    object returned by normalize() itself can be normalized. Note that the normalize() function
    must return a mapping object.

    Objects that can be represented as themselves are returned as themselves. Any other object will
    be represented as a map that includes the reserved key '__type__' which will map to a
    2-element list [module_name, class_name]; upon denomrlization, the module and class k are looked
    up and k.denomalize(data) is called.
    '''
    if data is None: return None
    elif pimms.is_array(data, 'complex') and not pimms.is_array(data, 'real'):
        # any complex number must be handled specially:
        return {normalize_type_key: [None, 'complex'], 're':np.real(data), 'im': np.imag(data)}
    elif is_set(data):
        # sets also have a special type:
        return {normalize_type_key: [None, 'set'], 'elements': normalize(list(data))}
    elif pimms.is_scalar(data, ('number', 'string', 'unicode', 'bool')):
        # scalars are already normalized
        return data
    elif sps.issparse(data):
        # sparse matrices always get encoded as if they were csr_matrices (for now)
        (i,j,v) = sps.find(data)
        return {normalize_type_key: [None, 'sparse_matrix'],
                'rows':i.tolist(), 'cols':j.tolist(), 'vals': v.tolist(),
                'shape':data.shape}
    elif pimms.is_map(data):
        newdict = {}
        for (k,v) in six.iteritems(data):
            if not pimms.is_str(k):
                raise ValueError('Only maps with strings for keys can be normalized')
            newdict[k] = normalize(v)
        return newdict
    elif pimms.is_array(data, ('number', 'string', 'unicode', 'bool')):
        # numpy arrays just get turned into lists
        return data.tolist() if pimms.is_nparray(data) else data
    elif data is Ellipsis:
        return {normalize_type_key: [None, 'ellipsis']}
    elif pimms.is_scalar(data):
        # we have an object of some type we don't really recognize
        try:    m = data.normalize()
        except Exception: raise ValueError('Failed to run obj.normalize() on unrecognized obj: %s' % data)
        if not pimms.is_map(m): raise ValueError('obj.normalize() returned non-map; obj: %s' % data)
        m = dict(m)
        tt = type(data)
        m[normalize_type_key] = [tt.__module__, tt.__name__]
        return m
    else:
        # we have an array/list of some kind that isn't a number, string, or boolean
        return [normalize(x) for x in data]
def denormalize(data):
    '''
    denormalize(data) yield a denormalized version of the given JSON-friendly normalized data. This
      is the inverse of the normalize(obj) function.

    The normalize and denormalize functions use the reserved keyword '__type__' along with the
    <obj>.normalize() and <class>.denormalize(data) functions to manage types of objects that are
    not JSON-compatible. Please see help(normalize) for more details.
    '''
    if   data is None: return None
    elif pimms.is_scalar(data, ('number', 'bool', 'string', 'unicode')): return data
    elif pimms.is_map(data):
        # see if it's a non-native map
        if normalize_type_key in data:
            (mdl,cls) = data[normalize_type_key]
            if mdl is None:
                if   cls == 'ellipsis': return Ellipsis
                elif cls == 'complex':  return np.array(data['re']) + 1j*np.array(data['im'])
                elif cls == 'set':      return set(denormalize(data['elements']))
                elif cls == 'sparse_matrix':
                    return sps.csr_matrix((data['vals'], (data['rows'],data['cols'])),
                                          shape=data['shape'])
                else: raise ValueError('unrecognized builtin denormalize class: %s' % cls)
            else:
                cls = getattr(importlib.import_module(mdl), cls)
                d = {k:denormalize(v) for (k,v) in six.iteritems(data) if k != normalize_type_key}
                return cls.denormalize(d)
        else: return {k:denormalize(v) for (k,v) in six.iteritems(data)} # native map
    else:
        # must be a list of some type
        if not hasattr(data, '__iter__'):
            msg = 'denormalize does not recognized object %s with type %s' % (data, type(data))
            raise ValueError(msg)
        # lists of primitives need not be changed
        if pimms.is_array(data, ('number', 'bool', 'string', 'unicode')): return data
        return [denormalize(x) for x in data]
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
        raise ValueError('%dD affine matrix must be %dx%d or %dx%d' % arg)
    return aff
def is_dataframe(d):
    '''
    is_dataframe(d) yields True if d is a pandas DataFrame object and False otherwise; if
      pandas cannot be loaded, this yields None.
    '''
    try: import pandas
    except Exception: return None
    return isinstance(d, pandas.DataFrame)
def to_dataframe(d, **kw):
    '''
    to_dataframe(d) attempts to coerce the object d to a pandas DataFrame object. If d is a
      tuple of 2 items whose second argument is a dictionary, then the dictionary will be taken
      as arguments for the dataframe constructor. These arguments may alternately be given as
      standard keyword arguments.
    '''
    import pandas
    if pimms.is_itable(d): d = d.dataframe
    if is_dataframe(d): return d if len(kw) == 0 else pandas.DataFrame(d, **kw)
    if is_tuple(d) and len(d) == 2 and pimms.is_map(d[1]):
        try: return to_dataframe(d[0], **pimms.merge(d[1], kw))
        except Exception: pass
    # try various options:
    try: return pandas.DataFrame(d, **kw)
    except Exception: pass
    try: return pandas.DataFrame.from_records(d, **kw)
    except Exception: pass
    try: return pandas.DataFrame.from_dict(d, **kw)
    except Exception: pass
    raise ValueError('Coersion to dataframe failed for object %s' % d)
def dataframe_select(df, *cols, **filters):
    '''
    dataframe_select(df, k1=v1, k2=v2...) yields df after selecting all the columns in which the
      given keys (k1, k2, etc.) have been selected such that the associated columns in the dataframe
      contain only the rows whose cells match the given values.
    dataframe_select(df, col1, col2...) selects the given columns.
    dataframe_select(df, col1, col2..., k1=v1, k2=v2...) selects both.
    
    If a value is a tuple/list of 2 elements, then it is considered a range where cells must fall
    between the values. If value is a tuple/list of more than 2 elements or is a set of any length
    then it is a list of values, any one of which can match the cell.
    '''
    ii = np.ones(len(df), dtype='bool')
    for (k,v) in six.iteritems(filters):
        vals = df[k].values
        if   pimms.is_set(v):                    jj = np.isin(vals, list(v))
        elif pimms.is_vector(v) and len(v) == 2: jj = (v[0] <= vals) & (vals < v[1])
        elif pimms.is_vector(v):                 jj = np.isin(vals, list(v))
        else:                                    jj = (vals == v)
        ii = np.logical_and(ii, jj)
    if len(ii) != np.sum(ii): df = df.loc[ii]
    if len(cols) > 0: df = df[list(cols)]
    return df
def dataframe_except(df, *cols, **filters):
    '''
    dataframe_except(df, k1=v1, k2=v2...) yields df after selecting all the columns in which the
      given keys (k1, k2, etc.) have been selected such that the associated columns in the dataframe
      contain only the rows whose cells match the given values.
    dataframe_except(df, col1, col2...) selects all columns except for the given columns.
    dataframe_except(df, col1, col2..., k1=v1, k2=v2...) selects on both conditions.
    
    The dataframe_except() function is identical to the dataframe_select() function with the single
    difference being that the column names provided to dataframe_except() are dropped from the
    result while column names passed to dataframe_select() are kept.

    If a value is a tuple/list of 2 elements, then it is considered a range where cells must fall
    between the values. If value is a tuple/list of more than 2 elements or is a set of any length
    then it is a list of values, any one of which can match the cell.
    '''
    ii = np.ones(len(df), dtype='bool')
    for (k,v) in six.iteritems(filters):
        vals = df[k].values
        if   pimms.is_set(v):                    jj = np.isin(vals, list(v))
        elif pimms.is_vector(v) and len(v) == 2: jj = (v[0] <= vals) & (vals < v[1])
        elif pimms.is_vector(v):                 jj = np.isin(vals, list(v))
        else:                                    jj = (vals == v)
        ii = np.logical_and(ii, jj)
    if len(ii) != np.sum(ii): df = df.loc[ii]
    if len(cols) > 0: df = df.drop(list(cols), axis=1, inplace=False)
    return df

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
def auto_dict(ival=None, miss=None):
    '''
    auto_dict() yields an auto-dict that vivifies value of {} on miss.
    auto_dict(ival) uses the given dict ival as an initializer.
    auto_dict(ival, miss) uses the given miss function.
    auto_dict(None, miss) is equivalent to auto_dict() with the given miss function.

    If the miss argument (also a named parameter, miss) is an empty list, an empty dict, or an
    empty set, then the miss is taken to be an anonymous lambda function that returns an empty
    item of the same type.
    '''
    if ival is None: d = AutoDict()
    else: d = AutoDict(ival)
    if miss == {} or miss is None: return d
    elif miss == []: d.on_miss = lambda:[]
    elif miss == set([]): d.on_miss = lambda:set([])
    else: d.on_miss = miss
    return d

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
    is_image(img) yields True if img is an instance if nibabel.spatialimages.SpatialImagee and False
      otherwise.
    '''
    return isinstance(image, nib.spatialimages.SpatialImage)
def is_image_header(x):
    '''
    is_image_header(x) yields True if x is a nibabel.spatialimages.SpatialHeader object and False
      otherwise.
    '''
    return isinstance(x, nib.spatialimages.SpatialHeader)

def is_address(data):
    '''
    is_address(addr) yields True if addr is a valid address dict for addressing positions on a mesh
      or in a cortical sheet and False otherwise.
    '''
    return (pimms.is_map(data) and 'faces' in data and 'coordinates' in data)
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

def numel(x):
    '''
    numel(x) yields the number of elements in x: the product of the shape of x.
    '''
    return int(np.prod(np.shape(x)))
def rows(x):
    '''
    rows(x) yields the number of rows in x; if x is a scalar, this is still 1.
    '''
    s = np.shape(x)
    return s[0] if len(s) > 0 else 1
def check_sparsity(x, fraction=0.6):
    '''
    check_sparsity(x) yields either x or an array equivalent to x with a different sparsity based on
      a heuristic: if x is a sparse array with more than 60% of its elements specified, it is made
      dense; otherwise, it is left alone.

    The optional argument fraction (default 0.6) specifies the fraction of elements that must be
    specified in the array for it to be un-sparsified.
    '''
    if not sps.issparse(x): return x
    n = numel(x)
    if n == 0: return x
    if len(x.data) / float(x) > 0.6: return x.toarray()
    else: return x
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
            nnm = np.prod(nm.shape)
            # if we have (sparse matrix) * (high-dim array), unbroadcast the dense array
            if   dnm == 0: return (sp, np.reshape(nm, (1,   1)))
            elif dnm == 1: return (sp, np.reshape(nm, (nnm, 1)))
            elif dnm == 2: return (sp, nm)
            else:          return unbroadcast(sp.toarray(), nm)
        return fix(a, b) if spa else tuple(reversed(fix(b, a)))
    # okay, no sparse matrices found:
    a = np.asarray(a)
    b = np.asarray(b)
    da = len(a.shape)
    db = len(b.shape)
    if   da > db: return (a, np.reshape(b, b.shape + tuple(np.ones(da-db, dtype=np.int))))
    elif da < db: return (np.reshape(a, a.shape + tuple(np.ones(db-da, dtype=np.int))), b)
    else:         return (a, b)
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
    if sps.issparse(a):
        if not sps.issparse(b):
            b = np.asarray(b)
            if len(b.shape) == 0: b = np.reshape(b, (1,1))
    elif sps.issparse(b):
        a = np.asarray(a)
        if len(a.shape) == 0: a = np.reshape(a, (1,1))
    else:
        a = np.asarray(a)
        b = np.asarray(b)
    return a + b
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
def cminus(a, b):
    '''
    cminus(a, b) returns the difference a - b as a numpy array object. Like numpy's subtract
      function or a - b syntax, minus will thread over the latest dimension possible.
    '''
    # adding/subtracting a constant to/from a sparse array is an error...
    spa = sps.issparse(a)
    spb = sps.issparse(b)
    if not spa: a = np.asarray(a)
    if not spb: b = np.asarray(b)
    if   spa: b = np.reshape(b, (1,1)) if len(np.shape(b)) == 0 else b
    elif spb: a = np.reshape(a, (1,1)) if len(np.shape(a)) == 0 else a
    return a - b
def minus(a, b):
    '''
    minus(a, b) returns the difference a - b as a numpy array object. Unlike numpy's subtract
      function or a - b syntax, minus will thread over the earliest dimension possible; thus if
      a.shape is (4,2) and b.shape is 4, a - b is a equivalent to [ai-bi for (ai,bi) in zip(a,b)].
    '''
    (a,b) = unbroadcast(a,b)
    return a - b
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
        except Exception: pass
        return x
    else:
        x = np.asarray(x)
        z = np.isclose(x, 0)
        r = np.logical_not(z) / (x + z)
        if null == 0: return r
        r[z] = null
        return r
def cdivide(a, b):
    '''
    cdivide(a, b) returns the quotient a / b as a numpy array object. Like numpy's divide function
      or a/b syntax, divide will thread over the latest dimension possible. Unlike numpy's divide,
      cdivide works with sparse matrices.

    Note that warnings/errors are raised by this function when divide-by-zero occurs, so it is
    usually not useful to use cdivide() with sparse matrices--see czdivide instead.
    '''
    if   sps.issparse(a): return a.multiply(inv(b))
    elif sps.issparse(b): return np.asarray(a) / b.toarray()
    else:                 return np.asarray(a) / np.asarray(b)
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
    return cdivide(a,b)
def czdivide(a, b, null=0):
    '''
    czdivide(a, b) returns the quotient a / b as a numpy array object. Like numpy's divide function
      or a/b syntax, czdivide will thread over the latest dimension possible. Unlike numpy's divide,
      czdivide works with sparse matrices. Additionally, czdivide multiplies a by the zinv of b, so
      divide-by-zero entries are replaced with 0 in the result.

    The optional argument null (default: 0) may be given to specify that zeros in the arary b should
    instead be replaced with the given value in the result. Note that if this value is not equal to
    0, then any sparse array passed as argument b must be reified.

    The czdivide function never raises an error due to divide-by-zero; if you desire this behavior,
    use the cdivide function instead.
    '''
    if null == 0:         return a.multiply(zinv(b)) if sps.issparse(a) else a * zinv(b)
    elif sps.issparse(b): b = b.toarray()
    else:                 b = np.asarray(b)
    z = np.isclose(b, 0)
    q = np.logical_not(z)
    zi = q / (b + z)
    if sps.issparse(a):
        r = a.multiply(zi).tocsr()
    else:
        r = np.asarray(a) * zi
    r[np.ones(a.shape, dtype=np.bool)*z] = null
    return r
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

    Note that zdivide(a,b, null=z) is not quite equivalent to a*zinv(b, null=z) unless z is 0; if z
    is not zero, then the same elements that are zet to z in zinv(b, null=z) are set to z in the
    result of zdivide(a,b, null=z) rather than the equivalent element of a times z.
    '''
    (a,b) = unbroadcast(a,b)
    return czdivide(a,b, null=null)
def cpower(a,b):
    '''
    cpower(a,b) is equivalent to a**b except that it also operates over sparse arrays; though it
    must reify them to do so.
    '''
    if sps.issparse(a): a = a.toarray()
    if sps.issparse(b): b = b.toarray()
    return a ** b
hpi    = np.pi / 2
tau    = 2 * np.pi
negpi  = -np.pi
neghpi = -hpi
negtau = -tau
def power(a,b):
    '''
    power(a,b) is equivalent to a**b except that, like the neuropythy.util.times function, it
      threads over the earliest dimension possible rather than the latest, as numpy's power function
      and ** syntax do. The power() function also works with sparse arrays; though it must reify
      them during the process.
    '''
    (a,b) = unbroadcast(a,b)
    return cpower(a,b)
def inner(a,b):
    '''
    inner(a,b) yields the dot product of a and b, doing so in a fashion that respects sparse
      matrices when encountered. This does not error check for bad dimensionality.

    If a or b are constants, then the result is just the a*b; if a and b are both vectors or both
    matrices, then the inner product is dot(a,b); if a is a vector and b is a matrix, this is
    equivalent to as if a were a matrix with 1 row; and if a is a matrix and b a vector, this is
    equivalent to as if b were a matrix with 1 column.
    '''
    if   sps.issparse(a): return a.dot(b)
    else: a = np.asarray(a)
    if len(a.shape) == 0: return a*b
    if sps.issparse(b):
        if len(a.shape) == 1: return b.T.dot(a)
        else:                 return b.T.dot(a.T).T
    else: b = np.asarray(b)
    if len(b.shape) == 0: return a*b
    if len(a.shape) == 1 and len(b.shape) == 2: return np.dot(b.T, a)
    else: return np.dot(a,b)
def sine(x):
    '''
    sine(x) is equivalent to sin(x) except that it also works on sparse arrays.
    '''
    if sps.issparse(x):
        x = x.copy()
        x.data = np.sine(x.data)
        return x
    else: return np.sin(x)
def cosine(x):
    '''
    cosine(x) is equivalent to cos(x) except that it also works on sparse arrays.
    '''
    # cos(0) = 1 so no point in keeping these sparse
    if sps.issparse(x): x = x.toarray(x)
    return np.cos(x)
def tangent(x, null=(-np.inf, np.inf), rtol=default_rtol, atol=default_atol):
    '''
    tangent(x) is equivalent to tan(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x == -pi/2 or -pi/2. If only one number is given, then it is
    used for both values; otherwise the first value corresponds to -pi/2 and the second to pi/2.
    A value of x is considered to be equal to one of these valids based on numpy.isclose. The
    optional arguments rtol and atol are passed along to isclose. If null is None, then no
    replacement is performed.
    '''
    if sps.issparse(x):
        x = x.copy()
        x.data = tangent(x.data, null=null, rtol=rtol, atol=atol)
        return x
    else: x = np.asarray(x)
    if rtol is None: rtol = default_rtol
    if atol is None: atol = default_atol
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    x = np.mod(x + pi, tau) - pi
    ii = None if nln is None else np.where(np.isclose(x, neghpi, rtol=rtol, atol=atol))
    jj = None if nlp is None else np.where(np.isclose(x, hpi,    rtol=rtol, atol=atol))
    x = np.tan(x)
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def cotangent(x, null=(-np.inf, np.inf), rtol=default_rtol, atol=default_atol):
    '''
    cotangent(x) is equivalent to cot(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x == 0 or pi. If only one number is given, then it is used for
    both values; otherwise the first value corresponds to 0 and the second to pi.  A value of x is
    considered to be equal to one of these valids based on numpy.isclose. The optional arguments
    rtol and atol are passed along to isclose. If null is None, then no replacement is performed.
    '''
    if sps.issparse(x): x = x.toarray()
    else:               x = np.asarray(x)
    if rtol is None: rtol = default_rtol
    if atol is None: atol = default_atol
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    x = np.mod(x + hpi, tau) - hpi
    ii = None if nln is None else np.where(np.isclose(x, 0,  rtol=rtol, atol=atol))
    jj = None if nlp is None else np.where(np.isclose(x, pi, rtol=rtol, atol=atol))
    x = np.tan(x)
    if ii: x[ii] = 1
    if jj: x[jj] = 1
    x = 1.0 / x
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def secant(x, null=(-np.inf, np.inf), rtol=default_rtol, atol=default_atol):
    '''
    secant(x) is equivalent to 1/sin(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x == -pi/2 or -pi/2. If only one number is given, then it is
    used for both values; otherwise the first value corresponds to -pi/2 and the second to pi/2.
    A value of x is considered to be equal to one of these valids based on numpy.isclose. The
    optional arguments rtol and atol are passed along to isclose. If null is None, then an error is
    raised when -pi/2 or pi/2 is encountered.
    '''
    if sps.issparse(x): x = x.toarray()
    else:               x = np.asarray(x)
    if rtol is None: rtol = default_rtol
    if atol is None: atol = default_atol
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    x = np.mod(x + pi, tau) - pi
    ii = None if nln is None else np.where(np.isclose(x, neghpi, rtol=rtol, atol=atol))
    jj = None if nlp is None else np.where(np.isclose(x, hpi,    rtol=rtol, atol=atol))
    x = np.cos(x)
    if ii: x[ii] = 1.0
    if jj: x[jj] = 1.0
    x = 1.0/x
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def cosecant(x, null=(-np.inf, np.inf), rtol=default_rtol, atol=default_atol):
    '''
    cosecant(x) is equivalent to 1/sin(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x == 0 or pi. If only one number is given, then it is used for
    both values; otherwise the first value corresponds to 0 and the second to pi. A value x is
    considered to be equal to one of these valids based on numpy.isclose. The optional arguments
    rtol and atol are passed along to isclose. If null is None, then an error is raised when -pi/2
    or pi/2 is encountered.
    '''
    if sps.issparse(x): x = x.toarray()
    else:               x = np.asarray(x)
    if rtol is None: rtol = default_rtol
    if atol is None: atol = default_atol
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    x = np.mod(x + hpi, tau) - hpi # center on pi/2 so that 0 and pi are easy to detect
    ii = None if nln is None else np.where(np.isclose(x, 0,  rtol=rtol, atol=atol))
    jj = None if nlp is None else np.where(np.isclose(x, pi, rtol=rtol, atol=atol))
    x = np.sin(x)
    if ii: x[ii] = 1.0
    if jj: x[jj] = 1.0
    x = 1.0/x
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def arcsine(x, null=(-np.inf, np.inf)):
    '''
    arcsine(x) is equivalent to asin(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x < -1 or x > 1. If only one number is given, then it is used
    for both values; otherwise the first value corresponds to <-1 and the second to >1.  If null is
    None, then an error is raised when invalid values are encountered.
    '''
    if sps.issparse(x):
        x = x.copy()
        x.data = arcsine(x.data, null=null, rtol=rtol, atol=atol)
        return x
    else: x = np.asarray(x)
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    ii = None if nln is None else np.where(x < -1)
    jj = None if nlp is None else np.where(x > 1)
    if ii: x[ii] = 0
    if jj: x[jj] = 0
    x = np.arcsin(x)
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def arccosine(x, null=(-np.inf, np.inf)):
    '''
    arccosine(x) is equivalent to acos(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x < -1 or x > 1. If only one number is given, then it is used
    for both values; otherwise the first value corresponds to <-1 and the second to >1.  If null is
    None, then an error is raised when invalid values are encountered.
    '''
    if sps.issparse(x): x = x.toarray()
    else:               x = np.asarray(x)
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    ii = None if nln is None else np.where(x < -1)
    jj = None if nlp is None else np.where(x > 1)
    if ii: x[ii] = 0
    if jj: x[jj] = 0
    x = np.arccos(x)
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def arctangent(y, x=None, null=0, broadcast=False, rtol=default_rtol, atol=default_atol):
    '''
    arctangent(x) is equivalent to atan(x) except that it also works on sparse arrays.
    arctangent(y,x) is equivalent to atan2(y,x) except that it also works on sparse arrays.

    The optional argument null (default: 0) specifies the result found when y and x both equal 0. If
    null is None, then an error is raised on this condition. Note that if null is not 0, then it is
    more likely that sparse arrays will have to be reified. If null is set to None, then no attempt
    is made to detect null values.

    The optional argument broadcast (default: False) specifies whether numpy-like (True) or
    Mathematica-like (False) broadcasting should be used. Broadcasting resolves ambiguous calls to
    arctangent, such as artangent([a,b,c], [[d,e,f],[g,h,i],[j,k,l]]). If broadcasting is True, 
    arctangent(y,x) behaves like numpy.arctan2(y,x), so [a,b,c] is interpreted like [[a,b,c],
    [a,b,c], [a,b,c]]. If broadcasting is False, [a,b,c] is interpreted like [[a,a,a], [b,b,b],
    [c,c,c]].
    '''
    if sps.issparse(y):
        if x is None:
            y = y.copy()
            y.data = np.arctan(y.data)
            return y
        elif null is not None and null != 0:
            # we need to reify anyway...
            y = y.toarray()
            if sps.issparse(x): x = x.toarray()
        else:
            # anywhere that y is zero must have an arctan of 0 or null (which is 0), so we only have
            # to look at those values that are non-zero in y
            (yr,yc,yv) = sps.find(y)
            xv = np.asarray(x[rr,rc].flat)
            res = y.copy()
            res.data = arctangent(yv, xv, null=null)
            res.eliminate_zeros()
            return res
    elif sps.issparse(x): x = x.toarray()
    # we should start by broadcasting if need be...
    if x is None: res = np.arctan(y)
    else:
        if not broadcast: (y,x) = unbroadcast(y,x)
        res = np.arctan2(y, x)
        # find the zeros, if need-be
        if null is not None:
            if rtol is None: rtol = default_rtol
            if atol is None: atol = default_atol
            # even if null is none, we do this because the rtol and atol may be more lenient than
            # the tolerance used by arctan2.
            z = np.isclose(y, 0, rtol=rtol, atol=atol) & np.isclose(x, 0, rtol=rtol, atol=atol)
            res[z] = null
    return res
def flattest(x):
    '''
    flattest(x) yields a 1D numpy vector equivalent to a flattened version of x. Unline
      np.asarray(x).flatten, flattest(x) works with sparse matrices. It does not, however, work with
      ragged arrays.
    '''
    x = x.toarray().flat if sps.issparse(x) else np.asarray(x).flat
    return np.array(x)
def flatter(x, k=1):
    '''
    flatter(x) yields a numpy array equivalent to x but whose first dimension has been flattened.
    flatter(x, k) yields a numpy array whose first k dimensions have been flattened; if k is
      negative, the last k dimensions are flattened. If np.inf or -np.inf is passed, then this is
      equivalent to flattest(x). Note that flatter(x) is equivalent to flatter(x,1).
    flatter(x, 0) yields x.
    '''
    if k == 0: return x
    x = x.toarray() if sps.issparse(x) else np.asarray(x)
    if len(x.shape) - abs(k) < 2: return x.flatten()
    k += np.sign(k)
    if k > 0: return np.reshape(x, (-1,) + x.shape[k:])
    else:     return np.reshape(x, x.shape[:k] + (-1,))
def part(x, *args):
    '''
    part(x, ii, jj...) is equivalent to x[ii, jj...] if x is a sparse matrix or numpy array and is
      equivalent to np.asarray(x)[ii][:, jj][...] if x is not. If only one argument is passed and
      it is a tuple, then it is passed like x[ii] alone.

    The part function is comparible with slices (though the must be entered using the slice(...)
    rather than the : syntax) and Ellipsis.
    '''
    n = len(args)
    sl = slice(None)
    if sps.issparse(x):
        if n == 1: return x[args[0]]
        elif n > 2: raise ValueError('Too many indices for sparse matrix')
        (ii,jj) = args
        if   ii is Ellipsis: ii = sl
        elif jj is Ellipsis: jj = sl
        ni = pimms.is_number(ii)
        nj = pimms.is_number(jj)
        if   ni and nj: return x[ii,jj]
        elif ni:        return x[ii,jj].toarray()[0]
        elif nj:        return x[ii,jj].toarray()[:,0]
        else:           return x[ii][:,jj]
    else:
        x = np.asarray(x)
        if n == 1: return x[args[0]]
        i0 = []
        for (k,arg) in enumerate(args):
            if arg is Ellipsis:
                # special case...
                #if Ellipsis in args[ii+1:]: raise ValueError('only one ellipsis allowed per part')
                left = n - k - 1
                i0 = [sl for _ in range(len(x.shape) - left)]
            else:
                x = x[tuple(i0 + [arg])]
                if not pimms.is_number(arg): i0.append(sl)
        return x
def hstack(tup):
    '''
    hstack(x) is equivalent to numpy.hstack(x) or scipy.sparse.hstack(x) except that it works
      correctly with both sparse and dense arrays (if any inputs are dense, it converts all inputs
      to dense arrays).
    '''
    if all([sps.issparse(u) for u in tup]): return sps.hstack(tup, format=tup[0].format)
    else: return np.hstack([u.toarray() if sps.issparse(u) else u for u in tup])
def vstack(tup):
    '''
    vstack(x) is equivalent to numpy.vstack(x) or scipy.sparse.vstack(x) except that it works
      correctly with both sparse and dense arrays (if any inputs are dense, it converts all inputs
      to dense arrays).
    '''
    if all([sps.issparse(u) for u in tup]): return sps.vstack(tup, format=tup[0].format)
    else: return np.vstack([u.toarray() if sps.issparse(u) else u for u in tup])
def repmat(x, r, c):
    '''
    repmat(x, r, c) is equivalent to numpy.matlib.repmat(x, r, c) except that it works correctly for
      sparse matrices.
    '''
    if sps.issparse(x):
        row = sps.hstack([x for _ in range(c)])
        return sps.vstack([row for _ in range(r)], format=x.format)
    else: return np.matlib.repmat(x, r, c)
    
def replace_close(x, xhat, rtol=default_rtol, atol=default_atol, copy=True):
    '''
    replace_close(x, xhat) yields x if x is not close to xhat and xhat otherwise. Closeness is
      determined by numpy's isclose(), and the atol and rtol options are passed along.

    The x and xhat arguments may be lists/arrays.

    The optional argument copy may also be set to False to chop x in-place.
    '''
    if rtol is None: rtol = default_rtol
    if atol is None: atol = default_atol
    x = np.array(x) if copy else np.asarray(x)
    w = np.isclose(x, xhat, rtol=rtol, atol=atol)
    x[w] = np.asarray(xhat)[w]
    return x
def chop(x, rtol=default_rtol, atol=default_atol, copy=True):
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
                 order=1, weights=None, smoothing=None, periodic=False,
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
        assert((ds >= 0).all())
        return ds
    @pimms.require
    def check_distances(distances, coordinates, periodic):
        if distances is None: return True
        if len(distances) != coordinates.shape[1] - 1:
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
        # we need to skip anything where t[i] and t[i+1] are too close
        wh = np.where(np.isclose(np.diff(t), 0))[0]
        if len(wh) > 0:
            (t,x,y) = [np.array(u) for u in (t,x,y)]
            ii = np.arange(len(t))
            for i in reversed(wh):
                ii[i+1:-1] = ii[i+2:]
                for u in (t,x,y):
                    u[i] = np.mean(u[i:i+2])
            ii = ii[:-len(wh)]
            (t,x,y) = [u[ii] for u in (t,x,y)]
        xtck = interpolate.splrep(t, x, k=order, s=smoothing, w=weights, per=periodic)
        ytck = interpolate.splrep(t, y, k=order, s=smoothing, w=weights, per=periodic)
        return tuple([tuple([pimms.imm_array(u) for u in tck])
                      for tck in (xtck,ytck)])
    def __repr__(self):
        return 'CurveSpline(<%d points>, order=%d, %f <= t <= %f)' % (
            self.coordinates.shape[1],
            self.order, self.t[0], self.t[-1])
    def __call__(self, t, derivative=0):
        from scipy import interpolate
        xint = interpolate.splev(t, self.splrep[0], der=derivative, ext=0)
        yint = interpolate.splev(t, self.splrep[1], der=derivative, ext=0)
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

def curve_spline(x, y=None, weights=None, order=1, even_out=True,
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
      * even_out (True) whether to even out the distances along
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
def is_curve_spline(obj):
    '''
    is_curve_spline(obj) yields True if obj is a curve spline object and False otherwise.
    '''
    return isinstance(obj, CurveSpline)
def to_curve_spline(obj):
    '''
    to_curve_spline(obj) obj if obj is a curve spline and otherwise attempts to coerce obj into a
      curve spline, raising an error if it cannot.
    '''
    if   is_curve_spline(obj):            return obj
    elif is_tuple(obj) and len(obj) == 2: (crds,opts) = obj
    else:                                 (crds,opts) = (obj,{})
    if pimms.is_matrix(crds) or is_curve_spline(crds): crds = [crds]
    spls = [c for c in crds if is_curve_spline(c)]
    opts = dict(opts)
    if 'weights' not in opts and len(spls) == len(crds):
        if all(c.weights is not None for c in crds):
            opts['weights'] = np.concatenate([c.weights for c in crds])
    if 'order' not in opts and len(spls) > 0:
        opts['order'] = np.min([c.order for c in spls])
    if 'smoothing' not in opts and len(spls) > 0:
        sm = set([c.smoothing for c in spls])
        if len(sm) == 1: opts['smoothing'] = list(sm)[0]
        else: opts['smoothing'] = None
    crds = [x.crds if is_curve_spline(crds) else np.asarray(x) for x in crds]
    crds = [x if x.shape[0] == 2 else x.T for x in crds]
    crds = np.hstack(crds)
    return curve_spline(crds, **opts)
def curve_intersection(c1, c2, grid=16):
    '''
    curve_intersect(c1, c2) yields the parametric distances (t1, t2) such that c1(t1) == c2(t2).
      
    The optional parameter grid may specify the number of grid-points
    to use in the initial search for a start-point (default: 16).
    '''
    from scipy.optimize import minimize
    from neuropythy.geometry import segment_intersection_2D
    if c1.coordinates.shape[1] > c2.coordinates.shape[1]:
        (t1,t2) = curve_intersection(c2, c1, grid=grid)
        return (t2,t1)
    # before doing a search, see if there are literal exact intersections of the segments
    x1s  = c1.coordinates.T
    x2s  = c2.coordinates
    for (ts,te,xs,xe) in zip(c1.t[:-1], c1.t[1:], x1s[:-1], x1s[1:]):
        pts = segment_intersection_2D((xs,xe), (x2s[:,:-1], x2s[:,1:]))
        ii = np.where(np.isfinite(pts[0]))[0]
        if len(ii) > 0:
            ii = ii[0]
            def f(t): return np.sum((c1(t[0]) - c2(t[1]))**2)
            t01 = 0.5*(ts + te)
            t02 = 0.5*(c2.t[ii] + c2.t[ii+1])
            (t1,t2) = minimize(f, (t01, t02)).x
            return (t1,t2)
    if pimms.is_vector(grid): (ts1,ts2) = [c.t[0] + (c.t[-1] - c.t[0])*grid for c in (c1,c2)]
    else:                     (ts1,ts2) = [np.linspace(c.t[0], c.t[-1], grid) for c in (c1,c2)]
    (pts1,pts2) = [c(ts) for (c,ts) in zip([c1,c2],[ts1,ts2])]
    ds = np.sqrt([np.sum((pts2.T - pp)**2, axis=1) for pp in pts1.T])
    (ii,jj) = np.unravel_index(np.argmin(ds), ds.shape)
    (t01,t02) = (ts1[ii], ts2[jj])
    ttt = []
    def f(t): return np.sum((c1(t[0]) - c2(t[1]))**2)
    (t1,t2) = minimize(f, (t01, t02)).x
    return (t1,t2)
def close_curves(*crvs, **kw):
    '''
    close_curves(crv1, crv2...) yields a single curve that merges all of the given list of curves
      together. The curves must be given in order, such that the i'th curve should be connected to
      to the (i+1)'th curve circularly to form a perimeter.

    The following optional parameters may be given:
      * grid may specify the number of grid-points to use in the initial search for a start-point
        (default: 16).
      * order may specify the order of the resulting curve; by default (None) uses the lowest order
        of all curves.
      * smoothing (None) the amount to smooth the points.
      * even_out (True) whether to even out the distances along the curve.
      * meta_data (None) an optional map of meta-data to give the spline representation.
    '''
    for k in six.iterkeys(kw):
        if k not in close_curves.default_options: raise ValueError('Unrecognized option: %s' % k)
    kw = {k:(kw[k] if k in kw else v) for (k,v) in six.iteritems(close_curves.default_options)}
    (grid, order) = (kw['grid'], kw['order'])
    crvs = [(crv if is_curve_spline(crv) else to_curve_spline(crv)).even_out() for crv in crvs]
    # find all intersections:
    isects = [curve_intersection(u,v, grid=grid)
              for (u,v) in zip(crvs, np.roll(crvs,-1))]
    # subsample curves
    crds = np.hstack([crv.subcurve(s1[1], s0[0]).coordinates[:,:-1]
                      for (crv,s0,s1) in zip(crvs, isects, np.roll(isects,1,0))])
    kw['order'] = np.min([crv.order for crv in crvs]) if order is None else order
    kw = {k:v for (k,v) in six.iteritems(kw)
          if v is not None and k in ('order','smoothing','even_out','meta_data')}
    return curve_spline(crds, periodic=True, **kw)
close_curves.default_options = dict(grid=16, order=None, even_out=True,
                                    smoothing=None, meta_data=None)
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

def dirpath_to_list(p):
    '''
    dirpath_to_list(path) yields a list of directories contained in the given path specification.

    A path may be either a single directory name (==> [path]), a :-separated list of directories
    (==> path.split(':')), a list of directory names (==> path), or None (==> []). Note that the
    return value filters out parts of the path that are not directories.
    '''
    if   p is None: p = []
    elif pimms.is_str(p): p = p.split(':')
    if len(p) > 0 and not pimms.is_vector(p, str):
        raise ValueError('Path is not equivalent to a list of dirs')
    return [pp for pp in p if os.path.isdir(pp)]
