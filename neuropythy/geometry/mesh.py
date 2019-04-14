####################################################################################################
# neuropythy/geometry/mesh.py
# Tools for interpolating from meshes.
# By Noah C. Benson

import numpy                        as np
import numpy.matlib                 as npml
import numpy.linalg                 as npla
import scipy                        as sp
import scipy.spatial                as space
import scipy.sparse                 as sps
import scipy.optimize               as spopt
import nibabel                      as nib
import nibabel.freesurfer.mghformat as fsmgh
import pyrsistent                   as pyr
import os, sys, six, types, logging, warnings, gzip, json, pimms

from .util  import (triangle_area, triangle_address, alignment_matrix_3D, rotation_matrix_3D,
                    cartesian_to_barycentric_3D, cartesian_to_barycentric_2D, vector_angle_cos,
                    segment_intersection_2D, segments_overlapping, points_close, point_in_segment,
                    barycentric_to_cartesian, point_in_triangle)
from ..util import (ObjectWithMetaData, to_affine, zinv, is_image, is_address, address_data, curry,
                    curve_spline, CurveSpline, chop, zdivide, flattest, inner, config, library_path,
                    dirpath_to_list, to_hemi_str, is_tuple, is_list, is_set, close_curves,
                    normalize, denormalize, AutoDict, auto_dict)
from ..io   import (load, importer, exporter)
from functools import reduce

try:              from StringIO import StringIO
except Exception: from io import StringIO

@pimms.immutable
class VertexSet(ObjectWithMetaData):
    '''
    VertexSet is a class that tracks a number of vertices, including properties for them. This class
    is intended as a base class for Tesselation and Mesh, both of which track vertex properties.
    Note that all VertexSet objects add/overwrite the keys 'index' and 'label' in their itables in
    order to store the vertex indices and labels as properties.
    '''

    def __init__(self, labels, properties=None, meta_data=None):
        self._properties = properties
        self.labels = labels
        self.meta_data = meta_data

    @pimms.param
    def labels(lbls):
        '''
        vset.labels is an array of the integer vertex labels.
        '''
        return pimms.imm_array(lbls)
    @pimms.param
    def _properties(props):
        '''
        obj._properties is an itable of property values given to the vertex-set obj; this is a
        pre-processed input version of the value obj.properties.
        '''
        if props is None: return None
        if pimms.is_itable(props): return props.persist()
        elif pimms.is_map(props): return pimms.itable(props).persist()
        else: raise ValueError('provided properties data must be a mapping')
    @pimms.value
    def vertex_count(labels):
        '''
        vset.vertex_count is the number of vertices in the given vertex set vset.
        '''
        return len(labels)
    @pimms.value
    def indices(vertex_count):
        '''
        vset.indices is the list of vertex indices for the given vertex-set vset.
        '''
        idcs = np.arange(0, vertex_count, 1, dtype=np.int)
        idcs.setflags(write=False)
        return idcs
    @pimms.require
    def validate_vertex_properties_size(_properties, vertex_count):
        '''
        validate_vertex_properties_size requires that _properties have the same number of rows as
        the vertex_count (unless _properties is empty).
        '''
        if   _properties is None: return True
        elif _properties.row_count == 0: return True
        elif _properties.row_count == vertex_count: return True
        else:
            s = (_properties.row_count, vertex_count)
            s = '_properties.row_count (%d) and vertex_count (%d) must be equal' % s
            raise ValueError(s)
    # The idea here is that _properties may be provided by the overloading class, then properties
    # can be overloaded by that class to add unmodifiable properties to the object; e.g., meshes
    # want coordinates to be a property that cannot be updated.
    @pimms.value
    def properties(_properties, labels, indices):
        '''
        obj.properties is an itable of property values given to the vertex-set obj.
        '''
        _properties = pyr.m() if _properties is None else _properties
        return _properties.set('index', indices).set('label', labels)
    @pimms.value
    def repr(vertex_count):
        '''
        obj.repr is the representation string returned by obj.__repr__().
        '''
        return 'VertexSet(<%d vertices>)' % self.vertex_count

    # Normal Methods
    def __repr__(self):
        return self.repr
    def prop(self, name):
        '''
        obj.prop(name) yields the vertex property in the given object with the given name.
        obj.prop(data) yields data if data is a valid vertex property list for the given object.
        obj.prop([p1, p2...]) yields a (d x n) vector of properties where d is the number of
          properties given and n is obj.properties.row_count.
        obj.prop(set([name1, name2...])) yields a mapping of the given names mapped to the
          appropriate property values.
        '''
        if pimms.is_str(name):
            return self.properties[name]
        elif is_set(name):
            return pyr.pmap({nm:self.properties[nm] for nm in name})
        elif pimms.is_vector(name):
            if len(name) == self.properties.row_count:
                return name
            else:
                return np.asarray([self.prop(nm) for nm in name])
        else:
            raise ValueError('unrecognized property')
    def with_prop(self, *args, **kwargs):
        '''
        obj.with_prop(...) yields a duplicate of the given object with the given properties added to
          it. The properties may be specified as a sequence of mapping objects followed by any
          number of keyword arguments, all of which are merged into a single dict left-to-right
          before application.
        '''
        pp = pimms.merge(self._properties if self._properties else {}, *(args + (kwargs,)))
        if pp is self._properties: return self
        pp = pimms.ITable(pp, n=self.vertex_count)
        return self if pp is self._properties else self.copy(_properties=pp)
    def wout_prop(self, *args):
        '''
        obj.wout_property(...) yields a duplicate of the given object with the given properties
          removed from it. The properties may be specified as a sequence of column names or lists of
          column names.
        '''
        pp = self._properties
        for a in args: pp = pp.discard(a)
        return self if pp is self._properties else self.copy(_properties=pp)
    def property(self, prop,
                 dtype=Ellipsis,
                 outliers=None,  data_range=None,    clipped=np.inf,
                 weights=None,   weight_min=0,       weight_transform=Ellipsis,
                 mask=None,      valid_range=None,   null=np.nan,
                 transform=None, yield_weight=False):
        '''
        vset.property(prop) is equivalent to to_property(vset, prop).
        '''
        return to_property(self, prop,
                           dtype=dtype,           null=null,
                           outliers=outliers,     data_range=data_range,
                           clipped=clipped,       weights=weights,
                           weight_min=weight_min, weight_transform=weight_transform,
                           mask=mask,             valid_range=valid_range,
                           transform=transform,   yield_weight=yield_weight)
    def isolines(self, prop, val, **kw):
        '''
        vset.isolines(prop, val, options...) is equivalent to isolines(tess, prop, val, options...).
        
        See the neurpythy.isolines() function for additional information.
        '''
        return isolines(self, prop, val, **kw) 
    def map(self, f):
        '''
        tess.map(f) is equivalent to tess.properties.map(f).
        '''
        return self.properties.map(f)
    def where(self, f, indices=False):
        '''
        obj.where(f) yields a list of vertex labels l such that f(p[l]) yields True, where p is the
          properties value for the vertex-set obj (i.e., p[l] is the property map for the vertex
          with label l. The function f should operate on a dict p which is identical to that passed
          to the method obj.properties.map().
        The optional third parameter indices (default: False) may be set to True to indicate that
        indices should be returned instead of labels.
        '''
        idcs = np.where(self.map(f))[0]
        return idcs if indices else self.labels[idcs]
    def mask(self, m, indices=False):
        '''
        obj.mask(m) yields the set of vertex labels from the given vertex-set object obj that
          correspond to the mask m.

        The mask m may take any of the following forms:
           * a list of vertex indices
           * a boolean array (one value per vertex)
           * a property name, which can be cast to a boolean array
           * a tuple (property, value) where property is a list of values, one per vertex, and value
             is the value that must match in order for a vertex to be included (this is basically
             equivalent to the mask (property == value); note that property may also be a property
             name
           * a tuple (property, min, max), which specifies that the property must be between min and
             max for a vertex to be included (min < p <= max)
           * a tuple (property, (val1, val2...)), which specifies that the property must be any of
             the values in (val1, val2...) for a vertex to be included
           * None, indicating that all labels should be returned
        
        Note that the optional argument indices (default: False) may be set to true to yield the
        vertex indices instead of the vertex labels.
        '''
        return to_mask(self, m, indices=indices)
def is_vset(v):
    '''
    is_vset(v) yields True if v is a VertexSet object and False otherwise. Note that topologies,
      tesselations, and meshes are all vertex sets.
    '''
    return isinstance(v, VertexSet)
        
def to_mask(obj, m=None, indices=None):
    '''
    to_mask(obj, m) yields the set of indices from the given vertex-set or itable object obj that
      correspond to the given mask m.
    to_mask((obj, m)) is equivalent to to_mask(obj, m).
    
    The mask m may take any of the following forms:
       * a list of vertex indices
       * a boolean array (one value per vertex)
       * a property name, which can be cast to a boolean array
       * a tuple (property, value) where property is a list of values, one per vertex, and value
         is the value that must match in order for a vertex to be included (this is basically
         equivalent to the mask (property == value); note that property may also be a property
         name
       * a tuple (property, min, max), which specifies that the property must be between min and
         max for a vertex to be included (min < p <= max)
       * a tuple (property, (val1, val2...)), which specifies that the property must be any of
         the values in (val1, val2...) for a vertex to be included
       * None, indicating that all labels should be returned
       * a dict/mapping with one item whose key is either 'and', or 'or' and whose value is a list,
         each of whose elements matches one of the above.
    
    Note that the optional argument indices (default: False) may be set to true to yield the
    vertex indices instead of the vertex labels. If obj is not a VertexSet object, then this
    option is ignored.
    '''
    if not pimms.is_map(obj) and pimms.is_vector(obj) and len(obj) < 4 and m is None:
        if   len(obj) == 1: obj = obj[0]
        elif len(obj) == 2: (obj, m) = obj
        else:
            (obj, m, q) = obj
            if indices is None:
                if pimms.is_map(q): indices = q.get('indices', False)
                else: indices = q
    if indices is None: indices = False
    if is_vset(obj):
        lbls = obj.labels
        idcs = obj.indices
        obj = obj.properties
    else:
        obj = pimms.itable(obj)
        lbls = np.arange(0, obj.row_count, 1, dtype=np.int)
        idcs = lbls
    if m is None: return idcs if indices else lbls
    if is_tuple(m):
        if len(m) == 0: return np.asarray([], dtype=np.int)
        p = to_property(obj, m[0])
        if len(m) == 2 and hasattr(m[1], '__iter__'):
            m = reduce(lambda q,u: np.logical_or(q, p == u), m[1], np.zeros(len(p), dtype=np.bool))
        elif len(m) == 2:
            m = (p == m[1])
        elif len(m) == 3:
            m = np.logical_and(m[1] < p, p <= m[2])
    elif pimms.is_str(m):
        m = np.asarray(obj[m], dtype=np.bool)
    elif pimms.is_map(m):
        if len(m) != 1: raise ValueError('Dicts used as masks must contain 1 item')
        (k,v) = next(six.iteritems(m))
        if not hasattr(v, '__iter__'): raise ValueError('Value of dict-mask must be an iterator')
        if not pimms.is_str(k): raise ValueError('Key of dict-mask must be "or", or "and"')
        v = [to_mask(obj, u, indices=indices) for u in v]
        if   k in ('and', 'intersect', 'intersection', '^', '&', '&&'):
            return reduce(np.intersect1d, v)
        elif k in ('or',  'union', 'v', '|' '||'):
            return reduce(np.union1d, v)
    # at this point, m should be a boolean array or a list of indices
    return idcs[m] if indices else lbls[m]
def to_property(obj, prop=None,
                dtype=Ellipsis,
                outliers=None,  data_range=None,    clipped=np.inf,
                weights=None,   weight_min=0,       weight_transform=Ellipsis,
                mask=None,      valid_range=None,   null=np.nan,
                transform=None, yield_weight=False):
    '''
    to_property(obj, prop) yields the given property from obj after performing a set of filters on
      the property, as specified by the options. In the property array that is returned, the values
      that are considered outliers (data out of some range) are indicated by numpy.inf, and values
      that are not in the optionally-specified mask are given the value numpy.nan; these may be
      changed with the clipped and null options, respectively.
    to_property((obj, prop)) is equivalent to to_property(obj, prop).

    The property argument prop may be either specified as a string (a property name in the object)
    or as a property vector. The weights option may also be specified this way. Additionally, the
    prop arg may be a list such as ['polar_angle', 'eccentricity'] where each element is either a
    string or a vector, in which case the result is a matrix of properties. Finally, prop may be
    a set of property names, in which case the return value is an itable whose keys are the property
    names.

    The obj argument may be either a VertexSet object (such as a Mesh or Tesselation) or a mapping
    object such as a pimms ITable. If no strings are used to specify properties, it may additionally
    be omitted or set to None.

    The following options are accepted:
      * outliers (default:None) specifies the vertices that should be considered outliers; this
        may be either None (no outliers explicitly specified), a list of indices, or a boolean
        mask.
      * data_range (default:None) specifies the acceptable data range for values in the
        property; if None then this paramter is ignored. If specified as a pair of numbers
        (min, max), then data that is less than the min or greater than the max is marked as an
        outlier (in addition to other explicitly specified outliers). The values np.inf or 
        -np.inf can be specified to indicate a one-sided range.
      * clipped (default:np.inf) specifies the value to be used to mark an out-of-range value in
        the returned array.
      * mask (default:None) specifies the vertices that should be included in the property 
        array; values are specified in the mask similarly to the outliers option, except that
        mask values are included rather than excluded. The mask takes precedence over the 
        outliers, in that a null (out-of-mask) value is always marked as null rather than
        clipped.
      * valid_range (default: None) specifies the range of values that are considered valid; 
        i.e., values outside of the range are marked as null. Specified the same way as
        data_range.
      * null (default: np.nan) specifies the value marked in the array as out-of-mask.
      * transform (default:None) may optionally provide a function to be passed the array prior
        to being returned (after null and clipped values are marked).
      * dtype (defaut:Ellipsis) specifies the type of the array that should be returned.
        Ellipsis indicates that the type of the given property should be used. If None, then a
        normal Python array is returned. Otherwise, should be a numpy type such as numpy.real64
        or numpy.complex128.
      * weights (default:Ellipsis) specifies the property or property array that should be
        examined as the weights. The default, Ellipsis, simply chops values that are close to or
        less than 0 such that they are equal to 0. None specifies that no transformation should
        be applied.
      * weight_min (default:0) specifies the value at-or-below which the weight is considered 
        insignificant and the value is marked as clipped.
      * weight_transform (default:None) specifies a function that should be applied to the
        weight array before being used in the function.
      * yield_weight (default:False) specifies, if True, that instead of yielding prop, yield
        the tuple (prop, weights).
    
    '''
    # was an arg given, or is obj a tuple?
    if pimms.is_vector(obj) and len(obj) < 4 and prop is None:
        kw0 = dict(dtype=dtype,           null=null,
                   outliers=outliers,     data_range=data_range,
                   clipped=clipped,       weights=weights,
                   weight_min=weight_min, weight_transform=weight_transform,
                   mask=mask,             valid_range=valid_range,
                   transform=transform,   yield_weight=yield_weight)
        if   len(obj) == 2: return to_property(obj[0], obj[1], **kw0)
        elif len(obj) == 3: return to_property(obj[0], obj[1], **pimms.merge(kw0, obj[2]))
        else: raise ValueError('Bad input vector given to to_property()')
    # we could have been given a property alone or a map/vertex-set and a property
    if prop is None: raise ValueError('No property given to to_property()')
    # if it's a vertex-set, we want to note that and get the map
    if isinstance(obj, VertexSet): (vset, obj) = (obj,  obj.properties)
    elif pimms.is_map(obj):        (vset, obj) = (None, obj)
    elif obj is None:              (vset, obj) = (None, None)
    else: ValueError('Data object given to to_properties() is neither a vertex-set nor a mapping')
    # Now, get the property array, as an array
    if pimms.is_str(prop):
        if obj is None: raise ValueError('a property name but no data object given to to_property')
        else: prop = obj[prop]
    if is_set(prop):
        def _lazy_prop(kk):
            return lambda:to_property(obj, kk,
                                      dtype=dtype,           null=null,
                                      outliers=outliers,     data_range=data_range,
                                      clipped=clipped,       weights=weights,
                                      weight_min=weight_min, weight_transform=weight_transform,
                                      mask=mask,             valid_range=valid_range,
                                      transform=transform,   yield_weight=yield_weight)
        return pimms.itable({k:_lazy_prop(k) for k in prop})
    elif (pimms.is_matrix(prop) or
          (pimms.is_vector(prop) and all(pimms.is_str(p) or pimms.is_vector(p) for p in prop))):
        return np.asarray([to_property(obj, k,
                                       dtype=dtype,           null=null,
                                       outliers=outliers,     data_range=data_range,
                                       clipped=clipped,       weights=weights,
                                       weight_min=weight_min, weight_transform=weight_transform,
                                       mask=mask,             valid_range=valid_range,
                                       transform=transform,   yield_weight=yield_weight)
                           for k in prop])
    elif not pimms.is_vector(prop):
        raise ValueError('prop must be a property name or a vector or a combination of these')
    else: prop = np.asarray(prop)
    if dtype is Ellipsis:  dtype = prop.dtype
    # Go ahead and process the weights
    if pimms.is_str(weights):
        if obj is None: raise ValueError('a weight name but no data object given to to_property')
        else: weights = obj[weights]
    weights_orig = weights
    if weights is None or weight_min is None: low_weight = np.asarray([], dtype=np.int)
    else:
        if weight_transform is Ellipsis:
            weights = np.array(weights, dtype=np.float)
            weights[weights < 0] = 0
        elif weight_transform is not None:
            weight = weight_transform(np.asarray(weights))
        if not pimms.is_vector(weights, 'real'):
            raise ValueError('weights must be a real-valued vector or property name for such')
        low_weight = (np.asarray([], dtype=np.int) if weight_min is None else
                      np.where(weights < weight_min)[0])
    # we can also process the outliers
    outliers = np.asarray([], dtype=np.int) if outliers is None else np.arange(len(prop))[outliers]
    outliers = np.union1d(outliers, low_weight) # low-weight vertices are treated as outliers
    # make sure we interpret mask correctly...
    mask = to_mask(obj, mask, indices=True)
    # Now process the property depending on whether the type is numeric or not
    if pimms.is_array(prop, 'number'):
        if pimms.is_array(prop, 'int'): prop = np.array(prop, dtype=np.float)
        else: prop = np.array(prop) # complex or reals can support nan
        if not np.isnan(null): prop[prop == null] = np.nan
        mask_nan = np.isnan(prop)
        mask_inf = np.isinf(prop)
        where_nan = np.where(mask_nan)[0]
        where_inf = np.where(mask_inf)[0]
        where_ok  = np.where(np.logical_not(mask_nan | mask_inf))[0]
        # look at the valid_range...
        if valid_range is None: where_inv = np.asarray([], dtype=np.int)
        else: where_inv = where_ok[(prop[where_ok] < valid_range[0]) |
                                   (prop[where_ok] > valid_range[1])]
        where_nan = np.union1d(where_nan, where_inv)
        mask = np.setdiff1d(mask, where_nan)
        # Find the outliers: values specified as outliers or inf values; will build this as we go
        outliers = np.intersect1d(outliers, mask) # outliers not in the mask don't matter anyway
        # If there's a data range argument, deal with how it affects outliers
        if data_range is not None:
            if not pimms.is_vector(data_range): data_range = (0, data_range)
            mii = mask[(prop[mask] < data_range[0]) | (prop[mask] > data_range[1])]
            outliers = np.union1d(outliers, mii)
        # no matter what, trim out the infinite values (even if inf was in the data range)
        outliers = np.union1d(outliers, mask[np.isinf(prop[mask])])
        # Okay, mark everything in the prop:
        unmask = np.setdiff1d(np.arange(len(prop), dtype=np.int), mask)
        if len(outliers) > 0:  prop[outliers]  = clipped
        if len(unmask) > 0: prop[unmask] = null
        prop = prop.astype(dtype)
    elif len(mask) < len(prop) or len(outliers) > 0:
        # not a number array; we cannot do fancy trimming of values
        tmp = np.full(len(prop), null, dtype=dtype)
        tmp[mask] = prop[mask]
        if len(outliers) > 0: tmp[outliers] = clipped
    if yield_weight:
        if weights is None or not pimms.is_vector(weights): weights = np.ones(len(prop))
        else: weights = np.array(weights, dtype=np.float)
        weights[where_nan] = 0
        weights[outliers] = 0
    # transform?
    if transform: prop = transform(prop)
    # That's it, just return
    return (prop, weights) if yield_weight else prop

@pimms.immutable
class TesselationIndex(object):
    '''
    TesselationIndex is an immutable helper-class for Tesselation. The TesselationIndex handles
    requests to obtain indices of vertices, edges, and faces in a tesselation object. Generally,
    this is done via the __getitem__ (index[item]) method. In the case that you wish to obtain the
    vertex indices for an edge or face but don't wish to obtain the index of the edge or face
    itself, the __call__ (index(item)) method can be used. Note that when looking up the indices of
    vertices, negative values are ignored/passed through. In this way, you can indicate a missing
    vertex in a list with a -1 without being affected by indexing.
    '''

    def __init__(self, vertex_index, edge_index, face_index):
        self.vertex_index = vertex_index
        self.edge_index = edge_index
        self.face_index = face_index

    @pimms.param
    def vertex_index(vi):
        if not pimms.is_pmap(vi): vi = pyr.pmap(vi)
        return vi
    @pimms.param
    def edge_index(ei):
        if not pimms.is_pmap(ei): ei = pyr.pmap(ei)
        return ei
    @pimms.param
    def face_index(fi):
        if not pimms.is_pmap(fi): fi = pyr.pmap(fi)
        return fi
    @pimms.value
    def vertex_matrix(vertex_index):
        ks = np.array(vertex_index.keys())
        vs = np.array(vertex_index.values())
        n = np.max(ks)
        return sps.csr_matrix((vs + 1, (np.ones(len(ks)), ks)), shape=(1, n), dtype=np.int)
    @pimms.value
    def vertex_matrix(vertex_index):
        ls = np.array(vertex_index.keys())
        ii = np.array(vertex_index.values())
        n = np.max(ls) + 1
        return sps.csr_matrix((ii + 1, (np.zeros(len(ls)), ls)), shape=(1, n), dtype=np.int)
    @pimms.value
    def edge_matrix(edge_index):
        (us,vs) = np.array(edge_index.keys()).T
        ii = np.array(edge_index.values())
        n = np.max([us,vs]) + 1
        return sps.csr_matrix((ii + 1, (us, vs)), shape=(n, n), dtype=np.int)
    @pimms.value
    def face_matrix(face_index):
        (a,b,c) = np.array(face_index.keys()).T
        ii = np.array(face_index.values())
        n = np.max([a,b,c]) + 1
        # we have to cheat with the last two
        bc = b*n + c
        return sps.csr_matrix((ii + 1, (a,bc)), shape=(n, n*n), dtype=np.int)
    
    def __repr__(self):
            return "TesselationIndex(<%d vertices>)" % len(self.vertex_index)
    def __getitem__(self, index):
        if is_tuple(index):
            if   len(index) == 3: return self.face_index.get(index, None)
            elif len(index) == 2: return self.edge_index.get(index, None)
            elif len(index) == 1: return self.vertex_index.get(index[0], None)
            else:                 raise ValueError('Unrecognized tesselation item: %s' % index)
        elif is_set(index):
            return {k:self[k] for k in index}
        elif pimms.is_vector(index):
            index = np.array(index)
            mtx = self.vertex_matrix
            yy = np.where((index >= 0) & (index < mtx.shape[1]))[0]
            if len(yy) < len(index):
                index = np.array(index)
                res = np.full(len(index), -1)
                res[yy] = flattest(mtx[0, index[yy]]) - 1
            else: res = flattest(mtx[0, index]) - 1
        elif pimms.is_matrix(index):
            m = np.asarray(index)
            if m.shape[0] != 2 and m.shape[0] != 3: m = m.T
            if m.shape[0] == 2:
                (u,v) = m
                xx = np.where((u < 0) | (v < 0) | (u >= mtx.shape[0]) | (v >= mtx.shape[1]))[0]
                if len(xx) > 0:
                    (u,v) = [np.array(x) for x in (u,v)]
                    u[xx] = 0
                    v[xx] = 0
                res = flattest(self.edge_matrix[(u,v)]) - 1
            else:
                (a,b,c) = m
                mtx = self.face_matrix
                bc = b*mtx.shape[0] + c
                xx = np.where((a >= mtx.shape[0]) | (bc >= mtx.shape[1]))[0]
                if len(xx) > 0:
                    bc[xx] = 0
                    a = np.array(a)
                    a[xx] = 0
                res = flattest(mtx[(a, bc)]) - 1
        else: return self.vertex_index.get(index, None)
        ii = np.where(res < 0)[0]
        if len(ii) == 0: return res
        res = res.astype(np.object)
        res[ii] = None
        return res
    def __call__(self, index):
        if pimms.is_scalar(index): return self.vertex_index.get(index, None)
        elif is_tuple(index):      return tuple([self[ii] for ii in index])
        else:                      return np.reshape(self[flattest(index)], np.shape(index))

@pimms.immutable
class Tesselation(VertexSet):
    '''
    A Tesselation object represents a triangle mesh with no particular coordinate embedding.
    Tesselation inherits from the immutable class VertexSet, which provides functionality for
    tracking the properties of the tesselation's vertices.
    '''
    def __init__(self, faces, properties=None, meta_data=None):
        self.faces = faces
        # we don't call VertexSet.__init__ because it sets vertex labels, which we have changed in
        # this class to a value instead of a param; instead we just set _properties directly
        self._properties = properties
        self.meta_data = meta_data

    # The immutable parameters:
    @pimms.param
    def faces(tris):
        '''
        tess.faces is a read-only numpy integer matrix of the triangle indices that make-up; the
          given tesselation object; the matrix is (3 x m) where m is the number of triangles, and
          the cells are valid indices into the rows of the coordinates matrix.
        '''
        tris = np.asarray(tris, dtype=np.int)
        if tris.shape[0] != 3:
            tris = tris.T
            if tris.shape[0] != 3:
                raise ValueError('faces must be a (3 x m) or (m x 3) matrix')
        return pimms.imm_array(tris)

    # The immutable values:
    @pimms.value
    def labels(faces):
        '''
        tess.labels is an array of the integer vertex labels; subsampling the tesselation object
        will maintain vertex labels (but not indices).
        '''
        return pimms.imm_array(np.unique(faces))
    @pimms.value
    def face_count(faces):
        '''
        tess.face_count is the number of faces in the given tesselation.
        '''
        return faces.shape[1]
    @pimms.value
    def face_index(faces):
        '''
        tess.face_index is a mapping that indexes the faces by vertex labels (not vertex indices).
        '''
        idx = {}
        for (a,b,c,i) in zip(faces[0], faces[1], faces[2], range(faces.shape[1])):
            idx[(a,b,c)] = i
            idx[(b,c,a)] = i
            idx[(c,b,a)] = i
            idx[(a,c,b)] = i
            idx[(b,a,c)] = i
            idx[(c,a,b)] = i
        return pyr.pmap(idx)
    @pimms.value
    def edge_data(faces):
        '''
        tess.edge_data is a mapping of data relevant to the edges of the given tesselation.
        '''
        limit = np.max(faces) + 1
        all_edges = np.hstack([[faces[0],faces[1]], [faces[1],faces[2]], [faces[2],faces[0]]])
        (edge_list,cnt) = np.unique(np.sort(all_edges, axis=0), axis=1, return_counts=True)
        rng = np.arange(edge_list.shape[1])
        poss_edges = np.hstack([edge_list, np.flipud(edge_list)])
        poss_idcs = np.concatenate([rng,rng])
        idx = {k:ii for (k,ii) in zip(zip(poss_edges[0],poss_edges[1]), poss_idcs)}
        rng = np.arange(faces.shape[1])
        face_idcs = np.concatenate([rng,rng,rng])
        esrt = np.sort(all_edges, axis=0)
        edge2face = {k:ii for (k,ii) in zip(zip(*all_edges), face_idcs)}
        for (e,er) in zip(zip(*edge_list), zip(*np.flipud(edge_list))):
            tup = tuple([eff for q in (e,er) for eff in [edge2face.get(q)] if eff is not None])
            assert(len(tup) > 0)
            for eidx in (e,er): edge2face[eidx] = tup
        edge_list.setflags(write=False)
        return pyr.m(edges=edge_list,
                     edge_index=pyr.pmap(idx),
                     edge_face_index=pyr.pmap(edge2face))
    @pimms.value
    def edges(edge_data):
        '''
        tess.edges is a (2 x p) numpy array containing the p edge pairs that are included in the
        given tesselation.
        '''
        return edge_data['edges']
    @pimms.value
    def edge_count(edges):
        '''
        tess.edge_count is the number of edges in the given tesselation.
        '''
        return edges.shape[1]
    @pimms.value
    def edge_index(edge_data):
        '''
        tess.edge_index is a mapping that indexes the edges by vertex labels (not vertex indices).
        '''
        return edge_data['edge_index']
    @pimms.value
    def edge_face_index(edge_data):
        '''
        tess.edge_face_index is a mapping that indexes the edges by vertex labels (not vertex
          indices) to a face index or pair of face indices. So for an edge from the vertex labeled
          u to the vertex labeled v, index.edge_face_index[(u,v)] is a tuple of the faces that are
          adjacent to the edge (u,v).
        '''
        return edge_data['edge_face_index']
    @pimms.value
    def edge_faces(edges, edge_face_index):
        '''
        tess.edge_faces is a tuple that contains one element per edge; each element
        tess.edge_faces[i] is a tuple of the 1 or two face indices of the faces that contain the
        edge with edge index i.
        '''
        return tuple([edge_face_index[e] for e in zip(*edges)])
    @pimms.value
    def face_neighbors(edge_faces, face_count):
        '''
        tess.face_neighbors is a tuple that contains one element per face; each element
        tess.face_neighbors[i] is a tuple of the 0-3 face indices of the faces that are adjacent to
        the face with index i.
        '''
        q = [[] for _ in range(face_count)]
        for fs in edge_faces:
            if len(fs) == 2:
                (a,b) = fs
                q[a].append(b)
                q[b].append(a)
        return tuple([tuple(qq) for qq in q])
    @pimms.value
    def vertex_index(indices, labels):
        '''
        tess.vertex_index is an index of vertex-label to vertex index for the given tesselation.
        '''
        return pyr.pmap({v:i for (i,v) in zip(indices, labels)})
    @pimms.value
    def index(vertex_index, edge_index, face_index):
        '''
        tess.index is a TesselationIndex object that indexed the faces, edges, and vertices in the
        given tesselation object. Vertex, edge, and face indices can be looked-up using the
        following syntax:
          # Note that in all of these, u, v, and w are vertex *labels*, not vertex indices:
          tess.index[u]       # for vertex-label u => vertex-index of u
          tess.index[(u,v)]   # for edge (u,v) => edge-index of (u,v)
          tess.index[(u,v,w)] # for face (u,v,w) => face-index of (u,v,w)
        Alternately, one may want to obtain the vertex indices of an object without obtaining the
        indices for it directly:
          # Assume ui,vi,wi are the vertex indices of the vertex labels u,v,w:
          tess.index(u)       # for vertex-label u => ui
          tess.index((u,v))   # for edge (u,v) => (ui,vi)
          tess.index((u,v,w)) # for face (u,v,w) => (ui,vi,wi)
        Finally, in addition to passing individual vertices or tuples, you may pass an appropriately
        sized vector (for vertices) or matrix (for edges and faces), and the result will be a list
        of the appropriate indices or an identically-sized array with the vertex indices.
        '''
        idx = TesselationIndex(vertex_index, edge_index, face_index)
        return idx.persist()
    @pimms.value
    def indexed_edges(edges, vertex_index):
        '''
        tess.indexed_edges is identical to tess.edges except that each element has been indexed.
        '''
        return pimms.imm_array([[vertex_index[u] for u in row] for row in edges])
    @pimms.value
    def indexed_faces(faces, vertex_index):
        '''
        tess.indexed_faces is identical to tess.faces except that each element has been indexed.
        '''
        return pimms.imm_array([[vertex_index[u] for u in row] for row in faces])
    @pimms.value
    def vertex_edge_index(labels, edges):
        '''
        tess.vertex_edge_index is a map whose keys are vertices and whose values are tuples of the
        edge indices of the edges that contain the relevant vertex.
        '''
        d = {k:[] for k in labels}
        for (i,(u,v)) in enumerate(edges.T):
            d[u].append(i)
            d[v].append(i)
        return pyr.pmap({k:tuple(v) for (k,v) in six.iteritems(d)})
    @pimms.value
    def vertex_edges(labels, vertex_edge_index):
        '''
        tess.vertex_edges is a tuple whose elements are tuples of the edge indices of the edges
        that contain the relevant vertex; i.e., for vertex u with vertex index i,
        tess.vertex_edges[i] will be a tuple of the edges indices that contain vertex u.
        '''
        return tuple([vertex_edge_index[u] for u in labels])
    @pimms.value
    def vertex_face_index(labels, faces):
        '''
        tess.vertex_face_index is a map whose keys are vertices and whose values are tuples of the
        indices of the faces that contain the relevant vertex.
        '''
        d = {k:[] for k in labels}
        for (i,(u,v,w)) in enumerate(faces.T):
            d[u].append(i)
            d[v].append(i)
            d[w].append(i)
        return pyr.pmap({k:tuple(v) for (k,v) in six.iteritems(d)})
    @pimms.value
    def vertex_faces(labels, vertex_face_index):
        '''
        tess.vertex_faces is a tuple whose elements are tuples of the face indices of the faces
        that contain the relevant vertex; i.e., for vertex u with vertex index i,
        tess.vertex_faces[i] will be a tuple of the face indices that contain vertex u.
        '''
        return tuple([vertex_face_index[u] for u in labels])
    @staticmethod
    def _order_neighborhood(edges):
        fres = [edges[0][1]]
        bres = [edges[0][1]]
        (fi,bi) = (0,0)
        for _ in edges:
            for e in edges:
                if e[0] == fres[fi]:
                    fres.append(e[1])
                    fi += 1
                    break
        if fres[-1] == fres[0]:
            fres = fres[:-1]
        else:
            for _ in range(len(edges) - fi):
                for e in edges:
                    if e[1] == bres[bi]:
                        bres.append(e[0])
                        bi += 1
        return tuple(reversed(bres[1:])) + tuple(fres)
    @pimms.value
    def neighborhoods(labels, faces, vertex_faces):
        '''
        tess.neighborhoods is a tuple whose contents are the neighborhood of each vertex in the
        tesselation.
        '''
        faces = faces.T
        nedges = [[((f[0], f[1]) if f[2] == u else (f[1], f[2]) if f[0] == u else (f[2], f[0]))
                   for f in faces[list(fs)]]
                  for (u, fs) in zip(labels, vertex_faces)]
        return tuple([Tesselation._order_neighborhood(nei) for nei in nedges])
    @pimms.value
    def indexed_neighborhoods(vertex_index, neighborhoods):
        '''
        tess.indexed_neighborhoods is a tuple whose contents are the neighborhood of each vertex in
        the given tesselation; this is identical to tess.neighborhoods except this gives the vertex
        indices where tess.neighborhoods gives the vertex labels.
        '''
        return tuple([tuple([vertex_index[u] for u in nei]) for nei in neighborhoods])

    # Requirements/checks
    @pimms.require
    def validate_properties(vertex_count, _properties):
        '''
        tess.validate_properties requres that all non-builtin properties have the same number of
          entries as the there are vertices in the tesselation.
        '''
        if _properties is None or len(_properties.column_names) == 0:
            return True
        if vertex_count != _properties.row_count:
            ns = (_properties.row_count, vertex_count)
            raise ValueError('_properties has incorrect number of entries %d; (should be %d)' % ns)
        return True

    # Normal Methods
    def __repr__(self):
        return 'Tesselation(<%d faces>, <%d vertices>)' % (self.face_count, self.vertex_count)
    def make_mesh(self, coords, properties=None, meta_data=None):
        '''
        tess.make_mesh(coords) yields a Mesh object with the given coordinates and with the
          meta_data and properties inhereted from the given tesselation tess.
        '''
        md = self.meta_data
        if meta_data is not None: md = pimms.merge(md, meta_data)
        return Mesh(self, coords, meta_data=md, properties=properties)
    def subtess(self, vertices, tag=None):
        '''
        tess.subtess(vertices) yields a sub-tesselation of the given tesselation object that only
          contains the given vertices, which may be specified as a boolean vector or as a list of
          vertex labels. Faces and edges are trimmed automatically, but the vertex labels for the
          new vertices remain the same as in the original graph.
        The optional argument tag may be set to True, in which case the new tesselation's meta-data
        will contain the key 'supertess' whose value is the original tesselation tess; alternately
        tag may be a string in which case it is used as the key name in place of 'supertess'.
        '''
        vertices = np.asarray(vertices)
        if len(vertices) != self.vertex_count or \
           not np.array_equal(vertices, np.asarray(vertices, np.bool)):
            tmp = self.index(vertices)
            vertices = np.zeros(self.vertex_count, dtype=np.bool)
            vertices[tmp] = 1
        vidcs = self.indices[vertices]
        if len(vidcs) == self.vertex_count: return self
        fsum = np.sum([vertices[f] for f in self.indexed_faces], axis=0)
        fids = np.where(fsum == 3)[0]
        faces = self.faces[:,fids]
        vidcs = self.index(np.unique(faces))
        props = self._properties
        if props is not None and len(props) > 1: props = props[vidcs]
        md = self.meta_data.set(tag, self) if pimms.is_str(tag)   else \
             self.meta_data.set('supertess', self) if tag is True else \
             self.meta_data
        dat = {'faces': faces}
        if props is not self._properties: dat['_properties'] = props
        if md is not self.meta_data: dat['meta_data'] = md
        return self.copy(**dat)
    def select(self, fn, tag=None):
        '''
        tess.select(fn) is equivalent to tess.subtess(tess.properties.map(fn)); any vertex whose
          property data yields True or 1 will be included in the new subtess and all other vertices
          will be excluded.
        The optional parameter tag is used identically as in tess.subtess().
        '''
        return self.subtess(self.map(fn), tag=tag)

def is_tess(t):
    '''
    is_tess(t) yields True if t is a Tesselation object and False otherwise.
    '''
    return isinstance(t, Tesselation)
def tess(faces, properties=None, meta_data=None):
    '''
    tess(faces) yields a Tesselation object from the given face matrix.
    '''
    return Tesselation(faces, properties=properties, meta_data=meta_data)

@pimms.immutable
class Mesh(VertexSet):
    '''
    A Mesh object represents a triangle mesh in either 2D or 3D space.
    To construct a mesh object, use Mesh(tess, coords), where tess is either a Tesselation object or
    a matrix of face indices and coords is a coordinate matrix for the vertices in the given
    tesselation or face matrix.
    '''

    def __init__(self, faces, coordinates, meta_data=None, properties=None):
        self.coordinates = coordinates
        self.tess = faces
        self.meta_data = meta_data
        self._properties = properties

    # The immutable parameters:
    @pimms.param
    def coordinates(crds):
        '''
        mesh.coordinates is a read-only numpy array of size (d x n) where d is the number of
          dimensions and n is the number of vertices in the mesh.
        '''
        crds = np.asarray(crds)
        if crds.shape[0] != 2 and crds.shape[0] != 3:
            crds = crds.T
            if crds.shape[0] != 2 and crds.shape[0] != 3:
                raise ValueError('coordinates must be a (d x n) or (n x d) array where d is 2 or 3')
        return pimms.imm_array(crds)
    @pimms.param
    def tess(tris):
        '''
        mesh.tess is the Tesselation object that represents the triangle tesselation of the given
        mesh object.
        '''
        if not isinstance(tris, Tesselation):
            try: tris = Tesselation(tris)
            except Exception: raise ValueError('mesh.tess must be a Tesselation object')
        return tris.persist()

    # The immutable values:
    @pimms.value
    def labels(tess):
        '''
        mesh.labels is the list of vertex labels for the given mesh.
        '''
        return tess.labels
    @pimms.value
    def indices(tess):
        '''
        mesh.indices is the list of vertex indicess for the given mesh.
        '''
        return tess.indices
    @pimms.value
    def properties(_properties, tess, coordinates):
        '''
        mesh.properties is the pimms Itable object of properties known to the given mesh.
        '''
        pp = {} if _properties is None else _properties
        tp = {} if tess.properties is None else tess.properties
        # note that tess.properties always already has labels and indices included
        if _properties is tess.properties:
            return pimms.itable(pp, {'coordinates': coordinates.T}).persist()
        else:
            return pimms.itable(tp, pp, {'coordinates': coordinates.T}).persist()
    @pimms.value
    def edge_coordinates(tess, coordinates):
        '''
        mesh.edge_coordinates is the (2 x d x p) array of the coordinates that define each edge in
          the given mesh; d is the number of dimensions that define the vertex positions in the mesh
          and p is the number of edges in the mesh.
        '''
        return pimms.imm_array([coordinates[:,e] for e in tess.indexed_edges])
    @pimms.value
    def face_coordinates(tess, coordinates):
        '''
        mesh.face_coordinates is the (3 x d x m) array of the coordinates that define each face in
          the given mesh; d is the number of dimensions that define the vertex positions in the mesh
          and m is the number of triange faces in the mesh.
        '''
        return pimms.imm_array([coordinates[:,f] for f in tess.indexed_faces])
    @pimms.value
    def edge_centers(edge_coordinates):
        '''
        mesh.edge_centers is the (d x n) array of the centers of each edge in the given mesh.
        '''
        return pimms.imm_array(0.5 * np.sum(edge_coordinates, axis=0))
    @pimms.value
    def face_centers(face_coordinates):
        '''
        mesh.face_centers is the (d x n) array of the centers of each triangle in the given mesh.
        '''
        return pimms.imm_array(np.sum(face_coordinates, axis=0) / 3.0)
    @pimms.value
    def face_normals(face_coordinates):
        '''
        mesh.face_normals is the (3 x m) array of the outward-facing normal vectors of each
          triangle in the given mesh. If mesh is a 2D mesh, these are all either [0,0,1] or
          [0,0,-1].
        '''
        u01 = face_coordinates[1] - face_coordinates[0]
        u02 = face_coordinates[2] - face_coordinates[0]
        if len(u01) == 2:
            zz = np.zeros((1,tmp.shape[1]))
            u01 = np.concatenate((u01,zz))
            u02 = np.concatenate((u02,zz))
        xp = np.cross(u01, u02, axisa=0, axisb=0).T
        norms = np.sqrt(np.sum(xp**2, axis=0))
        wz = np.isclose(norms, 0)
        return pimms.imm_array(xp * (np.logical_not(wz) / (norms + wz)))
    @pimms.value
    def vertex_normals(face_normals, tess):
        '''
        mesh.vertex_normals is the (3 x n) array of the outward-facing normal vectors of each
          vertex in the given mesh. If mesh is a 2D mesh, these are all either [0,0,1] or
          [0,0,-1].
        '''
        tmp = np.array([np.sum(face_normals[:,fs], axis=1) for fs in tess.vertex_faces]).T
        norms = np.sqrt(np.sum(tmp ** 2, axis=0))
        wz = np.isclose(norms, 0)
        return pimms.imm_array(tmp * (np.logical_not(wz) / (norms + wz)))
    @pimms.value
    def face_angle_cosines(face_coordinates):
        '''
        mesh.face_angle_cosines is the (3 x d x n) matrix of the cosines of the angles of each of
        the faces of the mesh; d is the number of dimensions of the mesh embedding and n is the
        number of faces in the mesh.
        '''
        X = face_coordinates
        X = np.asarray([x * (zs / (xl + np.logical_not(zs)))
                        for x  in [X[1] - X[0], X[2] - X[1], X[0] - X[2]]
                        for xl in [np.sqrt(np.sum(x**2, axis=0))]
                        for zs in [np.isclose(xl, 0)]])
        dps = np.asarray([np.sum(x1*x2, axis=0) for (x1,x2) in zip(X, -np.roll(X, 1, axis=0))])
        dps.setflags(write=False)
        return dps
    @pimms.value
    def face_angles(face_angle_cosines):
        '''
        mesh.face_angles is the (3 x d x n) matrix of the angles of each of the faces of the mesh;
        d is the number of dimensions of the mesh embedding and n is the number of faces in the
        mesh.
        '''
        tmp = np.arccos(face_angle_cosines)
        tmp.setflags(write=False)
        return tmp
    @pimms.value
    def face_areas(face_coordinates):
        '''
        mesh.face_areas is the length-m numpy array of the area of each face in the given mesh.
        '''
        return pimms.imm_array(triangle_area(*face_coordinates))
    @pimms.value
    def edge_lengths(edge_coordinates):
        '''
        mesh.edge_lengths is a numpy array of the lengths of each edge in the given mesh.
        '''
        tmp = np.sqrt(np.sum((edge_coordinates[1] - edge_coordinates[0])**2, axis=0))
        tmp.setflags(write=False)
        return tmp
    @pimms.value
    def face_hash(face_centers):
        '''
        mesh.face_hash yields the scipy spatial hash of triangle centers in the given mesh.
        '''
        try:              return space.cKDTree(face_centers.T)
        except Exception: return space.KDTree(face_centers.T)
    @pimms.value
    def vertex_hash(coordinates):
        '''
        mesh.vertex_hash yields the scipy spatial hash of the vertices of the given mesh.
        '''
        try:              return space.cKDTree(coordinates.T)
        except Exception: return space.KDTree(coordinates.T)

    # requirements/validators
    @pimms.require
    def validate_tess(tess, coordinates):
        '''
        mesh.validate_tess requires that all faces be valid indices into mesh.coordinates.
        '''
        (d,n) = coordinates.shape
        if tess.vertex_count != n:
            raise ValueError('mesh coordinate matrix size does not match vertex count')
        if d != 2 and d != 3:
            raise ValueError('Only 2D and 3D meshes are supported')
        return True
    @pimms.value
    def repr(coordinates, vertex_count, tess):
        '''
        mesh.repr is the representation string returned by mesh.__repr__().
        '''
        args = (coordinates.shape[0], tess.face_count, vertex_count)
        return 'Mesh(<%dD>, <%d faces>, <%d vertices>)' % args

    # Normal Methods
    def __repr__(self):
        return self.repr

    # this is tricky: because mesh inherits properties from its tess, we have to also update the
    # tess when we do this
    def wout_prop(self, *args):
        '''
        obj.wout_property(...) yields a duplicate of the given object with the given properties
          removed from it. The properties may be specified as a sequence of column names or lists of
          column names.
        '''
        new_tess = self.tess.wout_prop(*args)
        new_mesh = self if self.tess is new_tess else self.copy(tess=new_tess)
        pp = new_mesh._properties
        for a in args: pp = pp.discard(a)
        return new_mesh if pp is new_mesh._properties else new_mesh.copy(_properties=pp)

    def submesh(self, vertices, tag=None, tag_tess=Ellipsis):
        '''
        mesh.submesh(vertices) yields a sub-mesh of the given mesh object that only contains the
          given vertices, which may be specified as a boolean vector or as a list of vertex labels.
          Faces and edges are trimmed automatically, but the vertex labels for the new vertices
          remain the same as in the original graph.
        The optional argument tag may be set to True, in which case the new mesh's meta-data
        will contain the key 'supermesh' whose value is the original tesselation tess; alternately
        tag may be a string in which case it is used as the key name in place of 'supermesh'.
        Additionally, the optional argument tess_tag is the equivalent to the tag option except that
        it is passed along to the mesh's tesselation object; the value Ellipsis (default) can be
        given in order to specify that tag_tess should take the same value as tag.
        '''
        subt = self.tess.subtess(vertices, tag=tag_tess)
        if subt is self.tess: return self
        vidcs = self.tess.index(subt.labels)
        props = self._properties
        if props is not None and props.row_count > 0:
            props = props[vidcs]
        coords = self.coordinates[:,vidcs]
        md = self.meta_data.set(tag, self) if pimms.is_str(tag)   else \
             self.meta_data.set('supermesh', self) if tag is True else \
             self.meta_data
        dat = {'coordinates': coords, 'tess': subt}
        if props is not self._properties: dat['_properties'] = props
        if md is not self.meta_data: dat['meta_data'] = md
        return self.copy(**dat)
    def select(self, fn, tag=None, tag_tess=Ellipsis):
        '''
        mesh.select(fn) is equivalent to mesh.subtess(mesh.map(fn)); any vertex whose
          property data yields True or 1 will be included in the new submesh and all other vertices
          will be excluded.
        The optional parameters tag and tag_tess is used identically as in mesh.submesh().
        '''
        return self.submesh(self.map(fn), tag=tag, tag_tess=tag_tess)
    
    # True if the point is in the triangle, otherwise False; tri_no is an index into the faces
    def is_point_in_face(self, tri_no, pt):
        '''
        mesh.is_point_in_face(face_id, crd) yields True if the given coordinate crd is in the face
          of the given mesh with the given face id (i.e., mesh.faces[:,face_id] gives the vertex
          indices for the face in question).
        The face_id and crd values may be lists as well; either they may be lists the same length
        or one may be a list.
        '''
        pt = np.asarray(pt)
        tri_no = np.asarray(tri_no)
        if len(tri_no.shape) == 0:
            tri_no = [tri_no]
            tri = np.transpose([self.coordinates[:,t] for t in self.tess.indexed_faces[:,tri_no]],
                               (2,0,1))
            return point_in_triangle(tri, pt)[0]
        else:
            tri = np.transpose([self.coordinates[:,t] for t in self.tess.indexed_faces[:,tri_no]],
                               (2,0,1))
            return point_in_triangle(tri, pt)

    def _find_triangle_search(self, x, k=24, searched=set([]), n_jobs=-1):
        # This gets called when a container triangle isn't found; the idea is that k should
        # gradually increase until we find the container triangle; if k passes the max, then
        # we give up and assume no triangle is the container
        if k >= 288: return None
        try:              (d,near) = self.facee_hash.query(x, k=k, n_jobs=n_jobs)
        except Exception: (d,near) = self.facee_hash.query(x, k=k)
        near = [n for n in near if n not in searched]
        searched = searched.union(near)
        tri_no = next((kk for kk in near if self.is_point_in_face(kk, x)), None)
        return (tri_no if tri_no is not None
                else self._find_triangle_search(x, k=(2*k), searched=searched))
    
    def nearest_vertex(self, pt, n_jobs=-1):
        '''
        mesh.nearest_vertex(pt) yields the id number of the nearest vertex in the given
        mesh to the given point pt. If pt is an (n x dims) matrix of points, an id is given
        for each column of pt.
        '''
        try:              (d,near) = self.vertex_hash.query(pt, k=1, n_jobs=n_jobs)
        except Exception: (d,near) = self.vertex_hash.query(pt, k=1)
        return near

    def point_in_plane(self, tri_no, pt):
        '''
        r.point_in_plane(id, pt) yields the distance from the plane of the id'th triangle in the
        registration r to the given pt and the point in that plane as a tuple (d, x).
        If id or pt or both are lists, then the elements d and x of the returned tuple are a vector
        and matrix, respectively, where the size of x is (dims x n).
        '''
        tri_no = np.asarray(tri_no)
        pt = np.asarray(pt)
        if tri_no.shape is () and len(pt.shape) == 1:
            tx = self.coordinates[:, self.tess.faces[:, tri_no]]
            n = self.face_normals[:, tri_no]
            d = np.dot(n, pt - tx[0])
            return (np.abs(d), pt - n*d)
        if len(pt.shape) == 1:
            pt = npml.repmat(pt, len(tri_no), 1).T
        else:
            pt = pt.T if pt.shape[0] != self.coordinates.shape[0] else pt
            tri_no = np.full(pt.shape[1], tri_no, dtype=np.int)
        tx0 = self.coordinates[:,  self.tess.faces[0,tri_no]]
        n   = self.face_normals[:, tri_no]
        d   = np.sum(n * (pt - tx0), axis=0)
        return (np.abs(d), pt - n*d)
    
    def nearest_data(self, pt, k=2, n_jobs=-1):
        '''
        mesh.nearest_data(pt) yields a tuple (k, d, x) of the matrix x containing the point(s)
        nearest the given point(s) pt that is/are in the mesh; a vector d if the distances between
        the point(s) pt and x; and k, the face index/indices of the triangles containing the 
        point(s) in x.
        Note that this function and those of this class are made for spherical meshes and are not
        intended to work with other kinds of complex topologies; though they might work 
        heuristically.
        '''
        pt = np.asarray(pt, dtype=np.float32)
        if len(pt.shape) == 1:
            r = self.nearest_data([pt], k=k, n_jobs=n_jobs)[0];
            return (r[0][0], r[1][0], r[2][0])
        pt = pt.T if pt.shape[0] == self.coordinates.shape[0] else pt
        try:              (d, near) = self.face_hash.query(pt, k=k, n_jobs=n_jobs)
        except Exception: (d, near) = self.face_hash.query(pt, k=k)
        ids = [tri_no if tri_no is not None else self._find_triangle_search(x, 2*k, set(near_i))
               for (x, near_i) in zip(pt, near)
               for tri_no in [next((k for k in near_i if self.is_point_in_face(k, x)), None)]]
        pips = [self.point_in_plane(i, p) if i is not None else (0, None)
                for (i,p) in zip(ids, pt)]
        return (np.asarray(ids),
                np.asarray([d[0] for d in pips]),
                np.asarray([d[1] for d in pips]))

    def nearest(self, pt, k=2, n_jobs=-1):
        '''
        mesh.nearest(pt) yields the point in the given mesh nearest the given array of points pts.
        '''
        dat = self.nearest_data(pt, n_jobs=n_jobs)
        return dat[2]

    def nearest_vertex(self, x, n_jobs=-1):
        '''
        mesh.nearest_vertex(x) yields the vertex index or indices of the vertex or vertices nearest
          to the coordinate or coordinates given in x.
        '''
        x = np.asarray(x)
        if len(x.shape) == 1: return self.nearest_vertex([x], n_jobs=n_jobs)[0]
        if x.shape[0] == self.coordinates.shape[0]: x = x.T
        n = self.coordinates.shape[1]
        try:              (_, nei) = self.vertex_hash.query(x, k=1, n_jobs=n_jobs)
        except Exception: (_, nei) = self.vertex_hash.query(x, k=1)
        return nei

    def distance(self, pt, k=2, n_jobs=1):
        '''
        mesh.distance(pt) yields the distance to the nearest point in the given mesh from the points
        in the given matrix pt.
        '''
        dat = self.nearest_data(pt)
        return dat[1]

    def container(self, pt, k=2, n_jobs=-1):
        '''
        mesh.container(pt) yields the id number of the nearest triangle in the given
        mesh to the given point pt. If pt is an (n x dims) matrix of points, an id is given
        for each column of pt.

        Implementation Note:
          This method will fail to find the container triangle of a point if you have a very odd
          geometry; the requirement for this condition is that, for a point p contained in a
          triangle t with triangle center x0, there are at least n triangles whose centers are
          closer to p than x0 is to p. The value n depends on your starting parameter k, but is
          approximately 256.
        '''
        pt = np.asarray(pt)
        if len(pt.shape) == 1:
            return self.container([pt], k=k, n_jobs=n_jobs)[0]
        else:
            if pt.shape[0] == self.coordinates.shape[0]: pt = pt.T
            tcount = self.tess.faces.shape[1]
            max_k = 256 if tcount > 256 else tcount
            if k > tcount: k = tcount
            def try_nearest(sub_pts, cur_k=k, top_i=0, near=None):
                res = np.full(len(sub_pts), None, dtype=np.object)
                if k != cur_k and cur_k > max_k: return res
                if near is None:
                    try:              near = self.face_hash.query(sub_pts, k=cur_k, n_jobs=n_jobs)[1]
                    except Exception: near = self.face_hash.query(sub_pts, k=cur_k)[1]
                # we want to try the nearest then recurse on those that didn't match...
                guesses = near[:, top_i]
                in_tri_q = self.is_point_in_face(guesses, sub_pts)
                res[in_tri_q] = guesses[in_tri_q]
                if in_tri_q.all(): return res
                # recurse, trying the next nearest points
                out_tri_q = np.logical_not(in_tri_q)
                sub_pts = sub_pts[out_tri_q]
                top_i += 1
                res[out_tri_q] = (try_nearest(sub_pts, cur_k*2, top_i, None)
                                  if top_i == cur_k else
                                  try_nearest(sub_pts, cur_k, top_i, near[out_tri_q]))
                return res
            res = np.full(len(pt), None, dtype=np.object)
            # filter out points that aren't close enough to be in a triangle:
            (dmins, dmaxs) = [[f(x[np.isfinite(x)]) for x in self.coordinates]
                              for f in [np.min, np.max]]
            finpts = np.isfinite(np.sum(pt, axis=1))
            if finpts.all():
                if pt.shape[1] == 2:
                    inside_q = reduce(np.logical_and,
                                      [(x >= mn)&(x <= mx) for (x,mn,mx) in zip(pt.T,dmins,dmaxs)])
                else:
                    inside_q = np.full(len(pt), True, dtype=np.bool)
            else:
                inside_q = np.full(len(pt), False, dtype=np.bool)
                if pt.shape[1] == 2:
                    inside_q[finpts] = reduce(
                        np.logical_and,
                        [(x >= mn)&(x <= mx) for (x,mn,mx) in zip(pt[finpts].T,dmins,dmaxs)])
                else:
                    inside_q[finpts] = True
            if not inside_q.any(): return res
            res[inside_q] = try_nearest(pt[inside_q])
            return res

    @staticmethod
    def scale_interpolation(interp, mask=None, weights=None):
        '''
        Mesh.scale_interpolation(interp) yields a duplicate of interp in which all rows of the
          interpolation matrix have a sum of 1 or contain a nan element; those rows that correspond
          to interpolation points that are either not in the mask or have zero weight contributing
          to them will have a nan value somewhere that indicates that the interpolated value should
          always be nan.
        Mesh.scale_interpolation(interp, mask) additionally applies the given mask to the
          interpolation matrix; any interpolation point whose nearest neighbor is a vertex outside
          of the given mask will be assigned a value of numpy.nan upon interpolation with the
          resulting interpolation matrix.
        Mesh.scale_interpolation(interp, mask, weights) additionally applies the given vertex 
          weights to the interpolation matrix.

        The value mask may be specified as either None (no interpolation), a boolean array with the
        same number of elements as there are columns in interp, or a list of indices. The values in
        the mask are considered valid while the values not in the mask are considered invalid. The
        string value 'all' is also a valid mask (equivalent to no mask, or all elements in the
        mask).

        The interp argument should be a scipy.sparse.*_matrix; the object will not be modified
        in-place except to run eliminate_zeros(), and the returned matrix will always be of the
        csr_matrix type. Typical usage would be:
        interp_matrix = scipy.sparse.lil_matrix((n, self.vertex_count))
        # create your interpolation matrix here...
        return Mesh.rescale_interpolation(interp_matrix, mask_arg, weights_arg)
        '''
        if sps.issparse(interp): interp = interp.tocsr().copy()
        if weights is None and mask is None: return interp
        (m,n) = interp.shape # n: no. of vertices in mesh; m: no. points being interpolated
        # setup the weights:
        if weights is None:           weights = np.ones(n, dtype=np.float)
        elif np.shape(weights) == (): weights = np.full(n, weights, dtype=np.float)
        else:                         weights = np.array(weights, dtype=np.float)
        # figure out the mask
        if mask is not None:
            mask = self.mask(mask, indices=True)
            nmask = np.setdiff1d(np.arange(n), mask)
            weights[nmask] = np.nan
        # if the weights are nan, we only want that to apply if the heavist weight is nan; otherwise
        # we interpolate with what's not nan
        nansq = np.logical_not(np.isfinite(flattest(interp.sum(axis=1))))
        interp.data[~np.isfinite(interp.data)] = 0
        heaviest = flattest(interp.argmax(axis=1))
        hvals = interp[(np.arange(m), heaviest)]
        hnots = np.isclose(hvals, 0) | ~np.isfinite(weights[heaviest]) | nansq
        whnan = np.where(hnots)[0]
        # we can now eliminate the nan weights
        weights[~np.isfinite(weights)] = 0
        # and scale the interpolation matrix
        interp *= sps.diag(weights)
        # now, we may need to scale the rows
        rsums = flattest(interp.sum(axis=1))
        if not np.isclose(rsums, 1).all(): interp = sps.diag(zinv(rsums)).dot(interp)
        # Then put the nans back where they're needed
        if len(whnan) > 0: interp[(np.zeros(len(whnan), dtype=np.int), whnan)] = np.nan
        interp.eliminate_zeros()
        return interp
    def nearest_interpolation(self, coords, n_jobs=-1):
        '''
        mesh.nearest_interpolation(x) yields an interpolation matrix for the given coordinate or
          coordinate matrix x. An interpolation matrix is just a sparce array M such that for a
          column vector u with the same number of rows as there are vertices in mesh, M * u is the
          interpolated values of u at the coordinates in x.
        '''
        if is_address(coords):
            (fs, xs) = address_data(coords, dims=2, strict=False)
            ii = np.where(np.isfinite(np.sum(xs, axis=0)))[0]
            xs = np.vstack([xs, [1 - np.sum(xs, axis=0)]])
            nv = fs[(np.argmax(xs, axis=0), np.arange(fs.shape[1]))]
            nv = self.tess.index(nv[:,ii])
        else:
            coords = np.asarray(coords)
            if coords.shape[0] == self.coordinates.shape[0]: coords = coords.T
            nv = self.nearest_vertex(coords, n_jobs=n_jobs)
            ii = np.arange(len(nv))
        n = self.vertex_count
        m = len(ii)
        return sps.csr_matrix((np.ones(m, dtype=np.int), (np.arange(m)[ii], nv)),
                              shape=(m,n),
                              dtype=np.int)
    def linear_interpolation(self, coords, n_jobs=-1):
        '''
        mesh.linear_interpolation(x) yields an interpolation matrix for the given coordinate or 
          coordinate matrix x. An interpolation matrix is just a sparce array M such that for a
          column vector u with the same number of rows as there are vertices in mesh, M * u is the
          interpolated values of u at the coordinates in x.

        The coordinate matrix x may alternately be an address-data map, in which case interpolation
        is considerably faster.
        '''
        if is_address(coords):
            (fs, xs) = address_data(coords, dims=2, strict=False)
            xs = np.vstack([xs, [1 - np.sum(xs, axis=0)]])
            (n, m) = (xs.shape[1], self.vertex_count)
            ii = np.where(np.isfinite(np.sum(xs, axis=0)))[0]
            fsi = self.tess.index(fs[:,ii])
            return sps.csr_matrix(
                (xs[:,ii].flatten(), (np.tile(np.arange(n)[ii], 3), fsi.flatten())),
                shape=(n, m))
        else: return self.linear_interpolation(self.address(coords, n_jobs=n_jobs))
    def heaviest_interpolation(self, coords, n_jobs=-1):
        '''
        mesh.heaviest_interpolation(x) yields an interpolation matrix for the given coordinate or 
          coordinate matrix x. The heaviest interpolation matrix is like a linear interpolation
          except that it assigns to each value the vertex with the maximum weight.
        '''
        li = self.linear_interpolation(coords, n_jobs=n_jobs)
        # rescale the rows
        mx = flattest(li.argmax(axis=1))
        wh = np.where(~np.isclose(flattest(li.sum(axis=1)), 0))[0]
        mx = mx[wh]
        return sps.csr_matrix((np.ones(len(mx)), (wh, mx)), shape=li.shape, dtype=np.int)
    def apply_interpolation(self, interp, data):
        '''
        mesh.apply_interpolation(interp, data) yields the result of applying the given interpolation
          matrix (should be a scipy.sparse.csr_matrix), which can be obtained via
          mesh.nearest_interpolation or mesh.linear_interpolation, and applies it to the given data,
          which should be matched to the coordinates used to create the interpolation matrix.

        The data argument may be a list/vector of size m (where m is the number of columns in the
        matrix interp), a matrix of size (m x n) for any n, or a map whose values are lists of
        length m.

        Note that apply_interpolation() does not scale the interpolation matrix, so you must do that
        prior to passing it as an argument. It is generally recommended to use the interpolate()
        function or to use this function with the interpolation_matrix() function.
        '''
        # we can start by applying the mask to the interpolation
        if not sps.issparse(interp): interp = np.asarray(interp)
        (m,n) = interp.shape
        # if data is a map, we iterate over its columns:
        if pimms.is_str(data):
            return self.apply_interpolation(
                interp,
                self.properties if data.lower() == 'all' else self.properties[data])
        elif pimms.is_lazy_map(data):
            return pimms.lazy_map({curry(lambda k:self.apply_interpolation(interp, data[k]), k)
                                   for k in six.iterkeys(data)})
        elif pimms.is_map(data):
            return pyr.pmap({k:self.apply_interpolation(interp, data[k])
                             for k in six.iterkeys(data)})
        elif pimms.is_matrix(data):
            data = np.asarray(data)
            (data, tr) = (data.T, True) if data.shape[0] != n else (data, False)
            if not pimms.is_matrix(data, 'number'):
                data = np.asarray([self.apply_interpolation(interp, d) for d in data])
            else: data = inner(interp, data)
            return data.T if tr else data
        elif not pimms.is_vector(data):
            raise ValueError('cannot interpret input data argument')
        elif len(data) != n:
            return tuple([self.apply_interpolation(interp, d) for d in data])
        elif pimms.is_vector(data, 'number'):
            return inner(interp, data)
        else:
            maxs = flattest(interp.argmax(axis=1))
            bads = (~np.isfinite(interp.sum(axis=1))) | np.isclose(0, interp[(np.arange(m),maxs)])
            bads = np.where(bads)[0]
            if len(bads) == 0: res = data[maxs]
            else:
                res = np.array(data[maxs], dtype=np.object)
                res[bads] = np.nan
            return res
    def interpolation_matrix(self, x, mask=None, weights=None, method='linear', n_jobs=-1):
        '''
        mesh.interpolation_matrix(x) yields an interpolation matrix for the given point matrix x (x
          ay also be an address-data map).
        
        The following options are accepted:
          * mask (default: None) indicates that the given True/False or 0/1 valued list/array should
            be used; any point whose nearest neighbor (see below) is in the given mask will, instead
            of an interpolated value, be set to nan. The mask is interpreted by the mask() method.
          * method (default: 'linear') specifies what method to use for interpolation. The only
            currently supported methods are 'linear' and 'nearest'. Note that 'auto' cannot be used
            with interpolation_matrix() as it can with interpolate() because the data being
            interpolated is not given to interpolation_matrix() and thus cannot be used to deduce 
            the interpolation type. The 'nearest' method does not actually perform a
            nearest-neighbor interpolation but rather assigns to a destination vertex the value of
            the source vertex whose voronoi-like polygon contains the destination vertex; note that
            the term 'voronoi-like' is used here because it uses the Voronoi diagram that
            corresponds to the triangle mesh and not the true delaunay triangulation. The 'linear'
            method uses linear interpolation; though if the given data is non-numerical, then
            nearest interpolation is used instead. The 'automatic' method uses linear interpolation
            for any floating-point data and nearest interpolation for any integral or non-numeric
            data.
          * n_jobs (default: -1) is passed along to the cKDTree.query method, so may be set to an
            integer to specify how many processors to use, or may be -1 to specify all processors.
        '''
        if pimms.is_str(method): method = method.lower()
        if method in [None, Ellipsis, 'auto', 'automatic']:
            raise ValueError('interpolation_matrix() does not support method "automatic"')
        elif method in ['nn', 'nearest', 'near', 'nearest_neighbor', 'nearest-neighbor']:
            return Mesh.scale_interpolation(
                self.nearest_interpolation(x, n_jobs=n_jobs),
                mask=mask,
                weights=weights)
        elif method in ['linear', 'lin', 'trilinear']:
            return Mesh.scale_interpolation(
                self.linear_interpolation(x, n_jobs=n_jobs),
                mask=mask,
                weights=weights)
        elif method in ['heaviest', 'heavy', 'corner', 'h']:
            return Mesh.scale_interpolation(
                self.heaviest_interpolation(x, n_jobs=n_jobs),
                mask=mask,
                weights=weights)
        else: raise ValueError('unknown interpolation method: %s' % method)
    def interpolate(self, x, data, mask=None, weights=None, method='automatic', n_jobs=-1):
        '''
        mesh.interpolate(x, data) yields a numpy array of the data interpolated from the given
          array, data, which must contain the same number of elements as there are points in the
          Mesh object mesh, to the coordinates in the given point matrix x. Note that if x is a
          vector instead of a matrix, then just one value is returned.
        
        The following options are accepted:
          * mask (default: None) indicates that the given True/False or 0/1 valued list/array should
            be used; any point whose nearest neighbor (see below) is in the given mask will, instead
            of an interpolated value, be set to nan. The mask is interpreted by the mask() method.
          * method (default: 'automatic') specifies what method to use for interpolation. The only
            currently supported methods are 'automatic', 'linear', 'heaviest', or 'nearest'. The
            'heaviest' method is like a nearest-neighbor interpolation but rather assigns to
            a destination vertex the value of the source vertex whose voronoi-like polygon contains
            the destination vertex; note that the term 'voronoi-like' is used here because it uses
            the Voronoi diagram that corresponds to the triangle mesh and not the true delaunay
            triangulation. The 'linear' method uses linear interpolation; though if the given data
            is non-numerical, then heaviest interpolation is used instead. The 'automatic' method
            uses linear interpolation for any floating-point data and heaviest interpolation for any
            integral or non-numeric data. Note that heaviest interpolation is used for non-numeric
            data arrays if the method argument is 'linear'.
          * n_jobs (default: -1) is passed along to the cKDTree.query method, so may be set to an
            integer to specify how many processors to use, or may be -1 to specify all processors.
        '''
        n = self.vertex_count
        if isinstance(x, Mesh): x = x.coordinates
        # no matter what the input we want to calculate the interpolation matrices but once:
        interps = pimms.lazy_map(
            {'nearest':  lambda:self.interpolation_matrix(x,
                                                          n_jobs=n_jobs, method='nearest',
                                                          mask=mask, weights=weights),
             'heaviest': lambda:self.interpolation_matrix(x,
                                                          n_jobs=n_jobs, method='heaviest',
                                                          mask=mask, weights=weights),
             'linear':   lambda:self.interpolation_matrix(x,
                                                          n_jobs=n_jobs, method='linear',
                                                          mask=mask, weights=weights)})
        if pimms.is_str(method): method = method.lower()
        if method in [None, Ellipsis, 'auto', 'automatic']: method = None
        elif method in ['lin', 'linear', 'trilinear']: method = 'linear'
        elif method in ['heaviest', 'heavy', 'corner', 'h']: method = 'heaviest'
        elif method in ['nn','nearest','near','nearest-neighbor','nearest_neighbor']:
            method = 'nearest'
        else: raise ValueError('cannot interpret method: %s' % method)
        # we now need to look over data...
        def _apply_interp(dat):
            if pimms.is_str(dat):
                return _apply_interp(self.properties[dat])
            elif pimms.is_array(dat, np.inexact, (1,2)) and method in [None,'linear']:
                return self.apply_interpolation(interps['linear'], dat)
            elif pimms.is_array(dat, 'int', (1,2)) and method == 'linear':
                return self.apply_interpolation(interps['linear'], dat)
            elif method in [None, 'linear', 'heaviest']:
                return self.apply_interpolation(interps['heaviest'], dat)
            else:
                return self.apply_interpolation(interps['nearest'], dat)
        if pimms.is_str(data) and data.lower() == 'all':
            data = self.properties
        if pimms.is_lazy_map(data):
            return pimms.lazy_map({k:curry(lambda k:_apply_interp(data[k]), k)
                                   for k in six.iterkeys(data)})
        elif pimms.is_map(data):
            return pyr.pmap({k:_apply_interp(data[k]) for k in six.iterkeys(data)})
        elif pimms.is_matrix(data):
            # careful... the matrix could actually be a tuple of rows of different types...
            # if it's a numpy array object, though, this won't be the case
            if np.shape(data)[1] != n: data = np.transpose(data)
            if pimms.is_nparray(data): return _apply_interp(data)
            # we really don't want to apply to each row separately; instead, do a few big
            # multiplications: one for inexacts, one for integers and everything else
            data = [np.asarray(row) for row in data]
            if len(np.unique([row.dtype for row in data])) == 1:
                return _apply_interp(np.asarray(data))
            else: return tuple([_apply_interp(row) for row in data])
        elif pimms.is_vector(data, np.number) and len(data) == self.tess.vertex_count:
            return _apply_interp(data)
        elif pimms.is_vector(data):
            return tuple([_apply_interp(d) for d in data])
        else:
            return _apply_interp(data)
    def address(self, data, n_jobs=-1):
        '''
        mesh.address(X) yields a dictionary containing the address or addresses of the point or
          points given in the vector or coordinate matrix X. Addresses specify a single unique 
          topological location on the mesh such that deformations of the mesh will address the same
          points differently. To convert a point from one mesh to another isomorphic mesh, you can
          address the point in the first mesh then unaddress it in the second mesh.
        '''
        # we have to have a topology and registration for this to work...
        if isinstance(data, Mesh): return self.address(data.coordinates)
        data = np.asarray(data)
        idxfs = self.tess.indexed_faces
        coords = self.coordinates
        dims = coords.shape[0]
        if len(data.shape) == 1:
            try:              face_id = self.container(data, n_jobs=n_jobs)
            except Exception: face_id = self.container(data)
            if face_id is None:
                return {'faces':np.array([0,0,0]), 'coordinates':np.full(2,np.nan)}
            tx = coords[:, idxfs[:,face_id]].T
            faces = self.tess.faces[:,face_id]
        else:
            data = data if data.shape[1] == 3 or data.shape[1] == 2 else data.T
            n = data.shape[0]
            try:              face_id = np.asarray(self.container(data), n_jobs=n_jobs)
            except Exception: face_id = np.asarray(self.container(data))
            tx = np.full((3, dims, n), np.nan)
            oks = np.where(np.logical_not(np.equal(face_id, None)))[0]
            okfids = face_id[oks].astype('int')
            tx[:,:,oks] = np.transpose(
                np.reshape(coords[:,idxfs[:,okfids].flatten()], (dims, 3, oks.shape[0])),
                (1,0,2))
            faces = np.full((3, n), 0, dtype=np.int)
            faces[:,oks] = self.tess.faces[:,okfids]
        bc = cartesian_to_barycentric_3D(tx, data) if dims == 3 else \
             cartesian_to_barycentric_2D(tx, data)
        return {'faces': faces, 'coordinates': bc}

    def unaddress(self, data):
        '''
        mesh.unaddress(A) yields a coordinate matrix that is the result of unaddressing the given
          address dictionary A in the given mesh. See also mesh.address.
        '''
        (faces, coords) = address_data(data, 2)
        faces = self.tess.index(faces)
        selfx = self.coordinates
        if all(len(np.shape(x)) > 1 for x in (faces, coords)):
            tx = np.transpose([selfx[:,ff] if ff[0] >= 0 else null
                               for null in [np.full((3, selfx.shape[0]), np.nan)]
                               for ff in faces.T],
                              (2,1,0))
        elif faces == -1:
            return np.full(selfx.shape[0], np.nan)
        else:
            tx = selfx[:,faces].T
        return barycentric_to_cartesian(tx, coords)

    def from_image(self, image, affine=None, method=None, fill=0, dtype=None,
                   native_to_vertex_matrix=None, weights=None):
        '''
        mesh.from_image(image) interpolates the given 3D image array at the values in the given 
          mesh's coordinates and yields the property that results. If image is given as a string,
          this function will attempt to load it as an mgh/mgz file or a nifti file.

        The following options may be used:
          * affine (default: None) may specify the affine transform that aligns the vertex
            coordinates with the image (vertex-to-voxel transform). If None, then uses a
            FreeSurfer-like transformby default. Note that if image is an MGHImage or a Nifti1Image,
            then the included affine transform included in the header will be used by default if
            None is given.
          * method (default: None) may specify either 'linear' or 'nearest'; if None, then the
            interpolation is linear when the image data is real and nearest otherwise.
          * fill (default: 0) values filled in when a vertex falls outside of the image.
          * native_to_vertex_matrix (default: None) may optionally give a final transformation that
            converts from native subject orientation encoded in images to vertex positions.
          * weights (default: None) may optionally provide an image whose voxels are weights to use
            during the interpolation; these weights are in addition to trilinear weights and are
            ignored in the case of nearest interpolation.
          * native_to_vertex_matrix (default: None) specifies a matrix that aligns the surface
            coordinates with their subject's 'native' orientation; None is equivalnet to the
            identity matrix.
        '''
        if native_to_vertex_matrix is None:
            native_to_vertex_matrix = np.eye(4)
        native_to_vertex_matrix = to_affine(native_to_vertex_matrix)
        if pimms.is_str(image): image = load(image)
        if is_image(image):
            # we want to apply the image's affine transform by default
            if affine is None: affine = image.affine
            image = image.get_data()
        image = np.asarray(image)
        if affine is None:
            # wild guess: the inverse of FreeSurfer tkr_vox2ras matrix without alignment to native
            from neuropythy.freesurfer import tkr_vox2ras
            affine = np.dot(np.linalg.inv(native_to_vertex_matrix),
                            tkr_vox2ras(image.shape[0:3], (1.0, 1.0, 1.0)))
            ijk0 = np.asarray(image.shape) * 0.5
            affine = to_affine(([[-1,0,0],[0,0,-1],[0,1,0]], ijk0), 3)
        else: affine = to_affine(affine, 3)
        affine = np.dot(native_to_vertex_matrix, affine)
        affine = npla.inv(affine)
        if method is not None: method = method.lower()
        if method is None or method in ['auto', 'automatic']:
            method = 'linear' if np.issubdtype(image.dtype, np.inexact) else 'nearest'
        if dtype is None: dtype = image.dtype
        # okay, these are actually pretty simple; first transform the coordinates
        xyz = affine.dot(np.vstack((self.coordinates, np.ones(self.vertex_count))))[0:3]
        # remember: this might be a 4d or higher-dim image...
        res = np.full((self.vertex_count,) + image.shape[3:], fill, dtype=dtype)
        # now find the nearest voxel centers...
        # if we are doing nearest neighbor; we're basically done already:
        if method == 'nearest':
            ijk = np.asarray(np.round(xyz), dtype=np.int)
            ok = np.all((ijk >= 0) & [ii < sh for (ii,sh) in zip(ijk, image.shape)], axis=0)
            res[ok] = image[tuple(ijk[:,ok])]
            return res
        # otherwise, we do linear interpolation; start by parsing the weights if given
        if weights is None: weights = np.ones(image.shape)
        elif pimms.is_str(weights): weights = load(weights).get_data()
        elif is_image(weights): weights = weights.get_data()
        else: weights = np.asarray(weights)
        # find the 8 neighboring voxels
        mins = np.floor(xyz)
        maxs = np.ceil(xyz)
        ok = np.all((mins >= 0) & [ii < sh for (ii,sh) in zip(maxs, image.shape[0:3])], axis=0)
        (mins,maxs,xyz) = [x[:,ok] for x in (mins,maxs,xyz)]
        voxs = np.asarray([mins,
                           [mins[0], mins[1], maxs[2]],
                           [mins[0], maxs[1], mins[2]],
                           [mins[0], maxs[1], maxs[2]],
                           [maxs[0], mins[1], mins[2]],
                           [maxs[0], mins[1], maxs[2]],                           
                           [maxs[0], maxs[1], mins[2]],
                           maxs],
                          dtype=np.int)
        # trilinear weights
        wgts_tri = np.asarray([np.prod(1 - np.abs(xyz - row), axis=0) for row in voxs])
        # weight-image weights
        wgts_wgt = np.asarray([weights[tuple(row)] for row in voxs])
        # note that there might be a 4D image here
        if len(wgts_wgt.shape) > len(wgts_tri.shape):
            for _ in range(len(wgts_wgt.shape) - len(wgts_tri.shape)):
                wgts_tri = np.expand_dims(wgts_tri, -1)
        wgts = wgts_tri * wgts_wgt
        wgts *= zinv(np.sum(wgts, axis=0))
        vals = np.asarray([image[tuple(row)] for row in voxs])
        res[ok] = np.sum(wgts * vals, axis=0)
        return res
    
    # smooth a field on the cortical surface
    def smooth(self, prop, smoothness=0.5, weights=None, weight_min=None, weight_transform=None,
               outliers=None, data_range=None, mask=None, valid_range=None, null=np.nan,
               match_distribution=None, transform=None):
        '''
        mesh.smooth(prop) yields a numpy array of the values in the mesh property prop after they
          have been smoothed on the cortical surface. Smoothing is done by minimizing the square
          difference between the values in prop and the smoothed values simultaneously with the
          difference between values connected by edges. The prop argument may be either a property
          name or a list of property values.
        
        The following options are accepted:
          * weights (default: None) specifies the weight on each individual vertex that is in the
            mesh; this may be a property name or a list of weight values. Any weight that is <= 0 or
            None is considered outside the mask.
          * smoothness (default: 0.5) specifies how much the function should care about the
            smoothness versus the original values when optimizing the surface smoothness. A value of
            0 would result in no smoothing performed while a value of 1 indicates that only the
            smoothing (and not the original values at all) matter in the solution.
          * outliers (default: None) specifies which vertices should be considered 'outliers' in
            that, when performing minimization, their values are counted in the smoothness term but
            not in the original-value term. This means that any vertex marked as an outlier will
            attempt to fit its value smoothly from its neighbors without regard to its original
            value. Outliers may be given as a boolean mask or a list of indices. Additionally, all
            vertices whose values are either infinite or outside the given data_range, if any, are
            always considered outliers even if the outliers argument is None.
          * mask (default: None) specifies which vertices should be included in the smoothing and
            which vertices should not. A mask of None includes all vertices by default; otherwise
            the mask may be a list of vertex indices of vertices to include or a boolean array in
            which True values indicate inclusion in the smoothing. Additionally, any vertex whose
            value is either NaN or None is always considered outside of the mask.
          * data_range (default: None) specifies the range of the data that should be accepted as
            input; any data outside the data range is considered an outlier. This may be given as
            (min, max), or just max, in which case 0 is always considered the min.
          * match_distribution (default: None) allows one to specify that the output values should
            be distributed according to a particular distribution. This distribution may be None (no
            transform performed), True (match the distribution of the input data to the output,
            excluding outliers), a collection of values whose distribution should be matched, or a 
            function that accepts a single real-valued argument between 0 and 1 (inclusive) and
            returns the appropriate value for that quantile.
          * null (default: numpy.nan) specifies what value should be placed in elements of the
            property that are not in the mask or that were NaN to begin with. By default, this is
            NaN, but 0 is often desirable.
        '''
        # Do some argument processing ##############################################################
        n = self.tess.vertex_count
        all_vertices = np.asarray(range(n), dtype=np.int)
        # Parse the property data and the weights...
        (prop,weights) = self.property(prop, outliers=outliers, data_range=data_range,
                                       mask=mask, valid_range=valid_range,
                                       weights=weights, weight_min=weight_min,
                                       weight_transform=weight_transform, transform=transform,
                                       yield_weight=True)
        prop = np.array(prop)
        if not pimms.is_vector(prop, np.number):
            raise ValueError('non-numerical properties cannot be smoothed')
        # First, find the mask; these are values that can be included theoretically
        where_inf = np.where(np.isinf(prop))[0]
        where_nan = np.where(np.isnan(prop))[0]
        where_bad = np.union1d(where_inf, where_nan)
        where_ok  = np.setdiff1d(all_vertices, where_bad)
        # Whittle down the mask to what we are sure is in the minimization:
        mask = reduce(np.setdiff1d,
                      [all_vertices if mask is None else all_vertices[mask],
                       where_nan, np.where(~np.isfinite(weights))[0]])
        # Find the outliers: values specified as outliers or inf values; we'll build this as we go
        outliers = [] if outliers is None else all_vertices[outliers]
        outliers = np.intersect1d(outliers, mask) # outliers not in the mask don't matter anyway
        # no matter what, trim out the infinite values (even if inf was in the data range)
        outliers = np.union1d(outliers, mask[np.where(np.isinf(prop[mask]))[0]])
        outliers = np.union1d(outliers, mask[np.where(np.isclose(weights[mask], 0))[0]])
        outliers = np.asarray(outliers, dtype=np.int)
        # here are the vertex sets we will use below
        tethered = np.setdiff1d(mask, outliers)
        tethered = np.asarray(tethered, dtype=np.int)
        mask = np.asarray(mask, dtype=np.int)
        maskset  = frozenset(mask)
        # Do the minimization ######################################################################
        # start by looking at the edges
        el0 = self.tess.indexed_edges
        # give all the outliers mean values
        prop[outliers] = np.mean(prop[tethered])
        # x0 are the values we care about; also the starting values in the minimization
        x0 = np.array(prop[mask])
        # since we are just looking at the mask, look up indices that we need in it
        mask_idx = {v:i for (i,v) in enumerate(mask)}
        mask_tethered = np.asarray([mask_idx[u] for u in tethered])
        el = np.asarray([(mask_idx[a], mask_idx[b]) for (a,b) in el0.T
                         if a in maskset and b in maskset])
        # These are the weights and objective function/gradient in the minimization
        (ks, ke) = (smoothness, 1.0 - smoothness)
        (zs, ix) = (np.ones(len(el)), np.arange(len(el)))
        e2v = sps.csr_matrix((np.concatenate([zs,-zs]), (el.T.flatten(),np.concatenate([ix, ix]))),
                             shape=(len(x0), len(el)), dtype=np.int)
        (us, vs) = el.T
        weights_tth = weights[tethered]
        # build the optimization
        def _f(x):
            rs = np.dot(weights_tth, (x0[mask_tethered] - x[mask_tethered])**2)
            re = np.sum((x[us] - x[vs])**2)
            return ks*rs + ke*re
        def _f_jac(x):
            df = 2*ke*e2v.dot(x[us] - x[vs])
            df[mask_tethered] += 2*ks*weights_tth*(x[mask_tethered] - x0[mask_tethered])
            return df
        sm_prop = spopt.minimize(_f, x0, jac=_f_jac, method='L-BFGS-B').x
        # Apply output re-distributing if requested ################################################
        if match_distribution is not None:
            percentiles = 100.0 * np.argsort(np.argsort(sm_prop)) / (float(len(mask)) - 1.0)
            if match_distribution is True:
                sm_prop = np.percentile(x0[mask_tethered], percentiles)
            elif hasattr(match_distribution, '__iter__'):
                sm_prop = np.percentile(match_distribution, percentiles)
            elif hasattr(match_distribution, '__call__'):
                sm_prop = map(match_distribution, percentiles / 100.0)
            else:
                raise ValueError('Invalid match_distribution argument')
        result = np.full(len(prop), null, dtype=np.float)
        result[mask] = sm_prop
        return result
def is_mesh(m):
    '''
    is_mesh(m) yields True if m is a Mesh object and False otherwise.
    '''
    return isinstance(m, Mesh)
def is_flatmap(m):
    '''
    is_flatmap(m) yields True if m is a Mesh object with 2 coordinate dimensions and False
      otherwise.
    '''
    return isinstance(m, Mesh) and m.coordinates.shape[0] == 2
def mesh(faces, coordinates, meta_data=None, properties=None):
    '''
    mesh(faces, coordinates) yields a mesh with the given face and coordinate matrices.
    '''
    return Mesh(faces, coordinates, meta_data=meta_data, properties=properties)

@pimms.immutable
class MapProjection(ObjectWithMetaData):
    '''
    A MapProjection object stores information about the projection of a spherical 3D mesh to a
    flattened 2D mesh. This process involves a number of steps:
      * selection of the appropriate sub-mesh
      * alignment of the mesh to the appropriate coordinate system
      * projection of the mesh to 2D
    Additionally, MapProjection objects store the relevant data for reversing a projection; i.e.,
    a map projection object can be used to transfer points from the 2D to the 3D object and vice
    versa.
    '''

    ################################################################################################
    # The precise 3d -> 2d and 2d -> 3d functions
    @staticmethod
    def orthographic_projection_forward(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 3 else X.T
        return X[0:2]
    @staticmethod
    def orthographic_projection_inverse(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 2 else X.T
        Xnorm = X / sphere_radius
        return np.asarray([X[0], X[1], sphere_radius * np.sqrt(1.0 - (Xnorm ** 2).sum(0))])
    @staticmethod
    def equirectangular_projection_forward(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 3 else X.T
        X = X / np.sqrt((X ** 2).sum(0))
        return sphere_radius / np.pi * np.asarray([np.arctan2(X[0], X[2]), np.arcsin(X[1])])
    @staticmethod
    def equirectangular_projection_inverse(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 2 else X.T
        X = np.pi / sphere_radius * X
        cos1 = np.cos(X[1])
        return np.asarray([cos1 * np.sin(X[0]) * sphere_radius,
                           np.sin(X[1]) * sphere_radius,
                           cos1 * np.cos(X[0]) * sphere_radius])
    @staticmethod
    def mercator_projection_forward(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 3 else X.T
        X = X / np.sqrt((X ** 2).sum(0))
        return sphere_radius * np.asarray([np.arctan2(X[2], X[1]),
                                           np.log(np.tan(0.25 * np.pi + 0.5 * np.arcsin(X[0])))])
    @staticmethod
    def mercator_projection_inverse(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 2 else X.T
        X = X / sphere_radius
        return sphere_radius * np.asarray([np.sin(2 * (np.arctan(np.exp(X[1])) - 0.25*np.pi)),
                                           np.cos(X[0]), np.sin(X[0])])
    @staticmethod
    def sinusoidal_projection_forward(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 3 else X.T
        X = X / np.sqrt((X ** 2).sum(0))
        phi = np.arcsin(X[0])
        return sphere_radius / np.pi * np.asarray([np.arctan2(X[2], X[1]) * np.cos(phi), phi])
    @staticmethod
    def sinusoidal_projection_inverse(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 2 else X.T
        X = np.pi * X / sphere_radius
        cosphi = np.cos(X[1])
        return np.asarray([np.sin(X[1]) * sphere_radius,
                           np.cos(X[0] / cosphi) * sphere_radius,
                           np.sin(X[0] / cosphi) * sphere_radius])
    # These are given actual values just below the class definition
    projection_forward_methods = {}
    projection_inverse_methods = {}

    def __init__(self, mesh=None,
                 center=None, center_right=None, radius=None, method='orthographic',
                 registration='native', chirality=None, sphere_radius=None,
                 pre_affine=None, post_affine=None, meta_data=None):
        self.mesh = mesh
        self.center = center
        self.center_right = center_right
        self.radius = radius
        self.method = method
        self.registration = registration
        self.chirality = chirality
        self.sphere_radius = sphere_radius
        self.pre_affine = pre_affine
        self.post_affine = post_affine
        self.meta_data = meta_data
    @classmethod
    def denormalize(self, data):
        params = dict(mesh=None, center=None, center_right=None, radius=None, method=None,
                      registration='native', chirality=None, sphere_radius=None,
                      pre_affine=None, post_affine=None, meta_data=None)
        params = {k:data[k] for k in params.keys() if k in data}
        return MapProjection(**params)
    @staticmethod
    def load(filename,
             center=None, center_right=None, radius=None, method='orthographic',
             registration='native', chirality=None, sphere_radius=None,
             pre_affine=None, post_affine=None, meta_data=None):
        '''
        MapProjection.load(file) yields the map projection indicated by the given file name or file
          object file. Map projections define the parameters of a projection to the 2D cortical
          surface via a registartion name and projection parameters.

        Note that although load uses the normalize/denormalize system, it does not call
        denormalize(data) but rather MapProjection.denormalize(data); this is because
        MapProjection's load/save system expects that it will not encode type-data as the
        normalization system does.

        The following options may be given; all of these options represent parameters that may or
        may not be defined by the map-projection file; in the case that a parameter is defined by
        the file, the optional argument is ignored. To force parameters to have particular values,
        you must modify or copy the map projection that is yielded by this function (but note that
        all the parameters below match parameters of the map projection class). All of these
        parameters except the method (default: 'orthographic'), registration (default: 'native'),
        and meta_data parameters have default values of None, which means that they are, by default,
        handled by the MapProjection class. The meta_data parameter, if provided, is merged with any
        meta-data in the projection file such that the passed meta-data is overwritten by the file's
        meta-data.
          * center specifies the 3D vector that points toward the center of the map.
          * center_right specifies the 3D vector that points toward any point on the positive x-axis
            of the resulting map.
          * radius specifies the radius that should be assumed by the model in radians of the
            cortical sphere. If the default (None) is given, then pi/3.5 is used.
          * method specifies the projection method used (default: 'orthographic').
          * registration specifies the registration to which the map is aligned (default: 'native').
          * chirality specifies whether the projection applies to left or right hemispheres.
          * sphere_radius specifies the radius of the sphere that should be assumed by the model.
            Note that in Freesurfer, spheres have a radius of 100.
          * pre_affine specifies the pre-projection affine transform to use on the cortical sphere.
          * post_affine specifies the post-projection affine transform to use on the 2D map.
          * meta_data specifies any additional meta-data to attach to the projection.
        '''
        # import the file first:
        if pimms.is_str(filename):
            filename = os.path.expandvars(os.path.expanduser(filename))
            if not os.path.isfile(filename):
                raise ValueError('Given filename (%s) is not a file!' % filename)
            gz = (len(filename) > 3 and filename[-3:] == '.gz')
            dat = None
            with (gzip.open(filename, 'rt') if gz else open(filename, 'rt')) as f:
                dat = json.load(f)
            fname = filename
        else:
            dat = json.load(filename)
            fname = '<file-stream>'
        dat = {k.lower():denormalize(v) for (k,v) in six.iteritems(dat)}
        # make the object
        params = dict(center=center, center_right=center_right, radius=radius, method=method,
                      registration=registration, chirality=chirality, sphere_radius=sphere_radius,
                      pre_affine=pre_affine, post_affine=post_affine)
        dat = {k:(dat[k] if k in dat else v)
               for (k,v) in six.iteritems(params)
               if v is not None or k in dat}
        return MapProjection.denormalize(dat)
    def save(self, filename):
        '''
        map_projection.save(file) writes a json version of the given map_projection object to the
          given file name or stream object, file.
        '''
        # okay, we need to make a json-like structure of ourselves then turn it into a string;
        # note that we cannot save the mesh, so it is always left off
        dat = normalize(self.normalize())
        txt = json.dumps(dat)
        # if it's a filename, we'll need to open it then close it later
        if pimms.is_str(filename):
            filename = os.path.expandvars(os.path.expanduser(filename))
            gz = (len(filename) > 3 and filename[-3:] == '.gz')
            with (gzip.open(filename, 'wt') if gz else open(filename, 'wt')) as f: f.write(txt)
        else: filename.write(txt)
        return filename
    @pimms.param
    def mesh(m):
        '''
        proj.mesh is the mesh on which the given projection was performed; this mesh should be a 3D
        spherical mesh. If proj.mesh is None, then this projection has not yet been reified with a
        mesh object.
        '''
        if m is None: return None
        if not isinstance(m, Mesh):
            raise ValueError('projection mesh must be a Mesh object')
        return pimms.persist(m)
    @pimms.param
    def center(c):
        '''
        proj.center is the (x,y,z) coordinate of the center of the given projection on the sphere.
        '''
        if c is None: return None
        if not pimms.is_vector(c) or len(c) != 3:
            raise ValueError('map projection center must be a 3D coordinate')
        return pimms.imm_array(c)
    @pimms.param
    def center_right(cr):
        '''
        proj.center_right is the (x,y,z) coordinate of any point that should appear on the positive
        x-axis of the resulting map projection.
        '''
        if cr is None: return None
        if not pimms.is_vector(cr) or len(cr) != 3:
            raise ValueError('map projection center_right must be a 3D coordinate')
        return pimms.imm_array(cr)
    @pimms.param
    def radius(r):
        '''
        proj.radius is the radius of inclusion for the given map projection; this may be an value
        given as a number or a quantity in terms of radians, turns, or degrees (e.g.,
        pimms.quant(30, 'deg') will create a map projection all points within 30 degrees of the
        center). If None, then pi/3.5 radians is used.
        '''
        if r is None: return np.pi/3.5
        if not pimms.is_real(r) or r <= 0:
            raise ValueError('radius must be a real number that is greater than 0')
        if pimms.is_quantity(r) and not pimms.like_units(r, 'radians'):
            raise ValueError('radius must be a real number or a quantity with rotation units')
        return r
    @pimms.param
    def method(m):
        '''
        proj.method is the name of the method used to create the map projection proj.
        '''
        if not pimms.is_str(m):
            raise ValueError('projection method must be a string')
        m = m.lower()
        if m not in MapProjection.projection_forward_methods:
            raise ValueError('inrecognized map projection: %s' % m)
        return m
    @pimms.param
    def registration(r):
        '''
        proj.registration is either None or the name of a registration that should be used as the
        base sphere for the given map projection; note that None and 'native' are equivalent.
        '''
        if r is None: return 'native'
        if not pimms.is_str(r):
            raise ValueError('projection registration must be a string')
        return r
    @pimms.param
    def chirality(ch):
        '''
        proj.chirality is either 'lh', 'rh', or None, and specifies whether the given map projection
        is valid only for LH or RH chiralities; None indicates that the projection does not care
        about chirality.
        '''
        ch = to_hemi_str(ch)
        if ch == 'lr': return None
        else: return ch
    @pimms.param
    def sphere_radius(sr):
        '''
        proj.sphere_radius is either None or the radius of the initial sphere. Generally this is
        left as None until projection is performed so that the sphere_radius can be deduced from
        the mesh that serves as the projection domain. Because the MapProjection object that is
        attached to the meta-data of the resulting map is modified to have this value filled in,
        inverse projetions do not require that this value be set explicitly.
        '''
        if sr is None: return None
        if not pimms.is_real(sr) or sr <= 0:
            raise ValueError('sphere_radius must be a positive real number or None')
        return sr
    @pimms.param
    def pre_affine(pa):
        '''
        proj.pre_affine is a 4x4 matrix of the affine transformation that should be applied to the
        coordinates of the spherical mesh prior to projection via the rest of the standard map
        projection methods. This may be None to indicate no initial transformation.
        '''
        return pa if pa is None else pimms.imm_array(to_affine(pa, 3))
    @pimms.param
    def post_affine(pa):
        '''
        proj.post_affine is a 3x3 matrix of the affine transformation that should be applied to the
        coordinates of the flat map after projection via the rest of the standard map projection
        methods. This may be None to indicate no final transformation.
        '''
        return pa if pa is None else pimms.imm_array(to_affine(pa, 2))
    @pimms.value
    def alignment_matrix(pre_affine, center, center_right):
        '''
        proj.alignment_matrix is a 4x4 matrix that aligns the 3D spherical mesh such that the center
        of the projection lies on the positive z-axis and the center_right of the projection lies in
        the x-z plane with a positive x value.

        Note that the alignment_matrix applies the pre_affine prior to determining the rotation of
        the center and center_right; accordingly, the center should be specified in post-pre_affine
        transfomration coordinates.
        '''
        mtx = np.eye(4) if pre_affine is None else pre_affine
        cmtx = np.eye(4)
        if center is not None:
            tmp = alignment_matrix_3D(center, [0,0,1])
            cmtx[0:3,0:3] = tmp
            mtx = cmtx.dot(mtx)
        crmtx = np.eye(4)
        if center_right is not None:
            # Tricky: we need to run this coordinate through the center transform then align it with
            # the x-y plane:
            cr = cmtx.dot(np.concatenate([center_right, [1]]))[:3]
            # what angle do we need to rotate this?
            ang = np.arctan2(cr[1], cr[0])
            crmtx[0:3,0:3] = rotation_matrix_3D([0,0,1], -ang)
            mtx = crmtx.dot(mtx)
        # That's all that actually needs to be done in preprocessing
        return pimms.imm_array(mtx)
    @pimms.value
    def inverse_alignment_matrix(alignment_matrix):
        '''
        proj.inverse_alignment_matrix is a 4x4 matrix that is the inverse of proj.alignment_matrix.
        '''
        return None if alignment_matrix is None else pimms.imm_array(npla.inv(alignment_matrix))
    @pimms.value
    def inverse_post_affine(post_affine):
        '''
        proj.inverse_post_affine is a 4x4 matrix that is the inverse of proj.post_affine.
        '''
        return None if post_affine is None else pimms.imm_array(npla.inv(post_affine))
    @pimms.value
    def _sphere_radius(mesh, sphere_radius):
        '''
        proj._sphere_radius is identical to proj.sphere_radius unless proj.sphere_radius is None and
        proj.mesh is not None, in which case proj._sphere_radius is deduced from the mesh. For this
        reason, _sphere_radius is used internally.
        If both sphere_radius and mesh are None, then the default sphere radius is 100.
        '''
        if sphere_radius is not None: return sphere_radius
        if mesh is None: return 100.0
        rs = np.sqrt(np.sum(mesh.coordinates**2, axis=0))
        mu = np.mean(rs)
        sd = np.std(rs)
        if sd/mu > 0.05: warnings.war('Given mesh does not appear to be a sphere centered at 0')
        return mu
    @pimms.value
    def repr(chirality, registration):
        '''
        proj.repr is the representation string yielded by proj.__repr__().
        '''
        ch = 'LR' if chirality is None else chirality.upper()
        reg = 'native' if registration is None else registration
        return 'MapProjection(<%s>, <%s>)' % (ch, reg)
    
    def __repr__(self):
        return self.repr
    def in_domain(self, x):
        '''
        proj.in_domain(x) yields a boolean array whose elements indicate whether the coordinates in
          x are part of the domain of the given map projection. The argument x may be either a
          coordinate matrix, a coordinate vector, or a mesh (in which case this is equivalent to
          proj.in_domain(x.coordinates).
        '''
        x = x.coordinates if isinstance(x, Mesh) else np.asarray(x)
        if pimms.is_vector(x): return np.asarray(self.in_domain([x])[0], dtype=np.bool)
        if x.shape[0] != 3: x = x.T
        # no radius means we don't actually do any trimming
        if self.radius is None: return np.ones(x.shape[1], dtype=np.bool)
        # put the coordinates through the initial transformation:
        x = self.alignment_matrix.dot(np.concatenate((x, np.ones((1,x.shape[1])))))
        x = np.clip(x[2] / self._sphere_radius, -1, 1)
        # okay, we want the angle of the vertex [0,0,1] to these points...
        th = np.arccos(x)
        # and we want to know what points are within the angle given by the radius; if the radius
        # is a radian-like quantity, we use th itself; otherwise, we convert it to a distance
        rad = pimms.mag(self.radius, 'radians')
        return (th < rad)
    def select_domain(self, x):
        '''
        proj.select_domain(x) yields a subset of the coordinates in x that lie in the domain of
          the given map projection proj, assuming x is a 3D coordinate matrix.
        proj.select_domain(x0) yields either the point x0 or None.
        proj.select_domain(mesh) yields a copy of mesh that has been sub-sampled using
          mesh.submesh().
        '''
        inq = self.in_domain(x)
        if   isinstance(x, Mesh): return x.submesh(inq)
        elif pimms.is_vector(x):  return (None if inq[0] else x)
        x = np.asarray(x)
        return x[:,inq] if x.shape[0] == 3 else x[inq]            
    def forward(self, x):
        '''
        proj.forward(x) yields the result of projecting the given 3D coordinate or coordinates in x
          through the map projection proj; the result will be a 2D vector or matrix with the same
          shape as x (up to conversion of 3D to 2D).
        proj.forward(mesh) yeilds the 2D mesh that results from the projection.

        Note that proj.forward does not perform any trimming w.r.t. the radius parameter; this is
        intentional. For a full-featured projection with trimming use proj(x) or proj(mesh); this
        is equivalent to proj.forward(proj.select_domain(x)).
        '''
        if   pimms.is_vector(x, 'real'):     return self.forward([x])[0]
        elif isinstance(x, Mesh):            return x.copy(coordinates=self.forward(x.coordinates))
        elif not pimms.is_matrix(x, 'real'): raise ValueError('invalid input coordinates')
        x = np.asarray(x)
        if x.shape[0] != 3:
            if x.shape[1] != 3: raise ValueError('coordinates are not 3D')
            else: return self.forward(x.T).T
        ones = np.ones((1, x.shape[1]))
        # apply the alignment matrix first:
        aff0 = self.alignment_matrix
        x = aff0.dot(np.concatenate((x, ones)))[0:3]
        # okay, next call the transformation function...
        fwd = MapProjection.projection_forward_methods[self.method]
        x = fwd(x, sphere_radius=self._sphere_radius)
        # next, apply the post-transform
        ptx = self.post_affine
        if ptx is not None:
            x = ptx.dot(np.concatenate((x, ones)))[0:2]
        # that's it!
        return x
    def inverse(self, x):
        '''
        proj.inverse(x) yields the result of unprojecting the given 2D coordinate or coordinates in
          x back to the original 3D sphere. See also proj.forward().
        '''
        if   pimms.is_vector(x, 'real'):     return self.inverse([x])[0]
        elif isinstance(x, Mesh):            return x.copy(coordinates=self.inverse(x.coordinates))
        elif not pimms.is_matrix(x, 'real'): raise ValueError('invalid input coordinates')
        x = np.asarray(x)
        if x.shape[0] != 2:
            if x.shape[1] != 2: raise ValueError('coordinates are not 2D')
            else: return self.inverse(x.T).T
        ones = np.ones((1, x.shape[1]))
        # first, un-apply the post-transform
        ptx = self.inverse_post_affine
        if ptx is not None:
            x = ptx.dot(np.concatenate((x, ones)))[0:2]
        # okay, next call the inverse transformation function...
        inv = MapProjection.projection_inverse_methods[self.method]
        x = inv(x, sphere_radius=self._sphere_radius)
        # apply the alignment matrix first:
        aff0 = self.inverse_alignment_matrix
        x = aff0.dot(np.concatenate((x, ones)))[0:3]
        # that's it!
        return x
    def extract_mesh(self, obj):
        '''
        proj.extract_mesh(topo) yields the mesh registration object from the given topology topo
          that is registered to the given projection, or None if no such registration is found.
        proj.extract_mesh(subj) yields the registration for a subject.

        Note that None is also returned if the projection cannot find an appropriate hemisphere in
        the subject.
        '''
        from neuropythy.mri import is_subject
        if is_subject(obj):
            if self.chirality is None: return None
            else: ch = self.chirality
            if ch not in obj.hemis: return None
            else: obj = obj.hemis[ch]
        if isinstance(obj, Topology):
            # check the chiralities
            if (obj.chirality  is not None and
                self.chirality is not None and
                obj.chirality != self.chirality): return None
            # We need to figure out if there is a matching registration
            reg = self.registration
            if self.registration is None: reg = 'native'
            if reg in obj.registrations: return obj.registrations[reg]
            else: return None
        else: raise ValueError('extract_mesh arg must be a Subject or Topology object')
    def __call__(self, obj, tag='projection'):
        '''
        proj(x) performs the map projection proj on the given coordinate or coordinate matrix x and
          yields the resulting coordinate or coordinate matrix. If no coordinates in x are part of
          the domain, then an empty matrix is returned; if only one coordinate was provided, and it
          is not in the domain, then None is returned.
        proj(mesh) yields a 2D mesh that is the result of the given projection; in this case, the
          meta_data of the newly created mesh includes the tag 'projection', which will contain the
          projection used to create the map; this projection will have been reified with mesh. The
          optional argument tag may be used to change the name of 'projection'; None indicates that
          no tag should be included.
        proj(topo) yields a 2D mesh that is derived from one of the registrations in the given
          topology topo, determined by the proj.registration parameter.
        '''
        from neuropythy.mri import is_subject
        if is_topo(obj) or is_subject(obj):
            msh = self.extract_mesh(obj)
            if msh is None: raise ValueError('Could not find matching registration for %s' % obj)
            else: obj = msh
        if isinstance(obj, Mesh):
            proj = self if self.mesh is obj else self.copy(mesh=obj)
            submesh = self.select_domain(obj)
            res = self.forward(submesh)
            return res if tag is None else res.with_meta({tag:proj})
        elif pimms.is_vector(obj):
            return self.forward(obj) if self.in_domain(obj) else None
        else:
            return self.forward(self.select_domain(obj))
MapProjection.projection_forward_methods = pyr.m(
    orthographic    = MapProjection.orthographic_projection_forward,
    equirectangular = MapProjection.equirectangular_projection_forward,
    mercator        = MapProjection.mercator_projection_forward,
    sinusoidal      = MapProjection.sinusoidal_projection_forward)
MapProjection.projection_inverse_methods = pyr.m(
    orthographic    = MapProjection.orthographic_projection_inverse,
    equirectangular = MapProjection.equirectangular_projection_inverse,
    mercator        = MapProjection.mercator_projection_inverse,
    sinusoidal      = MapProjection.sinusoidal_projection_inverse)

def deduce_chirality(obj):
    '''
    deduce_chirality(x) attempts to deduce the chirality of x ('lh', 'rh', or 'lr') and yeilds the
      deduced string. If no chirality can be deduced, yields None. Note that a if x is either None
      or Ellipsis, this is converted into 'lr'.
    '''
    # few simple tests:
    try: return obj.chirality
    except Exception: pass
    try: return obj.meta_data['chirality']
    except Exception: pass
    try: return obj.meta_data['hemi']
    except Exception: pass
    try: return obj.meta_data['hemisphere']
    except Exception: pass
    if obj is None or obj is Ellipsis: return 'lr'
    try: return to_hemi_str(obj)
    except Exception: pass
    return None

@importer('map_projection', ('mp.json', 'mp.json.gz', 'map.json', 'map.json.gz',
                             'projection.json', 'projection.json.gz'))
def load_map_projection(filename,
                        center=None, center_right=None, radius=None, method='orthographic',
                        registration='native', chirality=None, sphere_radius=None,
                        pre_affine=None, post_affine=None, meta_data=None):
    '''
    load_map_projection(filename) yields the map projection indicated by the given file name. Map
      projections define the parameters of a projection to the 2D cortical surface via a
      registartion name and projection parameters.

    This function is primarily a wrapper around the MapProjection.load() function; for information
    about options, see MapProjection.load.
    '''
    return MapProjection.load(filename,
                              center=center, center_right=center_right, radius=radius,
                              method=method, registration=registration, chirality=chirality,
                              sphere_radius=sphere_radius, pre_affine=pre_affine,
                              post_affine=post_affine)
@exporter('map_projection', ('mp.json', 'mp.json.gz', 'map.json', 'map.json.gz',
                             'projection.json', 'projection.json.gz'))
def save_map_projection(filename, mp, **kw):
    '''
    save_map_projection(filename, map_projection) saves the given map projection to the given file
      or stream object, filename, and returns filename.
    '''
    return mp.save(filename, **kw)
# Auto-loaded Map Projections (#map_projections)
projections_libdir = (os.path.join(library_path(), 'projections'),)
# the projections in the neuropythy libdir
def load_projections_from_path(p):
    '''
    load_projections_from_path(p) yields a lazy-map of all the map projection files found in the
      given path specification p. The path specification may be either a directory name, a
      :-separated list of directory names, or a python list of directory names.

    In order to be included in the returned mapping of projections, a projection must have a
    filename that matches the following format:
      <hemi>.<name>[.mp | .map | .projection | <nothing>].json[.gz]
    For example, the following match:
      lh.occipital_pole.mp.json    => map_projections['lh']['occipital_pole']
      rh.frontal.json.gz           => map_projections['rh']['frontal']
      lr.motor.projection.json.gz  => map_projections['lr']['motor']
    '''
    p = dirpath_to_list(p)
    return pyr.pmap(
        {h:pimms.lazy_map(
            {parts[1]: curry(lambda flnm,h: load_map_projection(flnm, chirality=h),
                             os.path.join(pp, fl), h)
             for pp    in p
             for fl    in os.listdir(pp)  if fl.endswith('.json') or fl.endswith('.json.gz')
             for parts in [fl.split('.')] if len(parts) > 2 and parts[0] == h})
         for h in ('lh','rh','lr')})
# just the neuropythy lib-dir projections:
try: npythy_map_projections = load_projections_from_path(projections_libdir)
except Exception:
    warnings.warn('Error raised while loading neuropythy libdir map projections')
    npythy_map_projections = pyr.m(lh=pyr.m(), rh=pyr.m(), lr=pyr.m())
# all the map projections:
map_projections = npythy_map_projections
def check_projections_path(path):
    '''
    check_projections_path(path) yields the given path after checking that it is valid and updating
      neuropythy.map_projections to include any projections found on the given path.

    This function is called whenever neuropythy.config['projections_path'] is edited; it should not
    generally be called directly.
    '''
    path = dirpath_to_list(path)
    tmp = load_projections_from_path(path)
    # okay, seems like it passed; go ahead and update
    global map_projections
    map_projections = pyr.pmap({h: pimms.merge(npythy_map_projections[h], tmp[h])
                                for h in six.iterkeys(npythy_map_projections)})
    return path
config.declare('projections_path', filter=check_projections_path)
def projections_path(path=Ellipsis):
    '''
    projections_path() yields the projections path currently being used by neuropythy. This is
      equivalent to projections_path(Ellipsis).
    projections_path(path) sets the neuropythy projections path to be the given argument. The path
      may be either a single directory, a list of directories, or a :-separated list of directories.

    Before returnings, the map projections mapping stored in neuropythy.map_projections is updated
    to include all map projections found in the given directories. Old map projections found in the
    previous projections-path directories are discarded; though map projections found in
    neuropythy's lib directory are always kept (but may be overwritten by projections in the path
    with the same name)

    In order to be included in the returned mapping of projections, a projection must have a
    filename that matches the following format:
      <hemi>.<name>[.mp | .map | .projection | <nothing>].json[.gz]
    For example, the following match:
      lh.occipital_pole.mp.json    => map_projections['lh']['occipital_pole']
      rh.frontal.json.gz           => map_projections['rh']['frontal']
      lh.motor.projection.json.gz  => map_projections['lh']['motor']
    '''
    if path is Ellipsis: return config['projections_path']
    else: config['projections_path'] = path
def map_projection(name=None, chirality=Ellipsis,
                   center=Ellipsis, center_right=Ellipsis, radius=Ellipsis,
                   method=Ellipsis, registration=Ellipsis, sphere_radius=Ellipsis,
                   pre_affine=Ellipsis, post_affine=Ellipsis, meta_data=Ellipsis, remember=False):
    '''
    map_projection(name, hemi) yields the map projection with the given name if it exists; hemi must
      be either 'lh', 'rh', or 'lr'/None.
    map_projection(name, topo) yields a map projection using the given topology object topo to
      determine the hemisphere and assigning to the resulting projection's 'mesh' parameter the
      appropriate registration from the given topology.
    map_projection(name, mesh) uses the given mesh; the mesh's meta-data must specify the hemisphere
      for this to work--otherwise 'lr' is always used as the hemisphere.
    map_projection(affine, hemi) creates a map projection from the given affine matrix, which must
      align a set of spherical coordinates to a new set of 3D coordinates that are used as input to
      the method argument (default method 'orthographic' uses the first two of these coordinates as
      the x and y values of the map).
    map_projection() creates a new map projection using the optional arguments, if provided.

    All options that can be passed to load_map_projection and MapProjection can be passed to
    map_projection:
      * name is the first optional parameter appearing as name and affine above.
      * chirality is the second optional parameter, and, if set, will ensure that the resulting
        map projection's chirality is equivalent to the given chirality.
      * center specifies the 3D vector that points toward the center of the map.
      * center_right specifies the 3D vector that points toward any point on the positive x-axis
        of the resulting map.
      * radius specifies the radius that should be assumed by the model in radians of the
        cortical sphere; if the default value (Ellipsis) is given, then pi/3.5 is used.
      * method specifies the projection method used (default: 'equirectangular').
      * registration specifies the registration to which the map is aligned (default: 'native').
      * chirality specifies whether the projection applies to left or right hemispheres.
      * sphere_radius specifies the radius of the sphere that should be assumed by the model.
        Note that in Freesurfer, spheres have a radius of 100.
      * pre_affine specifies the pre-projection affine transform to use on the cortical sphere. Note
        that if the first (name) argument is provided as an affine transform, then that transform is
        applied after the pre_affine but before alignment of the center and center_right points.
      * post_affine specifies the post-projection affine transform to use on the 2D map.
      * meta_data specifies any additional meta-data to attach to the projection.
      * remember may be set to True to indicate that, after the map projection is constructed, the
        map_projection cache of named projections should be updated with the provided name. This
        can only be used when the provided name or first argument is a string.
    '''
    global map_projections # save flag lets us modify this
    # make a dict of the map parameters:
    kw = dict(center=center, center_right=center_right, radius=radius,
              method=method, registration=registration, sphere_radius=sphere_radius,
              pre_affine=pre_affine, post_affine=post_affine, meta_data=meta_data)
    kw = {k:v for (k,v) in six.iteritems(kw) if v is not Ellipsis}
    # interpret the hemi argument first
    hemi = chirality
    if hemi is None or hemi is Ellipsis:
        hemi = 'lr'
        topo = None
        mesh = None
    if pimms.is_str(hemi):
        hemi = to_hemi_str(hemi)
        topo = None
        mesh = None
    elif is_topo(hemi):
        topo = hemi
        hemi = hemi.chirality
        mesh = None
    elif is_mesh(hemi):
        mesh = hemi
        hemi = deduce_chirality(hemi)
        topo = None
    else: raise ValueError('Could not understand map_projection hemi argument: %s' % hemi)
    hemi = to_hemi_str(hemi)
    # name might be an affine matrix
    try:              aff = to_affine(name)
    except Exception: aff = None
    if pimms.is_matrix(aff):
        # see if this is an affine matrix
        aff = np.asarray(aff)
        (n,m) = aff.shape
        mtx = None
        # might be a transformation into 2-space or into 3-space:
        if n == 2:
            if   m == 3: mtx = np.vstack([np.hstack([mtx, [[0,0,0],[0,0,0]]]), [[0],[0],[0],[1]]])
            elif m == 4: mtx = np.vstack([mtx, [[0,0,0,0], [0,0,0,1]]])
        elif n == 3:
            if   m == 3: mtx = np.vstack([np.hstack([mtx, [[0,0,0]]]), [[0],[0],[0],[1]]])
            elif m == 4: mtx = np.vstack([mtx, [[0,0,0,1]]])
        elif n == 4:
            if   m == 4: mtx = aff
        if mtx is None: raise ValueError('Invalid affine matrix shape; must be {2,3}x{3,4} or 4x4')
        # Okay, since center and center-right ignore the pre-affine matrix, we can just use this as
        # the pre-affine and set the center to whatever comes out
        kw['pre_affine'] = mtx if kw.get('pre_affine') is None else mtx.dot(to_affine(pre_affine))
        name = None
    # load name if it's a string
    if pimms.is_str(name):
        if   name         in map_projections[hemi]: mp = map_projections[hemi][name]
        elif name.lower() in map_projections[hemi]: mp = map_projections[hemi][name.lower()]
        else:
            try: mp = load_map_projection(name, chirality=hemi)
            except Exception:
                raise ValueError('could neither find nor load projection %s (%s)' % (name,hemi))
        # update parameters if need-be:
        if len(kw) > 0: mp = mp.copy(**kw)
    elif name is None:
        # make a new map_projection
        if chirality is not Ellipsis: kw['chirality'] = hemi
        mp = MapProjection(**kw)
    elif is_map_projection(name):
        # just updating an existing projection
        if chirality is not Ellipsis: kw['chirality'] = hemi
        mp = name
        if len(kw) > 0: mp = mp.copy(**kw)
    else: raise ValueError('first argument must be affine, string, or None')
    # if we have a topology/mesh, we should add it:
    if topo is not None: mesh = mp.extract_mesh(topo)
    if mesh is not None: mp   = mp.copy(mesh=mesh)
    # if save was requested, save it
    mp = mp.persist()
    if remember is True:
        if pimms.is_str(name):
            name = name.lower()
            hval = map_projections.get(hemi, pyr.m())
            map_projections = map_projections.set(hemi, hval.set(name, mp))
        else: warnings.warn('Cannot save map-projection with non-string name')
    # okay, return the projection
    return mp
def is_map_projection(arg):
    '''
    is_map_projection(arg) yields True if arg is a map-projection object and False otherwise.
    '''
    return isinstance(arg, MapProjection)
def to_map_projection(arg, hemi=Ellipsis, chirality=Ellipsis,
                      center=Ellipsis, center_right=Ellipsis, radius=Ellipsis,
                      method=Ellipsis, registration=Ellipsis, sphere_radius=Ellipsis,
                      pre_affine=Ellipsis, post_affine=Ellipsis, meta_data=Ellipsis):
    '''
    to_map_projection(mp) yields mp if mp is a map projection object.
    to_map_projection((name, hemi)) is equivalent to map_projection(name, chirality=hemi).
    to_map_projection((name, opts)) uses the given options dictionary as options to map_projection;
      (name, hemi, opts) is also allowed as input.
    to_map_projection(filename) yields the map projection loaded from the given filename.
    to_map_projection('<name>:<hemi>') is equivalent to to_map_projection(('<name>', '<hemi>')).
    to_map_projection('<name>') is equivalent to to_map_projection(('<name>', 'lr')).
    to_map_projection((affine, hemi)) converts the given affine transformation, which must be a
      transformation from spherical coordinates to 2D map coordinates (once the transformed z-value
      is dropped), to a map projection. The hemi argument may alternately be an options mapping.

    The to_map_projection() function may also be called with the the elements of the above tuples
    passed directly; i.e. to_map_projection(name, hemi) is equivalent to
    to_map_projection((name,hemi)).

    Additionaly, all optional arguments to the map_projection function may be given and will be
    copied into the map_projection that is returned. Note that the named chirality argument is used
    to set the chirality of the returned map projection but never to specify the chirality of a
    map projection that is being looked up or loaded; for that use the second argument, second tuple
    entry, or hemi keyword.
    '''
    kw = dict(center=center, center_right=center_right, radius=radius, chirality=chirality,
              method=method, registration=registration, sphere_radius=sphere_radius,
              pre_affine=pre_affine, post_affine=post_affine, meta_data=meta_data)
    kw = {k:v for (k,v) in six.iteritems(kw) if v is not Ellipsis}
    if pimms.is_vector(arg):
        if   len(arg) == 1: arg = arg[0]
        elif len(arg) == 2:
            (arg, tmp) = arg
            if pimms.is_map(tmp):
                kw = {k:v for (k,v) in six.iteritems(pimms.merge(tmp, kw)) if v is not Ellipsis}
            elif hemi is Ellipsis: hemi = arg
        elif len(arg) == 3:
            (arg, h, opts) = arg
            kw = {k:v for (k,v) in six.iteritems(pimms.merge(opts, kw)) if v is not Ellipsis}
            if hemi is Ellipsis: hemi = h
        else: raise ValueError('Invalid vector argument given to to_map_projection()')
    hemi = deduce_chirality(hemi)
    mp = None
    if   is_map_projection(arg): mp = arg
    elif pimms.is_str(arg):
        # first see if there's a hemi appended
        if ':' in arg:
            spl = arg.split(':')
            (a,h) = (':'.join(spl[:-1]), spl[-1])
            try:
                (hemtmp, arg) = (to_hemi_str(h), a)
                if hemi is None: hemi = hemtmp
            except Exception: pass
        # otherwise, strings alone might be map projection names or filenames
        mp = map_projection(arg, hemi)
    else: raise ValueError('Cannot interpret argument to to_map_projection')
    if len(kw) == 0: return mp
    else: return mp.copy(**kw)
def to_flatmap(name, hemi=Ellipsis, 
               center=Ellipsis, center_right=Ellipsis, radius=Ellipsis, chirality=Ellipsis,
               method=Ellipsis, registration=Ellipsis, sphere_radius=Ellipsis,
               pre_affine=Ellipsis, post_affine=Ellipsis, meta_data=Ellipsis):
    '''
    to_flatmap(name, topo) yields a flatmap of the given topology topo using the map projection
      obtained via to_map_projection(name).
    to_flatmap(name, mesh) yields a flatmap of the given mesh. If no hemisphere is specified in the
      name argument nor in the mesh meta-data, then 'lr' is assumed.
    to_flatmap(name, subj) uses the given hemisphere from the given subject.
    to_flatmap((name, obj)) is equivalent to to_flatmap(name, obj).
    to_flatmap((name, obj, opts_map)) is equivalent to to_flatmap(name, obj, **opts_map).

    All optional arguments that can be passed to map_projection() may also be passed to the
    to_flatmap() function and are copied into the map projection object prior to its use in creating
    the resulting 2D mesh. Note that the named chirality argument is used to set the chirality of
    the returned map projection but never to specify the chirality of a map projection that is being
    looked up or loaded; for that use the second argument, second tuple entry, or hemi keyword.
    '''
    from neuropythy.mri import is_subject
    kw = dict(center=center, center_right=center_right, radius=radius, chirality=chirality,
              method=method, registration=registration, sphere_radius=sphere_radius,
              pre_affine=pre_affine, post_affine=post_affine, meta_data=meta_data)
    kw = {k:v for (k,v) in six.iteritems(kw) if v is not Ellipsis}
    if pimms.is_vector(name):
        if   len(name) == 1: name = name[0]
        elif len(name) == 2:
            (name, tmp) = name
            if pimms.is_map(tmp):
                kw = {k:v for (k,v) in six.iteritems(pimms.merge(tmp, kw)) if v is not Ellipsis}
            else: hemi = tmp
        elif len(name) == 3:
            (name, hemi, opts) = name
            kw = {k:v for (k,v) in six.iteritems(pimms.merge(opts, kw)) if v is not Ellipsis}
    if 'hemi' in kw:
        if hemi is Ellipsis: hemi = kw['hemi']
        del kw['hemi']
    # okay, we can get the map projection now...
    mp = to_map_projection(name, hemi, **kw)
    return mp(hemi)

@pimms.immutable
class Topology(VertexSet):
    '''
    A Topology object object represents a tesselation and a number of registered meshes; the
    registered meshes (registrations) must all share the same tesselation object; these are
    generally provided via vertex coordinates and not actualized Mesh objects.
    This class should only be instantiated by the neuropythy library and should generally not be
    constructed directly. See Hemisphere.topology objects to access a subject's topology objects.
    A Topology is a VertexSet that inherits its properties from its tess object; accordingly it
    can be used as a source of properties.
    '''

    def __init__(self, tess, registrations, properties=None, meta_data=None, chirality=None):
        self.tess = tess
        self.chirality = chirality
        self._registrations = registrations
        self._properties = properties
        self.meta_data = meta_data

    @pimms.param
    def tess(t):
        '''
        topo.tess is the tesselation object tracked by the given topology topo.
        '''
        if not isinstance(t, Tesselation):
            t = Tesselation(tess)
        return pimms.persist(t)
    @pimms.param
    def chirality(ch):
        '''
        topo.chirality gives the chirality ('lh' or 'rh') for the given topology; this may be None
        if no chirality has been specified.
        '''
        if ch is None: return None
        ch = ch.lower()
        if ch != 'lh' and ch != 'rh':
            raise ValueError('chirality must be \'lh\' or \'rh\'')
        return ch
    @pimms.param
    def _registrations(regs):
        '''
        topo._registrations is the list of registration coordinates provided to the given topology.
        See also topo.registrations.
        '''
        return regs if pimms.is_pmap(regs) else pyr.pmap(regs)
    @pimms.value
    def registrations(_registrations, tess, properties):
        '''
        topo.registrations is a persistent map of the mesh objects for which the given topology
        object is the tracker; this is generally a lazy map whose values are instantiated as 
        they are requested.
        '''
        # okay, it's possible that some of the objects in the _registrations map are in fact
        # already-instantiated meshes of tess; if so, we need to leave them be; at the same time
        # we can't assume that a value is correct without checking it...
        lazyq = pimms.is_lazy_map(_registrations)
        def _reg_check(key):
            def _lambda_reg_check():
                val = _registrations[key]
                if isinstance(val, Mesh): val = val.coordinates
                return Mesh(tess, val, properties=properties).persist()
            if lazyq and _registrations.is_lazy(key):
                return _lambda_reg_check
            else:
                return _lambda_reg_check()
        return pimms.lazy_map({k:_reg_check(k) for k in six.iterkeys(_registrations)})
    @pimms.value
    def labels(tess):
        '''
        topo.labels is the list of vertex labels for the given topology topo.
        '''
        return tess.labels
    @pimms.value
    def indices(tess):
        '''
        topo.indices is the list of vertex indicess for the given topology topo.
        '''
        return tess.indices
    @pimms.value
    def properties(_properties, tess):
        '''
        topo.properties is the pimms Itable object of properties known to the given topology topo.
        '''
        pp = {} if _properties is None else _properties
        tp = {} if tess.properties is None else tess.properties
        # note that tess.properties always already has labels and indices included
        itbl = pimms.ITable(pimms.merge(tp, pp) if _properties is not tess.properties else pp,
                            n=tess.vertex_count)
        return itbl.persist()
    @pimms.value
    def repr(chirality, tess):
        '''
        topo.repr is the representation string yielded by topo.__repr__().
        '''
        ch = 'XH' if chirality is None else chirality.upper()
        return 'Topology(<%s>, <%d faces>, <%d vertices>)' % (ch,tess.face_count,tess.vertex_count)
    
    def __repr__(self):
        return self.repr
    def make_mesh(self, coords, properties=None, meta_data=None):
        '''
        topo.make_mesh(coords) yields a Mesh object with the given coordinates and with the
          tesselation, meta_data, and properties inhereted from the topology topo.
        '''
        md = self.meta_data.set('topology', self)
        if meta_data is not None:
            md = pimms.merge(md, meta_data)
        ps = self.properties
        if properties is not None:
            ps = pimms.merge(ps, properties)
        return Mesh(self.tess, coords, meta_data=md, properties=ps)
    def register(self, name, coords):
        '''
        topology.register(name, coordinates) returns a new topology identical to topo that 
        additionally contains the new registration given by name and coordinates. If the argument
        coordinates is in fact a Mesh and this mesh is already in the topology with the given name
        then topology itself is returned.
        '''
        if not isinstance(coords, Mesh):
            coords = Mesh(self.tess, coords)
        elif name in self.registrations and not self.registrations.is_lazy(name):
            if self.registrations[name] is coords:
                return self
        return self.copy(_registrations=self.registrations.set(name, coords))
    def interpolate(self, topo, data,
                    registration=None, mask=None, weights=None,
                    method='automatic', n_jobs=1):
        '''
        topology.interpolate(topo, data) yields a numpy array of the data interpolated from the
          given array, data, which must contain the same number of elements as there are vertices
          tracked by the topology object (topology), to the coordinates in the given topology
          (topo). In order to perform interpolation, the topologies topology and topo must share at
          least one registration by name.
        
        The following options are accepted:
          * mask (default: None) indicates that the given True/False or 0/1 valued list/array should
            be used; any point whose nearest neighbor (see below) is in the given mask will, instead
            of an interpolated value, be set to the null value (see null option).
          * method (default: 'automatic') specifies what method to use for interpolation. The only
            currently supported methods are 'automatic', 'linear', or 'nearest'. The 'nearest'
            method does not  actually perform a nearest-neighbor interpolation but rather assigns to
            a destination vertex the value of the source vertex whose voronoi-like polygon contains
            the destination vertex; note that the term 'voronoi-like' is used here because it uses
            the Voronoi diagram that corresponds to the triangle mesh and not the true delaunay
            triangulation. The 'linear' method uses linear interpolation; though if the given data
            is non-numerical, then nearest interpolation is used instead. The 'automatic' method
            uses linear interpolation for any floating-point data and nearest interpolation for any
            integral or non-numeric data.
          * registration (default: None) specifies the registration to use. If None, interpolate
            will search for a shared registration aside from 'native'. If available, if will use the
            fsaverage, followed by the fs_LR.
          * n_jobs (default: 1) is passed along to the cKDTree.query method, so may be set to an
            integer to specify how many processors to use, or may be -1 to specify all processors.
        '''
        if is_address(topo):
            # we can use any surface since it's a mesh
            try: mesh = next(six.itervalues(self.registrations))
            except Exception: mesh = None
            try: mesh = next(six.itervalues(self.surfaces)) if mesh is None else mesh
            except Exception: mesh = None
            if mesh is None: raise ValueError('could not find mesh!')
            return mesh.interpolate(topo, data, mask=mask, weights=weights,
                                    method=method, n_jobs=n_jobs)
        elif not isinstance(topo, Topology):
            raise ValueError('Topologies can only be interpolated at a topology or an address')
        if registration is None:
            reg_names = [k for k in six.iterkeys(topo.registrations) if k in self.registrations
                         if k != 'native']
            # we want to apply some bit of ordering... fsaverage should be first if available, or
            # fs_LR for HCP subjects...
            if 'fs_LR' in reg_names:
                reg_names = ['fs_LR'] + [rn for rn in reg_names if rn != 'fs_LR']
            if 'fsaverage' in reg_names:
                reg_names = ['fsaverage'] + [rn for rn in reg_names if rn != 'fsaverage']
        else: reg_names = [registration]
        if not reg_names:
            raise RuntimeError('Topologies do not share a matching registration!')
        res = None
        errs = []
        for reg_name in reg_names:
            if True:
                res = self.registrations[reg_name].interpolate(
                    topo.registrations[reg_name], data,
                    mask=mask, method=method, n_jobs=n_jobs);
                break
            #except Exception as e: errs.append(e)
        if res is None:
            raise ValueError('All shared topologies raised errors during interpolation!', errs)
        return res
    def projection(self,
                   center=None, center_right=None, radius=None, method='orthographic',
                   registration='native', chirality=Ellipsis, sphere_radius=None,
                   pre_affine=None, post_affine=None, tag='projection'):
        '''
        topo.projection(...) is equivalent to MapProjection(...)(topo); effectively, this creates
          a map projection object, applies it to the topology topo (yielding a 2D mesh), and
          returns this mesh. To obtain the projection object itself, it is contained in the
          meta_data of the returned mesh (see also MapProjection).
        '''
        if chirality is Ellipsis: chirality = self.chirality
        proj = MapProjection(center=center, center_right=center_right, radius=radius,
                             method=method, registration=registration, chirality=chirality,
                             sphere_radius=sphere_radius,
                             pre_affine=pre_affine, post_affine=post_affine)
        return proj(self, tag=tag)
def is_topo(obj):
    '''
    is_topo(obj) yields True if obj is a Topology object and False otherwise.
    '''
    return isinstance(obj, Topology)
def topo(tess, registrations, properties=None, meta_data=None, chirality=None):
    '''
    topo(tess, regs) yields a Topology object with the given tesselation object tess and the given
      registration map regs.
    '''
    return Topology(tess, registrations, properties=properties, meta_data=meta_data,
                    chirarality=chirality)

####################################################################################################
# Curves, Loops, and Regions on the cortical surface
@pimms.immutable
class Path(ObjectWithMetaData):
    '''
    Path is a pimms immutable class that tracks curves or loops projected along a specific topology
    or cortex. A single path is defined in terms of a source topology/cortex (or map or mesh) and a
    set of ordered addresses within the given object. The addresses should include every point that
    is both on the path and is either a mesh vertex or a point along a mesh edge.

    To create a Path object, it is generally encourated that one use the path() function instead of
    the Path() constructor.
    '''
    def __init__(self, surface, addrs, meta_data=None):
        ObjectWithMetaData.__init__(self, meta_data)
        self.surface   = surface
        self.addresses = addrs
    @pimms.param
    def surface(s):
        '''
        path.surface is the surface mesh or topology on which the path is drawn.
        '''
        if isinstance(s, Topology) or isinstance(s, Mesh): return s.persist()
        else: raise ValueError('path surface must be a topology or a mesh')
    @pimms.param
    def addresses(addrs):
        '''
        path.addresses is the address map of the points in the curve.
        '''
        if not is_address(addrs): raise ValueError('addresses must be a valid address structure')
        return pimms.persist(addrs)
    @staticmethod
    def addresses_to_coordinates(surface, addresses):
        '''
        Path.addresses_to_coordinates(surface, addresses) yields the 3 x n coordinate matrix of the
        points represented in addresses along the given surface.
        '''
        x = surface.unaddress(addresses)
        x.setflags(write=False)
        return x
    @pimms.value
    def closed(addresses, surface):
        '''
        path.closed is True if, for the given path, the first and last points of the path are in
        the same face; otherwise, it is False. Empty curves are considered open, but are given a
        value of None. Note that a path of 1 point is considered closed, as is any path that lies
        entirely inside of a triangle.
        '''
        faces = address_data(addresses)[0]
        if faces is None: return None
        (f1,f2) = faces.T[[-1,0]]
        ol = len(np.unique([f1,f2]))
        return (ol == 3)
    @pimms.value
    def coordinates(surface, addresses):
        '''
        path.coordinates is, if path.surface is a mesh, a (3 x n) matrix of coordinates of the
        points in the path along the surface of path.surface, such that the path exactly follows
        the mesh. If path.surface is a topology, then this is a lazy-map of coordinate matrices
        with the same keys as path.surface.surfaces.
        '''
        if isinstance(surface, Mesh): return addresses_to_coordinates(surface, addresses)
        surfs = surface.surfaces
        fn = lambda k: Path.addresses_to_coordinates(surfs[k], addresses)
        return pimms.lazy_map({k:curry(fn, k) for k in six.iterkeys(surfs)})
    @pimms.value
    def length(coordinates, closed):
        '''
        path.length, if path.surface is a mesh, is the length of the given path along its mesh. If
        path.surface is a topology, path.length is a lazy map whose keys are the surface names and
        whose values are the lengths of the path along the respective surface.
        '''
        x = coordinates
        if pimms.is_map(coordinates):
            if closed: dfn = lambda k:np.sum(np.sqrt(np.sum((x[k] - np.roll(x[k], -1))**2,axis=0)))
            else:      dfn = lambda k:np.sum(np.sqrt(np.sum((x[k][:,:-1] - x[k][:,1:])**2,axis=0)))
            return pimms.lazy_map({k:curry(dfn,k) for k in six.iterkeys(coordinates)})
        else:
            if closed: return np.sum(np.sqrt(np.sum((x - np.roll(x, -1))**2, axis=0)))
            else:      return np.sum(np.sqrt(np.sum((x[:,:-1] - x[:,1:])**2, axis=0)))
    @pimms.value
    def edge_data(addresses, closed):
        '''
        path.edge_data is a tuple of (u, v, wu, wv, fs, ps) where u and v are arrays such that each
        edge in path.surface that intersects the given path is given by one of the (u[i],v[i]), and
        the relative distancea along the path is specified by the arrays of weights, wu and wv. If
        the given path is closed, then for each edge, one weight will be > 0.5 and one weight will
        be smaller; the relative position of 0.5 exactly specifies the relative distance along the
        edge that the intersection occurs. If the given path is not closed, then wu[i] + wv[i] will
        equal 0.5 instead of equalling 1.  The final two tuple element, fs and ps, are also tuples
        the same length as u and v. In fs[i], the face formed by the edges (u[i], v[i]) and
        (u[i+1], v[i+1]) is given; in ps[i], a tuple of all path point indices (i.e., into the
        path.addresses data arrays) in order that pass through fs[i] is given. If path is closed,
        then the final element of ps and fs will wrap around to the beginning of the path;
        otherwise both will be None. For any instance in which u[i] and v[i] are equal (i.e., the
        path passes through a vertex), fs[i] will be None and ps[i] will be a tuple containing
        only the vertex.

        If the path is closed, the the u are always on the inside of the label while the v are
        always on the outside.
        '''
        # walk along the address points...
        (faces, coords) = address_data(addresses, 2)
        coords = np.vstack([coords, [1 - np.sum(coords, axis=0)]])
        n = faces.shape[1]
        (u,v,wu,wv,fs,ps) = ([],[],[],[], [], [])
        pcur = []
        lastf = faces[:, -1] if closed else None
        maxv = np.max(faces) + 1
        mtx = sps.dok_matrix((maxv,maxv))
        for (ii,f,w) in zip(range(n), faces.T, coords.T):
            zs = np.isclose(w, 0, atol=1e-5)
            nz = np.sum(zs)
            pcur.append(ii)
            if nz == 0: # inside the triangle--no crossings
                lastf = f
                continue
            elif nz == 2: # exact crossing on a vertex
                vtx = f[~zs][0]
                for q in [u,v]:   q.append(vtx)
                for q in [wu,wv]: q.append(0.5)
            elif nz == 1: # crossing an edge
                k = [0,1] if zs[2] else [1,2] if zs[0] else [2,0]
                # see if this is specified in cw or ccw relative to edge
                rev = 5 if lastf is None else len(np.unique([lastf, f])) # will be 3, 4, or 5
                if   rev == 3: k = list(reversed(k))
                elif rev == 4:
                    rev = len(np.unique(np.concatenate([lastf, f[k]])))
                    if rev == 4 and (len(u) < 1 or not (u[-1] not in f or v[-1] not in f)):
                        k = list(reversed(k))
                if len(u) == 0: (uv,vtx) = (f[k],None)
                else:
                    uvl = [u[-1],v[-1]]
                    sdif = np.setdiff1d(f[k], uvl)
                    if len(sdif) == 0: (uv,vtx) = (uvl[::-1], np.setdiff1d(lastf, uv)[0])
                    else:              (uv,vtx) = (f[k],      sdif[0])
                for (q,qq) in zip([u,v],   uv):   q.append(qq)
                for (q,qq) in zip([wu,wv], w[k]): q.append(qq)
            else: raise ValueError('address contained all-zero weights',
                                   dict(faces=faces, coords=coords, ii=ii, f=f, w=w, pcur=pcur,
                                        u=u, v=v, vtx=vtx))
            if u[-1] != v[-1]: mtx[u[-1],v[-1]] += 1
            if lastf is None or vtx is None: ff = None
            elif u[-2] == v[-2]:
                # the last edge was actually a point; a couple possibilities: this point ii is...
                # (1) in the adjacent face (i.e., f contains u[-2])
                # (2) on the opposite edge but in the next face (nz == 1 and u[-2] not in f)
                # [3] equivalent to u[-2] (in which case condition (1) is also true)
                if   u[-2] in f: ff = f
                elif nz == 1:    ff = (u[-2], f[k[1]], f[k[0]])
                else: raise ValueError('point followed by non-deducible face',
                                       dict(faces=faces, coords=coords, ii=ii, f=f, w=w, pcur=pcur,
                                            u=u, v=v, vtx=vtx))
            elif vtx == u[-2] or vtx == v[-2]:
                if nz == 2:
                    # we're still at the same point;
                    ff = f
                else: raise ValueError(
                        'Unexpected condition deducing triangle: u[-2] or v[-2] equal vtx',
                        dict(faces=faces,coords=coords,ii=ii,f=f,w=w,pcur=pcur,u=u,v=v,vtx=vtx))
            else: ff = (u[-2], v[-2], vtx)
            assert(ff is None or len(np.unique(ff)) == 3)
            fs.append(ff)
            ps.append(tuple(pcur))
            pcur = [ii]
            lastf = f
        # okay, we need to fix the first/last elements and the fs/ps order: the fs and ps have been
        # appended *after* the edge that closes the triangle rather than between the opening and
        # closing edge. The first elements are also incomplete--in a closed loop they should be
        # joined to the last elements; in an open loop they are discarded and the last element is
        # set to None
        if closed:
            tmp = np.setdiff1d((u[0],v[0]), (u[-1],v[-1])) if len(u) > 1 else []
            if len(tmp) == 0:
                (f0,f1) = faces[:,[0,-1]].T
                (x0,x1) = coords[:,[0,-1]].T
                # most likely we start/end in same face but exit/enter through one edge
                f0but1 = np.setdiff1d(f0,f1)
                if len(f0but1) == 0:
                    if len(fs) > 0: fs[0] = f0
                    if len(ps) > 0: pcur = (ps[-1][-1],) + ps[0]
                    else:           pcur = pcur + [pcur[0]]
                elif len(f0but1) == 1:
                    f0ii = np.where(f0 == f0but1[0])[0]
                    if np.isclose(x0[foii], 0): fs[0] = f1
                    else: fs[0] = f0
                else: raise ValueError('closed path does not start/end correctly',
                                       dict(faces=faces, coords=coords, u=u, v=v, vtx=vtx,
                                            pcur=pcur, f0=f0, f1=f1, x0=x0, x1=x1))
                if len(ps) > 0: ps[0] = tuple(pcur)
            elif len(tmp) == 1:
                fs[0] = (u[-1], v[-1], tmp[0])
                ps[0] = tuple(pcur)[:-1] + ps[0]
            else: raise ValueError('closed path does not start/end in same face',
                                   dict(faces=faces, coords=coords, u=u, v=v, vtx=vtx))
        else:
            fs[0] = None
            ps[0] = None
        # any edge that was crossed by the path an even number of times isn't actually in the path
        # if closed, so we set both values to 0.25
        (u,v,wu,wv) = [np.asarray(x) for x in (u,v,wu,wv)]
        (ii,jj,qq) = sps.find(mtx + mtx.T)
        k = np.where(np.mod(qq, 2) == 0)[0]
        for (uu,vv) in zip(ii[k], jj[k]):
            wu[u == uu] = 0.25
            wv[v == vv] = 0.25
        fs = np.roll(fs, -1, axis=0) if len(fs) > 0 else np.zeros((0,3), dtype=np.int)
        ps = np.roll(ps, -1, axis=0) if len(ps) > 0 else np.array([],    dtype=np.object)
        if not closed: (wu,wv) = [np.asarray(w) * 0.5 for w in (wu,wv)]
        return tuple(map(pimms.imm_array, (u,v,wu,wv,fs,ps)))
    @pimms.value
    def intersected_edges(edge_data):
        '''
        path.intersected_edges is a (2 x n) matrix of edges that are crossed by the given path in
        order along the path; each column (u,v) of the returned matrix gives one edge, given in
        counter-clockwise order relative to path (i.e., u is always inside a closed path or on the
        left side of an open path).
        '''
        return pimms.imm_array([edge_data[0], edge_data[1]])
    @pimms.value
    def intersected_edge_indices(edge_data, surface):
        '''
        path.intersected_edge_indices is equivalent to to path.intersected_edges except that it
        stores a list of edge indices instead of a matrix of edges.
        '''
        idx = surface.tess.index
        return pimms.imm_array([idx[(u,v)] for (u,v) in zip(edge_data[0], edge_data[1])])
    @pimms.value
    def label(surface, edge_data, closed):
        '''
        path.label is either None if the given path is not a closed path or an array of values, one 
        per vertex in path.surface (i.e., a surface property), between 0 and 1 that specifies which
        vertices are inside the closed path and which are outside of it. The specific values will
        always be 1 for any vertex inside the area, 0 for any vertex outside the area, and a number
        between 0 and 1 for vertices that are adjacent to the boundary (i.e., the path intersects
        the vertex or one of its adjacent edges). For a given edge (u,v) that intersects the path,
        the label values of u and v (lu and lv) will be such that the exact position of the edge
        intersection with the path, for vertex positions xu and xv, is given by:
        (lu*xu + lv*xv) / (lu + lv)
        As a simple course measure, the value of (path.label >= 0.5) is a boolean property that is
        true for all the vertices inside the closed path.
        The inside of a label is always determined by the side of the label closed with respect to
        the left side of a counter-clockwise path.
        '''
        # we only need the tesselation...
        tess = surface.tess
        n = tess.vertex_count
        # we know from addresses where the intersections are freom edge_data
        (u,v,wu,wv) = edge_data[:4]
        # convert weights
        (wu,wv) = (0.75 + 0.5*(np.asarray(wu) - 0.5), 0.25 + 0.5*(np.asarray(wv) - 0.5))
        # labels need to be indexed...
        (u,v) = (tess.index(u), tess.index(v))
        same  = np.union1d(u,v)
        (q,wq) = [np.concatenate([a,b]) for (a,b) in [(u,v),(wu,wv)]]
        m = len(q)
        # for the labels, the u and v have repeats, so we want to average their values
        mm  = sps.csr_matrix((np.ones(m), (q, np.arange(m))), shape=(n, m))
        lbl = zdivide(mm.dot(wq), flattest(mm.sum(axis=1)))
        # we crawl across vertices by edges until we find all of them
        nei  = np.asarray(tess.indexed_neighborhoods)
        unk  = np.full(tess.vertex_count, True, dtype=np.bool)
        unk[q] = False
        q = np.unique(u[~np.isin(u, v)])
        while len(q) > 0:
            # get all their neighbors
            q = np.unique([k for neitup in nei[q] for k in neitup]).astype(np.int64) 
            q = q[unk[q]] # only not visited neighbors
            lbl[q] = 1.0 # they are inside the region now
            unk[q] = False # now we've visited them
        return lbl
    @pimms.value
    def minlabel(label):
        '''
        path.minlabel is equivalent to path.label if path.label contains fewer or equal to half of
          the vertices vertices than its
          complement and is equivalent to 1 - path.label if not.
        '''
        ls = np.sum(label[label != 0.5].astype('bool'))
        return label if ls <= len(label) - ls else pimms.imm_array(1 - label)
    @pimms.value
    def maxlabel(minlabel):
        '''
        path.maxlabel is equivalent to path.label if path.label contains more vertices than its
          complement and is equivalent to 1 - path.label if not.
        '''
        return pimms.imm_array(1 - minlabel)
    @pimms.value
    def contained_faces(surface, label):
        '''
        path.contained_faces is a matrix of faces in path.surface that are completely contained
          by the given path; any face that intersects the path will not be included. Faces are
          returned in a (3 x n) matrix of vertex labels.
        '''
        msk = np.where(label >= 0.5)[0]
        fs = np.where(np.sum(np.isin(surface.tess.indexed_faces, msk), axis=0) == 3)[0]
        return pimms.imm_array(surface.tess.faces[:,fs])
    @pimms.value
    def intersected_faces(edge_data, closed):
        '''
        path.intersected_faces is a matrix of faces in path.surface that intersect the given path.
          Faces are returned in a (3 x n) matrix of vertex labels. Note that faces that intersect
          the path at a single point are not included (i.e., the path must go through the face).

        All faces in intersected_faces are listed such that path.intersected_edges is equal to the
        first two rows of of intersected_faces, up to the last column; in other words, the faces
        are given in order along/around the path, and each face is ordered in the same direction
        (clockwise or counter-clockwise) as the path with respect to the path.surface outer normal,
        starting with the two vertices whose edge is crossed.
        '''
        res = []
        uv = np.transpose(edge_data[:2])
        (u0,v0) = uv[-1] if closed else (np.nan,np.nan)
        for (u,v) in uv:
            res.append([u0,v0,np.setdiff1d((u,v), (u0,v0))[0]])
            (u0,v0) = (u,v)
        res = np.array(res[1:] if not np.isfinite(res[0][0]) else res)
        res.setflags(write=False)
        return res
    @pimms.value
    def intersected_face_paths(edge_data, addresses, closed):
        '''
        path.intersected_face_paths is a dict of all the faces intersected by the the given path
        along with the barycentric coordinates of all points that lie in that face. This means that
        if you instantiate these points and list them all there will be duplicate points every time
        the path crosses an edge. Keys of this dict are the face tuples (a,b,c) rotated such that a
        is always the label with the lowest value.
        '''
        (faces, coords) = address_data(addresses, 2)
        coords = np.vstack([coords, [1 - np.sum(coords, axis=0)]])
        (faces, coords) = [x.T for x in (faces,coords)]
        def bc_conv(f0, x0, ftarg):
            r = np.zeros(len(x0))
            for (f,x) in zip(f0,x0):
                # if the node is not found we use a much more tolerant version of the error--such a
                # situation likely indicates a rounding error at an exact node crossing
                if   np.isclose(x, 0, atol=1e-5): continue
                elif f in ftarg:                  r[f == ftarg] = x
                elif np.isclose(x, 0, atol=1e-3): continue
                else: raise ValueError('Non-zero bc-conv value',
                                       dict(edge_data=edge_data, addresses=addresses, closed=closed,
                                            f0=f0, x0=x0, ftarg=ftarg, f=f, x=x))
            return r
        # walk along edge data; mostly this isn't too hard
        (fs,ps) = edge_data[4:6]
        if fs[-1] is None: fs = fs[:-1]
        # standardize them all without reflecting them
        fs = np.asarray([np.roll(f, -np.argmin(f)) for f in fs])
        idx = AutoDict()
        idx.on_miss = lambda:[]
        for (f,p) in zip(zip(*fs.T), ps): idx[f].append(p)
        # make each of these into a separate entry
        def to_bcs(f,ps):
            bcs = np.asarray([bc_conv(faces[p], coords[p], f) for p in ps]).T[:2]
            bcs.setflags(write=False)
            return bcs
        try: return pyr.pmap({f: tuple([to_bcs(f,p) for p in ps]) for (f,ps) in six.iteritems(idx)})
        except ValueError as e:
            if len(e.args) > 1 and isinstance(e.args[1], dict): e.args[1]['idx'] = idx
            raise
    @staticmethod
    def tesselate_triangle_paths(paths):
        '''
        tesselate_triangle_paths([path1, path2...]) yields a (3x2xN) array of N 2D triangles that
          tesselate a triangle without crossing any of the given paths. The paths themselves should
          be in barycentric coordinates and should each start and end on an edge without touching 
          any edge between. The 2nd dimension consists of the first two barycentric coordinates
          while the first dimensions corresponds to the edges of each tesselated triangle.

        This function is probably not particularly performant--it operates by trying to connect
        every pair of vertices and rejecting a pairing if it crosses any existing line.
        '''
        # these coords are used to reify BC triangles while figuring them out...
        A = np.array([0.0,              0.0])
        B = np.array([1.0/np.sqrt(2.0), 0.0])
        C = np.array([0.0,              1.0/np.sqrt(2.0)])
        (a21,b21,c21) = [np.reshape(x,(2,1)) for x in (A,B,C)]
        paths = [path if len(path) == 3 else np.vstack([path, [1 - np.sum(path,0)]])
                 for path in paths]
        # for starters, we want to build up a matrix of all the individual reified coordinates
        bccoords = np.hstack([[[1,0,0],[0,1,0],[0,0,1]], np.hstack(paths)]) # first three are A,B,C
        bccoords0 = np.array(bccoords)
        coords = a21*bccoords[0] + b21*bccoords[1] + c21*bccoords[2] # reify all
        coords0 = np.array(coords)
        (coords, bccoords) = [x.T for x in (coords, bccoords)]
        # turn paths into indices instead of coords
        (pidcs,k) = ([],3) # 3 is the first non-corner index in bccoords/coords
        for path in paths:
            qq = np.shape(path)[1]
            pidcs.append(k + np.arange(qq))
            k += qq
        # now we want to make sure there aren't duplicate points
        idcs = np.arange(len(coords)) # starting indices
        ridcs = []
        n = 0
        for (ii,x) in enumerate(coords):
            if idcs[ii] < ii: continue
            dists = np.sqrt(np.sum((coords[ii:] - x)**2, axis=1))
            idcs_eq = np.where(np.isclose(dists, 0, atol=1e-4))[0]
            idcs[idcs_eq + ii] = n
            n += 1
            ridcs.append(ii)
        # if we overwrote the last one, we actually want to keep it
        if np.sum(idcs[3:] == idcs[-1]) > 1:
            qii = ridcs[idcs[-1]]
            coords[qii] = coords[-1]
            bccoords[qii] = bccoords[-1]
        if n > 64: warnings.warn('tesselating face with %d points: poor performance is likely' % n)
        elif np.max(idcs) < 4:
            assert(len(coords) > 4)
            # intersection at a single point or at two points; regardless, we tesselate into the
            # original triangle only; however, we have to figure out which side is LHS and RHS
            if idcs[-1] == idcs[-2]:
                # intersecting at a point; we determine if this is LHS/RHS by checking the
                # (near-vertex) points that were given anyway
                t = [coords[-2], coords[-1], coords[idcs[-1]]]
                if np.cross(coords[-2] - coords[-1], coords[idcs[-1]] - coords[-1]) <= 0:
                    return (np.zeros([3,2,0]), np.reshape([[1,0],[0,1],[0,0]], (3,2,1)))
                else:
                    return (np.reshape([[1,0],[0,1],[0,0]], (3,2,1)), np.zeros([3,2,0]))
            else:
                # intersecting at two points; we determine LHS/RHS by whether the points are
                # ordered as in the triangle
                if (idcs[-2] + 1) % 3 == idcs[-1]:
                    return (np.reshape([[1,0],[0,1],[0,0]], (3,2,1)), np.zeros([3,2,0]))
                else:
                    return (np.zeros([3,2,0]), np.reshape([[1,0],[0,1],[0,0]], (3,2,1)))
        # fix the paths and coords matrices
        pidcs = [[i0 for (i0,i1) in zip(pii[:-1],pii[1:]) if i0 != i1] + [pii[-1]]
                 for p in pidcs
                 for pii in [idcs[p]]]
        coords   = coords[ridcs]
        bccoords = bccoords[ridcs]
        # we'll need to know what points are along what edge as part of this: we'll collect the
        # point indices here--doing so is easier with the barycentric than the reified
        # coordinates, so we'll use those and reify the coordinates as we go
        internal_paths = [[],[]] # [0] is start, [1] is end; we use coordinate ids
        on_edges = ([],[],[])
        for pii in pidcs:
            # make sure first/last are on an edge and the rest aren't; note that it's okay, though,
            # to have points along the edge that we enter/exit on
            bcx = bccoords[pii]
            x   = coords[pii]
            zz = np.isclose(bcx, 0, atol=1e-5)
            (k0,ke) = (1, len(bcx) - 2)
            for (k_, _k, dr) in zip([k0,ke], [ke,k0], [1,-1]):
                while dr*k_ < dr*_k and np.sum(zz[k_]) > 0:
                    wz = np.where(zz[k_])[0]
                    if len(wz) == 0 or not zz[k_-dr, wz].any(): break
                    k_ = k_ + dr
                if dr == 1: k0 = k_
                else:       ke = k_
            if np.sum(np.isclose(bcx[k0:ke], 0, atol=1e-5)) > 0:
                raise ValueError('path middle touches edge',
                                 dict(coords0=coords0, bccoords0=bccoords0,
                                      coords=coords, bccoords=bccoords,
                                      pii=pii, pidcs=pidcs, idcs=idcs, ridcs=ridcs))
            (p0,p1) = bcx[[0,-1]]
            (z0,z1) = [np.isclose(p,0,atol=1e-4) for p in (p0,p1)]
            (s0,s1) = [np.sum(z)                 for z in (z0,z1)]
            (e0,e1) = [((0 if     z[2] else 1 if     z[0] else 2) if s == 1 else
                        (0 if not z[0] else 1 if not z[1] else 2) if s == 2 else
                        None)
                       for (z,s) in [(z0,s0), (z1,s1)]]
            if e0 is None or e1 is None:
                raise ValueError('path-piece does not start/end on edge',
                                 dict(coords0=coords0, bccoords0=bccoords0,
                                      coords=coords, bccoords=bccoords,
                                      pii=pii, pidcs=pidcs, idcs=idcs, ridcs=ridcs))
            # find fractional distance from first to second vertex
            (w0,w1) = (bcx[0, np.mod(e0 + 1, 3)], bcx[-1, np.mod(e1 + 1, 3)])
            # put these in the on_edges
            on_edges[e0].append((pii[0],  w0))
            on_edges[e1].append((pii[-1], w1))
        # okay, we've validated and prepped the paths; now sort everything on each edges so that we
        # can make them into segment lists
        segs     = np.zeros((2,2,n*n))
        seg_idcs = np.zeros((2,n*n), dtype=np.int)
        m = 0 # number of segs so far
        # (segs[i][j][k] is the j'th coordinates (j=0:x, j=1:y) of the start (i=0) or end (i=1)
        #  of the k'th segment; this way segs can be passed straight to segments_intersection_2D)
        eidx = [None,None,None]
        for (ii,oes) in enumerate(on_edges):
            # first sort this edge's intersection points by distance
            idcs = np.asarray([u[0] for u in sorted(oes, key=lambda u:u[1])], dtype=np.int)
            eidx[ii] = idcs
            xs = coords[idcs]
            l = len(xs)
            seg_idcs[0,m]         = ii
            seg_idcs[0,m+1:m+l+1] = idcs
            seg_idcs[1,m:m+l]     = idcs
            seg_idcs[1,m+l]       = np.mod(ii+1, 3)
            m = m + l + 1
        # once those are added to segs, we just need to add the internal segs from pidcs
        for pii in pidcs:
            l = len(pii)
            seg_idcs[0,m:m+l-1] = pii[:-1]
            seg_idcs[1,m:m+l-1] = pii[1:]
            m = m + l - 1
        segs[:,:,:m] = np.asarray([coords[ii[:m]].T for ii in seg_idcs])
        # okay, we can now do the segmentation! Try every edge; take as many as possible that do not
        # cross each other or any other edge.
        seg_idcs = set(zip(*seg_idcs[:,:m]))
        for i in range(n):
            for j in range(i+1,n):
                # if this edge (a) is in the segs already or (b) is colinear with anything in segs
                # or (c) intersects anything in segs, then we reject it
                if (i,j) in seg_idcs or (j,i) in seg_idcs: continue
                ss = [coords[i], coords[j]]
                if segments_overlapping(segs, ss).any(): continue
                sii = np.asarray(segment_intersection_2D(segs, ss))
                sii = sii[:,np.isfinite(sii[0])]
                if len(sii) > 0:
                    sii = ~(points_close(sii, ss[0]) | points_close(sii, ss[1]))
                    if sii.any(): continue
                if point_in_segment(ss, coords.T).any(): continue
                # otherwise we add it to our collection of edges
                seg_idcs.add((i,j))
                segs[:,:,m] = [coords[i], coords[j]]
                m += 1
        # okay, we've drawn the edges; now we find faces by examining the neighborhood of each node
        tris = set([])
        neis = {u:set([]) for u in range(n)}
        for (u,v) in seg_idcs:
            if u == v: continue
            neis[u].add(v)
            neis[v].add(u)
        for (a,nei) in six.iteritems(neis):
            nei = np.asarray(list(nei))
            # we sort each of the neighbors by their angle around u
            (x,y) = (coords[nei] - coords[a]).T
            ths   = np.arctan2(y, x)
            ii    = np.argsort(ths)
            nei   = nei[ii]
            ths   = ths[ii]
            dths  = np.mod(np.roll(ths, -1) - ths, 2*np.pi)
            # every three must be a triangle
            for (u,v,th) in zip(nei, np.roll(nei, -1), dths):
                abc = [a,u,v]
                if a == u or u == v or v == a:
                    raise ValueError('bad triangle in neighbor search',
                                     dict(bccoords0=bccoords0, coords0=coords0,
                                          bccoords=bccoords, coords=coords,
                                          pidcs=pidcs, idcs=idcs, ridcs=ridcs,
                                          a=a, nei=nei, ths=ths, tris=tris, seg_idcs=seg_idcs))
                # avoid colinear points and angles >= 180, i.e., any three points on the same edge
                if np.isclose(th, [0,np.pi]).any() or th >= np.pi: continue
                if any(np.sum(np.isin(abc, ee)) == 3 for ee in eidx): continue
                tris.add(tuple(np.roll(abc, -np.argmin(abc))))
        # That's the set of triangles--we can detect which are on the RHS/LHS of the the path by
        # looking at the edges that make-up the path; we can be clever about this and use a sparse
        # matrix where (u,v) and (v,u) are different
        mtx = sps.lil_matrix((n,n), dtype=np.int)
        for pii in pidcs:
            for (u,v) in zip(pii[:-1],pii[1:]):
                mtx[u,v] = 1
                mtx[v,u] = -1
        # now go through the triangles and look for edges that indicate direction
        tskip = tris
        (lhs, rhs) = (set([]), set([]))
        qq = 0
        while len(tskip) > 0:
            qq = qq + 1
            tris = tskip
            tskip = set([])
            for abc in tris:
                (a,b,c) = abc
                itlst = list(zip([1,1,1,-1,-1,-1], [a,b,c,a,b,c], [b,c,a,c,a,b]))
                h = next((sgn*h for (sgn,u,v) in itlst for h in [mtx[u,v]] if h != 0), None)
                if h is None:
                    tskip.add(abc)
                    continue
                for (u,v) in zip(abc,(b,c,a)):
                    if mtx[v,u] == 0: mtx[v,u] = h
                (lhs if h == 1 else rhs).add(abc)
            if tskip == tris: raise ValueError('infinite loop: %s' % tskip)
        # Convert them back to addresses; we only want first 2 BC coordinates for return value
        bccoords = bccoords[:,:2]
        return tuple([np.transpose(ll, (1,2,0)) if len(ll) > 0 else np.array(ll)
                      for hs in (lhs,rhs)
                      for ll in [[bccoords[abc] for abc in np.asarray(list(hs))]]])
    @pimms.value
    def all_border_triangle_addresses(edge_data, addresses, closed, intersected_face_paths):
        '''
        path.all_border_triangle_addresses contains a nested tuple ((a,b,c), (d,e,f)); each of the
        (a,b,c) and (d,e,f) tuples represent arrays of triangles--each of the a-f represent an
        addresses-object of points in path.surface that form the triangles' corners. Together, all
        the triangles in the two triangle arrays are congruent to the triangles in 
        path.intersected_faces; however, the border_triangle_addresses have been split into sets of
        smaller triangles that are exclusively on the inner side of the path border (triangles in 
        (a,b,c)) and those exclusively on the outer side of the path border (triangles in (d,e,f)).
        Among other things, this splitting of the triangles is required to calculate precise
        surface area measurements of regions contained by closed paths.

        In the case that path is not closed, the triangles in (a,b,c) are on the left side of the
        path while those in (d,e,f) are on the right side, with respect to the direction of path.
        '''
        # we need to step through each of the intersected faces, tesselate it, then determine which
        # of the tesselated triangles are inside/outside the mesh
        (lhs,rhs,lfs,rfs) = ([],[],[],[])
        for (abc,bcxs) in six.iteritems(intersected_face_paths):
            try: (ll,rr) = Path.tesselate_triangle_paths(bcxs)
            except ValueError as e:
                if len(e.args) > 1 and isinstance(e.args[1], dict):
                    e.args[1]['abc'] = abc
                    e.args[1]['bcxs'] = bcxs
                    e.args[1]['idx'] = len(lhs)
                raise e
            if len(ll) == 0 or len(rr) == 0: continue
            lhs.append(ll)
            rhs.append(rr)
            lfs.append(np.matlib.repmat([abc],ll.shape[2],1))
            rfs.append(np.matlib.repmat([abc],rr.shape[2],1))
        lhs = np.concatenate(lhs, axis=2)
        rhs = np.concatenate(rhs, axis=2)
        lfs = np.vstack(lfs).T
        rfs = np.vstack(rfs).T
        for x in (lhs,rhs,lfs,rfs): x.setflags(write=False)
        return tuple([tuple([pyr.m(faces=fs, coordinates=xs) for xs in hs])
                      for (hs,fs) in zip([lhs,rhs],[lfs,rfs])])
    @pimms.value
    def all_border_triangles(all_border_triangle_addresses, surface):
        '''
        path.all_border_triangles is tuple of two (3 x d x n) arrays--the first dimension of each
        array corresponds to the triangle vertex, the second to the x/y/z coordinate of the vertex,
        and the final dimension to the triangle id. The first gives the inner or left border
        triangle coordinates while the second gives the right or the outer coordinates.

        If path.surface is a topology, then border_triangles is instead a lazy-map of the border
        triangle coordinates, one set for each surface in topo.
        '''
        conv = lambda surf: tuple([tuple([pimms.imm_array(surf.unaddress(xs).T) for xs in bta])
                                   for bta in all_border_triangle_addresses])
        if isinstance(surface, Topology):
            return pimms.lazy_map({k:curry(lambda k: conv(surface.surfaces[k]), k)
                                   for k in six.iterkeys(surface.surfaces)})
        else: return conv(surface)
    @pimms.value
    def border_triangles(all_border_triangles):
        '''
        path.border_triangles contains the coordinates of the inner partial-face triangles
        that lie on the boundary of the given path; if the path is not closed, these correspond to
        the left parts of the intersected faces. For the border triangles that are outside or right
        of the border use path.outer_border_triangles. See also all_border_triangles.
        '''
        if pimms.is_map(all_border_triangles):
            return pimms.lazy_map({k:curry(lambda k:all_border_triangles[k][0], k)
                                   for k in six.iterkeys(all_border_triangles)})
        else: return all_border_triangles[0]
    @pimms.value
    def outer_border_triangles(all_border_triangles):
        '''
        path.outer_border_triangles contains the coordinates of the outer partial-face triangles
        that lie on the boundary of the given path; if the path is not closed, these correspond to
        the right parts of the intersected faces. See also border_triangles and
        all_border_triangles.
        '''
        if pimms.is_map(all_border_triangles):
            return pimms.lazy_map({k:curry(lambda k:all_border_triangles[k][1], k)
                                   for k in six.iterkeys(all_border_triangles)})
        else: return all_border_triangles[1]
    @pimms.value
    def surface_area(border_triangles, contained_faces, surface, closed):
        '''
        path.surface_area is the surface area of a closed path; if path is not closed, then this
        is None.
        
        If path.surface is a topology rather than a mesh, then this is a lazy-map of surface names
        whose values are the surface area for the given surface. If the path is not closed, this
        remains None.
        '''
        if not closed: return None
        contained_faces = surface.tess.index(contained_faces)
        def sarea(srf):
            if pimms.is_str(srf): (srf,btris) = (surface.surfaces[srf],border_triangles[srf])
            else: btris = border_triangles
            btris = np.transpose(btris, [0,2,1])
            cxs = np.asarray([srf.coordinates[:,f] for f in contained_faces])
            return np.sum(triangle_area(*cxs)) + np.sum(triangle_area(*btris))
        if isinstance(surface, Topology):
            return pimms.lazy_map({k:curry(sarea, k) for k in six.iterkeys(surface.surfaces)})
        else: return sarea(surface)
    @staticmethod
    def estimate_distances(addresses, mesh):
        '''
        Path.estimate_distances(addresses, mesh) estimates all the distances between the vertices in
          the mesh and the path implied by the given set of addresses using a minimum-graph-distance
          algorithm over the mesh edges.
        '''
        # we're going to estimate an unsigned distance for every vertex, one can add the sign in for
        # a closed path using the contained_faces or similar if desired
        d = np.full(mesh.vertex_count, np.inf) # initial distances
        # for starters, we need to get distances from all the addresses
        (faces, coords) = address_data(addresses, 2)
        coords = np.vstack([coords, [1 - np.sum(coords, axis=0)]])
        faces  = mesh.tess.index(faces)
        # each face implies a minimum distance to each of its vertices
        for (f,x) in zip(faces.T, coords.T):
            fx = mesh.coordinates[:,f]
            pt = np.sum(fx * x, axis=1)
            # distances between...
            dd = np.sqrt(np.sum((fx.T - pt)**2, axis=1))
            ii = (d[f] > dd)
            d[f[ii]] = dd[ii]
        # okay, starting distances obtained; now let's do a search!
        ii0 = np.where(np.isfinite(d))[0]
        if len(ii0) == 0: raise ValueError('No distances obtained from addresses!')
        # we iterate until we cannot decrease the path-length of anything
        (max_iter,it) = (mesh.vertex_count + 1, 0)
        (u,v) = np.hstack([mesh.tess.indexed_edges, np.flipud(mesh.tess.indexed_edges)])
        elens = np.concatenate([mesh.edge_lengths, mesh.edge_lengths])
        updated = ii0
        while len(updated) > 0:
            if it > max_iter: raise ValueError('Max iterations exceeded in min-distance search loop')
            else: it += 1
            # what edges might be affected by this update
            ii = np.where(np.isin(u, updated))[0]
            (uu,vv) = (u[ii],v[ii])
            dnew = d[uu] + elens[ii]
            # we want to compare into vv, but vv will have repeats, so we must min across these;
            # easy way to do this is to put them all in a sparse matrix and min/argmin across
            (vv,ridx) = np.unique(vv, return_inverse=True)
            # we want to take the smallest value that is greater than 0, so we have to use a trick
            mx = np.max(dnew) + 1.0
            mtx = sps.csr_matrix((mx - dnew, (ridx, np.arange(len(dnew)))))
            dnew = mx - flattest(mtx.max(axis=1))
            # okay, find out where this distance is less than the previous dist
            jj = np.where(d[vv] > dnew)[0]
            updated = vv[jj]
            d[updated] = dnew[jj]
        return d
    @pimms.value
    def estimated_distances(addresses, surface):
        '''
        path.estimated_distances is a vector of estimated distance values from the given path. The
        distance estimates are upper bounds on the actual distance but are not exact.
        '''
        if is_topo(surface):
            return pimms.lazy_map(
                {k:curry(lambda a,k: Path.estimate_distances(a, surface.surfaces[k]), addresses, k)
                 for k in six.iterkeys(surface.surfaces)})
        else: return Path.estimate_distances(addresses, surface)
    @pimms.value
    def estimated_signed_distances(estimated_distances, contained_faces, surface):
        '''
        path.estimated_distances is a vector of estimated distance values from the given path. The
        distance estimates are upper bounds on the actual distance but are not exact.
        '''
        fs = surface.tess.index(np.unique(contained_faces))
        mlt = np.ones(surface.vertex_count, dtype=np.int)
        mlt[fs] = -1
        if is_topo(surface):
            return pimms.lazy_map(
                {k:curry(lambda k: estimated_distances[k]*mlt, k)
                 for k in six.iterkeys(estimated_distances)})
        else: return estimated_distances * mlt
    def reverse(self, meta_data=None):
        '''
        path.reverse() yields a path that is equivalent to the given path but reversed (thus it is
          considered to contain all the vertices not contained by path if path is closed).
        '''
        addrs = {k:np.fliplr(v) for (k,v) in six.iteritems(self.addresses)}
        return Path(self.surface, addrs, meta_data=meta_data)
    def boundary_vertices(self, distance, outer=False, indices=False):
        '''
        path.boundary_vertices(d) yields an array of vertex labels representing all vertices that
          are both on the on the inner/left-hand side of the given path and within a distance d of
          the boundary of the path.
        
        The following options may be given:
          * outer (default: False) may be set to True in order to use the right-hand/outer side of
            the path instead of the inner/left-hand side.
          * indices (default: False) may be set to True in order to yield indices instead of labels.
        '''
        #TODO
        raise NotImplementedError('Path.boundary_vertices is not yet implemented')
def is_path(p):
    '''
    is_path(p) yields True if p is a Path object and False otherwise.
    '''
    return isinstance(p, Path)

@pimms.immutable
class PathTrace(ObjectWithMetaData):
    '''
    PathTrace is a pimms immutable class that tracks curves or loops drawn on the cortical surface;
    a path trace is distinct from a path in that a trace specifies a path on a particular map
    projection without specifying the actual points along a topology. A path, on the other hand,
    reifies a path trace by finding all the intersections of the lines of the trace with the the
    edges of the native mesh of the topology.
    
    A path trace object stores a map projection and a set of points on the map projection that
    specify the path.

    To create a PathTrace object, it is generally encouraged that one use the path_trace function
    rather than the PathTrace() constructor.
    '''
    def __init__(self, map_projection, pts, closed=False, meta_data=None):
        ObjectWithMetaData.__init__(self, meta_data)
        self.map_projection = map_projection
        self.points         = pts
        self.closed         = closed
    @pimms.option(None)
    def map_projection(mp):
        mp = to_map_projection(mp).persist()
        if mp.radius is None: mp = mp.copy(radius=np.pi/2)
        return mp
    @pimms.param
    def points(x):
        '''
        trace.points is a either the coordinate matrix of the traced points represented by the given
        trace object or is the curve-spline object that represents the given path trace.
        '''
        if isinstance(x, CurveSpline): return x.persist()
        x = np.array(x)
        if not pimms.is_matrix(x, 'number'): return ValueError('trace points must be a matrix')
        if x.shape[0] != 2: x = x.T
        if x.shape[0] != 2: raise ValueError('trace points must be 2D')
        if x.flags['WRITEABLE']: x = pimms.imm_array(x)
        return x
    @pimms.param
    def closed(c):
        '''
        trace.closed is True if trace is a closed trace and False otherwise.
        '''
        return bool(c)
    @pimms.value
    def curve(points, closed):
        '''
        trace.curve is the curve-spline object that represents the given path-trace.
        '''
        if isinstance(points, CurveSpline):
            if closed == bool(points.periodic): return points.even_out()
            points = points.coordinates
        if closed:
            (x0,xx) = (points[:,0], points[:,-1])
            if not np.isclose(np.linalg.norm(xx - x0), 0):
                points = np.hstack([points, np.reshape(x0, (2,1))])
        return curve_spline(points[0], points[1], order=1, periodic=closed).persist()
    def to_path(self, obj, flatmap=None):
        '''
        trace.to_path(subj) yields a path reified on the given subject's cortical surface; the
          returned value is a Path object.
        trace.to_path(topo) yields a path reified on the given topology/cortex object topo.
        trace.to_path(mesh) yields a path reified on the given spherical mesh.

        The optional argument flatmap (default: None) may be given if a flatmap has already been made
        with the given path-trace's map_projection; this is recommended only if you know what you are
        doing and need to save computational resources.
        '''
        # make a flat-map of whatever we've been given...
        if   flatmap is not None: fmap = flatmap
        elif isinstance(obj, Mesh) and obj.coordinates.shape[0] == 2: fmap = obj
        else: fmap = self.map_projection(obj)
        crv = self.curve
        cids = fmap.container(crv.coordinates)
        pts = crv.coordinates.T
        if self.closed and not np.array_equal(pts[0], pts[-1]):
            pts = np.concatenate([pts, [pts[0]]])
        allpts = []
        for ii in range(len(pts) - 1):
            allpts.append([pts[ii]])
            seg  = pts[[ii,ii+1]]
            ipts = segment_intersection_2D(seg, fmap.edge_coordinates)
            ipts = np.transpose(ipts)
            ipts = ipts[np.isfinite(ipts[:,0])]
            # sort these by distance along the vector...
            dists = np.dot(ipts, seg[1] - seg[0])
            allpts.append(ipts[np.argsort(dists)])
        allpts.append([pts[-1]])
        allpts = np.concatenate(allpts)
        idcs = [0]
        for ii in range(1, len(allpts)):
            d = np.sqrt(np.sum((allpts[idcs[-1]] - allpts[ii])**2))
            if np.isclose(d, 0): continue
            idcs.append(ii)
        allpts = allpts[idcs]
        # okay, we have the points--address them and make a path
        addrs = fmap.address(allpts)
        return Path(obj, addrs, meta_data={'source_trace': self})
    def save(self, filename):
        '''
        path_trace.save(filename) saves the given path_trace object to the given filename. If
          filename is a stream object, writes path_trace to the stream. This function uses a json
          format for the path_trace; it will fail if the path trace includes any data that cannot
          be rendered as json, e.g. in the path trace's meta-data.
        '''
        # okay, we need to make a json-like structure of ourselves then turn it into a string;
        # note that we cannot save the mesh, so it is always left off
        dat = self.normalize()
        txt = json.dumps(dat)
        # if it's a filename, we'll need to open it then close it later
        if pimms.is_str(filename):
            filename = os.path.expandvars(os.path.expanduser(filename))
            gz = (len(filename) > 3 and filename[-3:] == '.gz')
            with (gzip.open(filename, 'wt') if gz else open(filename, 'wt')) as f: f.write(txt)
        else: filename.write(txt)
        return filename
    @classmethod
    def denormalize(self, dat):
        '''
        PathTrace.denormalize(data) yields a path trace object that is equivalent to the given 
        normalized data. The data should generally come from a path_trace.normalize() call or from
        a loaded json path-trace file.
        '''
        for k in ['map_projection', 'closed', 'points']:
            if k not in dat: raise ValueError('Missing field from path_trace data: %s' % k)
        return PathTrace(dat['map_projection'], dat['points'],
                         closed=dat['closed'], meta_data=dat.get('meta_data'))
    @staticmethod
    def load(filename):
        '''
        PathTrace.load(filename) loads a path trace object from the given filename and returns it.
        '''
        if pimms.is_str(filename):
            filename = os.path.expandvars(os.path.expanduser(filename))
            if not os.path.isfile(filename):
                raise ValueError('Given filename (%s) is not a file!' % filename)
            gz = (len(filename) > 3 and filename[-3:] == '.gz')
            dat = None
            with (gzip.open(filename, 'rt') if gz else open(filename, 'rt')) as f:
                dat = json.load(f)
            fname = filename
        else:
            dat = json.load(filename)
            fname = '<file-stream>'
        dat = {k.lower():denormalize(v) for (k,v) in six.iteritems(dat)}
        return PathTrace.denormalize(dat)
def is_path_trace(pt):
    '''
    is_path_trace(p) yields True if p is a PathTrace object and False otherwise.
    '''
    return isinstance(pt, PathTrace)
def path_trace(map_projection, pts, closed=False, meta_data=None):
    '''
    path_trace(proj, points) yields a path-trace object that represents the given path of points on
      the given map projection proj.
    
    The following options may be given:
      * closed (default: False) specifies whether the points form a closed loop. If they do form
        such a loop, the points should be given in the same ordering (counter-clockwise or
        clockwise) that mesh vertices are given in; usually counter-clockwise.
      * meta_data (default: None) specifies an optional additional meta-data map to append to the
        object.
    '''
    return PathTrace(map_projection, pts, closed=closed, meta_data=meta_data)
def close_path_traces(*args):
    '''
    close_path_traces(pt1, pt2...) yields the path-trace formed by joining the list of path traces
      at their intersection points in the order given. Note that the direction in which each
      individual path trace's coordinates are specified is ultimately ignored by this function--the
      only ordering that matters is the order in which the list of paths is given.

    Each path argument may alternately be a curve-spline object or coordinate matrix, so long as all
    paths and curves track the same 2D space.
    '''
    pts = [x for x in args if is_path_trace(x)]
    if len(pts) == 0:
        if len(args) == 1 and hasattr(args[0], '__iter__'): return close_path_traces(*args[0])
        raise ValueError('at least one argument to close_path_traces must be a path trace')
    mp0 = pts[0].map_projection
    if not all(mp is None or mp.normalize() == mp0.normalize()
               for x in pts[1:] for mp in [x.map_projection]):
        warnings.warn('path traces do not share a map projection')
    crvs = [x.curve if is_path_trace(x) else to_curve_spline(x) for x in args]
    loop = close_curves(*crvs)
    return path_trace(mp0, loop.coordinates, closed=True)

def isolines(obj, prop, val,
             outliers=None,  data_range=None,  clipped=np.inf,
             weights=None,   weight_min=0,     weight_transform=Ellipsis,
             mask=None,      valid_range=None, transform=None,
             smooth=False,   yield_addresses=False):
    '''
    isolines(mesh, prop, val) yields a 2 x D x N array of points that represent the lines of
      equality of the given property to the given value, according to linear interpolation
      of the property values over the mesh; D represents the dimensions of the mesh (2D or
      3D) and N is the number of faces whose prop values surround val.
    isolines(cortex, prop, val) yields the lines as addresses instead of coordinates.
    isolines(subject, prop, val) yields a lazy map whose keys are the keys of subject.hemis
      and whose values are equivalent to isolines(subject.hemis[key], prop, val).
    
    The following optional arguments may be given:
      * yield_addresses (default: False) may be set to True to instruct isolines to return the
        addresses instead of the coordinates, even for meshes; if the first argument is not a
        mesh then this argument is ignored.
      * smooth (default: None) may optionally be True or a value n or a tuple (n,m) that are
        used as arguments to the smooth_lines() function.
      * Almost all options that can be passed to the to_property() function can be passed to
        this function and are in turn passed along to to_property():
          * outliers       * data_range          * clipped        * weights
          * weight_min     * weight_transform    * mask           * valid_range
          * transform
        Weights, if specified, are not used in the isoline calculation; they are only used for
        thresholding.
    '''
    import scipy.sparse as sps
    from neuropythy import (is_subject, is_list, is_tuple, is_topo, is_mesh, is_tess)
    from neuropythy.util import flattest
    if is_subject(obj):
        kw = dict(outliers=outliers,     data_range=data_range,
                  clipped=clipped,       weights=weights,
                  weight_min=weight_min, weight_transform=weight_transform,
                  mask=mask,             valid_range=valid_range,
                  transform=transform,   smooth=smooth,
                  yield_addresses=yield_addresses)
        return pimms.lazy_map({h:curry(lambda h: isolines(mesh.hemis[h],prop,val,**kw),h)
                               for h in six.iterkeys(mesh.hemis)})
    elif not (is_topo(obj) or is_mesh(obj)):
        raise ValueError('argument must be a mesh, topology, or subject')
    if   smooth is True:          smooth = ()
    elif smooth in [False,None]:  smooth = None
    elif pimms.is_vector(smooth): smooth = tuple(smooth)
    else: raise ValueError('unrecognized smooth argument')
    # find the addresses over the faces:
    fs = obj.tess.indexed_faces if not is_tess(obj) else obj.indexed_faces
    N  = obj.vertex_count
    p  = obj.property(prop,
                      outliers=outliers,     data_range=data_range,
                      clipped=clipped,       weights=weights,
                      weight_min=weight_min, weight_transform=weight_transform,
                      mask=mask,             valid_range=valid_range,
                      transform=transform)
    ii  = np.isfinite(p)
    fii = np.where(np.sum([ii[f] for f in fs], axis=0) == 3)[0]
    fs  = fs[:,fii]
    fp  = np.asarray([p[f] for f in fs])
    lt  = (fp <= val)
    slt = np.sum(lt, 0)
    (ii1,ii2) = [(slt == k) for k in (1,2)]
    wmtx = sps.lil_matrix((N, N), dtype='float')
    emtx = sps.dok_matrix((N*N, N*N), dtype='bool')
    for (ii,fl) in zip([ii1,ii2], [True,False]):
        tmp = np.array(lt.T)
        tmp[~ii,:] = False
        if fl:
            w = fs.T[tmp]
            tmp[ii] = ~tmp[ii]
            (u,v) = np.reshape(fs.T[tmp], (-1,2)).T
        else:
            (u,v) = np.reshape(fs.T[tmp], (-1,2)).T
            tmp[ii] = ~tmp[ii]
            w = fs.T[tmp]
        (pu,pv,pw) = [p[q] for q in (u,v,w)]
        # the line is from somewhere on (u,w) to somewhere on (v,w)
        (wu, wv) = [(val - pw) / (px - pw) for px in (pu,pv)]
        # put these in the weight matrix
        wmtx[u,w] = wu
        wmtx[v,w] = wv
        wmtx[w,u] = (1 - wu)
        wmtx[w,v] = (1 - wv)
        # and in the edge matrix
        e1 = np.sort([u,w],axis=0).T.dot([N,1])
        e2 = np.sort([v,w],axis=0).T.dot([N,1])
        emtx[e1, e2] = True
        emtx[e2, e1] = True
    # okay, now convert these into sequential lines...
    wmtx = wmtx.tocsr()
    (rs,cs,xs) = sps.find(emtx)
    # we need to compress emtx
    ((rs,ridx),(cs,cidx)) = [np.unique(x, return_inverse=True) for x in (rs,cs)]
    assert(len(ridx) == len(cidx) and len(cidx) == len(xs) and np.array_equal(rs,cs))
    em = sps.csr_matrix((xs, (ridx,cidx)), dtype='bool')
    ecounts = flattest(em.sum(axis=1))
    ccs = sps.csgraph.connected_components(em, directed=False)[1]
    lines = {}
    loops = {}
    for (lbl,st) in zip(*np.unique(ccs, return_index=True)):
        if lbl < 0: continue
        o0 = sps.csgraph.depth_first_order(em, st, directed=False)[0]
        o = rs[o0]
        (u,v) = (np.floor(o/N).astype('int'), np.mod(o, N).astype('int'))
        w = flattest(wmtx[u,v])
        if ecounts[o0[-1]] == 2:
            loops[lbl] = (u,v,w)
        else:
            if ecounts[st] != 1:
                o0 = sps.csgraph.depth_first_order(em, o0[-1], directed=False)[0]
                o = rs[o0]
                (u,v) = (np.floor(o/N).astype('int'), np.mod(o, N).astype('int'))
                w = flattest(wmtx[u,v])
            lines[lbl] = (u,v,w)
    # okay, we add all of these to the addrs that we return; it's potentially
    # more useful to have each face's full crossing in its own BC coordinates
    # at times, and the duplicates can be eliminated with just a slice
    addrs = []
    for (lbl,(us,vs,ws)) in six.iteritems(lines):
        (u0s,v0s,u1s,v1s) = (us[:-1],vs[:-1],us[1:],vs[1:])
        qs = [np.setdiff1d([u1,v1], [u0,v0])[0] for (u0,v0,u1,v1) in zip(u0s,v0s,u1s,v1s)]
        fs = np.asarray([u0s, qs, v0s], dtype=np.int)
        fend = (u1s[-1], np.setdiff1d(fs[:,-1], (u1s[-1],v1s[-1])), v1s[-1])
        fs = np.hstack([fs, np.reshape(fend, (3,1))])
        # convert faces back to labels
        fs = np.asarray([obj.labels[f] for f in fs])
        # make the weights
        ws = np.asarray([ws, 0*ws])
        addrs.append({'faces': fs, 'coordinates': ws})
    addrs = list(sorted(addrs, key=lambda a:a['faces'].shape[0]))
    # if obj is a topology or addresses were requested, return them now
    if yield_addresses or not is_mesh(obj): return addrs
    # otherwise, we now convert these into coordinates
    xs = [obj.unaddress(addr) for addr in addrs]
    if smooth is not None: xs = smooth_lines(xs, *smooth)
    return xs
def smooth_lines(lns, n=1, inertia=0.5):
    '''
    smooth_lines(matrix) runs one smoothing iteration on the lines implied by the {2,3}xN matrix.
    smooth_lines(collection) iteratively descends lists, sets, and the values of maps, converting
      all valid matrices that it finds; note that if it encounters anything that cannot be
      interpreted as a line matrix (such as a string), an error is raised.
    
    Smoothing is performed by adding to the coordinates of each vertex with two neighbors half of
    the coordinates of each of its neighbors then dividing that sum by 2; i.e.:
      x_new[i] == x_old[i]/2 + x_old[i-1]/4 + x_old[i+1]/4.

    The following options may be given:
      * n (default: 1) indicates the number of times to perform the above smoothing operation.
      * inertia (default: 0.5) specifies the fraction of the new value that is derived from the
        old value of a node as opposed to the old values of its neighbors. If intertia is m, then
        x_new[i] == m*x_old[i] + (1-m)/2*(x_old[i-1] + x_old[i+1]).
        A higher intertia will smooth the line less each iteration.
    '''
    if pimms.is_lazy_map(lns):
        return pimms.lazy_map({k:curry(lambda k: smooth_lines(lns[k], n=n, inertia=inertia), k)
                               for k in six.iterkeys(lns)})
    elif pimms.is_map(lns):
        m = {k:smooth_lines(v,n=n,inertia=inertia) for (k,v) in six.iteritems(lns)}
        return pyr.pmap(m) if pimms.is_pmap(lns) else m
    elif pimms.is_matrix(lns, 'number'):
        (p,q) = (inertia, 1 - inertia)
        l = lns if lns.shape[0] in (2,3) else lns.T
        if np.allclose(l[:,0], l[:,-1]):
            for ii in range(n):
                lns = p*l + q/2*(np.roll(l, -1, 1) + np.roll(l, 1, 1))
        else:
            l = np.array(l)
            for ii in range(n):
                l[1:-1] = p*l[1:-1] + q/2*(l[:-2] + l[2:])
        if lns.shape != l.shape: l = l.T
        l.setflags(write=lns.flags['WRITEABLE'])
        return l
    elif not pimms.is_scalar(lns):
        u = [smooth_lines(u, n=n, inertia=inertia) for u in lns]
        return (tuple(u)       if is_tuple(lns)                else
                pyr.pvector(u) if isinstance(lns, pyr.PVector) else
                np.asarray(u)  if pimms.is_nparray(lns)        else
                type(lns)(u))
    else: raise ValueError('smooth_lines could not interpret object: %s' % lns)

@importer('path_trace', ('pt.json',         'pt.json.gz',
                         'pathtrace.json',  'pathtrace.json.gz',
                         'path_trace.json', 'path_trace.json.gz',
                         'path-trace.json', 'path-trace.json.gz',))
def load_path_trace(filename, **kw):
    '''
    load_path_trace(filename) yields the path trace indicated by the given file name or stream
      object, filename. The file or stream must be encoded in json.
    '''
    return PathTrace.load(filename, **kw)
@exporter('path_trace', ('pt.json',         'pt.json.gz',
                         'pathtrace.json',  'pathtrace.json.gz',
                         'path_trace.json', 'path_trace.json.gz',
                         'path-trace.json', 'path-trace.json.gz',))
def save_path_trace(filename, pt, **kw):
    '''
    save_path_trace(filename, map_projection) saves the given path trace object to the given file
      or stream object, filename, and returns filename.
    '''
    return pt.save(filename, **kw)
        
####################################################################################################
# Some Functions that deal with converting to/from the above classes

def to_tess(obj):
    '''
    to_tess(obj) yields a Tesselation object that is equivalent to obj; if obj is a tesselation
      object already and no changes are requested (see options) then obj is returned unmolested.

    The following objects can be converted into tesselations:
      * a tesselation object
      * a mesh or topology object (yields their tess objects)
      * a 3 x n or n x 3 matrix of integers (the faces)
      * a tuple of coordinates and faces that can be passed to to_mesh
    '''
    if   is_tess(obj): return obj
    elif is_mesh(obj): return obj.tess
    elif is_topo(obj): return obj.tess
    else:
        # couple things to try: (1) might specify a tess face matrix, (2) might be a mesh-like obj
        try:    return tess(obj)
        except Exception: pass
        try:    return to_mesh(obj).tess
        except Exception: pass
    raise ValueError('Could not convert argument to tesselation object')
def to_mesh(obj):
    '''
    to_mesh(obj) yields a Mesh object that is equivalent to obj or identical to obj if obj is itself
      a mesh object.

    The following objects can be converted into meshes:
      * a mesh object
      * a tuple (coords, faces) where coords is a coordinate matrix and faces is a matrix of
        coordinate indices that make-up the triangles
      * a tuple (faces, coords) where faces is a triangle matrix and coords is a coordinate matrix;
        note that if neither matrix is of integer type, then the latter ordering (which is the same
        as that accepted by the mesh() function) is assumed.
      * a tuple (topo, regname) specifying the registration name to use (note that regname may
        optionally start with 'reg:' which is ignored).
      * a tuple (cortex, surfname) specifying the surface name to use. Note that surfname may
        optionally start with 'surf:' or 'reg:', both of which are used only to determine whether
        to lookup a registration or a surface. If no 'surf:' or 'reg:' is given as a prefix, then
        a surface is tried first followed by a registration. The surface name 'sphere' is
        automatically translated to 'reg:native' and any surface name of the form '<name>_sphere' is
        automatically translated to 'reg:<name>'.
      * a tuple (topo/cortex, mesh) results in the mesh being returned.
      * a tuple (mesh, string) or (mesh, None) results in mesh with the second argument ignored.
      * a tuple (mesh1, mesh2) results in mesh2 with mesh1 ignored.

    Note that some of the behavior described above is desirable because of a common use case of the
    to_mesh function. When another function f accepts as arguments both a hemi/topology object as
    well as an optional surface argument, the purpose is often to obtain a specific mesh from the
    topology but to allow the user to specify which or to pass their own mesh.
    '''
    if   is_mesh(obj): return obj
    elif pimms.is_vector(obj) and len(obj) == 2:
        (a,b) = obj
        if   pimms.is_matrix(a, 'int') and pimms.is_matrix(b, 'real'): return mesh(a, b)
        elif pimms.is_matrix(b, 'int') and pimms.is_matrix(a, 'real'): return mesh(b, a)
        elif is_mesh(a) and (b is None or pimms.is_str(b)): return a
        elif is_mesh(a) and is_mesh(b): return b
        elif is_topo(a):
            from neuropythy import is_cortex
            if   is_mesh(b):          return b
            elif not pimms.is_str(b): raise ValueError('to_mesh: non-str surf/reg name: %s' % (b,))
            (b0, lb) = (b, b.lower())
            # check for translations of the name first:
            s = b[4:] if lb.startswith('reg:') else b[5:] if lb.startswith('surf:') else b
            ls = s.lower()
            if ls.endswith('_sphere'): b = ('reg:' + s[:-7])
            elif ls == 'sphere': b = 'reg:native'
            lb = b.lower()
            # we try surfaces first (if a is a cortex and has surfaces)
            if is_cortex(a) and not lb.startswith('reg:'):
                (s,ls) = (b[5:],lb[5:]) if lb.startswith('surf:') else (b,lb)
                if   s  in a.surfaces: return a.surfaces[s]
                elif ls in a.surfaces: return a.surfaces[ls]
            # then check registrations
            if not lb.startswith('surf:'):
                (s,ls) = (b[4:],lb[4:]) if lb.startswith('reg:') else (b,lb)
                if   s  in a.registrations: return a.registrations[s]
                elif ls in a.registrations: return a.registrations[ls]
            # nothing found
            raise ValueError('to_mesh: mesh named "%s" not found in topology %s' % (b0, a))
        else: raise ValueError('to_mesh: could not deduce meaning of row: %s' % (obj,))
    else: raise ValueError('Could not deduce how object can be convertex into a mesh')

# The Gifti importer goes here because it relies on Mesh
@importer('gifti', ('gii', 'gii.gz'))
def load_gifti(filename, to='auto'):
    '''
    load_gifti(filename) yields the nibabel gifti data structure loaded by nibabel from the given
      filename. Currently, this load method is not particlarly sophisticated and simply returns this
      data.
    
    The optional argument to may be used to coerce the resulting data to a particular format; the
    following arguments are understood:
      * 'auto' currently returns the nibabel data structure.
      * 'mesh' returns the data as a mesh, assuming that there are two darray elements stored in the
        gifti file, the first of which must be a coordinate matrix and a triangle topology.
      * 'coordinates' returns the data as a coordinate matrix.
      * 'tesselation' returns the data as a tesselation object.
      * 'raw' returns the entire gifti image object (None will also yield this result).
    '''
    dat = nib.load(filename)
    to = 'raw' if to is None else to.lower()
    if to in ['raw', 'image', 'gifti', 'all', 'full']:
        return dat
    if to in ['auto', 'automatic']:
        # is this a mesh gifti?
        pset = dat.get_arrays_from_intent('pointset')
        tris = dat.get_arrays_from_intent('triangle')
        if len(pset) == 1 and len(tris) == 1:
            (cor, tri) = (pset[0].data, tris[0].data)
            # okay, try making it:
            try: return Mesh(tri, cor)
            except Exception: pass
        elif len(pset) == 1 and len(tris) == 0:
            # just a pointset
            return pset[0].data
        elif len(tris) == 1 and len(pset) == 0:
            # Just a topology...
            return Tesselation(tris[0].data)
        # Maybe it's a stat? If so, we want to return the data array...
        # see the nifti1 header for these numbers, but stats are intent 2-24
        stats = [v for k in range(2,25) for v in dat.get_arrays_from_intent(k)]
        if len(stats) == 1: return np.squeeze(stats[0].data)
        # most other possibilities are also basic arrays, so if there's only one of them, we can
        # just yield that array
        if len(dat.darrays) == 1: return np.squeeze(dat.darrays[0].data)
        # We don't know what it is; return the whole thing:
        return dat
    elif to in ['coords', 'coordinates', 'xyz']:
        cor = dat.darrays[0].data
        if pimms.is_matrix(cor, np.inexact): return cor
        else: raise ValueError('give gifti file did not contain coordinates')
    elif to in ['tess', 'tesselation', 'triangles', 'tri', 'triangulation']:
        cor = dat.darrays[0].data
        if pimms.is_matrix(cor, 'int'): return Tesselation(cor)
        else: raise ValueError('give gifti file did not contain tesselation')
    elif to in ['mesh']:
        if len(dat.darrays) == 2: (cor,    tri) = dat.darrays
        else:                     (cor, _, tri) = dat.darrays
        cor = cor.data
        tri = tri.data
        # possible that these were given in the wrong order:
        if pimms.is_matrix(tri, np.inexact) and pimms.is_matrix(cor, 'int'):
            (cor,tri) = (tri,cor)
        # okay, try making it:
        return Mesh(tri, cor)
    else: raise ValueError('option "to" given to load_gift could not be understood')

