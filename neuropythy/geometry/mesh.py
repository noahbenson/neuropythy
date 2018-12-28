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
import collections                  as colls
import os, sys, six, types, logging, warnings, gzip, json, pimms

from .util  import (triangle_area, triangle_address, alignment_matrix_3D, rotation_matrix_3D,
                    cartesian_to_barycentric_3D, cartesian_to_barycentric_2D,
                    segment_intersection_2D,
                    barycentric_to_cartesian, point_in_triangle)
from ..util import (ObjectWithMetaData, to_affine, zinv, is_image, is_address, address_data, curry,
                    curve_spline, CurveSpline, chop, zdivide, flattest, inner, config, library_path,
                    dirpath_to_list, to_hemi_str)
from ..io   import (load, importer)
from functools import reduce

if six.PY2: (_tuple_type, _list_type) = (types.TupleType, types.ListType)
else:       (_tuple_type, _list_type) = (tuple, list)

# This function creates the tkr matrix for a volume given the dims
def tkr_vox2ras(img, zooms=None):
    '''
    tkr_vox2ras(img) yields the FreeSurfer tkr VOX2RAS matrix for the given nibabel image object
      img. The img must have a get_shape() method and header member with a get_zooms() method.
    tkr_vox2ras(hdr) operates on a nibabel image header object.
    tkr_vox2ras(shape, zooms) operates on the shape (e.g., for FreeSurfer subjects (256,256,256))
      and the zooms or voxel dimensions (e.g., for FreeSurfer subjects, (1.0, 1.0, 1.0)).
    '''
    if zooms is not None:
        # let's assume that they passed shape, zooms
        shape = img
    else:
        try:    img = img.header
        except: pass
        try:    (shape, zooms) = (img.get_data_shape(), img.get_zooms())
        except: raise ValueError('single argument must be nibabel image or header')
    # Okay, we have shape and zooms...
    zooms = zooms[0:3]
    shape = shape[0:3]
    (dC, dR, dS) = zooms
    (nC, nR, nS) = 0.5 * (np.asarray(shape) * zooms)
    return np.asarray([[-dC,   0,   0,  nC],
                       [  0,   0,  dS, -nS],
                       [  0, -dR,   0,  nR],
                       [  0,   0,   0,   1]])

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
        elif isinstance(name, colls.Set):
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
                 weight=None,    weight_min=0,       weight_transform=Ellipsis,
                 mask=None,      valid_range=None,   null=np.nan,
                 transform=None, yield_weight=False):
        '''
        vset.property(prop) is equivalent to to_property(vset, prop).
        '''
        return to_property(self, prop,
                           dtype=dtype,           null=null,
                           outliers=outliers,     data_range=data_range,
                           clipped=clipped,       weight=weight,
                           weight_min=weight_min, weight_transform=weight_transform,
                           mask=mask,             valid_range=valid_range,
                           transform=transform,   yield_weight=yield_weight)

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
        
def to_mask(obj, m, indices=False):
    '''
    to_mask(obj, m) yields the set of indices from the given vertex-set or itable object obj that
      correspond to the given mask m.
    
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
    if isinstance(obj, VertexSet):
        lbls = obj.labels
        idcs = obj.indices
        obj = obj.properties
    else:
        obj = pimms.itable(obj)
        lbls = np.arange(0, obj.row_count, 1, dtype=np.int)
        idcs = lbls
    if m is None: return idcs if indices else lbls
    if isinstance(m, tuple):
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
                weight=None,    weight_min=0,       weight_transform=Ellipsis,
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
    if pimms.is_vector(obj) and len(obj) == 2:
        return to_property(obj[0], obj[1],
                           dtype=dtype,           null=null,
                           outliers=outliers,     data_range=data_range,
                           clipped=clipped,       weight=weight,
                           weight_min=weight_min, weight_transform=weight_transform,
                           mask=mask,             valid_range=valid_range,
                           transform=transform,   yield_weight=yield_weight)
    # we could have been given a property alone or a map/vertex-set and a property
    if prop is None: (prop, obj) = (obj, None)
    # if it's a vertex-set, we want to note that and get the map
    if isinstance(obj, VertexSet): (vset, obj) = (obj,  obj.properties)
    elif pimms.is_map(obj):        (vset, obj) = (None, obj)
    elif obj is None:              (vset, obj) = (None, None)
    else: ValueError('Data object given to to_properties() is neither a vertex-set nor a mapping')
    # Now, get the property array, as an array
    if pimms.is_str(prop):
        if obj is None: raise ValueError('a property name but no data object given to to_property')
        else: prop = obj[prop]
    if isinstance(prop, colls.Set):
        def _lazy_prop(kk):
            return lambda:to_property(obj, kk,
                                      dtype=dtype,           null=null,
                                      outliers=outliers,     data_range=data_range,
                                      clipped=clipped,       weight=weight,
                                      weight_min=weight_min, weight_transform=weight_transform,
                                      mask=mask,             valid_range=valid_range,
                                      transform=transform,   yield_weight=yield_weight)
        return pimms.itable({k:_lazy_prop(k) for k in prop})
    elif (pimms.is_matrix(prop) or
          (pimms.is_vector(prop) and all(pimms.is_str(p) or pimms.is_vector(p) for p in prop))):
        return np.asarray([to_property(obj, k,
                                       dtype=dtype,           null=null,
                                       outliers=outliers,     data_range=data_range,
                                       clipped=clipped,       weight=weight,
                                       weight_min=weight_min, weight_transform=weight_transform,
                                       mask=mask,             valid_range=valid_range,
                                       transform=transform,   yield_weight=yield_weight)
                           for k in prop])
    elif not pimms.is_vector(prop):
        raise ValueError('prop must be a property name or a vector or a combination of these')
    if dtype is Ellipsis:  dtype = np.asarray(prop).dtype
    if not np.isnan(null): prop  = np.asarray([np.nan if x == null else x for x in prop])
    prop = np.asarray(prop, dtype=dtype)
    # Next, do the same for weight:
    if pimms.is_str(weight):
        if obj is None: raise ValueError('a weight name but no data object given to to_property')
        else: weight = obj[weight]
    weight_orig = weight
    if weight is None or weight_min is None:
        low_weight = []
    else:
        if weight_transform is Ellipsis:
            weight = np.array(weight, dtype=np.float)
            weight[weight < 0] = 0
            weight[np.isclose(weight, 0)] = 0
        elif weight_transform is not None:
            weight = weight_transform(np.asarray(weight))
        if not pimms.is_vector(weight, 'real'):
            raise ValueError('weight must be a real-valued vector or property name for such')
        low_weight = [] if weight_min is None else np.where(weight <= weight_min)[0]
    # Next, find the mask; these are values that can be included theoretically;
    all_vertices = np.asarray(range(len(prop)), dtype=np.int)
    where_nan = np.where(np.isnan(prop))[0]
    where_inf = np.where(np.isinf(prop))[0]
    where_ok  = reduce(np.setdiff1d, [all_vertices, where_nan, where_inf])
    # look at the valid_range...
    where_inv = [] if valid_range is None else \
                where_ok[(prop[where_ok] < valid_range[0]) | (prop[where_ok] > valid_range[1])]
    # Whittle down the mask to what we are sure is in the spec:
    where_nan = np.union1d(where_nan, where_inv)
    # make sure we interpret mask correctly...
    mask = to_mask(obj, mask, indices=True)
    mask = np.setdiff1d(all_vertices if mask is None else all_vertices[mask], where_nan)
    # Find the outliers: values specified as outliers or inf values; will build this as we go
    outliers = [] if outliers is None else all_vertices[outliers]
    outliers = np.intersect1d(outliers, mask) # outliers not in the mask don't matter anyway
    outliers = np.union1d(outliers, low_weight) # low-weight vertices are treated as outliers
    # If there's a data range argument, deal with how it affects outliers
    if data_range is not None:
        if hasattr(data_range, '__iter__'):
            outliers = np.union1d(outliers, mask[np.where(prop[mask] < data_range[0])[0]])
            outliers = np.union1d(outliers, mask[np.where(prop[mask] > data_range[1])[0]])
        else:
            outliers = np.union1d(outliers, mask[np.where(prop[mask] < 0)[0]])
            outliers = np.union1d(outliers, mask[np.where(prop[mask] > data_range)[0]])
    # no matter what, trim out the infinite values (even if inf was in the data range)
    outliers = np.union1d(outliers, mask[np.where(np.isinf(prop[mask]))[0]])
    # Okay, mark everything in the prop:
    unmask = np.setdiff1d(all_vertices, mask)
    where_nan = np.asarray(np.union1d(where_nan,unmask), dtype=np.int)
    outliers = np.asarray(outliers, dtype=np.int)
    if len(where_nan) + len(outliers) > 0:
        prop = np.array(prop)
        prop[where_nan] = null
        prop[outliers]  = clipped
    if yield_weight:
        weight = np.array(weight, dtype=np.float)
        weight[where_nan] = 0
        weight[outliers] = 0
    # transform?
    if transform: prop = transform(prop)
    # That's it, just return
    return (prop, weight) if yield_weight else prop
    

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
    
    def __repr__(self):
            return "TesselationIndex(<%d vertices>)" % len(self.vertex_index)
    def __getitem__(self, index):
        if isinstance(index, tuple):
            if   len(index) == 3: return self.face_index.get(index, None)
            elif len(index) == 2: return self.edge_index.get(index, None)
            elif len(index) == 1: return self.vertex_index.get(index[0], None)
            else:                 raise ValueError('Unrecognized tesselation item: %s' % index)
        elif isinstance(index, colls.Set):
            return {k:self[k] for k in index}
        elif pimms.is_vector(index):
            vi = self.vertex_index
            return np.asarray([vi[k] if k >= 0 else k for k in index])
        elif pimms.is_matrix(index):
            m = np.asarray(index)
            if m.shape[0] != 2 and m.shape[0] != 3: m = m.T
            idx = self.edge_index if m.shape[0] == 2 else self.face_index
            return pimms.imm_array([idx[k] for k in zip(*m)])
        else:
            return self.vertex_index[index]
    def __call__(self, index):
        vi = self.vertex_index
        if isinstance(index, tuple):
            return tuple([vi[k] if k >= 0 else k for k in index])
        elif isinstance(index, colls.Set):
            return set([vi[k] if k >= 0 else k for k in index])
        elif pimms.is_vector(index):
            return np.asarray([vi[k] if k >= 0 else k for k in index], dtype=np.int)
        elif pimms.is_matrix(index):
            return np.asarray([[vi[k] if k >= 0 else k for k in u] for u in index], dtype=np.int)
        else:
            return vi[index]

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
        edge2face = {}
        idx = {}
        edge_list = [None for i in range(3*faces.size)]
        k = 0
        rng = range(faces.shape[1])
        for (e,i) in zip(
            zip(np.concatenate((faces[0], faces[1], faces[2])),
                np.concatenate((faces[1], faces[2], faces[0]))),
            np.concatenate((rng, rng, rng))):
            e = tuple(sorted(e))
            if e in idx:
                edge2face[e].append(i)
            else:
                idx[e] = k
                idx[e[::-1]] = k
                edge_list[k] = e
                edge2face[e] = [i]
                k += 1
        edge2face = {k:tuple(v) for (k,v) in six.iteritems(edge2face)}
        for ((a,b),v) in six.iteritems(pyr.pmap(edge2face)):
            edge2face[(b,a)] = v
        return pyr.m(edges=pimms.imm_array(np.transpose(edge_list[0:k])),
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
            except: raise ValueError('mesh.tess must be a Tesselation object')
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
        try:    return space.cKDTree(face_centers.T)
        except: return space.KDTree(face_centers.T)
    @pimms.value
    def vertex_hash(coordinates):
        '''
        mesh.vertex_hash yields the scipy spatial hash of the vertices of the given mesh.
        '''
        try:    return space.cKDTree(coordinates.T)
        except: return space.KDTree(coordinates.T)

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
        try:    (d,near) = self.facee_hash.query(x, k=k, n_jobs=n_jobs)
        except: (d,near) = self.facee_hash.query(x, k=k)
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
        try:    (d,near) = self.vertex_hash.query(pt, k=1, n_jobs=n_jobs)
        except: (d,near) = self.vertex_hash.query(pt, k=1)
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
        try:    (d, near) = self.face_hash.query(pt, k=k, n_jobs=n_jobs)
        except: (d, near) = self.face_hash.query(pt, k=k)
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
        try:    (_, nei) = self.vertex_hash.query(x, k=1, n_jobs=n_jobs)
        except: (_, nei) = self.vertex_hash.query(x, k=1)
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
        pt = np.asarray(pt, dtype=np.float32)
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
                    try:    near = self.face_hash.query(sub_pts, k=cur_k, n_jobs=n_jobs)[1]
                    except: near = self.face_hash.query(sub_pts, k=cur_k)[1]
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
            (fs, xs) = address_data(coords, dims=2)
            xs = np.vstack([xs, [1 - np.sum(xs, axis=0)]])
            nv = fs[(np.argmax(xs, axis=0), np.arange(fs.shape[1]))]
            nv = self.tess.index(nv)
        else:
            coords = np.asarray(coords)
            if coords.shape[0] == self.coordinates.shape[0]: coords = coords.T
            nv = self.nearest_vertex(coords, n_jobs=n_jobs)
        n = self.vertex_count
        m = len(nv)
        return sps.csr_matrix((np.ones(m, dtype=np.int), (np.arange(m), nv)),
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
            (fs, xs) = address_data(coords, dims=2)
            xs = np.vstack([xs, [1 - np.sum(xs, axis=0)]])
            (n, m) = (xs.shape[1], self.vertex_count)
            fs = self.tess.index(fs)
            return sps.csr_matrix(
                (xs.flatten(), (np.tile(np.arange(n), 3), fs.flatten())),
                shape=(n, m))
        else: return self.linear_interpolation(self.address(coords, n_jobs=n_jobs))
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
            if data.shape[0] != n: data = data.T
            if not pimms.is_matrix(data, 'number'):
                return np.asarray([self.apply_interpolation(interp, d) for d in data.T])
            else: return inner(interp, data.T).T
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
            currently supported methods are 'automatic', 'linear', or 'nearest'. The 'nearest'
            method does not actually perform a nearest-neighbor interpolation but rather assigns to
            a destination vertex the value of the source vertex whose voronoi-like polygon contains
            the destination vertex; note that the term 'voronoi-like' is used here because it uses
            the Voronoi diagram that corresponds to the triangle mesh and not the true delaunay
            triangulation. The 'linear' method uses linear interpolation; though if the given data
            is non-numerical, then nearest interpolation is used instead. The 'automatic' method
            uses linear interpolation for any floating-point data and nearest interpolation for any
            integral or non-numeric data. Note that nearest-neighbor interpolation is used for
            non-numeric data arrays no matter what the method argument is.
          * n_jobs (default: -1) is passed along to the cKDTree.query method, so may be set to an
            integer to specify how many processors to use, or may be -1 to specify all processors.
        '''
        n = self.vertex_count
        if isinstance(x, Mesh): x = x.coordinates
        # no matter what the input we want to calculate the interpolation matrices but once:
        interps = pimms.lazy_map(
            {'nearest': lambda:self.interpolation_matrix(x,
                                                         n_jobs=n_jobs, method='nearest',
                                                         mask=mask, weights=weights),
             'linear':  lambda:self.interpolation_matrix(x,
                                                         n_jobs=n_jobs, method='linear',
                                                         mask=mask, weights=weights)})
        if pimms.is_str(method): method = method.lower()
        if method in [None, Ellipsis, 'auto', 'automatic']: method = None
        elif method in ['lin', 'linear', 'trilinear']: method = 'linear'
        elif method in ['nn', 'nearest', 'near', 'nearest-neighbor', 'nearest_neighbor']:
            method = 'nearest'
        else: raise ValueError('cannot interpret method: %s' % method)
        # we now need to look over data...
        def _apply_interp(dat):
            if pimms.is_str(dat):
                return _apply_interp(self.properties[dat])
            elif pimms.is_array(dat, np.inexact, (1,2)) and method != 'nearest':
                return self.apply_interpolation(interps['linear'], dat)
            elif pimms.is_array(dat, 'int', (1,2)) and method == 'linear':
                return self.apply_interpolation(interps['linear'], dat)
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
            try:    face_id = self.container(data, n_jobs=n_jobs)
            except: face_id = self.container(data)
            if face_id is None:
                return {'faces':np.array([-1,-1,-1]), 'coordinates':np.full(2,np.nan)}
            tx = coords[:, idxfs[:,face_id]].T
            faces = self.tess.faces[:,face_id]
        else:
            data = data if data.shape[1] == 3 or data.shape[1] == 2 else data.T
            n = data.shape[0]
            try:    face_id = np.asarray(self.container(data), n_jobs=n_jobs)
            except: face_id = np.asarray(self.container(data))
            tx = np.full((3, dims, n), np.nan)
            oks = np.where(np.logical_not(face_id == None))[0]
            okfids = face_id[oks].astype('int')
            tx[:,:,oks] = np.transpose(
                np.reshape(coords[:,idxfs[:,okfids].flatten()], (dims, 3, oks.shape[0])),
                (1,0,2))
            faces = np.full((3, n), -1, dtype=np.int)
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
                   native_to_vertex_matrix=None, weight=None):
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
          * weight (default: None) may optionally provide an image whose voxels are weights to use
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
            # wild guess: the inverse of the tkr_vox2ras matrix without alignment to native
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
        if weight is None: weight = np.ones(image.shape)
        elif pimms.is_str(weight): weight = load(weight).get_data()
        elif isinstance(weight, nib.analyze.SpatialImage): weight = weight.get_data()
        else: weight = np.asarray(weight)
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
        wgts_wgt = np.asarray([weight[tuple(row)] for row in voxs])
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
                       where_nan,
                       np.where(np.isclose(weights, 0))[0]])
        # Find the outliers: values specified as outliers or inf values; we'll build this as we go
        outliers = [] if outliers is None else all_vertices[outliers]
        outliers = np.intersect1d(outliers, mask) # outliers not in the mask don't matter anyway
        # no matter what, trim out the infinite values (even if inf was in the data range)
        outliers = np.union1d(outliers, mask[np.where(np.isinf(prop[mask]))[0]])
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
        e2v = lil_matrix((len(x0), len(el)), dtype=np.int)
        for (i,(u,v)) in enumerate(el):
            e2v[u,i] = 1
            e2v[v,i] = -1
        e2v = csr_matrix(e2v)
        (us, vs) = el.T
        weights_tth = weights[tethered]
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
        return X[1:3]
    @staticmethod
    def orthographic_projection_inverse(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 2 else X.T
        Xnorm = X / sphere_radius
        return np.asarray([sphere_radius * np.sqrt(1.0 - (Xnorm ** 2).sum(0)), X[0], X[1]])
    @staticmethod
    def equirectangular_projection_forward(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 3 else X.T
        X = X / np.sqrt((X ** 2).sum(0))
        return sphere_radius / np.pi * np.asarray([np.arctan2(X[1], X[0]), np.arcsin(X[2])])
    @staticmethod
    def equirectangular_projection_inverse(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 2 else X.T
        X = np.pi / sphere_radius * X
        cos1 = np.cos(X[1])
        return np.asarray([cos1 * np.cos(X[0]) * sphere_radius, 
                           cos1 * np.sin(X[0]) * sphere_radius,
                           np.sin(X[1]) * sphere_radius])
    @staticmethod
    def mercator_projection_forward(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 3 else X.T
        X = X / np.sqrt((X ** 2).sum(0))
        return sphere_radius * np.asarray([np.arctan2(X[1], X[0]),
                                           np.log(np.tan(0.25 * np.pi + 0.5 * np.arcsin(X[2])))])
    @staticmethod
    def mercator_projection_inverse(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 2 else X.T
        X = X / sphere_radius
        return sphere_radius * np.asarray([np.cos(X[0]), np.sin(X[0]),
                                           np.sin(2 * (np.arctan(np.exp(X[1])) - 0.25*np.pi))])
    @staticmethod
    def sinusoidal_projection_forward(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 3 else X.T
        X = X / np.sqrt((X ** 2).sum(0))
        phi = np.arcsin(X[2])
        return sphere_radius / np.pi * np.asarray([np.arctan2(X[1], X[0]) * np.cos(phi), phi])
    @staticmethod
    def sinusoidal_projection_inverse(X, sphere_radius=100.0):
        X = np.asarray(X)
        X = X if X.shape[0] == 2 else X.T
        X = np.pi * X / sphere_radius
        z = np.sin(X[1])
        cosphi = np.cos(X[1])
        return np.asarray([np.cos(X[0] / cosphi) * sphere_radius,
                           np.sin(X[0] / cosphi) * sphere_radius,
                           np.sin(X[1]) * sphere_radius])
    # These are given actual values just below the class definition
    projection_forward_methods = {}
    projection_inverse_methods = {}

    def __init__(self, mesh=None,
                 center=None, center_right=None, radius=None, method='equirectangular',
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

    @staticmethod
    def load(filename,
             center=None, center_right=None, radius=None, method='equirectangular',
             registration='native', chirality=None, sphere_radius=None,
             pre_affine=None, post_affine=None, meta_data=None):
        '''
        MapProjection.load(file) yields the map projection indicated by the given file name or file
          object file. Map projections define the parameters of a projection to the 2D cortical
          surface via a registartion name and projection parameters.

        The following options may be given; all of these options represent parameters that may or
        may not be defined by the map-projection file; in the case that a parameter is defined by
        the file, the optional argument is ignored. To force parameters to have particular values,
        you must modify or copy the map projection that is by this function (but note that all the
        parameters below match parameters of the map projection class. All of these parameters
        except the method (default: 'equirectangular'), registration (default: 'native'), and
        meta_data parameters have default values of None, which means that they are, by default,
        handled by the MapProjection class. The meta_data parameter, if provided, is merged with any
        meta-data in the projection file such that the passed meta-data is overwritten by the file's
        meta-data.
          * center specifies the 3D vector that points toward the center of the map.
          * center_right specifies the 3D vector that points toward any point on the positive x-axis
            of the resulting map.
          * radius specifies the radius that should be assumed by the model in radians of the
            cortical sphere.
          * method specifies the projection method used (default: 'equirectangular').
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
        dat = {k.lower():v for (k,v) in six.iteritems(dat)}
        # check version
        if 'version' not in dat: warnings.warn('projection file contains no version: %s' % fname)
        elif dat['version'] < 1: warnings.warn('projection file version < 1: %s' % fname)
        elif dat['version'] > 1: logging.info('projection file version > 1: %s' % fname)
        # build up the parameter dict we will use:
        params = dict(center=center, center_right=center_right, radius=radius, method=method,
                      registration=registration, chirality=chirality, sphere_radius=sphere_radius,
                      pre_affine=pre_affine, post_affine=post_affine)
        params = {k:(dat[k] if k in dat else v) for (k,v) in six.iteritems(params)}
        # get meta-data if included and/or set it up
        meta = dat['meta_data'] if 'meta_data' in dat else {}
        if not pimms.is_map(meta): raise ValueError('projection meta_data was not a mapping object')
        if meta_data is not None: meta = pimms.merge(meta_data, meta)
        meta = dict(meta)
        if 'version' not in meta: meta['version'] = dat['version'] if 'version' in dat else None
        if 'json' not in meta: meta['json'] = dat
        if 'filename' not in meta and pimms.is_str(filename): meta['filename'] = filename
        # add in meta-data
        params['meta_data'] = meta
        # and use these to create an object...
        return MapProjection(**params)
        
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
        center).
        '''
        if r is None: return None
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
        if ch is None: return None
        if not pimms.is_str(ch):
            raise ValueError('projection chirality must be either None or \'lh\' or \rh\'')
        ch = ch.lower()
        if ch not in ['lh', 'rh']:
            raise ValueError('projection chirality must be either None or \'lh\' or \rh\'')
        return ch
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
        of the projection lies on the positive x-axis and the center_right of the projection lies in
        the x-y plane.
        '''
        mtx = np.eye(4) if pre_affine is None else pre_affine
        cmtx = np.eye(4)
        if center is not None:
            tmp = alignment_matrix_3D(center, [1,0,0])
            cmtx[0:3,0:3] = tmp
            mtx = cmtx.dot(mtx)
        crmtx = np.eye(4)
        if center_right is not None:
            # Tricky: we need to run this coordinate through the center transform then align it with
            # the x-y plane:
            cr = cmtx[0:3,0:3].dot(center_right)
            # what angle do we need to rotate this?
            ang = np.arctan2(cr[2], cr[1])
            crmtx[0:3,0:3] = rotation_matrix_3D([1,0,0], -ang)
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
        ch = 'XH' if chirality is None else chirality.upper()
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
        x = np.clip(x[0] / self._sphere_radius, -1, 1)
        # okay, we want the angle of the vertex [1,0,0] to these points...
        th = np.arccos(x)
        # and we want to know what points are within the angle given by the radius; if the radius
        # is a radian-like quantity, we use th itself; otherwise, we convert it to a distance
        rad = pimms.mag(self.radius, 'radians') if pimms.is_quantity(self.radius) else self.radius
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
        from neuropythy.mri import Subject
        if isinstance(obj, Subject):
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
        from neuropythy.mri import Subject
        if isinstance(obj, Topology) or isinstance(obj, Subject):
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

@importer('map_projection', ('mp.json', 'mp.json.gz', 'map.json', 'map.json.gz',
                             'projection.json', 'projection.json.gz'))
def load_map_projection(filename,
                        center=None, center_right=None, radius=None, method='equirectangular',
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
      lh.motor.projection.json.gz  => map_projections['lh']['motor']
    '''
    p = dirpath_to_list(p)
    return pyr.pmap(
        {h:pimms.lazy_map(
            {parts[1]: curry(lambda flnm,h: load_map_projection(flnm, chirality=h),
                             os.path.join(pp, fl), h)
             for pp    in p
             for fl    in os.listdir(pp)  if fl.endswith('.json') or fl.endswith('.json.gz')
             for parts in [fl.split('.')] if len(parts) > 2 and parts[0] == h})
         for h in ('lh','rh')})
# just the neuropythy lib-dir projections:
try: npythy_map_projections = load_projections_from_path(projections_libdir)
except:
    warnings.warn('Error raised while loading neuropythy libdir map projections')
    npythy_map_projections = pyr.m(lh=pyr.m(), rh=pyr.m())
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
    map_projections = pimms.merge(npythy_map_projections, tmp)
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
def map_projection(name, arg,
                   center=None, center_right=None, radius=None, method='equirectangular',
                   registration='native', sphere_radius=None,
                   pre_affine=None, post_affine=None, meta_data=None):
    '''
    map_projection(name, hemi) yields the map projection with the given name if it exists; hemi must
      be either 'lh' or 'rh'.
    map_projection(name, topo) yields a map projection using the given topology object topo to
      determine the hemisphere and assigning to the resulting projection's 'mesh' parameter the
      appropriate registration from the given topology.
    map_projection(name, mesh) uses the given mesh; the mesh's meta-data must specify the hemisphere
      for this to work.

    All options that can be passed to load_map_projection and MapProjection can be passed to
    map_projection:
      * center specifies the 3D vector that points toward the center of the map.
      * center_right specifies the 3D vector that points toward any point on the positive x-axis
        of the resulting map.
      * radius specifies the radius that should be assumed by the model in radians of the
        cortical sphere.
      * method specifies the projection method used (default: 'equirectangular').
      * registration specifies the registration to which the map is aligned (default: 'native').
      * chirality specifies whether the projection applies to left or right hemispheres.
      * sphere_radius specifies the radius of the sphere that should be assumed by the model.
        Note that in Freesurfer, spheres have a radius of 100.
      * pre_affine specifies the pre-projection affine transform to use on the cortical sphere.
      * post_affine specifies the post-projection affine transform to use on the 2D map.
      * meta_data specifies any additional meta-data to attach to the projection.
    '''
    if not pimms.is_str(name): raise ValueError('map_projection name must be a string')
    if pimms.is_str(arg):
        hemi = to_hemi_str(arg)
        topo = None
        mesh = None
    elif isinstance(arg, Topology):
        hemi = arg.chirality
        topo = arg
        mesh = None
    elif isinstance(arg, Mesh):
        if   'chirality' in arg.meta_data: hemi = to_hemi_str(arg.meta_data['chirality'])
        elif 'hemi'      in arg.meta_data: hemi = to_hemi_str(arg.meta_data['hemi'])
        else: raise ValueError('Could not deduce hemisphere from mesh')
        topo = None
        mesh = arg
    else: raise ValueError('Could not understand map_projection argument: %s' % arg)
    if   name         in map_projections[hemi]: mp = map_projections[hemi][name]
    elif name.lower() in map_projections[hemi]: mp = map_projections[hemi][name.lower()]
    else:
        try: mp = load_map_projection(name, chirality=hemi,
                                      center=center, center_right=center_right, radius=radius,
                                      method=method, registration=registration,
                                      sphere_radius=sphere_radius,
                                      pre_affine=pre_affine, post_affine=post_affine)
        except: raise ValueError('could neither find nor load projection %s (%s)' % (arg, hemi))
    if topo: mesh = mp.extract_mesh(topo)
    # okay, return the projection with the mesh attached
    return mp if mesh is None else mp.copy(mesh=mesh)
def to_flatmap(name, obj, chirality=None,
               center=None, center_right=None, radius=None, method='equirectangular',
               registration='native', sphere_radius=None,
               pre_affine=None, post_affine=None, meta_data=None):
    '''
    to_flatmap(name, topo) yields a flatmap of the given topology topo using the map projection with
      the given name or path.
    to_flatmap(name, mesh, h) yields a flatmap of the given mesh, which is of the given hemisphere;
      if h is not given, then the hemisphere must be in the meta-data of the mesh.
    to_flatmap(name, subj, h) uses the given hemisphere from the given subject.
    '''
    from neuropythy.mri import Subject
    hemi = chirality
    if isinstance(obj, Subject):
        if hemi is None: raise ValueError('hemi is required when subject object is given')
        h = to_hemi_str(hemi) if hemi not in obj.hemis else hemi
        if h not in obj.hemis: raise ValueError('Given hemi not found for give subject')
        obj = obj.hemis[h]
    if isinstance(obj, Topology):
        mp = map_projection(name, obj)
        if mp.mesh is None: raise ValueError('could not match projection to topology')
        else: return mp(mp.mesh)
    elif not isinstance(obj, Mesh): raise ValueError('Could not interpret to_flatmap arg')
    if hemi is not None: obj = obj.with_meta(chirality=to_hemi_str(hemi))
    mp = map_projection(name, obj,
                        center=center, center_right=center_right, radius=radius,
                        method=method, registration=registration,
                        sphere_radius=sphere_radius,
                        pre_affine=pre_affine, post_affine=post_affine)
    if mp.mesh is None: warnings.warn('could not match projection to mesh')
    return mp(obj)

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
        if not isinstance(topo, Topology):
            raise ValueError('Topologies can only be interpolated with other topologies')
        if registration is None:
            reg_names = [k for k in topo.registrations.iterkeys() if k in self.registrations
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
                   center=None, center_right=None, radius=None, method='equirectangular',
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
        (faces, coords) = address_data(addresses)
        coords = np.vstack([coords, [1 - np.sum(coords, axis=0)]])
        n = faces.shape[1]
        (u,v,wu,wv,fs,ps) = ([],[],[],[], [], [])
        pcur = []
        lastf = faces[:, -1] if closed else None
        for (ii,f,w) in zip(range(n), faces.T, coords.T):
            ff = Ellipsis
            zs = np.isclose(w,0)
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
                    rev = len(np.unique(np.concatenate([lastf,f[k]])))
                    if rev == 4: k = list(reversed(k))
                vtx = None if len(u) == 0 else np.setdiff1d(f[k], [u[-1],v[-1]])[0]
                for (q,qq) in zip([u,v],   f[k]): q.append(qq)
                for (q,qq) in zip([wu,wv], w[k]): q.append(qq)
            else: raise ValueError('address contained all-zero weights')
            if ff is Ellipsis: ff = None if lastf is None or vtx is None else (u[-2], v[-2], vtx)
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
            tmp = np.setdiff1d((u[0],v[0]), (u[-1],v[-1]))
            # might be that we started/ended on a point
            if len(tmp) == 0:
                tmp = np.setdiff1d((u[-2],v[-2]), (u[-1],u[-2]))
                fs[0] = (u[-1],v[-1],tmp[0])
                ps[0] = tuple(pcur)
            elif len(tmp) == 1:
                fs[0] = (u[-1], v[-1], tmp[0])
                ps[0] = tuple(pcur)[:-1] + ps[0]
            else: raise ValueError('closed path does not start/end in same face')
        else:
            fs[0] = None
            ps[0] = None
        fs = np.roll(fs, -1, axis=0)
        ps = np.roll(ps, -1, axis=0)
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
        same  = np.union1d(u,v)
        other = np.setdiff1d(tess.labels, same)
        (q,wq) = [np.concatenate([a,b]) for (a,b) in [(u,v),(wu,wv)]]
        m = len(q)
        # for the labels, the u and v have repeats, so we want to average their values
        mm  = sps.csr_matrix((np.ones(m), (q, np.arange(m))), shape=(n, m))
        lbl = zdivide(mm.dot(wq), flattest(mm.sum(axis=1)))
        q   = np.unique(q)
        wq  = lbl[q]
        # we crawl across vertices by edges until we find all of them
        nei  = np.asarray(tess.neighborhoods)
        unk  = np.full(tess.vertex_count, True, dtype=np.bool)
        unk[q] = False
        q = np.unique(u[v != u])
        while len(q) > 0:
            q = np.unique([k for neitup in nei[q] for k in neitup]) # get all their neighbors
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
        fs = np.where(np.sum(np.isin(surface.tess.faces, msk), axis=0) == 3)[0]
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
        res = np.array(res[1:] if not np.isfinite(res[0][0]) else res)
        res.setflags(write=False)
        return res
    class BorderTriangleSplitter(object):
        '''
        The Path.BorderTriangleSplitter class performs the splitting of border triangles in a path
        into inner and outer triangles. It should not be used directly; rather Path objects use it
        as a tool.
        '''
        # these coords are used to reify BC triangles while figuring them out...
        A = pimms.imm_array([0.0,              0.0])
        B = pimms.imm_array([1.0/np.sqrt(2.0), 0.0])
        C = pimms.imm_array([0.0,              1.0/np.sqrt(2.0)])
        def __init__(self, edge_data, addresses, closed):
            self.edges  = edge_data[:2]
            self.faces  = edge_data[4]
            self.pieces = edge_data[5]
            (fs, xs) = address_data(addresses)
            xs = chop(np.vstack([xs, [1.0 - xs[0] - xs[1]]]))
            self.bc_faces = fs.T
            self.bc_coords = xs.T
            self.closed = closed
        @staticmethod
        def angle_order(pts, x0, x1=(1,0)): # order of pts around x0 starting at x1
            '''
            Given a set of points in the (A,B,C) triangle, yield their indices in order of the
            angle about point x0 starting with the point x1.
            '''
            pts = (np.asarray(pts) - x0).T
            x1  = np.asarray(x1) - x0
            rs  = np.sqrt(np.sum(pts**2, axis=0))
            th0 = np.arctan2(x1[1], x1[0])
            ths = np.mod(np.arctan2(pts[1], pts[0]) - th0, 2*np.pi)
            kk  = np.argsort(ths)
            return np.array([k for k in kk
                             if not np.isclose(pts[:,k], 0).all()
                             if not np.isclose(pts[:,k], x1).all()])
        @staticmethod
        def scan_face_points(pts):
            '''
            scan_face_points(points) splits a single face containing all the given points into
            left- and right-side sets of faces and returns the barycentric coordinates for these.
            The given points must be in path order. The points argument must be the barycentric
            weights on vertices A and B of the triangle.
            '''
            # turn all into coords for the calculation
            (A,B,C) = (Path.BorderTriangleSplitter.A,
                       Path.BorderTriangleSplitter.B,
                       Path.BorderTriangleSplitter.C)
            pts = np.asarray([A*x + B*y + C*z for (x,y,z) in pts])
            n = len(pts)
            allpts = np.vstack([pts, (A,B,C)])
            (rfs,lfs) = ([],[]) # left and right side triangles
            cfs = rfs # we always start with the right side
            # we walk across points then back sweeping triangles around as we go...
            path  = np.concatenate([np.arange(n), np.flip(np.arange(n-1))])
            plen  = len(path)
            # figure out what side we enter on...
            (u0,v0) = (0,1) if pts[0,1] == 0 else (2,0) if pts[0,0] == 0 else (1,2)
            prev_p = v0 + n
            skipto = None
            for k in range(plen-1):
                if skipto is not None and k != skipto: continue
                else: skipto = None
                (p,next_p) = path[[k,k+1]]
                pt = allpts[p]
                # order the points clockwise around this point
                ii = Path.BorderTriangleSplitter.angle_order(allpts, pt, allpts[prev_p])
                if p == n-1: # we've reached the edge/turnaround point, and i is the next
                    cfs = lfs
                    prev_p = ii[0]
                    ii = ii[1:]
                # we skip the first--it's the prev_pt
                for i in ii:
                    # for sure, this triad makes a triangle on whatever side we're on...
                    cfs.append((p, prev_p, i))
                    if i != next_p: prev_p = i
                    if i == next_p: pass
                    elif i < n: # another point in the path, but not the next
                        if k+2 < plen and path[k+2] == i:
                            cfs.append((p,i,next_p))
                        elif k+3 < plen and path[k+3] == i:
                            cfs.append((p,i,next_p))
                            cfs.append((next_p,i,path[k+1]))
                        else:
                            warnings.warn('skipping concave segment of curve in single face')
                        skipto = i
                    else: continue # haven't found the next point yet--keep going
                    break # all other conditions, we break
            # we have lfs and rfs now, but they're in embedded triangle coords instead of
            # barycentric coordinates...
            res = []
            for cfs in (lfs,rfs):
                cfs = np.transpose([allpts[p] for p in np.transpose(cfs)], (0,2,1))
                tri = np.array([[np.full(cfs.shape[2], x) for x in xx] for xx in (A,B,C)])
                cfs = np.array([cartesian_to_barycentric_2D(tri, x) for x in cfs])
                res.append(cfs)
            return tuple(res)
        def __call__(self):
            '''
            Scans the entire path to parcellate left and right border triangles, and yields the
            tuple (lhs, rhs) of the left (inner) and right (outer) border triangles of the given
            path. The lhs and rhs are 3-tuples (a,b,c) of the addresses of the three vertices of
            each border triangle (in counter-clockwise ordering). These should not be expected to
            be in a specific order; though they should be grouped by face.
            '''
            bcfs    = self.bc_faces
            bcxs    = self.bc_coords
            fs      = self.faces
            (us,vs) = self.edges
            ps      = self.pieces
            ne      = us.shape[0]
            nf      = fs.shape[0]
            (lfs,rfs,lxs,rxs) = ([],[],[],[])
            for (f,p) in zip(fs,ps):
                if f is None:
                    # last face -- we can just skip basically
                    continue
                # get the barycentric coords
                bxs = np.array(bcxs[list(p)])
                bfs = bcfs[list(p)]
                # fix the barycentric coords if need be
                for (ii,xx,ff) in zip(range(len(bxs)), bxs, bfs):
                    if not np.array_equal(ff,f):
                        bxs[ii] = [xx[ff == fi][0] if fi in ff else 0 for fi in f]
                (lhs,rhs) = [xx.T for xx in Path.BorderTriangleSplitter.scan_face_points(bxs)]
                # lhs and rhs are the barycentric coordinates...
                frow = [f for _ in range(len(lhs) + len(rhs))]
                for (xx,ff,hs) in zip([lxs,rxs],[lfs,rfs],[lhs,rhs]):
                    xx.append(hs)
                    ff.append([f for _ in range(len(hs))])
            (lfs,rfs,lxs,rxs) = [np.vstack(xx).T for xx in (lfs,rfs,lxs,rxs)]
            lfs = pimms.imm_array(lfs)
            rfs = pimms.imm_array(rfs)
            lxs = pimms.imm_array(lxs)
            rxs = pimms.imm_array(rxs)
            return tuple([tuple([pyr.m(coordinates=x, faces=fs) for x in xs])
                          for (xs,fs) in [(lxs,lfs), (rxs,rfs)]])
    @pimms.value
    def all_border_triangle_addresses(edge_data, addresses, closed):
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
        bts = Path.BorderTriangleSplitter(edge_data, addresses, closed)
        return bts()
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
        if isinstance(surface, Topology):
            conv1 = lambda s,xs:pimms.imm_array(surface[s].unaddress(xs).T)
            conv2 = lambda s: tuple([conv1(s,xs) for xs in all_border_triangle_addresses])
            return pimms.lazy_map({k:curry(conv2, k) for k in six.iterkeys(surface.surfaces)})
        else: return tuple([pimms.imm_array(surface.unaddress(xs).T)
                            for xs in all_border_triangle_addresses])
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
        def sarea(srf):
            if pimms.is_str(srf): (srf,btris) = (surface.surfaces[srf],border_triangles[srf])
            else: btris = border_triangles
            cxs = np.asarray([srf.coordinates[:,f] for f in contained_faces])
            return triangle_areas(*cxs) + triangle_areas(*btris)
        if isinstance(surface, Topology):
            return pimms.lazy_map({k:curry(sarea, k) for k in six.iterkeys(surface.surfaces)})
        else: return sarea(surface)
    def reverse(self, meta_data=None):
        '''
        path.reverse() yields a path that is equivalent to the given path but reversed (thus it is
          considered to contain all the vertices not contained by path if path is closed).
        '''
        addrs = {k:np.fliplr(v) for (k,v) in six.iteritems(self.addresses)}
        return Path(self.surface, addrs, meta_data=meta_data)

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
        if not isinstance(mp, MapProjection):
            raise ValueError('trace map_projection must be a MapProjection or None')
        return mp.persist()
    @pimms.param
    def points(x):
        '''
        trace.points is a either the coordinate matrix of the traced points represented by the given
        trace object or is the curve-spline object that represents the given path trace.
        '''
        if isinstance(x, CurveSpline): return x.persist()
        x = np.asarray(x)
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
            if closed != bool(points.periodic): return points
            else: return points.copy(periodic=closed)
        # default order to use is 0 for a trace
        return curve_spline(points[0], points[1], order=1, periodic=closed).persist()
    def to_path(self, obj):
        '''
        trace.to_path(subj) yields a path reified on the given subject's cortical surface; the
          returned value is a Path object.
        trace.to_path(topo) yields a path reified on the given topology/cortex object topo.
        trace.to_path(mesh) yields a path reified on the given spherical mesh.
        '''
        # make a flat-map of whatever we've been given...
        if isinstance(obj, Mesh) and obj.coordinates.shape[0] == 2: fmap = obj
        else: fmap = self.map_projection(obj) 
        # we are doing this in 2D
        fmap = self.map_projection(obj)
        crv = self.curve
        cids = fmap.container(crv.coordinates)
        pts = crv.coordinates.T
        if self.closed: pts = np.concatenate([pts, [pts[0]]])
        coords = pts.T
        allpts = []
        for ii in range(len(pts) - 1):
            allpts.append([pts[ii]])
            seg = coords[:,[ii,ii+1]].T
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
def close_path(*args):
    '''
    close_path(path1, path2...) yields the path formed by joining the list of paths at their
      intersection points in the order given. Note that the order in which each path is specified is
      ultimately ignored by this function--the only order that matters is the order in which the
      list of paths is given.
    '''
    #TODO
    pass
        
####################################################################################################
# Some Functions that deal with converting to/from the above classes

def to_tess(obj, properties=None, meta_data=None):
    '''
    to_tess(obj) yields a Tesselation object that is equivalent to obj; if obj is a tesselation
      object already and no changes are requested (see options) then obj is returned unmolested.

    The following objects can be converted into tesselations:
      * a tesselation object
      * a mesh or topology object (yields their tess objects)
      * a 3 x n or n x 3 matrix of integers (the faces)
      * a tuple (coords, faces), as returned when, for example, loading a freesurfer geometry file;
        in this cases, the result is equivalent to to_tess(faces)

    The following options are accepted:
      * properties (default: None) specifies properties that should be given to the tesselation
        object (see Tesselation and VertexSet).
      * meta_data (default: None) specifies meta-data that should be attached to the tesselation
        object (see Tesselation).
    '''
    if isinstance(obj, Tesselation): res = obj
    elif isinstance(obj, Mesh): res = obj.tess
    elif isinstance(obj, Topology): res = obj.tess
    elif pimms.is_matrix(obj, 'number'): res = Tesselation(obj)
    elif isinstance(obj, _tuple_type) and len(obj) == 2 and pimms.is_matrix(obj[1], 'int'):
        res = Tesselation(obj[1])
    else: raise ValueError('Cannot deduce how object is a tesselation')
    if properties is not None: res = res.with_prop(properties)
    if meta_data is not None:  res = res.with_meta(meta_data)
    return res
def to_mesh(obj, properties=None, meta_data=None):
    '''
    to_mesh(obj) yields a Mesh object that is equivalent to obj; if obj is a mesh object and no
      changes are requested (see options) then obj is returned unmolested.

    The following objects can be converted into meshes:
      * a mesh object
      * a tuple (coords, faces) where coords is a coordinate matrix and faces is a matrix of
        coordinate indices that make-up the triangles
    '''
    if isinstance(obj, Mesh):
        res = obj
        if properties is not None: res = res.with_prop(properties)
        if meta_data is not None: res = res.with_meta(meta_data)
        return res
    elif isinstance(obj, _tuple_type) and len(obj) == 2:
        return Mesh(obj[1], obj[0], properties=properties, meta_data=meta_data)
    else:
        raise ValueError('Could not deduce how object can be convertex into a mesh')

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
            except: pass
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

