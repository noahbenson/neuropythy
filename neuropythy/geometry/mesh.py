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
import sys, six, types, logging, pimms

from .util  import (triangle_area, triangle_address, alignment_matrix_3D, rotation_matrix_3D,
                    cartesian_to_barycentric_3D, cartesian_to_barycentric_2D,
                    barycentric_to_cartesian, point_in_triangle)
from ..util import (ObjectWithMetaData, to_affine, zinv, is_image, address_data)
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
        res = [edges[0][1]]
        for i in range(len(edges)):
            for e in edges:
                if e[0] == res[i]:
                    res.append(e[1])
                    break
        return tuple(res)
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
        if len(tri_no) == 0:
            tri = self.coordinates[:, self.tess.indexed_faces[:, tri_no]]
        else:
            tri = np.transpose([self.coordinates[:,t] for t in self.tess.indexed_faces[:,tri_no]],
                               (2,0,1))
        return point_in_triangle(tri, pt)

    def _find_triangle_search(self, x, k=24, searched=set([])):
        # This gets called when a container triangle isn't found; the idea is that k should
        # gradually increase until we find the container triangle; if k passes the max, then
        # we give up and assume no triangle is the container
        if k >= 288: return None
        (d,near) = self.facee_hash.query(x, k=k)
        near = [n for n in near if n not in searched]
        searched = searched.union(near)
        tri_no = next((kk for kk in near if self.is_point_in_face(kk, x)), None)
        return (tri_no if tri_no is not None
                else self._find_triangle_search(x, k=(2*k), searched=searched))
    
    def nearest_vertex(self, pt):
        '''
        mesh.nearest_vertex(pt) yields the id number of the nearest vertex in the given
        mesh to the given point pt. If pt is an (n x dims) matrix of points, an id is given
        for each column of pt.
        '''
        (d,near) = self.vertex_hash.query(pt, k=1)
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
    
    def nearest_data(self, pt, k=2, n_jobs=1):
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
        (d, near) = self.face_hash.query(pt, k=k)
        ids = [tri_no if tri_no is not None else self._find_triangle_search(x, 2*k, set(near_i))
               for (x, near_i) in zip(pt, near)
               for tri_no in [next((k for k in near_i if self.is_point_in_face(k, x)), None)]]
        pips = [self.point_in_plane(i, p) if i is not None else (0, None)
                for (i,p) in zip(ids, pt)]
        return (np.asarray(ids),
                np.asarray([d[0] for d in pips]),
                np.asarray([d[1] for d in pips]))

    def nearest(self, pt, k=2, n_jobs=1):
        '''
        mesh.nearest(pt) yields the point in the given mesh nearest the given array of points pts.
        '''
        dat = self.nearest_data(pt)
        return dat[2]

    def nearest_vertex(self, x, n_jobs=1):
        '''
        mesh.nearest_vertex(x) yields the vertex index or indices of the vertex or vertices nearest
          to the coordinate or coordinates given in x.
        '''
        x = np.asarray(x)
        if len(x.shape) == 1: return self.nearest_vertex([x], n_jobs=n_jobs)[0]
        if x.shape[0] == self.coordinates.shape[0]: x = x.T
        n = self.coordinates.shape[1]
        (_, nei) = self.vertex_hash.query(x, k=1) #n_jobs fails? version problem?
        return nei

    def distance(self, pt, k=2, n_jobs=1):
        '''
        mesh.distance(pt) yields the distance to the nearest point in the given mesh from the points
        in the given matrix pt.
        '''
        dat = self.nearest_data(pt)
        return dat[1]

    def container(self, pt, k=2, n_jobs=1):
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
                    near = self.face_hash.query(sub_pts, k=cur_k)[1]
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
        interp = interp.tocsr()
        interp.eliminate_zeros()
        (m,n) = interp.shape # n: no. of vertices in mesh; m: no. points being interpolated
        # We apply weights first, because they are fairly straightforward:
        if weights is not None:
            weights = np.array(weights)
            weights[np.logical_not(np.isfinite(weights))] = 0
            weights[weights < 0] = 0
            interp = interp.dot(sps.diags(weights))
        # we make a mask with 1 extra element, always out of the mask, as a way to flag vertices
        # that shouldn't appear in the mask for some other reason
        if mask is None or (pimms.is_str(mask) and mask.lower() == 'all'):
            mask = np.ones(n + 1, dtype=np.bool)
            mask[n] = 0
        else:
            interp = interp.dot(sps.diags(np.asarray(mask, dtype=np.float)))
            interp.eliminate_zeros()
            mask = np.concatenate([mask, [False]])
        # we may need to rescale the rows now:
        (closest, rowdivs) = np.transpose(
            [(r.indices[np.argsort(r.data)[-1]], 1/ss) if np.isfinite(ss) and ss > 0 else (n,0)
             for r in interp
             for ss in [r.data.sum()]])
        closest = np.asarray(closest, dtype=np.int)
        if not np.array_equal(rowdivs, np.ones(len(rowdivs))):
            # rescale the rows
            interp = sps.diags(rowdivs).dot(interp)
        # any row with no interpolation weights or that is nearest to a vertex not in the mesh
        # needs to be given a nan value upon interpolation
        bad_pts = np.logical_not(mask[closest])
        if bad_pts.sum() > 0:
            interp = interp.tolil()
            interp[bad_pts, 0] = np.nan
            interp = interp.tocsr()
        return interp
    def nearest_interpolation(self, coords, n_jobs=1):
        '''
        mesh.nearest_interpolation(x) yields an interpolation matrix for the given coordinate or
          coordinate matrix x. An interpolation matrix is just a sparce array M such that for a
          column vector u with the same number of rows as there are vertices in mesh, M * u is the
          interpolated values of u at the coordinates in x.
        '''
        coords = np.asarray(coords)
        if coords.shape[0] == self.coordinates.shape[0]: coords = coords.T
        n = self.coordinates.shape[1]
        m = coords.shape[0]
        nv = self.nearest_vertex(coords, n_jobs=n_jobs)
        return sps.csr_matrix(
            (np.ones(len(nv), dtype=np.int), (range(len(nv)), nv)),
            shape=(m,n),
            dtype=np.int)
    def linear_interpolation(self, coords, n_jobs=1):
        '''
        mesh.linear_interpolation(x) yields an interpolation matrix for the given coordinate or 
          coordinate matrix x. An interpolation matrix is just a sparce array M such that for a
          column vector u with the same number of rows as there are vertices in mesh, M * u is the
          interpolated values of u at the coordinates in x.
        '''
        coords = np.asarray(coords)
        if coords.shape[0] == self.coordinates.shape[0]: coords = coords.T
        n = self.coordinates.shape[1]
        m = coords.shape[0]
        mtx = sps.lil_matrix((m, n), dtype=np.float)
        tris = self.tess.indexed_faces
        # first, find the triangle containing each point...
        containers = self.container(coords, n_jobs=n_jobs)
        # which points are in a triangle at all...
        contained_q = np.asarray([x is not None for x in containers], dtype=np.bool)
        contained_idcs = np.where(contained_q)[0]
        containers = containers[contained_idcs].astype(np.int)
        # interpolate for these points
        tris = tris[:,containers]
        corners = np.transpose(self.face_coordinates[:,:,containers], (0,2,1))
        coords = coords[contained_idcs]
        # get the mini-triangles' areas
        a_area = triangle_area(coords.T, corners[1].T, corners[2].T)
        b_area = triangle_area(coords.T, corners[2].T, corners[0].T)
        c_area = triangle_area(coords.T, corners[0].T, corners[1].T)
        tot = a_area + b_area + c_area
        for (x,ii,f,aa,ba,ca,tt) in zip(coords, contained_idcs, tris.T, a_area,b_area,c_area,tot):
            if np.isclose(tt, 0):
                (aa,ba,ca) = np.sqrt(np.sum((coords[f] - x)**2, axis=1))
                # check if we can do line interpolation
                (zab,zbc,zca) = np.isclose((aa,ba,ca), (ba,ca,aa))
                (aa,ba,ca) = (1.0,   1.0,      1.0) if zab and zbc and zca else \
                             (ca,     ca,    aa+ba) if zab                 else \
                             (ba+ca,  aa,       aa) if zbc                 else \
                             (ba,     aa+ca,    ba)
                tt = aa + ba + ca
            mtx[ii, f] = (aa/tt, ba/tt, ca/tt)
        return mtx.tocsr()
    def apply_interpolation(self, interp, data, mask=None, weights=None):
        '''
        mesh.apply_interpolation(interp, data) yields the result of applying the given interpolation
          matrix (should be a scipy.sparse.csr_matrix), which can be obtained via
          mesh.nearest_interpolation or mesh.linear_interpolation, and applies it to the given data,
          which should be matched to the coordinates used to create the interpolation matrix.

        The data argument may be a list/vector of size m (where m is the number of columns in the
        matrix interp), a matrix of size (n x m) for any n, or a map whose values are lists of
        length m.

        The following options may be provided:
          * mask (default: None) specifies which elements should be in the mask.
          * weights (default: None) additional weights that should be applied to the vertices in the
            mesh during interpolation.
        '''
        # we can start by applying the mask to the interpolation
        interp = Mesh.scale_interpolation(interp, mask=mask, weights=weights)
        (m,n) = interp.shape
        # if data is a map, we iterate over its columns:
        if pimms.is_str(data):
            return self.apply_interpolation(
                interp,
                self.properties if data.lower() == 'all' else self.properties[data])
        elif pimms.is_lazy_map(data):
            def _make_lambda(kk): return lambda:self.apply_interpolation(interp, data[kk])
            return pimms.lazy_map({k:_make_lambda(k) for k in six.iterkeys(data)})
        elif pimms.is_map(data):
            return pyr.pmap({k:self.apply_interpolation(interp, data[k])
                             for k in six.iterkeys(data)})
        elif pimms.is_matrix(data):
            data = np.asarray(data)
            if data.shape[0] == n:
                return np.asarray([self.apply_interpolation(interp, row) for row in data.T]).T
            else:
                return np.asarray([self.apply_interpolation(interp, row) for row in data])
        elif pimms.is_vector(data) and len(data) != n:
            return tuple([self.apply_interpolation(interp, d) for d in data])
        # If we've made it here, we have a single vector to interpolate;
        # we might have non-finite values in this array (additions to the mask), so let's check:
        data = np.asarray(data)
        # numeric arrays can be handled relatively easily:
        if pimms.is_vector(data, np.number):
            numer = np.isfinite(data)
            if np.sum(numer) < n:
                # just remove these elements from the mask
                data = np.array(data)
                data[np.logical_not(numer)] = 0
                interp = Mesh.scale_interpolation(interp, mask=numer)
            return interp.dot(data)
        # not a numerical array; we just do nearest interpolation
        return np.asarray(
            [data[r.indices[np.argsort(r.data)[-1]]] if np.isfinite(ss) and ss > 0 else np.nan
             for r in interp
             for ss in [r.data.sum()]])

    def interpolate(self, x, data, mask=None, weights=None, method='automatic', n_jobs=1):
        '''
        mesh.interpolate(x, data) yields a numpy array of the data interpolated from the given
          array, data, which must contain the same number of elements as there are points in the
          Mesh object mesh, to the coordinates in the given point matrix x. Note that if x is a
          vector instead of a matrix, then just one value is returned.
        
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
          * n_jobs (default: 1) is passed along to the cKDTree.query method, so may be set to an
            integer to specify how many processors to use, or may be -1 to specify all processors.
        '''
        n = self.vertex_count
        if isinstance(x, Mesh): x = x.coordinates
        if method is None: method = 'auto'
        method = method.lower()
        if method == 'linear':
            interp = self.linear_interpolation(x, n_jobs=n_jobs)
        elif method == 'nearest':
            return self.apply_interpolation(self.nearest_interpolation(x, n_jobs=n_jobs),
                                            data, mask=mask, weights=weights)
        elif method == 'auto' or method == 'automatic':
            # unique challenge; we want to calculate the interpolation matrices but once:
            interps = pimms.lazy_map(
                {'nearest': lambda:Mesh.scale_interpolation(
                    self.nearest_interpolation(x, n_jobs=n_jobs),
                    mask=mask, weights=weights),
                 'linear': lambda:Mesh.scale_interpolation(
                    self.linear_interpolation(x, n_jobs=n_jobs),
                    mask=mask, weights=weights)})
            # we now need to look over data...
            def _apply_interp(dat):
                if pimms.is_str(dat):
                    return _apply_interp(self.properties[dat])
                elif pimms.is_vector(dat, np.inexact):
                    return self.apply_interpolation(interps['linear'], dat)
                else:
                    return self.apply_interpolation(interps['nearest'], dat)
            if pimms.is_str(data) and data.lower() == 'all':
                data = self.properties
            if pimms.is_lazy_map(data):
                def _make_lambda(kk): return lambda:_apply_interp(data[kk])
                return pimms.lazy_map({k:_make_lambda(k) for k in six.iterkeys(data)})
            elif pimms.is_map(data):
                return pyr.pmap({k:_apply_interp(data[k]) for k in six.iterkeys(data)})
            elif pimms.is_matrix(data):
                # careful... the rows could have a tuple of rows of different types...
                if len(data) == n:
                    # in this case we assume that all cells are the same type
                    data = np.asarray(data)
                    return np.asarray([_apply_interp(np.asarray(row)) for row in data.T]).T
                elif pimms.is_nparray(data):
                    return np.asarray([_apply_interp(row) for row in data])
                else:
                    return tuple([_apply_interp(row) for row in data])
            elif pimms.is_vector(data, np.number) and len(data) == self.tess.vertex_count:
                return _apply_interp(data)
            elif pimms.is_vector(data):
                return tuple([_apply_interp(d) for d in data])
            else:
                return _apply_interp(data)
        else:
            raise ValueError('method argument must be linear, nearest, or automatic')
        return self.apply_interpolation(interp, data, mask=mask, weights=weights)

    def address(self, data):
        '''
        mesh.address(X) yields a dictionary containing the address or addresses of the point or
        points given in the vector or coordinate matrix X. Addresses specify a single unique 
        topological location on the mesh such that deformations of the mesh will address the same
        points differently. To convert a point from one mesh to another isomorphic mesh, you can
        address the point in the first mesh then unaddress it in the second mesh.
        '''
        # we have to have a topology and registration for this to work...
        if isinstance(data, Mesh):
            return self.address(data.coordinates)
        data = np.asarray(data)
        idxfs = self.tess.indexed_faces
        if len(data.shape) == 1:
            face_id = self.container(data)
            if face_id is None: return None
            tx = self.coordinates[:, idxfs[:,face_id]].T
            faces = self.tess.faces[:,face_id]
        else:
            data = data if data.shape[1] == 3 or data.shape[1] == 2 else data.T
            face_id = np.asarray(self.container(data))
            null = np.full((idxfs.shape[0], self.coordinates.shape[0]), np.nan)
            tx = np.transpose([self.coordinates[:,idxfs[:,f]] if f is not None else null
                               for f in face_id],
                              (2,1,0))
            faces = self.tess.faces
            null = [-1, -1, -1]
            faces = np.transpose([faces[:,f] if f is not None else null for f in face_id])
        bc = cartesian_to_barycentric_3D(tx, data) if self.coordinates.shape[0] == 3 else \
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
        if isinstance(obj, Topology):
            # check the chiralities
            if obj.chirality is not None and self.chirality is not None:
                if obj.chirality != self.chirality:
                    raise ValueError('given topology is the wrong chirality for projection')
            # We need to figure out if there is a matching registration
            reg = self.registration
            if self.registration is None: reg = 'native'
            if reg in obj.registrations:
                return self(obj.registrations[reg], tag=tag)
            else:
                raise ValueError('given topology does not include the registration %s' % reg)
        elif isinstance(obj, Mesh):
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
    def interpolate(self, topo, data, mask=None, weights=None, method='automatic', n_jobs=1):
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
          * n_jobs (default: 1) is passed along to the cKDTree.query method, so may be set to an
            integer to specify how many processors to use, or may be -1 to specify all processors.
        '''
        if not isinstance(topo, Topology):
            raise ValueError('Topologies can only be interpolated with other topologies')
        reg_names = [k for k in topo.registrations.iterkeys() if k in self.registrations
                     if k != 'native']
        if not reg_names:
            raise RuntimeError('Topologies do not share a matching registration!')
        res = None
        for reg_name in reg_names:
            try:
                res = self.registrations[reg_name].interpolate(
                    topo.registrations[reg_name], data,
                    mask=mask, method=method, n_jobs=n_jobs);
                break
            except: pass
        if res is None:
            raise ValueError('All shared topologies raised errors during interpolation!')
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
      * 'mesh' returns the data as a mesh, assuming that there are two darray elements stored in the,
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
        # is this is mesh gifti?
        if len(dat.darrays) == 2 or len(dat.darrays) == 3:
            if len(dat.darrays) == 2: (cor,    tri) = dat.darrays
            else:                     (cor, _, tri) = dat.darrays
            cor = cor.data
            tri = tri.data
            # possible that these were given in the wrong order:
            if pimms.is_matrix(tri, np.inexact) and pimms.is_matrix(cor, np.signedinteger):
                (cor,tri) = (tri,cor)
            # okay, try making it:
            try: return Mesh(tri, cor)
            except: pass
        # is it a coord or topo?
        if len(dat.darrays) == 1:
            cor = dat.darrays[0].data
            if pimms.is_matrix(cor, np.inexact): return cor
            if pimms.is_matrix(cor, 'int'):     return Tesselation(cor)
        # We don't know what it is:
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
