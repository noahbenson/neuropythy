# -*- coding: utf-8 -*-
####################################################################################################
# neuropythy/util/vset.py
# This file contains code fo the VertexSet class and related classes.

import os, sys, six, pimms
import numpy        as np
import scipy.sparse as sps
import pyrsistent   as pyr
from functools import reduce

from .. import math as nym
from .core import (is_str, is_tuple, is_list, is_map)

def simplex_index_csrmatrix(simplex_matrix, max_label=None):
    """Constructs an index matrix from a simplex matrix.

    Parameters
    ----------
    simplex_matrix : matrix of int
        A matrix whose columns are each simplices.
    max_label : None or int, optional
        The maximum vertex label to use in the matrix. If `None` (the default),
        then uses the maximum value found in the `simplex_matrix`. Note that
        this should not in fact be the maximum label value, but that value
        plus one--i.e., tihs us the nnumberrr of unique labels if the labels
        are numbered 0 to 1.

    Returns
    -------
    scipy.sparse.csr_matrix
        A SciPy CSR sparse matrix of the index.
    """
    mtx = asarray(simplex_matrix, dtype='int')
    if max_label is None: max_label = nym.max(mtx)
    (d,n) = mtx.shape
    u = simplex_matrix[:,0]
    if d > 1:
        w = np.full(d-1, max_label)
        w[0] = 1
        v = np.dot(np.cumprod(w, out=w), simplex_matrix[1:,:])
        return sps.csr_matrix((np.arange(1, 1+n), (u,v)),
                              dtype=np.int,
                              shape=(max_label, max_label))
    else:
        zz = np.zeros(n, dtype=np.int)
        return sps.csr_matrix((np.arange(1, 1+n), (zz,v)),
                              dtype=np.int,
                              shape=(max_label, max_label))
def loworder_simplex_matrix(mtx, unordered=None):
    """Returns a simplex matrix for simplices one order lower.

    Parameters
    ----------
    mtx : matrix of int
        The simplex matrix whose order is to be lowered.
    unordered : boolean or None, optional
        Whether to eliminate duplicates with diifferent vertex orderings
        (`True`) or not (`False`). If `None`, then deletes duplicates up to the
        ordering if the returned matrix has 2 or fewer rows. The default is
        `None`.
    
    Returns
    -------
    matrix of int
        A simplex matrix one order lower than `mtx`.
    """
    # We just roll the indices we take.
    mtx = nym.promote(mtx, dtype='int')
    if mtx.ndim < 2: return None
    if mtx.ndim > 2: raise ValueError("simpleex matrix is required")
    (d,n) = mtx.shape
    if d == 1: return None
    if unordered is None: unordered = (d < 4)
    lm = np.hstack([np.roll(mtx, k, axis=0)[:-1,:] for k in np.arange(d)])
    if unordered:
        mx = np.max(lm) + 1
        w = np.full(len(lm)-1, mx)
        w[0] = 1
        v = np.unique(np.dot(lm, np.cumprod(w, out=w)))
        lm = np.empty((len(lm), len(v)), dtype=np.int)
        lm[1:,:] = [v // m for m in w[1:]]
        lm[0,:] = v % mx
    return lm
@pimms.immutable
class VertexIndex(object):
    """An index for arrangements of vertices.

    A `VertexIndex` represents vertices and simplices in a vertex-set in a way
    that can efficiently be queried for lookup operations. Examples of these
    operations include determining the vertex indices for a list of vertex
    labels or determining the face indices of a matrix of faces.

    Parameters
    ----------
    *args
        A sequence of simplex matrices. Each element `k` of `args` represents
        the order `k` simplices; for example, args[0] must always be a vector
        (or `1`x`n` matrix) of the vertex labels. The second argument, `args[1]`
        must be the `2`x`n` matrix of edges, where the elements of the matrix
        are vertex labels. `args[2]` must contain triangle simplices. Higher
        order simplices are supported as well. Each argument except the first
        may be `None` if that simplex type is not supported. If one of these
        values is instead `Ellipsis`, then the simplex matrix is constructed
        from higher order matrices.

    Attributes
    ----------
    simplex_matrices : tuple of read-only numpy arrays
        A tuple of the simplex matrices tracked by the index.
    max_label : int
        The maximum label value that occurs in the matrices.
    simplex_counts : tuple of int
        A tuple of integers, each indicating the number of simplices of each
        rank.
    indexes : tuple of scipy sparse csr_matrices
        A tuple of the index matrices.
    """
    def __init__(self, *args):
        self.simplex_matrices = args
    @pimms.param
    def simplex_matrices(m):
        m = list(m)
        n = len(m)
        if m[-1] is Ellipsis: raise ValueError("args cannot end with ...")
        for k in range(n-1,-1,-1):
            a = args[k]
            if a is None: continue
            elif a is Ellipsis:
                a1 = args[k+1]
                if a1 is None:
                    raise ValueError("Ellipsis cannot precede None")
                a = loworder_simplex_matrix(a1)
            m[k] = nym.to_readonly(a)
        # We're required to have a 0 matrix
        if len(m) < 1 or m[0] is None:
            raise ValueError("vertex labels must be provided to VertexIndex")
        return tuple(m)
    @pimms.value
    def max_label(simplex_matrices):
        return np.max([nym.max(u) for u in simplex_matrices if u is not None])
    @pimms.value
    def simplex_counts(simplex_matrices):
        return tuple([0 if u is None else u.shape[-1] for u in simplex_matrices])
    @pimms.value
    def indexes(simplex_matrices, max_label):
        return tuple([simplex_index_csrmatrix(m, max_label=max_label+1)
                      for m in simplex_matrices])
    def __call__(self, labels, error=True):
        """Converts an array of vertex labels into indices.

        `index(array)` returns an array the same shape as `array`, but whereas
        `array` is an array of labels, the return value is an array of
        equivalent vertex indices.

        In comparison to `index[array]`, `index(array)` is equivalent if `array`
        has fewer than 2 dimensions, as vertex indices are looked up from the
        vertex labels that are given in both cases. If `array` is a matrix, then
        whereas `index(array)` converts the label in each individual cell into a
        vertex index, `index[array]` returns a vector of the simplex-indices
        for the simplices represented in the columns of the matrix `array`.

        Parameters
        ----------
        labels : array-like of ints
            The labels that are to be converted into indices.
        error : boolean, optional
            Whether to raise an error when a label is not found. If `True` then
            a `ValueError` is raised when a label is not found in the index. If
            `False`, then these labels are given the value `-1`. The default is
            `True`.

        Returns
        -------
        array-like of ints
            The vertex indices that correspond to the given array of labels.

        Raises
        ------
        ValueError
            If a label cannot be found and `error` is `True`.
        """
        index = self.indexes[0]
        lbl = nym.toarray(labels, dtype=np.int)
        if not nym.is_numeric(lbl, '<=int'):
            raise ValueError("vertex index requires integer arguments")
        sh = lbl.shape
        nn = np.prod(sh, dtype=np.int)
        ll = lbl.view()
        ll.shape = (n,)
        ii = index[0,ll]
        if error:
            found = index.data.shape[0]
            if found != n:
                raise ValueError(f"vertex index failed to find {n-found} labels")
        lbl.shape = (1,n)
        ii.todense(out=lbl)
        lbl.shape = sh
        return nym.astensor(lbl, device=labels.device) if nym.istensor(labels) else lbl
    def __getitem__(self, simplex):
        """Converts a matrix, tuple, or argument list of simplces into indices.

        `index(array)` returns an array the same shape as `array`, but whereas
        `array` is an array of labels, the return value is an array of
        equivalent vertex indices.

        In comparison to `index[array]`, `index(array)` is equivalent if `array`
        has fewer than 2 dimensions, as vertex indices are looked up from the
        vertex labels that are given in both cases. If `array` is a matrix, then
        whereas `index(array)` converts the label in each individual cell into a
        vertex index, `index[array]` returns a vector of the simplex-indices
        for the simplices represented in the columns of the matrix `array`.

        Parameters
        ----------
        simplex
            The simplices, specified as columns of vertex labels, that are to be
            converted into simplex-indices. This may be a single matrix, vector,
            or value, or it may be a sequence of simplex corners that together
            form a matrix. A vector passed alone is always interpreted as a
            single simplex.

        Returns
        -------
        vector of ints
            The simplex indices that correspond to the columns of the given
            simplex matrix.

        Raises
        ------
        ValueError
            If a simplex cannot be found.
        """
        s = nym.asarray(simplex).view()
        if s.ndim == 0: s.shape = (1,1)
        elif s.ndim == 1: s.shape = (s.shape[0],1)
        elif s.ndim > 2: raise ValueError("simplex argument must be a matrix, vector, or value")
        (d,n) = s.shape
        if d > len(self.indexes): raise ValueError("simplex rank too high for index")
        index = self.indexes[d-1]
        if index is None:
            raise ValueError(f"index does not contain {d}-rank simplices")
        if d == 1:
            u = index[0, s[0]]
        elif d == 2:
            u = index[s[0],s[1]]
        else:
            w = np.full(d-1, self.max_label, dtype=np.int)
            w[0] = 1
            np.cumprod(w, out=w)
            a = s[0]
            b = np.dot(w, s)
            u = index[(a, b)]
        if len(u.data) != n: raise ValueError(f"{n-u.data} simplices not found in index")
        out = u.data
        return nym.astensor(out, device=simplex.device) if nym.istensor(simplex) else out
@pimms.immutable
class VertexSet(ObjectWithMetaData):
    """Sets of vertices that can have properties.

    `VertexSet` is a class that tracks a number of vertices, including
    properties for them. This class is intended as a base class for
    `Tesselation` and `Mesh`, both of which track vertex properties.  Note that
    all `VertexSet` objects add/overwrite the keys `'index'` and `'label'` in
    their propeties in order to store the vertex indices and labels as
    properties.

    `VertexSet` can produce subsets (also `VetexSet` objects) that use the same
    labels as their paent supersets. Through this mechanism, properrties
    computed using subsets can easily be applied back to superset meshes: the
    labels of the subset are the indices of the original superset.

    Parameters
    ----------
    labels : vector-like of ints
        The labels of the vertices in the set. The ordering of all label and
        property vector values is always presumed to be the same.  properties :
        mapping of strings to vector-like, optional
    properties : mapping of strings to array-likes, optional
        The mapping of properties. This must be a dictionary, persistent map, or
        similar (pimms lazy maps are respected); all keys must be strings
        (property names), and all values must be array-like objects whose first
        dimension is equal to the number of vertices (i.e., the length of
        `labels`).
    meta_data : mapping, optional
        An optional mapping of meta-data about the vertex-set.

    Attributes
    ----------
    labels : vector of ints
        A read-only numpy array of the integer labels for each vertex in the
        vertex-set. Labels are always non-negative.
    vertex_count : int
        The number of vertices in the vertex-set.
    indices : vector of ints
        A read-only numpy array of the non-negative integer indices for each
        vertex. This is always equivalent to `arange(vertex_count)`.
    index : VertexIndex
        A `VertexIndex` object that can convert from vertex labels to vertex
        indices.
    properties : mapping of str to array-like
        A read-only mapping of property names (keys) to array-like objects whose
        first dimension is equal to the number of vertices in the vertex-set.
    repr : str
        A string representation of the vertex-set.
    """
    def __init__(self, labels, properties=None, meta_data=None):
        self._properties = properties
        self.labels = labels
        self.meta_data = meta_data
    @pimms.param
    def labels(lbls):
        '''vset.labels is an array of the integer vertex labels.'''
        return nym.to_readonly(lbls)
    @pimms.param
    def _properties(props):
        '''
        obj._properties is an itable of property values given to the vertex-set obj; this is a
        pre-processed input version of the value obj.properties.
        '''
        if props is None: return None
        elif pimms.is_map(props): return pimms.persist(props)
        else: raise ValueError('provided properties data must be a mapping')
    @pimms.value
    def vertex_count(labels):
        '''vset.vertex_count is the number of vertices in the given vertex set vset.'''
        return len(labels)
    @pimms.value
    def indices(vertex_count):
        '''vset.indices is the list of vertex indices for the given vertex-set vset.'''
        idcs = np.arange(0, vertex_count, 1, dtype=np.int)
        return nym.as_readonly(idcs)
    @pimms.value
    def index(labels):
        '''vset.index is the index of the labels of the vertex-set.'''
        return VertexIndex(labels).persist()
    # The idea here is that _properties may be provided by the overloading class, then properties
    # can be overloaded by that class to add unmodifiable properties to the object; e.g., meshes
    # want coordinates to be a property that cannot be updated.
    @pimms.value
    def properties(_properties, labels, indices):
        '''obj.properties is an itable of property values given to the vertex-set obj.'''
        _properties = pyr.m() if _properties is None else _properties
        pp = _properties.set('index', indices).set('label', labels)
        return pimms.ITable(pp, row_count=len(labels)).persist()
    
    @pimms.value
    def repr(vertex_count):
        '''obj.repr is the representation string returned by obj.__repr__().'''
        return f'VertexSet(<{vertex_count} vertices>)'
    # Normal Methods
    def __repr__(self):
        return self.repr
    def prop(self, *args):
        """Returns the property array with the given name or raises an error.

        `obj.prop(name)` returns the vertex property in the given object with
        the given name.
        
        `obj.prop(data)` yields `data` if `data` is a valid vertex property for
        the given object; this means it must have a length of
        `obj.vertex_count`.

        `obj.prop(p1, p2...)` yields a `d`x`n` array of properties where `d`
        is the number of properties given and `n` is `obj.vertex_count`.

        `obj.prop(set([name1, name2...]))` yields a mapping of the given names
        mapped to the appropriate property values.

        Parameter
        ---------
        *args
            A property name, a valid property, or a set of property names, as
            described above, or a sequence of these.

        Returns
        -------
        array or mapping
            An array of properties or a mapping or properties, as described
            above.

        Raises
        ------
        ValueError
            When the property cannot be recognized.
        """
        narg = len(args)
        if nargs == 0:
            raise ValueError("prop() requires at least one argument")
        if nargs > 1:
            return nym.promote([self.prop(u) for u in args])
        if is_str(name):
            return self.properties[name]
        elif is_set(name):
            return {nm:self.properties[nm] for nm in name}
        elif nym.arraylike(name, shape=(self.vertex_count, Ellipsis)):
            return name
        else:
            raise ValueError('unrecognized property')
    def with_prop(self, *args, **kwargs):
        """Returns a duplicates the object with new properties.

        `obj.with_prop(...)` yields a duplicate of the given object with the
        properties provided in the argument-list added to it. The properties may
        be specified as a sequence of mapping objects such as python dicts
        followed by any number of keyword arguments, all of which are merged
        into a single dict left-to-right before application.

        Parameters
        ----------
        *args
            Any number of mapping objects, such as dicts, to be merged into the
            new vertex-set's properties.
        **kwargs
            Any number of keyword arguments to be merged into the new
            vertex-set's properties.

        Returns
        -------
        VertexSet
            A duplicate of the current vertex-set, but with the provided
            properties merged into its properties.
        """
        pp = {} if self._properties is None else self._properties
        pp = pimms.rmerge(pp, *(args + (kwargs,)))
        if pp is self._properties: return self
        if len(pp) == 0 and self._properties is None: return self
        pp = pimms.ITable(pp, n=self.vertex_count)
        return self if pp is self._properties else self.copy(_properties=pp)
    def wout_prop(self, *args):
        """Returns a duplicate object with new properties.

        `obj.wout_prop(...)` yields a duplicate of the given object with the
        properties named in the argument list removed from it. The properties
        may be specified as a sequence of column names or lists of column names.

        Parameters
        ----------
        *args
            A sequence of property names that should be removed from the object.

        Return
        ------
        VertexSet
            The duplicate object without the given properties.
        """
        pp = self._properties
        for a in args: pp = pp.discard(a)
        return self if pp is self._properties else self.copy(_properties=pp)
    def property(self, prop,
                 dtype=None,     filter=None,
                 outliers=None,  data_range=None,    clipped=np.inf,
                 mask=None,      valid_range=None,   null=np.nan):
        """Extracts properties from an object and applies filters to them.
        
        `obj.to_property(prop)` yields the given property `prop` from `obj`
        after performing a set of filters on the property, as specified by the
        options. In the property array that is returned, the values that are
        considered outliers (data out of some range) are indicated by `inf`, and
        values that are not in the optionally-specified mask are given the value
        `nan`; these may be changed with the clipped and null options,
        respectively.
    
        `obj.propert(prop)` is equivalent to `to_property((obj, prop))` and to
        `to_property(obj, prop)`.

        The property argument `prop` may be either specified as a string (a
        property name in the object) or as a property vector. The weights option
        may also be specified this way. Additionally, the `prop` argument may be
        a list such as `['polar_angle', 'eccentricity']` where each element is
        either a string or a vector, in which case the result is a matrix of
        properties. Finally, `prop` may be a set of property names, in which
        case the return value is a `pimms.ITable` whose keys are the property
        names.

        Parameters
        ----------
        prop : str or property vector or list, optional
            A proprety name, a property vector, or a list of these, as detailed
            above. `prop` should be provided only if `obj` is not itself a tuple
            `(obj,prop)`.
        mask : None or mask-like
            The specification of vertices that should be included in the
            property array; values are specified in the mask in accordance with
            the `to_mask()` function.
        null : object, optional
            The value marked in the array as out-of-mask. By default, this is
            `nan`.
        filter : function or None, optional
            A function to be passed the property array prior to being returned
            (after `null` and `clipped` values have been marked). The value
            returned by this function is returned instead of the property.
        dtype : dtype-like or None
            The type of the array that should be returned. `None`, the default,
            indicates that the type of the given property should be used.
            Otherwise, this must be a dtype-like. Both `numpy` and `torch`
            dtypes are supported.

        Returns
        -------
        array-like
            An array of properties, as described above. If the `dtype` parameter
            is a `torch.dtype` object, then the return value is a torch tensor;
            otherwise, it is a numpy array.
    
        Raises
        ------
        ValueError
            If the arguments cannot be interpreted according to the formats
            descibed above.
        """
        return to_property(self, prop, dtype=dtype, null=null, filter=filter, mask=mask)
    def mask(self, m, indices=False, invert=False):
        """Returns a vector of labels of the vertices that are in the mask.

        `obj.mask(m)` yields the set of vertex labels from the given vertex-set
        object `obj` that correspond to the mask `m`.

        The mask `m` may take any of the following forms:
           * a list of vertex indices;
           * a boolean array (one value per vertex);
           * a property name, which can be cast to a boolean array
           * a tuple `(property, value)` where property is a list of values, one
             per vertex, and `value` is the value that must match in order for a
             vertex to be included (this is basically equivalent to the mask 
             `(property == value)`; note that property may also be a property
             name;
           * a tuple `(property, min, max)`, which specifies that the property
             must be between `min` and `max` for a vertex to be included
             (`min < p <= max`);
           * a tuple `(property, (val1, val2...))`, which specifies that the
             property must be any of the values in `(val1, val2...)` for a
             vertex to be included; or
           * `None`, indicating that all labels should be returned

        `obj.mask(m, ...)` is equivalent to `to_mask(obj, m, ...)` and to
        `to_mask((obj,m), ...)`.
        
        Parameters
        ----------
        m : mask-like
            A list of vertex indices, a boolean array, a property name, or
            tuple, as described above, that defines a valid mask.
        indices : boolean, optional
            Whether to return vertex indices (`True`) instead of vertex labels
            (`False`). The default is `False`.
        invert : boolean, optional
            Whether to return the inverse of the mask (`True`) or the mask
            itself (`False`). The default is `False`.

        Returns
        -------
        vector of int
            A vector of vertex labels for those vertices in the mask, or a
            vector of indices if `indices` is `True`.

        Raises
        ------
        ValueError
            If `m` cannot be interpreted as a valid mask instruction.
        """
        return to_mask(self, m, indices=indices)
def is_vset(v):
    """`True` if `v` is a `VertexSet`, otherwise `False`.

    `is_vset(v)` returns `True` if `v` is a `VertexSet` object and `False`
    otherwise. Note that topologies, tesselations, and meshes are all vertex
    sets. This function is a shortcut for `isinstance(v, VertexSet)`.

    Parameters
    ----------
    v : object
        The object whose quality as a `VertexSet` is to be tested.

    Returns
    -------
    boolean
        `True` if `isinstance(v, VertexSet)` otherwise `False`.
    """
    return isinstance(v, VertexSet)

def is_mask(m):
    """Returns `True` for valid masks and `False` otherwise.

    In order to be a valid mask, an object must be a vector of non-negative
    integers.

    Parameters
    ----------
    m : object
        The object whose quality as a mask is to be tested.

    Returns
    -------
    boolean
        `True` if `m` is a valid mask and `False` otherwise.

    See also: `like_mask()`.
    """
    return nym.is_numeric(m, 'int', ndim=1) && nym.all(nym.ge(m, 0))
def like_mask(m):
    """Determines if an object can be converted into a mask.

    An object is like a mask if it either is already a valid mask or if it is an
    object whose format is accepted by the `to_msak()` function and that can
    potentially be converted to a mask when paired with a `VertexSet` using the
    `to_mask()` function. Note that this function only determines if an argument
    has the appropriate format: it doees not determiine whether it can produce a
    valid or sensible mask when paired with aa particular object. The following
    objects return `True` from `like_mask()`:
      * a list of vertex indices;
      * a boolean array (one value per vertex);
      * a property name, which can be cast to a boolean array
      * a tuple `(property, value)` where property is a list of values, one per
        vertex, and `value` is the value that must match in order for a vertex
        to be included (this is basically equivalent to the mask `(property ==
        value)`; note that property may also be a property name;
      * a tuple `(property, min, max)`, which specifies that the property must
        be between `min` and `max` for a vertex to be included (`min < p <=
        max`);
      * a tuple `(property, (val1, val2...))`, which specifies that the
        property must be any of the values in `(val1, val2...)` for a vertex to
        be included; or
      * `None`, indicating that all labels should be returned

    Note that although `to_mask((vset, mask))` is equivalent to `to_mask(vset,
    mask)` and is a valid way to call the `to_mask()` function, the
    `like_mask()` function only considers the `mask` part of the above call to
    be mask-like. In otherwords, `like_mask(mask)` would return `True` where
    `like_mask((vset, mask))` would return `False`, even though `to_mask((vset,
    mask))` would succeed.

    Parameters
    ----------
    m : object
        The object whose quality as a mask is to be tested.

    Returns
    -------
    boolean
        `True` if `m` is or can be converted into a valid mask and `False`
        otherwise.

    See also: `is_mask()`, `to_mask()`.
    """
    if m is None: return True
    elif is_tuple(m):
        n = len(m)
        if   n == 0:                        return True
        elif n < 2 or n > 3:                return False
        elif is_str(m[0]):                  return True
        elif nym.arraylike(m[0], ndim=1)): return True
        else: return False
    elif is_str(m): return True
    elif is_map(m): return 
        if len(m) != 1: return False
        k = next(six.iterkeys(m))
        if not is_str(k): return False
        kl = k.lower()
        if lk == 'not':
            return like_mask(m[k])
        elif kl in ('and', 'or'):
            v = m[k]
            if not hasattr(v, '__iter__'): return False
            return all(like_mask(u) for u in v)
        else:
            return False
    else:
        # At this point, it could still be a boolean vector.
        return nym.is_numeric(m, 'bool', ndim=1)
def to_mask(obj, m=None, indices=None, invert=None):
    """Returns a vector of labels of the vertices that are in the mask.

    `to_mask(obj, m)` yields the set of vertex labels from the given vertex-set
    object `obj` that correspond to the mask `m`.

    The mask `m` may take any of the following forms:
      * a list of vertex indices;
      * a boolean array (one value per vertex);
      * a property name, which can be cast to a boolean array
      * a tuple `(property, value)` where property is a list of values, one per
        vertex, and `value` is the value that must match in order for a vertex
        to be included (this is basically equivalent to the mask `(property ==
        value)`; note that property may also be a property name;
      * a tuple `(property, min, max)`, which specifies that the property must
        be between `min` and `max` for a vertex to be included (`min < p <=
        max`);
      * a tuple `(property, (val1, val2...))`, which specifies that the
        property must be any of the values in `(val1, val2...)` for a vertex to
        be included; or
      * `None`, indicating that all labels should be returned

    `obj.mask(m, ...)` is equivalent to `to_mask(obj, m, ...)` and to
    `to_mask((obj,m), ...)`.
    
    Parameters
    ----------
    m : mask-like
        A list of vertex indices, a boolean array, a property name, or tuple, as
        described above, that defines a valid mask.
    indices : boolean, optional
        Whether to return vertex indices (`True`) instead of vertex labels
        (`False`). The default is `False`. If the object `obj` is not a
        `VertexSet` object (e.g., if it is a mapping), then this option is
        ignored.
    invert : boolean, optional
        Whether to return the inverse of the mask (`True`) or the mask itself
        (`False`). The default is `False`.

    Returns
    -------
    vector of int
        A vector of vertex labels for those vertices in the mask, or a vector of
        indices if `indices` is `True`.

    Raises
    ------
    ValueError
        If `m` cannot be interpreted as a valid mask instruction.
    """
    if not is_map(obj) and (is_tuple(obj) or is_list(objb)) and len(obj) < 3 and m is None:
        if   len(obj) == 1: obj = obj[0]
        elif len(obj) == 2: (obj, m) = obj
        else:
            (obj, m, kw) = obj
            kw = dict(kw)
            if invert is not None: kw['invert'] = invert
            if indices is not None: kw['indices'] = indices
            return to_mask(obj, m, **kw)
    if indices is None: indices = False
    if invert is None: invert = False
    if is_vset(obj):
        lbls = obj.labels
        idcs = obj.indices
        obj = obj.properties
    else:
        obj = pimms.itable(obj)
        lbls = np.arange(0, obj.row_count, 1, dtype=np.int)
        idcs = lbls
    if m is None:
        m = idcs
    elif is_tuple(m):
        if len(m) == 0: return idcs[[]]
        p = to_property(obj, m[0])
        if len(m) == 2 and hasattr(m[1], '__iter__'):
            m = nym.isin(p, m[1])
        elif len(m) == 2:
            m = nym.eq(p, m[1])
        elif len(m) == 3:
            m = nym.logical_and(nym.lt(m[1], p), nym.le(p, m[2]))
        else:
            raise ValueError("invalid mask: %s" % (m,))
    elif is_str(m):
        m = nym.asdtype(obj[m], 'bool')
    elif is_map(m):
        if len(m) != 1: raise ValueError('dicts used as masks must contain 1 item')
        (k,v) = next(six.iteritems(m))
        if not is_str(k): raise ValueError('key of dict-mask must be "or", "and", or "not"')
        k = k.lower()
        # Here, when we call down, we are accumulating, so we always want the indices; otherwise,
        # we run the risk of getting back into this call-stack and having labels but a need to
        # return indices. It's easy to go from indices to labels but hard to go the other way.
        if k == 'not':
            v = to_mask(obj, v, indices=True, invert=True)
        else:
            if not hasattr(v, '__iter__'):
                raise ValueError('value of dict mask with "or" or "and" key must be an iterator')
            # We don't pass down the invert option, as that is performed once at the end.
            v = [to_mask(obj, u, indices=True) for u in v]
            if k == 'and':
                m = reduce(curry(nym.intersect1, assume_unique=True), v)
            elif k == 'or':
                m = reduce(curry(nym.union1d, assume_unique=True), v)
            else:
                raise ValueError('key of dict-mask must be "or", "and", or "not"')
    # At this point, m should be a boolean array or a list of indices; possiibly, none of the above
    # conditions were met, but in this case, m must have been a boolean array or list of indices to
    # start with, per the function requirements.
    if invert:
        if nym.is_numeric(m, dtype='bool', shape=len(lbls)): m = ~m
        else: m = nym.setdiff1d(idcs, m)
    if len(m) == len(lbls):
        return idcs if indices else lbls
    else:
        return idcs[m] if indices else lbls[m]
def to_property(obj, prop=None, dtype=None, filter=None, mask=None, null=np.nan):
    """Extracts properties from an object and applies filters to them.
        
    `to_property(obj, prop)` yields the given property `prop` from `obj` after
    performing a set of filters on the property, as specified by the options. In
    the property array that is returned, the values that are considered outliers
    (data out of some range) are indicated by `inf`, and values that are not in
    the optionally-specified mask are given the value `nan`; these may be
    changed with the clipped and null options, respectively.
    
    `obj.propert(prop)` is equivalent to `to_property((obj, prop))` and to
    `to_property(obj, prop)`.

    The property argument `prop` may be either specified as a string (a property
    name in the object) or as a property vector. The weights option may also be
    specified this way. Additionally, the `prop` argument may be a list such as
    `['polar_angle', 'eccentricity']` where each element is either a string or a
    vector, in which case the result is a matrix of properties. Finally, `prop`
    may be a set of property names, in which case the return value is a
    `pimms.ITable` whose keys are the property names.

    Parameters
    ----------
    prop : str or property vector or list, optional
        A proprety name, a property vector, or a list of these, as detailed
        above. `prop` should be provided only if `obj` is not itself a tuple
        `(obj,prop)`. 
    mask : None or mask-like
        The specification of vertices that should be included in the property
        array; values are specified in the mask in accordance with the
        `to_mask()` function.
    null : object, optional
        The value marked in the array as out-of-mask. By default, this is `nan`.
    filter : function or None, optional
        A function to be passed the property array prior to being returned
        (after `null` and `clipped` values have been marked). The value returned
        by this function is returned instead of the property.
    dtype : dtype-like or None
        The type of the array that should be returned. `None`, the default,
        indicates that the type of the given property should be used.
        Otherwise, this must be a dtype-like. Both `numpy` and `torch` dtypes
        are supported.

    Returns
    -------
    array-like
        An array of properties, as described above. If the `dtype` parameter is
        a `torch.dtype` object, then the return value is a torch tensor;
        otherwise, it is a numpy array.

    Raises
    ------
    ValueError
        If the arguments cannot be interpreted according to the formats descibed
        above.
    """
    kw0 = dict(dtype=dtype, null=null, filter=filter, mask=mask)
    # Was a prop arg given, or is obj a tuple?
    if prop is None and is_tuple(obj) and len(obj) < 4:
        if   len(obj) == 2:
            return to_property(obj[0], obj[1], **kw0)
        elif len(obj) == 3:
            kw0.update(obj[2])
            return to_property(obj[0], obj[1], **kw0)
        else: raise ValueError('Bad input vector given to to_property()')
    # we could have been given a property alone or a map/vertex-set and a property
    if prop is None: raise ValueError('No property given to to_property()')
    # if it's a vertex-set, we want to note that and get the map
    if isinstance(obj, VertexSet): (vset, obj) = (obj,  obj.properties)
    elif pimms.is_map(obj):        (vset, obj) = (None, obj)
    elif obj is None:              (vset, obj) = (None, None)
    else: ValueError('Data object given to to_properties() is neither a vertex-set nor a mapping')
    # Now, get the property array, as an array
    n = self.vertex_count
    if is_str(prop):
        if obj is None: raise ValueError('a property name but no data object given to to_property')
        else: prop = obj[prop]
    elif is_set(prop):
        if pimms.is_lmap(obj):
            lazy_prop = lambda kk: (lambda:to_property(obj, kk, **kw0))
            return pimms.itable({k:(lazy_prop(k) if obj.is_lazy(k) else obj[k]) for k in prop})
        else:
            return pimms.itable({k:obj[k] for k in prop})
    elif (nym.arraylike(prop, ndim=2)
          or ((is_list(prop) or is_tuple(prop))
              and all(is_str(p) or nym.arraylike(p, ndim=1) for p in prop))):
        return nym.promote([to_property(obj, k, **kw0) for k in prop])
    elif not nym.arraylike(prop, ndim=1):
        raise ValueError('prop must be a property name or a vector or a combination of these')
    else: prop = nym.promote(prop)
    if dtype is not None: prop = nym.asdtype(prop, dtype)
    n = len(prop)
    # Process the mask.
    mask = None if mask is None else to_mask(obj, mask, indices=True, invert=True)
    # If the outliers is empty and the mask is everyone, then we can just return prop.
    if mask is not None and len(mask) > 0: prop[mask] = null
    return prop if filter is None else filter(prop)
