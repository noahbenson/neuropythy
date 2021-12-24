####################################################################################################
# Address Functions
# by Noah C. Benson

import six, pyrsistent as pyr
import numpy as np
from .. import math as nym
from .. import util as nyu
from pimms import (is_tuple, is_list, is_set, is_map, is_str, curry)

@pimms.immutable
class Address(nyu.ObjectWithMetaData):
    """Points represented as relative positions on meshes or splines.

    Address data in `neuropythy` represent points on a a mesh or spline that are
    expressed in terms relative to the vertices of the mesh or spline rather
    than in raw coordinates. For triangle meshes, points are stored as 2D
    barycentric coordinates that are paired with the vertex labels of the
    vertices that make up the triangle containing the point in question.

    The use of `Address` data is intentionally left flexible. The intended use
    is for barycentric coordinates within a triangle of a mesh, but the use of
    the `weight` matrix to address points outside of a triangle or simplex by
    allowing the weights to sum to a number other than 1, for example, is
    allowed. The weights matrix is not explicitly checked for structure aside
    from its shape.

    Parameters
    ----------
    simplex : matrix of int
        The matrix of vertex labels for the corners of the simplices that
        contain the points addressed. The `simplices` are always vertex labels,
        not vertex indices. If this is either a vector or a scalar, it is first
        converted into a matrix whose number of columns is equal to 1. If this
        is a matrix, then it must be a `d`x`n` matrix where `d` is the number of
        vertices in a simplex, and `n` is the number of points in the simplex
        collection.
    weight : matrix of float
        The relative weights on each of the vertices in `simplex`. This must
        be the same shape as `simplex`. For most weight schemes, the sum of
        each column must be 1, but this is not enforced.
    height : float or vector of floats or None, optional
        The height of the point relative to the mesh or structure on which
        addresses were calculated. This is an optional piece of data about
        addresses that is not explicitly interpreted by the `Address` class
        but that is intended to represent the cortical depth of a point within
        the cortical sheet. The default is `None`.
    vset : VertexSet or None, optional
        The vertex set object from which the points are addressed or `None`.
    meta_data : mapping or None, optional
        Optional meta-data dictionary.

    Attributes
    ----------
    simplex : int or vector of int or matrix of int
        The read-only matrix or of vertex labels, or a single vertex label
        representing the simplices of the mesh or spline in which the address
        was found. If a point could not be addressed in the object that the
        `Address` reference, then the vector label is encoded as `-1`. If
        `simplex` is a vector, it is assumed to be a single simplex, not a
        vector of unitary simplices.
    weight : float or vector of floats matrix of floats
        A read-only matrix whose shape is equal to that of `simplices` and whose
        columns all sum to 1. The weights are the barycentric coordinates for
        triangles or equivalent weightings for any other simplex rank. If a
        point could not be addresses in the reference object, then `nan` values
        are encoded in that column.
    height : float or vector of floats or None
        The height parameters, if any.
    vset : VertexSet or None
        The `VertexSet` object from which the addresses were found, if provided.
    meta_data : mapping
        Any meta-data attached to the object.
    found_mask : boolean or vector of booleans
        A boolean or vector of booleans indicating for each point in the address
        space that was found in the object that the addresses reference. A point
        is considered not found whe the `weight` for any of its reference
        vertices is `nan`. If all vertices are found, then this is
        `slice(None)`.
    all_found : boolean
        `True` if all vertices were found in the addressing object and `False`
        otherwise (see als `found_mask`).
    """
    def __init__(self, simplex, weight, height=None, vset=None, meta_data=None):
        nyu.ObjectWithMetaData.__init__(self, meta_data=meta_data)
        self.simplex = simplex
        self.weight = weight
        self.height = height
        self.vset = vset
    @pimms.param
    def simplex(s):
        "The `ncorners`x`npoints` matrix of simplices containing the adressed points."
        s = nym.to_readonly(s)
        assert nym.is_numeric(s, 'int', ndims=(0,1,2)), \
            "simplex must be an integer array with ndims <= 2"
        return s
    @pimms.param
    def weight(w):
        "The `ncorners`x`npoints` matrix of weights on each of the vertices in `simplex`."
        w = nym.to_readonly(w)
        assert nym.is_numeric(w, '<=real', ndims=(0,1,2)), \
            "weight must be a real-valued array with ndims <= 2"
        return w
    @pimms.require
    def simplex_weight_same_shape(simplex, weight):
        "Requires that the simplex and weight arrays have the same shape."
        assert simplex.shape == weight.shape, \
            "simplex and weight must have the same shape"
        return True
    @pimms.value
    def simplex_matrix(simplex):
        s = simplex.view()
        if   s.ndim == 1: s.shape = (w.shape[0],1)
        elif s.ndim == 0: s.shape = (1,1)
        return s
    @pimms.value
    def weight_matrix(weight):
        w = weight.view()
        if   w.ndim == 1: w.shape = (w.shape[0],1)
        elif w.ndim == 0: w.shape = (1,1)
        return w
    @pimms.param
    def height(h):
        "The optional height value or vector. Interpretation of this value is not enforced."
        return h
    @pimms.param
    def vset(v):
        "The optional vertex set. Interpretation of this value is not enforced."
        return v
    @pimms.value
    def found_mask(simplex, weight):
        "A mask of all vertices found in the addressed object."
        if   simplex.ndim == 0: (s,w) = (nym.reshape(simplex, (1,1)), nym.reshape(weight, (1,1)))
        elif simplex.ndim == 1: (s,w) = (nym.reshape(simplex, (-1,1)), nym.reshape(weight, (-1,1)))
        ok = nym.all(nym.isfinite(weight), axis=0) & nym.all(nym.ge(simplex, 0), axis=0)
        if nym.all(ok): return slice(None)
        else: return nym.as_readonly(ok)
    @pimms.value
    def all_found(found_mask):
        "A shortcut for `isinstance(found_mask, slice)."
        return isinstance(found_mask, slice)
    def interpolate(self, prop, has_height=None, vset=Ellipsis, grayheight=True,
                    method=None, height=Ellipsis, strict=False, null=np.nan):
        """Interpolates a property at a set of points specified in an address.
        
        `addr.interpolate(prop)` returns the result of interpolating the given
        property `prop` at point or points that are specified by the given
        address `addr`. If `addr` contains height values and prop is a map of
        layer values whose keys are numbers, then the addresses respect and are
        interpolated from the appropriate layers. The `height` parameter
        controls the layer from which values are interpolated if no heights are
        included in `addr`, and the `has_height` parameter can be used to force
        a certain behavior or raise an error. The `grayheight` parameter
        controls whether the heights are interpreted using `to_grayheight()`.
        
        The address data `addr` is related too the property `prop` in that the
        vertex labels in the cells of the `simiplex` array of the address data
        correspond to a cell in the `prop` vector (or in each of `prop`'s
        columns, if prop is a dictionary of vectors). In other words, the
        address data must have been calculated from the set of points at which
        we are interpolating---in most interpolation function this set of points
        (or this topology/mesh object) is required, but the addresses in `addr`
        suffice here. The `vset` parameter may be used to override the
        vertex-set whose index is used to translate the vertex labels of the
        address into property indices.
        
        Parameters
        ----------
        prop : array-like or mapping
            The property to be interpolated onto the points encoded by
            `addr`. This must be an array-like object (a `numpy` array, `torch`
            tensor, or something that can be converted into these types) whose
            last dimension is equivalent to the number of vertices in the mesh
            from which interpolation occurs (i.e., the mesh on which the `addr`
            address data were calculated) or a mapping whose values are all such
            array-like objects and whose keys are height values (though see also
            the `grayheight` parameter). When `prop` is a mapping, the
            interpolation occurs either at the heights specified in the `addr`
            data or at the height specified by the `height` argument, if `addr`
            does not include height data. If neither is given, then the mean of
            the min and max heights are used.
        method : interpolation method-like, optional
            The interpolation method to use, which may be any value that, when
            filtered by the `to_interpolation_method()` function results in either
            `'heaviest'` or `'linear'` interpolations. The `'heaviest'`
            interpolation method is similar to nearest-neighbor interpolation except
            that it always chooses the nearest mesh vertex of a point from among the
            corners of the triangle containing the point whereas true
            nearest-neighbor interpolation might pick a closer vertex of a
            neighboring triangle. Linear interpolation interpolated linearly within
            the triangle containing the point. True nearest-neighbor interpolation
            is not possible using this function (the data necessary to perform such
            interpolation is not provided to this function). For true
            nearest-neighbor interpolation you must use a mesh's `interpolate()`
            method. If `None` is given (the default), then `'linear'` is used for
            all real and complex (inexact) numbers and `'heaviest'` is used for all
            others.
        height : float or vector of floats or None, optional
            If the `addr` data do not contain information about height and the
            `prop` data contain property data for multiple heights, including,
            at minimum, 0.0 and 1.0, then the depth at which interpolation
            occurs is provided by `height`. If `height` is `None` (the default),
            then the mean of the min and max heights in the `prop` map are used.
        strict : boolean, optional
            If `True`, an error is raised if any address coordinates have non-finite
            values (i.e., were "out-in-region" values); otherwise the associated
            interpolated values are silently set to `null`. The default is `False`.
        null : object, optional
            The value given to any "out-of-region" value found in the addresses if
            `strict` is `False`. The default is `nan`.
        vset : VertexSet, optional
            If the addresses werer calculated in reference to a mesh that is a
            flatmap or submesh of another mesh, then the vertex labels in the
            `addr` data's `'simplex'` matrix will not match up to the `prop`
            dimensions. In this case, `vset` may be the `VertexSet` object,
            which can translate from vertex labels to correct vertex
            indices. The default, `Ellipsis`, uses `addr.vset`. If `vset` is
            `None`, this results in no translation, meaning that `prop` must be
            from a mesh that has not been subsampled.
        
        Returns
        -------
        array-like
            An array or tensor of values interpolated from the given properrties
            (`prop`) onto the points encoded in the given address data (`addr`).
        
        Raises
        ------
        ValueError
            If any of the arguments cannot be interpreted as matching their required
            types or forms.
        """
        # Argument Parsing #########################################################################
        # Parse the index, if any.
        #TODO: Here, have to fix this functoin
        if index is not None:
            from neuropythy.geometry import (is_tess, is_mesh, is_topo)
            if   is_mesh(index): index = index.tess
            elif is_topo(index): index = index.tess
            if is_tess(index): index = index.index
            faces = index(faces)
        # Parse the properties into an array of depths and a list of the properties at those depths.
        if nym.arraylike(prop, shape=(-1,Ellipsis)):
            prop = promote(prop)
            n = prop.shape[-1]
            prop = {0.0:prop, 1.0:prop}
        elif not is_map(prop):
            raise ValueError('bad property arg of type %s' % type(prop))
        else:
            prop = to_cortical_depth(prop) # convert keys to floats
            if 0.0 not in prop or 1.0 not in prop:
                raise ValueError("property mappings must at minimum contain white and pial layers")
            prop = {k:promote(v) for (k,v) in six.iteritems(prop)}
            n = prop[0.0].shape[-1]
            for v in six.itervalues(prop):
                if v.shape[-1] != n:
                    raise ValueError("property mappings must contain arrays whose last dims match")
        # We now have a valid property map; convert to sorted keys and values.
        ks = nym.argsort(list(prop.keys()))
        vs = [prop[k] for k in ks] # Keep as a list because they may not actually have the same shapes.
        # Get faces and barycentric coordinates and cortical depth.
        (faces, (a,b,h)) = address_data(addr, 3, surface=surface, strict=strict)
        # Let's promote everything together now!
        promotions = promote(faces, a, b, h, *vs)
        vs = promotions[4:]
        (faces, a, b, h) = promotions[:4]
        # Calculate the barycentric c weight.
        c = 1.0 - a - b
        # Now we can parse the interpolation method.
        if method is None:
            if is_numeric(vs[0], '>int'): method = 'linear'
            else: method = 'heaviest'
        else:
            method = to_interpolation_method(method)
            if method not in ('linear', 'heaviest'):
                raise ValueError(f"method {method} is not supported for address interpolation")
        # Where are the nans? (No need to raise an error: strict will have done that above.)
        bad = nym.where(~nym.isfinite(a))[0]
        # Add infinite boundaries to our layers for depths outside of [0,1].
        ks = nym.cat([[-nym.inf], ks, [nym.inf]])
        vs = [vs[0]] + vs + [v[-1]]
        # where in each column is the height.
        q = nym.gt(h, nym.reshape(ks, (-1,1)))
        # qs[0] is always True, qs[-1] is always False; the first False indicates h's layer
        wh1 = nym.argmin(q, axis=0) # get first occurance of False; False means h >= the layer
        wh0 = wh1 - 1
        h = (h - ks[wh0]) / (ks[wh1] - ks[wh0])
        h[wh0 == 0] = 0.5
        h[wh1 == (len(ks) - 1)] = 0.5
        hup = (h > 0.5) # the heights above 0.5 (closer to the upper than the lower)
        # okay, figure out the actual values we use:
        vals = vs[:,faces]
        each = nym.arange(len(wh1))
        vals = nym.transpose(vals, (0,2,1))
        lower = nym.tr(vals[(wh0, each)])
        upper = nym.tr(vals[(wh1, each)])
        if method == 'linear':
            vals = lower*(1 - h) + upper*h
        else:
            ii = h > 0.5
            vals[:,ii] = upper[:,ii]
        # make sure that we only get inf/nan values using nearest (otherwise they spread!)
        ii = nym.where(~nym.isfinite(lower) & hup)
        vals[ii] = upper[ii]
        ii = nym.where(~nym.isfinite(upper) & ~hup)
        vals[ii] = lower[ii]
        # now, let's interpolate across a/b/c;
        if method == 'linear':
            w = nym.promote([a,b,c])
            ni = nym.where(~nym.isfinite(vals))
            if len(ni[0]) > 0:
                w[ni] = 0
                vals[ni] = 0
                ww = nym.zinv(nym.sum(w, axis=0))
                w *= ww
            else: ww = None
            res = nym.sum(vals * w, axis=0)
            if ww is not None: res[nym.isclose(ww, 0)] = null
        else:
            wh = nym.argmax([a,b,c], axis=0)
            res = vals[(wh, nym.arange(len(wh)))]
        if len(bad) > 0: res[bad] = null
        return res

def is_address(data, dims=None, all_found=None, has_height=None):
    """`True` if given an `Address` object, otherwise `False`.

    See `Address`.
    
    Parameters
    ----------
    data : object
        An object whose quality as an `Address` is to be assessed.
    dims : None or int, optional
        The required dimensionality of the simplices. If `None` (the default),
        then no dimension requirement is applied.
    has_height : boolean or None, optional
        Whether the address is required to have a height that is not `None`. If
        `None`, then no requirement is put on the height. If `True` or `False`,
        then `height is not None` must equal `has_height`.
    all_found : boolean or None, optional
        Whether the address is required to have all points found in the
        address's reference object. If `None`, then no requirement is put on the
        number of points found. If `True` or `False`, then `data.all_found` must
        equal `all_found`.

    Returns
    -------
    boolean
        `True` if `data` contains valid neuropythy address data and `False`
        otherwise.
    """
    if not isinstance(data, Address): return False
    if dims is not None and dims != data.simplex_matrix.shape[0]: return False
    if has_height is not None and has_height == (data.height is None): return False
    if all_found is not None and all_found != data.all_found: return False
    return True
def to_address(obj, dims=None, all_found=None, has_height=None):
    """Converts the given input into an `Address` and returns it.

    `to_address(obj)` returns `obj` if `obj` is already an `Address` object and
    otherwise returns an `Address` object representing `obj`. If `obj` cannot be
    interpreted as an address, then an error is raised.

    In order to be converted into aan address, an object must be address-like.
    An object is address-like if it is an address, if it is a dictionray
    containing the keys 'simplex' and 'weight', or if it is a tuple
    `(simplex,weight)` where in all cases, the `simplex` is integer-valued, the
    `weight` is real-valued, and they have the shape.

    Parameters
    ----------
    obj : object
        An object whose quality as an `Address` is to be assessed.
    dims : None or int, optional
        The required dimensionality of the simplices. If `None` (the default),
        then no dimension requirement is applied.
    has_height : boolean or None, optional
        Whether the address is required to have a height that is not `None`. If
        `None`, then no requirement is put on the height. If `True` or `False`,
        then `height is not None` must equal `has_height`.
    all_found : boolean or None, optional
        Whether the address is required to have all points found in the
        address's reference object. If `None`, then no requirement is put on the
        number of points found. If `True` or `False`, then `result.all_found`
        must equal `all_found`.

    Returns
    -------
    Address
        An address object.

    Raises
    ------
    ValueError
        If the given object cannot be converted into an `Address`.
    """
    if not is_address(obj):
        if is_map(obj) and 'simplex' in obj and 'weight' in obj:
            (s,w,h,v,m) = [obj.get(k) for k in ['simplex','weight','height','vset','meta_data']]
            obj = Address(s, w, height=height, vset=v, meta_data=m)
        elif is_tuple(obj) and len(obj) == 2:
                (s,w) = obj
                obj = Adderss(s, w)
        elif is_tuple(obj) and len(obj) == 3:
                (s,w,kw) = obj
                obj = Adderss(s, w, **kw)
        else:
            raise ValueError(f"cannot convert object to Adress: {obj}")
    # Make sure the various checks match!
    if not is_address(obj, dims=dims, has_height=has_height, all_found=all_found):
        raise ValueError("given address data failed requirements")
    return obj
