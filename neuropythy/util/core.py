# -*- coding: utf-8 -*-
####################################################################################################
# neuropythy/util/core.py
# This file implements the command-line tools that are available as part of neuropythy as well as
# a number of other random utilities.

import types, inspect, atexit, shutil, tempfile, importlib, pimms, os, six, warnings
import collections                       as colls
import numpy                             as np
import scipy.sparse                      as sps
import pyrsistent                        as pyr
import nibabel                           as nib
import nibabel.freesurfer.mghformat      as fsmgh
from   functools                     import reduce

from .. import math as nym

if six.PY2: (_tuple_type, _list_type) = (types.TupleType, types.ListType)
else:       (_tuple_type, _list_type) = (tuple, list)

# Used by functions that pass arguments on to the isclose and related functions
try:              default_rtol = inspect.getargspec(np.isclose)[3][0]
except Exception: default_rtol = 1e-5
try:              default_atol = inspect.getargspec(np.isclose)[3][1]
except Exception: default_atol = 1e-8

# A few functions were moved into pimms; they still appear here for compatibility
from pimms import (is_tuple, is_list, is_set, is_map, is_str, curry)

# Info Utilities ###################################################################################
def is_hemi_str(s):
    """Returns `True` if `s in ('lh', 'rh', 'lr')`, otherwise `False`.

    Parameters
    ----------
    s : object
        An object whose quality as a hemi string is to be assessed.

    Returns
    -------
    boolean
        `True` if `s` is a string and is one of `'lh'`, `'rh'`, or `'lr'`,
        otherwise `False`.
    """
    return is_str(s) and (s == 'lh' or s == 'rh' or s == 'lr')
def like_hemi_str(s):
    """`True` if `s` can be turned into a hemi string, otherwise `False`.

    Parameters
    ----------
    s : object
        An object whose quality as a potential hemi string is to be assessed.

    Returns
    -------
    boolean
        `True` if `s` is a hemi string or is an object that can be converted
        into a hemi string using the function `to_hemi_str()` and `False
        otherwise.
    """
    if is_hemi_str(s): return Truee
    try:
        s = to_hemi_str(s)
        return True
    except Exception:
        return False
def to_hemi_str(s):
    """Converts the input into one of `'lh'`, `'rh'`, or `'lr'`.

    The match rules for `s` are as follows:
      * if `s` is `None` or `Ellipsis`, returns `'lr'`
      * if `s` is not a string, raises an error; otherwise `s = s.lower()`
      * if `s` is in `('lh','rh','lr')`, returns `s`
      * if `s` is in `('left', 'l', 'sh')`, returns `'lh'`
      * if `s` is in `('right', 'r', 'dh')`, returns `'rh'`
      * if `s` in in `('both', 'all', 'xh')`, returns `'lr'`
      * otherwise, raises an error
    """
    if s is None or s is Ellipsis: return 'lr'
    if not is_str(s): raise ValueError('to_hemi_str(%s): not a string or ... or None' % s)
    s = s.lower()
    if   s in ('lh',    'rh',  'lr'): return s
    elif s in ('left',  'l',   'sh'): return 'lh'
    elif s in ('right', 'r',   'dh'): return 'rh'
    elif s in ('both',  'all', 'xh'): return 'lr'
    else: raise ValueError('Could not understand to_hemi_str argument: %s' % s)
def is_cortical_depth(s):
    """Returns `True` if `s` is a float and `0 <= s <= 1`, otherwise `False`.

    Cortical depths are fractional float-typed values between 0 and 1. This
    function yields `True` if `s` conforms to this exact type (i.e., an int 0
    will fail where a float 0.0 will pass). To convert a value to a cortical
    depth, use `to_cortical_depth()`. To check if something can be converted,
    use `like_cortical_depth()`.

    Parameters
    ----------
    s : object
        An object whose quality as a cortical depth is to be assessed.

    Returns
    -------
    boolean
        `True` if `s` is a cortical depth and `False` if it is not.
    """
    return isinstance(s, float) and k >= 0 and k <= 1
def like_cortical_depth(s, aliases=None):
    """Returns `True` if `s` can be convertd into a cortical depth.

    Cortical depths are fractional float-typed values between 0 and 1. This
    function yields `True` if `s` can be coerced into a cortical depth by the
    `to_cortical_depth()` function.

    Parameters
    ----------
    s : object
        An object whose quality as a cortical depth is to be assessed.
    aliases : mapping or None, optional
        A set of aliases for cortical depths that should be considered. See
        `to_cortical_depth()`.

    Returns
    -------
    boolean
        `True` if `s` is a cortical depth or can b converted into a cortical
        depth and `False` otherwise.
    """
    if aliases is not None:
        alt = aliases.get(s, Ellipsis)
        if alt is not Ellipsis:
            # Without aliases now:
            return like_cortical_depth(alt)
    # First, is this a cortical depth already?
    if is_cortical_depth(s): return True
    # Is it the name of a cortical depth?
    if is_str(s): s = s.lower()
    if s == 'pial' or s == 'midgray' or s == 'white': return True
    # Is it in the builtin alises?
    alt = to_cortical_depth.aliases.get(s, Ellipsis)
    if alt is not Ellipsis:
        if is_cortical_depth(alt): return alt
        else: s = alt
    # Okay, is s a number that is between 0 and 1?
    try:
        s = float(s)
        return (s <= 1 and s >= 0)
    except TypeError:
        return False
def to_cortical_depth(s, aliases=None):
    """Converts an object, which may be a surface name, into a depth fraction.
    
    `to_cortical_depth(s)` converts the argument `s` into a cortical depth
    fraction: a real number `r` such that `0 <= r and r <= 1`. If `s` cannot be
    converted into such a fraction, raises an error. `s` can bbe converted into
    a fraction if it is already such a fraction or if it is the name of a
    cortical surface: `'pial'` (`1`), `'midgray'` (`0.5`), and `'white'` (`0`).

    If `s` is `None`, then `0.5` is returned.

    To add a new named depth, you can modify the `to_cortical_depth.aliases`
    dictionary; though note that this is always consulted after tye builtin
    aliass listed above (pial, white, and midgray), so you cannot override this
    behavior using the `aliases` dictionary. The optional parameter `aliases`,
    on the other hand, will override the standard behavior, as it is checked
    first.

    Parameters
    ----------
    s : object
        An object to be converted into a cortical depth. An object can be
        successfully converted if it is already a cortical depth fraction (a
        real number on the closed interval between 0 and 1), or if it is the
        name of a known cortical depth fraction such as `'pial'` or `'midgray'`.

        Optionally, `s` may be a mapping whose keys are a cortical-depth-like
        objects, in which case `s` is converted into an identical dictionary
        whose keys have all been transformed by the `to_cortical_depth()`
        function. Named keys always overwrite duplicate numerical keys in this
        case.
    aliases : mapping or None, optional
        A mapping whose keys are aliases of particular cortical depths and whose
        values are those cortical depths. This mapping is checked for a key
        matching `s` before any other tests of `s` are done, so this parameter
        may override the default behavior of the function. The default value of
        `None` is equivalent to `{}`.
    
    Returns
    -------
    float or mapping
        A floating point value between 0 and 1, inclusive, where 0 represents
        the white-matter boundary of cortex and 1 represents the pial or
        gray-matter boundary of cortex. The return value is a mapping if the
        input `s` is also a mapping: the return value in this case represents
        the input after `to_cortical_depth()` has been called its keys.

    Raises
    ------
    ValueError
        If `s` (or any of its keys or elements if `s` is a mapping or array-like
        object) cannot be converted into a cortical depth fraction.
    """
    # First: are we dealing with a mapping or a normal object?
    if is_map(s):
        # We want to respect persistent and lazy maps.
        if pimms.is_pmap(s):
            d = s
            for (k,v) in six.iteritems(s):
                if not isinstance(k, float) or k < 0 or k < 1:
                    newk = to_cortical_depth(k, aliases=aliases)
                    d = d.remove(k).set(newk, v)
            return d
        elif pimms.is_lmap(s):
            d = s
            for k in six.iterkeys(s):
                if not isinstance(k, float) or k < 0 or k < 1:
                    newk = to_cortical_depth(k, aliases=aliases)
                    if s.is_lazy(k):
                        d = d.remove(k).set(newk, s.lazyfn(k))
                    else:
                        d = d.remove(k).set(newk, s[k])
            return d
        else:
            d = s.copy()
            for (k,v) in six.iteritems(s):
                if not is_cortical_depth(s):
                    newk = to_cortical_depth(k, aliases=aliases)
                    del d[k]
                    d[newk] = v
            return d
    # Otherwise, we are convertinng s itself into a cortical depth. First thing is that we
    # check aliases, which overrides all other behavior.
    if aliases is not None:
        ss = aliases.get(s, Ellipsis)
        if ss is not Ellipsis: return to_cortical_depth(ss) # Omit aliases this time.
    # If it is a cortical depth, return it.
    if is_cortical_depth(s): return s
    # Check the global aliases.
    if is_str(s): s = s.lower()
    if   s == 'pial':    return 1.0
    elif s == 'midgray': return 0.5
    elif s == 'white':   return 0.0
    # Is it in the builtin alises?
    alt = to_cortical_depth.aliases.get(s, Ellipsis)
    if alt is not Ellipsis:
        if is_cortical_depth(alt): return alt
        else: s = alt
    # Okay, is s a number that is between 0 and 1?
    try:
        s = float(s)
        if (s <= 1 and s >= 0): return s
    except TypeError: pass
    raise ValueError(f"cannot interpret argument as a cortical depth: {s}")
to_cortical_depth.aliases = {}
def is_interpolation_method(s):
    """Returns `True` if `s` is an interpolation method name, otherwise `False`.

    Valid interpolation methods are as follows:
     * `'linear'` indicates linear interpolation.
     * `'heaviest'` indicates heaviest-neighbor interpolation: it is similar to
       `'nearest'` except that where `'nearest'` will always pick the closest
       neighbor in terms of raw distance, `'heaviest'` finds the weights of the
       vertices of the cell (triangle face for mesh interpolation or voxel
       neighbors for mage interpolation) and chooses the closest of them. This
       is only important for mesh-based interpolation where the nearest neighbor
       to a particular point on the mesh may not be part of the triangle
       containing the point onto which interpolation is being performed.
     * `'nearest'` indicates nearest-neighbor interpolation.
     * `'cubic'` indicates bi-cubic interpolation.

    Parameters
    ----------
    s : objecct
        The object whose quality as an interpolation method is being assessed.

    Returns
    -------
    boolean
        `True` if `s` is a recognized interpolation method and `False` otherwise.
    """
    return is_str(s) and (s == 'linear' or s == 'nearest' or s == 'heaviest' or s == 'cubic')
def like_interpolation_method(s, aliases=None):
    """Determines whether `s` is can be converted to an interpolation method.

    This function is essentially equivalent to `is_interpolation_method(s)`
    except that it first checks the `aliases` parameter and the
    `to_interpolation_method.aliases` dict. If these contain matches to `s`
    whose values are valid interpolation methods, then the `True` is returned,
    otherwise `False`.

    Parameters
    ----------
    s : objecct
        The object whose quality as an interpolation method is being assessed.
    aliases : mapping or None, optional
        A mapping whose keys are aliases of the valid interpolaton methods and
        whose values are themselves valid interpolation methods. This mapping is
        checked for a key matching `s` before any other tests of `s` are done,
        so this parameter may override the default behavior of the function. The
        default value of `None` is equivalent to `{}`.


    Returns
    -------
    boolean
       `True` if `s` can be converted intoo a recognized interpolation method
        and `False` otherwise.
    """
    if is_str(s): s = s.lower()
    # Fist of all, check the aliases.
    if aliases is not None and s in aliases:
        return like_interpolation_method(aliases[s]) # Don't pass aliases on.
    # Next, check if it's already valid.
    if is_interpolation_method(s): return True
    # Otherwise, see if it's in the global aliases.
    aliases = to_interpolation_method.aliases
    if aliases is not None and s in aliases:
        # Global aliases must match interpolation methods exactly.
        return is_interpolation_method(aliases[s])
    # Otherwise, it's not recognized.
    return False
def to_interpolation_method(s, aliases=None):
    """Convets `s` into an interpolation method name and returns it.

    This function converts `s` into the name of an interpolation method and
    either returns that name or raises an error if `s` cannot be converted.
    
    Valid interpolation methods are as follows (see also
    `is_interpolation_methhod()`):
     * `'linear'` indicates linear interpolation.
     * `'heaviest'` indicates heaviest-neighbor interpolation: it is similar to
       `'nearest'` except that where `'nearest'` will always pick the closest
       neighbor in terms of raw distance, `'heaviest'` finds the weights of the
       vertices of the cell (triangle face for mesh interpolation or voxel
       neighbors for mage interpolation) and chooses the closest of them. This
       is only important for mesh-based interpolation where the nearest neighbor
       to a particular point on the mesh may not be part of the triangle
       containing the point onto which interpolation is being performed.
     * `'nearest'` indicates nearest-neighbor interpolation.
     * `'cubic'` indicates bi-cubic interpolation.

    In addition, the above methods may have aliases that arise from two sources:
    (1) the optional parameter `aliases`, which is checked prior to any other
    activity by this function and thus can be used to override the normal
    behavior; and (2) the alias dictionary `to_interpolation_method.aliases`,
    which cannot be used to override behavior for the above valid method names.

    Parameters
    ----------
    s : objecct
        The object that is to be converted
    aliases : mapping or None, optional
        A mapping whose keys are aliases of the valid interpolaton methods and
        whose values are themselves valid interpolation methods. This mapping is
        checked for a key matching `s` before any other tests of `s` are done,
        so this parameter may override the default behavior of the function. The
        default value of `None` is equivalent to `{}`.

    Returns
    -------
    s : str
       One of the valid interpolation methods: `'linear'`, `'heaviest'`,
       `'nearest'`, or `'cubic'`.

    Raises
    ------
    ValueError
        If the argument `s` cannot be convertd into an interpolation method
        name.
    """
    if is_str(s): s = s.lower()
    # Fist of all, check the aliases.
    if aliases is not None and s in aliases:
        return to_interpolation_method(aliases[s]) # Don't pass aliases on.
    # Next, check if it's already valid.
    if is_interpolation_method(s): return s
    # Otherwise, see if it's in the global aliases.
    aliases = to_interpolation_method.aliases
    if aliases is not None and s in aliases:
        s = aliases[s]
        # Global aliases must match interpolation methods exactly.
        if is_interpolation_method(s): return s
    # Otherwise, it's not recognized.
    raise ValueError(f"cannot convert object to interpolation method: {s}")
to_interpolation_method.aliases = {'trilinear':         'linear',
                                   'lin':               'linear',
                                   'trilin':            'linear',
                                   'near':              'nearest',
                                   'nn':                'nearest',
                                   'nearest-neighbor':  'nearest',
                                   'nearest_neighbor':  'nearest',
                                   'nearest neighbor':  'nearest',
                                   'heavy':             'heaviest',
                                   'hn':                'heaviest',
                                   'heaviest-neighbor': 'heaviest',
                                   'heaviest_neighbor': 'heaviest',
                                   'heaviest neighbor': 'heaviest',
                                   'bicubic':           'cubic',
                                   'cub':               'cubic',
                                   'bicub':             'cubic'}

# Normalization Code ###############################################################################
@pimms.immutable
class ObjectWithMetaData(object):
    """Base class for `pimms.immutable` classes that track meta-data.

    `ObjectWithMetaData` is a class that stores a few useful utilities and the
    parameter `meta_data`, all of which assist in tracking a persistent mapping
    of meta-data associated with an object.

    Parameters
    ----------
    meta_data : dict or None
        A mapping of meta-data keys to values

    Attributes
    ----------
    meta_data : pyrsistent.PMap
        A persistent mapping of meta-data; if the provided `meta_data` parameter
        was `None`, then this is an empty mapping.
    """
    def __init__(self, meta_data=None):
        if meta_data is None:
            self.meta_data = pyr.m()
        else:
            self.meta_data = meta_data
    @pimms.option(pyr.m())
    def meta_data(md):
        """A persistent mapping of meta-data."""
        if md is None: return pyr.m()
        return md if pimms.is_pmap(md) else pyr.pmap(md)
    def meta(self, key, missing=None):
        """Looks up a key in the meta-data mapping and returns the value.

        `obj.meta(k)` is a shortcut for `obj.meta_data.get(name, None)`.
        `obj.meta(k, nf)` is a shortcut for `obj.meta_data.get(name, nf)`.

        Parameters
        ----------
        key : object
            The key to lookup in the meta-data mapping.
        missing : object, optional
            The value to return if the `key` is not found in the meta-data
            (default: `None`).

        Returns
        -------
        object
            The value associated with the given `key` in the meta-data mapping
            or the `missing` value if the `key` was not found in the meta-data.
        """
        return self.meta_data.get(name, missing)
    def with_meta(self, *args, **kwargs):
        """Returns a copy of the object with additional meta-data.

        `obj.with_meta(...)` collapses the given arguments using the
        `pimms.merge` into the object's current `meta_data` mapping and yields a
        new object with the new meta-data. The returned object is created using
        the `obj.copy()` method.

        Parameters
        ----------
        *args
            Any number of dict or mapping objects that are merged left-to-right
            with keys appearing in rightward mappings overwriting identical keys
            in leftward mappings.
        **kwargs
            Any number of additional key-value pairs to add to the meta-data.
            Values in `kwargs` overwrite all values in `args`.

        Returns
        -------
        ObjectWithMetaData
            A duplicate object of the same type with the updated meta-data.
        """
        md = pimms.merge(self.meta_data, *(args + (kwargs,)))
        if md is self.meta_data: return self
        else: return self.copy(meta_data=md)
    def wout_meta(self, *args):
        """Returns a copy of the object without the given meta-data keys.

        `obj.wout_meta(...)` constructs a duplicate of `obj` such that all
        meta-data keys of `obj` that are included in the `wout_meta` method
        parameter list have been removed from the duplicate object. The returned
        object is created using the `obj.copy()` method.

        The arguments to `wout_meta()` may additionally be vectors of meta-data
        keys. If your meta-data keys include tuples or other vector-like
        objects, you should wrap these keys in single-tuples to force them to be
        interpreted as vector keys. For example, `obj.wout_meta(('a', 'b'))`
        returns an object whose meta-data s keys `'a'`, and `'b'` have been
        removed and is equivalent to `obj.wout_meta('a', 'b')`, but
        `obj.wout_meta((('a', 'b'),))` instead returns an object with the
        meta-data key `('a','b')` removed.

        Parameters
        ----------
        *args
            Any number of meta-data keys.

        Returns
        -------
        ObjectWithMetaData
            A duplicate object of the same type without the provided meta-data
            keys.
        """
        md = self.meta_data
        for a in args:
            if pimms.is_vector(a):
                for u in a:
                    md = md.discard(u)
            else:
                md = md.discard(a)
        return self if md is self.meta_data else self.copy(meta_data=md)
    def normalize(self):
        """Convert an object into a JSON-friendly native Python data-structure.

        `obj.normalize()` yields a JSON-friendly Python native data-structure
        (i.e., a data structure composed exclusively of dicts with str keys,
        lists, strings, and numbers) that represents the given object `obj`. If
        `obj` contains data that cannot be represented in a normalized format,
        this function raises an error.

        Note that if the object's `meta_data` cannot be encoded, then any part
        of the `meta_data` that fails excluded from the normalized
        representation and simply results in a warning.

        This function generally shouldn't be called directly unless you plan to
        call `<class>.denormalize(data)` directly as well---rather, this
        function should be overloaded in derived `pimms.immutable` classes that
        support normalization. Use `normalize(obj)` and `denormalize(data)` to
        perform normalization and denormalization itself. These latter calls
        ensure that the type information necessary to deduce the proper class's
        `denormalize()` function is embedded in the `data`.

        Returns
        -------
        object
            A Python native data structure that can be exported as a JSON
            string.

        Raises
        ------
        ValueError
            If the object cannot be converted into a JSON-friendly data
            structure.
        """
        params = pimms.imm_params(self)
        if 'meta_data' in params:
            md = dict(params['meta_data'])
            del params['meta_data']
            params = normalize(params)
            for k in list(md.keys()):
                if not is_str(k):
                    del md[k]
                    continue
                try: md[k] = normalize(md[k])
                except Exception:
                    msg = "ignoring meta-data key %s with JSON-incompatible value"
                    warnings.warn(msg % (k,))
                    del md[k]
            params['meta_data'] = md
        else: params = normalize(params)
        return params
    @classmethod
    def denormalize(self, params):
        """Denormalizes an object of a given type from a native Python form.

        `ObjectWithMetaData.denormalize(params)` is used to denormalize an
        object given a mapping of normalized JSON parameters, as produced via
        `obj.normalize()` or `normalize(obj)`. Note that `ObjectWithMetaData`
        here can be substituted out with one of its derived classes.

        This function should generally be called by the `denormalize()` function
        rather than being called directly unless the data you have was produced
        by a call to `obj.normalize()` rather than `normalize(obj)`.

        Parameters
        ----------
        params : mapping
            A mapping of the immutable parameter values that can be used to
            rehydrate an object.

        Returns
        -------
        object
            An object of the type used in the call signature, constructed from
            the given parameter mapping.

        Raises
        ------
        Exception
            If the parameters could not hydrate an object of the given type.
        """
        return self(**params)
def normalize(data):
    """Converts an object into a JSON-friendly normalized dscription object.
    
    `normalize(obj)` returns a JSON-friendly normalized description of the given
    object. If the data cannot be normalized an error is raised.

    Any object that implements a `normalize()` function can be normalized, so
    long as the mapping object returned by `normalize()` itself can be
    normalized. Note that the `normalize()` function must return a mapping
    object such as a dict.

    Objects that can be represented as themselves such as numbers, strings, or
    `None` are returned as themselves. Any other object will be represented as a
    map that includes the reserved key `'__type__'` which will map to a
    2-element list `[module_name, class_name]`; upon denormalization, the module
    and class `k` are looked up and `k.denomalize(data)` is called.

    Parameters
    ----------
    data : object
        The data to be normalized.

    Returns
    -------
    object
        A JSON-friendly object that can be serialized as a JSON-string.

    Raises
    ------
    Exception
        If the argument `data` cannot be normalized.
    """
    if data is None: return None
    elif pimms.is_array(data, 'complex') and not pimms.is_array(data, 'real'):
        # any complex number must be handled specially:
        return {normalize.type_key: [None, 'complex'],
                're':np.real(data).astype('float'), 'im': np.imag(data).astype('float')}
    elif is_set(data):
        # sets also have a special type:
        return {normalize.type_key: [None, 'set'], 'elements': normalize(list(data))}
    elif pimms.is_scalar(data, ('string', 'unicode', 'bool', 'integer')):
        # most scalars are already normalized
        return data
    elif pimms.is_scalar(data, 'number'):
        # make sure it's not a float32 object
        return float(data)
    elif sps.issparse(data):
        # sparse matrices always get encoded as if they were csr_matrices (for now)
        (i,j,v) = sps.find(data)
        return {normalize.type_key: [None, 'sparse_matrix'],
                'rows':i.tolist(), 'cols':j.tolist(), 'vals': v.tolist(),
                'shape':data.shape}
    elif is_map(data):
        newdict = {}
        for (k,v) in six.iteritems(data):
            if not is_str(k):
                raise ValueError('Only maps with strings for keys can be normalized')
            newdict[k] = normalize(v)
        return newdict
    elif pimms.is_array(data, ('number', 'string', 'unicode', 'bool')):
        # numpy arrays just get turned into lists
        return np.asarray(data).tolist()
    elif data is Ellipsis:
        return {normalize.type_key: [None, 'ellipsis']}
    elif pimms.is_scalar(data):
        # we have an object of some type we don't really recognize
        try:              m = data.normalize()
        except Exception: m = None
        if m is None:
            raise ValueError('could not run obj.normalize() on obj: %s' % (data,))
        if not is_map(m):
            raise ValueError('obj.normalize() returned non-map; obj: %s' % (data,))
        m = dict(m)
        tt = type(data)
        m[normalize.type_key] = [tt.__module__, tt.__name__]
        return m
    else:
        # we have an array/list of some kind that isn't a number, string, or boolean
        return [normalize(x) for x in data]
normalize.type_key = '__type__'
def denormalize(data):
    """Converts a normalized object into its standard Python representation.
    
    `denormalize(data)` yield a denormalized version of the given JSON-friendly
    normalized `data` argument. This is the inverse of the `normalize(obj)`
    function.

    The normalize and denormalize functions use the reserved keyword
    `'__type__'` along with the `<obj>.normalize()` and
    `<class>.denormalize(data)` functions to manage types of objects that are
    not JSON-compatible. Please see `help(normalize)` for more details.

    Parameters
    ----------
    data : object
        The data to be denormalized.

    Returns
    -------
    object
        A Python object equivalent to that from which the given `data` was
        denormalized.

    Raises
    ------
    Exception
        If the argument `data` cannot be denormalized.
    """
    if   data is None: return None
    elif pimms.is_scalar(data, ('number', 'bool', 'string', 'unicode')): return data
    elif is_map(data):
        # see if it's a non-native map
        if normalize.type_key in data:
            (mdl,cls) = data[normalize.type_key]
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
                d = {k:denormalize(v) for (k,v) in six.iteritems(data) if k != normalize.type_key}
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

# Affine Code ######################################################################################
def to_affine(aff, dims=None, dtype=None, requires_grad=False):
    """Converts the argument to an affine transform matrix.

    `to_affine(data)` returns an affine transformation matrix equivalent to that
    given in `data`. The value in `data` may be specified either as `(matrix,
    offset_vector)`, as an `n+1`x`n+1` matrix, or, as an `n`x`n+1` matrix.

    `to_affine(data, dims)` additionally requires that the dimensionality of the
    `data` be `dims`, meaning that the returned matrix will be of size
    `dims+1`x`dims+1`.

    Parameters
    ----------
    data : tuple or matrix
        One of the following representations of an affine transformation: (1) a
        4x4 or 4x3 matrix (3D affine), (2) a 3x3 or 3x2 matrix (2D affine), (3)
        a tuple `(matrix, offset)` that contains either a 3D matrix and 3D
        vector or a 2D matrix and 2D vector. Higher order affines may also be
        parsed.
    dims : int > 0, optional
        The dimensionality of the affine matrix that should be returned. The
        returned matrix will always have shape `(dims+1, dims+1)`. If the `dims`
        argument is not explicitly given or is `None`, then the dimensionality
        is detected automatically. Providing the `dims` argument explicitly
        forces an error to be raised should the `data` argument not match the
        requested dimensionality.
    dtype : numpy dtype or torch dtype or numpy dtype alias, optional
        The dtype to use in constructing the affine matrix. If the `dtype` is a
        `torch.dtype` object, then the result will be a `torch` tensor;
        otherwise the result will be a `numpy` array. The default value of dtype
        is `None`, which indicates that the dtype will be inferred from the
        `data`.
    requires_grad : boolean, optional
        If the `dtype` parameter is a `torch.dtype` object, then this specifies
        whether the created affine matrix will require gradient tracking. The
        default value is `False`.

    Returns
    -------
    numpy.ndarray or torch.tensor
        A square matrix representing the affine transformation in `data`. The
        return value is a `torch` tensor if the `dtype` parameter is a
        `torch.dtype` object; otherwise the return value is a `numpy.ndarray`
        object.

    Raises
    ------
    ValueError
        If `data` cannot be interpreted as an affine transformation or if the
        affine transformation provided in `data` does not have the
        dimensionality required by an explicit `dims` argument.
    """
    # Have we been given a tuple or a matrix?
    if pimms.is_tuple(aff) and len(aff) == 2:
        (mtx,off) = aff
        (mtx,off) = nym.promote(mtx, off, dtype=dtype)
        mtx_sh = mtx.shape
        off_sh = off.shape
        if mtx_sh == (3,) and off_sh == (3,):
            # This is actually a 2x3 matrix, and as such it is a valid 2D affine matrix.
            aff = nym.cat([mtx[None,:], off[None,:], [(0, 0, 1)]], axis=1)
        else:
            assert len(mtx_sh) == 2 and mtx_sh[0] == mtx_sh[1], \
                'affine tuples must contain a square matrix and a vector'
            assert len(off_sh) == 1 and off_sh[0] == mtx_sh[0], \
                'affine tuples must contain matrix and offset with matching shapes'
            aff = nym.cat([mtx, off[:,None]], axis=1)
            aff = nym.cat([aff, [0]*len(off) + [1]])
        aff_sh = np.shape(aff)
    else:
        aff = nym.promote(aff, dtype=dtype)
        aff_sh = aff.shape
        assert len(aff_sh) == 2, \
            'affine transform arrays must be matrices'
        if aff_sh[1] == aff_sh[0] + 1:
            aff = nym.cat([aff, [0]*aff_sh[0] + [1]], axis=0)
            aff_sh = aff.shape
        assert aff_sh[0] == aff_sh[1], \
            'affiine transform matrices must be n x n or n x (n+1)'
    if dims is not None:
        assert dims == aff_sh[0] - 1, \
            f'given affine matrix does not match required dimensioinality: {dims}'
    return aff
def apply_affine(affine, coords, T=False, dtype=None):
    """Applies an affine transform to a set of matrix of coordinates.

    `apply_affine(affine, coords)` applies the given affine transformation to
    the given coordinate or coordinates and returns the new coordinate or set of
    coordinates that results.

    This function requires that `coords` to be a `dims`x`n` matrix where `dims`
    is the also the dimensionality of the affine trannsform giiven in the
    argument `affine`.

    Parameters
    ----------
    affine : affine matrix or affine-like or None
        The affine transformation to apply to the `coords`. This may be a matrix
        with shape `(dims+1, dims+1)` where `dims` is the number of rows in the
        `coords` matrix, or it may be anything that can be converted to such a
        matrix using the `to_affine()` function. If `affine` is `None`, then no
        affine transformation is applied, and the `coords` are returned
        untouched.
    coords : matrix of numbers
        The coordinate matrix that is to be transformed by the `affine`. This
        must be a `dims`x`n` matrix or a `dims`-length vector. The return value
        is always the same shape as `coords`.
    T : boolean, optional
        If `True`, then `coords` must instead be a `n`x`dims`-shaped matrix.

    Returns
    -------
    matrix of numbers
        A matrix of numbers the same shape as `coords`. The return value is
        always the same shape as `coords` regardless of whether the `T`
        parameter has been set to `True`.

    Raises
    ------
    ValueError
        If `affine` cannot be interpreted as an affine transformation, if
        `coords` is not a matrix, or if the dimensionality of the `coords`
        matrix does not match that of the `affine` transformation.
    """
    # Get the coordinates:
    coords = nym.promote(coords, dtype=dtype)
    sh = coords.shape
    assert len(sh) == 2, "coords argument to apply_affine must be a matrix"
    if affine is None: return coords
    if T:
        coords = nym.tr(coords)
        sh = coords.shape
    # Get the affine transform:
    aff = to_affine(affine, dtype=coords.dtype, dims=sh[0])
    # Apply it.
    res = nym.add(nym.dot(aff[:-1,:-1], coords), aff[:-1, [-1]])
    if T:
        res = nym.tr(res)
    return res

# Dataframe Code ###################################################################################
def is_dataframe(d):
    """Returns `True` if given a `pandas.DataFrame`, otherwise `False`.
    
    `is_dataframe(d)` returns `True` if `d` is a `pandas.DataFrame` object and
    `False` otherwise; if `pandas` cannot be loaded, this yields `None`.

    Parameters
    ----------
    d : object
        The object that is to be tested: is this object a `pandas` `DataFrame`
        object?
    
    Returns
    -------
    boolean
        `True` if `d` is a `pandas.DataFrame` object and `False` otherwise. If
        `pandas` cannot be imported, instead returns `None`.
    """
    try: import pandas
    except Exception: return None
    return isinstance(d, pandas.DataFrame)
def to_dataframe(d, **kw):
    """Converts `d` into a `pandas.DataFrame` and returns it.

    `to_dataframe(d)` attempts to coerce the object `d` to a `pandas.DataFrame`
    object. If `d` is a tuple of 2 items whose second argument is a dictionary,
    then the dictionary will be taken as arguments for the dataframe
    constructo. (These arguments may alternately be given as standard keyword
    arguments.) Otherwise, `d` must be in one of two formats: (1) `d` can be a
    mapping (such as a dictionary) whose keys are the column names and whose
    values are the dataframe's columns, or (2) `d` can be an iterable whose rows
    are dicts representing the dataframe's individual rows. In the latter of
    these cases, the dictionaries that represent the rows must all have the same
    keys while in the former of these cases, the iterables that represent the
    columns must all have the same number of elements.

    Parameters
    ----------
    d : object
        The object that is to be converted into a dataframe.
    **kw
        Any additional keyword arguments that should be passed to the
        `pandas.DataFrame` constructor.

    Returns
    -------
    pandas.DataFrame
        A dataframe representation of `d`.

    Raises
    ------
    ValueError
        If `d` cannot be interpreted as a dataframe.
    """
    import pandas
    if pimms.is_itable(d): d = d.dataframe
    if is_dataframe(d): return d if len(kw) == 0 else pandas.DataFrame(d, **kw)
    if is_tuple(d) and len(d) == 2 and is_map(d[1]):
        try: return to_dataframe(d[0], **dict(d[1], **kw))
        except Exception: pass
    # try various options:
    try: return pandas.DataFrame(d, **kw)
    except Exception: pass
    try: return pandas.DataFrame.from_records(d, **kw)
    except Exception: pass
    try: return pandas.DataFrame.from_dict(d, **kw)
    except Exception: pass
    raise ValueError('Coersion to dataframe failed for object %s' % d)
def dataframe_select(df, *args, **kwargs):
    """Performs a simple selection on a dataframe object.

    `dataframe_select(df, k1=v1, k2=v2...)` yields a subset of the rows of `df`
    after selecting only those rows in which the given keys (`k1`, `k2`, etc.)
    map to values that match the filter instructiions (`v1`, `v2`, etc.; see
    below regarding filters).

    `dataframe_select(df, col1, col2...)` selects a subset of the columns of
    `df`: those that are listed only.

    `dataframe_select(df, col1, col2..., k1=v1, k2=v2...)` selects only the
    listed columns and only those rows that match the given key-value filters.
    
    **Filters**. Keys are mapped to values that act as filter instructons (i.e.,
    the `v1`, `v2`, etc. in `dataframe_select(df, k1=v1, k2=v2...)` are filter
    instructions). Depending on the instructioin, different rules will be used
    to select the rows that are included. These rules are as follows: (1) if a
    filter is a tuple or list of 2 elements, then it is considered a range where
    cells must fall between the values; (2) if a filter is a tuple or list of
    whose length is not 2, or if it is a set of any length, then any values in
    the filter are considered acceptable values regarding the selection; (3)
    otherwise, the filter itself is considered to be the only acceptable value
    regarding the selection.

    Parameters
    ----------
    df : pandas.DataFrame or dataframe-like
        The `pandas` dataframe to select on; if this is not a `pandas.DataFrame`
        object, the function will attempt to interpret it as a dataframe first
        using the `to_dataframe()` function.
    *args
        The names of the columns that should be subselected; if no column names
        are provided, then all columns are included.
    **kwargs
        The names of columns (as keys) mapped to filter instructions (as
        values); each of the filter instructoins is applied to the relevant
        column value of all rows, and only those rows that pass all filters are
        included in the selection.

    Returns
    -------
    pandas.DataFrame
        The resulting subselection of the dataframe `df`.
    """
    ii = np.ones(len(df), dtype='bool')
    for (k,v) in six.iteritems(filters):
        vals = df[k].values
        if   is_set(v):                    jj = np.isin(vals, list(v))
        elif pimms.is_vector(v) and len(v) == 2: jj = (v[0] <= vals) & (vals < v[1])
        elif pimms.is_vector(v):                 jj = np.isin(vals, list(v))
        else:                                    jj = (vals == v)
        ii = np.logical_and(ii, jj)
    if len(ii) != np.sum(ii): df = df.loc[ii]
    if len(cols) > 0: df = df[list(cols)]
    return df

# AutoDict code ####################################################################################
class AutoDict(dict):
    """A dictionary that automatically vivifies new keys.

    `AutoDict` is a handy kind of dictionary that automatically vivifies itself
    when a miss occurs. By default, the new value returned on miss is itself an
    `AutoDict`, thus allowing for arbitrary depths of nested dictionaries with
    ease, but this may be changed by setting the object's `on_miss` attribute to
    a custom function such as `lambda:[]` (to return an empty list).

    The `auto_dict()` function can also be used to create `AutoDict` objects.

    Parameters
    ----------
    *args
        Any number of dict objects, which are merged from left to right, to form
        the initial key-value pairs in the new `AutoDict`.
    **kwargs
        Any number of key-value pairs may be passed to `AutoDict()` and are
        treated as a final right-most dictionary when merging together the
        `*args` to form the initial contents of the new `AuroDict`.

    Attributes
    ----------
    on_miss : function
        The function that is called (as `on_miss()`) when a key miss occurs. The
        return value of this function is the new value that is inserted with the
        key that caused the miss.
    """
    def __init__(self, *args, **kwargs):
        dict.__init__(self, *args, **kwargs)
        self.on_miss = lambda:type(self)()
    def __missing__(self, key):
        value = self.on_miss()
        self[key] = value
        return value
def auto_dict(ival=None, miss=None):
    """Creates and returns an `AutoDict` object.

    `auto_dict()` is equivalent to `AutoDict()`.

    `auto_dict(ival=ival)` uses the given dict `ival` as an initializer for the
    `AutoDict` contents. The `ival` may be a mapping object or it may be a tuple
    or list of mapping objects, which are merged left-to-right.

    `auto_dict(miss=miss)` uses the given value `miss` as the `AutoDict`'s
    `on_miss` attribute. Usually this must be a function such as `lambda:[]`,
    but if the `miss` argument is equal to either `[]` or `set([])`, then the
    returned `AutoDict` will use `on_miss = lambda:[]` or `on_miss =
    lambda:set([])`, respectively. Similarly, if the `miss` argument is equal to
    `{}`, then the `on_miss = lambda:type(self)()`, which is equivalent to the
    function used for the default `miss` value of `None`.

    Parameters
    ----------
    ival : None or mapping or iterator of mappings, optional
        The initial value of the new dictionary.
    miss : None or [] or set([]) or {} or function, optional
        The on_miss function that the new dictionary should use.

    Returns
    -------
    AutoDict
        The new dictionary object.
    """
    if ival is None: d = AutoDict()
    elif is_map(d): d = AutoDict(ival)
    else: d = AutoDict(dict(*ival))
    if miss == {} or miss is None: return d
    elif miss == []: d.on_miss = lambda:[]
    elif miss == set([]): d.on_miss = lambda:set([])
    else: d.on_miss = miss
    return d

# Address Functions ################################################################################
def is_address(data, check_values=False):
    """Determines if the argument contains valid neuropythy address data.

    `is_address(data)` returns `True` if `data` is a valid address dict for
    addressing positions on a mesh or in a (3D) cortical sheet and returnsp
    `False` otherwise. In order to be a valid address dict, `data` must be a
    mapping object and must contain the keys `'faces'` and `'coordinates'`. The
    key `'faces'` must be mapped to a 3D vector or 3D matrix of integers (mesh
    vertex labels), and the key `'coordinates'` must be mapped to a 2D or 3D
    matrix of real numbers (barycentric coordinates with an optional depth
    coordinate for 3D cortical sheets).

    In order to be considered 2D or 3D, a matrix must have 2 or 3 rows,
    respectively (not 2 or 3 columns).

    Parameters
    ----------
    data : object
        An object whose quality as a set of address data is to be assessed.
    check_values : boolean, optional
        Whether or not to check that the values of the `data` mapping adhere to
        the address data format as well as the keys. The default behavior
        (`check_values=False`) does not require that the values be matrices of
        the appropriate shape and type. If this parameter is set to `True`,
        these checks will be carried out.

    Returns
    -------
    boolean
        `True` if `data` contains valid neuropythy address data and `False`
        otherwise.
    """
    if not (is_map(data) and 'faces' in data and 'coordinates' in data):
        return False
    if check_values:
        # We need to check the value shapes.
        faces = data['faces']
        coords = data['coords']
        fsh = nym.shape(faces)
        csh = nym.shape(coords)
        if not arraylike(faces, dtype='int', ndims=2): return False
        if not arraylike(coords, dtype='float', ndims=2): return False
        if fsh[0] != 3: return False
        if csh[0] != 2 and csh[0] != 3: return False
        if csh[1] != fsh[1]: return False
    return True
def address_data(data, dims=None, surface=0.5, strict=True):
    """Extracts the `(faces, coordinates)` from an address data dict.

    `address_data(data)` returns the tuple `(faces, coords)` of the address data
    where both faces and coords are guaranteed to be numpy arrays with sizes
    (`3`x`n`) and (`d`x`n`). If the data are not valid as an address, then an
    error is raised. If the address is empty, this returns `(None, None)`.

    Parameters
    ----------
    data : an address data mapping (see `is_address()`)
        An address data mapping whose face and coordinate matrices are being
        extracted.
    dims : 2 or 3 or None, optional
        The dimensionality requested of the output coordinate matrix. If the
        address data is not the requested dimensionality, then the depth
        dimension is truncated (in the case `dims=2`), or the value of `surface`
        (see below) is appended to all columns (in the case that `dims=3`). If
        `dims` is `None` (the default), then no change is made to the
        coordinates.
    surface : real number in [0,1], optional
        The depth value to append to the columns of the coordinates matrix in
        the case that 3D coordinates are requested but only 2D coordinates are
        available. In such a case, positions in the 2D triangle surface mesh are
        encoded in the address data, but the depth is not, so an arbitrary depth
        must be chosen. The depth must be a real number between 0 and 1 with 0
        representing the white-matter surface of cortex and 1 representing the
        pial surface. The aliases `'pial'`, `'white'`, and `'midgray'` can be
        used as aliases for `1`, `0`, and `0.5`, respectively. The default value
        is `0.5`.
    strict : boolean, optional
        Whether to rraise an error when there are non-finite values found in the
        faces or the coordinates matrices. These values are usually indicative
        of an attempt to address a point that was not inside the mesh/cortex.
        The default is `True`, which does not suppress errors. To suppress these
        errors use `strict=False`.

    Returns
    -------
    2-tuple of numpy.ndarray matrices
        The `(faces, coordinates)` matrices are returned as aa 2-tuple. The
        `faces` matrix is a `3`x`n` integer matrix whose columns are the mesh
        vertex labels of the corners of each triangle represented in the address
        data, and the `coordinates` matrix is a `2`x`n` or `3`x`n` real-valued
        matrix whose columns are the barycentric coordinates of the represented
        point in the associated triangle from `faces` and (optionally) the
        cortical depth fraction if the `coordinates` matrix has 3 rows.

    Raises
    ------
    ValueError
        If `data` is not a valid address data mapping.
    """
    if data is None: return (None, None)
    if not is_address(data, check_values=False): # We're gonna recheck the values.
        raise ValueError('argument is not a valid address data mapping')
    faces = promote(data['faces'])
    coords = promote(data['coordinates'])
    if len(faces.shape) > 2 or len(coords.shape) > 2:
        raise ValueError('address data contained high-dimensional arrays')
    if len(faces.shape) != len(coords.shape):
        raise ValueError('address data faces and coordinates are different shapes')
    if faces.shape[1:] != coordinates.shape[1:]:
        raise ValueError('address data faces and coordinates have different column counts')
    if faces.shape[0] != 3:
        raise ValueError('address contains a face matrix whose first dimension is not 3')
    if coords.shape[0] not in (2,3):
        raise ValueError('address coords are neither 2D nor 3D')
    if len(faces.shape) == 2 and faces.shape[1] == 0:
        return (None, None)
    if dims is None: dims = coords.shape[0]
    elif coords.shape[0] != dims:
        if dims == 2: coords = coords[:2]
        else:
            if surface is None:
                raise ValueError('address data must be 3D but 2D data was found')
            elif is_str(surface):
                surface = surface.lower()
                if surface == 'pial': surface = 1
                elif surface == 'white': surface = 0
                elif surface in ('midgray', 'mid', 'middle'): surface = 0.5
                else: raise ValueError('unrecognized surface name: ' + surface)
            if not pimms.is_real(surface) or surface < 0 or surface > 1:
                raise ValueError('surface must be a real number in [0,1]')
            coords = nym.cat([coords, nym.full((1, coords.shape[1]), surface)])
    if strict:
        cnans = nym.logical_not(nym.isfinite(coords))
        if nym.sum(cnans) > 0:
            w = nym.where(cnans)
            if len(w[0]) > 10:
                raise ValueError('address contains %d non-finite coords' % len(w[0]))
            else:
                raise ValueError('address contains %d non-finite coords (%s)' % (len(w),w))
        fnans = nym.logical_not(nym.isfinite(faces))
        if nym.sum(fnans) > 0:
            w = np.where(fnans)
            if len(w[0]) > 10:
                raise ValueError('address contains %d non-finite faces' % len(w[0]))
            else:
                raise ValueError('address contains %d non-finite faces (%s)' % (len(w[0]),w))
    return (faces, coords)
def address_interpolate(addr, prop, method=None, surface='midgray',
                        strict=False, null=np.nan, index=None):
    """Interpolates a property at a set of points specified using address data.

    `address_interpolate(addr, prop)` returns the result of interpolating the
    given property prop at points that are specified by the given addresses. If
    addr contains 3D addresses and prop is a map of layer values from 0 to 1
    (e.g., `{0:white_prop, 0.5:midgray_prop, 1:pial_prop}`), then the addresses
    are respect and are interpolated from the appropriate layers.

    The address data `addr` is related too the property `prop` in that the
    vertex labels in the cells of the `'faces'` matrix of the address data
    correspond to a cell in the `prop` vector (or in each of `prop`'s columns,
    if prop is a dictionary of vectors). In other words, the address data must
    have been calculated from the set of points at which we are
    interpolating---in most interpolation function this set of points (or this
    topology/mesh object) is required, but `addr` suffices here.

    If the `prop` and `addr` data comes from a flatmap or a submesh of a full
    mesh or topology object, then the vertex labels in the `addr` data will not
    be correct indices into the `prop` data. In this case, an index is needed,
    and either the flatmap's/submesh's `tess` attribute or its `tess.index` may
    be given as an opotional parameter in this case.

    Parameters
    ----------
    addr : address data dict
        The dictionary of address data (see `is_address` and `address_data`)
        that represents the points at which the property is to be interpolated.
    prop : array-like or mapping
        The property to be interpolated onto the points encoded by `addr`. This
        must be an array-like object (a `numpy` array, `torch` tensor, or
        something that can be converted into these types) whose last dimension
        is equivalent to the number of vertices in the mesh from which
        interpolation occurs (i.e., the mesh on which the `addr` address data
        were calculated) or a mapping whose values are all such array-like
        objects and whose keys are cortical depths or are like cortical depths
        (see `is_cortical_depth()` and `like_cortical_depth()`). When `prop` is
        a mapping, the interpolation occurs either at the depths specified in
        the `addr` data or at the depth specified by the `surface` argument,
        which must be like a cortical depth also, if no depth is encoded in
        `addr`.
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
    surface : cortical-depth-like, optional
        If the `addr` data do not contain informatioon about cortical depth and
        the `prop` data contain property data for multiple cortical layers,
        including, at minimum, 0.0 and 1.0, then the depth at which
        interpolation occurs is providd by `to_cortical_depth(surface)`. The
        default value is `midgray`.
    strict : boolean, optional
        If `True`, an error is raised if any address coordinates have non-finite
        values (i.e., were "out-in-region" values); otherwise the associated
        interpolated values are silently set to `null`. The default is `False`.
    null : object, optional
        The value given to any "out-of-region" value found in the addresses if
        `strict` is `False`. The default is `nan`.
    index : Index or Tesselation, optional
        If the addresses werer calculated in reference to a mesh that is a
        flatmap or submesh of another mesh, then the vertex labels in the `addr`
        data's `'faces'` matrix will not match up to the `prop` dimensions. In
        this case, `index` may be the `Tesselation` object or the tesselation's
        `Index` object, allowing `address_interpolate()` to translate from
        vertex labels to vertex indices. The default, `None`, results in no
        translaction, meaning that `prop` must be from a mesh that has not been
        subsampled.

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
    # Argument Parsing #############################################################################
    # Parse the index, if any.
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

# #TODO -- code cleaning: above is mostly clean, below needs work.
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

def nan_compare(f, x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nan_compare(f, x, y) is equivalent to f(x, y), which is assumed to be a boolean function that
      broadcasts over x and y (such as numpy.less), except that NaN values in either x or y result
      in a value of False instead of being run through f.

    The argument f must be a numpy comparison function such as numpy.less that accepts the optional
    arguments where and out.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to f(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to f(nan, non_nan).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to f(non_nan, nan).
    '''
    #TODO: This should work with sparse matrices as well
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    xii = np.isnan(x)
    yii = np.isnan(y)
    if not xii.any() and not yii.any(): return f(x, y)
    ii  = (~xii) & (~yii)
    out = np.zeros(ii.shape, dtype=np.bool)
    if nan_nan == nan_val and nan_val == val_nan:
        # All the nan-result values are the same; we can simplify a little...
        if nan_nan: out[~ii] = nan_nan
    else:
        if nan_nan: out[   xii &    yii] = nan_nan
        if nan_val: out[   xii & (~yii)] = nan_val
        if val_nan: out[(~xii) &    yii] = val_nan
    return f(x, y, out=out, where=ii)
def naneq(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    naneq(x, y) is equivalent to (x == y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to naneq(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to naneq(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to naneq(nan, 0).
    '''
    return nan_compare(np.equal, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nanne(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nanne(x, y) is equivalent to (x != y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanne(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanne(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanne(nan, 0).
    '''
    return nan_compare(np.not_equal, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nanlt(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nanlt(x, y) is equivalent to (x < y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanlt(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanlt(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nan;t(nan, 0).
    '''
    return nan_compare(np.less, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nanle(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nanle(x, y) is equivalent to (x <= y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanle(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanle(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanle(nan, 0).
    '''
    return nan_compare(np.less_equal, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nangt(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nangt(x, y) is equivalent to (x > y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nangt(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to nangt(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nangt(nan, 0).
    '''
    return nan_compare(np.greater, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nange(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nange(x, y) is equivalent to (x >= y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nange(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to nange(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nange(nan, 0).
    '''
    return nan_compare(np.greater_equal, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nanlog(x, null=np.nan):
    '''
    nanlog(x) is equivalent to numpy.log(x) except that it avoids calling log on 0 and non-finie
      values; in place of these values, it returns the value null (which is nan by default).
    '''
    x = np.asarray(x)
    ii0 = np.where(np.isfinite(x))
    ii  = np.where(x[ii0] > 0)[0]
    if len(ii) == numel(x): return np.log(x)
    res = np.full(x.shape, null)
    ii = tuple([u[ii] for u in ii0])
    res[ii] = np.log(x[ii])
    return res    

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
    def set(**kw):
        '''
        ds.set(a=b, c=d, ...) yields a copy of the data-struct ds in which the given keys have been
          set to the given values. If no keys are changed, then ds itself is returned.
        ds.set() yields ds.
        '''
        d = self.__dict__
        try:
            if all(v is d[k] for (k,v) in six.iteritems(kw)): return self
        except KeyError: pass
        d = dict(d, **kw)
        return DataStruct(d)
    def delete(*args):
        '''
        db.delete('a', 'b', ...) yields a copy of the data-struct ds in which the given keys have
          been dropped from the data-structure. If none of the keys are in ds, then ds itself is
          returned.
        ds.delete() yields ds.
        '''
        d = self.__dict__
        for k in args:
            if k in d:
                if d is self.__dict__: d = dict(d)
                del d[k]
        if d is self.__dict__: return self
        return DataStruct(d)
        
def data_struct(*args, **kw):
    '''
    data_struct(args...) collapses all arguments (which must be maps) and keyword arguments
      right-to-left into a single mapping and uses this mapping to create a DataStruct object.
    '''
    m = pimms.merge(*(args + (kw,)))
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
    elif is_str(p): p = p.split(':')
    if len(p) > 0 and not pimms.is_vector(p, str):
        raise ValueError('Path is not equivalent to a list of dirs')
    return [pp for pp in p if os.path.isdir(pp)]

def try_until(*args, **kw):
    '''
    try_until(f1, f2, f3...) attempts to return f1(); if this raises an Exception during its
      evaluation, however, it attempts to return f2(); etc. If none of the functions succeed, then
      an exception is raised.

    The following optional arguments may be given:
      * check (default: None) may specify a function of one argument that must return True when the
        passed value is an acceptable return value; for example, an option of
        `check=lambda x: x is not None`  would indicate that a function that returns None should not
        be considered to have succeeded.
    '''
    if 'check' in kw: check = kw.pop('check')
    else: check = None
    if len(kw) > 0: raise ValueError('unrecognized options given to try_until')
    for f in args:
        if not hasattr(f, '__call__'):
            raise ValueError('function given to try_until is not callable')
        try:
            rval = f()
            if check is None or check(rval): return rval
        except Exception: raise
    raise ValueError('try_until failed to find a successful function return')
