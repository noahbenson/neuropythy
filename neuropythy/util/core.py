# -*- coding: utf-8 -*-
####################################################################################################
# neuropythy/util/core.py
# This file implements the command-line tools that are available as part of neuropythy as well as
# a number of other random utilities.

import atexit, shutil, tempfile, importlib, pimms, os, six, warnings
import numpy        as np
import scipy.sparse as sps
import pyrsistent   as pyr

from .. import math as nym
# A few old functions were moved to pimms; they still appear here for compatibility reasons.
from pimms import (is_tuple, is_list, is_set, is_map, is_str, curry)

# Info Utilities ###################################################################################
def is_iterable(obj, map=True, set=True, str=True):
    """Determines if the given object is iterable.

    This function can be used to determine if an object is iterable. A few
    common types of objects, such as strings and dictionaries, may be flagged as
    special cases using the optional parameters.

    Parameters
    ----------
    obj : object
        The object whose iterability is to be assessed.
    map : boolean
        Whether an object that is a subclass of `Mapping` is considered an
        iterable object. The default is `True` (maps and dicts are considered
        iterablee).
    set : boolean
        Whether an object that is a subclass of `set` or `frozenset` is
        considered an iterable object. The default is `True` (sets are
        considered iterablee).
    str : boolean
        Whether an object that is a subclass of `str` is considered an iterable
        object. The default is `True` (strings are considered iterablee).

    Returns
    -------
    boolean
        `True` if `obj` is an iterable objects, subject to the above
        restrictions, and `False` otherwise.
    """
    if not map and is_map(obj): return False
    if not set and is_set(obj): return False
    if not str and is_str(obj): return False
    return hasattr(obj, '__iter__')
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
def is_grayheight(s):
    """Returns `True` if `s` is a float and `0 <= s <= 1`, otherwise `False`.

    Cortical depths are fractional float-typed values between 0 and 1. This
    function yields `True` if `s` conforms to this exact type (i.e., an int 0
    will fail where a float 0.0 will pass). To convert a value to a cortical
    depth, use `to_grayheight()`. To check if something can be converted,
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
def like_grayheight(s, aliases=None):
    """Returns `True` if `s` can be convertd into a cortical depth.

    Cortical depths are fractional float-typed values between 0 and 1. This
    function yields `True` if `s` can be coerced into a cortical depth by the
    `to_grayheight()` function.

    Parameters
    ----------
    s : object
        An object whose quality as a cortical depth is to be assessed.
    aliases : mapping or None, optional
        A set of aliases for cortical depths that should be considered. See
        `to_grayheight()`.

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
            return like_grayheight(alt)
    # First, is this a cortical depth already?
    if is_grayheight(s): return True
    # Is it the name of a cortical depth?
    if is_str(s): s = s.lower()
    if s == 'pial' or s == 'midgray' or s == 'white': return True
    # Is it in the builtin alises?
    alt = to_grayheight.aliases.get(s, Ellipsis)
    if alt is not Ellipsis:
        if is_grayheight(alt): return alt
        else: s = alt
    # Okay, is s a number that is between 0 and 1?
    try:
        s = float(s)
        return (s <= 1 and s >= 0)
    except TypeError:
        return False
def to_grayheight(s, aliases=None):
    """Converts an object, which may be a surface name, into a depth fraction.
    
    `to_grayheight(s)` converts the argument `s` into a cortical depth
    fraction: a real number `r` such that `0 <= r and r <= 1`. If `s` cannot be
    converted into such a fraction, raises an error. `s` can bbe converted into
    a fraction if it is already such a fraction or if it is the name of a
    cortical surface: `'pial'` (`1`), `'midgray'` (`0.5`), and `'white'` (`0`).

    If `s` is `None`, then `0.5` is returned.

    To add a new named depth, you can modify the `to_grayheight.aliases`
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
        whose keys have all been transformed by the `to_grayheight()`
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
        the input after `to_grayheight()` has been called its keys.

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
                    newk = to_grayheight(k, aliases=aliases)
                    d = d.remove(k).set(newk, v)
            return d
        elif pimms.is_lmap(s):
            d = s
            for k in six.iterkeys(s):
                if not isinstance(k, float) or k < 0 or k < 1:
                    newk = to_grayheight(k, aliases=aliases)
                    if s.is_lazy(k):
                        d = d.remove(k).set(newk, s.lazyfn(k))
                    else:
                        d = d.remove(k).set(newk, s[k])
            return d
        else:
            d = s.copy()
            for (k,v) in six.iteritems(s):
                if not is_grayheight(s):
                    newk = to_grayheight(k, aliases=aliases)
                    del d[k]
                    d[newk] = v
            return d
    # Otherwise, we are convertinng s itself into a cortical depth. First thing is that we
    # check aliases, which overrides all other behavior.
    if aliases is not None:
        ss = aliases.get(s, Ellipsis)
        if ss is not Ellipsis: return to_grayheight(ss) # Omit aliases this time.
    # If it is a cortical depth, return it.
    if is_grayheight(s): return s
    # Check the global aliases.
    if is_str(s): s = s.lower()
    if   s == 'pial':    return 1.0
    elif s == 'midgray': return 0.5
    elif s == 'white':   return 0.0
    # Is it in the builtin alises?
    alt = to_grayheight.aliases.get(s, Ellipsis)
    if alt is not Ellipsis:
        if is_grayheight(alt): return alt
        else: s = alt
    # Okay, is s a number that is between 0 and 1?
    try:
        s = float(s)
        if (s <= 1 and s >= 0): return s
    except TypeError: pass
    raise ValueError(f"cannot interpret argument as a cortical depth: {s}")
to_grayheight.aliases = {}
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
        A mapping of meta-data keys to values.

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
def is_metaobj(obj):
    """Determines if an object is an `ObjectWithMtaData` or not.
    
    `is_metaobj(obj)` returns `True` if `obj` is an immutable object that can
    have meta-data and `False` otherwise. This function is essentially an alias
    for `isinstance(obj, ObjectWithMetaData)`.

    Parameters
    ----------
    obj : object
        The object whose quality as an `ObjectWithMetaData` is to be assessed.

    Returns
    -------
    boolean
        `True` if `obj` is an `ObjectWithMetaData` and `False` otherwise.
    """
    return isinstance(obj, ObjectWithMetaData)
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
def library_path():
    """Returns the local path of the neuropythy `'lib'` directory.

    Returns
    -------
    str
        The absolute local path of the neuropythy `'lib'` directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'lib'))

# DataStruct #######################################################################################
class DataStruct(object):
    """A simple immutable structure with configurable attributes.

    A `DataStruct` object is an immutable map-like object that accepts any
    number of keyword arguments on construction and assigns all of them as
    members which are then immutable in the resulting (immutable) object.

    Parameters
    ----------
    **kwargs
        The names and values to be included in the resulting object.
    """
    def __init__(self, **kw):    self.__dict__.update(kw)
    def __setattr__(self, k, v): raise ValueError('DataStruct objects are immutable')
    def __delattr__(self, k):    raise ValueError('DataStruct objects are immutable')
    def set(**kwargs):
        """Returns a copy of the data structure with an update to its members.

        `ds.set(a=b, c=d, ...)` returns a copy of the data-struct `ds` in which
        the given keys have been set to the given values. If no keys are
        changed, then `ds` itself is returned.  ds.set() yields ds.

        Parameters
        ----------
        **kwargs
            The attribute names to give the new data structure and their values.

        Returns
        -------
        DataStruct
            A data structure object that is a copy of the existing data
            structure but with updated attributes.
        """
        d = self.__dict__
        if all(k in d and v is d[k] for (k,v) in six.iteritems(kwargs)):
            return self
        d = dict(d, **kwargs)
        return DataStruct(d)
    def delete(*args):
        """Returns a copy of thhe data structure without the given attributes.

        `db.delete('a', 'b', ...)` yields a copy of the data-struct `ds` in
        which the given attributes have been dropped from the data-structure. If
        none of the keys are in `ds`, then `ds` itself is returned.  `ds.delete()`
        returns `ds`.

        Parameters
        ----------
        *args
            The names of the attributes that should be deleted.

        Returns
        -------
        DataStruct
            A new data structure object without the given attributes.
        """
        d = self.__dict__
        for k in args:
            if k in d:
                if d is self.__dict__: d = dict(d)
                del d[k]
        if d is self.__dict__: return self
        return DataStruct(d)
def is_data_struct(obj):
    """Returns `True` if given a `DataStruct` object and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a `DataStruct` is to be assessed.
    
    Returns
    -------
    boolean
        `True` if `obj` is a `DataStruct` and `False` otherwise.
    """
    return isinstance(obj, DataStruct)
def to_data_struct(*args, **kwargs):
    """Creates a `DataStruct` object with the given attribute dictionary.

    `to_data_struct(args...)` collapses all arguments (which must be mappings)
    and keyword arguments right-to-left into a single dictionary and uses this
    dict to create a `DataStruct` object. Any keys in the resulting dictionary
    are included as attributs in the resulting `DataStruct` object.

    This function is essentially an alias for the `DataStruct()` constructor
    itself.

    Parameters
    ----------
    *args
        Any number of dict or mapping objects that are to be mergeed
        left-to-right.
    **kwargs
        Any number of key-value pairs to be added to the attributes.

    Returns
    -------
    DataStruct
        A new `DataStruct` object with the attributes given in the arguments.
    """
    if len(args) == 1 and len(kwargs) == 0 and is_data_struct(args[0]):
        return args[0]
    args = [a.__dict__ if is_data_struct(a) else a for a in args]
    if len(kwargs) > 0: args.append(kwargs)
    d = {}
    for a in args: d.update(a)
    return DataStruct(**d)

# File System Tools ################################################################################
def tmpdir(prefix='npythy_tempdir_', delete=True):
    """Returns a temporary directory path.

    `tmpdir()` creates a temporary directory and returns its path. At python
    exit, the directory and all of its contents are recursively deleted (so long
    as the the normal python exit process is allowed to call the `atexit`
    handlers).

    `tmpdir(prefix)` uses the given prefix in the `tempfile.mkdtemp()` call that
    is used to create the directory.
    
    Parameters
    ----------
    prefix : str, optional
        The prefix to give the name of the temporary directory.
    delete : boolean, optional
         Whether to automatically delete the temporary directory when Python
         exits. The default is `True`.

    Returns
    -------
    str
        The path of the temporary directory that is created.

    Raises
    ------
    ValueError
        When a temporary directory cannot be created.
    """
    path = tempfile.mkdtemp(prefix=prefix)
    if not os.path.isdir(path): raise ValueError('Could not find or create temp directory')
    if delete: atexit.register(shutil.rmtree, path)
    return path
def to_pathlist(path, error_on_missing=False):
    """Converts a colon-separated string of paths into a Python list of paths.

    `to_pathlist(path)` returns a list of directories contained in the given
    path specification. The specification should be similar to how `PATH`
    variables are encoded in POSIX shells in which multiple paths are separated
    by the colon (:) character.

    A path may be either a single directory name (resulting in `[path]`), a
    colon-separated list of directories (resulting in `path.split(':')`), a list
    of directory names (resulting in `path`), or `None` (resulting in
    `[]`). Note that the return value filters out parts of the path that are not
    directories but does not raise an error.

    Parameters
    ----------
    path : str or list of str or None
        The object to be converted into a list of paths.
    error_on_missing : boolean, optional
        Whether to raise an error when any of the paths are not found. The
        default is `False`.

    Returns
    -------
    list of str
        A list of paths represented in the object `path`.
    """
    if   path is None: return []
    elif is_str(path): path = path.split(':')
    # Otherwise, we assume a list of strings and expect an error if it's not.
    if error_on_missing:
        for pp in path:
            if not os.path.isdir(pp):
                raise ValueError(f"pathlist directory not found: {pp}")
        return list(path)
    else:
        return [pp for pp in path if os.path.isdir(pp)]

# Other Utilities ##################################################################################
def is_callable(f):
    """Returns `True` if `f` has an `__call__` attribute and `False` otherwise.

    Parameters
    ----------
    f : object
        The object whose quality as a callable is to be determined.

    Returns
    -------
    boolean
        `True` if `hasattr(f, '__call__')`, otherwise `False`.
    """
    return hasattr(f, '__call__')
def try_through(*args, check=None, default=None, error_on_fail=True):
    """Attempts to run multiple functions and returns the first to succeed.

    `try_through(f1, f2, f3...)` attempts to return `f1()`; if this raises an
    `Exception` during its evaluation, however, it attempts to return `f2()`;
    etc. If none of the functions succeed, then an exception is raised.

    Parameters
    ----------
    *args
        The list of functions to try to run, in order.
    check : function or None, optional
        An optional function of one argument that must return `True` when the
        passed value is an acceptable return value; for example, an option of
        `check=lambda x: x is not None` would indicate that a function that
        returns `None` should not be considered to have succeeded.
    error_on_fail : boolean, optional
        Whether to raise an error on failure or to return the `default` optional
        argument instead. If `True` (the default) then an error is raised on
        failure. If `False`, then `default` is returned instead.
    default : object, optional
        The object to return if both `error_on_failure` is `False` and none of
        the functions in `*args` succeeds.

    Returns
    -------
    object
        The return value of the first function in `args` to successfully return
        a value that also passes the `check` function, if any.

    Raises
    ------
    ValueError
        If none of the given functions succeed or if one of the arguments is not
        callable.
    """
    for f in args:
        if not is_callable(f):
            raise ValueError('function given to try_through is not callable')
        try:
            rval = f()
            if check is None or check(rval):
                return rval
        except Exception: pass
    if error_on_fail:
        raise ValueError('try_until failed to find a successful function return')
    else:
        return default
