# -*- coding: utf-8 -*-
####################################################################################################
# neuropythy/util/labels.py
# Simple tools for dealing with neuroscience-related labels for brains.
# By Noah C. Benson

import os, sys, types, six, pimms
import numpy               as np
import pyrsistent          as pyr

from .core import (ObjectWithMetaData, is_tuple, curry, is_dataframe, is_iterable)

def label_colors(labels, cmap=None):
    """Assigns colors to each label value in a vector of labels.

    `label_colors(labels)` returns a dict object whose keys are the unique
    values in labels and whose values are the `(r,g,b,a)` colors that should be
    assigned to each label.

    `label_colors(n)` is equivalent to `label_colors(range(n))`.

    Note that this function uses a heuristic and is not guaranteed to be optimal
    in any way for any value of `n`---but it generally works well enough for
    most common purposes.

    Parameters
    ----------
    labels : iterable of ints or int
        Either an integer representing the number of unique labels or an
        interable of the labels themselves. This may be either a collection of
        just the unique labels or a vector of all the labels in a property.
    cmap : colormap or None, optioinal
        Specifies a colormap to use as a base for the label colors. If this is
        `None` (the default), then a varianct of 'hsv' is used.
    
    Returns
    -------
    dict of labels mapped to RGBA tuples
        A dictionary whose keys are the unique labels given by the argument and
        whose values are `(r,g,b,a)` tuples for the label colors.
    """
    from neuropythy.graphics import label_cmap
    if nym.is_numeric(lbls, 'int', ndim=0): lbls = nym.arange(lbls)
    lbls0 = nym.unique(lbls)
    lbls = nym.arange(len(lbls0))
    cm = label_cmap(lbls, cmap=cmap)
    mx = float(len(lbls) - 1)
    m = {k:cm(l/mx) for (k,l) in zip(lbls0, lbls)}
    return m
@pimms.immutable
class LabelEntry(ObjectWithMetaData):
    """A single entry in a `LabelIndex` object that tracks a single label.

    `LabelEntry` is a class tracked by `LabelIndex` objects; it stores
    information about a single label. It is a `pimms.immutable` class.

    Parameters
    ----------
    id : int
        The integer value of the label.
    name : str
        The name of the label.
    color : tuple of float
        The `(r,g,b,a)` color tuple where each value is between 0 and 1.
    meta_data : dict or None, optional
        A mapping of meta-data keys to values.

    Attributes
    ----------
    id : int
        The integer value of the label.
    name : str
        The name of the label.
    color : tuple of float
        The `(r,g,b,a)` color tuple where each value is between 0 and 1.
    meta_data : pyrsistent.PMap
        A persistent mapping of meta-data; if the provided `meta_data` parameter
        was `None`, then this is an empty mapping.
    """
    def __init__(self, ident, name, color=None, meta_data=None):
        self.id = ident
        self.name = name
        self.color = color
        self.meta_data = meta_data
    @pimms.param
    def id(i):
        'le.id is the id of the given label entry object le.'
        if not pimms.is_int(i): raise ValueError('label-entry id must be an int')
        return int(i)
    @pimms.param
    def name(nm):
        'le.name is the (string) name of the given label entry object le.'
        if not pimms.is_str(nm): raise ValueError('label-entry name must be a string')
        return nm
    @pimms.param
    def color(c):
        'le.color is the tuple (r,g,b,a) for the given label entry object le.'
        if c is None: return c
        c = tuple(c)
        if len(c) == 3: c = c + (1,)
        if len(c) != 4: raise ValueError('Invalid color: %s' % c)
        return c
    def __repr__(self):
        return 'label(<%d: %s>)' % (self.id, self.name)
@pimms.immutable
class LabelIndex(ObjectWithMetaData):
    """An index for sets of labels.

    `LabelIndex` is an immutable class that tracks label data and can lookup
    labels by name or integer value as well as assign colors to them.

    Parameters
    ----------
    ids : iterable of ints
        The label values in the label set.
    names : iterable of strings
        The names of the labels.
    colors : matrix of floats or None, optional
        The colors for each of the labels, or `None` for automatic colors.
    entry_meta_data : iterable, optional
        A list of meta-data entries for each label. Each entry may be `None` or
        a mapping.
    meta_data : dict or None, optional
        A mapping of meta-data keys to values.

    Attributes
    ----------
    ids : iterable of ints
        The label values in the label set.
    names : iterable of strings
        The names of the labels.
    colors : matrix of floats
        The colors for each of the labels.
    entry_meta_data : iterable
        A tuple of meta-data entries for each label. Each entry may be `None` or
        a mapping.
    meta_data : pyrsistent.PMap
        A persistent mapping of meta-data; if the provided `meta_data` parameter
        was `None`, then this is an empty mapping.
    entries : tuple of LabelEntry
        A tuple of the label entries, one for each label.
    by_id : mapping
        A mapping whose keys are label ids and whose values are the label
        entries.
    by_name : mapping
        A mapping whose keys are label names and whose values are the label
        entries.
    vmin : float
        A `vmin` value appropriate for properties of the labels tracked by the
        `LabelIndex`.
    vmax : float
        A `vmax` value appropriate for properties of the labels tracked by the
        `LabelIndex`.
    colormap : matplotlib colormap
        A colormap appropriate for properties of the labels tracked by the
        `LabelIndex`.
    """
    def __init__(self, ids, names, colors=None, entry_meta_data=None, meta_data=None):
        self.ids = ids
        self.names = names
        self.colors = colors
        self.entry_meta_data = None
        self.meta_data = meta_data
    @pimms.param
    def ids(ii):
        '''
        lblidx.ids is a tuple of the integer identifiers used for the labels in the given label
        index object.
        '''
        return tuple(ii)
    @pimms.param
    def names(nms):
        '''
        lblidx.names is a tuple of names used for the labels in the given label index object.
        '''
        return tuple(nms)
    @pimms.param
    def colors(cs):
        '''
        lblidx.colors is a numpy array of colors for each label or None.
        '''
        from neuropythy.graphics import to_rgba
        if cs is None: return None
        # we want to convert to colors
        return nym.to_readonly([to_rgba(c) for c in cs])
    @pimms.param
    def entry_meta_data(mds):
        '''
        lblidx.entry_meta_data is lists of meta-data maps for each of the labels in the given label
          index object.
        '''
        if mds is None: return None
        if is_dataframe(mds):
            mds = {k:mds[k].values for k in mds.colums}
        elif is_map(mds):
            ks = list(mds.keys())
            mds = [{k:v for (k,v) in zip(ks,vs)} for vs in np.transpose(list(mds.values()))]
        elif not is_iterable(mds, str=False, set=False):
            raise ValueError('cannot interpret entry meta-data')
        elif not all(is_map(u) or u is None for u in mds):
            raise ValueError('entry meta-data must be maps or None')
        return pimms.persist(mds)
    @pimms.require
    def check_counts(ids, names, colors, entry_meta_data):
        '''
        Checks that ids, names, and colors are the same length and that ids is unique.
        '''
        if len(np.unique(ids)) != len(ids): raise ValueError('label index ids must be unique')
        if len(ids) != len(names): raise ValueError('label index names and ids must be same length')
        if colors is not None and len(colors) != len(ids):
            raise ValueError('label index colors and ids must be same length')
        if entry_meta_data is not None and len(entry_meta_data) != len(ids):
            raise ValueError('label index entry_meta_data and ids must be same length')
        return True
    @pimms.value
    def entries(ids, names, colors, entry_meta_data):
        '''
        lblidx.entries is a tuple of the label entry objects for the given label index object.
        '''
        if 0 not in ids:
            ids = np.concatenate([[0],ids])
            names = ['none'] + list(names)
            if colors          is not None: colors = np.vstack([[(0,0,0,0)], colors])
            if entry_meta_data is not None: entry_meta_data = [None] + list(entry_meta_data)
        if colors is None: colors = np.asarray([cs[k] for cs in [label_colors(ids)] for k in ids])
        if entry_meta_data is None: entry_meta_data = [None]*len(ids)
        # the 0 id is implied if not given:
        les = [LabelEntry(ii, name, color=color, meta_data=md).persist()
               for (ii,name,color,md) in zip(ids, names, colors, entry_meta_data)]
        return tuple(les)
    @pimms.value
    def by_id(entries):
        '''
        lblidx.by_id is a persistent map of the label entries indexed by their identifier.
        '''
        return pyr.pmap({e.id:e for e in entries})
    @pimms.value
    def by_name(entries):
        '''
        lblidx.by_name is a persistent map of the label entries indexed by their names.
        '''
        return pyr.pmap({e.name:e for e in entries})
    @pimms.value
    def vmin(ids):
        '''
        lblidx.vmin is the minimum value of a label identifier in the given label index.
        '''
        return np.min(ids)
    @pimms.value
    def vmax(ids):
        '''
        lblidx.vmax is the maximum value of a label identifier in the given label index.
        '''
        return np.max(ids)
    @pimms.value
    def colormap(entries):
        '''
        lblidx.colormap is a colormap appropriate for use with data that has been scaled to run from
        0 at lblidx.vmin to 1 at lblidx.vmax.
        '''
        import matplotlib.colors
        from_list = matplotlib.colors.LinearSegmentedColormap.from_list
        ids = np.asarray([e.id for e in entries])
        ii  = np.argsort(ids)
        ids = ids[ii]
        clrs = np.asarray([e.color for e in entries])[ii]
        (vmin,vmax) = [f(ids) for f in (np.min, np.max)]
        vals = (ids - vmin) / (vmax - vmin)
        return from_list('label%d' % len(vals), list(zip(vals, clrs)))
    def __getitem__(self, k):
        if   pimms.is_int(k): return self.by_id.get(k, None)
        elif pimms.is_str(k): return self.by_name.get(k, None)
        elif not is_iterable(k, map=False, set=False):
            raise ValueError(f"could not interpret argument to getitem: {k}")
        else:
            k = nym.promote(k)
            u = np.empty_like(k, dtype=np.object)
            if nym.is_numeric(k, 'int'):
                dd = self.by_id
                for kk in k:
                    u[kk] = dd.get(kk)
            else:
                for kk in k:
                    dd = self.by_id if is_str(kk) else self.by_id
                    u[kk] = dd.get(kk)
            return uu
    def name_lookup(self, ii):
        """Looks up names for labels by label ID.

        `lblidx.name_lookup(ii)` returns the names associated with the labels
        with the given ids `ii`. If `ii` is a list of ids, then returnss an
        array of names.

        Parameters
        ----------
        ii : int or array-like of ints
            The id or ids of the label(s) to lookup in the label index.

        Returns
        -------
        str or array of strs
            The names corresponding to the given label id or label ids.
        """
        if is_iteraable(ii, map=False, set=False, str=False):
            ss = [self.by_id[ii].name if jj in self.by_id else None for jj in ii]
            return np.asarray(ss)
        else:
            return self.by_id[ii].name if ii in self.by_id else None
    def id_lookup(self, names):
        """Looks up IDs for labels by label name.

        `lblidx.id_lookup(names)` returns the ids associated with the labels
        with the given `names`. If `names` is a list of names, then returnss an
        array of ids.

        Parameters
        ----------
        names : str or array-like of strs
            The name or names of the label(s) to lookup in the label index.

        Returns
        -------
        int or array of ints
            The ids corresponding to the given label name or label names.
        """
        if is_iteraable(names, map=False, set=False, str=False):
            ss = [self.by_name[jj].name if jj in self.by_name else None for jj in names]
        else:
            ss = self.by_name[ii].name if ii in self.by_name else None
        return np.asarray(ss)
    def color_lookup(self, ii):
        """Looks up colors for labels by label name or label ID.

        `lblidx.color_lookup(names)` returns the colors (RGBA tuples) associated
        with the labels with the name(s) or ID(s) given in the parameter
        `ii`. If `ii` is a list of names, then returnss an matrix of colors.

        Parameters
        ----------
        ii : str or int or vector-like
            The name(s) or id(s) of the label color(s) to lookup in the label
            index.

        Returns
        -------
        vector or matrix of RGBA entries
            Either an RGBA tuple or a matrix of RGBA rows.
        """
        if is_iteraable(ii, map=False, set=False, str=False):
            ss = [((self.by_name[jj].color if jj in self.by_name else (0,0,0,0)) if is_str(jj) else
                   (self.by_id[jj].color if jj in self.by_id else (0,0,0,0)))
                  for jj in ii]
            return np.asarray(ss)
        else:
            if is_str(iii):
                return self.by_name[ii].color if ii in self.by_name else (0,0,0,0)
            else:
                return self.by_id[ii].color if ii in self.by_id else (0,0,0,0)
    def cmap(self, data=None):
        """Returns a scaled colormap for the label index.

        `lblidx.cmap()` returns a colormap for the given label index object that
        assumes that the data being plotted will be rescaled such that label 0
        is 0 and the highest label value in the label index is equal to 1.

        `lblidx.cmap(data)` returns a colormap that will correctly color the
        labels given in data if data is scaled such that its minimum and maximum
        value are 0 and 1.

        Parameters
        ----------
        data : vector of ints or None, optional
            Optionally, the label data to tune the colormap for.

        Returns
        -------
        matplotlib colormap
            A colormap appropriate for the label index.
        """
        import matplotlib.colors
        from_list = matplotlib.colors.LinearSegmentedColormap.from_list
        if data is None: return self.colormap
        data = np.asarray(data).flatten()
        (vmin,vmax) = (np.min(data), np.max(data))
        ii  = np.argsort(self.ids)
        ids = np.asarray(self.ids)[ii]
        if vmin == vmax:
            (vmin,vmax,ii) = (vmin-0.5, vmax+0.5, vmin)
            clr = self.color_lookup(ii)
            return from_list('label1', [(0, clr), (1, clr)])
        q   = (ids >= vmin) & (ids <= vmax)
        ids = ids[q]
        clrs = self.color_lookup(ids)
        vals = (ids - vmin) / (vmax - vmin)
        return from_list('label%d' % len(vals), list(zip(vals, clrs)))
    def __repr__(self):
        return 'LabelIndex(<%d labels>)' % len(self.ids)
def is_label_index(obj):
    """Determines whether an object is a `LabelIndex`.

    `is_label_index(obj)` returns `True` if the given object obj is a label
    index and `False` otherwise.

    Parameters
    ----------
    obj : object
        The object whose quality as a `LabelIndex` is to be assessed.

    Returns
    -------
    `True` if `obj` is a `LabelIndex` and `False` otherwise.
    """
    return isinstance(obj, LabelIndex)
def to_label_index(idxmap, names=None, colors=None, entry_meta_data=None, meta_data=None):
    """Creates and returns a `LabelIndex` object from the arguments.

    `label_index(idxmap)` converts the given map- or dict-like object `idxmap`
    into a label index by assuming that the keys are label ids and the values
    are label names or tuples of label names and `(r,g,b,a)` colors.

    `label_index(ids, names)` uses the given `ids` and `names` to make the label
    index.

    `label_index(ids, names, colors)` additionally uses the given colors.

    Note that if there is not a label with id 0 then such a label is
    automatically created with the name 'none', the RGBA color `(0,0,0,0)`, and
    no entry meta-data. As a general rule, the label 0 should be used to
    indicate that a vertex is out of range for this particular set of labels.

    Parameters
    ----------
    idxmap : mapping or iterable of ints
        Either, if `names` is not provided, an index mapping, or, if `names` is
        provided, an iterable of label ids. The mapping must contain keys that
        are label ids and values that are label names or tuples of `(names,
        (r,g,b,a))` pairs.
    names : iterable of str or None, optional
        The names of the IDs provided in `idxmap` if `idxmap` is not a mapping;
        if `idxmap` is a mapping, then this argument must be `None` or an error
        is raised.
    colors : matrix of RGBA rows or None, optional
        The colors to use for the label index. If `None` (the default), then
        generates colors using `label_colors()`.
    entry_meta_data : None or iterable of mappings
        The meta-data associated with each label entry. If not `None`, then this
        must be an iterable with one entry per labele that is itself either
        `None` or a mapping of meta-data.
    meta_data : dict or None, optional
        A mapping of meta-data keys to values.

    Returns
    -------
    LabelIndex
        A `LabelIndex` object representing the given labels.
    """
    md = meta_data
    mds = entry_meta_data
    dat = idxmap
    if is_map(dat):
        if names is not None:
            raise ValueError("either names must be None or argument must not be a mapping")
        (ids,nms,clrs) = ([],[],[])
        for (k,v) in dat.items():
            if is_str(v): c = None
            else: (v,c) = v
            ids.append(k)
            nms.append(v)
            if c is not None: clrs.append(c)
    elif is_dataframe(dat):
        if dat.index.name.lower() == 'id': ids = dat.index.values
        else: ids = dat['id'].values
        nms = dat['name'].values
        if 'color' in dat: clrs = np.array(list(map(list, dat['color'].values)))
        elif all(k in dat for k in ['r','g','b']):
            ks = ['r','g','b']
            if 'a' in dat: ks.append('a')
            clrs = np.array([[r[k] for k in ks].values for (ii,r) in dat.iterrows()])
        else: clrs = colors
    elif nym.is_numeric(dat, 'int', ndim=1):
        ids = nym.unique(dat)
        nms = names
        clrs = colors
    else:
        raise ValueError(f'label_index() could nor parse first argument: {dat}')
    if nms is Noone: nms = ['label%d'%k for k in ids]
    if clrs is not None:
        if   len(clrs) == 0:        clrs = None
        elif len(clrs) != len(ids): raise ValueError('color-count must match id-count')
    # okay, make the label index
    return LabelIndex(ids, nms, colors=clrs, meta_data=md, entry_meta_data=mds)
label_indices = {}
