####################################################################################################
# neuropythy/util/labels.py
# Simple tools for dealing with neuroscience-related labels for brains.
# By Noah C. Benson

import numpy               as np
import pyrsistent          as pyr
import collections         as colls
import os, sys, types, six, pimms

from .core import (ObjectWithMetaData, is_tuple, curry, is_dataframe)

def label_colors(lbls, cmap=None):
    '''
    label_colors(labels) yields a dict object whose keys are the unique values in labels and whose
      values are the (r,g,b,a) colors that should be assigned to each label.
    label_colors(n) is equivalent to label_colors(range(n)).

    Note that this function uses a heuristic and is not guaranteed to be optimal in any way for any
    value of n--but it generally works well enough for most common purposes.
    
    The following optional arguments may be given:
      * cmap (default: None) specifies a colormap to use as a base. If this is None, then a varianct
        of 'hsv' is used.
    '''
    from neuropythy.graphics import label_cmap
    if pimms.is_int(lbls): lbls = np.arange(lbls)
    lbls0 = np.unique(lbls)
    lbls = np.arange(len(lbls0))
    cm = label_cmap(lbls, cmap=cmap)
    mx = float(len(lbls) - 1)
    m = {k:cm(l/mx) for (k,l) in zip(lbls0, lbls)}
    return m
@pimms.immutable
class LabelEntry(ObjectWithMetaData):
    '''
    LabelEntry is a class tracked by LabelIndex objects; it stores information about a single
    label.
    '''
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
    '''
    LabelIndex is an immutable class that tracks label data and can lookup labels by name or integer
    value as well as assign colors to them.
    '''
    def __init__(self, ids, names, colors=None, entry_meta_data=None, meta_data=None):
        '''
        LabelIndex(ids, names) constructs a label index object with the given ids and names.
        '''
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
        return pimms.imm_array([to_rgba(c) for c in cs])
    @pimms.param
    def entry_meta_data(mds):
        '''
        lblidx.entry_meta_data is lists of meta-data maps for each of the labels in the given label
          index object.
        '''
        if mds is None: return None
        if is_dataframe(mds):
            mds = {k:mds[k].values for k in mds.colums}
        elif pimms.is_map(mds):
            ks = list(mds.keys())
            mds = [{k:v for (k,v) in zip(ks,vs)} for vs in np.transpose(list(mds.values()))]
        elif not pimms.is_array(mds) or not all(pimms.is_map(u) for u in mds):
            raise ValueError('unbalanced or non-map entry meta-data')
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
        if   pimms.is_int(k):              return self.by_id.get(k, None)
        elif pimms.is_str(k):              return self.by_name.get(k, None)
        elif pimms.is_vector(k, 'int'):    return np.asarray([self.by_id.get(k, None) for k in k])
        else: return np.asarray([(self.by_name if pimms.is_str(k) else self.by_id).get(k, None)
                                 for k in k])
    def name_lookup(self, ii):
        '''
        lblidx.name_lookup(ii) yields the names associated with the labels with the given ids. If
          ii is a list of ids, then yields an array of names.
        '''
        if   pimms.is_int(ii): return self.by_id[ii].name if ii in self.by_id else None
        elif pimms.is_str(ii): return self.by_name[ii].name if ii in self.by_name else None
        else: return np.asarray([tbl[ii].name if ii in tbl else None
                                 for ii in ii
                                 for tbl in [self.by_name if pimms.is_str(ii) else self.by_id]])
    def id_lookup(self, names):
        '''
        lblidx.id_lookup(names) yields the ids associated with the labels with the given names. If
          names is a list of names, then yields an array of ids.
        '''
        if   pimms.is_str(names): return self.by_name[names].id if names in self.by_name else None
        elif pimms.is_int(names): return self.by_id[names].id if names in self.by_id else None
        else: return np.asarray([tbl[ii].id if ii in tbl else None
                                 for ii in names
                                 for tbl in [self.by_name if pimms.is_str(ii) else self.by_id]])
    def color_lookup(self, ii):
        '''
        lblidx.color_lookup(ids) yields the color(s) associated with the labels with the given ids.
          If ids is a list of ids, then yields a matrix of colors.
        lblidx.color_lookup(names) uses the names to lookup the label colors.
        '''
        if pimms.is_int(ii): return self.by_id[ii].color if ii in self.by_id else None
        elif pimms.is_str(ii): return self.by_name[ii].color if ii in self.by_name else None
        else: return np.asarray([tbl[ii].color if ii in tbl else None
                                 for ii in ii
                                 for tbl in [self.by_name if pimms.is_str(ii) else self.by_id]])
    def cmap(self, data=None):
        '''
        lblidx.cmap() yields a colormap for the given label index object that assumes that the data
          being plotted will be rescaled such that label 0 is 0 and the highest label value in the
          label index is equal to 1.
        lblidx.cmap(data) yields a colormap that will correctly color the labels given in data if
          data is scaled such that its minimum and maximum value are 0 and 1.
        '''
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
def is_label_index(le):
    '''
    is_label_index(le) yields True if the given object le is a label index and False otherwise.
    '''
    return isinstance(le, LabelIndex)
def label_index(dat, *args, **kw):
    '''
    label_index(idx_map) converts the given map- or dict-like object idx_map into a label index by
      assuming that the keys are label ids and the values are label names or tuples of label names
      and (r,g,b,a) colors.
    label_index(ids, names) uses the given ids and names to make the label index.
    label_index(ids, names, colors) additionally uses the given colors.

    Note that if there is not a label with id 0 then such a label is automatically created with the
    name 'none', the rgba color [0,0,0,0], and no entry meta-data. As a general rule, the label 0
    should be used to indicate that a label is missing.

    The optional arguments meta_data and entry_meta_data may specify both the meta-data for the
    label-index object itself as well as the meta-data for the individual entries.
    '''
    md = kw.pop('meta_data', {})
    mds = kw.pop('entry_meta_data', None)
    if len(kw) > 0: raise ValueError('unrecognized optional argument(s) given to label_index')
    if len(args) == 0:
        if pimms.is_map(dat):
            (ids,nms,clrs) = ([],[],[])
            for (k,v) in six.iteritems(dat):
                if pimms.is_scalar(v): c = None
                else: (v,c) = v
                if pimms.is_str(k):
                    ids.append(v)
                    nms.append(k)
                else:
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
            else: clrs = []
        elif pimms.is_vector(dat, 'int'):
            ids = np.unique(dat)
            nms = ['label%d'%k for k in ids]
            clrs = []
        else: raise ValueError('label_index(idx_map) given non-map argument')
    elif len(args) == 1: (ids,nms,clrs) = (dat, args[0], [])
    elif len(args) == 2: (ids,nms,clrs) = (dat, args[0], args[1])
    else: raise ValueError('Too many arguments given to label_index()')
    if clrs is None or len(clrs) == 0: clrs = None
    elif len(clrs) != len(ids): raise ValueError('color-count must match id-count')
    # okay, make the label index
    return LabelIndex(ids, nms, colors=clrs, meta_data=md, entry_meta_data=mds)
def to_label_index(obj):
    '''
    to_label_index(obj) attempts to coerce the given object into a label index object; if obj is
      already a label index object, then obj itself is returned. If obj cannot be coerced into a
      label index, then an error is raised.

    The obj argument can be any of the following:
      * a label index
      * a label list (i.e., an integer vector)
      * a tuple of arguments, potentially ending with a kw-options map, that can be passed to the
        label_index function successfully.
    '''
    if   is_label_index(obj): return obj
    elif pimms.is_vector(obj, 'int'): return label_index(obj)
    elif is_dataframe(obj): return label_index(obj)
    elif is_tuple(obj):
        if len(obj) > 1 and pimms.is_map(obj[-1]): return label_index(*obj[:-1], **obj[-1])
        else: return label_index(*obj)
    else: raise ValueError('could not parse to_label_index parameter: %s' % obj)
label_indices = {}
