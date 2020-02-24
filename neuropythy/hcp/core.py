####################################################################################################
# neuropythy/hcp/core.py
# Simple tools for use with the Human Connectome Project (HCP) database in Python
# By Noah C. Benson

import numpy                        as np
import nibabel                      as nib
import nibabel.freesurfer.io        as fsio
import nibabel.freesurfer.mghformat as fsmgh
import pyrsistent                   as pyr
from   six.moves                import collections_abc as collections
import os, warnings, six, pimms

from ..        import geometry      as geo
from ..        import mri           as mri
from ..        import io            as nyio
from ..        import vision        as nyvis

from ..util    import (library_path, curry, config, to_pseudo_path, is_pseudo_path)
from .files    import (subject_paths, clear_subject_paths, add_subject_path, find_subject_path,
                       subject_file_map, is_hcp_subject_path)

def to_default_alignment_value(x):
    if not pimms.is_str(x): raise ValueError('hcp_default_alignment must be a string')
    x = x.lower()
    if   x in ('msmsulc', 'sulc'):  x = 'MSMSulc'
    elif x in ('msmall', 'all'):    x = 'MSMAll'
    elif x in ('fs', 'freesurfer'): x = 'FS'
    else: raise ValueError('invalid value for hcp_default_alignment: %s')
    return x
config.declare('hcp_default_alignment', environ_name='HCP_DEFAULT_ALIGNMENT',
               filter=to_default_alignment_value, default_value='MSMAll')
def cortex_from_filemap(fmap, name, affine=None):
    '''
    cortex_from_filemap(filemap, name) yields a cortex object from the given filemap.
    '''
    chirality = name[:2].lower()
    # get the relevant hemi-data
    hdat = fmap.data_tree.hemi[name]
    # we need the tesselation at build-time, so let's create that now:
    tris = hdat.tess['white']
    # this tells us the max number of vertices
    n = np.max(tris) + 1
    # Properties: we want to merge a bunch of things together...
    # for labels, weights, annots, we need to worry about loading alts:
    def _load_with_alt(k, s0, sa, trfn):
        if s0 is not None:
            try: u = s0.get(k, None)
            except Exception: u = None
        else: u = None
        if u is None and sa is not None:
            try: u = sa.get(k, None)
            except Exception: u = None
        if u is None: raise ValueError('Exception while loading property %s' % k)
        else: return u if trfn is None else trfn(u)
    def _lbltr(ll):
        l = np.zeros(n, dtype='bool')
        l[ll[0]] = True
        l.setflags(write=False)
        return l
    def _wgttr(ll):
        w = np.zeros(n, dtype='float')
        w[ll[0]] = ll[1]
        w.setflags(write=False)
        return w
    def _anotr(ll):
        ll[0].setflags(write=False)
        return ll[0]
    p = {}
    from itertools import chain
    l = hdat.label if hasattr(hdat, 'label') else {}
    al = hdat.alt_label if hasattr(hdat, 'alt_label') else {}
    for k in set(chain(six.iterkeys(l), six.iterkeys(al))):
        p[k+'_label'] = curry(_load_with_alt, k, l, al, _lbltr)
    w = hdat.weight if hasattr(hdat, 'weight') else {}
    aw = hdat.alt_weight if hasattr(hdat, 'alt_weight') else {}
    for k in set(chain(six.iterkeys(w), six.iterkeys(aw))):
        p[k+'_weight'] = curry(_load_with_alt, k, w, aw, _wgttr)
    a = hdat.annot if hasattr(hdat, 'annot') else {}
    aa = hdat.alt_annot if hasattr(hdat, 'alt_annot') else {}
    for k in set(chain(six.iterkeys(a), six.iterkeys(aa))):
        p[k] = curry(_load_with_alt, k, a, aa, _anotr)
    props = pimms.merge(hdat.property if hasattr(hdat, 'property') else {}, pimms.lazy_map(p))
    tess = geo.Tesselation(tris, properties=props)
    # if this is a subject that exists in the library, we may want to add some files:
    if name is None:
        pd = fmap.pseudo_paths[None]._path_data
        name = pd['pathmod'].split(fmap.actual_path)[1]
    regs = hdat.registration if hasattr(hdat, 'registration') else {}
    # Okay, make the cortex object!
    md = {'file_map': fmap}
    if name is not None: md['subject_id'] = name
    srfs = hdat.surface if hasattr(hdat, 'surface') else {}
    return mri.Cortex(chirality, tess, srfs, regs, affine=affine, meta_data=md).persist()
def images_from_filemap(fmap):
    '''
    images_from_filemap(fmap) yields a persistent map of MRImages tracked by the given subject with
      the given name and path; in freesurfer subjects these are renamed and converted from their
      typical freesurfer filenames (such as 'ribbon') to forms that conform to the neuropythy naming
      conventions (such as 'gray_mask'). To access data by their original names, use the filemap.
    '''
    imgmap = fmap.data_tree.image
    def img_loader(k): return lambda:imgmap[k]
    imgs = {k:img_loader(k) for k in six.iterkeys(imgmap)}
    def _make_mask(val, eq=True):
        rib = imgmap['ribbon']
        img = np.asarray(rib.dataobj)
        arr = (img == val) if eq else (img != val)
        arr.setflags(write=False)
        return type(rib)(arr, rib.affine, rib.header)
    imgs['lh_gray_mask']  = lambda:_make_mask(3)
    imgs['lh_white_mask'] = lambda:_make_mask(2)
    imgs['rh_gray_mask']  = lambda:_make_mask(42)
    imgs['rh_white_mask'] = lambda:_make_mask(41)
    imgs['brain_mask']    = lambda:_make_mask(0, False)
    # merge in with the typical images
    return pimms.merge(fmap.data_tree.image, pimms.lazy_map(imgs))
def subject_from_filemap(fmap, name=None, meta_data=None, check_path=True,
                         default_alignment='MSMAll'):
    '''
    subject_from_filemap(fmap) yields an HCP subject from the given filemap.
    '''
    # start by making a pseudo-dir:
    if check_path and not is_hcp_subject_path(fmap.pseudo_paths[None]):
        raise ValueError('given path does not appear to hold an HCP subject')
    # we need to go ahead and load the ribbon...
    rib = fmap.data_tree.image['ribbon']
    vox2nat = rib.affine
    # make images and hems
    imgs = images_from_filemap(fmap)
    # many hemispheres to create:
    hems = pimms.lazy_map({h:curry(cortex_from_filemap, fmap, h)
                           for h in ['lh_native_MSMAll',  'rh_native_MSMAll',
                                     'lh_nat32k_MSMAll',  'rh_nat32k_MSMAll',
                                     'lh_nat59k_MSMAll',  'rh_nat59k_MSMAll',
                                     'lh_LR32k_MSMAll',   'rh_LR32k_MSMAll',
                                     'lh_LR59k_MSMAll',   'rh_LR59k_MSMAll',
                                     'lh_LR164k_MSMAll',  'rh_LR164k_MSMAll',
                                     'lh_native_MSMSulc', 'rh_native_MSMSulc',
                                     'lh_nat32k_MSMSulc', 'rh_nat32k_MSMSulc',
                                     'lh_nat59k_MSMSulc', 'rh_nat59k_MSMSulc',
                                     'lh_LR32k_MSMSulc',  'rh_LR32k_MSMSulc',
                                     'lh_LR59k_MSMSulc',  'rh_LR59k_MSMSulc',
                                     'lh_LR164k_MSMSulc', 'rh_LR164k_MSMSulc',
                                     'lh_native_FS',      'rh_native_FS',
                                     'lh_nat32k_FS',      'rh_nat32k_FS',
                                     'lh_nat59k_FS',      'rh_nat59k_FS',
                                     'lh_LR32k_FS',       'rh_LR32k_FS',
                                     'lh_LR59k_FS',       'rh_LR59k_FS',
                                     'lh_LR164k_FS',      'rh_LR164k_FS']})
    # now, setup the default alignment aliases:
    if default_alignment is not None:
        for h in ['lh_native',  'rh_native',  'lh_nat32k',  'rh_nat32k',
                  'lh_nat59k',  'rh_nat59k',  'lh_LR32k',   'rh_LR32k',
                  'lh_LR59k',   'rh_LR59k',   'lh_LR164k',  'rh_LR164k']:
            hems = hems.set(h, curry(lambda h:hems[h+'_'+default_alignment], h))
        hems = hems.set('lh', lambda: hems['lh_native'])
        hems = hems.set('rh', lambda: hems['rh_native'])
    meta_data = pimms.persist({} if meta_data is None else meta_data)
    meta_data = meta_data.set('raw_images', fmap.data_tree.raw_image)
    if default_alignment is not None:
        meta_data = meta_data.set('default_alignment', default_alignment)
    return mri.Subject(name=name, pseudo_path=fmap.pseudo_paths[None],
                       hemis=hems, images=imgs,
                       meta_data=meta_data).persist()
@nyio.importer('hcp_subject', sniff=is_hcp_subject_path)
def subject(path, name=Ellipsis, meta_data=None, check_path=True, filter=None,
            default_alignment=Ellipsis):
    '''
    subject(name) yields an HCP-based Subject object for the subject with the given name or path.
      Subjects are cached and not reloaded, so multiple calls to subject(name) will yield the same
      immutable subject object.

    The name argument is allowed to take a variety of forms:
      * a local (absolute or relative) path to a valid HCP subject directory
      * a url or pseudo-path to a valid HCP subject
      * an integer, in which case the neuropythy.data['hcp'] dataset is used (i.e., the subject data
        are auto-downloaded from the HCP Amazon-S3 bucket as required)
    If you request a subject by path, the HCP module has no way of knowing for sure if that subject
    should be auto-downloaded, so is not; if you want subjects to be auto-downloaded from the HCP
    database, you should represent the subjects by their integer ids.

    Note that subects returned by hcp_subject() are always persistent Immutable objects; this
    means that you must create a transient version of the subject to modify it via the member
    function sub.transient(). Better, you can make copies of the objects with desired modifications
    using the copy method--see the pimms library documentation regarding immutable classes and
    objects.

    If you wish to modify all subjects loaded by this function, you may set its attribute 'filter'
    to a function or list of functions that take a single argument (a subject object) and returns a
    single argument (a potentially-modified subject object).

    The argument name may alternately be a pseudo_path object or a path that can be converted into a
    pseudo_path object.

    The following options are accepted:
      * name (default: Ellipsis) may optionally specify the subject's name; if Ellipsis, then
        attempts to deduce the name from the initial argument (which may be a name or a path).
      * meta_data (default: None) may optionally be a map that contains meta-data to be passed along
        to the subject object (note that this meta-data will not be cached).
      * check_path (default: True) may optionally be set to False to ignore the requirement that a
        directory contain at least the mri/, label/, and surf/ directories to be considered a valid
        HCP subject directory. Subject objects returned when this argument is not True are not
        cached. Additionally, check_path may be set to None instead of False, indicating that no
        sanity checks or search should be performed whatsoever: the string name should be trusted 
        to be an exact relative or absolute path to a valid HCP subejct.
      * filter (default: None) may optionally specify a filter that should be applied to the subject
        before returning. This must be a function that accepts as an argument the subject object and
        returns a (potentially) modified subject object. Filtered subjects are cached by using the
        id of the filters as part of the cache key.
      * default_alignment (default: Ellipsis) specifies the default alignment to use with HCP
        subjects; this may be either 'MSMAll', 'MSMSulc', or 'FS'; the deafult (Ellipsis) indicates
        that the 'hcp_default_alignment' configuration value should be used (by default this is
        'MSMAll').
    '''
    from neuropythy import data
    if pimms.is_str(default_alignment):
        default_alignment = to_default_alignment_value(default_alignment)
    elif default_alignment in (Ellipsis, None):
        default_alignment = config['hcp_default_alignment']
    # first thing: if the sid is an integer, we try to get the subject from the hcp dataset;
    # in this case, because the hcp dataset actually calls down through here (with a pseudo-path),
    # we don't need to run any of the filters that are run below (they have already been run)
    if pimms.is_int(path):
        if default_alignment != 'MSMAll':
            warnings.warn('%s alignment requested, but MSMAll used for HCP release subject'
                          % (default_alignment,))
        try: return data['hcp'].subjects[path]
        except Exception: pass
    # convert the path to a pseudo-dir; this may fail if the user is requesting a subject by name...
    try: pdir = to_pseudo_path(path)
    except Exception: pdir = None
    if pdir is None: # search for a subject with this name
        tmp = find_subject_path(path, check_path=check_path)
        if tmp is not None:
            pdir = to_pseudo_path(tmp)
            path = tmp
    if pdir is None:
        if default_alignment != 'MSMAll':
            warnings.warn('%s alignment requested, but MSMAll used for HCP release subject'
                          % (default_alignment,))
        # It's possible that we need to check the hcp dataset
        try: return data['hcp'].subjects[int(path)]
        except: pass
        raise ValueError('could not find HCP subject: %s' % (path,))
    path = pdir.actual_source_path
    # okay, before we continue, lets check the cache...
    tup = (path, default_alignment)
    if tup in subject._cache: sub = subject._cache[tup]
    else:
        # extract the name if need-be
        if name is Ellipsis:
            import re
            (pth,name) = (path, '.')
            while name == '.': (pth, name) = pdir._path_data['pathmod'].split(pth)
            name = name.split(':')[-1]
            name = pdir._path_data['pathmod'].split(name)[1]
            if '.tar' in name: name = name.split('.tar')[0]
        # make the filemap
        fmap = subject_file_map(pdir, name=name)
        # and make the subject!
        sub = subject_from_filemap(fmap, name=name, check_path=check_path, meta_data=meta_data,
                                   default_alignment=default_alignment)
        if mri.is_subject(sub):
            sub = sub.persist()
            sub = sub.with_meta(file_map=fmap)
            subject._cache[(path, default_alignment)] = sub
    # okay, we have the initial subject; let's organize the filters
    if pimms.is_list(subject.filter) or pimms.is_tuple(subject.filter): filts = list(subject.filter)
    else: filts = []
    if pimms.is_list(filter) or pimms.is_tuple(filter): filter = list(filter)
    else: filter = []
    filts = filts + filter
    if len(filts) == 0: return sub
    fids = tuple([id(f) for f in filts])
    tup = fids + (path, default_alignment)
    if tup in subject._cache: return subject._cache[tup]
    for f in filts: sub = f(sub)
    if mri.is_subject(sub): subject._cache[tup] = sub
    return sub.persist()
subject._cache = {}
subject.filter = None

def forget_subject(sid):
    '''
    forget_subject(sid) causes neuropythy's hcp module to forget about cached data for the subject
      with subject id sid. The sid may be any sid that can be passed to the subject() function.
    
    This function is useful for batch-processing of subjects in a memory-limited environment; e.g.,
    if you run out of memory while processing hcp subjects it is possibly because neuropythy is
    caching all of their data instead of freeing it.
    '''
    sub = subject(sid)
    if sub.path in subject._cache:
        del subject._cache[sub.path]
    else:
        for (k,v) in six.iteritems(subject._cache):
            if v is sub:
                del subject._cache[k]
                break
    return None
def forget_all():
    '''
    forget_all() causes neuropythy's hcp module to forget all cached subjects. See also
    forget_subject.
    '''
    subject._cache = {}
    return None
    
def download(sid):
    '''
    neuropythy.hcp.download(sid) is equivalent to neuropythy.data['hcp'].download(sid).
    '''
    import neuropythy as ny
    return ny.data['hcp'].download(sid)

# This is copied from ny.freesurfer.core; changes here should be duplicated there. This isn't really
# a good way to organize this--#TODO is to unify the subject-path interfaces somehow.
class SubjectDir(collections.Mapping):
    '''
    SubjectsDir objects are dictionary-like objects that represent a particular subject directory.
    They satisfy their queries (such as `111312 in spath`) by querying the filesystem itself.

    For more information see the subjects_path function.
    '''
    def __init__(self, path, bids=False, filter=None, meta_data=None, check_path=True):
        path = os.path.expanduser(os.path.expandvars(path))
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path): raise ValueError('given path is not a directory')
        self.bids = bool(bids)
        self.options = dict(filter=filter, meta_data=meta_data, check_path=bool(check_path))
    def __contains__(self, sub):
        if pimms.is_int(sub): sub = str(sub)
        # first check the item straight-up:
        sd = os.path.join(self.path, sub)
        if os.path.isdir(sd) and (not check_path or is_hcp_subject_path(sd)): return True
        if not bids: return False
        if sub.startswith('sub-'): sd = os.path.join(self.path, sub[4:])
        else: sd = os.path.join(self.path, 'sub-' + sub)
        return os.path.isdir(sd) and (not check_path or is_hcp_subject_path(sd))
    def _get_subject(self, sd, name):
        try:
            if name == str(int(name)): name = int(name)
        except Exception: pass
        opts = dict(**self.options)
        opts['name'] = name
        return subject(sd, **opts)
    def __getitem__(self, sub):
        if pimms.is_int(sub): sub = str(sub)
        check_path = self.options['check_path']
        if self.bids:
            if sub.startswith('sub-'): (sub, name) = (sub, name[4:])
            else: (sub,name) = ('sub-' + sub, sub)
        else: name = sub
        sd = os.path.join(self.path, sub)
        if os.path.isdir(sd) and (not check_path or is_hcp_subject_path(sd)):
            return self._get_subject(sd, name)
        if not bids: return False
        # try without the 'sub-' (e.g. for fsaverage)
        sub = name
        sd = os.path.join(self.path, sub)
        if os.path.isdir(sd) and (not check_path or is_hcp_subject_path(sd)):
            return self._get_subject(sd, name)
        raise KeyError(sub)
    def asmap(self):
        check_path = self.options['check_path']
        # search through the subjects in this directory
        res = {}
        for sub in os.listdir(self.path):
            if self.bids and sub.startswith('sub-'): (sub,name) = (sub,sub[4:])
            else: name = sub
            sdir = os.path.join(self.path, sub)
            if not check_path or is_hcp_subject_path(sdir):
                res[name] = curry(self._get_subject, sdir, name)
        return pimms.lmap(res)
    def __len__(self): return len(self.asmap())
    def __iter__(self): return iter(self.asmap())
    def __repr__(self): return 'freesurfer.SubjectsPath(' + repr(self.asmap()) + ')'
    def __str__(self): return 'freesurfer.SubjectsPath(' + str(self.asmap()) + ')'
# Functions for handling freesurfer subject directories themselves
def is_hcp_subject_dir_path(path):
    '''
    is_hcp_subject_dir_path(path) yields True if path is a directory that contains at least
      one HCP subejct subdirectory.
    '''
    if not os.path.isdir(path): return False
    for p in os.listdir(path):
        pp = os.path.join(path, p)
        if not os.path.isdir(pp): continue
        if is_hcp_subject_path(pp): return True
    return False
@nyio.importer('hcp_subject_dir', sniff=is_hcp_subject_dir_path)
def subject_dir(path, bids=False, filter=None, meta_data=None, check_path=True):
    '''
    subject_dir(path) yields a dictionary-like object containing the subjects in the FreeSurfer
      subjects directory given by path.

    The following optional arguments are accepted:
      * bids (default: False) may be set to True to indicate that the directory is part of a BIDS
        directory; in this case the 'sub-' prefix is stripped from directory names to deduce subject
        names.
      * check_path (default: True) may optionally be set to False to ignore the requirement that a
        subject directory contain at least the mri/, label/, and surf/ directories to be considered
        a valid FreeSurfer subject directory. See help(neuropythy.freesurfer.subject) for more
        information.
      * filter (default: None) may optionally specify a filter that should be applied to the subject
        before returning. See help(neuropythy.freesurfer.subject) for more information.
      * meta_data (default: None) may optionally be a map that contains meta-data to be passed along
        to the subject object (note that this meta-data will not be cached).
    '''
    return SubjectDir(path, bids=bids, filter=filter, meta_data=meta_data, check_path=check_path)
    
    
