####################################################################################################
# neuropythy/hcp/core.py
# Simple tools for use with the Human Connectome Project (HCP) database in Python
# By Noah C. Benson

import numpy                        as np
import nibabel                      as nib
import nibabel.freesurfer.io        as fsio
import nibabel.freesurfer.mghformat as fsmgh
import pyrsistent                   as pyr
import os, warnings, six, pimms

from ..        import geometry      as geo
from ..        import mri           as mri
from ..        import io            as nyio
from ..        import vision        as nyvis

from ..util    import (library_path, curry)
from .files    import (subject_paths, clear_subject_paths, add_subject_path, find_subject_path,
                       to_subject_id, subject_filemap, retinotopy_prefix, lowres_retinotopy_prefix,
                       inferred_retinotopy_prefix, lowres_inferred_retinotopy_prefix,
                       load_retinotopy_cache, save_retinotopy_cache)

# An hcp.Subject is much like an mri.Subject, but its dependency structure all comes from the
# path rather than data provided to the constructor:
@pimms.immutable
class Subject(mri.Subject):
    '''
    A neuropythy.hcp.Subject is an instance of neuropythy.mri.Subject that depends only on
    the path of the subject represented; all other data are automatically derived from this.
    
    Do not call Subject() directly; use neuropythy.hcp.subject() instead.
    '''
    def __init__(self, sid, path, meta_data=None, default_alignment='MSMAll'):
        sid = None if sid is None else to_subject_id(sid)
        path = os.path.abspath(path)
        name = path.split(os.sep)[-1] if sid is None else str(sid)
        # get the name...
        self.subject_id = sid
        self.name = name
        self.path = path
        self.default_alignment = default_alignment
        self.meta_data = meta_data
        # these are the only actually required data for the constructor; the rest is values

    @staticmethod
    def _hmap_has_retinotopy(hmaps, size):
        # by looking at hmaps, check if both hemispheres have retinotopy data
        props = [('%s_%s' % (retinotopy_prefix, s))
                 for s in ['polar_angle', 'eccentricity', 'radius', 'variance_explained']]
        hsuf = '_LR%dk_MSMAll' % size
        return all(p in hmaps[h + hsuf]['properties'] for p in props for h in ('lh','rh'))
    @staticmethod
    def _cortex_setup_native_retinotopy(subdir, sid, hemmap, hmaps, align='MSMAll'):
        # Here, hemmap is the map that will be cortex.hemis and hmaps is filemap['hemis'].
        size = next((s for s in [59,32] if Subject._hmap_has_retinotopy(hmaps, s)), None)
        # okay, size is the best we can do from the database files, but let's see if we have any
        # cache files; only use the 32k cache file if we can't load the 59k database file
        nsuf = '_native_%s' % align
        rp = retinotopy_prefix + '_'
        props = [('%s%s' % (rp, s))
                 for s in ['polar_angle', 'eccentricity', 'radius', 'variance_explained']]
        cache = load_retinotopy_cache(subdir, sid, alignment=align)
        # merge cache in now
        for (hname,hcache) in six.iteritems(cache):
            if hname not in hemmap: continue
            hemmap = hemmap.set(hname, curry(lambda m,n,c:m[n].with_prop(c), hemmap,hname,hcache))
        hfrompat = '%%s_LR%%dk_%s' % align
        # now make a filter that adds the properties when the hemi is loaded
        def check_hemi_retinotopy(hname):
            h = hname[:2]
            def lazy_hemi_fixer():
                tohem = hemmap[hname] # where we interpolate to; also the hem to fix
                # we only interpolate onto native hemispheres:
                if hname.split('_')[1] != 'native': return tohem
                # we want to make a lazy loader of the whole retinotopy dataset here
                def get_interpolated_retinotopy():
                    if get_interpolated_retinotopy.val is not None:
                        return get_interpolated_retinotopy.val
                    m = {}
                    # we interpolate both prf and lowres-prf
                    for (rp,res) in zip([retinotopy_prefix, lowres_retinotopy_prefix], [59,32]):
                        rp = rp + '_'
                        # first, if we have the retinotopy from cache, we can skip this...
                        if all(k in tohem.properties for k in props): continue
                        fromhem = hemmap[hfrompat % (h,res)] # where we interpolate from
                        if fromhem is None: continue
                        fromdat = {k:fromhem.prop(k)
                                   for k in six.iterkeys(fromhem.properties)
                                   if k.startswith(rp)}
                        # if the data isn't in the from hemi, we can't interpolate
                        if len(fromdat) == 0: continue
                        # convert to x/y for interpolation (avoid circular angle mean issues)
                        try: (x,y) = nyvis.as_retinotopy(fromdat, 'geographical', prefix=rp)
                        except Exception: continue
                        del fromdat[rp + 'polar_angle']
                        del fromdat[rp + 'eccentricity']
                        fromdat['x'] = x
                        fromdat['y'] = y
                        todat = fromhem.interpolate(tohem, fromdat, method='linear')
                        # back to visual coords
                        (a,e) = nyvis.as_retinotopy(todat, 'visual')
                        todat = todat.remove('x').remove('y')
                        # save results in our tracking variable, m
                        for (k,v) in six.iteritems(todat):                       m[k]      = v
                        for (k,v) in zip(['polar_angle','eccentricity'], [a,e]): m[rp + k] = v
                    m = pyr.pmap(m)
                    # we can save the cache...
                    get_interpolated_retinotopy.val = m
                    if len(m) > 0: save_retinotopy_cache(subdir, sid, hname, m, alignment=align)
                    return m
                get_interpolated_retinotopy.val = None
                # figure out what properties we'll get from this
                props = []
                for (rp,res) in zip([retinotopy_prefix, lowres_retinotopy_prefix], [59,32]):
                    fromh = hemmap[hfrompat % (h,res)] # where we interpolate from
                    if fromh is None: props = []
                    else: props = props + [k for k in six.iterkeys(fromh.properties)
                                           if k.startswith(rp) and k not in tohem.properties]
                if len(props) == 0: return tohem
                m = pimms.lazy_map({p:curry(lambda k:get_interpolated_retinotopy()[k], p)
                                    for p in props})
                return tohem.with_prop(m)
            return lazy_hemi_fixer
        # now update the hemmap
        return pimms.lazy_map({h:check_hemi_retinotopy(h) for h in six.iterkeys(hemmap)})
    @staticmethod
    def _cortex_from_hemimap(sid, hname, hmap):
        chirality = hname[0:2]
        # get a tess from the registration or surface map; first check for non-lazy or memoized
        # values that don't require loading
        (srfs, regs) = [hmap[k] for k in ('surfaces','registrations')]
        tess = next((m[k] for m in (srfs,regs) for k in six.iterkeys(m) if not m.is_lazy(k)), None)
        if tess is None: tess = regs['fs_LR']
        tess = tess.tess.with_prop(hmap['properties'])
        return mri.Cortex(chirality, tess, srfs, regs)
    @pimms.param
    def subject_id(sid):
        '''
        sub.subject_id is the 6-digit HCP subject id of the subject sub, or None if sub was loaded
        from a non-standard source.
        '''
        if sid is None: return None
        try: sid = int(sid)
        except Exception: raise ValueError('Subject IDs must be 6-digit ints or strings')
        if sid < 100000 or sid > 999999: raise ValueError('Subject IDs must be 6 digits')
        return sid
    @pimms.param
    def default_alignment(da):
        '''
        sub.default_alignment is either 'MSMAll' or 'MSMSulc' and specifies what alignment algorithm
        is used as the default hemisphere set. For example, if this is 'MSMAll', then sub.lh_native
        will be identical to sub.lh_native_MSMAll instead of to sub.lh_native_MSMSulc.
        '''
        if da.lower() in ('msmsulc', 'sulc'): return 'MSMSulc'
        elif da.lower() in ('msmall', 'all'): return 'MSMAll'
        else: raise ValueError('default_alignment must be MSMAll or MSMSulc')
    @pimms.value
    def filemap(path):
        '''
        sub.filemap is a lazily-loading map of the files imported for the given sub. It should
        generally not be necessary to access this value directly.
        '''
        return subject_filemap(path)
    @pimms.value
    def hemis(path, subject_id, filemap, default_alignment):
        '''
        sub.hemis is a persistent map of hemispheres/cortex objects for the given HCP subject sub.
        HCP subjects have many hemispheres of the name <chirality>_<topology>_<alignment> where
        chirality is lh or rh, topology is native or an atlas name, and alignment is MSMAll or
        MSMSulc. The lh and rh hemispheres are aliases for lh_native and rh_native. The
        default_alignment of the subject determines whether MSMAll or MSMSulc alignments are aliased
        as <chirality>_<alignment> hemispheres.
        '''
        sid = subject_id
        hmaps = filemap['hemis']
        def _ctx_loader(k):
            def _f():
                try: return Subject._cortex_from_hemimap(sid, k, hmaps[k])
                except Exception: return None
            return _f
        hemmap0 = pimms.lazy_map({k:_ctx_loader(k) for k in six.iterkeys(hmaps)})
        # one special thing: we can auto-load retinotopy from another hemisphere and interpolate it
        # onto the native hemisphere; set that up if possible (note that retinotopy is always
        # aligned to MSMAll):
        hemmap0 = Subject._cortex_setup_native_retinotopy(path, sid, hemmap0, hmaps, 'MSMAll')
        # now setup the aliases
        def _ctx_lookup(k): return lambda:hemmap0[k]
        hemmap = {}
        da = '_' + default_alignment
        for k in six.iterkeys(hmaps):
            hemmap[k] = _ctx_lookup(k)
            if k.endswith(da):
                kk = k[:-len(da)]
                hemmap[kk] = hemmap[k]
                if kk.endswith('_native'): hemmap[k[0:2]] = hemmap[k]
        return pimms.lazy_map(hemmap)
    @pimms.value
    def images(filemap):
        '''
        sub.images is a persistent map of MRImages tracked by the given subject sub; in HCP subjects
        these are renamed and converted from their typical HCP filenames (such as 'ribbon') to forms
        that conform to the neuropythy naming conventions (such as 'gray_mask').
        '''
        imgmap = filemap['images']
        def _img_loader(k): return lambda:imgmap[k]
        imgs = {k:_img_loader(k) for k in six.iterkeys(imgmap)}
        def _make_mask(val, eq=True):
            rib = imgmap['ribbon']
            arr = (rib.get_data() == val) if eq else (rib.get_data() != val)
            arr.setflags(write=False)
            return type(rib)(arr, rib.affine, rib.header)
        imgs['lh_gray_mask']  = lambda:_make_mask(3)
        imgs['lh_white_mask'] = lambda:_make_mask(2)
        imgs['rh_gray_mask']  = lambda:_make_mask(42)
        imgs['rh_white_mask'] = lambda:_make_mask(41)
        imgs['brain_mask']    = lambda:_make_mask(0, False)
        return pimms.lazy_map(imgs)
    @pimms.value
    def voxel_to_vertex_matrix(images):
        '''
        See neuropythy.mri.Subject.voxel_to_vertex_matrix.
        '''
        return pimms.imm_array(images['ribbon'].affine)
    @pimms.value
    def voxel_to_native_matrix(images):
        '''
        See neuropythy.mri.Subject.voxel_to_native_matrix.
        '''
        return pimms.imm_array(np.eye(4))

@nyio.importer('hcp_subject')
def subject(sid, subjects_path=None, meta_data=None, default_alignment='MSMAll'):
    '''
    subject(sid) yields a HCP Subject object for the subject with the given subject id; sid may be a
      path to a subject or a subject id, in which case the subject paths are searched for it.
    subject(None, path) yields a non-standard HCP subject at the given path.
    subject(sid, path) yields the specific HCP Subject at the given path.

    Subjects are cached and not reloaded.  Note that subects returned by subject() are always
    persistent Immutable objects; this means that you must create a transient version of the subject
    to modify it via the member function sub.transient(). Better, you can make copies of the objects
    with desired modifications using the copy method.

    This function works with the neuropythy.hcp.auto_download() function; if you have enabled auto-
    downloading, then subjects returned from this subject may be downloading themselves lazily.
    '''
    if subjects_path is None:
        if os.path.isdir(str(sid)):
            (fdir, fnm) = os.path.split(str(sid))
            try: sid = to_subject_id(fnm)
            except Exception: sid = None
            pth = fdir
        else:
            sid = to_subject_id(sid)
            fnm = str(sid)
            fdir = find_subject_path(sid)
            if fdir is None: raise ValueError('Could not locate subject with id \'%s\'' % sid)
            pth = os.path.split(fdir)[0]
    else:
        if sid is None:
            (pth, fnm) = os.path.split(subjects_path)
        else:
            sid = to_subject_id(sid)
            fnm = str(sid)
            fdir = subjects_path
    fdir = os.path.abspath(os.path.join(pth, fnm))
    if fdir in subject._cache: return subject._cache[fdir]
    sub = Subject(sid, fdir, meta_data=meta_data, default_alignment=default_alignment).persist()
    if isinstance(sub, Subject): subject._cache[fdir] = sub
    return sub
subject._cache = {}
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
    

    
    
    
