####################################################################################################
# neuropythy/hcp/core.py
# Simple tools for use with the Human Connectome Project (HCP) database in Python
# By Noah C. Benson

import numpy                        as np
import nibabel                      as nib
import nibabel.freesurfer.io        as fsio
import nibabel.freesurfer.mghformat as fsmgh
import pyrsistent                   as pyr
import neuropythy.geometry          as geo
import neuropythy.mri               as mri
import neuropythy.io                as nyio
import os, warnings, six, pimms

from neuropythy.util import library_path
from .files import (subject_paths, clear_subject_paths, add_subject_path, find_subject_path,
                    to_subject_id, to_credentials, load_credentials, detect_credentials,
                    subject_filemap)

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
        except: raise ValueError('Subject IDs must be 6-digit ints or strings representing them')
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
    def hemis(subject_id, filemap, default_alignment):
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
        def _ctx_loader(k): return lambda:Subject._cortex_from_hemimap(sid, k, hmaps[k])
        hemmap0 = pimms.lazy_map({k:_ctx_loader(k) for k in six.iterkeys(hmaps)})
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
        imgs['lh_gray_mask']  = lambda:_make_mask(42)
        imgs['lh_white_mask'] = lambda:_make_mask(41)
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
            except: sid = None
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

    
    
    
