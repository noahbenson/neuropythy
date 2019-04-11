####################################################################################################
# neuropythy/hcp/files.py
# Stored data regarding the organization of the files in HCP subjects.
# by Noah C. Benson

import os, six, logging, pimms, pyrsistent as pyr, nibabel as nib, numpy as np
from .. import io as nyio
from ..util import (config, is_image, to_credentials, file_map)

# this isn't required, but if we can load it we will use it for auto-downloading subject data
try:              import s3fs
except Exception: s3fs = None

####################################################################################################
# Subject Directory and where to find Subjects

   
def to_subject_paths(paths):
    '''
    to_subject_paths(paths) accepts either a string that is a :-separated list of directories or a
      list of directories and yields a list of all the existing directories.
    '''
    if paths is None: return []
    if pimms.is_str(paths): paths = paths.split(':')
    paths = [os.path.expanduser(p) for p in paths]
    return [p for p in paths if os.path.isdir(p)]

config.declare('hcp_subject_paths', environ_name='HCP_SUBJECTS_DIR', filter=to_subject_paths)

def subject_paths():
    '''
    subject_paths() yields a list of paths to HCP subject directories in which subjects are
      automatically searched for when identified by subject-name only. These paths are searched in
      the order returned from this function.

    If you must edit these paths, it is recommended to use add_subject_path, and clear_subject_paths
    functions.
    '''
    return config['hcp_subject_paths']
def clear_subject_paths(subpaths):
    '''
    clear_subject_paths() resets the HCP subject paths to be empty and yields the previous
      list of subject paths.
    '''
    sd = config['hcp_subject_paths']
    config['hcp_subject_paths'] = []
    return sd
def add_subject_path(path, index=None):
    '''
    add_subject_path(path) will add the given path to the list of subject directories in which to
      search for HCP subjects. The optional argument index may be given to specify the precedence of
      this path when searching for a new subject; the default, 0, always inserts the path at the
      front of the list; a value of k indicates that the new list should have the new path at index
      k.
    The path may contain :'s, in which case the individual directories are separated and added.  If
    the given path is not a directory or the path could not be inserted, yields False; otherwise,
    yields True. If the string contains a : and multiple paths, then True is yielded only if all
    paths were successfully inserted.  See also subject_paths.
    '''
    paths = [p for p in path.split(':') if len(p) > 0]
    if len(paths) > 1:
        tests = [add_subject_path(p, index=index) for p in reversed(paths)]
        return all(t for t in tests)
    else:
        spaths = config['hcp_subject_paths']
        path = os.path.expanduser(path)
        if not os.path.isdir(path): return False
        if path in spaths: return True
        try:
            if index is None or index is Ellipsis:
                sd = spaths + [path]
            else:
                sd = spaths + []
                sd.insert(index, path)
            config['hcp_subject_paths'] = sd
            return True
        except Exception:
            return False

def find_subject_path(sid):
    '''
    find_subject_path(sub) yields the full path of a HCP subject with the name given by the string
      sub, if such a subject can be found in the HCP search paths. See also add_subject_path.

    If no subject is found, then None is returned.
    '''
    # if it's a full/relative path already, use it:
    sub = str(sid)
    sdirs = config['hcp_subject_paths']
    if _auto_download_options is not None and 'subjects_path' in _auto_download_options:
        sdirs = list(sdirs) + [_auto_download_options['subjects_path']]
    if os.path.isdir(sub): return os.path.abspath(sub)
    pth = next((os.path.abspath(p) for sd in sdirs
                for p in [os.path.join(sd, sub)]
                if os.path.isdir(p)),
               None)
    if pth is not None: return pth
    # see if we can create them
    if _auto_download_options is None or not _auto_download_options['structure'] or \
       not _auto_downloadable(sid):
        return None
    # first see if the subject existst there
    pth = os.path.join(_auto_download_options['subjects_path'], sub)
    if os.path.isdir(pth): return pth
    try: os.makedirs(os.path.abspath(pth), 0o755)
    except Exception: return None
    return pth

if config['hcp_subject_paths'] is None:
    # if a path wasn't found, there are a couple environment variables we want to look at...
    if 'HCPSUBJS_DIR' in os.environ: add_subject_path(os.environ['HCPSUBJS_DIR'])
    for varname in ['HCP_ROOT', 'HCP_DIR']:
        if varname in os.environ:
            dirname = os.path.join(os.environ[varname], 'subjects')
            if os.path.isdir(dirname):
                add_subject_path(dirname)


####################################################################################################
# Utilities

def to_subject_id(s):
    '''
    to_subject_id(s) coerces the given string or number into an integer subject id. If s is not a
      valid subejct id, raises an exception.
    '''
    if not pimms.is_number(s) and not pimms.is_str(s):
        raise ValueError('invalid type for subject id: %s' % str(type(s)))
    if pimms.is_str(s):
        try: s = os.path.expanduser(s)
        except Exception: pass
        if os.path.isdir(s): s = s.split(os.sep)[-1]
    s = int(s)
    if s > 999999 or s < 100000:
        raise ValueError('subject ids must be 6-digit integers whose first digit is > 0')
    return s

config.declare_credentials('hcp_credentials',
                           environ_name='HCP_CREDENTIALS',
                           extra_environ=[('HCP_KEY', 'HCP_SECRET'),
                                          'S3FS_CREDENTIALS',
                                          ('S3FS_KEY', 'S3FS_SECRET')],
                           filenames=['~/.hcp-passwd', '~/.passwd-hcp',
                                      '~/.s3fs-passwd', '~/.passwd-s3fs'],
                           aws_profile_name=['HCP', 'hcp', 'S3FS', 's3fs'])


####################################################################################################
# Subject Data Structure
# This structure details how neuropythy understands an HCP subject to be structured.

def gifti_to_array(gii):
    '''
    gifti_to_array(gii) yields the squeezed array of data contained in the given gifti object, gii,
      Note that if gii does not contain simple data in its darray object, then this will produce
      undefined results. This operation is effectively equivalent to:
      np.squeeze([x.data for x in gii.darrays]).
    gifti_to_array(gii_filename) is equivalent to gifti_to_array(neyropythy.load(gii_filename)).
    '''
    if pimms.is_str(gii): return gifti_to_array(ny.load(gii, 'gifti'))
    elif pimms.is_nparray(gii): return gii #already done
    elif isinstance(gii, nib.gifti.gifti.GiftiImage):
        return np.squeeze(np.asarray([x.data for x in gii.darrays]))
    else: raise ValueError('Could not understand argument to gifti_to_array')

# A few loading functions used by the description below
# Used to auto-download a single file
def _auto_download_file(filename, data):
    sid = data['id']
    if os.path.isfile(filename): return filename
    if _auto_download_options is None or not _auto_download_options['structure'] \
       or not _auto_downloadable(sid):
        return None
    fs = _auto_download_options['s3fs']
    db = _auto_download_options['database']
    rl = _auto_download_options['release']
    # parse the path apart by subject directory
    splt = str(sid) + os.sep
    relpath = splt.join(filename.split(splt)[1:])
    hcp_sdir = '/'.join([db, rl, str(sid)])
    if not fs.exists(hcp_sdir):
        raise ValueError('Subject %d not found in release' % sid)
    hcp_flnm = '/'.join([hcp_sdir, relpath])
    # make sure it exists
    if not fs.exists(hcp_flnm): return None
    # download it...
    basedir = os.path.split(filename)[0]
    if not os.path.isdir(basedir): os.makedirs(os.path.abspath(basedir), 0o755)
    logging.info('neuropythy: Fetching HCP file "%s"', filename)
    fs.get(hcp_flnm, filename)
    return filename
# Used to load immutable-like mgh objects
def _data_load(filename, data):
    sid = data['id']
    # First, see if the file exists
    if not os.path.isfile(filename): raise ValueError('File %s not found' % filename)
    # If the data says it's a cifti...
    if 'cifti' in data and data['cifti']:
        res = nib.load(filename).get_data()
        res = np.squeeze(res)
    elif data['type'] in ('surface', 'registration'):
        res = nyio.load(filename)
    elif data['type'] == 'property':
        if filename.endswith('.gii') or filename.endswith('.gii.gz'):
            res = gifti_to_array(nyio.load(filename))
        else:
            res = nyio.load(filename)
        res = np.squeeze(res)
    elif data['type'] == 'image':
        res = nyio.load(filename)
    else:
        raise ValueError('unrecognized data type: %s' % data['type'])
    return res
def _load(filename, data):
    # firs check about auto-downloading
    if not os.path.isfile(filename): _auto_download_file(filename, data)
    # file may not exist...
    if not os.path.isfile(filename): return None
    if 'load' in data and data['load'] is not None:
        res = data['load'](filename, data)
    else:
        res = _data_load(filename, data)
    # do the filter if there is one
    if 'filt' in data and data['filt'] is not None:
        res = data['filt'](res)
    # persist and return
    if is_image(res):
        res.get_data().setflags(write=False)
    elif pimms.is_imm(res):
        res.persist()
    elif pimms.is_nparray(res):
        res.setflags(write=False)
    return res
def _load_atlas_sphere(filename, data):
    atlases = _load_atlas_sphere.atlases
    (fdir, fnm) = os.path.split(filename)
    (sid, h, x1, atlas, x2, x3) = fnm.split('.')
    sid = int(sid)
    h = h.lower() + 'h'
    if x2 != 'surf':
        raise ValueError('bad filename for atlas sphere: %s' % filename)
    cache = atlases[h]
    addr = atlas + '.' + x1
    if addr not in cache:
        cache[addr] = _load(filename, pimms.merge(data, {'load':None}))
    return cache[addr]
_load_atlas_sphere.atlases = {'lh':{}, 'rh':{}}
def _load_fsLR_atlasroi(filename, data):
    '''
    Loads the appropriate atlas for the given data; data may point to a cifti file whose atlas is
    needed or to an atlas file.
    '''
    (fdir, fnm) = os.path.split(filename)
    fparts = fnm.split('.')
    atl = fparts[-3]
    if atl in _load_fsLR_atlasroi.atlases: return _load_fsLR_atlasroi.atlases[atl]
    sid = data['id']
    fnm = [os.path.join(fdir, '%d.%s.atlasroi.%s.shape.gii' % (sid, h, atl))  for h in ('L', 'R')]
    if data['cifti']:
        dat = [{'id':data['id'], 'type':'property', 'name':'atlas', 'hemi':h} for h in data['hemi']]
    else:
        dat = [{'id':data['id'], 'type':'property', 'name':'atlas', 'hemi':(h + data['hemi'][2:])}
               for h in ('lh','rh')]
    # loading an atlas file; this is easier
    rois = tuple([_load(f, d).astype('bool') for (f,d) in zip(fnm, dat)])
    # add these to the cache
    if atl != 'native': _load_fsLR_atlasroi.atlases[atl] = rois
    return rois
_load_fsLR_atlasroi.atlases = {}
def _load_fsLR_atlasroi_for_size(size, sid=100610):
    '''
    Loads the appropriate atlas for the given size of data; size should be the number of stored
    vertices and sub-corticel voxels stored in the cifti file.
    '''
    from .core import subject
    # it doesn't matter what subject we request, so just use any one
    fls = _load_fsLR_atlasroi_for_size.sizes
    if size not in fls: raise ValueError('unknown fs_LR atlas size: %s' % size)
    (n,fls) = _load_fsLR_atlasroi_for_size.sizes[size]
    fl = os.path.join(subject(sid).path, 'MNINonLinear', *fls)
    dat = {'id':sid, 'cifti':True, 'hemi':('lh_LR%dk_MSMAll' % n ,'rh_LR%dk_MSMAll' % n)}
    return _load_fsLR_atlasroi(fl, dat)
_load_fsLR_atlasroi_for_size.sizes = {
    # two sizes for each atlas: one for when the cifti file includes subcortical voxels and one for
    # when it includes only the surface vertices
    91282:  (32,  ['fsaverage_LR32k', '{0[id]}.curvature_MSMAll.32k_fs_LR.dscalar.nii']),
    59412:  (32,  ['fsaverage_LR32k', '{0[id]}.curvature_MSMAll.32k_fs_LR.dscalar.nii']),
    
    170494: (59,  ['fsaverage_LR59k', '{0[id]}.MyelinMap_BC_1.6mm_MSMAll.59k_fs_LR.dscalar.nii']),
    108441: (59,  ['fsaverage_LR59k', '{0[id]}.MyelinMap_BC_1.6mm_MSMAll.59k_fs_LR.dscalar.nii']),

    # not sure what the bigger size is for this...
    #???: (164, ['{0[id]}.curvature_MSMAll.164k_fs_LR.dscalar.nii'],
    298261: (164, ['{0[id]}.curvature_MSMAll.164k_fs_LR.dscalar.nii'])}
def _load_fsmorph(filename, data):
    return nyio.load(filename, 'freesurfer_morph')
        

# The description of the entire subject directory that we care about:
subject_directory_structure = {
    'T1w': {'type':'dir',
            'contents': {
                'BiasField_acpc_dc.nii.gz':         {'type':'image', 'name':'bias'},
                'T1wDividedByT2w.nii.gz':           {'type':'image', 'name':'T1_to_T2_ratio_all'},
                'T1wDividedByT2w_ribbon.nii.gz':    {'type':'image', 'name':'T1_to_T2_ratio'},
                'T1w_acpc_dc_restore.nii.gz':       {'type':'image', 'name':'T1'},
                'T1w_acpc_dc.nii.gz':               {'type':'image', 'name':'T1_unrestored'},
                'T1w_acpc_dc_restore_brain.nii.gz': {'type':'image', 'name':'brain'},
                'T2w_acpc_dc_restore.nii.gz':       {'type':'image', 'name':'T2'},
                'T2w_acpc_dc.nii.gz':               {'type':'image', 'name':'T2_unrestored'},
                'T2w_acpc_dc_restore_brain.nii.gz': {'type':'image', 'name':'T2_brain'},
                'aparc+aseg.nii.gz':                {'type':'image', 'name':'parcellation2005'},
                'aparc.a2009s+aseg.nii.gz':         {'type':'image', 'name':'parcellation'},
                'brainmask_fs.nii.gz':              {'type':'image', 'name':'brainmask'},
                'ribbon.nii.gz':                    {'type':'image', 'name':'ribbon'},
                'wmparc.nii.gz':                    {'type':'image', 'name':'wm_parcellation'},
                '{0[id]}': {
                    'type': 'dir',
                    'contents': {
                        'surf': {
                            'type': 'dir',
                            'contents': {
                                'lh.area': (
                                    {'type':'property',          'name':'white_surface_area',
                                     'hemi':'lh_native_MSMSulc', 'load':_load_fsmorph},
                                    {'type':'property',          'name':'white_surface_area',
                                     'hemi':'lh_native_MSMAll',  'load':_load_fsmorph}),
                                'lh.area.mid': (
                                    {'type':'property',          'name':'midgray_surface_area',
                                     'hemi':'lh_native_MSMSulc', 'load':_load_fsmorph},
                                    {'type':'property',          'name':'midgray_surface_area',
                                     'hemi':'lh_native_MSMAll',  'load':_load_fsmorph}),
                                'lh.area.pial': (
                                    {'type':'property',          'name':'pial_surface_area',
                                     'hemi':'lh_native_MSMSulc', 'load':_load_fsmorph},
                                    {'type':'property',          'name':'pial_surface_area',
                                     'hemi':'lh_native_MSMAll',  'load':_load_fsmorph}),
                                'rh.area': (
                                    {'type':'property',          'name':'white_surface_area',
                                     'hemi':'rh_native_MSMSulc', 'load':_load_fsmorph},
                                    {'type':'property',          'name':'white_surface_area',
                                     'hemi':'rh_native_MSMAll',  'load':_load_fsmorph}),
                                'rh.area.mid': (
                                    {'type':'property',          'name':'midgray_surface_area',
                                     'hemi':'rh_native_MSMSulc', 'load':_load_fsmorph},
                                    {'type':'property',          'name':'midgray_surface_area',
                                     'hemi':'rh_native_MSMAll',  'load':_load_fsmorph}),
                                'rh.area.pial': (
                                    {'type':'property',          'name':'pial_surface_area',
                                     'hemi':'rh_native_MSMSulc', 'load':_load_fsmorph},
                                    {'type':'property',          'name':'pial_surface_area',
                                     'hemi':'rh_native_MSMAll',  'load':_load_fsmorph})}}}},
                'Native': {
                    'type':'dir',
                    'contents': {
                        '{0[id]}.L.white.native.surf.gii': (
                            {'type':'surface',
                             'name':'white',
                             'hemi':'lh_native_MSMSulc'},
                            {'type':'surface',
                             'name':'white',
                             'hemi':'lh_native_MSMAll'}),
                        '{0[id]}.L.midthickness.native.surf.gii': (
                            {'type':'surface',
                             'name':'midgray',
                             'hemi':'lh_native_MSMSulc'},
                            {'type':'surface',
                             'name':'midgray',
                             'hemi':'lh_native_MSMAll'}),
                        '{0[id]}.L.pial.native.surf.gii':(
                            {'type':'surface',
                             'name':'pial',
                             'hemi':'lh_native_MSMSulc'},
                            {'type':'surface',
                             'name':'pial',
                             'hemi':'lh_native_MSMAll'}),
                        '{0[id]}.L.inflated.native.surf.gii': (
                            {'type':'surface',
                             'name':'inflated',
                             'hemi':'lh_native_MSMSulc'},
                            {'type':'surface',
                             'name':'inflated',
                             'hemi':'lh_native_MSMAll'}),
                        '{0[id]}.L.very_inflated.native.surf.gii': (
                            {'type':'surface',
                             'name':'very_inflated',
                             'hemi':'lh_native_MSMSulc'},
                            {'type':'surface',
                             'name':'very_inflated',
                             'hemi':'lh_native_MSMAll'}),
                        '{0[id]}.R.white.native.surf.gii': (
                            {'type':'surface',
                             'name':'white',
                             'hemi':'rh_native_MSMSulc'},
                            {'type':'surface',
                             'name':'white',
                             'hemi':'rh_native_MSMAll'}),
                        '{0[id]}.R.midthickness.native.surf.gii': (
                            {'type':'surface',
                             'name':'midgray',
                             'hemi':'rh_native_MSMSulc'},
                            {'type':'surface',
                             'name':'midgray',
                             'hemi':'rh_native_MSMAll'}),
                        '{0[id]}.R.pial.native.surf.gii':(
                            {'type':'surface',
                             'name':'pial',
                             'hemi':'rh_native_MSMSulc'},
                            {'type':'surface',
                             'name':'pial',
                             'hemi':'rh_native_MSMAll'}),
                        '{0[id]}.R.inflated.native.surf.gii': (
                            {'type':'surface',
                             'name':'inflated',
                             'hemi':'rh_native_MSMSulc'},
                            {'type':'surface',
                             'name':'inflated',
                             'hemi':'rh_native_MSMAll'}),
                        '{0[id]}.R.very_inflated.native.surf.gii': (
                            {'type':'surface',
                             'name':'very_inflated',
                             'hemi':'rh_native_MSMSulc'},
                            {'type':'surface',
                             'name':'very_inflated',
                             'hemi':'rh_native_MSMAll'})}},
                'fsaverage_LR32k': {
                    'type':'dir',
                    'contents': {
                        '{0[id]}.L.inflated.32k_fs_LR.surf.gii':      {
                            'type':'surface',
                            'name':'inflated',
                            'hemi':'lh_lowres_MSMSulc'},
                        '{0[id]}.L.midthickness.32k_fs_LR.surf.gii':  {
                            'type':'surface',
                            'name':'midgray',
                            'hemi':'lh_lowres_MSMSulc'},
                        '{0[id]}.L.pial.32k_fs_LR.surf.gii':          {
                            'type':'surface',
                            'name':'pial',
                            'hemi':'lh_lowres_MSMSulc'},
                        '{0[id]}.L.very_inflated.32k_fs_LR.surf.gii': {
                            'type':'surface',
                            'name':'very_inflated',
                            'hemi':'lh_lowres_MSMSulc'},
                        '{0[id]}.L.white.32k_fs_LR.surf.gii':         {
                            'type':'surface',
                            'name':'white',
                            'hemi':'lh_lowres_MSMSulc'},
                        '{0[id]}.R.inflated.32k_fs_LR.surf.gii':      {
                            'type':'surface',
                            'name':'inflated',
                            'hemi':'rh_lowres_MSMSulc'},
                        '{0[id]}.R.midthickness.32k_fs_LR.surf.gii':  {
                            'type':'surface',
                            'name':'midgray',
                            'hemi':'rh_lowres_MSMSulc'},
                        '{0[id]}.R.pial.32k_fs_LR.surf.gii':          {
                            'type':'surface',
                            'name':'pial',
                            'hemi':'rh_lowres_MSMSulc'},
                        '{0[id]}.R.very_inflated.32k_fs_LR.surf.gii': {
                            'type':'surface',
                            'name':'very_inflated',
                            'hemi':'rh_lowres_MSMSulc'},
                        '{0[id]}.R.white.32k_fs_LR.surf.gii':         {
                            'type':'surface',
                            'name':'white',
                            'hemi':'rh_lowres_MSMSulc'},
                        '{0[id]}.L.inflated_MSMAll.32k_fs_LR.surf.gii':      {
                            'type':'surface',
                            'name':'inflated',
                            'hemi':'lh_lowres_MSMAll'},
                        '{0[id]}.L.midthickness_MSMAll.32k_fs_LR.surf.gii':  {
                            'type':'surface',
                            'name':'midgray',
                            'hemi':'lh_lowres_MSMAll'},
                        '{0[id]}.L.pial_MSMAll.32k_fs_LR.surf.gii':          {
                            'type':'surface',
                            'name':'pial',
                            'hemi':'lh_lowres_MSMAll'},
                        '{0[id]}.L.very_inflated_MSMAll.32k_fs_LR.surf.gii': {
                            'type':'surface',
                            'name':'very_inflated',
                            'hemi':'lh_lowres_MSMAll'},
                        '{0[id]}.L.white_MSMAll.32k_fs_LR.surf.gii':         {
                            'type':'surface',
                            'name':'white',
                            'hemi':'lh_lowres_MSMAll'},
                        '{0[id]}.R.inflated_MSMAll.32k_fs_LR.surf.gii':      {
                            'type':'surface',
                            'name':'inflated',
                            'hemi':'rh_lowres_MSMAll'},
                        '{0[id]}.R.midthickness_MSMAll.32k_fs_LR.surf.gii':  {
                            'type':'surface',
                            'name':'midgray',
                            'hemi':'rh_lowres_MSMAll'},
                        '{0[id]}.R.pial_MSMAll.32k_fs_LR.surf.gii':          {
                            'type':'surface',
                            'name':'pial',
                            'hemi':'rh_lowres_MSMAll'},
                        '{0[id]}.R.very_inflated_MSMAll.32k_fs_LR.surf.gii': {
                            'type':'surface',
                            'name':'very_inflated',
                            'hemi':'rh_lowres_MSMAll'},
                        '{0[id]}.R.white_MSMAll.32k_fs_LR.surf.gii':         {
                            'type':'surface',
                            'name':'white',
                            'hemi':'rh_lowres_MSMAll'}}}}},
    'MNINonLinear': {
        'type': 'dir',
        'contents': {
            'BiasField.nii.gz':         {'type':'image', 'name':'bias_warped'},
            'T1w_restore.nii.gz':       {'type':'image', 'name':'T1_warped'},
            'T1w.nii.gz':               {'type':'image', 'name':'T1_warped_unrestored'},
            'T1w_restore_brain.nii.gz': {'type':'image', 'name':'brain_warped'},
            'T2w_restore.nii.gz':       {'type':'image', 'name':'T2_warped'},
            'T2w.nii.gz':               {'type':'image', 'name':'T2_warped_unrestored'},
            'T2w_restore_brain.nii.gz': {'type':'image', 'name':'T2_brain_warped'},
            'aparc+aseg.nii.gz':        {'type':'image', 'name':'parcellation2005_warped'},
            'aparc.a2009s+aseg.nii.gz': {'type':'image', 'name':'parcellation_warped'},
            'brainmask_fs.nii.gz':      {'type':'image', 'name':'brainmask_warped'},
            'ribbon.nii.gz':            {'type':'image', 'name':'ribbon_warped'},
            'wmparc.nii.gz':            {'type':'image', 'name':'wm_parcellation_warped'},
            '{0[id]}.L.ArealDistortion_FS.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'areal_distortion_FS',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.ArealDistortion_MSMSulc.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'areal_distortion',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.MyelinMap.164k_fs_LR.func.gii': {
                'type':'property',
                'name':'myelin',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.MyelinMap_BC.164k_fs_LR.func.gii': {
                'type':'property',
                'name':'myelin_bc',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.SmoothedMyelinMap.164k_fs_LR.func.gii': {
                'type':'property',
                'name':'myelin_smooth',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.SmoothedMyelinMap_BC.164k_fs_LR.func.gii': {
                'type':'property',
                'name':'myelin_smooth_bc',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.RefMyelinMap.164k_fs_LR.func.gii': {
                'type':'property',
                'name':'myelin_ref',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.BA.164k_fs_LR.label.gii': (
                {'type':'property',
                 'name':'brodmann_area',
                 'hemi':'lh_LR164k_MSMSulc'},
                {'type':'property',
                 'name':'brodmann_area',
                 'hemi':'lh_LR164k_MSMAll'}),
            '{0[id]}.L.aparc.164k_fs_LR.label.gii': (
                {'type':'property',
                 'name':'parcellation_2005',
                 'hemi':'lh_LR164k_MSMSulc'},
                {'type':'property',
                 'name':'parcellation_2005',
                 'hemi':'lh_LR164k_MSMAll'}),
            '{0[id]}.L.aparc.a2009s.164k_fs_LR.label.gii': (
                {'type':'property',
                 'name':'parcellation',
                 'hemi':'lh_LR164k_MSMSulc'},
                {'type':'property',
                 'name':'parcellation',
                 'hemi':'lh_LR164k_MSMAll'}),
            '{0[id]}.L.atlasroi.164k_fs_LR.shape.gii': (
                {'type':'property',
                 'name':'atlas',
                 'hemi':'lh_LR164k_MSMSulc',
                 'load':_load_fsLR_atlasroi,
                 'filt':lambda x:x[0].astype(np.bool)},
                {'type':'property',
                 'name':'atlas',
                 'hemi':'lh_LR164k_MSMAll',
                 'load':_load_fsLR_atlasroi,
                 'filt':lambda x:x[0].astype(np.bool)}),
            '{0[id]}.L.curvature.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'curvature',
                'hemi':'lh_LR164k_MSMSulc',
                'filt':lambda c: -c},
            '{0[id]}.L.sulc.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'convexity',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.corrThickness.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'thickness',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.thickness.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'thickness_uncorrected',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.white.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'white',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.midthickness.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'midgray',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.pial.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'pial',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.inflated.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'inflated',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.very_inflated.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'very_inflated',
                'hemi':'lh_LR164k_MSMSulc'},
            '{0[id]}.L.white_MSMAll.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'white',
                'hemi':'lh_LR164k_MSMAll'},
            '{0[id]}.L.midthickness_MSMAll.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'midgray',
                'hemi':'lh_LR164k_MSMAll'},
            '{0[id]}.L.pial_MSMAll.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'pial',
                'hemi':'lh_LR164k_MSMAll'},
            '{0[id]}.L.inflated_MSMAll.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'inflated',
                'hemi':'lh_LR164k_MSMAll'},
            '{0[id]}.L.very_inflated_MSMAll.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'very_inflated',
                'hemi':'lh_LR164k_MSMAll'},
            '{0[id]}.L.sphere.164k_fs_LR.surf.gii': (
                {'type':'registration',
                 'name':'fs_LR',
                 'hemi':'lh_LR164k_MSMSulc',
                 'load':_load_atlas_sphere},
                {'type':'registration',
                 'name':'fs_LR',
                 'hemi':'lh_LR164k_MSMAll',
                 'load':_load_atlas_sphere}),
            # disabled until I decide how they should be handled (their tesselations not like the
            # spheres, so they don't really belong in the same hemisphere)
            #'{0[id]}.L.flat.164k_fs_LR.surf.gii': (
            #    {'type':'surface',
            #     'name':'flat',
            #     'hemi':'lh_LR164k_MSMSulc',
            #     'load':_load_atlas_sphere},
            #    {'type':'surface',
            #     'name':'flat',
            #     'hemi':'lh_LR164k_MSMAll',
            #     'load':_load_atlas_sphere}),
            '{0[id]}.R.ArealDistortion_FS.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'areal_distortion_FS',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.ArealDistortion_MSMSulc.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'areal_distortion',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.MyelinMap.164k_fs_LR.func.gii': {
                'type':'property',
                'name':'myelin',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.MyelinMap_BC.164k_fs_LR.func.gii': {
                'type':'property',
                'name':'myelin_bc',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.SmoothedMyelinMap.164k_fs_LR.func.gii': {
                'type':'property',
                'name':'myelin_smooth',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.SmoothedMyelinMap_BC.164k_fs_LR.func.gii': {
                'type':'property',
                'name':'myelin_smooth_bc',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.RefMyelinMap.164k_fs_LR.func.gii': {
                'type':'property',
                'name':'myelin_ref',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.BA.164k_fs_LR.label.gii': (
                {'type':'property',
                 'name':'brodmann_area',
                 'hemi':'rh_LR164k_MSMSulc'},
                {'type':'property',
                 'name':'brodmann_area',
                 'hemi':'rh_LR164k_MSMAll'}),
            '{0[id]}.R.aparc.164k_fs_LR.label.gii': (
                {'type':'property',
                 'name':'parcellation_2005',
                 'hemi':'rh_LR164k_MSMSulc'},
                {'type':'property',
                 'name':'parcellation_2005',
                 'hemi':'rh_LR164k_MSMAll'}),
            '{0[id]}.R.aparc.a2009s.164k_fs_LR.label.gii': (
                {'type':'property',
                 'name':'parcellation',
                 'hemi':'rh_LR164k_MSMSulc'},
                {'type':'property',
                 'name':'parcellation',
                 'hemi':'rh_LR164k_MSMAll'}),
            '{0[id]}.R.atlasroi.164k_fs_LR.shape.gii': (
                {'type':'property',
                 'name':'atlas',
                 'hemi':'rh_LR164k_MSMSulc',
                 'load':_load_fsLR_atlasroi,
                 'filt':lambda x:x[1].astype(np.bool)},
                {'type':'property',
                 'name':'atlas',
                 'hemi':'rh_LR164k_MSMAll',
                 'load':_load_fsLR_atlasroi,
                 'filt':lambda x:x[1].astype(np.bool)}),
            '{0[id]}.R.curvature.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'curvature',
                'hemi':'rh_LR164k_MSMSulc',
                'filt':lambda c: -c},
            '{0[id]}.R.sulc.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'convexity',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.corrThickness.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'thickness',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.thickness.164k_fs_LR.shape.gii': {
                'type':'property',
                'name':'thickness_uncorrected',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.sulc.164k_fs_LR.shape.gii': {
                'type':'surface',
                'name':'convexity',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.white.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'white',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.midthickness.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'midgray',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.pial.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'pial',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.inflated.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'inflated',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.very_inflated.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'very_inflated',
                'hemi':'rh_LR164k_MSMSulc'},
            '{0[id]}.R.white_MSMAll.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'white',
                'hemi':'rh_LR164k_MSMAll'},
            '{0[id]}.R.midthickness_MSMAll.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'midgray',
                'hemi':'rh_LR164k_MSMAll'},
            '{0[id]}.R.pial_MSMAll.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'pial',
                'hemi':'rh_LR164k_MSMAll'},
            '{0[id]}.R.inflated_MSMAll.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'inflated',
                'hemi':'rh_LR164k_MSMAll'},
            '{0[id]}.R.very_inflated_MSMAll.164k_fs_LR.surf.gii': {
                'type':'surface',
                'name':'very_inflated',
                'hemi':'rh_LR164k_MSMAll'},
            '{0[id]}.R.sphere.164k_fs_LR.surf.gii': (
                {'type':'registration',
                 'name':'fs_LR',
                 'hemi':'rh_LR164k_MSMSulc',
                 'load':_load_atlas_sphere},
                {'type':'registration',
                 'name':'fs_LR',
                 'hemi':'rh_LR164k_MSMAll',
                 'load':_load_atlas_sphere}),
            #'{0[id]}.R.flat.164k_fs_LR.surf.gii': (
            #    {'type':'surface',
            #     'name':'flat',
            #     'hemi':'rh_LR164k_MSMSulc',
            #     'load':_load_atlas_sphere},
            #    {'type':'surface',
            #     'name':'flat',
            #     'hemi':'rh_LR164k_MSMAll',
            #     'load':_load_atlas_sphere}),
            '{0[id]}.ArealDistortion_MSMAll.164k_fs_LR.dscalar.nii': {
                'type':'property',
                'name':'areal_distortion',
                'hemi':('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
            '{0[id]}.MyelinMap_BC_MSMAll.164k_fs_LR.dscalar.nii': {
                'type':'property',
                'name':'myelin_bc',
                'hemi':('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
            '{0[id]}.SmoothedMyelinMap_BC_MSMAll.164k_fs_LR.dscalar.nii': {
                'type':'property',
                'name':'myelin_smooth_bc',
                'hemi':('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
            '{0[id]}.curvature_MSMAll.164k_fs_LR.dscalar.nii': {
                'type':'property',
                'name':'curvature',
                'hemi':('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll'),
                'filt':lambda c: -c},
            '{0[id]}.sulc.164k_fs_LR.dscalar.nii': {
                'type':'property',
                'name':'convexity',
                'hemi':('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
            '{0[id]}.corrThickness.164k_fs_LR.dscalar.nii': {
                'type':'property',
                'name':'thickness',
                'hemi':('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
            '{0[id]}.thickness.164k_fs_LR.dscalar.nii': {
                'type':'property',
                'name':'thickness_uncorrected',
                'hemi':('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
            'Native': {
                'type':'dir',
                'contents': {
                    '{0[id]}.L.ArealDistortion_FS.native.shape.gii': (
                        {'type':'property',
                         'name':'areal_distortion_FS',
                         'hemi':'lh_native_MSMSulc'},
                        {'type':'property',
                         'name':'areal_distortion_FS',
                         'hemi':'lh_native_MSMAll'}),
                    '{0[id]}.L.ArealDistortion_MSMSulc.native.shape.gii': {
                        'type':'property',
                        'name':'areal_distortion',
                        'hemi':'lh_native_MSMSulc'},
                    '{0[id]}.L.ArealDistortion_MSMAll.native.shape.gii': {
                        'type':'property',
                        'name':'areal_distortion',
                        'hemi':'lh_native_MSMAll'},
                    '{0[id]}.L.MyelinMap.native.func.gii': {
                        'type':'property',
                        'name':'myelin',
                        'hemi':'lh_native_MSMSulc'},
                    '{0[id]}.L.MyelinMap_BC.native.func.gii': {
                        'type':'property',
                        'name':'myelin_bc',
                        'hemi':'lh_native_MSMSulc'},
                    '{0[id]}.L.SmoothedMyelinMap.native.func.gii': {
                        'type':'property',
                        'name':'myelin_smooth',
                        'hemi':'lh_native_MSMSulc'},
                    '{0[id]}.L.SmoothedMyelinMap_BC.native.func.gii': {
                        'type':'property',
                        'name':'myelin_smooth_bc',
                        'hemi':'lh_native_MSMSulc'},
                    '{0[id]}.L.RefMyelinMap.native.func.gii': {
                        'type':'property',
                        'name':'myelin_ref',
                        'hemi':'lh_native_MSMSulc'},
                    '{0[id]}.L.BA.native.label.gii': (
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'lh_native_MSMSulc'},
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'lh_native_MSMAll'}),
                    '{0[id]}.L.aparc.native.label.gii':  (
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'lh_native_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'lh_native_MSMAll'}),
                    '{0[id]}.L.aparc.a2009s.native.label.gii': (
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'lh_native_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'lh_native_MSMAll'}),
                    '{0[id]}.L.atlasroi.native.shape.gii': (
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'lh_native_MSMSulc',
                         'filt':lambda x:x.astype(np.bool)},
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'lh_native_MSMAll',
                         'filt':lambda x:x.astype(np.bool)}),
                    '{0[id]}.L.curvature.native.shape.gii': (
                        {'type':'property',
                         'name':'curvature',
                         'hemi':'lh_native_MSMSulc',
                         'filt':lambda c: -c},
                        {'type':'property',
                         'name':'curvature',
                         'hemi':'lh_native_MSMAll',
                         'filt':lambda c: -c}),
                    '{0[id]}.L.sulc.native.shape.gii': (
                        {'type':'property',
                         'name':'convexity',
                         'hemi':'lh_native_MSMSulc'},
                        {'type':'property',
                         'name':'convexity',
                         'hemi':'lh_native_MSMAll'}),
                    '{0[id]}.L.corrThickness.native.shape.gii': (
                        {'type':'property',
                         'name':'thickness',
                         'hemi':'lh_native_MSMSulc'},
                        {'type':'property',
                         'name':'thickness',
                         'hemi':'lh_native_MSMAll'}),
                    '{0[id]}.L.thickness.native.shape.gii': (
                        {'type':'property',
                         'name':'thickness_uncorrected',
                         'hemi':'lh_native_MSMSulc'},
                        {'type':'property',
                         'name':'thickness_uncorrected',
                         'hemi':'lh_native_MSMAll'}),
                    '{0[id]}.L.roi.native.shape.gii': (
                        {'type':'property',
                         'name':'roi',
                         'hemi':'lh_native_MSMSulc',
                         'filt':lambda r: r.astype(bool)},
                        {'type':'property',
                         'name':'roi',
                         'hemi':'lh_native_MSMAll',
                         'filt':lambda r: r.astype(bool)}),
                    '{0[id]}.L.sphere.native.surf.gii': (
                        {'type':'registration',
                         'name':'native',
                         'hemi':'lh_native_MSMSulc'},
                        {'type':'registration',
                         'name':'native',
                         'hemi':'lh_native_MSMAll'}),
                    '{0[id]}.L.sphere.reg.native.surf.gii': (
                        {'type':'registration',
                         'name':'fsaverage',
                         'hemi':'lh_native_MSMSulc'},
                        {'type':'registration',
                         'name':'fsaverage',
                         'hemi':'lh_native_MSMAll'}),
                    '{0[id]}.L.sphere.MSMAll.native.surf.gii': {
                        'type':'registration',
                        'name':'fs_LR',
                        'tool':'MSMAll',
                        'hemi':'lh_native_MSMAll'},
                    '{0[id]}.L.sphere.MSMSulc.native.surf.gii': {
                        'type':'registration',
                        'name':'fs_LR',
                        'tool':'MSMSulc',
                        'hemi':'lh_native_MSMSulc'},
                    '{0[id]}.R.ArealDistortion_FS.native.shape.gii': (
                        {'type':'property',
                         'name':'areal_distortion_FS',
                         'hemi':'rh_native_MSMSulc'},
                        {'type':'property',
                         'name':'areal_distortion_FS',
                         'hemi':'rh_native_MSMAll'}),
                    '{0[id]}.R.ArealDistortion_MSMSulc.native.shape.gii': {
                        'type':'property',
                        'name':'areal_distortion',
                        'hemi':'rh_native_MSMSulc'},
                    '{0[id]}.R.ArealDistortion_MSMAll.native.shape.gii': {
                        'type':'property',
                        'name':'areal_distortion',
                        'hemi':'rh_native_MSMAll'},
                    '{0[id]}.R.MyelinMap.native.func.gii': {
                        'type':'property',
                        'name':'myelin',
                        'hemi':'rh_native_MSMSulc'},
                    '{0[id]}.R.MyelinMap_BC.native.func.gii': {
                        'type':'property',
                        'name':'myelin_bc',
                        'hemi':'rh_native_MSMSulc'},
                    '{0[id]}.R.SmoothedMyelinMap.native.func.gii': {
                        'type':'property',
                        'name':'myelin_smooth',
                        'hemi':'rh_native_MSMSulc'},
                    '{0[id]}.R.SmoothedMyelinMap_BC.native.func.gii': {
                        'type':'property',
                        'name':'myelin_smooth_bc',
                        'hemi':'rh_native_MSMSulc'},
                    '{0[id]}.R.RefMyelinMap.native.func.gii': {
                        'type':'property',
                        'name':'myelin_ref',
                        'hemi':'rh_native_MSMSulc'},
                    '{0[id]}.R.BA.native.label.gii': (
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'rh_native_MSMSulc'},
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'rh_native_MSMAll'}),
                    '{0[id]}.R.aparc.native.label.gii':  (
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'rh_native_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'rh_native_MSMAll'}),
                    '{0[id]}.R.aparc.a2009s.native.label.gii': (
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'rh_native_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'rh_native_MSMAll'}),
                    '{0[id]}.R.atlasroi.native.shape.gii': (
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'rh_native_MSMSulc',
                         'filt':lambda x:x.astype(np.bool)},
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'rh_native_MSMAll',
                         'filt':lambda x:x.astype(np.bool)}),
                    '{0[id]}.R.curvature.native.shape.gii': (
                        {'type':'property',
                         'name':'curvature',
                         'hemi':'rh_native_MSMSulc',
                        'filt':lambda c: -c},
                        {'type':'property',
                         'name':'curvature',
                         'hemi':'rh_native_MSMAll',
                         'filt':lambda c: -c}),
                    '{0[id]}.R.sulc.native.shape.gii': (
                        {'type':'property',
                         'name':'convexity',
                         'hemi':'rh_native_MSMSulc'},
                        {'type':'property',
                         'name':'convexity',
                         'hemi':'rh_native_MSMAll'}),
                    '{0[id]}.R.corrThickness.native.shape.gii': (
                        {'type':'property',
                         'name':'thickness',
                         'hemi':'rh_native_MSMSulc'},
                        {'type':'property',
                         'name':'thickness',
                         'hemi':'rh_native_MSMAll'}),
                    '{0[id]}.R.thickness.native.shape.gii': (
                        {'type':'property',
                         'name':'thickness_uncorrected',
                         'hemi':'rh_native_MSMSulc'},
                        {'type':'property',
                         'name':'thickness_uncorrected',
                         'hemi':'rh_native_MSMAll'}),
                    '{0[id]}.R.roi.native.shape.gii': (
                        {'type':'property',
                         'name':'roi',
                         'hemi':'rh_native_MSMSulc',
                         'filt':lambda r: r.astype(bool)},
                        {'type':'property',
                         'name':'roi',
                         'hemi':'rh_native_MSMAll',
                         'filt':lambda r: r.astype(bool)}),
                    '{0[id]}.R.sphere.native.surf.gii': (
                        {'type':'registration',
                         'name':'native',
                         'hemi':'rh_native_MSMSulc'},
                        {'type':'registration',
                         'name':'native',
                         'hemi':'rh_native_MSMAll'}),
                    '{0[id]}.R.sphere.reg.native.surf.gii': (
                        {'type':'registration',
                         'name':'fsaverage',
                         'hemi':'rh_native_MSMSulc'},
                        {'type':'registration',
                         'name':'fsaverage',
                         'hemi':'rh_native_MSMAll'}),
                    '{0[id]}.R.sphere.MSMAll.native.surf.gii': {
                        'type':'registration',
                        'name':'fs_LR',
                        'tool':'MSMAll',
                        'hemi':'rh_native_MSMAll'},
                    '{0[id]}.R.sphere.MSMSulc.native.surf.gii': {
                        'type':'registration',
                        'name':'fs_LR',
                        'tool':'MSMSulc',
                        'hemi':'rh_native_MSMSulc'}}},
            'fsaverage_LR59k': {
                'type':'dir',
                'contents': {
                    '{0[id]}.L.BA.59k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'lh_LR59k_MSMSulc'},
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'lh_LR59k_MSMAll'}),
                    '{0[id]}.L.aparc.59k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'lh_LR59k_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'lh_LR59k_MSMAll'}),
                    '{0[id]}.L.aparc.a2009s.59k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'lh_LR59k_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'lh_LR59k_MSMAll'}),
                    '{0[id]}.L.atlasroi.59k_fs_LR.shape.gii': (
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'lh_LR59k_MSMSulc',
                         'load':_load_fsLR_atlasroi,
                         'filt':lambda x:x[0].astype(np.bool)},
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'lh_LR59k_MSMAll',
                         'load':_load_fsLR_atlasroi,
                         'filt':lambda x:x[0].astype(np.bool)}),
                    '{0[id]}.L.ArealDistortion_FS.59k_fs_LR.shape.gii': (
                        {'type':'property',
                         'name':'areal_distortion_fs',
                         'hemi':'lh_LR59k_MSMSulc'},
                        {'type':'property',
                         'name':'areal_distortion_fs',
                         'hemi':'lh_LR59k_MSMAll'}),
                    '{0[id]}.L.ArealDistortion_MSMSulc.59k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'areal_distortion',
                        'hemi':'lh_LR59k_MSMSulc'},
                    '{0[id]}.L.MyelinMap.59k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin',
                        'hemi':'lh_LR59k_MSMSulc'},
                    #'{0[id]}.L.MyelinMap_BC.59k_fs_LR.func.gii': {
                    #    'type':'property',
                    #    'name':'myelin_bc',
                    #    'hemi':'lh_LR59k_MSMSulc'},
                    '{0[id]}.L.SmoothedMyelinMap.59k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin_smooth',
                        'hemi':'lh_LR59k_MSMSulc'},
                    #'{0[id]}.L.SmoothedMyelinMap_BC.59k_fs_LR.func.gii': {
                    #    'type':'property',
                    #    'name':'myelin_smooth_bc',
                    #    'hemi':'lh_LR59k_MSMSulc'},
                    '{0[id]}.L.curvature.59k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'curvature',
                        'hemi':'lh_LR59k_MSMSulc',
                        'filt':lambda c: -c},
                    '{0[id]}.L.sulc.59k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'convexity',
                        'hemi':'lh_LR59k_MSMSulc'},
                    #'{0[id]}.L.corrThickness.59k_fs_LR.shape.gii': {
                    #    'type':'property',
                    #    'name':'thickness',
                    #    'hemi':'lh_LR59k_MSMSulc'},
                    '{0[id]}.L.thickness.59k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'thickness_uncorrected',
                        'hemi':'lh_LR59k_MSMSulc'},
                    '{0[id]}.L.white.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'white',
                        'hemi':'lh_LR59k_MSMSulc'},
                    '{0[id]}.L.midthickness.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'midgray',
                        'hemi':'lh_LR59k_MSMSulc'},
                    '{0[id]}.L.pial.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'pial',
                        'hemi':'lh_LR59k_MSMSulc'},
                    '{0[id]}.L.inflated.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'inflated',
                        'hemi':'lh_LR59k_MSMSulc'},
                    '{0[id]}.L.very_inflated.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'very_inflated',
                        'hemi':'lh_LR59k_MSMSulc'},
                    '{0[id]}.L.white_1.6mm_MSMAll.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'white',
                        'hemi':'lh_LR59k_MSMAll'},
                    '{0[id]}.L.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'midgray',
                        'hemi':'lh_LR59k_MSMAll'},
                    '{0[id]}.L.pial_1.6mm_MSMAll.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'pial',
                        'hemi':'lh_LR59k_MSMAll'},
                    '{0[id]}.L.inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'inflated',
                        'hemi':'lh_LR59k_MSMAll'},
                    '{0[id]}.L.very_inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'very_inflated',
                        'hemi':'lh_LR59k_MSMAll'},
                    #'{0[id]}.L.flat.59k_fs_LR.surf.gii': (
                    #    {'type':'surface',
                    #     'name':'flat',
                    #     'hemi':'lh_LR59k_MSMSulc'},
                    #    {'type':'surface',
                    #     'name':'flat',
                    #     'hemi':'lh_LR59k_MSMAll'}),
                    '{0[id]}.L.sphere.59k_fs_LR.surf.gii': (
                        {'type':'registration',
                         'name':'fs_LR',
                         'hemi':'lh_LR59k_MSMSulc'},
                        {'type':'registration',
                         'name':'fs_LR',
                         'hemi':'lh_LR59k_MSMAll'}),
                    '{0[id]}.R.BA.59k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'rh_LR59k_MSMSulc'},
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'rh_LR59k_MSMAll'}),
                    '{0[id]}.R.aparc.59k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'rh_LR59k_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'rh_LR59k_MSMAll'}),
                    '{0[id]}.R.aparc.a2009s.59k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'rh_LR59k_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'rh_LR59k_MSMAll'}),
                    '{0[id]}.R.atlasroi.59k_fs_LR.shape.gii': (
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'rh_LR59k_MSMSulc',
                         'load':_load_fsLR_atlasroi,
                         'filt':lambda x:x[1].astype(np.bool)},
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'rh_LR59k_MSMAll',
                         'load':_load_fsLR_atlasroi,
                         'filt':lambda x:x[1].astype(np.bool)}),
                    '{0[id]}.R.ArealDistortion_FS.59k_fs_LR.shape.gii': (
                        {'type':'property',
                         'name':'areal_distortion_fs',
                         'hemi':'rh_LR59k_MSMSulc'},
                        {'type':'property',
                         'name':'areal_distortion_fs',
                         'hemi':'rh_LR59k_MSMAll'}),
                    '{0[id]}.R.ArealDistortion_MSMSulc.59k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'areal_distortion',
                        'hemi':'rh_LR59k_MSMSulc'},
                    '{0[id]}.R.MyelinMap.59k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin',
                        'hemi':'rh_LR59k_MSMSulc'},
                    #'{0[id]}.R.MyelinMap_BC.59k_fs_LR.func.gii': {
                    #    'type':'property',
                    #    'name':'myelin_bc',
                    #    'hemi':'rh_LR59k_MSMSulc'},
                    '{0[id]}.R.SmoothedMyelinMap.59k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin_smooth',
                        'hemi':'rh_LR59k_MSMSulc'},
                    #'{0[id]}.R.SmoothedMyelinMap_BC.59k_fs_LR.func.gii': {
                    #    'type':'property',
                    #    'name':'myelin_smooth_bc',
                    #    'hemi':'rh_LR59k_MSMSulc'},
                    '{0[id]}.R.curvature.59k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'curvature',
                        'hemi':'rh_LR59k_MSMSulc',
                        'filt':lambda c: -c},
                    '{0[id]}.R.sulc.59k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'convexity',
                        'hemi':'rh_LR59k_MSMSulc'},
                    #'{0[id]}.R.corrThickness.59k_fs_LR.shape.gii': {
                    #    'type':'property',
                    #    'name':'thickness',
                    #    'hemi':'rh_LR59k_MSMSulc'},
                    '{0[id]}.R.thickness.59k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'thickness_uncorrected',
                        'hemi':'rh_LR59k_MSMSulc'},
                    '{0[id]}.R.white.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'white',
                        'hemi':'rh_LR59k_MSMSulc'},
                    '{0[id]}.R.midthickness.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'midgray',
                        'hemi':'rh_LR59k_MSMSulc'},
                    '{0[id]}.R.pial.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'pial',
                        'hemi':'rh_LR59k_MSMSulc'},
                    '{0[id]}.R.inflated.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'inflated',
                        'hemi':'rh_LR59k_MSMSulc'},
                    '{0[id]}.R.very_inflated.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'very_inflated',
                        'hemi':'rh_LR59k_MSMSulc'},
                    '{0[id]}.R.white_1.6mm_MSMAll.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'white',
                        'hemi':'rh_LR59k_MSMAll'},
                    '{0[id]}.R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'midgray',
                        'hemi':'rh_LR59k_MSMAll'},
                    '{0[id]}.R.pial_1.6mm_MSMAll.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'pial',
                        'hemi':'rh_LR59k_MSMAll'},
                    '{0[id]}.R.inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'inflated',
                        'hemi':'rh_LR59k_MSMAll'},
                    '{0[id]}.R.very_inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'very_inflated',
                        'hemi':'rh_LR59k_MSMAll'},
                    #'{0[id]}.R.flat.59k_fs_LR.surf.gii': (
                    #    {'type':'surface',
                    #     'name':'flat',
                    #     'hemi':'rh_LR59k_MSMSulc',
                    #     'load':_load_atlas_sphere},
                    #    {'type':'surface',
                    #     'name':'flat',
                    #     'hemi':'rh_LR59k_MSMAll',
                    #     'load':_load_atlas_sphere}),
                    '{0[id]}.R.sphere.59k_fs_LR.surf.gii': (
                        {'type':'registration',
                         'name':'fs_LR',
                         'hemi':'rh_LR59k_MSMSulc',
                         'load':_load_atlas_sphere},
                        {'type':'registration',
                         'name':'fs_LR',
                         'hemi':'rh_LR59k_MSMAll',
                         'load':_load_atlas_sphere}),
                    '{0[id]}.ArealDistortion_1.6mm_MSMAll.59k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'areal_distortion',
                        'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')},
                    '{0[id]}.MyelinMap_BC_1.6mm_MSMAll.59k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'myelin_bc',
                        'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')},
                    '{0[id]}.SmoothedMyelinMap_BC_1.6mm_MSMAll.59k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'myelin_smooth_bc',
                        'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')},
                    '{0[id]}.curvature_1.6mm_MSMAll.59k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'curvature',
                        'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll'),
                        'filt':lambda c: -c},
                    '{0[id]}.sulc_1.6mm_MSMAll.59k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'convexity',
                        'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')},
                    '{0[id]}.corrThickness_1.6mm_MSMAll.59k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'thickness',
                        'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')},
                    '{0[id]}.thickness_1.6mm_MSMAll.59k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'thickness_uncorrected',
                        'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')}}},
            'fsaverage_LR32k': {
                'type':'dir',
                'contents': {
                    '{0[id]}.L.BA.32k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'lh_LR32k_MSMSulc'},
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'lh_LR32k_MSMAll'}),
                    '{0[id]}.L.aparc.32k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'lh_LR32k_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'lh_LR32k_MSMAll'}),
                    '{0[id]}.L.aparc.a2009s.32k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'lh_LR32k_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'lh_LR32k_MSMAll'}),
                    '{0[id]}.L.atlasroi.32k_fs_LR.shape.gii': (
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'lh_LR32k_MSMSulc',
                         'load':_load_fsLR_atlasroi,
                         'filt':lambda x:x[0].astype(np.bool)},
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'lh_LR32k_MSMAll',
                         'load':_load_fsLR_atlasroi,
                         'filt':lambda x:x[0].astype(np.bool)}),
                    '{0[id]}.L.ArealDistortion_FS.32k_fs_LR.shape.gii': (
                        {'type':'property',
                         'name':'areal_distortion_fs',
                         'hemi':'lh_LR32k_MSMSulc'},
                        {'type':'property',
                         'name':'areal_distortion_fs',
                         'hemi':'lh_LR32k_MSMAll'}),
                    '{0[id]}.L.ArealDistortion_MSMSulc.32k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'areal_distortion',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.MyelinMap.32k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.MyelinMap_BC.32k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin_bc',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.SmoothedMyelinMap.32k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin_smooth',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.SmoothedMyelinMap_BC.32k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin_smooth_bc',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.curvature.32k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'curvature',
                        'hemi':'lh_LR32k_MSMSulc',
                        'filt':lambda c: -c},
                    '{0[id]}.L.sulc.32k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'convexity',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.corrThickness.32k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'thickness',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.thickness.32k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'thickness_uncorrected',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.white.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'white',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.midthickness.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'midgray',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.pial.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'pial',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.inflated.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'inflated',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.very_inflated.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'very_inflated',
                        'hemi':'lh_LR32k_MSMSulc'},
                    '{0[id]}.L.white_MSMAll.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'white',
                        'hemi':'lh_LR32k_MSMAll'},
                    '{0[id]}.L.midthickness_MSMAll.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'midgray',
                        'hemi':'lh_LR32k_MSMAll'},
                    '{0[id]}.L.pial_MSMAll.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'pial',
                        'hemi':'lh_LR32k_MSMAll'},
                    '{0[id]}.L.inflated_MSMAll.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'inflated',
                        'hemi':'lh_LR32k_MSMAll'},
                    '{0[id]}.L.very_inflated_MSMAll.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'very_inflated',
                        'hemi':'lh_LR32k_MSMAll'},
                    #'{0[id]}.L.flat.32k_fs_LR.surf.gii': (
                    #    {'type':'surface',
                    #     'name':'flat',
                    #     'hemi':'lh_LR32k_MSMSulc'},
                    #    {'type':'surface',
                    #     'name':'flat',
                    #     'hemi':'lh_LR32k_MSMAll'}),
                    '{0[id]}.L.sphere.32k_fs_LR.surf.gii': (
                        {'type':'registration',
                         'name':'fs_LR',
                         'hemi':'lh_LR32k_MSMSulc'},
                        {'type':'registration',
                         'name':'fs_LR',
                         'hemi':'lh_LR32k_MSMAll'}),
                    '{0[id]}.R.BA.32k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'rh_LR32k_MSMSulc'},
                        {'type':'property',
                         'name':'brodmann_area',
                         'hemi':'rh_LR32k_MSMAll'}),
                    '{0[id]}.R.aparc.32k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'rh_LR32k_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation_2005',
                         'hemi':'rh_LR32k_MSMAll'}),
                    '{0[id]}.R.aparc.a2009s.32k_fs_LR.label.gii': (
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'rh_LR32k_MSMSulc'},
                        {'type':'property',
                         'name':'parcellation',
                         'hemi':'rh_LR32k_MSMAll'}),
                    '{0[id]}.R.atlasroi.32k_fs_LR.shape.gii': (
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'rh_LR32k_MSMSulc',
                         'load':_load_fsLR_atlasroi,
                         'filt':lambda x:x[1].astype(np.bool)},
                        {'type':'property',
                         'name':'atlas',
                         'hemi':'rh_LR32k_MSMAll',
                         'load':_load_fsLR_atlasroi,
                         'filt':lambda x:x[1].astype(np.bool)}),
                    '{0[id]}.R.ArealDistortion_FS.32k_fs_LR.shape.gii': (
                        {'type':'property',
                         'name':'areal_distortion_fs',
                         'hemi':'rh_LR32k_MSMSulc'},
                        {'type':'property',
                         'name':'areal_distortion_fs',
                         'hemi':'rh_LR32k_MSMAll'}),
                    '{0[id]}.R.ArealDistortion_MSMSulc.32k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'areal_distortion',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.MyelinMap.32k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.MyelinMap_BC.32k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin_bc',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.SmoothedMyelinMap.32k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin_smooth',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.SmoothedMyelinMap_BC.32k_fs_LR.func.gii': {
                        'type':'property',
                        'name':'myelin_smooth_bc',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.curvature.32k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'curvature',
                        'hemi':'rh_LR32k_MSMSulc',
                        'filt':lambda c: -c},
                    '{0[id]}.R.sulc.32k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'convexity',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.corrThickness.32k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'thickness',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.thickness.32k_fs_LR.shape.gii': {
                        'type':'property',
                        'name':'thickness_uncorrected',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.white.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'white',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.midthickness.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'midgray',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.pial.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'pial',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.inflated.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'inflated',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.very_inflated.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'very_inflated',
                        'hemi':'rh_LR32k_MSMSulc'},
                    '{0[id]}.R.white_MSMAll.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'white',
                        'hemi':'rh_LR32k_MSMAll'},
                    '{0[id]}.R.midthickness_MSMAll.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'midgray',
                        'hemi':'rh_LR32k_MSMAll'},
                    '{0[id]}.R.pial_MSMAll.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'pial',
                        'hemi':'rh_LR32k_MSMAll'},
                    '{0[id]}.R.inflated_MSMAll.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'inflated',
                        'hemi':'rh_LR32k_MSMAll'},
                    '{0[id]}.R.very_inflated_MSMAll.32k_fs_LR.surf.gii': {
                        'type':'surface',
                        'name':'very_inflated',
                        'hemi':'rh_LR32k_MSMAll'},
                    #'{0[id]}.R.flat.32k_fs_LR.surf.gii': (
                    #    {'type':'surface',
                    #     'name':'flat',
                    #     'hemi':'rh_LR32k_MSMSulc',
                    #     'load':_load_atlas_sphere},
                    #    {'type':'surface',
                    #     'name':'flat',
                    #     'hemi':'rh_LR32k_MSMAll',
                    #     'load':_load_atlas_sphere}),
                    '{0[id]}.R.sphere.32k_fs_LR.surf.gii': (
                        {'type':'registration',
                         'name':'fs_LR',
                         'hemi':'rh_LR32k_MSMSulc',
                         'load':_load_atlas_sphere},
                        {'type':'registration',
                         'name':'fs_LR',
                         'hemi':'rh_LR32k_MSMAll',
                         'load':_load_atlas_sphere}),
                    '{0[id]}.ArealDistortion_MSMAll.32k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'areal_distortion',
                        'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')},
                    '{0[id]}.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'myelin_bc',
                        'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')},
                    '{0[id]}.SmoothedMyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'myelin_smooth_bc',
                        'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')},
                    '{0[id]}.curvature_MSMAll.32k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'curvature',
                        'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll'),
                        'filt':lambda c: -c},
                    '{0[id]}.sulc_MSMAll.32k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'convexity',
                        'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')},
                    '{0[id]}.corrThickness_MSMAll.32k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'thickness',
                        'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')},
                    '{0[id]}.thickness_MSMAll.32k_fs_LR.dscalar.nii': {
                        'type':'property',
                        'name':'thickness_uncorrected',
                        'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')}}}}}}

# Okay, convert that to something organized by hemisphere/image
def _organize_subject_directory_structure(ss):
    imgs = {}
    hems = {}
    fmap = {}
    # for walking the structure:
    def _visit(u, path, key):
        if isinstance(u, (tuple, list)):
            for uu in u: _visit(uu, path, key)
            return
        t = u['type']
        # dir is a slightly special case:
        if t == 'dir':
            newpath = os.path.join(path, key)
            for (k,v) in six.iteritems(u['contents']):
                _visit(v, newpath, k)
            return
        # Not a dir, so we can get the name and filename
        name = u['name']
        flnm = os.path.join(path, key)
        fmdat = {'filt':u.get('filt', None), 'load':u.get('load', None), 'type':t,
                 'hemi':u.get('hemi', None)}
        if t == 'image':
            if name in imgs: raise ValueError('Duplicate image in file spec: %s' % name)
            imgs[name] = flnm
            fmap[flnm] = fmdat
            return
        # not an image, so also has a hemisphere...
        h = u['hemi']
        if isinstance(h, tuple):
            hs = h
            fmdat['cifti'] = True
        else:
            hs = [h]
            fmdat['cifti'] = False
        for hh in hs:
            if hh not in hems:
                hems[hh] = {'registrations':{}, 'surfaces':{}, 'properties':{}}
        if t == 'surface':
            if name in hems[h]['surfaces']:
                raise ValueError('Duplicate surface %s in hemi %s' % (name, h))
            hems[h]['surfaces'][name] = flnm
            fmap[flnm] = fmdat
        elif t == 'registration':
            if name in hems[h]['registrations']:
                raise ValueError('Duplicate registration %s in hemi %s' % (name, h))
            hems[h]['registrations'][name] = flnm
            fmap[flnm] = fmdat
        elif t == 'property':
            if fmdat['cifti']:
                # this is a cifti file...
                fmdat['hemi'] = h
                for hh in h:
                    if name in hems[hh]['properties']:
                        raise ValueError('Duplicate property (cifti) %s in hemi %s' % (name, hh))
                    hems[hh]['properties'][name] = flnm
            else:
                # normal gifti file
                if name in hems[h]['properties']:
                    raise ValueError('Duplicate property %s in hemi %s' % (name, h))
                hems[h]['properties'][name] = flnm
            fmap[flnm] = fmdat
        else:
            raise ValueError('Unrecognized node type: %s' % t)
    # The lowest level is a directory...
    for (k,v) in six.iteritems(ss):
        _visit(v, '', k)
    # That should fix it all up!
    return {'hemis':   hems,
            'images':  imgs,
            'filemap': fmap}

subject_structure = _organize_subject_directory_structure(subject_directory_structure)


####################################################################################################
# Downloaders
# First, we can download a subject using s3fs, assuming we have the appropriate credentials.
# We can also set things up to auto-download a subject whenever they are requested but not detected.

def download(sid, credentials=None, subjects_path=None, overwrite=False, release='HCP_1200',
             database='hcp-openaccess', file_list=None):
    '''
    download(sid) downloads the data for subject with the given subject id. By default, the subject
      will be placed in the first HCP subject directory in the subjects directories list.

    Note: In order for downloading to work, you must have s3fs installed. This is not a requirement
    for the neuropythy library and does not install automatically when installing via pip. The
    github repository for this library can be found at https://github.com/dask/s3fs. Installation
    instructions can be found here: http://s3fs.readthedocs.io/en/latest/install.html

    Accepted options include:
      * credentials (default: None) may be used to specify the Amazon AWS Bucket credentials, which
        can be generated from the HCP db (https://db.humanconnectome.org/). If this argument can be
        coerced to a credentials tuple via the to_credentials function, that result will be used. If
        None, then the function will try to use the hcp_credentials configuration item in
        neuropythy.config; otherwise an error is raised.
      * subjects_path (default: None) specifies where the subject should be placed. If None, then
        the first directory in the subjects paths list is used. If there is not one of these then
        an error is raised.
      * overwrite (default: False) specifies whether or not to overwrite files that already exist.
        In addition to True (do overwrite) and False (don't overwrite), the value 'error' indicates
        that an error should be raised if a file already exists.
    '''
    if s3fs is None:
        raise RuntimeError('s3fs was not successfully loaded, so downloads may not occur; check '
                           'your Python configuration to make sure that s3fs is installed. See '
                           'http://s3fs.readthedocs.io/en/latest/install.html for details.')
    if credentials is None: credentials = config['hcp_credentials']
    if credentials is None: raise ValueError('No hcp_credentials specified or found')
    (s3fs_key, s3fs_secret) = to_credentials(credentials)
    if subjects_path is None:
        sdirs = config['hcp_subject_paths']
        subjects_path = next((sd for sd in sdirs if os.path.isdir(sd)), None)
        if subjects_path is None: raise ValueError('No subjects path given or found')
    else: subjects_path = os.path.expanduser(subjects_path)
    # Make sure we can connect to the bucket first...
    fs = s3fs.S3FileSystem(key=s3fs_key, secret=s3fs_secret)
    # Okay, make sure the release is found
    if not fs.exists('/'.join([database, release])):
        raise ValueError('database/release (%s/%s) not found' % (database, release))
    # Check on the subject id to
    sid = to_subject_id(sid)
    hcp_sdir = '/'.join([database, release, str(sid)])
    if not fs.exists(hcp_sdir): raise ValueError('Subject %d not found in release' % sid)
    # Okay, lets download this subject!
    loc_sdir = os.path.join(subjects_path, str(sid))
    # walk through the subject structures
    pulled = []
    for flnm in six.iterkeys(subject_structure['filemap']):
        flnm = flnm.format({'id':sid})
        loc_flnm = os.path.join(loc_sdir, flnm)
        hcp_flnm = '/'.join([hcp_sdir, flnm])
        if not overwrite and os.path.isfile(loc_flnm): continue
        # gotta download it!
        basedir = os.path.split(loc_flnm)[0]
        if not os.path.isdir(basedir): os.makedirs(os.path.abspath(basedir), 0o755)
        fs.get(hcp_flnm, loc_flnm)
        pulled.append(loc_flnm)
    return pulled

_retinotopy_path = None
_retinotopy_file = {32:'prfresults.mat', 59:'prfresults59k.mat'}
_retinotopy_url  = {32:'https://osf.io/yus6t/download', 59:None}
retinotopy_prefix = 'prf'
lowres_retinotopy_prefix = 'lowres-prf'
inferred_retinotopy_prefix = 'inf-prf'
lowres_inferred_retinotopy_prefix = 'inf-lowres-prf'

# If _auto_download_options is None, then no auto-downloading is enabled; if it is a map of
# options (even an empty one) then auto-downloading is enabled using the given options
_auto_download_options = None
def auto_download(status,
                  credentials=None, subjects_path=None, overwrite=False, release='HCP_1200',
                  database='hcp-openaccess', retinotopy_path=None, retinotopy_cache=True):
    '''
    auto_download(True) enables automatic downloading of HCP subject data when the subject ID
      is requested. The optional arguments are identical to those required for the function
      download(), and they are passed to download() when auto-downloading occurs.
    auto_download(False) disables automatic downloading.

    Automatic downloading is disabled by default unless the environment variable
    HCP_AUTO_DOWNLOAD is set to true. In this case, the database and release are derived from
    the environment variables HCP_AUTO_DATABASE and HCP_AUTO_RELEASE, and the variable
    HCP_AUTO_PATH can be used to override the default subjects path.
    '''
    global _auto_download_options, _retinotopy_path
    status = (['structure','retinotopy'] if status is True       else
              []                         if status is False      else
              [status]                   if pimms.is_str(status) else
              status)
    _auto_download_options = {'structure':False, 'retinotopy':False}
    for s in status:
        if s.lower() == 'structure':
            if s3fs is None:
                raise RuntimeError(
                    's3fs was not successfully loaded, so downloads may not occur; check'
                    ' your Python configuration to make sure that s3fs is installed. See'
                    ' http://s3fs.readthedocs.io/en/latest/install.html for details.')
            if credentials is None: credentials = config['hcp_credentials']
            if credentials is None: raise ValueError('No HCP credentials detected or found')
            (s3fs_key, s3fs_secret) = to_credentials(credentials)
            if subjects_path is None:
                sdirs = config['hcp_subject_paths']
                subjects_path = next((sd for sd in sdirs if os.path.isdir(sd)), None)
                if subjects_path is None: raise ValueError('No subjects path given or found')
            else: subjects_path = os.path.expanduser(subjects_path)
            fs = s3fs.S3FileSystem(key=s3fs_key, secret=s3fs_secret)
            hcpbase = '/'.join([database, release])
            if not fs.exists(hcpbase):
                raise ValueError('database/release (%s/%s) not found' % (database, release))
            sids = set([])
            for f in fs.ls(hcpbase):
                f = os.path.split(f)[-1]
                if len(f) == 6 and f[0] != '0':
                    try: sids.add(int(f))
                    except Exception: pass
            _auto_download_options['structure'] = True
            _auto_download_options['subjects_path'] = subjects_path
            _auto_download_options['overwrite'] = overwrite
            _auto_download_options['release'] = release
            _auto_download_options['database'] = database
            _auto_download_options['subject_ids'] = frozenset(sids)
            _auto_download_options['s3fs'] = fs
        elif s.lower() == 'retinotopy':
            if retinotopy_path is None:
                dirs = config['hcp_subject_paths']
                if subjects_path is not None: dirs = [subjects_path] + list(dirs)
                if _retinotopy_path is not None: dirs = [_retinotopy_path] + list(dirs)
                retinotopy_path = next((sd for sd in dirs if os.path.isdir(sd)), None)
            if retinotopy_path is None: raise ValueError('No retinotopy path given or found')
            else: retinotopy_path = os.path.expanduser(retinotopy_path)
            _auto_download_options['retinotopy'] = True
            _auto_download_options['retinotopy_path'] = retinotopy_path
            _auto_download_options['retinotopy_cache'] = retinotopy_cache
        else: raise ValueError('unrecognized auto_download argument: %s' % s)
    if all(v is False for v in six.itervalues(_auto_download_options)):
        _auto_download_options = None
# See if the config/environment lets auto-downloading start in the "on" state
def to_auto_download_state(arg):
    '''
    to_auto_download_state(arg) attempts to coerce the given argument into a valid auto-downloading
      instruction. Essentially, if arg is "on", "yes", "true", "1", True, or 1, then True is
      returned; if arg is "structure" then "structure" is returned; otherwise False is returned.
    '''
    if pimms.is_str(arg):
        arg = arg.lower().strip()
        if arg in ('on', 'yes', 'true', '1'): return True
        elif arg in ('struct', 'structure', 'structural'): return 'structure'
        elif arg in ('retinotopy', 'retino', 'ret'): return 'retinotopy'
        else: return False
    return arg in (True, 1)

config.declare('hcp_auto_release',  environ_name='HCP_AUTO_RELEASE')
config.declare('hcp_auto_database', environ_name='HCP_AUTO_DATABASE')
config.declare('hcp_auto_path',     environ_name='HCP_AUTO_PATH')
config.declare('hcp_auto_download', environ_name='HCP_AUTO_DOWNLOAD',
               filter=to_auto_download_state, default_value=False)
if config['hcp_auto_download'] is not False:
    try:
        args = {}
        if config['hcp_auto_release']:  args['release']       = config['hcp_auto_release']
        if config['hcp_auto_database']: args['database']      = config['hcp_auto_database']
        if config['hcp_auto_path']:     args['subjects_path'] = config['hcp_auto_path']
        auto_download(True, **args)
    except Exception:
        logging.warn('Could not initialize HCP auto-downloading from configuration data.')
def _auto_downloadable(sid):
    if _auto_download_options is None: return False
    elif sid == 'retinotopy': return _auto_download_options['retinotopy']
    elif not _auto_download_options['structure']: return False
    else: return to_subject_id(sid) in _auto_download_options['subject_ids']
# these expect value % (hemi, alignment, surfacename)
_retinotopy_cache_tr = {
    'native': {
        (retinotopy_prefix + '_polar_angle')                  : '%s.%s_angle.native59k.mgz',
        (retinotopy_prefix + '_eccentricity')                 : '%s.%s_eccen.native59k.mgz',
        (retinotopy_prefix + '_radius')                       : '%s.%s_prfsz.native59k.mgz',
        (retinotopy_prefix + '_variance_explained')           : '%s.%s_vexpl.native59k.mgz',
        (lowres_retinotopy_prefix + '_polar_angle')           : '%s.%s_angle.native32k.mgz',
        (lowres_retinotopy_prefix + '_eccentricity')          : '%s.%s_eccen.native32k.mgz',
        (lowres_retinotopy_prefix + '_radius')                : '%s.%s_prfsz.native32k.mgz',
        (lowres_retinotopy_prefix + '_variance_explained')    : '%s.%s_vexpl.native32k.mgz',
        (inferred_retinotopy_prefix + '_polar_angle')         : '%s.inf-%s_angle.native59k.mgz',
        (inferred_retinotopy_prefix + '_eccentricity')        : '%s.inf-%s_eccen.native59k.mgz',
        (inferred_retinotopy_prefix + '_radius')              : '%s.inf-%s_sigma.native59k.mgz',
        (inferred_retinotopy_prefix + '_visual_area')         : '%s.inf-%s_varea.native59k.mgz',
        (lowres_inferred_retinotopy_prefix + '_polar_angle')  : '%s.inf-%s_angle.native32k.mgz',
        (lowres_inferred_retinotopy_prefix + '_eccentricity') : '%s.inf-%s_eccen.native32k.mgz',
        (lowres_inferred_retinotopy_prefix + '_radius')       : '%s.inf-%s_sigma.native32k.mgz',
        (lowres_inferred_retinotopy_prefix + '_visual_area')  : '%s.inf-%s_varea.native32k.mgz'},
    'LR32k': {
        (lowres_retinotopy_prefix + '_polar_angle')           : '%s.%s_angle.32k.mgz',
        (lowres_retinotopy_prefix + '_eccentricity')          : '%s.%s_eccen.32k.mgz',
        (lowres_retinotopy_prefix + '_radius')                : '%s.%s_prfsz.32k.mgz',
        (lowres_retinotopy_prefix + '_variance_explained')    : '%s.%s_vexpl.32k.mgz'},
    'LR59k': {
        (retinotopy_prefix + '_polar_angle')                  : '%s.%s_angle.59k.mgz',
        (retinotopy_prefix + '_eccentricity')                 : '%s.%s_eccen.59k.mgz',
        (retinotopy_prefix + '_radius')                       : '%s.%s_prfsz.59k.mgz',
        (retinotopy_prefix + '_variance_explained')           : '%s.%s_vexpl.59k.mgz'}}
_retinotopy_cache_surf_tr = {
    'native59k': '%s_native_%s',
    'native32k': '%s_native_%s',
    '59k': '%s_LR59k_%s',
    '32k': '%s_LR32k_%s'}
def load_retinotopy_cache(sdir, sid, alignment='MSMAll'):
    '''
    Returns the subject's retinotopy cache as a lazy map, or None if no cache exists. The hemi keys
    in the returned map will spell out the cortex name (e.g., lh_native_MSMAll or rh_LR32k_MSMAll).
    '''
    fs = {h:{(k,kk):os.path.join(sdir, 'retinotopy', v % (h, alignment))
             for (kk,vv) in six.iteritems(_retinotopy_cache_tr)
             for (k,v)   in six.iteritems(vv)}
          for h in ('lh','rh')}
    files = {}
    for h in ['lh','rh']:
        for ((k,kk),v) in six.iteritems(fs[h]):
            if not os.path.isfile(v): continue
            hh = _retinotopy_cache_surf_tr[v.split('.')[-2]] % (h, alignment)
            if hh not in files: files[hh] = {}
            files[hh][k] = v
    if len(files) == 0: return files
    def _loader(fls, h, k): return (lambda:nyio.load(fls[h][k]))
    return {h:pimms.lazy_map({k:_loader(files, h, k) for k in six.iterkeys(v)})
            for (h,v) in six.iteritems(files)}
def save_retinotopy_cache(sdir, sid, hemi, props, alignment='MSMAll', overwrite=False):
    '''
    Saves the subject's retinotopy cache from the given properties. The first argument is the
    subject's directory (not the subjects' directory).
    '''
    h = hemi[:2]
    htype = hemi.split('_')[1]
    if _auto_download_options is None \
       or 'retinotopy_cache' not in _auto_download_options \
       or not _auto_download_options['retinotopy_cache']:
        return
    files = {k:os.path.join(sdir, 'retinotopy', v % (h, alignment))
             for (k,v) in six.iteritems(_retinotopy_cache_tr[htype])}
    for (p,fl) in six.iteritems(files):
        if p not in props or (not overwrite and os.path.exists(fl)): continue
        p = np.asarray(props[p])
        if np.issubdtype(p.dtype, np.floating): p = np.asarray(p, np.float32)
        dr = os.path.split(os.path.abspath(fl))[0]
        if not os.path.isdir(dr): os.makedirs(os.path.abspath(dr), 0o755)
        nyio.save(fl, p)
def _find_retinotopy_path(size=59):
    dirs = config['hcp_subject_paths']
    if _auto_download_options is not None \
       and 'retinotopy' in _auto_download_options and _auto_download_options['retinotopy'] \
       and 'retinotopy_path' in _auto_download_options \
       and _auto_download_options['retinotopy_path'] is not None \
       and _retinotopy_url[size] is not None:
        pth = os.path.join(_auto_download_options['retinotopy_path'], _retinotopy_file[size])
        if os.path.isfile(pth): return pth
        # okay, try to download it!
        import shutil
        from six.moves import urllib
        logging.info('neuropythy: Fetchinging HCP retinotopy database "%s"', pth)
        if six.PY2:
            response = urllib.request.urlopen(_retinotopy_url[size])
            with open(pth, 'wb') as fl:
                shutil.copyfileobj(response, fl)
        else:
            with urllib.request.urlopen(_retinotopy_url[size]) as response:
                with open(pth, 'wb') as fl:
                    shutil.copyfileobj(response, fl)
        return pth
    if _retinotopy_path is not None: dirs = [_retinotopy_path] + config['hcp_subject_paths']
    d = next((sd for sd in dirs if os.path.isfile(os.path.join(sd, _retinotopy_file[size]))), None)
    return d if d is None else os.path.join(d, _retinotopy_file[size])
def _retinotopy_open(fn, size=59):
    pth = _find_retinotopy_path(size=size)
    if pth is not None:
        import h5py
        with h5py.File(pth, 'r') as f: return fn(f)
    return None
def _retinotopy_submap(size=59):
    if _retinotopy_submap.cache is None: _retinotopy_submap.cache = {}
    if size not in _retinotopy_submap.cache:
        tmp = _retinotopy_open(
            lambda f: np.squeeze(np.array(f['subjectids'], dtype=np.int)),
            size=size)
        if tmp is None: return None
        tmp = pyr.pmap({sid:k for (k,sid) in enumerate(tmp)})
        _retinotopy_submap.cache[size] = tmp
    return _retinotopy_submap.cache[size]
_retinotopy_submap.cache = None
def _retinotopy_dset(name, size=59):
    name = name.lower()
    if name in ['full', 'type1', '1', 'all']:  name = 0
    elif name in ['half1', 'split1', 'type1']: name = 1
    elif name in ['half2', 'split2', 'type3']: name = 2
    else: raise ValueError('name must be "full", "half1", or "half2"')
    if _retinotopy_dset.cache[size][name] is None:
        arr = _retinotopy_open(lambda f: f['allresults'][name], size=size)
        if arr is None: return None
        arr = np.array(arr)
        arr.setflags(write=False)
        _retinotopy_dset.cache[size][name] = arr
    return _retinotopy_dset.cache[size][name]
_retinotopy_dset.cache = {32:[None,None,None], 59:[None,None,None]}
def _cifti_to_hemis(data, sid=100610):
    (la, ra) = _load_fsLR_atlasroi_for_size(data.shape[0])
    (ln, rn) = [aa.shape[0]     for aa in (la, ra)]
    (li, ri) = [np.where(aa)[0] for aa in (la, ra)]
    (lu, ru) = [len(ai)         for ai in (li, ri)]
    lsl = slice(0,  lu)
    rsl = slice(lu, lu + ru)
    (ldat, rdat) = [np.zeros((len(aa),) + data.shape[1:], dtype=data.dtype) for aa in (la,ra)]
    for (dat,sl,ii) in zip([ldat,rdat],[lsl,rsl],[li,ri]): dat[ii] = data[sl]
    return (ldat, rdat)
def _retinotopy_data(name, sid, size=59):
    smap = _retinotopy_submap(size=size)
    if smap is None or sid not in smap: return None
    arr = _retinotopy_dset(name, size=size)
    dat = arr[smap[sid]]
    return pyr.m(
        prf_polar_angle        = _cifti_to_hemis(np.mod(90 - dat[0] + 180, 360) - 180, sid),
        prf_eccentricity       = _cifti_to_hemis(dat[1], sid),
        prf_radius             = _cifti_to_hemis(dat[5], sid),
        prf_variance_explained = _cifti_to_hemis(dat[4]/100.0, sid))
    
def subject_filemap(sid, subject_path=None):
    '''
    subject_filemap(sid) yields a persistent lazy map structure that loads the relevant files as
      requested for the given subject. The sid may be a subject id or the path of a subject
      directory. If a subject id is given, then the subject is searched for in the known subject
      paths.

    The optional argument subject_path may be set to a specific path to ensure that the subject
    id is only searched for in the given path.
    '''
    # see if sid is a subject id
    if pimms.is_int(sid):
        if subject_path is None: sdir = find_subject_path(sid)
        else: sdir = os.path.expanduser(os.path.join(subject_path, str(sid)))
    elif pimms.is_str(sid):
        try: sid = os.path.expanduser(sid)
        except Exception: pass
        sdir = sid if os.path.isdir(sid) else find_subject_path(sid)
        sid  = int(sdir.split(os.sep)[-1])
    else: raise ValueError('Cannot understand HCP subject ID %s' % sid)
    if sdir is None:
        if _auto_download_options is not None and _auto_download_options['structure'] \
           and _auto_downloadable(sid):
            # we didn't find it, but we have a place to put it
            sdir = _auto_download_options['subjects_path']
            sdir = os.path.join(sdir, str(sid))
            if not os.path.isdir(sdir): os.makedirs(os.path.abspath(sdir), 0o755)
        else:
            raise ValueError('Could not find HCP subject %s' % sid)
    ff = {'id':sid}
    def _make_lambda(flnm, dat): return lambda:_load(flnm, dat)
    # walk through the subject structure's filemap to make a lazy map that loads things
    dats = {}
    fls = {}
    for (k,v) in six.iteritems(subject_structure['filemap']):
        flnm = os.path.join(sdir, k.format(ff))
        dat  = pimms.merge(v, ff)
        fls[flnm]  = _make_lambda(flnm, dat)
        dats[flnm] = dat
    fls = pimms.lazy_map(fls)
    # and the the hemispheres to make hemispheres...
    def _lookup_fl(flnm):
        def _f():
            obj = fls[flnm]
            dat = dats[flnm]
            if 'cifti' in dat and dat['cifti']:
                # we need to handle the cifti files by splitting them up according to atlases
                (ldat,rdat) = _cifti_to_hemis(np.asarray(obj), sid)
                obj = ldat if dat['hemi'][0:2] == 'lh' else rdat
                obj.setflags(write=False)
            return obj
        return _f
    hems = pyr.pmap(
        {h: pyr.pmap(
            {k: pimms.lazy_map({nm: _lookup_fl(fnm)
                                for (nm,fl) in six.iteritems(v)
                                for fnm in [os.path.join(sdir, fl.format(ff))]})
             for (k,v) in six.iteritems(entries)})
         for (h,entries) in six.iteritems(subject_structure['hemis'])})
    # and the images
    imgs = pimms.lazy_map({k: _lookup_fl(os.path.join(sdir, v.format(ff)))
                           for (k,v) in six.iteritems(subject_structure['images'])})
    # and retinotopy if appropriate
    for size in [32, 59]:
        retsubs = _retinotopy_submap(size=size)
        if retsubs is not None and sid in retsubs:
            def _make_loader(sz):
                rp = retinotopy_prefix if sz == 59 else lowres_retinotopy_prefix
                ldr = pimms.lazy_map({0:lambda:_retinotopy_data('full', sid, size=sz)})
                return pyr.pmap(
                    {('lh_LR%dk_MSMAll' % sz):pimms.lazy_map(
                        {(rp+'_polar_angle'):lambda:ldr[0]['prf_polar_angle'][0],
                         (rp+'_eccentricity'):lambda:ldr[0]['prf_eccentricity'][0],
                         (rp+'_radius'):lambda:ldr[0]['prf_radius'][0],
                         (rp+'_variance_explained'):lambda:ldr[0]['prf_variance_explained'][0]}),
                    ('rh_LR%dk_MSMAll' % sz):pimms.lazy_map(
                        {(rp+'_polar_angle'):lambda:ldr[0]['prf_polar_angle'][1],
                         (rp+'_eccentricity'):lambda:ldr[0]['prf_eccentricity'][1],
                         (rp+'_radius'):lambda:ldr[0]['prf_radius'][1],
                         (rp+'_variance_explained'):lambda:ldr[0]['prf_variance_explained'][1]})})
            ret = _make_loader(size)
            # merge into the hemis
            for h in six.iterkeys(ret):
                hdat = hems[h]
                prop = pimms.merge(hdat['properties'], ret[h])
                hems = hems.set(h, hdat.set('properties', prop))
    return pyr.pmap({'images': imgs, 'hemis': hems})

hcp_filemap_data_hierarchy = [['image'],
                              ['hemi', 'surface'],
                              ['hemi', 'registration'],
                              ['hemi', 'property']]
hcp_filemap_instructions = [
    'T1w', [
        'BiasField_acpc_dc.nii.gz',         {'image':'bias'},
        'T1wDividedByT2w.nii.gz',           {'image':'T1_to_T2_ratio_all'},
        'T1wDividedByT2w_ribbon.nii.gz',    {'image':'T1_to_T2_ratio'},
        'T1w_acpc_dc_restore.nii.gz',       {'image':'T1'},
        'T1w_acpc_dc.nii.gz',               {'image':'T1_unrestored'},
        'T1w_acpc_dc_restore_brain.nii.gz', {'image':'brain'},
        'T2w_acpc_dc_restore.nii.gz',       {'image':'T2'},
        'T2w_acpc_dc.nii.gz',               {'image':'T2_unrestored'},
        'T2w_acpc_dc_restore_brain.nii.gz', {'image':'T2_brain'},
        'aparc+aseg.nii.gz',                {'image':'parcellation2005'},
        'aparc.a2009s+aseg.nii.gz',         {'image':'parcellation'},
        'brainmask_fs.nii.gz',              {'image':'brainmask'},
        'ribbon.nii.gz',                    {'image':'ribbon'},
        'wmparc.nii.gz',                    {'image':'wm_parcellation'},
        '{0[id]}', [
            'surf', [
                'lh.area', (
                    {'property':'white_surface_area',
                     'hemi':'lh_native_MSMSulc', 'load':_load_fsmorph},
                    {'property':'white_surface_area',
                     'hemi':'lh_native_MSMAll',  'load':_load_fsmorph}),
                'lh.area.mid', (
                    {'property':'midgray_surface_area',
                     'hemi':'lh_native_MSMSulc', 'load':_load_fsmorph},
                    {'property':'midgray_surface_area',
                     'hemi':'lh_native_MSMAll',  'load':_load_fsmorph}),
                'lh.area.pial', (
                    {'property':'pial_surface_area',
                     'hemi':'lh_native_MSMSulc', 'load':_load_fsmorph},
                    {'property':'pial_surface_area',
                     'hemi':'lh_native_MSMAll',  'load':_load_fsmorph}),
                'rh.area', (
                    {'property':'white_surface_area',
                     'hemi':'rh_native_MSMSulc', 'load':_load_fsmorph},
                    {'property':'white_surface_area',
                     'hemi':'rh_native_MSMAll',  'load':_load_fsmorph}),
                'rh.area.mid', (
                    {'property':'midgray_surface_area',
                     'hemi':'rh_native_MSMSulc', 'load':_load_fsmorph},
                    {'property':'midgray_surface_area',
                     'hemi':'rh_native_MSMAll',  'load':_load_fsmorph}),
                'rh.area.pial', (
                    {'property':'pial_surface_area',
                     'hemi':'rh_native_MSMSulc', 'load':_load_fsmorph},
                    {'property':'pial_surface_area',
                     'hemi':'rh_native_MSMAll',  'load':_load_fsmorph})]],
        'Native', [
            '{0[id]}.L.white.native.surf.gii', (
                {'surface':'white', 'hemi':'lh_native_MSMSulc'},
                {'surface':'white', 'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.midthickness.native.surf.gii', (
                {'surface':'midgray', 'hemi':'lh_native_MSMSulc'},
                {'surface':'midgray', 'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.pial.native.surf.gii', (
                {'surface':'pial', 'hemi':'lh_native_MSMSulc'},
                {'surface':'pial', 'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.inflated.native.surf.gii', (
                {'surface':'inflated', 'hemi':'lh_native_MSMSulc'},
                {'surface':'inflated', 'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.very_inflated.native.surf.gii', (
                {'surface':'very_inflated', 'hemi':'lh_native_MSMSulc'},
                {'surface':'very_inflated', 'hemi':'lh_native_MSMAll'}),
            '{0[id]}.R.white.native.surf.gii', (
                {'surface':'white', 'hemi':'rh_native_MSMSulc'},
                {'surface':'white', 'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.midthickness.native.surf.gii', (
                {'surface':'midgray', 'hemi':'rh_native_MSMSulc'},
                {'surface':'midgray', 'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.pial.native.surf.gii', (
                {'surface':'pial', 'hemi':'rh_native_MSMSulc'},
                {'surface':'pial', 'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.inflated.native.surf.gii', (
                {'surface':'inflated', 'hemi':'rh_native_MSMSulc'},
                {'surface':'inflated', 'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.very_inflated.native.surf.gii', (
                {'surface':'very_inflated', 'hemi':'rh_native_MSMSulc'},
                {'surface':'very_inflated', 'hemi':'rh_native_MSMAll'})],
        'fsaverage_LR32k', [
            '{0[id]}.L.inflated.32k_fs_LR.surf.gii',             {'surface':'inflated',
                                                                  'hemi':'lh_lowres_MSMSulc'},
            '{0[id]}.L.midthickness.32k_fs_LR.surf.gii',         {'surface':'midgray',
                                                                  'hemi':'lh_lowres_MSMSulc'},
            '{0[id]}.L.pial.32k_fs_LR.surf.gii',                 {'surface':'pial',
                                                                  'hemi':'lh_lowres_MSMSulc'},
            '{0[id]}.L.very_inflated.32k_fs_LR.surf.gii',        {'surface':'very_inflated',
                                                                  'hemi':'lh_lowres_MSMSulc'},
            '{0[id]}.L.white.32k_fs_LR.surf.gii',                {'surface':'white',
                                                                  'hemi':'lh_lowres_MSMSulc'},
            '{0[id]}.R.inflated.32k_fs_LR.surf.gii',             {'surface':'inflated',
                                                                  'hemi':'rh_lowres_MSMSulc'},
            '{0[id]}.R.midthickness.32k_fs_LR.surf.gii',         {'surface':'midgray',
                                                                  'hemi':'rh_lowres_MSMSulc'},
            '{0[id]}.R.pial.32k_fs_LR.surf.gii',                 {'surface':'pial',
                                                                  'hemi':'rh_lowres_MSMSulc'},
            '{0[id]}.R.very_inflated.32k_fs_LR.surf.gii',        {'surface':'very_inflated',
                                                                  'hemi':'rh_lowres_MSMSulc'},
            '{0[id]}.R.white.32k_fs_LR.surf.gii',                {'surface':'white',
                                                                  'hemi':'rh_lowres_MSMSulc'},
            '{0[id]}.L.inflated_MSMAll.32k_fs_LR.surf.gii',      {'surface':'inflated',
                                                                  'hemi':'lh_lowres_MSMAll'},
            '{0[id]}.L.midthickness_MSMAll.32k_fs_LR.surf.gii',  {'surface':'midgray',
                                                                  'hemi':'lh_lowres_MSMAll'},
            '{0[id]}.L.pial_MSMAll.32k_fs_LR.surf.gii',          {'surface':'pial',
                                                                  'hemi':'lh_lowres_MSMAll'},
            '{0[id]}.L.very_inflated_MSMAll.32k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                                  'hemi':'lh_lowres_MSMAll'},
            '{0[id]}.L.white_MSMAll.32k_fs_LR.surf.gii',         {'surface':'white',
                                                                  'hemi':'lh_lowres_MSMAll'},
            '{0[id]}.R.inflated_MSMAll.32k_fs_LR.surf.gii',      {'surface':'inflated',
                                                                  'hemi':'rh_lowres_MSMAll'},
            '{0[id]}.R.midthickness_MSMAll.32k_fs_LR.surf.gii',  {'surface':'midgray',
                                                                  'hemi':'rh_lowres_MSMAll'},
            '{0[id]}.R.pial_MSMAll.32k_fs_LR.surf.gii',          {'surface':'pial',
                                                                  'hemi':'rh_lowres_MSMAll'},
            '{0[id]}.R.very_inflated_MSMAll.32k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                                  'hemi':'rh_lowres_MSMAll'},
            '{0[id]}.R.white_MSMAll.32k_fs_LR.surf.gii',         {'surface':'white',
                                                                  'hemi':'rh_lowres_MSMAll'}]],
    'MNINonLinear', [
        'BiasField.nii.gz',         {'image':'bias_warped'},
        'T1w_restore.nii.gz',       {'image':'T1_warped'},
        'T1w.nii.gz',               {'image':'T1_warped_unrestored'},
        'T1w_restore_brain.nii.gz', {'image':'brain_warped'},
        'T2w_restore.nii.gz',       {'image':'T2_warped'},
        'T2w.nii.gz',               {'image':'T2_warped_unrestored'},
        'T2w_restore_brain.nii.gz', {'image':'T2_brain_warped'},
        'aparc+aseg.nii.gz',        {'image':'parcellation2005_warped'},
        'aparc.a2009s+aseg.nii.gz', {'image':'parcellation_warped'},
        'brainmask_fs.nii.gz',      {'image':'brainmask_warped'},
        'ribbon.nii.gz',            {'image':'ribbon_warped'},
        'wmparc.nii.gz',            {'image':'wm_parcellation_warped'},
        '{0[id]}.L.ArealDistortion_FS.164k_fs_LR.shape.gii',      {'property':'areal_distortion_FS',
                                                                   'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.ArealDistortion_MSMSulc.164k_fs_LR.shape.gii', {'property':'areal_distortion',
                                                                   'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.MyelinMap.164k_fs_LR.func.gii',                {'property':'myelin',
                                                                   'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.MyelinMap_BC.164k_fs_LR.func.gii',             {'property':'myelin_bc',
                                                                   'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.SmoothedMyelinMap.164k_fs_LR.func.gii',        {'property':'myelin_smooth',
                                                                   'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.SmoothedMyelinMap_BC.164k_fs_LR.func.gii',     {'property':'myelin_smooth_bc',
                                                                   'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.RefMyelinMap.164k_fs_LR.func.gii',             {'property':'myelin_ref',
                                                                   'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.BA.164k_fs_LR.label.gii',                      ({'property':'brodmann_area',
                                                                    'hemi':'lh_LR164k_MSMSulc'},
                                                                   {'property':'brodmann_area',
                                                                    'hemi':'lh_LR164k_MSMAll'}),
        '{0[id]}.L.aparc.164k_fs_LR.label.gii',                   ({'property':'parcellation_2005',
                                                                    'hemi':'lh_LR164k_MSMSulc'},
                                                                   {'property':'parcellation_2005',
                                                                    'hemi':'lh_LR164k_MSMAll'}),
        '{0[id]}.L.aparc.a2009s.164k_fs_LR.label.gii',            ({'property':'parcellation',
                                                                    'hemi':'lh_LR164k_MSMSulc'},
                                                                   {'property':'parcellation',
                                                                    'hemi':'lh_LR164k_MSMAll'}),
        '{0[id]}.L.atlasroi.164k_fs_LR.shape.gii',      ({'property':'atlas',
                                                          'hemi':'lh_LR164k_MSMSulc',
                                                          'load':_load_fsLR_atlasroi,
                                                          'filt':lambda x:x[0].astype(np.bool)},
                                                         {'property':'atlas',
                                                          'hemi':'lh_LR164k_MSMAll',
                                                          'load':_load_fsLR_atlasroi,
                                                          'filt':lambda x:x[0].astype(np.bool)}),
        '{0[id]}.L.curvature.164k_fs_LR.shape.gii',     {'property':'curvature',
                                                         'hemi':'lh_LR164k_MSMSulc',
                                                         'filt':lambda c: -c},
        '{0[id]}.L.sulc.164k_fs_LR.shape.gii',          {'property':'convexity',
                                                         'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.corrThickness.164k_fs_LR.shape.gii', {'property':'thickness',
                                                         'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.thickness.164k_fs_LR.shape.gii',     {'property':'thickness_uncorrected',
                                                         'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.white.164k_fs_LR.surf.gii',         {'surface':'white',
                                                        'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.midthickness.164k_fs_LR.surf.gii',  {'surface':'midgray',
                                                        'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.pial.164k_fs_LR.surf.gii',          {'surface':'pial',
                                                        'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.inflated.164k_fs_LR.surf.gii',      {'surface':'inflated',
                                                        'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.very_inflated.164k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                        'hemi':'lh_LR164k_MSMSulc'},
        '{0[id]}.L.white_MSMAll.164k_fs_LR.surf.gii',  {'surface':'white',
                                                        'hemi':'lh_LR164k_MSMAll'},
        '{0[id]}.L.midthickness_MSMAll.164k_fs_LR.surf.gii',  {'surface':'midgray',
                                                               'hemi':'lh_LR164k_MSMAll'},
        '{0[id]}.L.pial_MSMAll.164k_fs_LR.surf.gii',          {'surface':'pial',
                                                               'hemi':'lh_LR164k_MSMAll'},
        '{0[id]}.L.inflated_MSMAll.164k_fs_LR.surf.gii',      {'surface':'inflated',
                                                               'hemi':'lh_LR164k_MSMAll'},
        '{0[id]}.L.very_inflated_MSMAll.164k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                               'hemi':'lh_LR164k_MSMAll'},
        '{0[id]}.L.sphere.164k_fs_LR.surf.gii',               ({'registration':'fs_LR',
                                                                'hemi':'lh_LR164k_MSMSulc',
                                                                'load':_load_atlas_sphere},
                                                               {'registration':'fs_LR',
                                                                'hemi':'lh_LR164k_MSMAll',
                                                                'load':_load_atlas_sphere}),
        # disabled until I decide how they should be handled (their tesselations not like the
        # spheres, so they don't really belong in the same hemisphere)
        #'{0[id]}.L.flat.164k_fs_LR.surf.gii', (
        #    {'surface':'flat',
        #     'hemi':'lh_LR164k_MSMSulc',
        #     'load':_load_atlas_sphere},
        #    {'surface':'flat',
        #     'hemi':'lh_LR164k_MSMAll',
        #     'load':_load_atlas_sphere}),
        '{0[id]}.R.ArealDistortion_FS.164k_fs_LR.shape.gii',      {'property':'areal_distortion_FS',
                                                                   'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.ArealDistortion_MSMSulc.164k_fs_LR.shape.gii', {'property':'areal_distortion',
                                                                   'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.MyelinMap.164k_fs_LR.func.gii',                {'property':'myelin',
                                                                   'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.MyelinMap_BC.164k_fs_LR.func.gii',             {'property':'myelin_bc',
                                                                   'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.SmoothedMyelinMap.164k_fs_LR.func.gii',        {'property':'myelin_smooth',
                                                                   'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.SmoothedMyelinMap_BC.164k_fs_LR.func.gii',     {'property':'myelin_smooth_bc',
                                                                   'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.RefMyelinMap.164k_fs_LR.func.gii',             {'property':'myelin_ref',
                                                                   'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.BA.164k_fs_LR.label.gii',                      ({'property':'brodmann_area',
                                                                    'hemi':'rh_LR164k_MSMSulc'},
                                                                   {'property':'brodmann_area',
                                                                    'hemi':'rh_LR164k_MSMAll'}),
        '{0[id]}.R.aparc.164k_fs_LR.label.gii',                   ({'property':'parcellation_2005',
                                                                    'hemi':'rh_LR164k_MSMSulc'},
                                                                   {'property':'parcellation_2005',
                                                                    'hemi':'rh_LR164k_MSMAll'}),
        '{0[id]}.R.aparc.a2009s.164k_fs_LR.label.gii',            ({'property':'parcellation',
                                                                    'hemi':'rh_LR164k_MSMSulc'},
                                                                   {'property':'parcellation',
                                                                    'hemi':'rh_LR164k_MSMAll'}),
        '{0[id]}.R.atlasroi.164k_fs_LR.shape.gii',      ({'property':'atlas',
                                                          'hemi':'rh_LR164k_MSMSulc',
                                                          'load':_load_fsLR_atlasroi,
                                                          'filt':lambda x:x[1].astype(np.bool)},
                                                         {'property':'atlas',
                                                          'hemi':'rh_LR164k_MSMAll',
                                                          'load':_load_fsLR_atlasroi,
                                                          'filt':lambda x:x[1].astype(np.bool)}),
        '{0[id]}.R.curvature.164k_fs_LR.shape.gii',     {'property':'curvature',
                                                         'hemi':'rh_LR164k_MSMSulc',
                                                         'filt':lambda c: -c},
        '{0[id]}.R.sulc.164k_fs_LR.shape.gii',          {'property':'convexity',
                                                         'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.corrThickness.164k_fs_LR.shape.gii', {'property':'thickness',
                                                         'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.thickness.164k_fs_LR.shape.gii',     {'property':'thickness_uncorrected',
                                                         'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.sulc.164k_fs_LR.shape.gii',          {'surface':'convexity',
                                                         'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.white.164k_fs_LR.surf.gii',          {'surface':'white',
                                                         'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.midthickness.164k_fs_LR.surf.gii',   {'surface':'midgray',
                                                         'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.pial.164k_fs_LR.surf.gii',           {'surface':'pial',
                                                         'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.inflated.164k_fs_LR.surf.gii',       {'surface':'inflated',
                                                         'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.very_inflated.164k_fs_LR.surf.gii',  {'surface':'very_inflated',
                                                         'hemi':'rh_LR164k_MSMSulc'},
        '{0[id]}.R.white_MSMAll.164k_fs_LR.surf.gii',   {'surface':'white',
                                                         'hemi':'rh_LR164k_MSMAll'},
        '{0[id]}.R.midthickness_MSMAll.164k_fs_LR.surf.gii',  {'surface':'midgray',
                                                               'hemi':'rh_LR164k_MSMAll'},
        '{0[id]}.R.pial_MSMAll.164k_fs_LR.surf.gii',          {'surface':'pial',
                                                               'hemi':'rh_LR164k_MSMAll'},
        '{0[id]}.R.inflated_MSMAll.164k_fs_LR.surf.gii',      {'surface':'inflated',
                                                               'hemi':'rh_LR164k_MSMAll'},
        '{0[id]}.R.very_inflated_MSMAll.164k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                               'hemi':'rh_LR164k_MSMAll'},
        '{0[id]}.R.sphere.164k_fs_LR.surf.gii', ({'registration':'fs_LR',
                                                  'hemi':'rh_LR164k_MSMSulc',
                                                  'load':_load_atlas_sphere},
                                                 {'registration':'fs_LR',
                                                  'hemi':'rh_LR164k_MSMAll',
                                                  'load':_load_atlas_sphere}),
        #'{0[id]}.R.flat.164k_fs_LR.surf.gii', (
        #    {'surface':'flat',
        #     'hemi':'rh_LR164k_MSMSulc',
        #     'load':_load_atlas_sphere},
        #    {'surface':'flat',
        #     'hemi':'rh_LR164k_MSMAll',
        #     'load':_load_atlas_sphere}),
        '{0[id]}.ArealDistortion_MSMAll.164k_fs_LR.dscalar.nii', {
            'property': 'areal_distortion',
            'hemi':     ('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
        '{0[id]}.MyelinMap_BC_MSMAll.164k_fs_LR.dscalar.nii', {
            'property': 'myelin_bc',
            'hemi':     ('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
        '{0[id]}.SmoothedMyelinMap_BC_MSMAll.164k_fs_LR.dscalar.nii', {
            'property': 'myelin_smooth_bc',
            'hemi':     ('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
        '{0[id]}.curvature_MSMAll.164k_fs_LR.dscalar.nii', {
            'property': 'curvature',
            'hemi':     ('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll'),
            'filt':     lambda c: -c},
        '{0[id]}.sulc.164k_fs_LR.dscalar.nii', {
            'property': 'convexity',
            'hemi':     ('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
        '{0[id]}.corrThickness.164k_fs_LR.dscalar.nii', {
            'property': 'thickness',
            'hemi':     ('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
        '{0[id]}.thickness.164k_fs_LR.dscalar.nii', {
            'property': 'thickness_uncorrected',
            'hemi':     ('lh_LR164k_MSMAll', 'rh_LR164k_MSMAll')},
        'Native', [
            '{0[id]}.L.ArealDistortion_FS.native.shape.gii', (
                {'property':'areal_distortion_FS', 'hemi':'lh_native_MSMSulc'},
                {'property':'areal_distortion_FS', 'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.ArealDistortion_MSMSulc.native.shape.gii', {
                    'property':'areal_distortion',
                    'hemi':'lh_native_MSMSulc'},
            '{0[id]}.L.ArealDistortion_MSMAll.native.shape.gii', {'property':'areal_distortion',
                                                                  'hemi':'lh_native_MSMAll'},
            '{0[id]}.L.MyelinMap.native.func.gii', {'property':'myelin',
                                                    'hemi':'lh_native_MSMSulc'},
            '{0[id]}.L.MyelinMap_BC.native.func.gii', {'property':'myelin_bc',
                                                       'hemi':'lh_native_MSMSulc'},
            '{0[id]}.L.SmoothedMyelinMap.native.func.gii', {'property':'myelin_smooth',
                                                            'hemi':'lh_native_MSMSulc'},
            '{0[id]}.L.SmoothedMyelinMap_BC.native.func.gii', {'property':'myelin_smooth_bc',
                                                               'hemi':'lh_native_MSMSulc'},
            '{0[id]}.L.RefMyelinMap.native.func.gii', {'property':'myelin_ref',
                                                       'hemi':'lh_native_MSMSulc'},
            '{0[id]}.L.BA.native.label.gii', ({'property':'brodmann_area',
                                               'hemi':'lh_native_MSMSulc'},
                                              {'property':'brodmann_area',
                                               'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.aparc.native.label.gii',  ({'property':'parcellation_2005',
                                                   'hemi':'lh_native_MSMSulc'},
                                                  {'property':'parcellation_2005',
                                                   'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.aparc.a2009s.native.label.gii', ({'property':'parcellation',
                                                         'hemi':'lh_native_MSMSulc'},
                                                        {'property':'parcellation',
                                                         'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.atlasroi.native.shape.gii', ({'property':'atlas',
                                                     'hemi':'lh_native_MSMSulc',
                                                     'filt':lambda x:x.astype(np.bool)},
                                                    {'property':'atlas',
                                                     'hemi':'lh_native_MSMAll',
                                                     'filt':lambda x:x.astype(np.bool)}),
            '{0[id]}.L.curvature.native.shape.gii', ({'property':'curvature',
                                                      'hemi':'lh_native_MSMSulc',
                                                      'filt':lambda c: -c},
                                                     {'property':'curvature',
                                                      'hemi':'lh_native_MSMAll',
                                                      'filt':lambda c: -c}),
            '{0[id]}.L.sulc.native.shape.gii', ({'property':'convexity',
                                                 'hemi':'lh_native_MSMSulc'},
                                                {'property':'convexity',
                                                 'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.corrThickness.native.shape.gii', ({'property':'thickness',
                                                          'hemi':'lh_native_MSMSulc'},
                                                         {'property':'thickness',
                                                          'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.thickness.native.shape.gii', ({'property':'thickness_uncorrected',
                                                      'hemi':'lh_native_MSMSulc'},
                                                     {'property':'thickness_uncorrected',
                                                      'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.roi.native.shape.gii', ({'property':'roi',
                                                'hemi':'lh_native_MSMSulc',
                                                'filt':lambda r: r.astype(bool)},
                                               {'property':'roi',
                                                'hemi':'lh_native_MSMAll',
                                                'filt':lambda r: r.astype(bool)}),
            '{0[id]}.L.sphere.native.surf.gii', ({'registration':'native',
                                                  'hemi':'lh_native_MSMSulc'},
                                                 {'registration':'native',
                                                  'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.sphere.reg.native.surf.gii', ({'registration':'fsaverage',
                                                      'hemi':'lh_native_MSMSulc'},
                                                     {'registration':'fsaverage',
                                                      'hemi':'lh_native_MSMAll'}),
            '{0[id]}.L.sphere.MSMAll.native.surf.gii', {'registration':'fs_LR',
                                                        'tool':'MSMAll',
                                                        'hemi':'lh_native_MSMAll'},
            '{0[id]}.L.sphere.MSMSulc.native.surf.gii', {'registration':'fs_LR',
                                                         'tool':'MSMSulc',
                                                         'hemi':'lh_native_MSMSulc'},
            '{0[id]}.R.ArealDistortion_FS.native.shape.gii', ({'property':'areal_distortion_FS',
                                                               'hemi':'rh_native_MSMSulc'},
                                                              {'property':'areal_distortion_FS',
                                                               'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.ArealDistortion_MSMSulc.native.shape.gii', {'property':'areal_distortion',
                                                                   'hemi':'rh_native_MSMSulc'},
            '{0[id]}.R.ArealDistortion_MSMAll.native.shape.gii', {'property':'areal_distortion',
                                                                  'hemi':'rh_native_MSMAll'},
            '{0[id]}.R.MyelinMap.native.func.gii', {'property':'myelin',
                                                    'hemi':'rh_native_MSMSulc'},
            '{0[id]}.R.MyelinMap_BC.native.func.gii', {'property':'myelin_bc',
                                                       'hemi':'rh_native_MSMSulc'},
            '{0[id]}.R.SmoothedMyelinMap.native.func.gii', {'property':'myelin_smooth',
                                                            'hemi':'rh_native_MSMSulc'},
            '{0[id]}.R.SmoothedMyelinMap_BC.native.func.gii', {'property':'myelin_smooth_bc',
                                                               'hemi':'rh_native_MSMSulc'},
            '{0[id]}.R.RefMyelinMap.native.func.gii', {'property':'myelin_ref',
                                                       'hemi':'rh_native_MSMSulc'},
            '{0[id]}.R.BA.native.label.gii', ({'property':'brodmann_area',
                                               'hemi':'rh_native_MSMSulc'},
                                              {'property':'brodmann_area',
                                               'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.aparc.native.label.gii',  ({'property':'parcellation_2005',
                                                   'hemi':'rh_native_MSMSulc'},
                                                  {'property':'parcellation_2005',
                                                   'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.aparc.a2009s.native.label.gii', ({'property':'parcellation',
                                                         'hemi':'rh_native_MSMSulc'},
                                                        {'property':'parcellation',
                                                         'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.atlasroi.native.shape.gii', ({'property':'atlas',
                                                     'hemi':'rh_native_MSMSulc',
                                                     'filt':lambda x:x.astype(np.bool)},
                                                    {'property':'atlas',
                                                     'hemi':'rh_native_MSMAll',
                                                     'filt':lambda x:x.astype(np.bool)}),
            '{0[id]}.R.curvature.native.shape.gii', ({'property':'curvature',
                                                      'hemi':'rh_native_MSMSulc',
                                                      'filt':lambda c: -c},
                                                     {'property':'curvature',
                                                      'hemi':'rh_native_MSMAll',
                                                      'filt':lambda c: -c}),
            '{0[id]}.R.sulc.native.shape.gii', ({'property':'convexity',
                                                 'hemi':'rh_native_MSMSulc'},
                                                {'property':'convexity',
                                                 'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.corrThickness.native.shape.gii', ({'property':'thickness',
                                                          'hemi':'rh_native_MSMSulc'},
                                                         {'property':'thickness',
                                                          'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.thickness.native.shape.gii', ({'property':'thickness_uncorrected',
                                                      'hemi':'rh_native_MSMSulc'},
                                                     {'property':'thickness_uncorrected',
                                                      'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.roi.native.shape.gii', ({'property':'roi',
                                                'hemi':'rh_native_MSMSulc',
                                                'filt':lambda r: r.astype(bool)},
                                               {'property':'roi',
                                                'hemi':'rh_native_MSMAll',
                                                'filt':lambda r: r.astype(bool)}),
            '{0[id]}.R.sphere.native.surf.gii', ({'registration':'native',
                                                  'hemi':'rh_native_MSMSulc'},
                                                 {'registration':'native',
                                                  'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.sphere.reg.native.surf.gii', ({'registration':'fsaverage',
                                                      'hemi':'rh_native_MSMSulc'},
                                                     {'registration':'fsaverage',
                                                      'hemi':'rh_native_MSMAll'}),
            '{0[id]}.R.sphere.MSMAll.native.surf.gii', {'registration':'fs_LR',
                                                        'tool':'MSMAll',
                                                        'hemi':'rh_native_MSMAll'},
            '{0[id]}.R.sphere.MSMSulc.native.surf.gii', {'registration':'fs_LR',
                                                         'tool':'MSMSulc',
                                                         'hemi':'rh_native_MSMSulc'}],
        'fsaverage_LR59k', [
            '{0[id]}.L.BA.59k_fs_LR.label.gii', ({'property':'brodmann_area',
                                                  'hemi':'lh_LR59k_MSMSulc'},
                                                 {'property':'brodmann_area',
                                                  'hemi':'lh_LR59k_MSMAll'}),
            '{0[id]}.L.aparc.59k_fs_LR.label.gii', ({'property':'parcellation_2005',
                                                     'hemi':'lh_LR59k_MSMSulc'},
                                                    {'property':'parcellation_2005',
                                                     'hemi':'lh_LR59k_MSMAll'}),
            '{0[id]}.L.aparc.a2009s.59k_fs_LR.label.gii', ({'property':'parcellation',
                                                            'hemi':'lh_LR59k_MSMSulc'},
                                                           {'property':'parcellation',
                                                            'hemi':'lh_LR59k_MSMAll'}),
            '{0[id]}.L.atlasroi.59k_fs_LR.shape.gii', ({'property':'atlas',
                                                        'hemi':'lh_LR59k_MSMSulc',
                                                        'load':_load_fsLR_atlasroi,
                                                        'filt':lambda x:x[0].astype(np.bool)},
                                                       {'property':'atlas',
                                                        'hemi':'lh_LR59k_MSMAll',
                                                        'load':_load_fsLR_atlasroi,
                                                        'filt':lambda x:x[0].astype(np.bool)}),
            '{0[id]}.L.ArealDistortion_FS.59k_fs_LR.shape.gii', ({'property':'areal_distortion_fs',
                                                                  'hemi':'lh_LR59k_MSMSulc'},
                                                                 {'property':'areal_distortion_fs',
                                                                  'hemi':'lh_LR59k_MSMAll'}),
            '{0[id]}.L.ArealDistortion_MSMSulc.59k_fs_LR.shape.gii', {'property':'areal_distortion',
                                                                      'hemi':'lh_LR59k_MSMSulc'},
            '{0[id]}.L.MyelinMap.59k_fs_LR.func.gii', {'property':'myelin',
                                                       'hemi':'lh_LR59k_MSMSulc'},
            #'{0[id]}.L.MyelinMap_BC.59k_fs_LR.func.gii', {
            #    'property':'myelin_bc',
            #    'hemi':'lh_LR59k_MSMSulc'},
            '{0[id]}.L.SmoothedMyelinMap.59k_fs_LR.func.gii', {'property':'myelin_smooth',
                                                               'hemi':'lh_LR59k_MSMSulc'},
            #'{0[id]}.L.SmoothedMyelinMap_BC.59k_fs_LR.func.gii', {
            #    'property':'myelin_smooth_bc',
            #    'hemi':'lh_LR59k_MSMSulc'},
            '{0[id]}.L.curvature.59k_fs_LR.shape.gii', {'property':'curvature',
                                                        'hemi':'lh_LR59k_MSMSulc',
                                                        'filt':lambda c: -c},
            '{0[id]}.L.sulc.59k_fs_LR.shape.gii', {'property':'convexity',
                                                   'hemi':'lh_LR59k_MSMSulc'},
            #'{0[id]}.L.corrThickness.59k_fs_LR.shape.gii', {
            #    'property':'thickness',
            #    'hemi':'lh_LR59k_MSMSulc'},
            '{0[id]}.L.thickness.59k_fs_LR.shape.gii', {'property':'thickness_uncorrected',
                                                        'hemi':'lh_LR59k_MSMSulc'},
            '{0[id]}.L.white.59k_fs_LR.surf.gii', {'surface':'white',
                                                   'hemi':'lh_LR59k_MSMSulc'},
            '{0[id]}.L.midthickness.59k_fs_LR.surf.gii', {'surface':'midgray',
                                                          'hemi':'lh_LR59k_MSMSulc'},
            '{0[id]}.L.pial.59k_fs_LR.surf.gii', {'surface':'pial',
                                                  'hemi':'lh_LR59k_MSMSulc'},
            '{0[id]}.L.inflated.59k_fs_LR.surf.gii', {'surface':'inflated',
                                                      'hemi':'lh_LR59k_MSMSulc'},
            '{0[id]}.L.very_inflated.59k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                           'hemi':'lh_LR59k_MSMSulc'},
            '{0[id]}.L.white_1.6mm_MSMAll.59k_fs_LR.surf.gii', {'surface':'white',
                                                                'hemi':'lh_LR59k_MSMAll'},
            '{0[id]}.L.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii', {'surface':'midgray',
                                                                       'hemi':'lh_LR59k_MSMAll'},
            '{0[id]}.L.pial_1.6mm_MSMAll.59k_fs_LR.surf.gii', {'surface':'pial',
                                                               'hemi':'lh_LR59k_MSMAll'},
            '{0[id]}.L.inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii', {'surface':'inflated',
                                                                   'hemi':'lh_LR59k_MSMAll'},
            '{0[id]}.L.very_inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                                        'hemi':'lh_LR59k_MSMAll'},
            #'{0[id]}.L.flat.59k_fs_LR.surf.gii', (
            #    {'surface':'flat',
            #     'hemi':'lh_LR59k_MSMSulc'},
            #    {'surface':'flat',
            #     'hemi':'lh_LR59k_MSMAll'}),
            '{0[id]}.L.sphere.59k_fs_LR.surf.gii', ({'registration':'fs_LR',
                                                     'hemi':'lh_LR59k_MSMSulc'},
                                                    {'registration':'fs_LR',
                                                     'hemi':'lh_LR59k_MSMAll'}),
            '{0[id]}.R.BA.59k_fs_LR.label.gii', ({'property':'brodmann_area',
                                                  'hemi':'rh_LR59k_MSMSulc'},
                                                 {'property':'brodmann_area',
                                                  'hemi':'rh_LR59k_MSMAll'}),
            '{0[id]}.R.aparc.59k_fs_LR.label.gii', ({'property':'parcellation_2005',
                                                     'hemi':'rh_LR59k_MSMSulc'},
                                                    {'property':'parcellation_2005',
                                                     'hemi':'rh_LR59k_MSMAll'}),
            '{0[id]}.R.aparc.a2009s.59k_fs_LR.label.gii', ({'property':'parcellation',
                                                            'hemi':'rh_LR59k_MSMSulc'},
                                                           {'property':'parcellation',
                                                            'hemi':'rh_LR59k_MSMAll'}),
            '{0[id]}.R.atlasroi.59k_fs_LR.shape.gii', ({'property':'atlas',
                                                        'hemi':'rh_LR59k_MSMSulc',
                                                        'load':_load_fsLR_atlasroi,
                                                        'filt':lambda x:x[1].astype(np.bool)},
                                                       {'property':'atlas',
                                                        'hemi':'rh_LR59k_MSMAll',
                                                        'load':_load_fsLR_atlasroi,
                                                        'filt':lambda x:x[1].astype(np.bool)}),
            '{0[id]}.R.ArealDistortion_FS.59k_fs_LR.shape.gii', ({'property':'areal_distortion_fs',
                                                                  'hemi':'rh_LR59k_MSMSulc'},
                                                                 {'property':'areal_distortion_fs',
                                                                  'hemi':'rh_LR59k_MSMAll'}),
            '{0[id]}.R.ArealDistortion_MSMSulc.59k_fs_LR.shape.gii', {'property':'areal_distortion',
                                                                      'hemi':'rh_LR59k_MSMSulc'},
            '{0[id]}.R.MyelinMap.59k_fs_LR.func.gii', {'property':'myelin',
                                                       'hemi':'rh_LR59k_MSMSulc'},
            #'{0[id]}.R.MyelinMap_BC.59k_fs_LR.func.gii', {
            #    'property':'myelin_bc',
            #    'hemi':'rh_LR59k_MSMSulc'},
            '{0[id]}.R.SmoothedMyelinMap.59k_fs_LR.func.gii', {'property':'myelin_smooth',
                                                               'hemi':'rh_LR59k_MSMSulc'},
            #'{0[id]}.R.SmoothedMyelinMap_BC.59k_fs_LR.func.gii', {
            #    'property':'myelin_smooth_bc',
            #    'hemi':'rh_LR59k_MSMSulc'},
            '{0[id]}.R.curvature.59k_fs_LR.shape.gii', {'property':'curvature',
                                                        'hemi':'rh_LR59k_MSMSulc',
                                                        'filt':lambda c: -c},
            '{0[id]}.R.sulc.59k_fs_LR.shape.gii', {'property':'convexity',
                                                   'hemi':'rh_LR59k_MSMSulc'},
            #'{0[id]}.R.corrThickness.59k_fs_LR.shape.gii', {
            #    'property':'thickness',
            #    'hemi':'rh_LR59k_MSMSulc'},
            '{0[id]}.R.thickness.59k_fs_LR.shape.gii', {'property':'thickness_uncorrected',
                                                        'hemi':'rh_LR59k_MSMSulc'},
            '{0[id]}.R.white.59k_fs_LR.surf.gii', {'surface':'white',
                                                   'hemi':'rh_LR59k_MSMSulc'},
            '{0[id]}.R.midthickness.59k_fs_LR.surf.gii', {'surface':'midgray',
                                                          'hemi':'rh_LR59k_MSMSulc'},
            '{0[id]}.R.pial.59k_fs_LR.surf.gii', {'surface':'pial',
                                                  'hemi':'rh_LR59k_MSMSulc'},
            '{0[id]}.R.inflated.59k_fs_LR.surf.gii', {'surface':'inflated',
                                                      'hemi':'rh_LR59k_MSMSulc'},
            '{0[id]}.R.very_inflated.59k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                           'hemi':'rh_LR59k_MSMSulc'},
            '{0[id]}.R.white_1.6mm_MSMAll.59k_fs_LR.surf.gii', {'surface':'white',
                                                                'hemi':'rh_LR59k_MSMAll'},
            '{0[id]}.R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii', {'surface':'midgray',
                                                                       'hemi':'rh_LR59k_MSMAll'},
            '{0[id]}.R.pial_1.6mm_MSMAll.59k_fs_LR.surf.gii', {'surface':'pial',
                                                               'hemi':'rh_LR59k_MSMAll'},
            '{0[id]}.R.inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii', {'surface':'inflated',
                                                                   'hemi':'rh_LR59k_MSMAll'},
            '{0[id]}.R.very_inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                                        'hemi':'rh_LR59k_MSMAll'},
            #'{0[id]}.R.flat.59k_fs_LR.surf.gii', (
            #    {'surface':'flat',
            #     'hemi':'rh_LR59k_MSMSulc',
            #     'load':_load_atlas_sphere},
            #    {'surface':'flat',
            #     'hemi':'rh_LR59k_MSMAll',
            #     'load':_load_atlas_sphere}),
            '{0[id]}.R.sphere.59k_fs_LR.surf.gii', ({'registration':'fs_LR',
                                                     'hemi':'rh_LR59k_MSMSulc',
                                                     'load':_load_atlas_sphere},
                                                    {'registration':'fs_LR',
                                                     'hemi':'rh_LR59k_MSMAll',
                                                     'load':_load_atlas_sphere}),
            '{0[id]}.ArealDistortion_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', {
                'property':'areal_distortion',
                'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')},
            '{0[id]}.MyelinMap_BC_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', {
                'property':'myelin_bc',
                'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')},
            '{0[id]}.SmoothedMyelinMap_BC_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', {
                'property':'myelin_smooth_bc',
                'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')},
            '{0[id]}.curvature_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', {
                'property':'curvature',
                'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll'),
                'filt':lambda c: -c},
            '{0[id]}.sulc_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', {
                'property':'convexity',
                'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')},
            '{0[id]}.corrThickness_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', {
                'property':'thickness',
                'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')},
            '{0[id]}.thickness_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', {
                'property':'thickness_uncorrected',
                'hemi':('lh_LR59k_MSMAll', 'rh_LR59k_MSMAll')}],
        'fsaverage_LR32k', [
            '{0[id]}.L.BA.32k_fs_LR.label.gii', ({'property':'brodmann_area',
                                                  'hemi':'lh_LR32k_MSMSulc'},
                                                 {'property':'brodmann_area',
                                                  'hemi':'lh_LR32k_MSMAll'}),
            '{0[id]}.L.aparc.32k_fs_LR.label.gii', ({'property':'parcellation_2005',
                                                     'hemi':'lh_LR32k_MSMSulc'},
                                                    {'property':'parcellation_2005',
                                                     'hemi':'lh_LR32k_MSMAll'}),
            '{0[id]}.L.aparc.a2009s.32k_fs_LR.label.gii', ({'property':'parcellation',
                                                            'hemi':'lh_LR32k_MSMSulc'},
                                                           {'property':'parcellation',
                                                            'hemi':'lh_LR32k_MSMAll'}),
            '{0[id]}.L.atlasroi.32k_fs_LR.shape.gii', ({'property':'atlas',
                                                        'hemi':'lh_LR32k_MSMSulc',
                                                        'load':_load_fsLR_atlasroi,
                                                        'filt':lambda x:x[0].astype(np.bool)},
                                                       {'property':'atlas',
                                                        'hemi':'lh_LR32k_MSMAll',
                                                        'load':_load_fsLR_atlasroi,
                                                        'filt':lambda x:x[0].astype(np.bool)}),
            '{0[id]}.L.ArealDistortion_FS.32k_fs_LR.shape.gii', ({'property':'areal_distortion_fs',
                                                                  'hemi':'lh_LR32k_MSMSulc'},
                                                                 {'property':'areal_distortion_fs',
                                                                  'hemi':'lh_LR32k_MSMAll'}),
            '{0[id]}.L.ArealDistortion_MSMSulc.32k_fs_LR.shape.gii', {'property':'areal_distortion',
                                                                      'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.MyelinMap.32k_fs_LR.func.gii', {'property':'myelin',
                                                       'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.MyelinMap_BC.32k_fs_LR.func.gii', {'property':'myelin_bc',
                                                          'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.SmoothedMyelinMap.32k_fs_LR.func.gii', {'property':'myelin_smooth',
                                                               'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.SmoothedMyelinMap_BC.32k_fs_LR.func.gii', {'property':'myelin_smooth_bc',
                                                                  'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.curvature.32k_fs_LR.shape.gii', {'property':'curvature',
                                                        'hemi':'lh_LR32k_MSMSulc',
                                                        'filt':lambda c: -c},
            '{0[id]}.L.sulc.32k_fs_LR.shape.gii', {'property':'convexity',
                                                   'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.corrThickness.32k_fs_LR.shape.gii', {'property':'thickness',
                                                            'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.thickness.32k_fs_LR.shape.gii', {'property':'thickness_uncorrected',
                                                        'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.white.32k_fs_LR.surf.gii', {'surface':'white',
                                                   'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.midthickness.32k_fs_LR.surf.gii', {'surface':'midgray',
                                                          'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.pial.32k_fs_LR.surf.gii', {'surface':'pial',
                                                  'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.inflated.32k_fs_LR.surf.gii', {'surface':'inflated',
                                                      'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.very_inflated.32k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                           'hemi':'lh_LR32k_MSMSulc'},
            '{0[id]}.L.white_MSMAll.32k_fs_LR.surf.gii', {'surface':'white',
                                                          'hemi':'lh_LR32k_MSMAll'},
            '{0[id]}.L.midthickness_MSMAll.32k_fs_LR.surf.gii', {'surface':'midgray',
                                                                 'hemi':'lh_LR32k_MSMAll'},
            '{0[id]}.L.pial_MSMAll.32k_fs_LR.surf.gii', {'surface':'pial',
                                                         'hemi':'lh_LR32k_MSMAll'},
            '{0[id]}.L.inflated_MSMAll.32k_fs_LR.surf.gii', {'surface':'inflated',
                                                             'hemi':'lh_LR32k_MSMAll'},
            '{0[id]}.L.very_inflated_MSMAll.32k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                                  'hemi':'lh_LR32k_MSMAll'},
            #'{0[id]}.L.flat.32k_fs_LR.surf.gii', (
            #    {'surface',
            #     'flat',
            #     'hemi':'lh_LR32k_MSMSulc'},
            #    {'surface',
            #     'flat',
            #     'hemi':'lh_LR32k_MSMAll'}),
            '{0[id]}.L.sphere.32k_fs_LR.surf.gii', ({'registration':'fs_LR',
                                                     'hemi':'lh_LR32k_MSMSulc'},
                                                    {'registration':'fs_LR',
                                                     'hemi':'lh_LR32k_MSMAll'}),
            '{0[id]}.R.BA.32k_fs_LR.label.gii', ({'property':'brodmann_area',
                                                  'hemi':'rh_LR32k_MSMSulc'},
                                                 {'property':'brodmann_area',
                                                  'hemi':'rh_LR32k_MSMAll'}),
            '{0[id]}.R.aparc.32k_fs_LR.label.gii', ({'property':'parcellation_2005',
                                                     'hemi':'rh_LR32k_MSMSulc'},
                                                    {'property':'parcellation_2005',
                                                     'hemi':'rh_LR32k_MSMAll'}),
            '{0[id]}.R.aparc.a2009s.32k_fs_LR.label.gii', ({'property':'parcellation',
                                                            'hemi':'rh_LR32k_MSMSulc'},
                                                           {'property':'parcellation',
                                                            'hemi':'rh_LR32k_MSMAll'}),
            '{0[id]}.R.atlasroi.32k_fs_LR.shape.gii', ({'property':'atlas',
                                                        'hemi':'rh_LR32k_MSMSulc',
                                                        'load':_load_fsLR_atlasroi,
                                                        'filt':lambda x:x[1].astype(np.bool)},
                                                       {'property':'atlas',
                                                        'hemi':'rh_LR32k_MSMAll',
                                                        'load':_load_fsLR_atlasroi,
                                                        'filt':lambda x:x[1].astype(np.bool)}),
            '{0[id]}.R.ArealDistortion_FS.32k_fs_LR.shape.gii', ({'property':'areal_distortion_fs',
                                                                  'hemi':'rh_LR32k_MSMSulc'},
                                                                 {'property':'areal_distortion_fs',
                                                                  'hemi':'rh_LR32k_MSMAll'}),
            '{0[id]}.R.ArealDistortion_MSMSulc.32k_fs_LR.shape.gii', {'property':'areal_distortion',
                                                                      'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.MyelinMap.32k_fs_LR.func.gii', {'property':'myelin',
                                                       'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.MyelinMap_BC.32k_fs_LR.func.gii', {'property':'myelin_bc',
                                                          'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.SmoothedMyelinMap.32k_fs_LR.func.gii', {'property':'myelin_smooth',
                                                               'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.SmoothedMyelinMap_BC.32k_fs_LR.func.gii', {'property':'myelin_smooth_bc',
                                                                  'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.curvature.32k_fs_LR.shape.gii', {'property':'curvature',
                                                        'hemi':'rh_LR32k_MSMSulc',
                                                        'filt':lambda c: -c},
            '{0[id]}.R.sulc.32k_fs_LR.shape.gii', {'property':'convexity',
                                                   'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.corrThickness.32k_fs_LR.shape.gii', {'property':'thickness',
                                                            'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.thickness.32k_fs_LR.shape.gii', {'property':'thickness_uncorrected',
                                                        'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.white.32k_fs_LR.surf.gii', {'surface':'white',
                                                   'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.midthickness.32k_fs_LR.surf.gii', {'surface':'midgray',
                                                          'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.pial.32k_fs_LR.surf.gii', {'surface':'pial',
                                                  'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.inflated.32k_fs_LR.surf.gii', {'surface':'inflated',
                                                      'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.very_inflated.32k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                           'hemi':'rh_LR32k_MSMSulc'},
            '{0[id]}.R.white_MSMAll.32k_fs_LR.surf.gii', {'surface':'white',
                                                          'hemi':'rh_LR32k_MSMAll'},
            '{0[id]}.R.midthickness_MSMAll.32k_fs_LR.surf.gii', {'surface':'midgray',
                                                                 'hemi':'rh_LR32k_MSMAll'},
            '{0[id]}.R.pial_MSMAll.32k_fs_LR.surf.gii', {'surface':'pial',
                                                         'hemi':'rh_LR32k_MSMAll'},
            '{0[id]}.R.inflated_MSMAll.32k_fs_LR.surf.gii', {'surface':'inflated',
                                                             'hemi':'rh_LR32k_MSMAll'},
            '{0[id]}.R.very_inflated_MSMAll.32k_fs_LR.surf.gii', {'surface':'very_inflated',
                                                                  'hemi':'rh_LR32k_MSMAll'},
            #'{0[id]}.R.flat.32k_fs_LR.surf.gii', (
            #    {'surface',
            #     'flat',
            #     'hemi':'rh_LR32k_MSMSulc',
            #     'load':_load_atlas_sphere},
            #    {'surface',
            #     'flat',
            #     'hemi':'rh_LR32k_MSMAll',
            #     'load':_load_atlas_sphere}),
            '{0[id]}.R.sphere.32k_fs_LR.surf.gii', ({'registration':'fs_LR',
                                                     'hemi':'rh_LR32k_MSMSulc',
                                                     'load':_load_atlas_sphere},
                                                    {'registration':'fs_LR',
                                                     'hemi':'rh_LR32k_MSMAll',
                                                     'load':_load_atlas_sphere}),
            '{0[id]}.ArealDistortion_MSMAll.32k_fs_LR.dscalar.nii', {
                'property':'areal_distortion',
                'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')},
            '{0[id]}.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii', {
                'property':'myelin_bc',
                'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')},
            '{0[id]}.SmoothedMyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii', {
                'property':'myelin_smooth_bc',
                'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')},
            '{0[id]}.curvature_MSMAll.32k_fs_LR.dscalar.nii', {
                'property':'curvature',
                'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll'),
                'filt':lambda c: -c},
            '{0[id]}.sulc_MSMAll.32k_fs_LR.dscalar.nii', {
                'property':'convexity',
                'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')},
            '{0[id]}.corrThickness_MSMAll.32k_fs_LR.dscalar.nii', {
                'property':'thickness',
                'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')},
            '{0[id]}.thickness_MSMAll.32k_fs_LR.dscalar.nii', {
                'property':'thickness_uncorrected',
                'hemi':('lh_LR32k_MSMAll', 'rh_LR32k_MSMAll')}]]]
