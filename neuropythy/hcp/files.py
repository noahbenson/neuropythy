####################################################################################################
# neuropythy/hcp/files.py
# Stored data regarding the organization of the files in HCP subjects.
# by Noah C. Benson

import os, six, logging, copy, pimms, pyrsistent as pyr, nibabel as nib, numpy as np
from .. import io as nyio
from ..util import (config, is_image, to_credentials, file_map, to_pseudo_path, is_pseudo_path,
                    curry, library_path)
from ..freesurfer import (freesurfer_subject_filemap_instructions,
                          freesurfer_subject_data_hierarchy)

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
    pp = [os.path.expanduser(os.path.expandvars(p)) for p in paths]
    return [p for p in pp if os.path.isdir(p)]
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
def is_hcp_subject_path(path):
    '''
    is_hcp_subject_path(path) yields True if the given path appears to be an HCP subject directory;
      specifically, the path must be a directory that contains the subdirectories 'MNINonLinear' and
      'T1w'.
    '''
    needed = ['T1w', 'MNINonLinear']
    if   is_pseudo_path(path): return all(path.find(k) is not None for k in needed)
    elif os.path.isdir(path):  return all(os.path.isdir(os.path.join(path, k)) for k in needed)
    else:                      return False
def find_subject_path(sid, check_path=True):
    '''
    find_subject_path(sub) yields the full path of a HCP subject with the name given by the string
      sub, if such a subject can be found in the HCP search paths. See also add_subject_path.

    If no subject is found, then None is returned.
    '''
    # if it's a full/relative path already, use it:
    sub = str(sid)
    if ((not check_path or is_hcp_subject_path(sub)) and
        (check_path is None or os.path.isdir(sub))):
        return sub
    # check the subject directories:
    sdirs = config['hcp_subject_paths']
    return next((os.path.abspath(p) for sd in sdirs
                 for p in [os.path.join(sd, sub)]
                 if ((not check_path or is_hcp_subject_path(p)) and
                     (check_path is None or os.path.isdir(p)))),
                None)

if config['hcp_subject_paths'] is None:
    # if a path wasn't found, there are a couple environment variables we want to look at...
    if 'HCPSUBJS_DIR' in os.environ: add_subject_path(os.environ['HCPSUBJS_DIR'])
    for varname in ['HCP_ROOT', 'HCP_DIR']:
        if varname in os.environ:
            dirname = os.path.join(os.environ[varname], 'subjects')
            if os.path.isdir(dirname):
                add_subject_path(dirname)

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

def cifti_label_data(obj):
    '''
    cifti_label_data(obj) yields an ordered dictionary of label names mapped to label data in the
      given cifti object obj; if obj is not an interpretable label structure, None is returned.
    
    If the argument obj is a CIFTI image object then a list of values, one per axis in the CIFTI
    object, is returned where None indicates that the axis was not a label axis. If obj is an axis,
    then either None or the datatable alone is returned.

    Note that the label data for a particular label name may be empty/None; this indicates that the
    axis is for scalar data rather than integer label data.
    '''
    import nibabel.cifti2 as cii
    from neuropythy import label_index
    if obj is None: return None
    elif isinstance(obj, cii.Cifti2Image):
        return cifti_label_data(obj.header)
    elif isinstance(obj, cii.Cifti2Header):
        return [cifti_label_data(obj.get_index_map(k)) for k in obj.mapped_indices]
    elif not isinstance(obj, cii.Cifti2MatrixIndicesMap):
        return None
    # okay, it's a matrix-indices-map object... see if it has a named map
    nmaps = list(obj.named_maps)
    if len(nmaps) == 0: return None
    # it has a named map and label-table; we just have to interpret it
    from collections import OrderedDict
    res = OrderedDict()
    for nmap in nmaps:
        lbls = {}
        for k in ([] if nmap.label_table is None else nmap.label_table):
            lbl = nmap.label_table[k]
            lbls[lbl.key] = (lbl.label, lbl.rgba)
        if nmap.metadata is None or len(nmap.metadata) == 0: md = pyr.m()
        else: md = pimms.persist(nmap.metadata.data)
        if len(lbls) > 0: lbls = label_index(lbls, meta_data=md)
        else: lbls = None
        res[nmap.map_name] = lbls
    return res
def cifti_axis_spec(ax):
    '''
    cifti_axis_spec(obj) yields a persistent mapping of data about the given cifti2 axis object obj.

    The argument obj may alternately be a cifti image object or an image header, in which case a
    list of axis specs are given, one per axis.

    The map that is returned by this function contains the keys 'axis_type' ('CIFTI_INDEX_TYPE_' +
    one of 'SURFACE', 'VOLUME', 'PARCELS', 'BRAIN_MODELS', 'SERIES', 'SCALARS', or 'LABELS') and
    various data relevant to each specific type.
    '''
    import nibabel.cifti2 as cii
    from neuropythy import label_index
    if ax is None: return None
    elif isinstance(ax, cii.Cifti2Image):
        return cifti_axis_spec(ax.header)
    elif isinstance(ax, cii.Cifti2Header):
        return [cifti_axis_spec(ax.get_index_map(k)) for k in ax.mapped_indices]
    elif not isinstance(ax, cii.Cifti2MatrixIndicesMap):
        return None
    d = {}
    # start by checking for surface and volume data; they are used in parcels and brain models
    sdat = {}
    for srf in ax.surfaces:
        sdat[srf.brain_structure] = {'vertex_count': srf.surface_number_of_vertices}
    if len(sdat) > 0: d['surfaces'] = sdat
    if ax.volume is not None:
        d['volume_shape'] = ax.volume.volume_dimensions
        mtx = ax.volume.transformation_matrix_voxel_indices_ijk_to_xyz.matrix
        # 3 + exp because -3 is mm and we prefer these data to be in mm
        exp = 3 + ax.volume.transformation_matrix_voxel_indices_ijk_to_xyz.meter_exponent
        if exp != 0: mtx = mtx * 10**exp
        d['volume_affine'] = mtx
    # Okay, note the axis type:
    axtype = ax.indices_map_to_data_type
    d['axis_type'] = axtype
    # Check for series data:
    (i0,n,step) = (ax.series_start, ax.number_of_series_points, ax.series_step)
    if n is not None:
        idcs = np.arange(i0, i0 + n*step, step)
        #imsh.append(len(idcs))
        if ax.series_unit is not None:
            try: idcs = pimms.quant(idcs, ax.series_unit.lower())
            except Exception: pass
        d['series'] = idcs
        d['size'] = len(idcs)
    # Check for brain-model data:
    bmdls = list(ax.brain_models)
    if len(bmdls) > 0:
        maxii = 0
        idxdat = {}
        for mdl in sorted(bmdls, key=lambda m:m.index_offset):
            sl = slice(mdl.index_offset, mdl.index_offset + mdl.index_count)
            if maxii < sl.stop: maxii = sl.stop
            if mdl.model_type == 'CIFTI_MODEL_TYPE_SURFACE':
                idxdat[mdl.brain_structure] = {'vertex_count': mdl.surface_number_of_vertices,
                                               'vertex_indices': np.array(mdl.vertex_indices),
                                               'cifti_indices': sl}
            elif mdl.model_type == 'CIFTI_MODEL_TYPE_VOXELS':
                idxdat[mdl.brain_structure] = {'voxel_indices': np.array(mdl.voxel_indices_ijk),
                                               'cifti_indices': sl}
            else: raise ValueError('Unrecognized CIFTI model type: %s' % mdl.model_type)
        d['structures'] = idxdat
        d['size'] = maxii
    # Check for label and scalar data:
    lbls = cifti_label_data(ax)
    if lbls is not None:
        d['size'] = len(lbls)
        d['names'] = list(lbls.keys())
        if not all(v is None for v in six.itervalues(lbls)): d['labels'] = list(lbls.values())
    # Check for parcel data:
    parcs = {}
    maxii = 0
    for p in ax.parcels:
        pp = {}
        if p.voxel_indices_ijk is not None:
            pp['voxels'] = np.array(p.voxel_indices_ijk)
            maxii += len(pp['voxels'])
        vv = {}
        for vrt in p.vertices:
            vv[vrt.brain_structure] = np.array(vrt.vertices)
            maxii += len(vv[vrt.brain_structure])
        if len(vv) > 0: pp['vertices'] = vv
        parcs[p.name] = pp
    if len(parcs) > 0:
        d['parcels'] = parcs
        d['size'] = maxii
    # That's all we check for currently
    return d

def cifti_split(cii, label=('lh', 'rh', 'rest'), subject=None, hemi=None, null=np.nan):
    '''
    cifti_split(cii, label) yields the rows or columns of the given cifti file that correspond to
      the given label (see below).
    cifti_split(cii) is equivalent to cifti_split(cii, ('lh', 'rh', 'rest')).

    The label argument may be any of the following:
      * a valid CIFTI label name such as 'CIFTI_STRUCTURE_CEREBELLUM' or
        'CIFTI_STRUCTURE_CORTEX_LEFT';
      * an abbreviated name such as 'cerebellum' for 'CIFTI_STRUCTURE_CEREBELLUM'.
      * the abbreviations 'lh' and 'rh' which stand for 'CIFTI_STRUCTURE_CORTEX_LEFT' and 
        'CIFTI_STRUCTURE_CORTEX_RIGHT';
      * the special keyword 'rest', which represents all the rows/columns not collected by any other
        instruction ('rest', by itself, results in the whole matrix being returned); or
      * A tuple of the above, indicating that each of the items listed should be returned
        sequentially in a tuple.

    The following optional arguments may be given:
      * subject (default: None) may specify the subject
      * hemi (default: None) can specify the hemisphere object that 
    '''
    if cii is None: raise ValueError('cifti_split() called with argument None')
    dat = np.asanyarray(cii.dataobj if is_image(cii) else cii)
    n = dat.shape[-1]
    atlas = cifti_split._size_data.get(n, None)
    if atlas is None: raise ValueError('cannot split cifti with size %d' % n)
    if atlas not in cifti_split._atlas_cache:
        patt = os.path.join('data', 'fs_LR', '%s.atlasroi.%dk_fs_LR.shape.gii')
        lgii = nib.load(os.path.join(library_path(), patt % ('lh', atlas)))
        rgii = nib.load(os.path.join(library_path(), patt % ('rh', atlas)))
        cifti_split._atlas_cache[atlas] = tuple([pimms.imm_array(gii.darrays[0].data.astype('bool'))
                                                 for gii in (lgii, rgii)])
    (lroi,rroi) = cifti_split._atlas_cache[atlas]
    (ln,lN) = (np.sum(lroi), len(lroi))
    (rn,rN) = (np.sum(rroi), len(rroi))
    (ldat,rdat,sdat) = [np.full(dat.shape[:-1] + (k,), null) for k in [lN, rN, n - ln - rn]]
    ldat[..., lroi] = dat[..., :ln]
    rdat[..., rroi] = dat[..., ln:(ln+rn)]
    sdat[...] = dat[..., (ln+rn):]
    if ln + rn >= n: sdat = None
    return (ldat, rdat, sdat)
cifti_split._size_data = {
    # two sizes for each atlas: one for when the cifti file includes subcortical voxels and one for
    # when it includes only the surface vertices
    91282:  32,
    59412:  32,
    170494: 59,
    108441: 59,
    # not sure what the bigger size is for this...
    #?????: 164,
    298261: 164}
cifti_split._atlas_cache = {}
def cifti_extract(cii, h, null=np.nan):
    '''
    cifti_extract(cii, h) yields the portion of the cifti vector or matrix that is associated
      with the given hemisphere h of the given subject. If h is None, then yields any subcortical
      voxels (or None if none are included).
    '''
    (l,r,s) = cifti_split(cii, null=null)
    if h is None:            return s
    elif h.startswith('lh'): return l
    else:                    return r

####################################################################################################
# The filemap instructions and related helper data/functions

def subject_file_map(path, name=None):
    '''
    subject_file_map(path) yields a filemap object for the given HCP subject path.
    '''
    if name is None:
        if is_pseudo_path(path):
            pmod = path._path_data['pathmod']
            name = pmod.split(path.source_path)[-1]
        else: name = os.path.split(path)[-1]
    return file_map(path, hcp_filemap_instructions,
                    data_hierarchy=hcp_filemap_data_hierarchy,
                    path_parameters={'id':name})

# We adapt a section of the FreeSurfer spec to the HCP because the HCP directories include a
# mostly-complete FreeSurfer directory
def _hcp_adapt_fsinstructions(instr):
    instr = copy.copy(instr)
    if pimms.is_list(instr):
        res = []
        for (fnm,desc) in zip(instr[0::2], instr[1::2]):
            desc = _hcp_adapt_fsinstructions(desc)
            desc = (desc,) if pimms.is_map(desc) else desc
            if len(desc) == 0: continue
            res.append(fnm)
            res.append(desc[0] if len(desc) == 1 else desc)
        return res
    elif pimms.is_map(instr):
        if   'image'     in instr: instr['image'] = 'freesurfer_' + instr['image']
        elif 'raw_image' in instr: instr['raw_image'] = 'freesurfer_' + instr['raw_image']
        elif 'surface'   in instr: instr['surface'] = 'freesurfer_' + instr['surface']
        if   'hemi'      in instr:
            h = instr['hemi']
            if h.endswith('x'): return ()
            instr['hemi'] = h + '_native_MSMAll'
            instr = (instr, copy.copy(instr), copy.copy(instr))
            instr[1]['hemi'] = h + '_native_MSMSulc'
            instr[2]['hemi'] = h + '_native_FS'
        # others: label/annot/property -- these can stay the same
        return instr
    elif pimms.is_tuple(instr):
        tup = [_hcp_adapt_fsinstructions(u) for u in instr]
        tup = tuple([v for u in tup for v in (u if pimms.is_tuple(u) else [u])])
        return tup
    else: raise ValueError('Unrecognized instruction type: %s [%s]' % (type(instr), instr))
hcp_adapted_freesurfer_subject_filemap_instructions = []
# setup the freesurfer subject 
for (dname,contents) in zip(freesurfer_subject_filemap_instructions[0::2],
                            freesurfer_subject_filemap_instructions[1::2]):
    if dname == 'xhemi': continue
    hcp_adapted_freesurfer_subject_filemap_instructions.append(dname)
    hcp_adapted_freesurfer_subject_filemap_instructions.append(_hcp_adapt_fsinstructions(contents))
def _filt_coords(msh): return msh.coordinates
def _filt_faces(msh): return msh.tess.faces

def _surf(name, hemibase, suffix=['_MSMAll', '_MSMSulc', '_FS'], tess=False, format='gifti'):
    res = tuple([{'surface':name, 'hemi':(hemibase+suf), 'filt':_filt_coords, 'format':format}
                 for suf in (suffix if pimms.is_vector(suffix, str) else [suffix])])
    if tess:
        rst = tuple([{'tess':name, 'hemi':(hemibase+suf), 'filt':_filt_faces, 'format':format}
                     for suf in (suffix if pimms.is_vector(suffix, str) else [suffix])])
        res = res + rst
    return res
def _surf_fs(name, hemibase, tess=False, format='gifti'):
    return _surf(name, hemibase, suffix=['_FS'], tess=tess, format=format)
def _surf_msmsulc(name, hemibase, tess=False, format='gifti'):
    return _surf(name, hemibase, suffix=['_MSMSulc'], tess=tess, format=format)
def _surf_fssulc(name, hemibase, tess=False, format='gifti'):
    return _surf(name, hemibase, suffix=['_MSMSulc', '_FS'], tess=tess, format=format)
def _surf_msmall(name, hemibase, tess=False, format='gifti'):
    return _surf(name, hemibase, suffix=['_MSMAll'], tess=tess, format=format)
def _reg(name, hemibase, suffix=['_MSMAll', '_MSMSulc', '_FS'], format='gifti'):
    return tuple([{'registration':name, 'hemi':(hemibase+suf), 'filt':_filt_coords, 'format':format}
                  for suf in (suffix if pimms.is_vector(suffix, str) else [suffix])])
def _reg_fs(name, hemibase, format='gifti'):
    return _reg(name, hemibase, suffix=['_FS'], format=format)
def _reg_msmsulc(name, hemibase, format='gifti'):
    return _reg(name, hemibase, suffix=['_MSMSulc'], format=format)
def _reg_fssulc(name, hemibase, format='gifti'):
    return _reg(name, hemibase, suffix=['_FS', '_MSMSulc'], format=format)
def _reg_msmall(name, hemibase, format='gifti'):
    return _reg(name, hemibase, suffix=['_MSMAll'], format=format)
def _nofilt(x): return x
def _tobool(x): return np.asarray(x).astype('bool')
def _prop(name, hemibase, suffix=['_MSMAll','_MSMSulc', '_FS'],
          format='gifti', key='property', filt=None):
    tup = tuple([{key:name, 'hemi':(hemibase+suf), 'format':format}
                 for suf in (suffix if pimms.is_vector(suffix, str) else [suffix])])
    if format == 'gifti':
        for u in tup: u['filt'] = gifti_to_array
    elif format == 'cifti':
        for u in tup: u['filt'] = curry(lambda h,cii: cifti_extract(cii, h), u['hemi'])
    if key == 'label':
        for u in tup:
            u['filt'] = curry(lambda f,x: _tobool(f(x)), u['filt']) if 'filt' in u else _tobool
    if filt is not None and filt is not False:
        for u in tup:
            u['filt'] = curry(lambda ff,f,x: ff(f(x)), filt, u['filt']) if 'filt' in u else filt
    return tup
def _prop_fs(name, hemibase, format='gifti', key='property', filt=None):
    return _prop(name, hemibase, suffix=['_FS'], format=format, key=key)
def _prop_msmsulc(name, hemibase, format='gifti', key='property', filt=None):
    return _prop(name, hemibase, suffix=['_MSMSulc'], format=format, key=key)
def _prop_fssulc(name, hemibase, format='gifti', key='property', filt=None):
    return _prop(name, hemibase, suffix=['_FS', '_MSMSulc'], format=format, key=key)
def _prop_msmall(name, hemibase, format='gifti', key='property', filt=None):
    return _prop(name, hemibase, suffix=['_MSMAll'], format=format, key=key)
def _prop_cifti(name, hemires, filt=None):
    return (_prop_msmall(name, 'lh_LR%dk' % hemires, format='cifti', filt=filt) + 
            _prop_msmall(name, 'rh_LR%dk' % hemires, format='cifti', filt=filt))
def _load_imm_img(flnm, u):
    import neuropythy.mri as mri, neuropythy.io as nyio
    im = nyio.load(flnm)
    try: im.dataobj.flags['WRITEABLE'] = False
    except Exception:
        d = np.asanyarray(im.dataobj)
        d.flags['WRITEABLE'] = False
        im = type(im)(d, im.affine, im.header)
    return im

hcp_filemap_data_hierarchy = [['image'], ['raw_image'], ['flatmap'],
                              ['hemi', 'surface'], ['hemi', 'tess'],
                              ['hemi', 'registration'],
                              ['hemi', 'property'],
                              ['hemi', 'label'], ['hemi', 'alt_label'],
                              ['hemi', 'weight'], ['hemi', 'alt_weight'],
                              ['hemi', 'annot'], ['hemi', 'alt_annot']]
hcp_filemap_instructions = [
    'T1w', [
        'BiasField_acpc_dc.nii.gz',         {'image':'bias', 'load':_load_imm_img},
        'T1wDividedByT2w.nii.gz',           {'image':'T1_to_T2_ratio_all', 'load':_load_imm_img},
        'T1wDividedByT2w_ribbon.nii.gz',    {'image':'T1_to_T2_ratio', 'load':_load_imm_img},
        'T1w_acpc_dc_restore.nii.gz',       {'image':'T1', 'load':_load_imm_img},
        'T1w_acpc_dc.nii.gz',               {'image':'T1_unrestored', 'load':_load_imm_img},
        'T1w_acpc_dc_restore_brain.nii.gz', {'image':'brain', 'load':_load_imm_img},
        'T2w_acpc_dc_restore.nii.gz',       {'image':'T2', 'load':_load_imm_img},
        'T2w_acpc_dc.nii.gz',               {'image':'T2_unrestored', 'load':_load_imm_img},
        'T2w_acpc_dc_restore_brain.nii.gz', {'image':'T2_brain', 'load':_load_imm_img},
        'aparc+aseg.nii.gz',                {'image':'Desikan06_parcellation', 'load':_load_imm_img},
        'aparc.a2009s+aseg.nii.gz',         ({'image':'parcellation', 'load':_load_imm_img},
                                             {'image':'Destrieux09_parcellation', 'load':_load_imm_img}),
        'brainmask_fs.nii.gz',              {'image':'masked_brain', 'load':_load_imm_img},
        'ribbon.nii.gz',                    {'image':'ribbon', 'load':_load_imm_img},
        'wmparc.nii.gz',                    {'image':'white_parcellation', 'load':_load_imm_img},
        '{id}', hcp_adapted_freesurfer_subject_filemap_instructions,
        'Native', [
            '{id}.L.white.native.surf.gii',         _surf('white', 'lh_native', tess=True),
            '{id}.L.midthickness.native.surf.gii',  _surf('midgray', 'lh_native'),
            '{id}.L.pial.native.surf.gii',          _surf('pial', 'lh_native'),
            '{id}.L.inflated.native.surf.gii',      _surf('inflated', 'lh_native'),
            '{id}.L.very_inflated.native.surf.gii', _surf('very_inflated', 'lh_native'),
            '{id}.R.white.native.surf.gii',         _surf('white', 'rh_native', tess=True),
            '{id}.R.midthickness.native.surf.gii',  _surf('midgray', 'rh_native'),
            '{id}.R.pial.native.surf.gii',          _surf('pial', 'rh_native'),
            '{id}.R.inflated.native.surf.gii',      _surf('inflated', 'rh_native'),
            '{id}.R.very_inflated.native.surf.gii', _surf('very_inflated', 'rh_native')],
        'fsaverage_LR32k', [
            '{id}.L.inflated.32k_fs_LR.surf.gii',      _surf_fssulc('inflated',      'lh_nat32k'),
            '{id}.L.midthickness.32k_fs_LR.surf.gii',  _surf_fssulc('migray',        'lh_nat32k'),
            '{id}.L.pial.32k_fs_LR.surf.gii',          _surf_fssulc('pial',          'lh_nat32k'),
            '{id}.L.very_inflated.32k_fs_LR.surf.gii', _surf_fssulc('very_inflated', 'lh_nat32k'),
            '{id}.L.white.32k_fs_LR.surf.gii',         _surf_fssulc('white',         'lh_nat32k',
                                                                    tess=True),
            '{id}.R.inflated.32k_fs_LR.surf.gii',      _surf_fssulc('inflated',      'rh_nat32k'),
            '{id}.R.midthickness.32k_fs_LR.surf.gii',  _surf_fssulc('migray',        'rh_nat32k'),
            '{id}.R.pial.32k_fs_LR.surf.gii',          _surf_fssulc('pial',          'rh_nat32k'),
            '{id}.R.very_inflated.32k_fs_LR.surf.gii', _surf_fssulc('very_inflated', 'rh_nat32k'),
            '{id}.R.white.32k_fs_LR.surf.gii',         _surf_fssulc('white',         'rh_nat32k',
                                                                     tess=True),
            
            '{id}.L.inflated_MSMAll.32k_fs_LR.surf.gii',      _surf_msmall('inflated', 'lh_nat32k'),
            '{id}.L.midthickness_MSMAll.32k_fs_LR.surf.gii',  _surf_msmall('midgray', 'lh_nat32k'),
            '{id}.L.pial_MSMAll.32k_fs_LR.surf.gii',          _surf_msmall('pial', 'lh_nat32k'),
            '{id}.L.very_inflated_MSMAll.32k_fs_LR.surf.gii', _surf_msmall('very_inflated',
                                                                           'lh_nat32k'),
            '{id}.L.white_MSMAll.32k_fs_LR.surf.gii',         _surf_msmall('white', 'lh_nat32k',
                                                                           tess=True),
            '{id}.R.inflated_MSMAll.32k_fs_LR.surf.gii',      _surf_msmall('inflated', 'rh_nat32k'),
            '{id}.R.midthickness_MSMAll.32k_fs_LR.surf.gii',  _surf_msmall('midgray', 'rh_nat32k'),
            '{id}.R.pial_MSMAll.32k_fs_LR.surf.gii',          _surf_msmall('pial', 'rh_nat32k'),
            '{id}.R.very_inflated_MSMAll.32k_fs_LR.surf.gii', _surf_msmall('very_inflated',
                                                                           'rh_nat32k'),
            '{id}.R.white_MSMAll.32k_fs_LR.surf.gii',         _surf_msmall('white', 'rh_nat32k',
                                                                           tess=True)],
        'fsaverage_LR59k', [
            '{id}.L.inflated.59k_fs_LR.surf.gii',      _surf_fssulc('inflated',      'lh_nat59k'),
            '{id}.L.midthickness.59k_fs_LR.surf.gii',  _surf_fssulc('midgray',       'lh_nat59k'),
            '{id}.L.pial.59k_fs_LR.surf.gii',          _surf_fssulc('pial',          'lh_nat59k'),
            '{id}.L.very_inflated.59k_fs_LR.surf.gii', _surf_fssulc('very_inflated', 'lh_nat59k'),
            '{id}.L.white.59k_fs_LR.surf.gii',         _surf_fssulc('white',         'lh_nat59k',
                                                                    tess=True),
            '{id}.R.inflated.59k_fs_LR.surf.gii',      _surf_fssulc('inflated',      'rh_nat59k'),
            '{id}.R.midthickness.59k_fs_LR.surf.gii',  _surf_fssulc('midgray',       'rh_nat59k'),
            '{id}.R.pial.59k_fs_LR.surf.gii',          _surf_fssulc('pial',          'rh_nat59k'),
            '{id}.R.very_inflated.59k_fs_LR.surf.gii', _surf_fssulc('very_inflated', 'rh_nat59k'),
            '{id}.R.white.59k_fs_LR.surf.gii',         _surf_fssulc('white',         'rh_nat59k',
                                                                    tess=True),
            
            '{id}.L.inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii',      _surf_msmall('inflated',
                                                                                 'lh_nat59k'),
            '{id}.L.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii',  _surf_msmall('midgray',
                                                                                 'lh_nat59k'),
            '{id}.L.pial_1.6mm_MSMAll.59k_fs_LR.surf.gii',          _surf_msmall('pial',
                                                                                 'lh_nat59k'),
            '{id}.L.very_inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii', _surf_msmall('very_inflated',
                                                                                 'lh_nat59k'),
            '{id}.L.white_1.6mm_MSMAll.59k_fs_LR.surf.gii',         _surf_msmall('white',
                                                                                 'lh_nat59k',
                                                                                 tess=True),
            '{id}.R.inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii',      _surf_msmall('inflated',
                                                                                 'rh_nat59k'),
            '{id}.R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii',  _surf_msmall('midgray',
                                                                                 'rh_nat59k'),
            '{id}.R.pial_1.6mm_MSMAll.59k_fs_LR.surf.gii',          _surf_msmall('pial',
                                                                                 'rh_nat59k'),
            '{id}.R.very_inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii', _surf_msmall('very_inflated',
                                                                                 'rh_nat59k'),
            '{id}.R.white_1.6mm_MSMAll.59k_fs_LR.surf.gii',         _surf_msmall('white',
                                                                                 'rh_nat59k',
                                                                                 tess=True)]],
    'MNINonLinear', [
        'BiasField.nii.gz',         {'image':'bias_warped', 'load':_load_imm_img},
        'T1w_restore.nii.gz',       {'image':'T1_warped', 'load':_load_imm_img},
        'T1w.nii.gz',               {'image':'T1_warped_unrestored', 'load':_load_imm_img},
        'T1w_restore_brain.nii.gz', {'image':'brain_warped', 'load':_load_imm_img},
        'T2w_restore.nii.gz',       {'image':'T2_warped', 'load':_load_imm_img},
        'T2w.nii.gz',               {'image':'T2_warped_unrestored', 'load':_load_imm_img},
        'T2w_restore_brain.nii.gz', {'image':'T2_brain_warped', 'load':_load_imm_img},
        'aparc+aseg.nii.gz',        {'image':'Desikan06_parcellation_warped', 'load':_load_imm_img},
        'aparc.a2009s+aseg.nii.gz', ({'image':'parcellation_warped', 'load':_load_imm_img},
                                     {'image':'Destrieux09_parcellation_warped', 'load':_load_imm_img}),
        'brainmask_fs.nii.gz',      {'image':'masked_brain_warped', 'load':_load_imm_img},
        'ribbon.nii.gz',            {'image':'ribbon_warped', 'load':_load_imm_img},
        'wmparc.nii.gz',            {'image':'white_parcellation_warped', 'load':_load_imm_img},
        '{id}.L.ArealDistortion_FS.164k_fs_LR.shape.gii',      _prop_fs('areal_distortion',
                                                                        'lh_LR164k'),
        '{id}.L.ArealDistortion_MSMSulc.164k_fs_LR.shape.gii', _prop_msmsulc('areal_distortion',
                                                                             'lh_LR164k'),
        '{id}.L.ArealDistortion_MSMAll.164k_fs_LR.shape.gii',  _prop_msmall('areal_distortion',
                                                                            'lh_LR164k'),
        '{id}.L.MyelinMap.164k_fs_LR.func.gii',                _prop_fssulc('myelin_uncorrected',
                                                                            'lh_LR164k'),
        '{id}.L.MyelinMap_BC.164k_fs_LR.func.gii',             _prop_fssulc('myelin',
                                                                            'lh_LR164k'),
        '{id}.L.SmoothedMyelinMap.164k_fs_LR.func.gii',  _prop_fssulc('myelin_smooth_uncorrected',
                                                                      'lh_LR164k'),
        '{id}.L.SmoothedMyelinMap_BC.164k_fs_LR.func.gii',     _prop_fssulc('myelin_smooth',
                                                                            'lh_LR164k'),
        '{id}.L.RefMyelinMap.164k_fs_LR.func.gii',             _prop_fssulc('myelin_ref',
                                                                            'lh_LR164k'),
        '{id}.L.BA.164k_fs_LR.label.gii',                      _prop_msmall('brodmann_area',
                                                                            'lh_LR164k'),
        '{id}.L.aparc.164k_fs_LR.label.gii',         _prop('Desikan06_parcellation', 'lh_LR164k'),
        '{id}.L.aparc.a2009s.164k_fs_LR.label.gii',  (_prop('parcellation', 'lh_LH164k') +
                                                      _prop('Destrieux09_parcellation',
                                                            'lh_LH164k')),
        '{id}.L.atlasroi.164k_fs_LR.shape.gii',      _prop('atlas', 'lh_LR164k', key='label'),
        '{id}.L.curvature.164k_fs_LR.shape.gii',     _prop_fssulc('curvature', 'lh_LR164k',
                                                                  filt=lambda c:-c),
        '{id}.L.sulc.164k_fs_LR.shape.gii',          _prop_fssulc('convexity', 'lh_LR164k'),
        '{id}.L.corrThickness.164k_fs_LR.shape.gii', _prop_fssulc('thickness', 'lh_LR164k'),
        '{id}.L.thickness.164k_fs_LR.shape.gii',     _prop_fssulc('thickness_uncorrected',
                                                                   'lh_LR164k'),
        '{id}.L.white.164k_fs_LR.surf.gii',         _surf_fssulc('white', 'lh_LR164k', tess=True),
        '{id}.L.midthickness.164k_fs_LR.surf.gii',  _surf_fssulc('midgray', 'lh_LR164k'),
        '{id}.L.pial.164k_fs_LR.surf.gii',          _surf_fssulc('pial', 'lh_LR164k'),
        '{id}.L.inflated.164k_fs_LR.surf.gii',      _surf_fssulc('inflated', 'lh_LR164k'),
        '{id}.L.very_inflated.164k_fs_LR.surf.gii', _surf_fssulc('very_inflated', 'lh_LR164k'),
        '{id}.L.midthickness_MSMAll.164k_fs_LR.surf.gii',  _surf_msmall('midgray', 'lh_LR164k'),
        '{id}.L.pial_MSMAll.164k_fs_LR.surf.gii',          _surf_msmall('pial', 'lh_LR164k'), 
        '{id}.L.inflated_MSMAll.164k_fs_LR.surf.gii',      _surf_msmall('inflated', 'lh_LR164k'),
        '{id}.L.very_inflated_MSMAll.164k_fs_LR.surf.gii', _surf_msmall('very_inflated',
                                                                        'lh_LR164k'),
        '{id}.L.white_MSMAll.164k_fs_LR.surf.gii',          _surf_msmall('white', 'lh_LR164k',
                                                                         tess=True),
        '{id}.L.sphere.164k_fs_LR.surf.gii',               _reg('fs_LR', 'lh_LR164k'),
        '{id}.L.flat.164k_fs_LR.surf.gii', ({'flatmap':'lh_LR164k_FS',      'format':'gifti'},
                                            {'flatmap':'lh_LR164k_MSMSulc', 'format':'gifti'},
                                            {'flatmap':'lh_LR164k_MSMAll',  'format':'gifti'}),
        '{id}.R.ArealDistortion_FS.164k_fs_LR.shape.gii',      _prop_fs('areal_distortion',
                                                                        'rh_LR164k'),
        '{id}.R.ArealDistortion_MSMSulc.164k_fs_LR.shape.gii', _prop_msmsulc('areal_distortion',
                                                                             'rh_LR164k'),
        '{id}.R.ArealDistortion_MSMAll.164k_fs_LR.shape.gii',  _prop_msmall('areal_distortion',
                                                                            'rh_LR164k'),
        '{id}.R.MyelinMap.164k_fs_LR.func.gii',                _prop_fssulc('myelin_uncorrected',
                                                                            'rh_LR164k'),
        '{id}.R.MyelinMap_BC.164k_fs_LR.func.gii',             _prop_fssulc('myelin', 'rh_LR164k'),
        '{id}.R.SmoothedMyelinMap.164k_fs_LR.func.gii',   _prop_fssulc('myelin_smooth_uncorrected',
                                                                       'rh_LR164k'),
        '{id}.R.SmoothedMyelinMap_BC.164k_fs_LR.func.gii',     _prop_fssulc('myelin_smooth',
                                                                            'rh_LR164k'),
        '{id}.R.RefMyelinMap.164k_fs_LR.func.gii',             _prop_fssulc('myelin_ref',
                                                                            'rh_LR164k'),
        '{id}.R.BA.164k_fs_LR.label.gii',                      _prop_msmall('brodmann_area',
                                                                            'rh_LR164k'),
        '{id}.R.aparc.164k_fs_LR.label.gii',         _prop('Desikan06_parcellation',
                                                           'rh_LR164k'),
        '{id}.R.aparc.a2009s.164k_fs_LR.label.gii',  (_prop('parcellation', 'rh_LH164k') +
                                                      _prop('Destrieux09_parcellation',
                                                            'rh_LH164k')),
        '{id}.R.atlasroi.164k_fs_LR.shape.gii',      _prop('atlas', 'rh_LR164k', key='label'),
        '{id}.R.curvature.164k_fs_LR.shape.gii',     _prop_fssulc('curvature', 'rh_LR164k',
                                                                  filt=lambda c:-c),
        '{id}.R.sulc.164k_fs_LR.shape.gii',          _prop_fssulc('convexity', 'rh_LR164k'),
        '{id}.R.corrThickness.164k_fs_LR.shape.gii', _prop_fssulc('thickness', 'rh_LR164k'),
        '{id}.R.thickness.164k_fs_LR.shape.gii',     _prop_fssulc('thickness_uncorrected',
                                                                  'rh_LR164k'),
        '{id}.R.white.164k_fs_LR.surf.gii',         _surf_fssulc('white', 'rh_LR164k', tess=True),
        '{id}.R.midthickness.164k_fs_LR.surf.gii',  _surf_fssulc('midgray', 'rh_LR164k'),
        '{id}.R.pial.164k_fs_LR.surf.gii',          _surf_fssulc('pial', 'rh_LR164k'),
        '{id}.R.inflated.164k_fs_LR.surf.gii',      _surf_fssulc('inflated', 'rh_LR164k'),
        '{id}.R.very_inflated.164k_fs_LR.surf.gii', _surf_fssulc('very_inflated', 'rh_LR164k'),
        '{id}.R.midthickness_MSMAll.164k_fs_LR.surf.gii',  _surf_msmall('midgray', 'rh_LR164k'),
        '{id}.R.pial_MSMAll.164k_fs_LR.surf.gii',          _surf_msmall('pial', 'rh_LR164k'), 
        '{id}.R.inflated_MSMAll.164k_fs_LR.surf.gii',      _surf_msmall('inflated', 'rh_LR164k'),
        '{id}.R.very_inflated_MSMAll.164k_fs_LR.surf.gii', _surf_msmall('very_inflated',
                                                                        'lh_LR164k'),
        '{id}.R.white_MSMAll.164k_fs_LR.surf.gii',          _surf_msmall('white', 'rh_LR164k',
                                                                         tess=True),
        '{id}.R.sphere.164k_fs_LR.surf.gii',               _reg('fs_LR', 'rh_LR164k'),
        '{id}.R.flat.164k_fs_LR.surf.gii', ({'flatmap':'rh_LR164k_FS',      'format':'gifti'},
                                            {'flatmap':'rh_LR164k_MSMSulc', 'format':'gifti'},
                                            {'flatmap':'rh_LR164k_MSMAll',  'format':'gifti'}),
        '{id}.ArealDistortion_MSMAll.164k_fs_LR.dscalar.nii',      _prop_cifti('areal_distortion',
                                                                               164),
        '{id}.MyelinMap_BC_MSMAll.164k_fs_LR.dscalar.nii',         _prop_cifti('myelin', 164),
        '{id}.SmoothedMyelinMap_BC_MSMAll.164k_fs_LR.dscalar.nii', _prop_cifti('myelin_smooth',
                                                                               164),
        '{id}.curvature_MSMAll.164k_fs_LR.dscalar.nii',            _prop_cifti('curvature', 164,
                                                                               filt=lambda c:-c),
        '{id}.sulc.164k_fs_LR.dscalar.nii',                        _prop_cifti('convexity', 164),
        '{id}.corrThickness.164k_fs_LR.dscalar.nii',               _prop_cifti('thickness', 164),
        '{id}.thickness.164k_fs_LR.dscalar.nii', _prop_cifti('thickness_uncorrected', 164),
        'Native', [
            '{id}.L.ArealDistortion_FS.native.shape.gii', _prop_fs('areal_distortion', 'lh_native'),
            '{id}.L.ArealDistortion_MSMSulc.native.shape.gii', _prop_msmsulc('areal_distortion',
                                                                             'lh_native'),
            '{id}.L.ArealDistortion_MSMAll.native.shape.gii',  _prop_msmall('areal_distortion',
                                                                            'lh_native'),
            '{id}.L.MyelinMap.native.func.gii', _prop_fssulc('myelin_uncorrected', 'lh_native'),
            '{id}.L.MyelinMap_BC.native.func.gii', _prop_fssulc('myelin', 'lh_native'),
            '{id}.L.SmoothedMyelinMap.native.func.gii', _prop_fssulc('myelin_smooth_uncorrected',
                                                                     'lh_native'),
            '{id}.L.SmoothedMyelinMap_BC.native.func.gii', _prop_fssulc('myelin_smooth',
                                                                        'lh_native'),
            '{id}.L.RefMyelinMap.native.func.gii', _prop_fssulc('myelin_ref', 'lh_native'),
            '{id}.L.BA.native.label.gii', _prop('brodmann_area', 'lh_native'),
            '{id}.L.aparc.native.label.gii',  _prop('Deskian06_parcellation', 'lh_native'),
            '{id}.L.aparc.a2009s.native.label.gii', (_prop('Destrieux09_parcellation', 'lh_native')
                                                     + _prop('parcellation', 'lh_native')),
            '{id}.L.atlasroi.native.shape.gii', _prop('atlas', 'lh_native', key='label'),
            '{id}.L.curvature.native.shape.gii', _prop('curvature', 'lh_native', filt=lambda c:-c),
            '{id}.L.sulc.native.shape.gii', _prop('convexity', 'lh_native'),
            '{id}.L.corrThickness.native.shape.gii', _prop('thickness', 'lh_native'),
            '{id}.L.thickness.native.shape.gii', _prop('thickness_uncorrected', 'lh_native'),
            '{id}.L.roi.native.shape.gii', _prop('roi', 'lh_native', filt=_tobool),
            '{id}.L.sphere.native.surf.gii',     _reg('native', 'lh_native'),
            '{id}.L.sphere.reg.native.surf.gii', _reg('fsaverage', 'lh_native'),
            '{id}.L.sphere.reg.reg_LR.native.surf.gii', _reg_fs('fs_LR', 'lh_native'),
            '{id}.L.sphere.MSMAll.native.surf.gii', _reg_msmall('fs_LR', 'lh_native'),
            '{id}.L.sphere.MSMSulc.native.surf.gii',_reg_msmsulc('fs_LR', 'lh_native'),
            '{id}.R.ArealDistortion_FS.native.shape.gii', _prop_fs('areal_distortion', 'rh_native'),
            '{id}.R.ArealDistortion_MSMSulc.native.shape.gii', _prop_msmsulc('areal_distortion',
                                                                             'rh_native'),
            '{id}.R.ArealDistortion_MSMAll.native.shape.gii',  _prop_msmall('areal_distortion',
                                                                            'rh_native'),
            '{id}.R.MyelinMap.native.func.gii', _prop_fssulc('myelin_uncorrected', 'rh_native'),
            '{id}.R.MyelinMap_BC.native.func.gii', _prop_fssulc('myelin', 'rh_native'),
            '{id}.R.SmoothedMyelinMap.native.func.gii', _prop_fssulc('myelin_smooth_uncorrected',
                                                                      'rh_native'),
            '{id}.R.SmoothedMyelinMap_BC.native.func.gii', _prop_fssulc('myelin_smooth',
                                                                         'rh_native'),
            '{id}.R.RefMyelinMap.native.func.gii', _prop_fssulc('myelin_ref', 'rh_native'),
            '{id}.R.BA.native.label.gii', _prop('brodmann_area', 'rh_native'),
            '{id}.R.aparc.native.label.gii',  _prop('Deskian06_parcellation', 'rh_native'),
            '{id}.R.aparc.a2009s.native.label.gii', (_prop('Destrieux09_parcellation', 'rh_native')
                                                    + _prop('parcellation', 'rh_native')),
            '{id}.R.atlasroi.native.shape.gii', _prop('atlas', 'rh_native', key='label'),
            '{id}.R.curvature.native.shape.gii', _prop('curvature', 'rh_native', filt=lambda c:-c),
            '{id}.R.sulc.native.shape.gii', _prop('convexity', 'rh_native'),
            '{id}.R.corrThickness.native.shape.gii', _prop('thickness', 'rh_native'),
            '{id}.R.thickness.native.shape.gii', _prop('thickness_uncorrected', 'rh_native'),
            '{id}.R.roi.native.shape.gii', _prop('roi', 'rh_native', key='label'),
            '{id}.R.sphere.native.surf.gii',     _reg('native', 'rh_native'),
            '{id}.R.sphere.reg.native.surf.gii', _reg('fsaverage', 'rh_native'),
            '{id}.R.sphere.reg.reg_LR.native.surf.gii', _reg_fs('fs_LR', 'rh_native'),
            '{id}.R.sphere.MSMAll.native.surf.gii', _reg_msmall('fs_LR', 'rh_native'),
            '{id}.R.sphere.MSMSulc.native.surf.gii',_reg_msmsulc('fs_LR', 'rh_native')],
        'fsaverage_LR59k', [
            '{id}.L.BA.59k_fs_LR.label.gii', _prop('brodmann_area', 'lh_LR59k'),
            '{id}.L.aparc.59k_fs_LR.label.gii', _prop('Desikan06_parcellation', 'lh_LR59k'),
            '{id}.L.aparc.a2009s.59k_fs_LR.label.gii', (_prop('Destrieux09_parcellation',
                                                              'lh_LR59k') + 
                                                        _prop('parcellation', 'lh_LR59k')),
            '{id}.L.ArealDistortion_FS.59k_fs_LR.shape.gii', _prop_fs('areal_distortion',
                                                                      'lh_LR59k'),
            '{id}.L.ArealDistortion_MSMSulc.59k_fs_LR.shape.gii', _prop_msmsulc('areal_distortion',
                                                                                'lh_LR59k'),
            '{id}.L.ArealDistortion_MSMAll.59k_fs_LR.shape.gii',  _prop_msmall('areal_distortion',
                                                                               'lh_LR59k'),
            '{id}.L.MyelinMap.59k_fs_LR.func.gii', _prop_fssulc('myelin_uncorrected', 'lh_LR59k'),
            '{id}.L.MyelinMap_BC.59k_fs_LR.func.gii', _prop_fssulc('myelin', 'lh_LR59k'),
            '{id}.L.SmoothedMyelinMap.59k_fs_LR.func.gii',_prop_fssulc('myelin_smooth_uncorrected',
                                                                       'lh_LR59k'),
            '{id}.L.SmoothedMyelinMap_BC.59k_fs_LR.func.gii', _prop_fssulc('myelin_smooth',
                                                                           'lh_LR59k'),
            '{id}.L.RefMyelinMap.59k_fs_LR.func.gii', _prop_fssulc('myelin_ref', 'lh_LR59k'),
            '{id}.L.atlasroi.59k_fs_LR.shape.gii', _prop('atlas', 'lh_LR59k', key='label'),
            '{id}.L.curvature.59k_fs_LR.shape.gii', _prop('curvature', 'lh_LR59k',
                                                          filt=lambda c:-c),
            '{id}.L.sulc.59k_fs_LR.shape.gii', _prop('convexity', 'lh_LR59k'),
            '{id}.L.corrThickness.59k_fs_LR.shape.gii', _prop('thickness', 'lh_LR59k'),
            '{id}.L.thickness.59k_fs_LR.shape.gii', _prop('thickness_uncorrected', 'lh_LR59k'),
            '{id}.L.midthickness.59k_fs_LR.surf.gii',  _surf_fssulc('midgray', 'lh_LR59k'),
            '{id}.L.pial.59k_fs_LR.surf.gii',          _surf_fssulc('pial', 'lh_LR59k'),
            '{id}.L.inflated.59k_fs_LR.surf.gii',      _surf_fssulc('inflated', 'lh_LR59k'),
            '{id}.L.very_inflated.59k_fs_LR.surf.gii', _surf_fssulc('very_inflated', 'lh_LR59k'),
            '{id}.L.white.59k_fs_LR.surf.gii',         _surf_fssulc('white', 'lh_LR59k',
                                                                     tess=True),
            '{id}.L.white_1.6mm_MSMAll.59k_fs_LR.surf.gii',         _surf_msmall('white',
                                                                                 'lh_LR59k',
                                                                                 tess=True),
            '{id}.L.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii',  _surf_msmall('midgray',
                                                                                 'lh_LR59k'),
            '{id}.L.pial_1.6mm_MSMAll.59k_fs_LR.surf.gii',          _surf_msmall('pial',
                                                                                 'lh_LR59k'),
            '{id}.L.inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii',      _surf_msmall('inflated',
                                                                                 'lh_LR59k'),
            '{id}.L.very_inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii', _surf_msmall('very_inflated',
                                                                                 'lh_LR59k'),
            '{id}.L.sphere.59k_fs_LR.surf.gii', _reg('fs_LR', 'lh_LR59k'),
            '{id}.L.flat.59k_fs_LR.surf.gii', ({'flatmap':'lh_LR59k_FS',      'format':'gifti'},
                                               {'flatmap':'lh_LR59k_MSMSulc', 'format':'gifti'},
                                               {'flatmap':'lh_LR59k_MSMAll',  'format':'gifti'}),
            '{id}.R.BA.59k_fs_LR.label.gii', _prop('brodmann_area', 'rh_LR59k'),
            '{id}.R.aparc.59k_fs_LR.label.gii', _prop('Desikan06_parcellation', 'rh_LR59k'),
            '{id}.R.aparc.a2009s.59k_fs_LR.label.gii', (_prop('Destrieux09_parcellation',
                                                              'rh_LR59k') + 
                                                        _prop('parcellation', 'rh_LR59k')),
            '{id}.R.ArealDistortion_FS.59k_fs_LR.shape.gii', _prop_fs('areal_distortion',
                                                                      'rh_LR59k'),
            '{id}.R.ArealDistortion_MSMSulc.59k_fs_LR.shape.gii', _prop_msmsulc('areal_distortion',
                                                                                'rh_LR59k'),
            '{id}.R.ArealDistortion_MSMAll.59k_fs_LR.shape.gii', _prop_msmall('areal_distortion',
                                                                              'rh_LR59k'),
            '{id}.R.MyelinMap.59k_fs_LR.func.gii', _prop_fssulc('myelin_uncorrected', 'rh_LR59k'),
            '{id}.R.MyelinMap_BC.59k_fs_LR.func.gii', _prop_fssulc('myelin', 'rh_LR59k'),
            '{id}.R.SmoothedMyelinMap.59k_fs_LR.func.gii',_prop_fssulc('myelin_smooth_uncorrected',
                                                                       'rh_LR59k'),
            '{id}.R.SmoothedMyelinMap_BC.59k_fs_LR.func.gii', _prop_fssulc('myelin_smooth',
                                                                           'rh_LR59k'),
            '{id}.R.RefMyelinMap.59k_fs_LR.func.gii', _prop_fssulc('myelin_ref', 'rh_LR59k'),
            '{id}.R.atlasroi.59k_fs_LR.shape.gii', _prop('atlas', 'rh_LR59k', key='label'),
            '{id}.R.curvature.59k_fs_LR.shape.gii', _prop('curvature', 'rh_LR59k',
                                                          filt=lambda c:-c),
            '{id}.R.sulc.59k_fs_LR.shape.gii', _prop_fssulc('convexity', 'rh_LR59k'),
            '{id}.R.corrThickness.59k_fs_LR.shape.gii', _prop_fssulc('thickness', 'rh_LR59k'),
            '{id}.R.thickness.59k_fs_LR.shape.gii', _prop_fssulc('thickness_uncorrected',
                                                                 'rh_LR59k'),
            '{id}.R.midthickness.59k_fs_LR.surf.gii',  _surf_fssulc('midgray', 'rh_LR59k'),
            '{id}.R.pial.59k_fs_LR.surf.gii',          _surf_fssulc('pial', 'rh_LR59k'),
            '{id}.R.inflated.59k_fs_LR.surf.gii',      _surf_fssulc('inflated', 'rh_LR59k'),
            '{id}.R.very_inflated.59k_fs_LR.surf.gii', _surf_fssulc('very_inflated', 'rh_LR59k'),
            '{id}.R.white.59k_fs_LR.surf.gii',         _surf_fssulc('white', 'rh_LR59k',
                                                                    tess=True),
            '{id}.R.white_1.6mm_MSMAll.59k_fs_LR.surf.gii',         _surf_msmall('white',
                                                                                 'rh_LR59k',
                                                                                 tess=True),
            '{id}.R.midthickness_1.6mm_MSMAll.59k_fs_LR.surf.gii',  _surf_msmall('midgray',
                                                                                 'rh_LR59k'),
            '{id}.R.pial_1.6mm_MSMAll.59k_fs_LR.surf.gii',          _surf_msmall('pial',
                                                                                 'rh_LR59k'),
            '{id}.R.inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii',      _surf_msmall('inflated',
                                                                                 'rh_LR59k'),
            '{id}.R.very_inflated_1.6mm_MSMAll.59k_fs_LR.surf.gii', _surf_msmall('very_inflated',
                                                                                 'rh_LR59k'),
            '{id}.R.flat.59k_fs_LR.surf.gii', ({'flatmap':'rh_LR59k_FS',      'format':'gifti'},
                                               {'flatmap':'rh_LR59k_MSMSulc', 'format':'gifti'},
                                               {'flatmap':'rh_LR59k_MSMAll',  'format':'gifti'}),
            '{id}.R.sphere.59k_fs_LR.surf.gii', _reg('fs_LR', 'rh_LR59k'),
            '{id}.MyelinMap_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', _prop_cifti('myelin_uncorrected',
                                                                             59),
            '{id}.MyelinMap_BC_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', _prop_cifti('myelin', 59),
            '{id}.SmoothedMyelinMap_BC_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', _prop_cifti(
                'myelin_smooth', 59),
            '{id}.ArealDistortion_1.6mm_MSMAll.59k_fs_LR.shape.nii', _prop_cifti('areal_distortion',
                                                                                 59),
            '{id}.curvature_1.6mm_MSMAll.59k_fs_LR.shape.nii', _prop_cifti('curvature', 59,
                                                                           filt=lambda c:-c),
            '{id}.sulc_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', _prop_cifti('convexity', 59),
            '{id}.thickness_1.6mm_MSMAll.59k_fs_LR.dscalar.nii',_prop_cifti('thickness_uncorrected',
                                                                            59),
            '{id}.corrThickness_1.6mm_MSMAll.59k_fs_LR.dscalar.nii', _prop_cifti('thickness', 59)],
        'fsaverage_LR32k', [
            '{id}.L.BA.32k_fs_LR.label.gii', _prop('brodmann_area', 'lh_LR32k'),
            '{id}.L.aparc.32k_fs_LR.label.gii', _prop('Desikan06_parcellation', 'lh_LR32k'),
            '{id}.L.aparc.a2009s.32k_fs_LR.label.gii', (_prop('Destrieux09_parcellation',
                                                              'lh_LR32k') + 
                                                        _prop('parcellation', 'lh_LR32k')),
            '{id}.L.ArealDistortion_FS.32k_fs_LR.shape.gii', _prop_fs('areal_distortion',
                                                                      'lh_LR32k'),
            '{id}.L.ArealDistortion_MSMSulc.32k_fs_LR.shape.gii', _prop_msmsulc('areal_distortion',
                                                                                'lh_LR32k'),
            '{id}.L.ArealDistortion_MSMAll.32k_fs_LR.shape.gii', _prop_msmall('areal_distortion',
                                                                              'lh_LR32k'),
            '{id}.L.MyelinMap.32k_fs_LR.func.gii', _prop_fssulc('myelin_uncorrected', 'lh_LR32k'),
            '{id}.L.MyelinMap_BC.32k_fs_LR.func.gii', _prop_fssulc('myelin', 'lh_LR32k'),
            '{id}.L.SmoothedMyelinMap.32k_fs_LR.func.gii',_prop_fssulc('myelin_smooth_uncorrected',
                                                                       'lh_LR32k'),
            '{id}.L.SmoothedMyelinMap_BC.32k_fs_LR.func.gii', _prop_fssulc('myelin_smooth',
                                                                           'lh_LR32k'),
            '{id}.L.RefMyelinMap.32k_fs_LR.func.gii', _prop_fssulc('myelin_ref', 'lh_LR32k'),
            '{id}.L.atlasroi.32k_fs_LR.shape.gii', _prop('atlas', 'lh_LR32k', key='label'),
            '{id}.L.curvature.32k_fs_LR.shape.gii', _prop('curvature', 'lh_LR32k',
                                                          filt=lambda c:-c),
            '{id}.L.sulc.32k_fs_LR.shape.gii', _prop('convexity', 'lh_LR32k'),
            '{id}.L.corrThickness.32k_fs_LR.shape.gii', _prop('thickness', 'lh_LR32k'),
            '{id}.L.thickness.32k_fs_LR.shape.gii', _prop('thickness_uncorrected', 'lh_LR32k'),
            '{id}.L.midthickness.32k_fs_LR.surf.gii',  _surf_fssulc('midgray', 'lh_LR32k'),
            '{id}.L.pial.32k_fs_LR.surf.gii',          _surf_fssulc('pial', 'lh_LR32k'),
            '{id}.L.inflated.32k_fs_LR.surf.gii',      _surf_fssulc('inflated', 'lh_LR32k'),
            '{id}.L.very_inflated.32k_fs_LR.surf.gii', _surf_fssulc('very_inflated', 'lh_LR32k'),
            '{id}.L.white.32k_fs_LR.surf.gii',         _surf_fssulc('white', 'lh_LR32k',
                                                                     tess=True),
            '{id}.L.white_MSMAll.32k_fs_LR.surf.gii',         _surf_msmall('white',
                                                                                 'lh_LR32k',
                                                                                 tess=True),
            '{id}.L.midthickness_MSMAll.32k_fs_LR.surf.gii',  _surf_msmall('midgray',
                                                                                 'lh_LR32k'),
            '{id}.L.pial_MSMAll.32k_fs_LR.surf.gii',          _surf_msmall('pial',
                                                                                 'lh_LR32k'),
            '{id}.L.inflated_MSMAll.32k_fs_LR.surf.gii',      _surf_msmall('inflated',
                                                                                 'lh_LR32k'),
            '{id}.L.very_inflated_MSMAll.32k_fs_LR.surf.gii', _surf_msmall('very_inflated',
                                                                                 'lh_LR32k'),
            '{id}.L.sphere.32k_fs_LR.surf.gii', _reg('fs_LR', 'lh_LR32k'),
            '{id}.L.flat.32k_fs_LR.surf.gii', ({'flatmap':'lh_LR32k_FS',      'format':'gifti'},
                                               {'flatmap':'lh_LR32k_MSMSulc', 'format':'gifti'},
                                               {'flatmap':'lh_LR32k_MSMAll',  'format':'gifti'}),
            '{id}.R.BA.32k_fs_LR.label.gii', _prop('brodmann_area', 'rh_LR32k'),
            '{id}.R.aparc.32k_fs_LR.label.gii', _prop('Desikan06_parcellation', 'rh_LR32k'),
            '{id}.R.aparc.a2009s.32k_fs_LR.label.gii', (_prop('Destrieux09_parcellation',
                                                              'rh_LR32k') + 
                                                        _prop('parcellation', 'rh_LR32k')),
            '{id}.R.ArealDistortion_FS.32k_fs_LR.shape.gii', _prop_fs('areal_distortion',
                                                                      'rh_LR32k'),
            '{id}.R.ArealDistortion_MSMSulc.32k_fs_LR.shape.gii', _prop_msmsulc('areal_distortion',
                                                                                'rh_LR32k'),
            '{id}.R.ArealDistortion_MSMAll.32k_fs_LR.shape.gii',  _prop_msmall('areal_distortion',
                                                                               'rh_LR32k'),
            '{id}.R.MyelinMap.32k_fs_LR.func.gii', _prop_fssulc('myelin_uncorrected', 'rh_LR32k'),
            '{id}.R.MyelinMap_BC.32k_fs_LR.func.gii', _prop_fssulc('myelin', 'rh_LR32k'),
            '{id}.R.SmoothedMyelinMap.32k_fs_LR.func.gii',_prop_fssulc('myelin_smooth_uncorrected',
                                                                        'rh_LR32k'),
            '{id}.R.SmoothedMyelinMap_BC.32k_fs_LR.func.gii', _prop_fssulc('myelin_smooth',
                                                                            'rh_LR32k'),
            '{id}.R.RefMyelinMap.32k_fs_LR.func.gii', _prop_fssulc('myelin_ref', 'rh_LR32k'),
            '{id}.R.atlasroi.32k_fs_LR.shape.gii', _prop('atlas', 'rh_LR32k', key='label'),
            '{id}.R.curvature.32k_fs_LR.shape.gii', _prop('curvature', 'rh_LR32k',
                                                          filt=lambda c:-c),
            '{id}.R.sulc.32k_fs_LR.shape.gii', _prop_fssulc('convexity', 'rh_LR32k'),
            '{id}.R.corrThickness.32k_fs_LR.shape.gii', _prop_fssulc('thickness', 'rh_LR32k'),
            '{id}.R.thickness.32k_fs_LR.shape.gii', _prop_fssulc('thickness_uncorrected',
                                                                 'rh_LR32k'),
            '{id}.R.midthickness.32k_fs_LR.surf.gii',  _surf_fssulc('midgray', 'rh_LR32k'),
            '{id}.R.pial.32k_fs_LR.surf.gii',          _surf_fssulc('pial', 'rh_LR32k'),
            '{id}.R.inflated.32k_fs_LR.surf.gii',      _surf_fssulc('inflated', 'rh_LR32k'),
            '{id}.R.very_inflated.32k_fs_LR.surf.gii', _surf_fssulc('very_inflated', 'rh_LR32k'),
            '{id}.R.white.32k_fs_LR.surf.gii',         _surf_fssulc('white', 'rh_LR32k', tess=True),
            '{id}.R.white_MSMAll.32k_fs_LR.surf.gii',         _surf_msmall('white', 'rh_LR32k',
                                                                           tess=True),
            '{id}.R.midthickness_MSMAll.32k_fs_LR.surf.gii',  _surf_msmall('midgray', 'rh_LR32k'),
            '{id}.R.pial_MSMAll.32k_fs_LR.surf.gii',          _surf_msmall('pial', 'rh_LR32k'),
            '{id}.R.inflated_MSMAll.32k_fs_LR.surf.gii',      _surf_msmall('inflated', 'rh_LR32k'),
            '{id}.R.very_inflated_MSMAll.32k_fs_LR.surf.gii', _surf_msmall('very_inflated',
                                                                           'rh_LR32k'),
            '{id}.R.flat.32k_fs_LR.surf.gii', ({'flatmap':'rh_LR32k_FS',      'format':'gifti'},
                                               {'flatmap':'rh_LR32k_MSMSulc', 'format':'gifti'},
                                               {'flatmap':'rh_LR32k_MSMAll',  'format':'gifti'}),
            '{id}.R.sphere.32k_fs_LR.surf.gii', _reg('fs_LR', 'rh_LR32k'),
            '{id}.MyelinMap_MSMAll.32k_fs_LR.dscalar.nii', _prop_cifti('myelin_uncorrected', 32),
            '{id}.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii', _prop_cifti('myelin', 32),
            '{id}.SmoothedMyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii', _prop_cifti('myelin_smooth',
                                                                                  32),
            '{id}.ArealDistortion_MSMAll.32k_fs_LR.shape.nii', _prop_cifti('areal_distortion', 32),
            '{id}.curvature_MSMAll.32k_fs_LR.shape.nii', _prop_cifti('curvature', 32,
                                                                     filt=lambda c:-c),
            '{id}.sulc_MSMAll.32k_fs_LR.dscalar.nii', _prop_cifti('convexity', 32),
            '{id}.thickness_MSMAll.32k_fs_LR.dscalar.nii', _prop_cifti('thickness_uncorrected', 32),
            '{id}.corrThickness_MSMAll.32k_fs_LR.dscalar.nii', _prop_cifti('thickness', 32)]]]

@nyio.importer('cifti', ('nii',))
def load_cifti(filename, to='auto'):
    '''
    load_cifti(filename) yields the cifti image referened by the given filename by using the nibabel
      load function.
    
    The optional argument to may be used to coerce the resulting data to a particular format; the
    following arguments are understood:
      * 'header' will yield just the image header
      * 'data' will yield the image's data-array
      * 'field' will yield a squeezed version of the image's data-array and will raise an error if
        the data object has more than 2 non-unitary dimensions (appropriate for loading surface
        properties stored in image files)
      * 'image' will yield the raw image object
      * 'auto' is equivalent to 'image' unless the image has no more than 2 non-unitary dimensions,
        in which case it is assumed to be a surface-field and the return value is equivalent to
        the 'field' value.
    '''
    img = nib.load(filename)
    if not isinstance(img, nib.cifti2.Cifti2Image):
        raise ValueError('given file is not a cifti image')
    to = 'auto' if to is None else to.lower()
    if   to == 'image':  return img
    elif to == 'data':   return img.dataobj
    elif to == 'header': return img.header
    elif to == 'field':
        dat = np.squeeze(np.asarray(img.dataobj))
        if len(dat.shape) > 2:
            raise ValueError('image requested as field has more than 2 non-unitary dimensions')
        return dat
    elif to in ['auto', 'automatic']:
        dims = set(np.shape(img.dataobj))
        if 1 < len(dims) < 4 and 1 in dims: return np.squeeze(np.asarray(img.dataobj))
        else:                               return img
    else:
        raise ValueError('unrecognized \'to\' argument \'%s\'' % to)
