####################################################################################################
# neuropythy/freesurfer/core.py
# Simple tools for use with FreeSurfer in Python
# By Noah C. Benson

import numpy                        as np
import nibabel                      as nib
import nibabel.freesurfer.io        as fsio
import nibabel.freesurfer.mghformat as fsmgh
import pyrsistent                   as pyr
from   six.moves                import collections_abc as collections
import os, warnings, six, pimms

from .. import geometry as geo
from .. import mri      as mri
from .. import io       as nyio
#import ..io as nyio

from ..util import (config, library_path, curry, pseudo_path, is_pseudo_path, data_struct,
                    label_indices, label_index, to_label_index, is_tuple, file_map, to_pseudo_path,
                    try_until)

def library_freesurfer_subjects():
    '''
    library_freesurfer_subjects() yields the path of the neuropythy library's FreeSurfer subjects
      directory.
    '''
    return os.path.join(library_path(), 'data')

####################################################################################################
# FreeSurfer home and where to find FreeSurfer LUTs and such
# some functions to load in the color luts:
@nyio.importer('FreeSurferLUT', 'LUT.txt')
def load_LUT(filename, to='data'):
    '''
    load_LUT(filename) loads the given filename as a FreeSurfer LUT.

    The optional argument to (default: 'data') specifies how the LUT should be interpreted; it can
    be any of the following:
      * 'data' specifies that a dataframe should be returned.
    '''
    from neuropythy.util import to_dataframe
    import pandas
    # start by slurping in the text:
    dat = pandas.read_csv(filename, comment='#', sep='\s+', names=['id', 'name', 'r','g','b','a'])
    # if all the alpha values are 0, we set them to 1 (not sure why freesurfer does this)
    dat['a'] = 255 - dat['a']
    if pimms.is_str(to): to = to.lower()
    if   to is None: return dat
    elif to == 'data':
        df = to_dataframe({'id': dat['id'].values, 'name': dat['name'].values})
        df['color'] = dat.apply(lambda r: [r[k]/255.0 for k in ['r','g','b','a']], axis=1)
        df.set_index('id', inplace=True)
        return df
    else: raise ValueError('Unknown to argument: %s' % to)
# A function to load in default data from the freesurfer home: e.g., the default LUTs
def _load_fshome(path):
    luts = {'aseg': 'ASegStatsLUT.txt',
            'wm':   'WMParcStatsLUT.txt',
            'all':  'FreeSurferColorLUT.txt'}
    luts = {k: curry(load_LUT, os.path.join(path, v)) for (k,v) in six.iteritems(luts)}
    luts = pimms.lazy_map(luts)
    # put these into the label indices
    global label_indices
    label_indices['freesurfer']      = label_index(luts['all'])
    label_indices['freesurfer_aseg'] = label_index(luts['aseg'])
    label_indices['freesurfer_wm']   = label_index(luts['wm'])
    return data_struct(luts=luts, path=path)
config.declare_dir('freesurfer_home', environ_name='FREESURFER_HOME', filter=_load_fshome)

####################################################################################################
# Subject Directory and where to find Subjects
def to_subject_paths(paths, previous_paths=None):
    '''
    to_subject_paths(paths) accepts either a string that is a :-separated list of directories or a
      list of directories and yields a list of all the existing directories.
    '''
    if paths is None: return []
    if pimms.is_str(paths): paths = paths.split(':')
    paths = [os.path.expanduser(p) for p in paths]
    if previous_paths is not None: paths = previous_paths + paths
    return [p for p in paths if os.path.isdir(p)]
def freesurfer_paths_merge(p0, p1):
    '''
    freesurfer_paths_merge(p0, p1) yields to_subject_paths(p0) + to_subject_paths(p1).
    '''
    return to_subject_paths(p0) + to_subject_paths(p1)
config.declare('freesurfer_subject_paths', environ_name='SUBJECTS_DIR', filter=to_subject_paths,
               merge=freesurfer_paths_merge)

def subject_paths():
    '''
    subject_paths() yields a list of paths to Freesurfer subject directories in which subjects are
      automatically searched for when identified by subject-name only. These paths are searched in
      the order returned from this function.

    If you must edit these paths, it is recommended to use add_subject_path, and clear_subject_paths
    functions.
    '''
    return config['freesurfer_subject_paths']

def clear_subject_paths(subpaths):
    '''
    clear_subject_paths() resets the freesurfer subject paths to be empty and yields the previous
      list of subject paths.
    '''
    sd = config['freesurfer_subject_paths']
    config['freesurfer_subject_paths'] = []
    return sd

def add_subject_path(path, index=0):
    '''
    add_subject_path(path) will add the given path to the list of subject directories in which to
      search for Freesurfer subjects. The optional argument index may be given to specify the
      precedence of this path when searching for a new subject; the default, 0, always inserts the
      path at the front of the list; a value of k indicates that the new list should have the new
      path at index k.
    The path may contain :'s, in which case the individual directories are separated and added.
    If the given path is not a directory or the path could not be inserted, yields False;
    otherwise, yields True. If the string contains a : and multiple paths, then True is yielded
    only if all paths were successfully inserted.
    See also subject_paths.
    '''
    paths = [p for p in path.split(':') if len(p) > 0]
    if len(paths) > 1:
        tests = [add_subject_path(p, index=index) for p in reversed(paths)]
        return all(t for t in tests)
    else:
        fsp = config['freesurfer_subject_paths']
        if not os.path.isdir(path): return False
        if path in fsp: return True
        try:
            if index is None or index is Ellipsis:
                sd = fsp + [path]
            else:
                sd = fsp + []
                sd.insert(index, path)
            config['freesurfer_subject_paths'] = sd
            return True
        except Exception:
            return False

if config['freesurfer_subject_paths'] is None:
    # If the subjects path wasn't provided, try a few common subject paths
    add_subject_path('/Applications/freesurfer/subjects')
    add_subject_path('/opt/freesurfer/subjects')
# Regardless, make sure we check the FreeSurfer home
if config['freesurfer_home'] is not None:
    add_subject_path(os.path.join(config['freesurfer_home'].path, 'subjects'), None)

def is_freesurfer_subject_path(path):
    '''
    is_freesurfer_subject_path(path) yields True if the given path appears to be a valid freesurfer
      subject path and False otherwise.
    is_freesurfer_subject_path(pdir) performs the same check for the given pseudo-dir object pdir.

    A path is considered to be freesurfer-subject-like if it contains the directories mri/, surf/,
    and label/. 
    '''
    needed = ['mri', 'surf', 'label']
    if   is_pseudo_path(path): return all(path.find(d) is not None for d in needed)
    elif os.path.isdir(path): return all(os.path.isdir(os.path.join(path, d)) for d in needed)
    else:                     return False
  
def find_subject_path(sub, check_path=True):
    '''
    find_subject_path(sub) yields the full path of a Freesurfer subject with the name given by the
      string sub, if such a subject can be found in the Freesurfer search paths. See also
      add_subject_path.

    If no subject is found, then None is returned.

    The optional argument check_path (default: True) may be set to False to indicate that sanity
    checks regarding the state of the file-system should be skipped. If check_path is set to True,
    then a directory will only be returned if it is a directory on the file-system and it contains
    at least the subdirectories mri/, label/, and surf/. If check_path is set to False, then
    find_subject_path will first search for any directory that satisfies the above requirements,
    and, failing that, will search for any directory matching sub that at least exists. If this also
    fails, then sub itself is returned (i.e., sub is presumed to be a valid path).
    '''
    sdirs = config['freesurfer_subject_paths']
    # if it's a full/relative path already, use it:
    if ((not check_path or is_freesurfer_subject_path(sub)) and
        (check_path is None or os.path.isdir(sub))):
        return sub
    return next((p for sd in sdirs for p in [os.path.join(sd, sub)]
                 if ((not check_path or is_freesurfer_subject_path(p)) and
                     (check_path is None or os.path.isdir(p)))),
                None)

# Used to load immutable-like mgh objects
def _load_imm_mgh(flnm,u):
    img = fsmgh.load(flnm)
    try: img.dataobj.flags['WRITEABLE'] = False
    except Exception:
        d = np.asanyarray(img.dataobj)
        d.flags['WRITEABLE'] = False
        img = fsmgh.MGHImage(d, img.affine, img.header)
    return img

####################################################################################################
# Filemap
# This is the FreeSurfer file-map data, as neuropythy understands it.
def _load_label(flnm,u):
    (a,b) = nyio.load(flnm, 'freesurfer_label', to='all')
    a.setflags(write=False)
    b.setflags(write=False)
    return (a,b)
def _load_annot(flnm,u):
    (a,b) = nyio.load(flnm, 'freesurfer_annot', to='all')
    a.setflags(write=False)
    return (a,b.persist())
def _load_surfx(flnm,u):
    x = fsio.read_geometry(flnm)[0].T
    x.setflags(write=False)
    return x
def _load_prop(flnm,u):
    x = fsio.read_morph_data(flnm)
    x.setflags(write=False)
    return x
def _load_tris(flnm,u):
    x = fsio.read_geometry(flnm)[1]
    x.setflags(write=False)
    return x
def _load_geo(flnm,u):
    (x,t) = fsio.read_geometry(flnm)[:2]
    x = x.T
    x.setflags(write=False)
    t.setflags(write=False)
    return (x,t)    
freesurfer_subject_data_hierarchy = [['image'],                ['raw_image'],
                                     ['hemi', 'surface'],      ['hemi', 'tess'],
                                     ['hemi', 'registration'], ['hemi', 'property'],
                                     ['hemi', 'label'],        #['hemi', 'alt_label'],
                                     ['hemi', 'weight'],       ['hemi', 'alt_weight'],
                                     ['hemi', 'annot'],        ['hemi', 'alt_annot']]
freesurfer_subject_filemap_instructions = [
    'mri', [
        'rawavg.mgz',              ({'raw_image':'rawavg', 'load': _load_imm_mgh},
                                    {'image':'raw', 'load': _load_imm_mgh}),                     
        'orig.mgz',                ({'raw_image':'orig', 'load': _load_imm_mgh},
                                    {'image':'original', 'load': _load_imm_mgh}),                
        'orig_nu.mgz',             ({'raw_image':'orig_nu', 'load': _load_imm_mgh},
                                    {'image':'conformed', 'load': _load_imm_mgh}),
        'nu.mgz',                  ({'raw_image':'nu', 'load': _load_imm_mgh},
                                    {'image':'uniform', 'load': _load_imm_mgh}),
        'T1.mgz',                  ({'raw_image':'T1', 'load': _load_imm_mgh},
                                    {'image':'intensity_normalized', 'load': _load_imm_mgh}),
        'brainmask.auto.mgz',      ({'raw_image':'brainmask.auto', 'load': _load_imm_mgh},
                                    {'image':'auto_masked_brain', 'load': _load_imm_mgh}),
        'brainmask.mgz',           ({'raw_image':'brainmask', 'load': _load_imm_mgh},
                                    {'image':'masked_brain', 'load': _load_imm_mgh}),
        'norm.mgz',                ({'raw_image':'norm', 'load': _load_imm_mgh},
                                    {'image':'normalized', 'load': _load_imm_mgh}),
        'aseg.auto_noCCseg.mgz',   {'raw_image':'aseg.auto_noCCseg', 'load': _load_imm_mgh},
        'aseg.auto.mgz',           ({'raw_image':'aseg.auto', 'load': _load_imm_mgh},
                                    {'image':'auto_segmentation', 'load': _load_imm_mgh}),
        'aseg.presurf.hypos.mgz',  {'raw_image':'aseg.presurf.hypos', 'load': _load_imm_mgh},
        'aseg.presurf.mgz',        ({'raw_image':'aseg.presurf', 'load': _load_imm_mgh},
                                    {'image':'presurface_segmentation', 'load': _load_imm_mgh}), 
        'brain.mgz',               ({'raw_image':'brain', 'load': _load_imm_mgh},
                                    {'image':'brain', 'load': _load_imm_mgh}),
        'brain.finalsurfs.mgz',    {'raw_image':'brain.finalsurfs', 'load': _load_imm_mgh},
        'wm.seg.mgz',              {'raw_image':'wm.seg', 'load': _load_imm_mgh},
        'wm.asegedit.mgz',         {'raw_image':'wm.asegedit', 'load': _load_imm_mgh},
        'wm.mgz',                  ({'raw_image':'wm', 'load': _load_imm_mgh},
                                    {'image':'white_matter', 'load': _load_imm_mgh}),
        'filled.mgz',              {'raw_image':'filled', 'load': _load_imm_mgh},
        'T2.mgz',                  ({'raw_image':'T2', 'load': _load_imm_mgh},
                                    {'image':'T2', 'load': _load_imm_mgh}),
        'ribbon.mgz',              ({'raw_image':'ribbon', 'load': _load_imm_mgh},
                                    {'image':'ribbon', 'load': _load_imm_mgh}),
        'aparc+aseg.mgz',          ({'raw_image':'aparc+aseg', 'load': _load_imm_mgh},
                                    {'image':'Desikan06_parcellation', 'load': _load_imm_mgh}),
        'aparc.DKTatlas+aseg.mgz', ({'raw_image':'aparc.DKTatlas+aseg', 'load': _load_imm_mgh},
                                    {'image':'DKT40_parcellation', 'load': _load_imm_mgh}),
        'aparc.a2009s+aseg.mgz',   ({'raw_image':'aparc.a2009s+aseg', 'load': _load_imm_mgh},
                                    {'image':'Destrieux09_parcellation', 'load': _load_imm_mgh},
                                    {'image':'parcellation', 'load': _load_imm_mgh}),
        'ctrl_pts.mgz',            {'raw_image':'ctrl_pts', 'load': _load_imm_mgh},
        'lh.ribbon.mgz',           ({'raw_image':'lh.ribbon', 'load': _load_imm_mgh},
                                    {'image':'lh_ribbon', 'load': _load_imm_mgh}),
        'rh.ribbon.mgz',           ({'raw_image':'rh.ribbon', 'load': _load_imm_mgh},
                                    {'image':'rh_ribbon', 'load': _load_imm_mgh}),
        'wmparc.mgz',              ({'raw_image':'wmparc', 'load': _load_imm_mgh},
                                    {'image':'white_parcellation', 'load': _load_imm_mgh}),
        'aseg.mgz',                ({'raw_image':'aseg', 'load': _load_imm_mgh},
                                    {'image':'segmentation', 'load': _load_imm_mgh})],
    'surf', [
        'lh.area',           ({'hemi':'lh',  'property':'white_surface_area', 'load':_load_prop},
                              {'hemi':'rhx', 'property':'white_surface_area', 'load':_load_prop}),
        'lh.area.mid',       ({'hemi':'lh',  'property':'midgray_surface_area', 'load':_load_prop},
                              {'hemi':'lh',  'property':'surface_area', 'load':_load_prop},
                              {'hemi':'rhx', 'property':'midgray_surface_area', 'load':_load_prop},
                              {'hemi':'rhx', 'property':'surface_area', 'load':_load_prop}),
        'lh.area.pial',      ({'hemi':'lh',  'property':'pial_surface_area', 'load':_load_prop},
                              {'hemi':'rhx', 'property':'pial_surface_area', 'load':_load_prop}),
        'lh.avg_curv',       ({'hemi':'lh',  'property':'atlas_curvature', 'load':_load_prop},
                              {'hemi':'rhx', 'property':'atlas_curvature', 'load':_load_prop}),
        'lh.curv',           ({'hemi':'lh',  'property':'white_curvature', 'load':_load_prop},
                              {'hemi':'lh',  'property':'curvature', 'load':_load_prop},
                              {'hemi':'rhx', 'property':'white_curvature', 'load':_load_prop},
                              {'hemi':'rhx', 'property':'curvature', 'load':_load_prop}),
        'lh.curv.pial',      ({'hemi':'lh',  'property':'pial_curvature', 'load':_load_prop},
                              {'hemi':'rhx', 'property':'pial_curvature', 'load':_load_prop}),
        'lh.jacobian_white', ({'hemi':'lh',  'property':'jacobian_norm', 'load':_load_prop},
                              {'hemi':'rhx', 'property':'jacobian_norm', 'load':_load_prop}),
        'lh.inflated',       {'hemi':'lh',  'surface':'inflated', 'load':_load_surfx},
        'lh.orig',           {'hemi':'lh',  'surface':'white_original', 'load':_load_surfx},
        'lh.pial',           {'hemi':'lh',  'surface':'pial', 'load':_load_surfx},
        'lh.smoothwm',       {'hemi':'lh',  'surface':'white_smooth', 'load':_load_surfx},
        'lh.sphere',         ({'hemi':'lh',  'surface':'sphere', 'load':_load_surfx},
                              {'hemi':'lh',  'registration':'native', 'load':_load_surfx}),
        'lh.sphere.reg',     {'hemi':'lh',  'registration':'fsaverage', 'load':_load_surfx},
        'lh.white',          ({'hemi':'lh',  'surface':'white',
                               'load':_load_geo, 'filt':lambda u:u[0]},
                              {'hemi':'lh',  'tess':'white',
                               'load':_load_geo, 'filt':lambda u:u[1]}),
        'lh.sulc',           ({'hemi':'lh',  'property':'convexity', 'load':_load_prop},
                              {'hemi':'rhx', 'property':'convexity', 'load':_load_prop}),
        'lh.thickness',      ({'hemi':'lh',  'property':'thickness', 'load':_load_prop},
                              {'hemi':'rhx', 'property':'thickness', 'load':_load_prop}),
        'rh.area',           ({'hemi':'rh',  'property':'white_surface_area', 'load':_load_prop},
                              {'hemi':'lhx', 'property':'white_surface_area', 'load':_load_prop}),
        'rh.area.mid',       ({'hemi':'rh',  'property':'midgray_surface_area', 'load':_load_prop},
                              {'hemi':'rh',  'property':'surface_area', 'load':_load_prop},
                              {'hemi':'lhx', 'property':'midgray_surface_area', 'load':_load_prop},
                              {'hemi':'lhx', 'property':'surface_area', 'load':_load_prop}),
        'rh.area.pial',      ({'hemi':'rh',  'property':'pial_surface_area', 'load':_load_prop},
                              {'hemi':'lhx', 'property':'pial_surface_area', 'load':_load_prop}),
        'rh.avg_curv',       ({'hemi':'rh',  'property':'atlas_curvature', 'load':_load_prop},
                              {'hemi':'lhx', 'property':'atlas_curvature', 'load':_load_prop}),
        'rh.curv',           ({'hemi':'rh',  'property':'white_curvature', 'load':_load_prop},
                              {'hemi':'rh',  'property':'curvature', 'load':_load_prop},
                              {'hemi':'lhx', 'property':'white_curvature', 'load':_load_prop},
                              {'hemi':'lhx', 'property':'curvature', 'load':_load_prop}),
        'rh.curv.pial',      ({'hemi':'rh',  'property':'pial_curvature', 'load':_load_prop},
                              {'hemi':'lhx', 'property':'pial_curvature', 'load':_load_prop}),
        'rh.inflated',       ({'hemi':'rh',  'surface':'inflated', 'load':_load_surfx},
                              {'hemi':'lhx', 'surface':'inflated', 'load':_load_surfx}),
        'rh.jacobian_white', ({'hemi':'rh',  'property':'jacobian_norm', 'load':_load_prop},
                              {'hemi':'lhx', 'property':'jacobian_norm', 'load':_load_prop}),
        'rh.orig',           ({'hemi':'rh',  'surface':'white_original', 'load':_load_surfx},
                              {'hemi':'lhx', 'surface':'white_original', 'load':_load_surfx}),
        'rh.pial',           ({'hemi':'rh',  'surface':'pial', 'load':_load_surfx},
                              {'hemi':'lhx', 'surface':'pial', 'load':_load_surfx}),
        'rh.smoothwm',       ({'hemi':'rh',  'surface':'white_smooth', 'load':_load_surfx},
                              {'hemi':'lhx', 'surface':'white_smooth', 'load':_load_surfx}),
        'rh.sphere',         ({'hemi':'rh',  'surface':'sphere', 'load':_load_surfx},
                              {'hemi':'rh',  'registration':'native', 'load':_load_surfx},
                              {'hemi':'lhx', 'surface':'sphere', 'load':_load_surfx},
                              {'hemi':'lhx', 'registration':'native', 'load':_load_surfx}),
        'rh.sphere.reg',     ({'hemi':'rh',  'registration':'fsaverage', 'load':_load_surfx},
                              {'hemi':'lhx', 'registration':'fsaverage', 'load':_load_surfx}),
        'rh.sulc',           ({'hemi':'rh',  'property':'convexity', 'load':_load_prop},
                              {'hemi':'lhx', 'property':'convexity', 'load':_load_prop}),
        'rh.thickness',      ({'hemi':'rh',  'property':'thickness', 'load':_load_prop},
                              {'hemi':'lhx', 'property':'thickness', 'load':_load_prop}),
        'rh.white',          ({'hemi':'rh',  'surface':'white',
                               'load':_load_geo, 'filt':lambda u:u[0]},
                              {'hemi':'rh',  'tess':'white',
                               'load':_load_geo, 'filt':lambda u:u[1]},
                              {'hemi':'lhx', 'surface':'white',
                               'load':_load_geo, 'filt':lambda u:u[0]},
                              {'hemi':'lhx', 'tess':'white',
                               'load':_load_geo, 'filt':lambda u:u[1]}),
        # a few other registrations that we recognize
        'lh.fsaverage_sym.sphere.reg',        {'hemi':'lh',  'registration':'fsaverage_sym',
                                               'load':_load_surfx},
        #'rh.fsaverage_sym.sphere.reg',        {'hemi':'lhx', 'registration':'fsaverage_sym',
        #                                       'load':_load_surfx},
        'lh.benson14_retinotopy.sphere.reg',  ({'hemi':'lh', 'registration':'benson14_retinotopy',
                                                'load':_load_surfx},
                                               {'hemi':'lh', 'registration':'benson14',
                                                'load':_load_surfx}),
        'rh.benson14_retinotopy.sphere.reg',  ({'hemi':'rh', 'registration':'benson14_retinotopy',
                                                'load':_load_surfx},
                                               {'hemi':'rh', 'registration':'benson14',
                                                'load':_load_surfx}),
        'lh.benson17_retinotopy.sphere.reg',  ({'hemi':'lh', 'registration':'benson17_retinotopy',
                                                'load':_load_surfx},
                                               {'hemi':'lh', 'registration':'benson17',
                                                'load':_load_surfx}),
        'rh.benson17_retinotopy.sphere.reg',  ({'hemi':'rh', 'registration':'benson17_retinotopy',
                                                'load':_load_surfx},
                                               {'hemi':'rh', 'registration':'benson17',
                                                'load':_load_surfx})],
    'xhemi', [
        'surf', [
            #'rh.fsaverage_sym.sphere.reg',        {'hemi':'lhx', 'registration':'fsaverage_sym',
            #                                   'load':_load_surfx},
            'lh.fsaverage_sym.sphere.reg',    {'hemi':'rh',  'registration':'fsaverage_sym',
                                               'load':_load_surfx},
            'lh.inflated',   {'hemi':'rhx',  'surface':'inflated', 'load':_load_surfx},
            'lh.orig',       {'hemi':'rhx',  'surface':'white_original', 'load':_load_surfx},
            'lh.pial',       {'hemi':'rhx',  'surface':'pial', 'load':_load_surfx},
            'lh.smoothwm',   {'hemi':'rhx',  'surface':'white_smooth', 'load':_load_surfx},
            'lh.sphere',     ({'hemi':'rhx',  'surface':'sphere', 'load':_load_surfx},
                              {'hemi':'rhx',  'registration':'native', 'load':_load_surfx}),
            'lh.sphere.reg', {'hemi':'rhx',  'registration':'fsaverage', 'load':_load_surfx},
            'lh.white',      ({'hemi':'rhx',  'surface':'white',
                               'load':_load_geo, 'filt':lambda u:u[0]},
                              {'hemi':'rhx',  'tess':'white',
                               'load':_load_geo, 'filt':lambda u:u[1]}),
            'rh.inflated',   {'hemi':'lhx',  'surface':'inflated', 'load':_load_surfx},
            'rh.orig',       {'hemi':'lhx',  'surface':'white_original', 'load':_load_surfx},
            'rh.pial',       {'hemi':'lhx',  'surface':'pial', 'load':_load_surfx},
            'rh.smoothwm',   {'hemi':'lhx',  'surface':'white_smooth', 'load':_load_surfx},
            'rh.sphere',     ({'hemi':'lhx',  'surface':'sphere', 'load':_load_surfx},
                             {'hemi':'lhx',  'registration':'native', 'load':_load_surfx}),
            'rh.sphere.reg', {'hemi':'lhx',  'registration':'fsaverage', 'load':_load_surfx},
            'rh.white',      ({'hemi':'lhx',  'surface':'white',
                               'load':_load_geo, 'filt':lambda u:u[0]},
                              {'hemi':'lhx',  'tess':'white',
                               'load':_load_geo, 'filt':lambda u:u[1]})]],
    'label', [
        # cortex labels:
        'lh.cortex.label',           ({'hemi':'lh',  'label':'cortex', 'load':_load_label},
                                      {'hemi':'rhx', 'label':'cortex', 'load':_load_label}),
        'rh.cortex.label',           ({'hemi':'rh',  'label':'cortex', 'load':_load_label},
                                      {'hemi':'lhx', 'label':'cortex', 'load':_load_label}),
        # the annotation files:
        'lh.BA_exvivo.annot',        ({'hemi':'lh',  'annot':'brodmann_area_wide',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'brodmann_area_wide',
                                       'load':_load_annot}),
        'lh.BA_exvivo.thresh.annot', ({'hemi':'lh',  'annot':'brodmann_area', 'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'brodmann_area', 'load':_load_annot}),
        'lh.BA.annot',               ({'hemi':'lh',  'alt_annot':'brodmann_area_wide',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'alt_annot':'brodmann_area_wide',
                                       'load':_load_annot}),
        'lh.BA.thresh.annot',        ({'hemi':'lh',  'alt_annot':'brodmann_area',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'alt_annot':'brodmann_area',
                                       'load':_load_annot}),
        'lh.aparc.annot',            ({'hemi':'lh',  'annot':'Desikan06_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'Desikan06_parcellation',
                                       'load':_load_annot}),
        'lh.aparc.a2009s.annot',     ({'hemi':'lh',  'annot':'Destrieux09_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'lh',  'annot':'parcellation', 'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'Destrieux09_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'parcellation', 'load':_load_annot}),
        'lh.aparc.DKTatlas.annot',   ({'hemi':'lh',  'annot':'DKT40_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'DKT40_parcellation',
                                       'load':_load_annot}),
        'lh.BA_exvivo.annot',        ({'hemi':'lh',  'annot':'brodmann_area_wide',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'brodmann_area_wide',
                                       'load':_load_annot}),
        'lh.BA_exvivo.thresh.annot', ({'hemi':'lh',  'annot':'brodmann_area', 'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'brodmann_area', 'load':_load_annot}),
        'lh.BA.annot',               ({'hemi':'lh',  'alt_annot':'brodmann_area_wide',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'alt_annot':'brodmann_area_wide',
                                       'load':_load_annot}),
        'lh.BA.thresh.annot',        ({'hemi':'lh',  'alt_annot':'brodmann_area',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'alt_annot':'brodmann_area',
                                       'load':_load_annot}),
        'lh.aparc.annot',            ({'hemi':'lh',  'annot':'Desikan06_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'Desikan06_parcellation',
                                       'load':_load_annot}),
        'lh.aparc.a2009s.annot',     ({'hemi':'lh', 'annot':'Destrieux09_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'lh', 'annot':'parcellation', 'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'Destrieux09_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'parcellation', 'load':_load_annot}),
        'lh.aparc.DKTatlas.annot',   ({'hemi':'lh',  'annot':'DKT40_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'rhx', 'annot':'DKT40_parcellation',
                                       'load':_load_annot}),
        'rh.BA_exvivo.annot',        ({'hemi':'rh',  'annot':'brodmann_area_wide',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'brodmann_area_wide',
                                       'load':_load_annot}),
        'rh.BA_exvivo.thresh.annot', ({'hemi':'rh',  'annot':'brodmann_area', 'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'brodmann_area', 'load':_load_annot}),
        'rh.BA.annot',               ({'hemi':'rh',  'alt_annot':'brodmann_area_wide',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'alt_annot':'brodmann_area_wide',
                                       'load':_load_annot}),
        'rh.BA.thresh.annot',        ({'hemi':'rh',  'alt_annot':'brodmann_area',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'alt_annot':'brodmann_area',
                                       'load':_load_annot}),
        'rh.aparc.annot',            ({'hemi':'rh',  'annot':'Desikan06_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'Desikan06_parcellation',
                                       'load':_load_annot}),
        'rh.aparc.a2009s.annot',     ({'hemi':'rh',  'annot':'Destrieux09_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'rh',  'annot':'parcellation', 'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'Destrieux09_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'parcellation', 'load':_load_annot}),
        'rh.aparc.DKTatlas.annot',   ({'hemi':'rh',  'annot':'DKT40_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'DKT40_parcellation',
                                       'load':_load_annot}),
        'rh.BA_exvivo.annot',        ({'hemi':'rh',  'annot':'brodmann_area_wide',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'brodmann_area_wide',
                                       'load':_load_annot}),
        'rh.BA_exvivo.thresh.annot', ({'hemi':'rh',  'annot':'brodmann_area', 'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'brodmann_area', 'load':_load_annot}),
        'rh.BA.annot',               ({'hemi':'rh',  'alt_annot':'brodmann_area_wide',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'alt_annot':'brodmann_area_wide',
                                       'load':_load_annot}),
        'rh.BA.thresh.annot',        ({'hemi':'rh',  'alt_annot':'brodmann_area',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'alt_annot':'brodmann_area',
                                       'load':_load_annot}),
        'rh.aparc.annot',            ({'hemi':'rh',  'annot':'Desikan06_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'Desikan06_parcellation',
                                       'load':_load_annot}),
        'rh.aparc.a2009s.annot',     ({'hemi':'rh', 'annot':'Destrieux09_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'rh', 'annot':'parcellation', 'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'Destrieux09_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'parcellation', 'load':_load_annot}),
        'rh.aparc.DKTatlas.annot',   ({'hemi':'rh',  'annot':'DKT40_parcellation',
                                       'load':_load_annot},
                                      {'hemi':'lhx', 'annot':'DKT40_parcellation',
                                       'load':_load_annot})
    ] + [
        x
        for (h,hx) in zip(['lh','rh'],['rhx','lhx']) for p in (True, False) for alt in (True,False)
        for s in ['BA1', 'BA2', 'BA3a', 'BA3b', 'BA44', 'BA45', 'BA4a', 'BA4p', 'BA6',
                  'MT', 'V1', 'V2', 'entorhinal', 'perirhinal']
        for x in ['%s.%s%s.%slabel' % (h, s, '' if alt else '_exvivo', '' if p else 'thresh.'),
                  ({'hemi':h, (('alt_' if alt else '') + ('weight' if p else 'label')):s,
                    'load':_load_label},
                  {'hemi':hx, (('alt_' if alt else '') + ('weight' if p else 'label')):s,
                    'load':_load_label})]]]
def subject_file_map(path):
    '''
    subject_file_map(path) yields a filemap object for the given freesurfer subject path.
    '''
    return file_map(path, freesurfer_subject_filemap_instructions,
                    data_hierarchy=freesurfer_subject_data_hierarchy)
_registration_aliases = {'benson14_retinotopy.v4_0': ['benson17', 'benson17_retinotopy']}
def cortex_from_filemap(fmap, chirality, name, subid=None, affine=None):
    '''
    cortex_from_filemap(filemap, chirality, name) yields a cortex object from the given filemap;
      if the chirality and name do not match, then the result must be an xhemi.
    '''
    chirality = chirality.lower()
    name = name.lower()
    # get the relevant hemi-data
    hdat = fmap.data_tree.hemi[name]
    # we need the tesselation at build-time, so let's create that now:
    tris = hdat.tess['white']
    # this tells us the max number of vertices
    n = np.max(tris) + 1
    # Properties: we want to merge a bunch of things together...
    # for labels, weights, annots, we need to worry about loading alts:
    def _load_with_alt(k, s0, sa, trfn):
        try: u = s0.get(k, None)
        except Exception: u = None
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
    l  = hdat.label     if hasattr(hdat, 'label')     else {}
    al = hdat.alt_label if hasattr(hdat, 'alt_label') else {}
    for k in set(chain(six.iterkeys(l), six.iterkeys(al))):
        p[k+'_label'] = curry(_load_with_alt, k, l, al, _lbltr)
    w  = hdat.weight     if hasattr(hdat, 'weight')     else {}
    aw = hdat.alt_weight if hasattr(hdat, 'alt_weight') else {}
    for k in set(chain(six.iterkeys(w), six.iterkeys(aw))):
        p[k+'_weight'] = curry(_load_with_alt, k, w, aw, _wgttr)
    a  = hdat.annot     if hasattr(hdat, 'annot')     else {}
    aa = hdat.alt_annot if hasattr(hdat, 'alt_annot') else {}
    for k in set(chain(six.iterkeys(a), six.iterkeys(aa))):
        p[k] = curry(_load_with_alt, k, a, aa, _anotr)
    props = pimms.merge(hdat.property, pimms.lazy_map(p))
    tess = geo.Tesselation(tris, properties=props)
    def _make_midgray():
        x = np.mean([hdat.surface['white'], hdat.surface['pial']], axis=0)
        x.setflags(write=False)
        return x
    surfs = hdat.surface.set('midgray', _make_midgray)
    # if this is a subject that exists in the library, we may want to add some files:
    if subid is None:
        pd = fmap.pseudo_paths[None]._path_data
        subid = pd['pathmod'].split(fmap.actual_path)[1]
    regs = hdat.registration
    extra_path = os.path.join(library_freesurfer_subjects(), subid, 'surf')
    if os.path.isdir(extra_path):
        for flnm in os.listdir(extra_path):
            if flnm.startswith(chirality + '.') and flnm.endswith('.sphere.reg'):
                mid = flnm[(len(chirality)+1):-11]
                if mid == '': mid = ['fsaverage']
                else:         mid = _registration_aliases.get(mid, [mid])
                load_mid_fn = curry(_load_surfx, os.path.join(extra_path, flnm), {})
                for mid in mid:
                    f = load_mid_fn if mid not in regs else \
                        curry(try_until,
                              curry(lambda regs: regs[mid], regs), load_mid_fn,
                              check=pimms.is_matrix)
                    regs = regs.set(mid, f)

    # Okay, make the cortex object!
    md = {'file_map': fmap}
    if subid is not None: md['subject_id'] = subid
    return mri.Cortex(chirality, tess, surfs, regs, affine=affine, meta_data=md).persist()
def images_from_filemap(fmap):
    '''
    images_from_filemap(fmap) yields a persistent map of MRImages tracked by the given subject with
      the given name and path; in freesurfer subjects these are renamed and converted from their
      typical freesurfer filenames (such as 'ribbon') to forms that conform to the neuropythy naming
      conventions (such as 'gray_mask'). To access data by their original names, use the filemap.
    '''
    ims = {}
    raw_images = fmap.data_tree.raw_image
    def _make_imm_mask(rib, val, eq=True):
        arr = np.asarray(rib.dataobj)
        arr = (arr == val) if eq else (arr != val)
        arr.setflags(write=False)
        return fsmgh.MGHImage(arr, rib.affine, rib.header)
    # start with the ribbon:
    ims['lh_gray_mask']  = lambda:_make_imm_mask(raw_images['ribbon'], 3)
    ims['lh_white_mask'] = lambda:_make_imm_mask(raw_images['ribbon'], 2)
    ims['rh_gray_mask']  = lambda:_make_imm_mask(raw_images['ribbon'], 42)
    ims['rh_white_mask'] = lambda:_make_imm_mask(raw_images['ribbon'], 41)
    ims['brain_mask']    = lambda:_make_imm_mask(raw_images['ribbon'], 0, False)
    # merge in with the typical images
    return pimms.merge(fmap.data_tree.image, pimms.lazy_map(ims))
def subject_from_filemap(fmap, name=None, meta_data=None, check_path=True):
    # start by making a pseudo-dir:
    if check_path and not is_freesurfer_subject_path(fmap.pseudo_paths[None]):
        raise ValueError('given path does not appear to hold a freesurfer subject')
    # we need to go ahead and load the ribbon...
    rawims = fmap.data_tree.raw_image
    rib = next(rawims[k]
               for k in ['ribbon', 'brain', 'intensity_normalized', 'conformed']
               if k in rawims and rawims[k] is not None)
    vox2nat = rib.affine
    vox2vtx = rib.header.get_vox2ras_tkr()
    vtx2nat = np.dot(vox2nat, np.linalg.inv(vox2vtx))
    # make images and hems
    imgs = images_from_filemap(fmap)
    hems = pimms.lazy_map({h:curry(cortex_from_filemap, fmap, h, ch, name, vtx2nat)
                           for (h,ch) in zip(['lh','rhx','rh','lhx'], ['lh','lh','rh','rh'])})
    meta_data = pimms.persist({} if meta_data is None else meta_data)
    meta_data = meta_data.set('raw_images', fmap.data_tree.raw_image)
    return mri.Subject(name=name, pseudo_path=fmap.pseudo_paths[None],
                       hemis=hems, images=imgs,
                       meta_data=meta_data)
@nyio.importer('freesurfer_subject', sniff=is_freesurfer_subject_path)
def subject(path, name=Ellipsis, meta_data=None, check_path=True, filter=None):
    '''
    subject(name) yields a freesurfer Subject object for the subject with the given name. Subjects
      are cached and not reloaded, so multiple calls to subject(name) will yield the same immutable
      subject object.

    Note that subects returned by freesurfer_subject() are always persistent Immutable objects; this
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
        FreeSurfer subject directory. Subject objects returned using this argument are not cached.
        Additionally, check_path may be set to None instead of False, indicating that no checks or
        search should be performed; the string name should be trusted to be an exact relative or
        absolute path to a valid FreeSurfer subejct.
      * filter (default: None) may optionally specify a filter that should be applied to the subject
        before returning. This must be a function that accepts as an argument the subject object and
        returns a (potentially) modified subject object. Filtered subjects are cached by the id
        of the filters.
    '''
    # convert the path to a pseudo-dir; this may fail if the user is requesting a subject by name...
    try: pdir = to_pseudo_path(path)
    except Exception: pdir = None
    if pdir is None: # search for a subject with this name
        tmp = find_subject_path(path, check_path=check_path)
        if tmp is not None:
            pdir = to_pseudo_path(tmp)
            path = tmp
        elif path == 'fsaverage':
            try:
                import neuropythy as ny
                return ny.data['benson_winawer_2018'].subjects['fsaverage']
            except: pass
    if pdir is None: raise ValueError('could not find freesurfer subject: %s' % path)
    path = pdir.source_path
    # okay, before we continue, lets check the cache...
    if path in subject._cache: sub = subject._cache[path]
    else:
        # make the filemap
        fmap = subject_file_map(pdir)
        # extract the name if need-be
        if name is Ellipsis:
            import re
            (pth,name) = (path, '.')
            while name == '.': (pth, name) = pdir._path_data['pathmod'].split(pth)
            name = name.split(':')[-1]
            name = pdir._path_data['pathmod'].split(name)[1]
            if '.tar' in name: name = name.split('.tar')[0]
        # and make the subject!
        sub = subject_from_filemap(fmap, name=name, check_path=check_path, meta_data=meta_data)
        if mri.is_subject(sub):
            sub.persist()
            sub = sub.with_meta(file_map=fmap)
            subject._cache[path] = sub
    # okay, we have the initial subject; let's organize the filters
    if pimms.is_list(subject.filter) or pimms.is_tuple(subject.filter): filts = list(subject.filter)
    else: filts = []
    if pimms.is_list(filter) or pimms.is_tuple(filter): filter = list(filter)
    else: filter = []
    filts = filts + filter
    if len(filts) == 0: return sub
    fids = tuple([id(f) for f in filts])
    tup = fids + (path,)
    if tup in subject._cache: return subject._cache[tup]
    for f in filts: sub = f(sub)
    if mri.is_subject(sub): subject._cache[tup] = sub
    return sub
subject._cache = {}
subject.filter = None
def forget_subject(sid):
    '''
    forget_subject(sid) causes neuropythy's freesurfer module to forget about cached data for the
      subject with subject id sid. The sid may be any sid that can be passed to the subject()
      function.

    This function is useful for batch-processing of subjects in a memory-limited environment; e.g.,
    if you run out of memory while processing FreeSurfer subjects it is possibly because neuropythy
    is caching all of their data instead of freeing it.
    '''
    sub = subject(sid)
    if sub.path in subject._cache:
        del subject._cache[sub.path]
    for (k,v) in six.iteritems(subject._cache):
        if pimms.is_tuple(k) and k[-1] == sub.path:
            del subject._cache[k]
    return None
def forget_all():
    '''
    forget_all() causes neuropythy's freesurfer module to forget all cached subjects. See also
    forget_subject.
    '''
    subject._cache = {}
    return None

# This function creates the tkr matrix for a volume given the dims
def tkr_vox2ras(img, zooms=None):
    '''
    tkr_vox2ras(img) yields the FreeSurfer tkr VOX2RAS matrix for the given nibabel image object
      img. The img must have a get_shape() method and header member with a get_zooms() method.
    tkr_vox2ras(hdr) operates on a nibabel image header object.
    tkr_vox2ras(shape, zooms) operates on the shape (e.g., for FreeSurfer subjects (256,256,256))
      and the zooms or voxel dimensions (e.g., for FreeSurfer subjects, (1.0, 1.0, 1.0)).
    '''
    if zooms is not None:
        # let's assume that they passed shape, zooms
        shape = img
    else:
        try:              img = img.header
        except Exception: pass
        try:              (shape, zooms) = (img.get_data_shape(), img.get_zooms())
        except Exception: raise ValueError('single argument must be nibabel image or header')
    # Okay, we have shape and zooms...
    zooms = zooms[0:3]
    shape = shape[0:3]
    (dC, dR, dS) = zooms
    (nC, nR, nS) = 0.5 * (np.asarray(shape) * zooms)
    return np.asarray([[-dC,   0,   0,  nC],
                       [  0,   0,  dS, -nS],
                       [  0, -dR,   0,  nR],
                       [  0,   0,   0,   1]])

class SubjectDir(collections.Mapping):
    '''
    SubjectsDir objects are dictionary-like objects that represent a particular subject directory.
    They satisfy their queries (such as `'bert' in spath`) by querying the filesystem itself.

    For more information see the subjects_path function.
    '''
    def __init__(self, path, bids=False, filter=None, meta_data=None, check_path=True):
        path = os.path.expanduser(os.path.expandvars(path))
        self.path = os.path.abspath(path)
        if not os.path.isdir(self.path): raise ValueError('given path is not a directory')
        self.bids = bool(bids)
        self.options = dict(filter=filter, meta_data=meta_data, check_path=bool(check_path))
    def __contains__(self, sub):
        # first check the item straight-up:
        sd = os.path.join(self.path, sub)
        if os.path.isdir(sd) and (not check_path or is_freesurfer_subject_path(sd)): return True
        if not bids: return False
        if sub.startswith('sub-'): sd = os.path.join(self.path, sub[4:])
        else: sd = os.path.join(self.path, 'sub-' + sub)
        return os.path.isdir(sd) and (not check_path or is_freesurfer_subject_path(sd))
    def _get_subject(self, sd, name):
        opts = dict(**self.options)
        opts['name'] = name
        return subject(sd, **opts)
    def __getitem__(self, sub):
        check_path = self.options['check_path']
        if self.bids:
            if sub.startswith('sub-'): (sub, name) = (sub, name[4:])
            else: (sub,name) = ('sub-' + sub, sub)
        else: name = sub
        sd = os.path.join(self.path, sub)
        if os.path.isdir(sd) and (not check_path or is_freesurfer_subject_path(sd)):
            return self._get_subject(sd, name)
        if not bids: return False
        # try without the 'sub-' (e.g. for fsaverage)
        sub = name
        sd = os.path.join(self.path, sub)
        if os.path.isdir(sd) and (not check_path or is_freesurfer_subject_path(sd)):
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
            if not check_path or is_freesurfer_subject_path(sdir):
                res[name] = curry(self._get_subject, sdir, name)
        return pimms.lmap(res)
    def __len__(self): return len(self.asmap())
    def __iter__(self): return iter(self.asmap())
    def __repr__(self): return 'freesurfer.SubjectDir(' + repr(self.asmap()) + ')'
    def __str__(self): return 'freesurfer.SubjectDir(' + str(self.asmap()) + ')'
# Functions for handling freesurfer subject directories themselves
def is_freesurfer_subject_dir_path(path):
    '''
    is_freesurfer_subject_dir_path(path) yields True if path is a directory that contains at least
      one FreeSurfer subejct subdirectory.
    '''
    if not os.path.isdir(path): return False
    for p in os.listdir(path):
        pp = os.path.join(path, p)
        if not os.path.isdir(pp): continue
        if is_freesurfer_subject_path(pp): return True
    return False
@nyio.importer('freesurfer_subject_dir', sniff=is_freesurfer_subject_dir_path)
def subject_dir(path, bids=False, filter=None, meta_data=None, check_path=True):
    '''
    subject_dir(path yields a dictionary-like object containing the subjects in the FreeSurfer
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

####################################################################################################
# import/export code that works with neuropythy.io

# MGH Images!
@nyio.importer('mgh', ('mgh', 'mgh.gz', 'mgz'))
def load_mgh(filename, to='auto'):
    '''
    load_mgh(filename) yields the MGHImage referened by the given filename by using the
      nibabel.freesurfer.mghformat.load function.
    
    The optional argument 'to' may be used to coerce the resulting data to a particular format; the
    following arguments are understood:
      * 'header' will yield just the image header
      * 'data' will yield the image's data-array
      * 'field' will yield a squeezed version of the image's data-array and will raise an error if
        the data object has more than 2 non-unitary dimensions (appropriate for loading surface
        properties stored in image files)
      * 'affine' will yield the image's affine transformation
      * 'image' will yield the raw image object
      * 'auto' is equivalent to 'image' unless the image has no more than 2 non-unitary dimensions,
        in which case it is assumed to be a surface-field and the return value is equivalent to
        the 'field' value.
    '''
    img = fsmgh.load(filename)
    to = to.lower()
    if to == 'image':    return img
    elif to == 'data':   return img.dataobj
    elif to == 'affine': return img.affine
    elif to == 'header': return img.header
    elif to == 'field':
        dat = np.squeeze(img.dataobj)
        if len(dat.shape) > 2:
            raise ValueError('image requested as field has more than 2 non-unitary dimensions')
        return dat
    elif to in ['auto', 'automatic']:
        dims = set(img.dataobj.shape)
        if 1 < len(dims) < 4 and 1 in dims:
            return np.squeeze(img.dataobj)
        else:
            return img
    else:
        raise ValueError('unrecognized \'to\' argument \'%s\'' % to)
def to_mgh(obj, like=None, header=None, affine=None, extra=Ellipsis):
    '''
    to_mgh(obj) yields an MGHmage object that is as equivalent as possible to the given object obj.
      If obj is an MGHImage already and no changes are requested, then it is returned unmolested;
      otherwise, the optional arguments can be used to edit the header, affine, and exta.

    The following options are accepted:
      * like (default: None) may be provided to give a guide for the various header- and meta-data
        that is included in the image. If this is a nifti image object, its meta-data are used; if
        this is a subject, then the meta-data are deduced from the subject's voxel and native
        orientation matrices. All other specific options below override anything deduced from the
        like argument.
      * header (default: None) may be a Nifti1 or Niti2 image header to be used as the nifti header
        or to replace the header in a new image.
      * affine (default: None) may specify the affine transform to be given to the image object.
      * extra (default: Ellipsis) may specify an extra mapping for storage with the image data; the
        default value Ellipsis indicates that such a mapping should be left as is if it already
        exists in the obj or in the like option and should otherwise be empty.
    '''
    obj0 = obj
    # First go from like to explicit versions of affine and header:
    if like is not None:
        if isinstance(like, nib.analyze.AnalyzeHeader) or isinstance(obj, fsmgh.MGHHeader):
            if header is None: header = like
        elif isinstance(like, nib.analyze.SpatialImage):
            if header is None: header = like.header
            if affine is None: affine = like.affine
            if extra is Ellipsis: extra = like.extra
        elif isinstance(like, mri.Subject):
            if affine is None: affine = like.images['brain'].affine
        else:
            raise ValueError('Could not interpret like argument with type %s' % type(like))
    # check to make sure that we have to change something:
    elif (isinstance(obj, fsmgh.MGHImage)):
        if ((header is None or obj.header is header) and
            (extra is Ellipsis or extra == obj.extra or (extra is None and len(obj.extra) == 0))):
            return obj
    # okay, now look at the header and affine etc.
    if header is None:
        if isinstance(obj, nib.analyze.SpatialImage):
            header = obj.header
        else:
            header = None
    if affine is None:
        if isinstance(obj, nib.analyze.SpatialImage):
            affine = obj.affine
        else:
            affine = np.eye(4)
    if extra is None: extra = {}
    # Figure out what the data is
    if isinstance(obj, nib.analyze.SpatialImage):
        obj = obj.dataobj
    else:
        obj = np.asarray(obj)
    if len(obj.shape) < 3: obj = np.asarray([[obj]])
    # make sure the dtype isn't a high-bit integer or float
    if np.issubdtype(obj.dtype, np.integer) and obj.dtype.itemsize > 4:
        obj = np.asarray(obj, dtype=np.int32)
    elif np.issubdtype(obj.dtype, np.floating) and obj.dtype.itemsize > 4:
        obj = np.asarray(obj, dtype=np.float32)
    # Okay, make a new object now...
    obj = fsmgh.MGHImage(obj, affine, header=header, extra=extra)
    # Okay, that's it!
    return obj
@nyio.exporter('mgh', ('mgh', 'mgh.gz', 'mgz'))
def save_mgh(filename, obj, like=None, header=None, affine=None, extra=Ellipsis):
    '''
    save_mgh(filename, obj) saves the given object to the given filename in the mgh format and
      returns the filename.

    All options that can be given to the to_mgh function can also be passed to this function; they
    are used to modify the object prior to exporting it.
    '''
    obj = to_mgh(obj, like=like, header=header, affine=affine, extra=extra)
    obj.to_filename(filename)
    return filename

# For other freesurfer formats
@nyio.importer('freesurfer_geometry', ('white', 'pial', 'sphere', 'sphere.reg', 'inflated'))
def load_freesurfer_geometry(filename, to='mesh', warn=False):
    '''
    load_freesurfer_geometry(filename) yields the data stored at the freesurfer geometry file given
      by filename. The optional argument 'to' may be used to change the kind of data that is
      returned.

    The following are valid settings for the 'to' keyword argument:
      * 'mesh' (the default) yields a mesh object
      * 'tess' yields a tess object (discarding coordinates)
      * 'raw' yields a tuple of numpy arrays, identical to the read_geometry return value.
    '''
    if not warn:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    category=UserWarning,
                                    module='nibabel')
            (xs, fs, info) = fsio.read_geometry(filename, read_metadata=True)
    else:
        (xs, fs, info) = fsio.read_geometry(filename, read_metadata=True)
    # see if there's chirality data here...
    filename = os.path.split(filename)[1]
    filename = filename.lower()
    if   filename.startswith('lh'): info['chirality'] = 'lh.'
    elif filename.startswith('rh'): info['chirality'] = 'rh.'
    # parse it into something
    to = to.lower()
    if to in ['mesh', 'auto', 'automatic']:
        return geo.Mesh(fs, xs, meta_data=info)
    elif to in ['tess', 'tesselation']:
        return geo.Tesselation(fs, meta_data=info)
    elif to in ['coords', 'coordinates']:
        return xs
    elif to in ['triangles', 'faces']:
        return fs
    elif to in ['meta', 'meta_data']:
        return info
    elif to =='raw':
        return (xs, fs)
    else:
        raise ValueError('Could not understand \'to\' argument: %s' % to)
@nyio.exporter('freesurfer_geometry', ('sphere.reg',))
def save_freesurfer_geometry(filename, obj, volume_info=None, create_stamp=None):
    '''
    save_mgh(filename, obj) saves the given object to the given filename in the mgh format and
      returns the filename.

    All options that can be given to the to_mgh function can also be passed to this function; they
    are used to modify the object prior to exporting it.
    '''
    obj = geo.to_mesh(obj)
    fsio.write_geometry(filename, obj.coordinates.T, obj.tess.faces.T,
                        volume_info=volume_info, create_stamp=create_stamp)
    return filename
@nyio.importer('freesurfer_morph', ('curv',))
def load_freesurfer_morph(filename):
    '''
    load_freesurfer_morph(filename) yields the result of loading the given filename as FreeSurfer
      morph-data (e.g., lh.curv).
    '''
    return fsio.read_morph_data(filename)
@nyio.exporter('freesurfer_morph', ('curv',))
def save_freesurfer_morph(filename, obj, face_count=0):
    '''
    save_freesurfer_morph(filename, obj) saves the given object using nibabel.freesurfer.io's
      write_morph_data function, and returns the given filename.
    '''
    fsio.write_morph_data(filename, obj, fnum=face_count)
    return filename
@nyio.importer('freesurfer_label', ('label',))
def load_freesurfer_label(filename, to='all'):
    '''
    load_freesurfer_label(filename) yields the boolean label property found in the given freesurfer
      label file.

    The return value is determined by the option `to`, which by default is 'all'. The following
    values may be given:
      * 'vertices' is equivalent to nibabel.freesurfer.io.read_label(filename) as is.
      * 'all' is equivalent to nibabel.freesurfer.io.read_label(filename, read_scalars=True) as is.
    '''
    if to in [None,Ellipsis]: to = 'all'
    to = to.lower()
    if to == 'vertices': fsio.read_label(filename, read_scalars=False)
    (ls,ps) = fsio.read_label(filename, read_scalars=True)
    if to == 'all': return (ls,ps)
    else: raise ValueError('could not parse to option: %s' % to)    
@nyio.importer('freesurfer_annot', ('annot',))
def load_freesurfer_annot(filename, to='property'):
    '''
    load_freesurfer_annot(filename) yields the result of loading an annotation file given by the
      provided filename.

    The precise return value is determined by the optional argument `to`; it is 'property' by
    default but may be set to any of the following:
      * 'raw': nibabel.freesurfer.io.read_annot(filename, orig_ids=False)
      * 'raw_orig': nibabel.freesurfer.io.read_annot(filename, orig_ids=True)
      * 'property': returns the label ids found in the file as a property array.
      * 'index': returns the label_index object stored in the annotation file.
      * 'all': returns (property, index).
    '''
    if to in [None,Ellipsis]: to = 'property'
    to = to.lower()
    if to == 'raw_orig': return fsio.read_annot(filename, orig_ids=True)
    dat = fsio.read_annot(filename, orig_ids=False)
    if to == 'raw': return dat
    (ls,clrs,nms) = dat
    nms = np.asarray([nm.decode('utf-8') for nm in nms])
    lbls = clrs[:,-1]
    clrs = clrs[:,:4]
    clrs[:,3] = 255 - clrs[:,3]
    clrs = clrs / 255.0
    oks = ls >= 0
    p = np.array(ls)
    p[~oks] = 0
    lils = np.arange(len(nms))
    if to in ['property', 'prop', 'p', 'auto', 'automatic']: return p
    elif to in ['index', 'label_index', 'lblidx', 'idx']: return label_index(lils, nms, clrs)
    elif to in ['all', 'full']: return (p, label_index(lils, nms, clrs))
    else: raise ValueError('bad to conversion: %s' % to)
@nyio.exporter('freesurfer_annot', ('annot',))
def save_freesurfer_annot(filename, obj, index=None):
    '''
    save_freesurfer_annot(filename, prop) saves the given integer property prop to the given
      filename as a FreeSurfer annotation file.

    The optional argument index specifies how the colortab and names of the property labels should
    be handles. By default this is None, in which case names are generated using the formatter
    'label%d' and color values are generated using the label_colors function. If index is not None,
    then it is coerced to an index using the to_label_index() function, and the label index is
    used to create the colortable and names.
    '''
    # first parse the index
    if index is None:
        if len(obj) == 2 and pimms.is_vector(obj[0]): (obj, index) = obj
        else: index = label_index(obj)
    index = to_label_index(index)
    # okay, let's get the data in the right format
    (u,ris) = np.unique(obj, return_inverse=True)
    es = [index[l] for l in u]
    clrs = np.round([np.asarray(e.color)*255 for e in es]).astype('int')
    clrs[:,3] = 255 - clrs[:,3] # alpha -> transparency
    nms  = [e.name for e in es]
    lbls = [e.id for e in es]
    fsio.write_annot(filename, ris, clrs, nms, fill_ctab=True)
    return filename

# A few annot labels we can just save:
brodmann_label_index = label_index(
    np.arange(15),
    ['none', 'BA1', 'BA2', 'BA3a', 'BA3b', 'BA4a', 'BA4p', 'BA6', 'BA44', 'BA45', 'V1', 'V2', 'MT',
     'perirhinal', 'enterorhinal'])
label_indices['freesurfer_brodmann'] = brodmann_label_index.persist()
