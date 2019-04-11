####################################################################################################
# neuropythy/freesurfer/core.py
# Simple tools for use with FreeSurfer in Python
# By Noah C. Benson

import numpy                        as np
import nibabel                      as nib
import nibabel.freesurfer.io        as fsio
import nibabel.freesurfer.mghformat as fsmgh
import pyrsistent                   as pyr
import os, warnings, six, pimms

from .. import geometry as geo
from .. import mri      as mri
from .. import io       as nyio
#import ..io as nyio

from ..util import (config, library_path)

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
config.declare('freesurfer_subject_paths', environ_name='SUBJECTS_DIR', filter=to_subject_paths)

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
if 'FREESURFER_HOME' in os.environ:
    add_subject_path(os.path.join(os.environ['FREESURFER_HOME'], 'subjects'), None)

def is_freesurfer_subject_path(path):
    '''
    is_freesurfer_subject_path(path) yields True if the given path appears to be a valid freesurfer
      subject path and False otherwise.
    A path is considered to be freesurfer-subject-like if it contains the directories mri/, surf/,
    and label/.
    '''
    if not os.path.isdir(path): return False
    else: return all(os.path.isdir(os.path.join(path, d)) for d in ['mri', 'surf', 'label'])

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
    if is_freesurfer_subject_path(sub): return sub
    path = next((p for sd in sdirs for p in [os.path.join(sd, sub)]
                 if is_freesurfer_subject_path(p)),
                None)
    if path is not None: return path
    if check_path: return None
    if os.path.isdir(sub): return sub
    return next((p for sd in sdirs for p in [os.path.join(sd, sub)]
                 if os.path.isdir(p)),
                sub)

# Used to load immutable-like mgh objects
def _load_imm_mgh(flnm):
    img = fsmgh.load(flnm)
    img.get_data().setflags(write=False)
    return img

# A freesurfer.Subject is much like an mri.Subject, but its dependency structure all comes from the
# path rather than data provided to the constructor:
@pimms.immutable
class Subject(mri.Subject):
    '''
    A neuropythy.freesurfer.Subject is an instance of neuropythy.mri.Subject that depends only on
    the path of the subject represented; all other data are automatically derived from this.
    '''
    def __init__(self, path, meta_data=None, check_path=True):
        if check_path and not is_freesurfer_subject_path(path):
            raise ValueError('given path does not appear to hold a freesurfer subject')
        # get the name...
        path = os.path.abspath(path)
        name = os.path.split(path)[-1]
        self.name = name
        self.path = path
        self.meta_data = meta_data
        self.hemis = Subject.load_hemis(name, path)
        self.images = Subject.load_images(name, path)
        # these are the only actually required data for the constructor; the rest is values

    # This [private] function and this variable set up automatic properties from the FS directory
    # in order to be auto-loaded, a property must appear in this dictionary:
    _auto_retino_names = pyr.pmap({
        (tag + sep + name): (ptag + pname) for
        (tag,ptag) in [('', ''),
                       ('rf',         'rf_'        ), ('prf',        'prf_'       ),
                       ('meas',       'measured_'  ), ('measured',   'measured_'  ),
                       ('emp',        'empirical_' ), ('empirical',  'empirical_' ),
                       ('trn',        'training_'  ), ('train',      'training_'  ),
                       ('training',   'training_'  ), ('val',        'validation_'),
                       ('valid',      'validation_'), ('validation', 'validation_'),
                       ('test',       'validation_'), ('gold',       'gold_'      ),
                       ('retinotopy', ''           ), ('retino',     ''           ),
                       ('predict',    'predicted_' ), ('pred',       'predicted_' ),
                       ('model',      'model_'     ), ('mdl',        'model_'     ),
                       ('inferred',   'inferred_'  ), ('bayes',      'inferred_'  ),
                       ('inf',        'inferred_'  ), ('benson14',   'benson14_'  ),
                       ('benson17',   'benson17_'  ), ('atlas',      'atlas_'     ),
                       ('template',   'template_'  )]
        for sep in (['_', '.', '-'] if len(tag) > 0 else [''])
        for (name, pname) in [
                ('eccen',  'eccentricity'      ),
                ('angle',  'polar_angle'       ),
                ('theta',  'theta'             ),
                ('rho',    'rho'               ),
                ('prfsz',  'size'              ),
                ('size',   'size'              ),
                ('radius', 'radius'            ),
                ('sigma',  'sigma'             ),
                ('varex',  'variance_explained'),
                ('vexpl',  'variance_explained'),
                ('varexp', 'variance_explained'),
                ('weight', 'weight'            ),
                ('varea',  'visual_area'       ),
                ('vsroi',  'visual_area'       ),
                ('vroi',   'visual_area'       ),
                ('vslab',  'visual_area'       )]})
    _auto_properties = pyr.pmap({k: (a, lambda f: fsio.read_morph_data(f))
                                 for d in [{'sulc':      'convexity',
                                            'thickness': 'thickness',
                                            'volume':    'volume',
                                            'area':      'white_surface_area',
                                            'area.mid':  'midgray_surface_area',
                                            'area.pial': 'pial_surface_area',
                                            'curv':      'curvature'},
                                           _auto_retino_names]
                                 for (k,a) in six.iteritems(d)})
    _registration_aliases = {'benson14_retinotopy.v4_0': 'benson17'}
    @staticmethod
    def _cortex_from_path(subid, chirality, name, surf_path, data_path, data_prefix=Ellipsis):
        '''
        Subject._cortex_from_path(subid, chirality, name, spath, dpath) yields a Cortex object
          that has been loaded from the given path. The given spath should be the path from which
          to load the structural information (like lh.sphere and rh.white) while the dpath is the
          path from which to load the non-structural information (like lh.thickness or rh.curv).
        '''
        # data prefix is ellipsis when we use the same as the chirality; unless the name ends with
        # X, in which case, it's considered a reversed-hemisphere
        chirality = chirality.lower()
        if data_prefix is Ellipsis:
            if name.lower().endswith('x'): data_prefix = 'rh' if chirality == 'lh' else 'lh'
            else:                          data_prefix = chirality
            # we can start by making a lazy-map of the auto-properties
        def _make_prop_loader(flnm):
            def _load_fn():
                p = fsio.read_morph_data(flnm)
                p.setflags(write=False)
                return p
            return _load_fn
        def _make_mghprop_loader(flnm):
            def _load_fn():
                p = fsmgh.load(flnm).get_data().flatten()
                p.setflags(write=False)
                return p
            return _load_fn
        props = {}
        for (k,(a,_)) in six.iteritems(Subject._auto_properties):
            f = os.path.join(data_path, data_prefix + '.' + k)
            if os.path.isfile(f):
                props[a] = _make_prop_loader(f)
        # That takes care of the defauly properties; now look for auto-retino properties
        for flnm in os.listdir(data_path):
            if flnm[0] == '.' or not flnm.startswith(data_prefix + '.'): continue
            if flnm.endswith('.mgz') or flnm.endswith('.mgh'):
                mid = flnm[3:-4]
                loader = _make_mghprop_loader
            else:
                mid = flnm[3:]
                loader = _make_prop_loader
            if mid in Subject._auto_retino_names:
                tr = Subject._auto_retino_names[mid]
                props[tr] = loader(os.path.join(data_path, flnm))
        props = pimms.lazy_map(props)
        # we need a tesselation in order to make the surfaces or the cortex object
        white_surf_name = os.path.join(surf_path, chirality + '.white')
        # We need the tesselation at instantiation-time, so we can load it now
        tess = geo.Tesselation(fsio.read_geometry(white_surf_name)[1], properties=props)
        # start by creating the surface file names
        def _make_surf_loader(flnm):
            def _load_fn():
                x = fsio.read_geometry(flnm)[0].T
                x.setflags(write=False)
                return x
            return _load_fn
        surfs = {}
        for s in ['white', 'pial', 'inflated', 'sphere']:
            surfs[s] = _make_surf_loader(os.path.join(surf_path, chirality + '.' + s))
        surfs = pimms.lazy_map(surfs)
        def _make_midgray():
            x = np.mean([surfs['white'], surfs['pial']], axis=0)
            x.setflags(write=False)
            return x
        surfs = surfs.set('midgray', _make_midgray)
        # okay, now we can do the same for the relevant registrations; since the sphere registration
        # is the same as the sphere surface, we can just copy that one over:
        regs = {'native': lambda:surfs['sphere']}
        # see if our subject id has special neuropythy datafiles...
        surf_paths = [surf_path]
        extra_path = os.path.join(library_path(), 'data', subid, 'surf')
        regals = Subject._registration_aliases
        if os.path.isdir(extra_path): surf_paths.insert(0, extra_path)
        for surf_path in surf_paths:
            for flnm in os.listdir(surf_path):
                if flnm.startswith(chirality + '.') and flnm.endswith('.sphere.reg'):
                    mid = flnm[(len(chirality)+1):-11]
                    if mid == '': mid = 'fsaverage'
                    elif mid in regals: mid = regals[mid]
                    regs[mid] = _make_surf_loader(os.path.join(surf_path, flnm))
        regs = pimms.lazy_map(regs)
        # great; now we can actually create the cortex object itself
        return mri.Cortex(chirality, tess, surfs, regs).persist()

    @staticmethod
    def load_hemis(name, path):
        '''
        Subject.load_hemis(name, path) yields a persistent map of hemisphere names ('lh', 'rh',
          possibly others) for the given freesurfer subject sub. Other hemispheres may include lhx
          and rhx (mirror-inverted hemisphere objects).
        '''
        surf_path = os.path.join(path, 'surf')
        # basically, we want to create a lh and rh hemisphere object with automatically-loaded
        # properties based on the above auto-property data
        ctcs = {}
        for h in ['lh', 'rh']:
            ctcs[h] = Subject._cortex_from_path(name, h, h, surf_path, surf_path)
        # we also want to check for the xhemi subject
        xpath = os.path.join(path, 'xhemi', 'surf')
        if os.path.isdir(xpath):
            for (h,xh) in zip(['lh', 'rh'], ['rhx', 'lhx']):
                ctcs[xh] = Subject._cortex_from_path(name, h, xh, xpath, surf_path)
        # that's all!
        return pimms.lazy_map(ctcs)
    @staticmethod
    def load_mgh_images(name, path):
        '''
        Subject.load_mgh_images(name, path) yields a persistent map of the MGHImage objects for all
          the valid mgz files in the relevant subject's FreeSurfer mri/ directory (where the subject
          directory is given by path).
        '''
        # These are just given their basic filenames; nothing fancy here
        path = os.path.join(path, 'mri')
        fls = [f
               for p in [path, os.path.join(path, 'orig')]
               if os.path.isdir(p)
               for fname in os.listdir(p) for f in [os.path.join(p, fname)]
               if f[0] != '.'
               if len(f) > 4 and f[-4:].lower() in ['.mgz', '.mgh']
               if os.path.isfile(f)]
        def _make_loader(fname):
            def _loader():
                return _load_imm_mgh(fname)
            return _loader
        return pimms.lazy_map({os.path.split(flnm)[-1][:-4]: _make_loader(flnm) for flnm in fls})
    @staticmethod
    def load_images(name, path):
        '''
        Subject.load_images(name, path) yields a persistent map of MRImages tracked by the given
          subject sub; in freesurfer subjects these are renamed and converted from their typical
          freesurfer filenames (such as 'ribbon') to forms that conform to the neuropythy naming
          conventions (such as 'gray_mask'). To access data by their original names, use the
          Subject.load_mgh_images() function.
        '''
        ims = {}
        mgh_images = Subject.load_mgh_images(name, path)
        def _make_imm_mask(arr, val, eq=True):
            arr = (arr == val) if eq else (arr != val)
            arr.setflags(write=False)
            return fsmgh.MGHImage(arr, mgh_images['ribbon'].affine, mgh_images['ribbon'].header)
        # start with the ribbon:
        ims['lh_gray_mask']  = lambda:_make_imm_mask(mgh_images['ribbon'].get_data(), 3)
        ims['lh_white_mask'] = lambda:_make_imm_mask(mgh_images['ribbon'].get_data(), 2)
        ims['rh_gray_mask']  = lambda:_make_imm_mask(mgh_images['ribbon'].get_data(), 42)
        ims['rh_white_mask'] = lambda:_make_imm_mask(mgh_images['ribbon'].get_data(), 41)
        ims['brain_mask']    = lambda:_make_imm_mask(mgh_images['ribbon'].get_data(), 0, False)
        # next, do the standard ones:
        def _make_accessor(nm): return lambda:mgh_images[nm]
        for (tname, name) in zip(['original', 'normalized', 'segmentation', 'brain'],
                                 ['orig',     'norm',       'aseg',         'brain']):
            ims[tname] = _make_accessor(name)
        # last, check for auto-retino-properties:
        for k in six.iterkeys(mgh_images):
            if k in Subject._auto_retino_names:
                tr = Subject._auto_retino_names[k]
                ims[tr] = _make_accessor(k)
        return pimms.lazy_map(ims)
    @pimms.value
    def mgh_images(name, path):
        '''
        sub.mgh_images is a persistent map of MRImages, represented as MGHImage objects, tracked by
        the given FreeSurfer subject sub.
        '''
        return Subject.load_mgh_images(name, path)
    @pimms.value
    def voxel_to_vertex_matrix(mgh_images):
        '''
        See neuropythy.mri.Subject.voxel_to_vertex_matrix.
        '''
        return pimms.imm_array(mgh_images['ribbon'].header.get_vox2ras_tkr())
    @pimms.value
    def voxel_to_native_matrix(mgh_images):
        '''
        See neuropythy.mri.Subject.voxel_to_native_matrix.
        '''
        return pimms.imm_array(mgh_images['ribbon'].header.get_affine())

@nyio.importer('freesurfer_subject', sniff=is_freesurfer_subject_path)
def subject(name, meta_data=None, check_path=True):
    '''
    subject(name) yields a freesurfer Subject object for the subject with the given name. Subjects
      are cached and not reloaded, so multiple calls to subject(name) will yield the same immutable
      subject object..

    Note that subects returned by freesurfer_subject() are always persistent Immutable objects; this
    means that you must create a transient version of the subject to modify it via the member
    function sub.transient(). Better, you can make copies of the objects with desired modifications
    using the copy method--see the pimms library documentation regarding immutable classes and
    objects.

    The following options are accepted:
      * meta_data (default: None) may optionally be a map that contains meta-data to be passed along
        to the subject object (note that this meta-data will not be cached).
      * check_path (default: True) may optionally be set to False to ignore the requirement that a
        directory contain at least the mri/, label/, and surf/ directories to be considered a valid
        FreeSurfer subject directory. Subject objects returned using this argument are not cached.
        Additionally, check_path may be set to None instead of False, indicating that no checks or
        search should be performed; the string name should be trusted to be an exact relative or
        absolute path to a valid FreeSurfer subejct.
    '''
    name = os.path.expanduser(os.path.expandvars(name))
    if check_path is None:
        sub = Subject(name, check_path=False)
        if isinstance(sub, Subject): sub.persist()
    else:
        subpath = find_subject_path(name, check_path=check_path)
        if subpath is None and name == 'fsaverage':
            # we can use the benson and winawer 2018 dataset
            import neuropythy as ny
            try: return ny.data['benson_winawer_2018'].subjects['fsaverage']
            except Exception: pass # error message below is more accurate...
        if subpath is None:
            raise ValueError('Could not locate subject with name \'%s\'' % name)
        elif check_path:
            fpath = '/' + os.path.relpath(subpath, '/')
            if fpath in subject._cache:
                sub = subject._cache[fpath]
            else:
                sub = Subject(subpath)
                if isinstance(sub, Subject): subject._cache[fpath] = sub.persist()
        else:
            sub = Subject(subpath, check_path=False)
            if isinstance(sub, Subject): sub.persist()
    return (None                     if sub is None           else
            sub.with_meta(meta_data) if meta_data is not None else
            sub)
subject._cache = {}
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
    else:
        for (k,v) in six.iteritems(subject._cache):
            if v is sub:
                del subject._cache[k]
                break
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
    elif to == 'data':   return img.get_data()
    elif to == 'affine': return img.affine
    elif to == 'header': return img.header
    elif to == 'field':
        dat = np.squeeze(img.get_data())
        if len(dat.shape) > 2:
            raise ValueError('image requested as field has more than 2 non-unitary dimensions')
        return dat
    elif to in ['auto', 'automatic']:
        dims = set(img.dataobj.shape)
        if 1 < len(dims) < 4 and 1 in dims:
            return np.squeeze(img.get_data())
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
            if affine is None: affine = like.voxel_to_native_matrix
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
def load_freesurfer_label(filename, read_scalars=False):
    '''
    load_freesurfer_label(filename) is equivalent to nibabel.freesurfer.io.read_label(filename).
    '''
    return fsio.read_label(filename, read_scalars=read_scalars)
@nyio.importer('freesurfer_annot', ('annot',))
def load_freesurfer_annot(filename, orig_ids=False):
    '''
    load_freesurfer_annot(filename) is equivalent to nibabel.freesurfer.io.read_annot(filename).
    '''
    return fsio.read_annot(filename, orig_ids=orig_ids)
    

    
    
    
