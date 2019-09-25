####################################################################################################
# neuropythy/datasets/hcp.py
# The HCP_1200 dataset
# by Noah C. Benson

import os, six, shutil, tarfile, logging, warnings, pimms
import numpy as np
import pyrsistent as pyr

from six.moves import urllib

if six.PY3: from functools import reduce

from .core        import (Dataset, add_dataset)
from ..util       import (config, curry, auto_dict, to_credentials, pseudo_path)
from ..vision     import as_retinotopy
from ..           import io      as nyio
from .. import hcp

config.declare_credentials('hcp_credentials',
                           environ_name='HCP_CREDENTIALS',
                           extra_environ=[('HCP_KEY', 'HCP_SECRET'),
                                          'S3FS_CREDENTIALS',
                                          ('S3FS_KEY', 'S3FS_SECRET')],
                           filenames=['~/.hcp-passwd', '~/.passwd-hcp',
                                      '~/.s3fs-passwd', '~/.passwd-s3fs'],
                           aws_profile_name=['HCP', 'hcp', 'S3FS', 's3fs'])
def to_nonempty(s):
    '''
    to_nonempty(s) yields s if s is a nonempty string and otherwise raises an exception.
    '''
    if not pimms.is_str(s) or s == '': raise ValueError('cannot convert object to non-empty string')
    return s
config.declare('hcp_auto_release', environ_name='HCP_AUTO_RELEASE', default_value='HCP_1200',
               filter=to_nonempty)
config.declare('hcp_auto_database',environ_name='HCP_AUTO_DATABASE',default_value='hcp-openaccess',
               filter=to_nonempty)
config.declare('hcp_auto_path',    environ_name='HCP_AUTO_PATH',    default_value=Ellipsis,
               filter=to_nonempty)
config.declare('hcp_auto_default_alignment',
               environ_name='HCP_AUTO_DEFAULT_ALIGNMENT',
               default_value='MSMAll',
               filter=to_nonempty)
def to_interpolation_method(im):
    '''
    to_interpolation_method(x) yields either 'linear' or 'nearest' or raises an exception, depending
      on x; if x is None or a string resembling 'nearest', then 'nearest' is returned, while a
      string resembling 'linear' results in 'linear' being returned.
    '''
    if im is None: return 'nearest'
    im = im.lower()
    if   im in ['linear','lin','trilinear','trilin']: return 'linear'
    elif im in ['nearest','near','nn','nearest-neighbor','nearest_neighbor']: return 'nearest'
    else: raise ValueError('given interpolation_method (%s) is not recognized' % (im,))
config.declare('hcp_retinotopy_interpolation_method',
               environ_name='HCP_RETINOTOPY_INTERPOLATION_METHOD',
               default_value='nearest', filter=to_interpolation_method)
# See if the config/environment lets auto-downloading start in the "on" state
def to_auto_download_state(arg):
    '''
    to_auto_download_state(arg) attempts to coerce the given argument into a valid auto-downloading
      instruction. Essentially, if arg is "on", "yes", "true", "1", True, or 1, then True is
      returned; if arg is "structure" then "structure" is returned; otherwise False is returned. The
      special value Ellipsis yields True if the 'hcp_credentials' have been provided and False
      otherwise.
    '''
    if   arg is Ellipsis:       return config['hcp_credentials'] is not None
    elif not pimms.is_str(arg): return arg in (True, 1)
    elif arg.strip() == '': raise ValueError('auto-download argument may not be an empty string')
    else:
        arg = arg.lower().strip()
        return (True        if arg in ('on', 'yes', 'true', '1')            else
                'structure' if arg in ('struct', 'structure', 'structural') else
                False)
config.declare('hcp_auto_download', environ_name='HCP_AUTO_DOWNLOAD',
               filter=to_auto_download_state, default_value=Ellipsis)

@pimms.immutable
class HCPDataset(Dataset):
    '''
    neuropythy.data['hcp'] is a Dataset containing the publicly provided structural data from the
    Human Connectome Project; see https://db.humanconnectome.org/ for more information about the
    dataset itself.

    In order to use this dataset, you will need to configure a few items; primarily, you will need
    to provide some credentials via the neuropythy.config interface (see also:
    https://github.com/noahbenson/neuropythy/wiki/Configuration). You can generate HCP S3
    credentials at the db.humanconnectome.org site; once these are generated, they should either be
    placed in your neuropythy config file or set manually:
      `ny.config['hcp_credentials'] = 'dummykey:dummysecret'`

    You will probably also want to specify a location to cache the HCP subject data; if this is not
    explicitly provided, then neuropythy will either create a directory "HCP" in its default cache
    directory or, if there is no default cache directory, it will create a temporary directory just
    for the current Python session (this is deleted when Python exits).

    The ny.data['hcp'] object itself is a neuropythy Dataset object; the only relevant data field
    this dataset object contains is the subjects member, which is a lazy-map of the  HCP_1200
    subjects; e.g., ny.data['hcp'].subjects[100610] will yield the subject object for HCP subject
    100610.
    '''
    def __init__(self,
                 credentials=Ellipsis, release=Ellipsis, database=Ellipsis,
                 default_alignment=Ellipsis, cache_directory=Ellipsis,
                 meta_data=None, create_directories=True, create_mode=0o755):
        cdir = cache_directory
        if cdir is Ellipsis: cdir = config['hcp_auto_path']
        if cdir is Ellipsis:
            cdir = config['hcp_subject_paths']
            if cdir is not None: cdir = next(iter(cdir), None)
        Dataset.__init__(self, 'hcp',
                         meta_data=meta_data,
                         custom_directory=cdir,
                         create_directories=create_directories,
                         create_mode=create_mode)
        self.release = release
        self.database = database
        self.credentials = credentials
        self.default_alignment = default_alignment
    @pimms.param
    def default_alignment(df):
        '''
        hcp.default_alignment is the default alignment used by the HCP dataset. By default this is
        'MSMAll'.
        '''
        if df is None or df is Ellipsis: df = config['hcp_auto_default_alignment']
        df = df.lower()
        if   df in ['all', 'msmall', 'auto', 'automatic']: return 'MSMAll'
        elif df in ['sulc', 'msmsulc']: return 'MSMSulc'
        else: raise ValueError('default_alignment must be MSMAll or MSMSulc')
    @pimms.param
    def release(r):
        '''
        hcp.release is the release name for the HCP dataset object; by default this is 'HCP_1200' or
        whatever value is configured in neuropythy.config.
        '''
        if r is Ellipsis or r is None: return config['hcp_auto_release']
        elif not pimms.is_str(r): raise ValueError('HCP release must be a string')
        else: return r
    @pimms.param
    def database(r):
        '''
        hcp.database is the database name for the HCP dataset object; by default this is
        'hcp-openaccess' or whatever value is configured in neuropythy.config.
        '''
        if r is Ellipsis or r is None: return config['hcp_auto_database']
        elif not pimms.is_str(r): raise ValueError('HCP database must be a string')
        else: return r
    @pimms.param
    def credentials(c):
        '''
        hcp.credentials is the credential pair used by the HCP Dataset object hcp.
        '''
        if c is Ellipsis or c is None: return config['hcp_credentials']
        else: return to_credentials(c)
    @pimms.value
    def url(database, release):
        '''
        hcp.url is the base url of the HCP S3 subjects path; this is typically
        's3://hcp-openaccess/HCP_1200'.
        '''
        return 's3://' + database + '/' + release
    @pimms.value
    def s3_path(url, credentials, cache_directory):
        '''
        hcp.s3_path is the pseudo_path object that is used to create the various HCP subjects.
        '''
        return pseudo_path(url, cache_path=cache_directory, delete=False, credentials=credentials)
    @pimms.value
    def s3fs(s3_path):
        '''
        hcp.s3fs is the s3fs object that maintains a connection to the amazon s3.
        '''
        return s3_path._path_data['pathmod'].s3fs
    @staticmethod
    def _load_subject(ppath, url, credentials, cache_directory, df, sid):
        '''
        Creates a pseudo-path for the given subject and loads that subject as an hcp subject then
        returns it.
        '''
        from neuropythy.hcp import subject
        from posixpath import join as urljoin
        pm    = ppath._path_data['pathmod']
        creds = ppath.credentials
        # Make the pseudo-path; if auto-downloading is set to false, we want to create a pseudo-path
        # from only the cache path
        if config['hcp_auto_download'] in [False, None]:
            ppath = pseudo_path(os.path.join(cache_directory, str(sid)), delete=False)
        else:
            ppath = pseudo_path((pm.s3fs, urljoin(url, str(sid))),
                                credentials=credentials,
                                cache_path=os.path.join(cache_directory, str(sid)),
                                delete=False) # the base dir will be deleted if needs be
        return subject(ppath, default_alignment=df, name=str(sid))
    @pimms.value
    def _subjects(url, s3_path, default_alignment, cache_directory):
        '''
        hcp._subjects is a lazy persistent map of all the subjects that are part of the HCP_1200
        dataset. Unlike the hcp.subjects member, _subjects do not contain retinotopic mapping data
        for those subjects with available retinotopies from the ny.data['hcp_retinotopy'] dataset.
        '''
        # we get the subject list by doing an ls of the amazon bucket, but we fallback to a constant
        # list (in neuropythy.hcp)
        logging.info('HCPDataset: Getting HCP subject list...')
        aws_ips = ['99.84.127.26', '99.84.127.10', '99.84.127.3', '99.84.127.119']
        try:
            # this will raise an error if the internet is not alive
            dummy = urllib.request.urlopen('http://216.58.192.142', timeout=1)
            # internet is alive, so continue...
            pm = s3_path._path_data['pathmod']
            subdirs = []
            for s in pm.s3fs.ls(url):
                s = pm.split(s)[-1]
                try: s = int(s)
                except Exception: continue
                subdirs.append(s)
        except Exception:
            logging.info('       ...  Failed; falling back on builtin list.')
            subdirs = hcp.subject_ids
        creds = s3_path.credentials
        # okay, we will make a loader for each of these:
        ss = {sid: curry(HCPDataset._load_subject,
                         s3_path, url, creds, cache_directory, default_alignment, sid)
              for sid in subdirs}
        return pimms.lazy_map(ss)
    @pimms.value
    def subject_ids(_subjects):
        '''
        hcp.subject_ids is a tuple of the subject ids found in the HCP dataset.
        '''
        return tuple(_subjects.keys())
    @pimms.value
    def subjects(_subjects):
        '''
        hcp.subjects is a lazy persistent map of all the subjects that are part of the HCP_1200
        dataset. Subjects with valid retinotopic mapping data (assuming that the
        ny.data['hcp_retinotopy'] dataset has been initialized) include retinotopic mapping data
        as part of their property data.
        '''
        try:
            from neuropythy import data
            dset = data['hcp_retinotopy']
            subs = dset.subjects
        except Exception: return _subjects
        # okay, so far so good; let's setup the subject updating function:
        sids = set(list(_subjects.keys()))
        def _add_retino(sid):
            if sid in subs: return subs[sid]
            else:           return _subjects[sid]
        return pimms.lazy_map({sid: curry(_add_retino, sid) for sid in six.iterkeys(_subjects)})
    def download(self, sid):
        '''
        ny.data['hcp'].download(sid) downloads all the data understood by neuropythy for the given
        HCP subject id; the data are downloaded from the Amazon S3 into the path given by the 
        'hcp_auto_path' config item then returns a list of the downloaded files.
        '''
        # we can do this in quite a sneaky way: get the subject, get their filemap, force all the
        # paths in the subject to be downloaded using the pseudo-path, return the cache path!
        sub   = self.subjects[sid]
        fmap  = sub.meta_data['file_map']
        ppath = fmap.path
        fls   = []
        logging.info('Downloading HCP subject %s structure data...' % (sid,))
        for fl in six.iterkeys(fmap.data_files):
            logging.info('  * Downloading file %s for subject %s' % (fl, sid))
            try:
                fls.append(ppath.local_path(fl))
            except ValueError as e:
                if len(e.args) != 1 or not e.args[0].startswith('getpath:'): raise
                else: logging.info('    (File %s not found for subject %s)' % (fl, sid))
        logging.info('Subject %s donwnload complete!' % (sid,))
        return fls
# we wrap this in a lambda so that it gets loaded when requested (in case the config changes between
# when this gets run and when the dataset gets requested)
add_dataset('hcp', lambda:HCPDataset().persist())

####################################################################################################
# HCP Retinotopy

def to_retinotopy_cache_path(p):
    '''
    to_retinotopy_cache_path(p) yields p if p is a directory and raises an exception otherwise.
    '''
    if pimms.is_str(p) and os.path.isdir(p): return os.path.normpath(p)
    elif p is Ellipsis: return p
    else: return None
config.declare('hcp_retinotopy_cache_path', environ_name='HCP_RETINOTOPY_CACHE_PATH',
               filter=to_retinotopy_cache_path, default_value=Ellipsis)
def _prefix_keys(m, prefix):
    '''
    Lazily adds the given prefix to all keys in m, which must all be strings, and returns the new
    lazy map.
    '''
    f = lambda k:m[k]
    return pimms.lazy_map({(prefix+k): curry(f, k) for k in six.iterkeys(m)})    

hcp_retinotopy_property_names = ('polar_angle', 'eccentricity', 'radius', 'variance_explained',
                                 'mean_signal', 'gain', 'x', 'y')
hcp_retinotopy_property_files = pyr.pmap({'polar_angle':'angle', 'eccentricity':'eccen', 
                                          'mean_signal':'means', 'gain':'const', 'radius':'prfsz',
                                          'variance_explained':'vexpl', 'x':'xcrds', 'y':'ycrds'})
@pimms.calc(*hcp_retinotopy_property_names)
def calc_native_properties(native_hemi, fs_LR_hemi, prefix, resolution, subject_id=None,
                           method='nearest', cache_directory=None, alignment='MSMAll'):
    '''
    calc_native_properties is a pimms calculator that requires a native hemisphere, an fs_LR
    hemisphere, and a property prefix and yields the interpolated retinotopic mapping properties.
    '''
    method = to_interpolation_method(method)
    # first: check for a cache file:
    if cache_directory is not None:
        if subject_id is None: pth = cache_directory
        else: pth = os.path.join(cache_directory, str(subject_id), 'retinotopy')
        # we load all or none here:
        ftr = hcp_retinotopy_property_files
        mtd = '.' if method == 'nearest' else '.linear.'
        flp = '%s.%s%%s.%s%snative%dk.mgz' % (native_hemi.chirality,prefix,alignment,mtd,resolution)
        fls = {k:os.path.join(pth, flp % ftr[k]) for k in hcp_retinotopy_property_names}
        try: return {k:nyio.load(v, 'mgh') for (k,v) in six.iteritems(fls)}
        except Exception: pass
    else: fls = None
    logging.info('HCP Dataset: Interpolating retinotopy for HCP subject %s / %s (method: %s)...'
                 % (subject_id, native_hemi.chirality, method))
    p = fs_LR_hemi.interpolate(native_hemi,
                               {k:(prefix+k) for k in hcp_retinotopy_property_names
                                if k not in ['polar_angle', 'eccentricity']},
                               method=method)
    # calculate the angle/eccen from the x and y values
    theta = np.arctan2(p['y'], p['x'])
    p = pimms.assoc(p,
                    polar_angle=np.mod(90 - 180/np.pi*theta + 180, 360) - 180,
                    eccentricity=np.hypot(p['x'], p['y']))
    # write cache and return
    if fls is not None:
        try:
            for (k,v) in six.iteritems(p): nyio.save(fls[k], v)
        except Exception as e:
            tup = (subject_id, native_hemi.chirality, type(e).__name__ + str(e.args))
            warnings.warn('cache write failed for HCP retinotopy subject %s / %s: %s' % tup,
                          RuntimeWarning)
    return p
interpolate_native_properties = pimms.plan({'native_properties': calc_native_properties})

@pimms.immutable
class HCPRetinotopyDataset(Dataset):
    '''
    neuropythy.data['hcp_retinotopy'] is a Dataset containing the publicly provided data from the
    Benson et al. (2018; DOI:10.1167/18.13.23) paper on the HCP 7T retinotopy dataset. For more
    information see the paper's OSF site (https://osf.io/bw9ec/)

    You do not have to explicitly configure anything in order to use this dataset: the default
    behavior is to use the the same cache directory as the 'hcp' dataset, meaning that your subject
    directories will gain 'retinotopy' subdirectories containing cached mgz files. These
    'retinotopy' subdirectories exist at the same level as 'MNINonLinear' and 'T1w'.

    It is recommended that you interact with the data via the neuropythy.data['hcp'] dataset--this
    dataset stores the structural data necessary to make sense of the retinotopic data and will
    automatically include the relevant retinotopic data as properties attached to any subject
    requested that has retinotopic data.

    If you wish to explicitly disable hcp_retinotopy downloading, you can do so by setting the
    'hcp_auto_download' config item to either False or 'structure' (indicating that structural
    downloads should continue but retinotopic ones should not). Note that this latter setting does
    not actually prevent you from inducing downloading via the 'hcp_retinotopy' dataset directly,
    but it will prevent the 'hcp' dataset from inducing such downloads.

    The default behavior of the hcp_retinotopy dataset is to put cache files in the same directories
    as the auto-downloaded HCP subject data; a separate cache directory can be provided via the
    neuropythy hcp_retinotopy_cache_path config item.
    '''
    default_url = 'osf://bw9ec/'
    retinotopy_files = pyr.pmap({32:'prfresults.mat', 59:'prfresults59k.mat'})
    retinotopy_prefix = 'prf'
    lowres_retinotopy_prefix = 'lowres-prf'
    highres_retinotopy_prefix = 'highres-prf'
    subject_ids = tuple([100610, 102311, 102816, 104416, 105923, 108323, 109123, 111312,
                         111514, 114823, 115017, 115825, 116726, 118225, 125525, 126426,
                         128935, 130114, 130518, 131217, 131722, 132118, 134627, 134829,
                         135124, 137128, 140117, 144226, 145834, 146129, 146432, 146735,
                         146937, 148133, 150423, 155938, 156334, 157336, 158035, 158136,
                         159239, 162935, 164131, 164636, 165436, 167036, 167440, 169040,
                         169343, 169444, 169747, 171633, 172130, 173334, 175237, 176542,
                         177140, 177645, 177746, 178142, 178243, 178647, 180533, 181232,
                         181636, 182436, 182739, 185442, 186949, 187345, 191033, 191336,
                         191841, 192439, 192641, 193845, 195041, 196144, 197348, 198653,
                         199655, 200210, 200311, 200614, 201515, 203418, 204521, 205220,
                         209228, 212419, 214019, 214524, 221319, 233326, 239136, 246133,
                         249947, 251833, 257845, 263436, 283543, 318637, 320826, 330324,
                         346137, 352738, 360030, 365343, 380036, 381038, 385046, 389357,
                         393247, 395756, 397760, 401422, 406836, 412528, 429040, 436845,
                         463040, 467351, 525541, 536647, 541943, 547046, 550439, 552241,
                         562345, 572045, 573249, 581450, 585256, 601127, 617748, 627549,
                         638049, 644246, 654552, 671855, 680957, 690152, 706040, 724446,
                         725751, 732243, 751550, 757764, 765864, 770352, 771354, 782561,
                         783462, 789373, 814649, 818859, 825048, 826353, 833249, 859671,
                         861456, 871762, 872764, 878776, 878877, 898176, 899885, 901139,
                         901442, 905147, 910241, 926862, 927359, 942658, 943862, 951457,
                         958976, 966975, 971160, 973770, 995174, 999997, 999998, 999999])

    # these expect value % (hemi, alignment, surfacename)
    _retinotopy_cache_tr = pimms.persist({
        # the native filename's here aren't actually used; they would also include a '.linear.' for
        # linear interpolation right before the 'native59k' or 'native32k' (these are the filenames
        # for nearest interpolation)
        'native': { 
            (highres_retinotopy_prefix + '_polar_angle')        :'%s.split%s_angle.%s.native59k.mgz',
            (highres_retinotopy_prefix + '_eccentricity')       :'%s.split%s_eccen.%s.native59k.mgz',
            (highres_retinotopy_prefix + '_radius')             :'%s.split%s_prfsz.%s.native59k.mgz',
            (highres_retinotopy_prefix + '_variance_explained') :'%s.split%s_vexpl.%s.native59k.mgz',
            (highres_retinotopy_prefix + '_mean_signal')        :'%s.split%s_means.%s.native59k.mgz',
            (highres_retinotopy_prefix + '_gain')               :'%s.split%s_const.%s.native59k.mgz',
            (lowres_retinotopy_prefix + '_polar_angle')         :'%s.split%s_angle.%s.native32k.mgz',
            (lowres_retinotopy_prefix + '_eccentricity')        :'%s.split%s_eccen.%s.native32k.mgz',
            (lowres_retinotopy_prefix + '_radius')              :'%s.split%s_prfsz.%s.native32k.mgz',
            (lowres_retinotopy_prefix + '_variance_explained')  :'%s.split%s_vexpl.%s.native32k.mgz',
            (lowres_retinotopy_prefix + '_mean_signal')         :'%s.split%s_means.%s.native32k.mgz',
            (lowres_retinotopy_prefix + '_gain')                :'%s.split%s_const.%s.native32k.mgz'},
        'LR32k': {                                             
            (lowres_retinotopy_prefix + '_polar_angle')         :'%s.split%s_angle.32k.mgz',
            (lowres_retinotopy_prefix + '_eccentricity')        :'%s.split%s_eccen.32k.mgz',
            (lowres_retinotopy_prefix + '_radius')              :'%s.split%s_prfsz.32k.mgz',
            (lowres_retinotopy_prefix + '_variance_explained')  :'%s.split%s_vexpl.32k.mgz',
            (lowres_retinotopy_prefix + '_mean_signal')         :'%s.split%s_means.32k.mgz',
            (lowres_retinotopy_prefix + '_gain')                :'%s.split%s_const.32k.mgz'},
        'LR59k': {                                             
            (highres_retinotopy_prefix + '_polar_angle')        :'%s.split%s_angle.59k.mgz',
            (highres_retinotopy_prefix + '_eccentricity')       :'%s.split%s_eccen.59k.mgz',
            (highres_retinotopy_prefix + '_radius')             :'%s.split%s_prfsz.59k.mgz',
            (highres_retinotopy_prefix + '_variance_explained') :'%s.split%s_vexpl.59k.mgz',
            (highres_retinotopy_prefix + '_mean_signal')        :'%s.split%s_means.59k.mgz',
            (highres_retinotopy_prefix + '_gain')               :'%s.split%s_const.59k.mgz'}})
    
    def __init__(self, url=Ellipsis, cache_directory=Ellipsis, interpolation_method=Ellipsis,
                 meta_data=None, create_directories=True, create_mode=0o755):
        cdir = cache_directory
        if cdir is Ellipsis: cdir = config['hcp_auto_path']
        if cdir is Ellipsis:
            cdir = config['hcp_subject_paths']
            if cdir is not None: cdir = next(iter(cdir), None)
        Dataset.__init__(self, 'hcp_retinotopy',
                         meta_data=meta_data,
                         custom_directory=cdir,
                         create_directories=create_directories,
                         create_mode=create_mode)
        if url is Ellipsis: url = HCPRetinotopyDataset.default_url
        self.url = url
        self.interpolation_method = interpolation_method
    @pimms.param
    def url(u):
        '''
        ny.data['hcp_retinotopy'].url is the url from which the retinotopy data is loaded.
        '''
        if not pimms.is_str(u): raise ValueError('url must be a string')
        return u
    @pimms.param
    def interpolation_method(im):
        '''
        ny.data['hcp_retinotopy'].interpolation_method is a string, either 'nearest' (default) or
        'linear', which specifies whether nearest or linear interpolation should be used when
        interpolating retinotopy data from the fs_LR meshes onto the native meshes.
        '''
        if im is Ellipsis or im is None: return config['hcp_retinotopy_interpolation_method']
        else: return to_interpolation_method(im)
    @pimms.value
    def pseudo_path(url, cache_directory):
        '''
        ny.data['hcp_retinotopy'].pseudo_path is the psueod-path object responsible for loading the
        retinotopy data.
        '''
        return pseudo_path(url, cache_path=cache_directory).persist()
    @pimms.value
    def cifti_data(pseudo_path):
        '''
        ny.data['hcp_retinotopy'].cifti_data is a tuple of lazy maps of the 32k and 59k data arrays,
        reorganized into 'visual' retinotopic coordinates. The tuple elements represent the
        (full, split1, split2) solutions.
        '''
        # our loader function:
        def _load(res, split):
            import h5py
            flnm = HCPRetinotopyDataset.retinotopy_files[res]
            logging.info('HCPRetinotopyDataset: Loading split %d from file %s...' % (split, flnm))
            flnm = pseudo_path.local_path(flnm)
            with h5py.File(flnm, 'r') as f:
                sids = np.array(f['subjectids'][0], dtype='int')
                data = np.array(f['allresults'][split])
            sids.setflags(write=False)
            # convert these into something more coherent
            tmp = hcp.cifti_split(data)
            for q in tmp: q.setflags(write=False)
            return pyr.pmap(
                {h: pyr.m(prf_polar_angle        = np.mod(90 - dat[:,0] + 180, 360) - 180,
                          prf_eccentricity       = dat[:,1],
                          prf_radius             = dat[:,5],
                          prf_variance_explained = dat[:,4]/100.0,
                          prf_mean_signal        = dat[:,3],
                          prf_gain               = dat[:,2],
                          prf_x                  = dat[:,1]*np.cos(np.pi/180*dat[:,0]),
                          prf_y                  = dat[:,1]*np.sin(np.pi/180*dat[:,0]))
                 for (h,dat) in zip(['lh','rh','subcortical'], tmp)})
        splits = [pimms.lazy_map({res: curry(_load, res, split)
                                  for res in six.iterkeys(HCPRetinotopyDataset.retinotopy_files)})
                  for split in [0,1,2]]
        return tuple(splits)
    @pimms.value
    def subject_order(pseudo_path):
        '''
        subject_order is a mapping of subject ids to the offset at which they appear in the cifti
        data.
        '''
        import h5py
        def _load(res):
            flnm = HCPRetinotopyDataset.retinotopy_files[res]
            flnm = pseudo_path.local_path(flnm)
            logging.info('HCPRetinotopyDataset: Loading subjects from file %s...' % flnm)
            with h5py.File(flnm, 'r') as f:
                sids = np.array(f['subjectids'][0], dtype='int')
            return pyr.pmap({sid:ii for (ii,sid) in enumerate(sids)})
        return pimms.lazy_map({res: curry(_load, res)
                               for res in six.iterkeys(HCPRetinotopyDataset.retinotopy_files)})
    @pimms.value
    def retinotopy_data(cifti_data, subject_order, cache_directory,create_directories,create_mode):
        '''
        ny.data['hcp_retinotopy'].retinotopy_data is a nested-map data structure representing the
        retinotopic mapping data for each subject. The first layer of keys in retinotopy_data is
        the subject id, including the 99999* average subjects. The second layer of keys is the name
        of the hemisphere, and the third layer of keys are property names (the final layer is always
        constructed of lazy maps).
        '''
        rpfx = HCPRetinotopyDataset.retinotopy_prefix
        sids = HCPRetinotopyDataset.subject_ids
        # how we load data:
        def _load_LR(split, sid, h, res, prop, flpatt):
            pth = os.path.join(cache_directory, str(sid), 'retinotopy')
            if not os.path.exists(pth) and create_directories: os.makedirs(pth, create_mode)
            flnm = os.path.join(pth, flpatt % (h,split))
            # deduce the generic property name from the given prop name
            pp = rpfx + prop.split(rpfx)[1]
            # as a general rule: we don't want to touch cifti_split lazy-maps because they force the
            # entire .mat file to load... BUT if we have already loaded it, we're better off sharing
            # the loaded data instead of reloading cache
            # parse the prop -- we can remove the front of it
            cdat = cifti_data[split]
            okwrite = True
            sii = subject_order[res][sid]
            if not cdat.is_lazy(res) or not os.path.isfile(flnm): dat = cdat[res][h][pp][sii]
            else:
                try: return nyio.load(flnm)
                except Exception:
                    warnings.warn('failed to load HCP retinotopy cache file %s'%flnm)
                    okwrite = False
                dat = cdat[res][h][pp][sii]
            if okwrite: nyio.save(flnm, dat)
            return dat
        # okay, the base properties for the 32k and 59k meshes:
        trs = HCPRetinotopyDataset._retinotopy_cache_tr
        base_props = {
            sid: {h: {split: {res: pimms.lazy_map({p: curry(_load_LR, split, sid, h, res, p, flp)
                                                   for (p,flp) in six.iteritems(trs['LR%dk'%res])})
                              for res in [32, 59]}
                      for split in [0,1,2]}
                  for h in ['lh','rh']}
            for sid in sids}
        # okay, now we build on top of these: we add in the x/y properties
        def _add_xy(dat):
            k = next(six.iterkeys(dat))
            prefix = k.split('_')[0] + '_'
            (ang,ecc) = [dat[prefix + k] for k in ('polar_angle', 'eccentricity')]
            tht = np.pi/180 * (90 - ang)
            (x,y) = [ecc*np.cos(tht), ecc*np.sin(tht)]
            return pimms.assoc(dat, prefix + 'x', x, prefix + 'y', y)
        xy_props = {
            sid: {h: {split: pimms.lazy_map({res: curry(_add_xy, rdat)
                                             for (res,rdat) in six.iteritems(spdat)})
                      for (split,spdat) in six.iteritems(hdat)}
                  for (h,hdat) in six.iteritems(sdat)}
            for (sid,sdat) in six.iteritems(base_props)}
        # okay, that's it; just organize it into the desired shape
        def _reorg(hdat, res, ks):
            pfx = ks[0].split('_')[0] + '_'
            ks = ks + [pfx + 'x', pfx + 'y']
            f = lambda s,k:s[res][k]
            return pimms.lazy_map(
                {(pre+k): curry(f, s, k)
                 for k in ks
                 for (s,pre) in zip([hdat[0],hdat[1],hdat[2]], ['', 'split1-', 'split2-'])})
        r = {sid: {('%s_LR%dk'%(h,res)): _reorg(hdat, res, list(base_props[sid][h][0][res].keys()))
                   for (h,hdat) in six.iteritems(sdat)
                   for res in [32,59]}
             for (sid,sdat) in six.iteritems(xy_props)}
        return pimms.persist(r)
    @pimms.value
    def subjects(retinotopy_data, cache_directory, create_mode, create_directories,
                 interpolation_method):
        '''
        hcp_retinotopy.subjects is a lazy map whose keys are HCP subject IDs (of those subjects with
        valid retinotopic mapping data) and whose values are subject objects (obtained from the
        ny.data['hcp']._subjects map) with the addition of retinotopic mapping properties.
        '''
        alltrs = HCPRetinotopyDataset._retinotopy_cache_tr
        lrtrs = {32: alltrs['LR32k'], 59: alltrs['LR59k']}
        lpfx = HCPRetinotopyDataset.lowres_retinotopy_prefix
        hpfx = HCPRetinotopyDataset.highres_retinotopy_prefix
        nttrs = {32: {k:v for (k,v) in six.iteritems(alltrs['native']) if k.startswith(lpfx)},
                 59: {k:v for (k,v) in six.iteritems(alltrs['native']) if k.startswith(hpfx)}}
        def _prep(sid):
            # see if we can get the subject from the 'hcp' dataset:
            try:
                from neuropythy import data
                hcp = data['hcp']
                sub = hcp._subjects[sid]
            except Exception: hcp = None
            if hcp is None: raise ValueError('could not load subject object for sid %s' % (sid,))
            # okay, we need to prep this subject; the initial part is easy: copy the properties
            # for the fs_LR meshes over to them:
            hems = hems0 = sub.hemis
            def _add_LR(hnm, res):
                hem = hems0[hnm]
                return hem.with_prop(retinotopy_data[sid]['%s_LR%dk' % (hnm[:2], res)])
            for h in ['lh','rh']:
                for res in [32,59]:
                    for align in ['MSMAll', 'MSMSulc']:
                        hnm = '%s_LR%dk_%s' % (h,res,align)
                        hems = hems.set(hnm, curry(_add_LR, hnm, res))
            # okay, we have the data transferred over the the fs_LR hemispheres now; we just need
            # to do some interpolation for the native hemispheres
            hems1 = hems
            def _get(inp, k): return inp[k]
            def _interp_hem(hemi, h, res, align):
                pfls = nttrs[res]
                pfx = next(six.iterkeys(pfls)).split('_')[0] + '_'
                inpargs = dict(native_hemi=hemi, fs_LR_hemi=hems1['%s_LR%dk_%s' % (h, res, align)],
                               subject_id=sid,   method=interpolation_method,  resolution=res,
                               alignment=align,  cache_directory=cache_directory)
                lm = pimms.lazy_map({(p+k): curry(_get, inp, k)
                                     for spl in ['', 'split1-', 'split2-'] for p in [spl + pfx]
                                     for inp in [interpolate_native_properties(prefix=p, **inpargs)]
                                     for k   in hcp_retinotopy_property_names})
                return hemi.with_prop(lm)
            def _get_highest_res(hemi, k):
                try: x = hemi.prop(hpfx + '_' + k)
                except Exception: x = None
                if x is not None: return x
                try: return hemi.prop(lpfx + '_' +  k)
                except Exception: x = None
                if x is not None: return x
                raise ValueError('no retinotopy successfully loaded for hemi', hemi)
            def _interp_nat(h, align):
                # okay, let's get the hemisphere we are modifying...
                hemi = hems1['%s_native_%s' % (h, align)]
                # we're going to interp from both the 32k and 59k meshes (if possible)
                for res in [32, 59]: hemi = _interp_hem(hemi, h, res, align)
                # add the 'best/standard' prf header:
                pfx = HCPRetinotopyDataset.retinotopy_prefix
                lm = pimms.lazy_map({(pfx + '_' + k): curry(_get_highest_res, hemi, k)
                                     for k in hcp_retinotopy_property_names})
                # return with these properties:
                return hemi.with_prop(lm)
            for h in ['lh','rh']:
                for align in ['MSMAll','MSMSulc']:
                    hems = hems.set(h + '_native_' + align, curry(_interp_nat, h, align))
            # fix the hemisphere aliases based on default alignment:
            default_alignment = sub.meta_data.get('default_alignment',
                                                  config['hcp_default_alignment'])
            hems2 = hems
            for h in ['lh_native',  'rh_native',  'lh_LR32k',   'rh_LR32k',
                      'lh_LR59k',   'rh_LR59k',   'lh_LR164k',  'rh_LR164k']:
                hems = hems.set(h, curry(lambda h:hems2[h+'_'+default_alignment], h))
            hems = hems.set('lh', lambda:hems2['lh_native_' + default_alignment])
            hems = hems.set('rh', lambda:hems2['rh_native_' + default_alignment])
            return sub.copy(hemis=hems)
        # we just need to call down to this prep function lazily:
        return pimms.lazy_map({sid: curry(_prep, sid) for sid in six.iterkeys(retinotopy_data)})
add_dataset('hcp_retinotopy', lambda:HCPRetinotopyDataset().persist())
