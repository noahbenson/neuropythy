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
def to_nonempty_path(s):
    '''
    to_nonempty_path(s) yields s if s is a nonempty string and otherwise raises an exception. If s
      is a string, then the variable- and user-expanded form is returned.
    '''
    if not pimms.is_str(s) or s == '': raise ValueError('cannot convert object to non-empty string')
    return os.path.expanduser(os.path.expandvars(s))
config.declare('hcp_auto_release', environ_name='HCP_AUTO_RELEASE', default_value='HCP_1200',
               filter=to_nonempty)
config.declare('hcp_auto_database',environ_name='HCP_AUTO_DATABASE',default_value='hcp-openaccess',
               filter=to_nonempty)
config.declare('hcp_auto_path',    environ_name='HCP_AUTO_PATH',    default_value=Ellipsis,
               filter=to_nonempty_path)
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
def to_boolean(arg):
    '''
    to_boolean(arg) yields True or False based on the given arg. This function is intended to
      convert a neuropythy config argument into a boolean value. Accordingly, it accepts True or
      False values, numbers, or the strings "true", "on", "yes", "1", etc. as true and "false",
      "off", etc. as false. If arg has no clear True or False value (such as an arbitrary non-empty
      string) then None is returned.
    '''
    # Basically, we want to do a boolean test, but there are some special cases.
    if   arg is None: return False
    elif arg is Ellipsis: return None
    elif pimms.is_str(arg):
        arg = arg.strip().lower()
        return (True  if arg in ('true',  't', 'yes', '1',  'on', '+') else
                False if arg in ('false', 'f',  'no', '0', 'off',  '') else
                None)
    elif pimms.is_number(arg):
        return None if np.isnan(arg) else (arg != 0)
    elif pimms.is_nparray(arg):
        # Handle numpy arrays specially, because they don't like being bool'ed.
        return np.prod(arg.shape) > 0
    else:
        return bool(arg)
def to_auto_download_state(arg):
    '''
    to_auto_download_state(arg) attempts to coerce the given argument into a valid auto-downloading
      instruction. Essentially, if arg is "on", "yes", "true", "1", True, or 1, then True is
      returned; if arg is "structure" then "structure" is returned; otherwise False is returned. The
      special value Ellipsis yields True if the 'hcp_credentials' have been provided and False
      otherwise.
    '''
    if arg is Ellipsis: return config['hcp_credentials'] is not None
    else:               return to_boolean(arg)
config.declare('hcp_auto_download', environ_name='HCP_AUTO_DOWNLOAD',
               filter=to_auto_download_state, default_value=Ellipsis)

# HHCPMetaDataset ##################################################################################
config.declare('hcp_metadata_path', environ_name='HCP_METADATA_PATH', default_value=None,
               filter=to_nonempty_path)
config.declare('hcp_behavioral_path', environ_name='HCP_BEHAVIORAL_PATH', default_value=None,
               filter=to_nonempty_path)
config.declare('hcp_genetic_path', environ_name='HCP_GENETIC_PATH', default_value=None,
               filter=to_nonempty_path)
@pimms.immutable
class HCPMetaDataset(Dataset):
    '''
    The HCPMetaDataset (in neuropythy.data['hcp_metadata'] is a common repository for the various
    behavioral/genetic/meta-data provided by the Human Connectome Project. Because these data are
    not generally available for auto-download and/or are behind a registration process, you have
    to obtain these data manually then configure neuropythy to know about them. Neuropythy does not
    actually use these itself, but other datasets can use them and simply require you to configure
    the dataset through this interface instead of telling each dataset individually where it should
    find the files. A couple of summaries (gender and agegroup) are also provided here.

    To tell neuropythy where to find the relevant files, you use the configuration interface,
    ideally by making a JSON file named .npythyrc in your home directory that contains the paths for
    the relevant files. These files and configuration names are described below. If any of these
    paths is omitted from the configuration, there are a couple of default places that neuropythy
    will look for files with particular names. These are in the "hcp_subjects_path", in the
    "cache_data_root", and in the "hcp_metadata" and "hcp/metadata" subdirectories of the 
    "cache_data_root" configuration item. Both "hcp_subject_paths" and "cache_data_root" are
    configured via neuropythy.config as well and are documented elsewhere. If multiple HCP subject
    paths are included, then all of them are searched. Finally, the optional configuration item,
    "hcp_metadata_path" can be set to a directory which should be searched for files with the
    default names.

      * "hcp_behavioral_path" should be the path of the behavioral data CSV file provided by the
        Human Connectome Project (https://db.humanconnectome.org/). The default search name for
        this file is behvaioral_data_1200_subjects.csv.
      * "hcp_genetic_path" should be the path of the restricted family-structure data CSV file
        provided by the Human Connectome Project. The default search name for this file is the
        "RESTRICTED_hcpfamilystructure.csv"

    If you have declared the object `data = neuropythy.data['hcp_metadata']` then the following
    are the relevant members of data.
      * data.
    '''
    def __init__(self, name='hcp_metadata', cache_directory=None,
                 metadata_path=None, genetic_path=None, behavioral_path=None,
                 create_mode=0o755, create_directories=False,
                 meta_data=None, cache_required=False):
        '''
        HCPMetaDataset() creates a new HCP meta-dataset object. Constructing a new object will
        reset the object based on the current state of the neuropythy config data; alternately,
        values for metadata_path, genetic_path, and behavioral_path may be passed.
        '''
        Dataset.__init__(self, name,
                         meta_data=meta_data,
                         custom_directory=cache_directory,
                         create_directories=create_directories,
                         create_mode=create_mode,
                         cache_required=cache_required)
        self.metadata_path = metadata_path
        self._genetic_path = genetic_path
        self._behavioral_path = behavioral_path
    # How we search for a file
    @staticmethod
    def _findfile(paths, name):
        import os
        for path in paths:
            if path is None or path is Ellipsis: continue
            path = os.path.expanduser(os.path.expandvars(path))
            fname = os.path.join(path, name)
            if os.path.isfile(fname): return fname
        return None
    @pimms.param
    def metadata_path(p):
        '''
        metadata_path is either None (when no metadata-path has been provided/configured) or the
        path to the meta-data directory of the HCP. This directory is simply a location in which
        neuropythy will look for behavioral and genetic metadata about HCP subjects.
        '''
        import os
        if p is None or p is Ellipsis:
            # We want to see if it's in the config (this is None if not)
            return config['hcp_metadata_path']
        else:
            p = os.path.expanduser(os.path.expandvars(p))
            if not os.path.isdir(p):
                raise ValueError('metadata_path must be a directory')
            return p
    @pimms.param
    def _behavioral_path(p):
        '''
        _behavioral_path is the path to the behavioral/meta-data CSV file provided by the Human
        Connectome Project. This is the argument given to the dataset, not the actual name of
        the file used (which may be found elsewhere if None is given, for example).
        '''
        import os
        if p is None or p is Ellipsis: return None
        if not pimms.is_str(p): raise ValueError('behavioral_path must be None or a string')
        return os.path.expanduser(os.path.expandvars(p))
    @pimms.param
    def _genetic_path(p):
        '''
        _genetic_path is the path to the family-structure data CSV file provided by the Human
        Connectome Project. This is the argument given to the dataset, not the actual name of
        the file used (which may be found elsewhere if None is given, for example).
        '''
        import os
        if p is None or p is Ellipsis: return None
        if not pimms.is_str(p): raise ValueError('genetic_path must be None or a string')
        return os.path.expanduser(os.path.expandvars(p))
    @pimms.value
    def behavioral_path(_behavioral_path, metadata_path):
        '''
        behavioral_path is the path of the file being used for behavioral data. If no such file
        is found this is None.
        '''
        if _behavioral_path is not None:
            if not os.path.isfile(_behavioral_path):
                raise ValueError('provided behavioral_path is not a file')
            return _behavioral_path
        f = config['hcp_behavioral_path']
        if f is not None:
            if os.path.isfile(f): return f
            else: warnings.warn('provided config item hcp_behavioral_path is not a file')
        # we look through a few default places
        hcps = config['hcp_subject_paths']
        hcps = [] if hcps is None else hcps
        paths = [metadata_path] + hcps + [config['data_cache_root']]
        f = HCPMetaDataset._findfile(paths, 'behavioral_data_1200_subjects.csv')
        return f
    @pimms.value
    def genetic_path(_genetic_path, metadata_path):
        '''
        genetic_path is the path of the file being used for the restricted family-structure data.
        If no such file is found this is None.
        '''
        if _genetic_path is not None:
            if not os.path.isfile(_genetic_path):
                raise ValueError('provided genetic_path is not a file')
            return _genetic_path
        f = config['hcp_genetic_path']
        if f is not None:
            if os.path.isfile(f): return f
            else: warnings.warn('provided config item hcp_genetic_path is not a file')
        # we look through a few default places
        hcps = config['hcp_subject_paths']
        hcps = [] if hcps is None else hcps
        paths = [metadata_path] + hcps + [config['data_cache_root']]
        f = HCPMetaDataset._findfile(paths, 'RESTRICTED_hcpfamilystructure.csv')
        return f
    @pimms.value
    def behavioral_table(behavioral_path):
        '''
        behavioral_table is a pandas datafame of the behavioral data provided by the Human
        Connectome Project.
        '''
        if behavioral_path is None: return None
        return nyio.load(behavioral_path)
    @pimms.value
    def genetic_table(genetic_path):
        '''
        genetic_table is a pandas dataframe of the restricted genetic data for all subjects from
        the HCP.
        '''
        if genetic_path is None: return None
        return nyio.load(genetic_path)
    @pimms.value
    def behavioral_maps(behavioral_table):
        '''
        behavioral_maps is a dictionary of meta-data dictionaries, one per subject by subject ID.
        '''
        from pyrsistent import pmap
        from neuropythy.util import dataframe_select
        if behavioral_table is None: return None
        mdat = {}
        sids = np.unique(behavioral_table['Subject'].values)
        for sid in sids:
            tmp = dataframe_select(behavioral_table, Subject=sid)
            mdat[sid] = pmap(dict(tmp.iloc[0]))
        return pmap(mdat)
    @pimms.value
    def gender(behavioral_maps):
        '''
        gender is a dictionary of genders ('F' or 'M') for each subject, based on the behavioral
        data provided by the HCP.
        '''
        from pyrsistent import pmap
        from six import iteritems
        if behavioral_maps is None: return None
        return pmap({sid: v['Gender'] for (sid,v) in iteritems(behavioral_maps)})
    @pimms.value
    def agegroup(behavioral_maps):
        '''
        agegroup is a dictionary of agegroup numbers, as provided by the HCP's behavioral data.
        The agegroup number is the mean of the highest and lowest age in the agegroup except for
        the '36+' agegroup, which is coded as a 40.
        '''
        from pyrsistent import pmap
        from six import iteritems
        if behavioral_maps is None: return None
        agegroup = pyr.pmap(
            {sid: np.mean([int(a1), int(a2)])
             for (sid,v) in iteritems(behavioral_maps)
             for age in [v['Age']]
             for (a1,a2) in [age.split('-') if age != '36+' else ['40','40']]})
        return pmap(agegroup)
    @pimms.value
    def siblings(genetic_table):
        '''
        siblings is a mapping of the siblings in the HCP restricted genetics dataset.
        '''
        families = {}
        for (ii,r) in genetic_table.iterrows():
            (sid,fid,zyg) = [r[k] for k in ('Subject','Family_ID','ZygosityGT')]
            zyg = zyg.strip()
            if fid not in families: families[fid] = {}
            fam = families[fid]
            if zyg not in fam: fam[zyg] = []
            fam[zyg].append(sid)
        siblings = {'':{}, 'MZ':{}, 'DZ':{}}
        for (ii,r) in genetic_table.iterrows():
            (sid,fid,zyg) = [r[k] for k in ('Subject','Family_ID','ZygosityGT')]
            zyg = zyg.strip()
            sib = siblings[zyg]
            rel = families[fid][zyg]
            sib[sid] = [k for k in rel if k != sid]
        # clean the siblings up (twins have only one entry so clear the lists)
        for tw in ['DZ','MZ']: siblings[tw] = {k:v[0] for (k,v) in six.iteritems(siblings[tw])}
        siblings[''] = {k:v for (k,v) in six.iteritems(siblings['']) if len(v) > 0}
        return pimms.persist(siblings)
    @pimms.value
    def retinotopy_siblings(siblings):
        '''
        retinotopy_siblings is a mapping like siblings but restricted to just the subjects with
        retinotopic maps.
        '''
        # make the retinotopy subset of subjects:
        slist = HCPRetinotopyDataset.subject_ids
        retinotopy_siblings = {
            kk: {k:v for (k,v) in six.iteritems(vv)
                 if k in slist
                 if all(u in slist for u in ([v] if pimms.is_int(v) else v))}
            for (kk,vv) in six.iteritems(siblings)}
        return pimms.persist(retinotopy_siblings)
    @staticmethod
    def _siblings_to_pairs(rs):
        subject_list = [u for v in six.itervalues(rs)
                        for uuu in [[six.iterkeys(v)], six.itervalues(v)]
                        for uu in uuu for u in ([uu] if pimms.is_int(uu) else uu)]
        subject_list = np.unique(subject_list)
        # setup twin numbers so that we can export anonymized twin data (i.e.,
        # files containing twin data but not the subject IDs)
        twin_pairs = {tw: pimms.imm_array(list(sorted(dat)))
                      for tw  in ['MZ','DZ']
                      for dat in [set([tuple(sorted([k,v])) for (k,v) in six.iteritems(rs[tw])])]}
        # also get a list of all siblings so we can track who is/isn't related
        siblings = {}
        for s1 in subject_list:
            q = []
            for sibs in six.itervalues(rs):
                if s1 not in sibs: continue
                ss = sibs[s1]
                if pimms.is_int(ss): ss = [ss]
                for s2 in ss: q.append(s2)
            if len(q) > 0: siblings[s1] = q
        # Make up a list of all possible unrelated pairs
        unrelated_pairs = []
        for sid in subject_list:
            # find a random subject to pair them with
            urs = np.setdiff1d(subject_list, [sid] + siblings.get(sid,[]))
            unrelated_pairs.append([urs, np.full(len(urs), sid)])
        unrelated_pairs = np.unique(np.sort(np.hstack(unrelated_pairs), axis=0), axis=1).T
        unrelated_pairs.setflags(write=False)
        # Having made those unrelated pairs, we can add them to the twin pairs
        twin_pairs['UR'] = unrelated_pairs
        # finally, let's figure out the non-twin siblings:
        sibs = [(k,v) for (k,vv) in six.iteritems(rs['']) for v in vv]
        twin_pairs['SB'] = np.unique(np.sort(sibs, axis=1), axis=0)
        twin_pairs['SB'].setflags(write=False)
        return pyr.pmap({'monozygotic_twins': twin_pairs['MZ'],
                         'dizygotic_twins':   twin_pairs['DZ'],
                         'nontwin_siblings':  twin_pairs['SB'],
                         'unrelated_pairs':   twin_pairs['UR']})
    @pimms.value
    def sibling_pairs(siblings):
        '''
        sibling_pairs is a persistent map of twin pairs, sibling pairs, and unrelated
        pairs; each of these categories stores a (n x 2) matrix of the n pairs of subjects
        associated with that category.

        The keys are 'monozygotic_twins', 'dizygotic_twins', 'nontwin_siblings',
        and 'unrelated_pairs'.
        '''
        return HCPMetaDataset._siblings_to_pairs(siblings)
    @pimms.value
    def retinotopy_sibling_pairs(retinotopy_siblings):
        '''
        retinotopy_sibling_pairs is a persistent map of twin pairs, sibling pairs, and
        unrelated pairs; each of these categories stores a (n x 2) matrix of the n pairs of subjects
        associated with that category. This dataset is equivalent to sibling_pairs, except that it
        is restricted to subjects with retinotopic mapping data.

        The keys are 'monozygotic_twins', 'dizygotic_twins', 'nontwin_siblings',
        and 'unrelated_pairs'.
        '''
        return HCPMetaDataset._siblings_to_pairs(retinotopy_siblings)
add_dataset('hcp_metadata', lambda:HCPMetaDataset().persist())

# HCPDataset #######################################################################################
@pimms.immutable
class HCPDataset(HCPMetaDataset):
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
                 metadata_path=None, genetic_path=None, behavioral_path=None,
                 default_alignment=Ellipsis, cache_directory=Ellipsis,
                 meta_data=None, create_directories=True, create_mode=0o755):
        cdir = cache_directory
        if cdir is Ellipsis: cdir = config['hcp_auto_path']
        if cdir is Ellipsis:
            cdir = config['hcp_subject_paths']
            if cdir is not None: cdir = next(iter(cdir), None)
        HCPMetaDataset.__init__(self, 'hcp',
                                meta_data=meta_data,
                                cache_directory=cdir,
                                create_directories=create_directories,
                                create_mode=create_mode,
                                cache_required=True,
                                metadata_path=metadata_path,
                                genetic_path=genetic_path,
                                behavioral_path=behavioral_path)
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

# HCPRetinotopyDataset #############################################################################
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

    # make sure we can grab one of the properties before we print anything...
    ptest = fs_LR_hemi.prop(prefix+hcp_retinotopy_property_names[0])
    # if that didn't fail, print the message...
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
class HCPRetinotopyDataset(HCPMetaDataset):
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
                 metadata_path=None, genetic_path=None, behavioral_path=None,
                 meta_data=None, create_directories=True, create_mode=0o755):
        cdir = cache_directory
        if cdir is Ellipsis: cdir = config['hcp_auto_path']
        if cdir is Ellipsis:
            cdir = config['hcp_subject_paths']
            if cdir is not None: cdir = next(iter(cdir), None)
        HCPMetaDataset.__init__(self, 'hcp_retinotopy',
                                meta_data=meta_data,
                                cache_directory=cdir,
                                create_directories=create_directories,
                                create_mode=create_mode,
                                cache_required=True,
                                metadata_path=metadata_path,
                                genetic_path=genetic_path,
                                behavioral_path=behavioral_path)
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
            # Before we get this from the 
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
            if not os.path.exists(pth) and create_directories:
                try:
                    os.makedirs(pth, create_mode)
                except Exception as e:
                    w = "an exception of type %s was raised while trying to make cache directory %s"
                    warnings.warn(w % (type(e), pth))
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
            if okwrite:
                try: nyio.save(flnm, dat)
                except Exception as e:
                    warnings.warn("Error when trying to save cache file: %s" % (type(e),))
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
            if sid > 999990: raise ValueError("IDs 999997-999999 are not real HCP subjects")
            elif hcp is None: raise ValueError('could not load subject object for sid %s' % (sid,))
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
                except Exception as e: x = None
                if x is not None: return x
                try: return hemi.prop(lpfx + '_' +  k)
                except Exception as e: x = None
                if x is not None: return x
                raise ValueError('no retinotopy successfully loaded for hemi', hemi)
            def _interp_nat(h, align):
                # okay, let's get the hemisphere we are modifying...
                hemi = hem0 = hems1['%s_native_%s' % (h, align)]
                # we're going to interp from both the 32k and 59k meshes (if possible)
                for res in [32, 59]:
                    # These may fail due to not having files for one or the other (if not loading
                    # from S3); this is okay, though.
                    try: hemi = _interp_hem(hemi, h, res, align)
                    except Exception: pass
                if hemi is hem0:
                    warnings.warn('unable to interpolate retinotopy for any hemisphere '
                                  'of hcp_retinotopy subject')
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
