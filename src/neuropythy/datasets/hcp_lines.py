####################################################################################################
# neuropythy/datasets/hcp_lines.py
# Code for defining the HCP-lines dataset: read in and preprocess raw data, cache it in various
# directories, and provide it to the user as a set of data-structures.
# by Noah C. Benson

import os, sys, six, warnings, pimms
import pyrsistent as pyr
import numpy      as np
import scipy      as sp
import scipy.io   as spio

# Import neuropythy items.
from .core        import (Dataset, add_dataset)
from ..util       import (config, curry, auto_dict, address_data, pseudo_path)
from ..vision     import as_retinotopy
from .hcp         import (HCPMetaDataset, to_boolean)
from ..hcp        import subject as hcp_subject


config.declare_dir('hcp_lines_path',
                   environ_name='HCP_LINES_PATH', default_value=None)
config.declare('hcp_lines_auto_download', environ_name='HCP_LINES_AUTO_DOWNLOAD',
               filter=to_boolean, default_value=True)
# useful function for currying something that needs to be try'ed/catch'ed
def try_curry(f, failval, *args, **kw):
    '''
    try_curry(f, failval, args...) is equivalent to curry(f, args) except that if an exception is
      raised during the evaluation of f, it is caught and failval is returned instead.
    '''
    def _curried(*xargs, **xkw):
        act_args = args + xargs
        act_kw = {k:v for m in (kw,xkw) for (k,v) in six.iteritems(m)}
        try:              return f(*act_args, **act_kw)
        except Exception: return failval
    return _curried
def safe_curry(f, *args, **kw):
    '''
    safe_curry(f, args...) is equivalent to curry(f, args) except that if an exception is raised
      during the evaluation of f, it is caught and None is returned instead.

    Note that safe_curry(f, ...) is equivalent to try_curry(f, None, ...).
    '''
    return try_curry(f, None, *args, **kw)
def mapsmap(f, m, skip_none=True, clear_null=True):
    '''
    mapsmap(f, m) applies the function f to the leaves of the nested-map structure m.
    '''
    def recur(k, v, into):
        if pimms.is_map(v):
            new_into = {}
            for (kk,vv) in six.iteritems(v): recur(kk, vv, new_into)
            if not clear_null or len(new_into) > 0: into[k] = new_into
        else:
            if v is not None or not skip_none:  v = f(v)
            if v is not None or not clear_null: into[k] = v
    res = {}
    for (k,v) in six.iteritems(m): recur(k, v, res)
    if clear_null and len(res) == 0: return None
    else: return res
def mapswalk(fn, ft, m, skip_none=True, clear_null=True):
    '''
    mapswalk(fn, ft, m) applies the functions fn and ft to the nodes of the nested-map structure m.
      The function fn is applied to non-terminals and the function ft is applied to terminals. The
      argument fn is called after ft has been called on the node's children.
    '''
    if fn is None or ft is None:
        ident = lambda x:x
        if fn is None: fn = ident
        if ft is None: ft = ident
    def recur(k, v, into):
        if pimms.is_map(v):
            new_into = {}
            for (kk,vv) in six.iteritems(v): recur(kk, vv, new_into)
            if not clear_null or len(new_into) > 0: into[k] = fn(new_into)
        else:
            if v is not None or not skip_none:  v = ft(v)
            if v is not None or not clear_null: into[k] = v
    res = {}
    for (k,v) in six.iteritems(m): recur(k, v, res)
    if clear_null and len(res) == 0: return None
    else: return fn(res)
def _mapsmerge_pick(maps, k):
    (v,mapq) = (None,False)
    for m in maps:
        if k not in m: continue
        u = m[k]
        if pimms.is_map(u):
            # we know this will be a map; 
            if mapq: v.append(u)
            else:    v = [u]
            mapq = True
        elif not mapq and u is not None:
            v = u
            break
    if mapq: return v[0] if len(v) == 1 else mapsmerge(*v)
    else:    return v
def mapsmerge(*maps):
    '''
    mapsmerge(maps...) is used below to merge a lazily loading cache mapping structure with a lazy
      loading calculated data structure.
    '''
    maps = [m for m in maps if m]
    if len(maps) == 0: return None
    ks = set([k for m in maps for k in six.iterkeys(m)])
    if any(pimms.is_lazy_map(m) for m in maps):
        def _pick(k):
            for m in maps:
                u = m.get(k)
                if u is not None: return u
            return None
        return pimms.lazy_map({k:curry(_pick, k) for k in ks})
    else:
        def _pick(k):
            ms = []
            for m in maps:
                if k in m: ms.append(m[k])
            return mapsmerge(*ms)
        return pyr.pmap({k:_pick(k) for k in ks})

# Here, we define a neuropythy dataset object to handle all the data from the lines project
@pimms.immutable
class HCPLinesDataset(HCPMetaDataset):
    '''
    The HCPLinesDataset class is a neuropythy dataset class that handles details of the HCP-lines
    dataset. The dataset consists of sets of path-traces and paths for each HCP subject for whom
    retinotopic mapping data is available.

    In order to use this dataset, you must either set the HCP_LINES_PATH os environment variable to
    the lines data path (i.e., the directory containing 'linesData' and 'manualdefinition'
    directories) or you must have this directory mounted in
    /acadia/<user>/Projects/HCP/retinotopy/lines. (Later this will be configurable via neuropythy's
    config interface).

    Given an object, data, that is an instance of this class, the raw lines are represented as
    neuropythy path_trace objects; these essentially store the clicked points as an array along with
    a map_projection object that describes how to make a map out of the subject's native spherical
    surface that aligns with the clicked points. These map projections are somewhat odd since the
    clicked points are stored in pixel (row,col) format, but they are accurate to the clicks made
    by the raters.

    The path_trace objects themselves are stored in nested (lazy) dictionaries; for example,
      data.raw_path_traces['A2'][100610]['lh']['iso_angle']['V1_mid']
    is the path_trace for anatomist 'A2', subject 100610, the left hemisphere V1_mid line;
      data.raw_path_traces['A1'][111312]['rh']['iso_eccen']['0.5']
    is the path trace for anatomist 'A1', subject 111312, the right hemisphere 0.5-degree
    eccentricity line. If you try to load a particular path_trace and either there is an error
    raised or the result is None, then that means that there is either an error with the line or
    the line was not drawn by that anatomist.

    In addition to the raw_path_traces, there are also native_path_traces, which are identical to
    the raw path traces aside from undergoing some clean-up such as ensuring that the lines form
    propert boundaries (e.g., V1_ventral and V1_dorsal start at the same point). Additionally, there
    are fsaverage_path_traces and fsaverage500_path_traces. The fsaverage_path_traces are made by
    projecting the native_path_traces onto the fsaverage cortical surface such that lines can be
    compared across subjects. The fsaverage500_path_traces are made by then dividing each of the
    fsaverage_path_traces into 500 uniformly-spaced points, making it easier to compare equivalent
    points along the lines between subjects. All of these valies are organized in the same hierarchy
    as the raw_path_traces dictionary structure.

    There are also raw_paths, native_paths, and fsaverage_paths; each of these represents a path
    object, which is simply a path_trace object that has been wed to the cortical surface. The
    details of this difference are somewhat unimportant here, but essentially, the path_trace object
    represents the points clicked on a 2D map while the path object represents those lines etched
    onto the cortical surface such that exact cortical distances and areas can be calculated. Paths
    are constructed from path_traces by calling pathtrace.to_path(hemisphere).
    '''
    def __init__(self,
                 cache_directory=Ellipsis, create_mode=0o775, create_directories=True,
                 metadata_path=None, genetic_path=None, behavioral_path=None, meta_data=None,
                 trust_exclusions=True, anat_distances=False):
        cdir = cache_directory
        if cdir is Ellipsis: cdir = config['hcp_lines_path']
        # If we're configured for auto-downloading, do it!
        if config['hcp_lines_auto_download']:
            self.source_path = HCPLinesDataset.osf_path
        elif cdir is None:
            raise ValueError('No HCP lines path given and auto-download is disabled')
        elif not os.path.exists(cdir):
            raise ValueError('HCP lines path does not exist and auto-download is disabled')
        elif not os.path.isdir(cdir):
            raise ValueError('HCP lines path is not a directory')
        else:
            self.source_path = cdir
            cdir = None
        self.trust_exclusions = trust_exclusions
        self.anat_distances = anat_distances
        HCPMetaDataset.__init__(self, 'hcp_lines',
                                metadata_path=metadata_path, genetic_path=genetic_path,
                                behavioral_path=behavioral_path,
                                cache_directory=cdir,
                                create_directories=create_directories,
                                create_mode=create_mode,
                                meta_data=meta_data,
                                cache_required=True)
    
    osf_path = 'osf://gqnp8/'
        
    subject_list = (100610, 118225, 140117, 158136, 172130, 182436, 197348, 214524,
                    346137, 412528, 573249, 724446, 825048, 905147, 102311, 125525,
                    144226, 159239, 173334, 182739, 198653, 221319, 352738, 429040,
                    581450, 725751, 826353, 910241, 102816, 126426, 145834, 162935,
                    175237, 185442, 199655, 233326, 360030, 436845, 585256, 732243,
                    833249, 926862, 104416, 128935, 146129, 164131, 176542, 186949,
                    200210, 239136, 365343, 463040, 601127, 751550, 859671, 927359,
                    105923, 130114, 146432, 164636, 177140, 187345, 200311, 246133,
                    380036, 467351, 617748, 757764, 861456, 942658, 108323, 130518,
                    146735, 165436, 177645, 191033, 200614, 249947, 381038, 525541,
                    627549, 765864, 871762, 943862, 109123, 131217, 146937, 167036,
                    177746, 191336, 201515, 251833, 385046, 536647, 638049, 770352,
                    872764, 951457, 111312, 131722, 148133, 167440, 178142, 191841,
                    203418, 257845, 389357, 541943, 644246, 771354, 878776, 958976,
                    111514, 132118, 150423, 169040, 178243, 192439, 204521, 263436,
                    393247, 547046, 654552, 782561, 878877, 966975, 114823, 134627,
                    155938, 169343, 178647, 192641, 205220, 283543, 395756, 550439,
                    671855, 783462, 898176, 971160, 115017, 134829, 156334, 169444,
                    180533, 193845, 209228, 318637, 397760, 552241, 680957, 789373,
                    899885, 973770, 115825, 135124, 157336, 169747, 181232, 195041,
                    212419, 320826, 401422, 562345, 690152, 814649, 901139, 995174,
                    116726, 137128, 158035, 171633, 181636, 196144, 214019, 330324,
                    406836, 572045, 706040, 818859, 901442)
    anatomist_list = ('A1', 'A2', 'A3', 'A4')
    mean_anatomist_name = 'mean'
    mean_subject_name   = 999999
    full_anatomist_list = anatomist_list + (mean_anatomist_name,)
    full_subject_list   = subject_list   + (mean_subject_name,)
    raw_angle_list = ('V1_mid', 'V1_ventral', 'V2_ventral', 'V3_ventral',
                      'V1_dorsal', 'V2_dorsal', 'V3_dorsal')
    raw_eccen_list = ('0.5', '1', '2', '4', '7')
    mean_sampling_resolution = 500
    normalized_directory_name = 'normalized'
    
    @pimms.param
    def source_path(sp):
        '''
        hcplines.source_path is the source (input) path of the HCP-lines data.
        '''
        return sp
    @pimms.value
    def pseudo_path(source_path, cache_directory):
        '''
        hcplines.pseudo_path is the pseudo-path object that handles the loading and caching of the
        HCP-lines raw data.
        '''
        pp = pseudo_path(source_path, cache_path=cache_directory)
        # If the source path is the known OSF path, we can drastically speed things up by manually
        # loading in the OSF tree.
        if source_path == HCPLinesDataset.osf_path:
            import neuropythy as ny
            try:
                tree = ny.load(os.path.join(ny.library_path(), 'data', 'hcp_lines_osftree.json.gz'))
                object.__setattr__(pp._path_data.pathmod, 'osf_tree', tree)
            except:
                s = "Could not pre-load OSF-tree; initial loading of data from the OSF may be slow"
                warnings.warn(s)
        return pp
    @pimms.param
    def anat_distances(ad):
        '''
        hcplines.anat_distances is True if distance maps for all anatomists are included in the
        hcplines.subject_boundary_distances map and False if not. The 'mean' anatomist's distance
        maps are always included.
        Typically the individual anatomist maps are not included as they are large and require a lot
        of computation time to produce.
        '''
        return bool(ad)


    # Cache ########################################################################################
    @staticmethod
    def save_hdf5(filename, data):
        '''
        Saves the given data, which must be a set of nested maps ending in valid hdf5 data (such as
          numpy arrays) to the given filename using h5py.
        '''
        import h5py
        def recur(group, name, dat):
            if pimms.is_map(dat):
                g = group.create_group(name)
                for (k,v) in six.iteritems(dat): recur(g, k, v)
            else: group.create_dataset(name, data=dat)
        with h5py.File(filename, 'w') as fl:
            for (k,v) in six.iteritems(data):
                recur(fl, k, v)
        return filename
    @staticmethod
    def load_hdf5(filename):
        '''
        Loads the given filename as an hdf5-structured map; yields a dict-structure of the results.
        '''
        import h5py
        def recur(name, entry, into):
            if isinstance(entry, h5py.Group):
                next_into = {}
                for (k,v) in entry.items(): recur(k, v, next_into)
                into[name] = next_into
            else: into[name] = np.array(entry)
        res = {}
        with h5py.File(filename, 'r') as fl:
            for (k,v) in fl.items():
                recur(k, v, res)
        return res
    @staticmethod
    def cache_path(pseudo_path, *drs, **kw):
        '''
        cache_path(pd, dirparts...) is like os.path.join(dirparts...) except that it finds the given
          cache path inside the pseudo_path directory given by pd.

        The following optional arguments may be given:
          * create_directories (default: True) specifies whether directories should be created if
            they do not already exist.
          * create_mode (default 0o755) specifies the mode for creating directories.
          * prepend_cache_directory (default: True) indicates whether the function should
            automatically prepend the name of the normalized-data directory to the path.
        '''
        k = 0
        if 'create_directories' in kw:
            create_directories = kw['create_directories']
            k += 1
        else: create_directories = True
        if 'create_mode' in kw:
            create_mode = kw['create_mode']
            k += 1
        else: create_mode = 0o755
        if 'prepend_cache_directory' in kw:
            prepend_cache_directory = kw['prepend_cache_directory']
            k += 1
        else: prepend_cache_directory = True
        if k != len(kw): raise ValueError('Unrecognized keyword arguments given to cache_path')
        if prepend_cache_directory: drs = (HCPLinesDataset.normalized_directory_name,) + drs
        drs = [str(dd) for dd in drs]
        try: pth = pseudo_path.local_path(*drs)
        except Exception: pth = None
        if pth is None:
            pth = pseudo_path.local_cache_path(*drs)
            if create_directories:
                pdir = os.path.dirname(pth)
                if not os.path.isdir(pdir): os.makedirs(pdir, mode=create_mode)
        return pth
    @staticmethod
    def find_path(pseudo_path, *drs, **kw):
        '''
        HCPLinesDataset.find_path(...) is like HCPLinesDataset.cache_path(...) except that it does
          not create the directory and only returns either None (if the directory does not exist)
          or the directory itself.
        '''
        if 'prepend_cache_directory' in kw:
            prepend_cache_directory = kw.pop('prepend_cache_directory')
        else: prepend_cache_directory = True
        if 0 != len(kw): raise ValueError('Unrecognized keyword arguments given to find_path')
        if prepend_cache_directory: drs = (HCPLinesDataset.normalized_directory_name,) + drs
        drs = [str(dd) for dd in drs]
        tmp = pseudo_path.find(*drs)
        if tmp is None: return None
        try: return pseudo_path.local_path(*drs)
        except Exception: return None
    @staticmethod
    def save_paths_file(filename, paths):
        '''
        Saves the address path data to an hdf5-formatted file for a single subject.
        '''
        struct = mapsmap(lambda p:p.addresses, paths)
        HCPLinesDataset.save_hdf5(filename, struct)
        return filename
    @staticmethod
    def load_paths_file(filename, sid):
        '''
        Loads the address path data from an hdf5-formatted file for a single subject.
        '''
        from neuropythy.geometry import Path
        sub = hcp_subject(sid)
        struct = HCPLinesDataset.load_hdf5(filename)
        # first level of map is always hemi:
        m = {h: mapswalk(lambda q: (Path(sub.hemis[h], q) if 'faces' in q else q), None, hdat)
             for (h,hdat) in six.iteritems(struct)}
        return pimms.persist(m)
    def save_paths(self, anat, sid, name, overwrite=False,
                   create_directories=True, create_mode=0o755):
        '''
        save_paths(anat, sid, name) saves cache files for the given anatomist and subject ID; the
          type of path saved is determined by name, which may either be one of 'raw', 'native', 
          'fsaverage', 'area', or 'sector'. If no file was written (e.g., because the anatomist or
          subject did not exist) then None is returned; otherwise the filename is returned. If
          overwrite is False but the data exists, then the filename is returned as if the data were
          successfully written.
        '''
        name = name.lower()
        data = (self.raw_paths       if name == 'raw'       else
                self.native_paths    if name == 'native'    else
                self.fsaverage_paths if name == 'fsaverage' else
                self.native_areas    if name == 'area'      else
                self.native_sectors  if name == 'sector'    else
                None)
        if data is None: raise ValueError('Unknown path type: %s' % name)
        # make sure the anat/subject exist...
        data = data.get(anat, {}).get(sid)
        if data is None: return None
        pp = HCPLinesDataset.cache_path(self.pseudo_path, anat, '%s.%s_paths.hdf5' % (str(sid),name),
                                        create_directories=create_directories,
                                        create_mode=create_mode)
        if not overwrite and os.path.isfile(pp): return pp
        self.save_paths_file(pp, data)
        return pp
    @staticmethod
    def load_paths(pseudo_path, anat, sid, name):
        '''
        load_paths(pd, anat, sid, name) loads the cache for the given anatomist, subject, and data
          name; the name must be one of 'raw', 'native', 'fsaverage', 'area', or 'sector'. The data
          are loaded from the given pseudo_path pd.

        If no cache data for the given anatomist and sid are found, then None is returned.
        '''
        name = name.lower()
        if name not in ['raw', 'native', 'fsaverage', 'area', 'sector']:
            raise ValueError('Unknown path type: %s' % name)
        pp = HCPLinesDataset.cache_path(pseudo_path, anat, '%s.%s_paths.hdf5' % (sid, name),
                                        create_directories=False)
        if pp is None or not os.path.isfile(pp): return None
        else: return HCPLinesDataset.load_paths_file(pp, sid)
    def save_traces(self, anat, sid, name, overwrite=False,
                    create_directories=True, create_mode=0o755):
        '''
        save_traces(anat, sid, name) saves cache files for the given anatomist and subject ID; the
          type of path-trace saved is determined by name, which must be one of 'raw', 'native', 
          'fsaverage', 'fsaverage500', 'area', or 'sector'. If no file was written (e.g., because
          the anatomist or subject did not exist) then None is returned; otherwise the filename is
          returned. If overwrite is False but the data exists, then the filename is returned as if
          the data were successfully written.
        '''
        from neuropythy import save
        name = name.lower()
        data = (self.raw_path_traces          if name == 'raw'          else
                self.native_path_traces       if name == 'native'       else
                self.fsaverage_path_traces    if name == 'fsaverage'    else
                self.fsaverage500_path_traces if name == 'fsaverage500' else
                self.native_area_traces       if name == 'area'         else
                self.native_sector_traces     if name == 'sector'       else
                None)
        if data is None: raise ValueError('Unknown trace type: %s' % name)
        # make sure the anat/subject exist...
        data = data.get(anat, {}).get(sid)
        if data is None: return None
        pp = HCPLinesDataset.cache_path(self.pseudo_path, anat,
                                        '%s.%s_traces.json.gz' % (str(sid), name),
                                        create_directories=create_directories,
                                        create_mode=create_mode)
        if not overwrite and os.path.isfile(pp): return pp
        return save(pp, data, 'json')
    @staticmethod
    def load_traces(pseudo_path, anat, sid, name):
        '''
        load_traces(pd, anat, sid, name) loads the cache for the given anatomist, subject, and data
          name; the name must be one of 'raw', 'native', 'fsaverage', 'fsaverage500', 'area', or
          'sector'. The data is loaded from the given pseudo_path pd.

        If no cache data for the given anatomist and sid are found, then None is returned.
        '''
        from neuropythy import load
        name = name.lower()
        if name not in ['raw', 'native', 'fsaverage', 'fsaverage500', 'area', 'sector']:
            raise ValueError('Unknown path type: %s' % name)
        pp = HCPLinesDataset.cache_path(pseudo_path, anat, '%s.%s_traces.json.gz' % (sid, name),
                                        create_directories=False)
        if pp is None or not os.path.isfile(pp): return None
        else: return load(pp, 'json')
    def save_traces(self, anat, sid, name, overwrite=False,
                    create_directories=True, create_mode=0o755):
        '''
        save_traces(anat, sid, name) saves cache files for the given anatomist and subject ID; the
          type of path-trace saved is determined by name, which must be one of 'raw', 'native', 
          'fsaverage', 'fsaverage500', 'area', or 'sector'. If no file was written (e.g., because
          the anatomist or subject did not exist) then None is returned; otherwise the filename is
          returned. If overwrite is False but the data exists, then the filename is returned as if
          the data were successfully written.
        '''
        from neuropythy import save
        name = name.lower()
        data = (self.raw_path_traces          if name == 'raw'          else
                self.native_path_traces       if name == 'native'       else
                self.fsaverage_path_traces    if name == 'fsaverage'    else
                self.fsaverage500_path_traces if name == 'fsaverage500' else
                self.native_area_traces       if name == 'area'         else
                self.native_sector_traces     if name == 'sector'       else
                None)
        if data is None: raise ValueError('Unknown trace type: %s' % name)
        # make sure the anat/subject exist...
        data = data.get(anat, {})
        data = None if data is None else data.get(sid)
        if data is None: return None
        pp = HCPLinesDataset.cache_path(self.pseudo_path, anat,
                                        '%s.%s_traces.json.gz' % (str(sid), name),
                                        create_directories=create_directories,
                                        create_mode=create_mode)
        if not overwrite and os.path.isfile(pp): return pp
        return save(pp, data, 'json')
    def save_surface_areas(self, anat, sid, name, overwrite=False,
                           create_directories=True, create_mode=0o755):
        '''
        save_surface_areas(anat, sid, name) saves cache files for the given anatomist and subject
          ID; the type of surface area saved should be either 'sct' or 'roi'. If no file was written
          (e.g., because the anatomist or subject did not exist) then None is returned; otherwise
          the filename is returned. If overwrite is False but the data exists, then the filename is
          returned as if the data were successfully written.
        '''
        from neuropythy import save
        name = name.lower()
        data = (self.area_surface_areas   if name == 'roi' else
                self.sector_surface_areas if name == 'sct' else
                self.label_surface_areas  if name == 'lbl' else
                None)
        if data is None: raise ValueError('Unknown surface-area type: %s' % name)
        # make sure the anat/subject exist...
        if name != 'lbl': data = data.get(anat, None)
        data = None if data is None else data.get(sid)
        if data is None: return None
        pp = HCPLinesDataset.cache_path(self.pseudo_path, anat,
                                        '%s.%s_sareas.json.gz' % (str(sid), name),
                                        create_directories=create_directories,
                                        create_mode=create_mode)
        if not overwrite and os.path.isfile(pp): return pp
        return save(pp, data, 'json')
    def save_properties(self, anat, sid, name,
                        overwrite=False, create_directories=True, create_mode=0o755):
        '''
        save_properties(pd, anat, sid, name) saves cache files for the given anatomist, subject ID,
          and data-name. The data name must be 'labels' (subject_labels) or 'distances'
          (boundary_distances). If no file was written (e.g., because the anatomist or subject did
          not exist) then None is returned; otherwise the filename is returned. If overwrite is
          False but the data exists, then the filename is returned as if the data were successfully
          written. The data is loaded from the given pseudo_path pd.
        '''
        # make sure the anat/subject exist...
        name = name.lower()
        if   name == 'labels':    data = self.subject_labels
        elif name == 'distances': data = self.subject_boundary_distances
        elif name == 'clean':     data = self.clean_retinotopic_maps
        elif name == 'cmag':      data = self.subject_cortical_magnifications
        else: raise ValueError('Property name must be "labels" or "distances"')
        data = data.get(anat, {})
        if data is None: return None
        data = data.get(sid)
        if data is None: return None
        pp = HCPLinesDataset.cache_path(self.pseudo_path, anat, '%s.%s.hdf5' % (str(sid), name),
                                        create_directories=create_directories,
                                        create_mode=create_mode)
        if not overwrite and os.path.isfile(pp): return pp
        HCPLinesDataset.save_hdf5(pp, data)
        return pp
    @staticmethod
    def load_properties(pseudo_path, anat, sid, name):
        '''
        load_properties(pd, anat, sid, name) loads the properties cache for the given anatomist,
          subject, and data-name, which must be either 'labels' or 'distances'. If no cache data for
          the given anatomist and sid are found, then None is returned. The data is loaded from the
          given pseudo_path pd.
        '''
        name = name.lower()
        if name not in ['labels', 'distances', 'clean', 'cmag']:
            raise ValueError('Property name must be "labels", "distances", "cmag", or "clean"')
        pp = HCPLinesDataset.cache_path(pseudo_path, anat, '%s.%s.hdf5' % (sid, name),
                                        create_directories=False)
        if pp is None or not os.path.isfile(pp): return None
        return HCPLinesDataset.load_hdf5(pp)
    @staticmethod
    def load_dataframe(pseudo_path, *paths):
        '''
        load_dataframe(pd, paths...) loads the dataframe hdf5 cache, if it exists, and returns it
          as a pandas dataframe object. The path of the dataframe object is given by the pseudo-path
          pd plus the list of directories and files in paths. These paths are searched using
          pd.find(paths...). If the file does not exist, this function yields None.

        Note that the 'normalized' directory is automatically prepended.
        '''
        import pandas
        p = HCPLinesDataset.find_path(pseudo_path, *paths)
        if p is None: return None
        try: return pandas.read_hdf(p, 'dataframe')
        except Exception:
            warnings.warn('hcp-lines: Failed to read dataframe file: %s' % p)
            return None
    @staticmethod
    def load_surface_areas(pseudo_path, anat, sid, name):
        '''
        load_surface_areas(pd, anat, sid, name) loads either the 'roi' or 'sct' surface area data
          for the given anatomist and subject id.
        '''
        from neuropythy import load
        name = name.lower()
        if name not in ['roi', 'sct', 'lbl']:
            raise ValueError('Unknown surface area type: %s' % name)
        if name == 'lbl':
            # no anatomist in this case!
            pp = HCPLinesDataset.cache_path(pseudo_path, 'mean', '%s.lbl_sareas.json.gz' % sid,
                                            create_directories=False)
        else:
            pp = HCPLinesDataset.cache_path(pseudo_path, anat, '%s.%s_sareas.json.gz' % (sid,name),
                                            create_directories=False)
        if pp is None or not os.path.isfile(pp): return None
        else: return load(pp, 'json')
    @pimms.value
    def _cached_data(pseudo_path):
        '''
        _cached_data is a pimms lazy-map of all the cached data that is found in the HCP-lines
        dataset. Any cached data that is not found is automatically generated and saved when
        requested.
        '''
        import h5py
        from neuropythy import load
        # see what anatomist directories are there
        anatomists = HCPLinesDataset.full_anatomist_list
        subjects   = HCPLinesDataset.subject_list
        anatomists = tuple([anat for anat in anatomists
                            if pseudo_path.find('normalized', anat) is not None])
        pd         = pseudo_path
        hh         = HCPLinesDataset
        # we just build up lazy-maps that load in the content as requested
        lmap = pimms.lazy_map
        c = [{(name + '_path_traces'): {a: lmap({s: curry(hh.load_traces, pd, a, s, name)
                                                 for s in subjects})
                                        for a in anatomists}
              for name in ('raw','native','fsaverage','fsaverage500','area','sector')},
             {(name + '_paths'): {a: lmap({s: curry(hh.load_paths, pd, a, s, name)
                                           for s in subjects})
                                  for a in anatomists}
              for name in ('raw','native','fsaverage','area','sector')},
             {name: {a: lmap({s: curry(hh.load_properties, pd, a, s, name)
                              for s in subjects})
                     for a in anatomists}
              for name in ('labels', 'distances', 'clean', 'cmag')}]
        c = {k:pyr.pmap(v) for m in c for (k,v) in six.iteritems(m)}
        c['subject_dataframes'] = lmap(
            {s: curry(hh.load_dataframe, pseudo_path, 'mean', str(s), 'dataframe.hdf5')
             for s in subjects})
        c['dataframe'] = curry(hh.load_dataframe, pseudo_path, 'dataframe.hdf5')
        # surface area data...
        c['area_surface_areas'] = {a: lmap({s: curry(hh.load_surface_areas, pd, a, s, 'roi')
                                            for s in subjects})
                                   for a in anatomists}
        c['sector_surface_areas'] = {a: lmap({s: curry(hh.load_surface_areas, pd, a, s, 'sct')
                                              for s in subjects})
                                     for a in anatomists}
        c['label_surface_areas'] = lmap({s: curry(hh.load_surface_areas, pd, 'mean', s, 'lbl')
                                         for s in subjects})
        c['surface_area_dataframe'] = curry(hh.load_dataframe, pd, 'surface_areas.hdf5')
        return pimms.lazy_map(c)
    @staticmethod
    def load_raw_data(pseudo_path, anat, sid):
        '''
        load_raw_data(filename) loads the MATLAB file given by the given filename; this must be
          a file made for use with the HCPLinesDataset.
        '''
        from scipy.io import loadmat
        try: filename = pseudo_path.local_path('raw', anat, str(sid) + '.mat')
        except Exception: filename = None
        if filename is None: raise ValueError('No raw data file for %s / %s found' % (anat, sid))
        data = loadmat(filename)
        if int(data['subject'][0]) != int(sid) or data['anatomist'][0] != anat:
            raise ValueError('corrput raw file: wrong subject or anatomist')
        affs = {'lh': data['affine'][0], 'rh': data['affine'][1]}
        try: comment = data['comment'].astype(str)[0].strip()
        except Exception: comment = None
        conf   = {h: {nm[len(pref):]: np.squeeze(data['confidence'][nm][0][0]).astype('bool')
                      for nm in data['confidence'].dtype.names if nm.startswith(pref)}
                  for (h,pref) in zip(['lh','rh'], ['left_','right_'])}
        traces = {h: {nm[len(pref):]: data['traces'][nm][0][0]
                      for nm in data['traces'].dtype.names if nm.startswith(pref)}
                  for (h,pref) in zip(['lh','rh'], ['left_','right_'])}
        return pimms.persist({'affines': affs, 'traces': traces, 'comment': comment,
                              'confidence': conf})
    @pimms.value
    def raw_data(pseudo_path):
        '''
        hcplines.raw_data is a persistent map of the raw data in the HCP-lines dataset. It is a
        nested mapping data-structure in which the raw_data[a] for an anatomist a is itself a
        mapping; raw_data[a][sid] for subject-id sid itself contains keys 'affines' and 'traces',
        each of which contains a key for each hemisphere. The data are loaded lazily at the
        subject level.
        '''
        anatomists = HCPLinesDataset.anatomist_list
        subjects   = HCPLinesDataset.subject_list
        loadfn     = HCPLinesDataset.load_raw_data
        return pyr.pmap({anat: pimms.lazy_map({sid: curry(loadfn, pseudo_path, anat, sid)
                                               for sid in subjects})
                         for anat in anatomists})
    @pimms.value
    def anatomist_comments(raw_data):
        '''
        hcplines.anatomist_comments is a nested pimms lazy-map structure whose keys are the
        anatomists then subject IDs. The values of these nested maps are the general comments
        (strings) made by the anatomists while drawing the subject lines.

        Comments were written by the anatomist and are not expected to follow any particular format
        or lexicon.
        '''
        return pyr.pmap(
            {anat: pimms.lazy_map({sid: curry(lambda a,s: raw_data[a][s]['comment'], anat, sid)
                                   for sid in HCPLinesDataset.subject_list})
             for anat in HCPLinesDataset.anatomist_list})
    @pimms.value
    def anatomist_confidence(raw_data):
        '''
        hcplines.anatomist_confidence is a nested pimms lazy-map structure whose keys are the
        anatomists -> subject IDs -> hemispheres -> 'iso_angle' or 'iso_eccen' -> raw-line-name;
        this is identical to the structure of hcplines.raw_path_traces. Each of the eventual values
        gives the anatomist's declared confidence in the points along the raw line to which the
        value corresponds. The confidence is binary (1 indicating "certain" and 0 indicating
        "not-certain") and is stored in an array of boolean values, one per point in the raw path
        trace.
        '''
        return pyr.pmap(
            {anat: pimms.lazy_map({sid: curry(lambda a,s: raw_data[a][s]['confidence'], anat, sid)
                                   for sid in HCPLinesDataset.subject_list})
             for anat in HCPLinesDataset.anatomist_list})
    @pimms.value
    def subject_errors(pseudo_path):
        '''
        hcplines.subject_errors is a persistent mapping structure whose keys are the HCP subjects
        and whose values document the errors that occurred during the processing of the HCP line
        data. Errors generally correspond to topographic errors in the lines drawn by an anatomist
        such as iso-eccentricity lines that intersect each other or V1 boundaries that do not
        intersect the 0.5 degree iso-eccentricity line.
        '''
        import neuropythy as ny
        try: dat = ny.load(pseudo_path.local_path('normalized', 'logs', 'errors.json'))
        except Exception: dat = None
        return pimms.persist({int(k):v for (k,v) in six.iteritems(dat)})
    @pimms.value
    def exclusions(pseudo_path):
        '''
        hcplines.exclusions is a frozenset whose elements are tuples of (anatomist, sid, hemi) that
        should been excluded from calculations due to errors encountered during processing of the
        anatomists' lines. For more information see the help for subejct_errors as well. Tuples
        ('mean' subject_id, hemi) are also included in the exclusions for all subject/hemi pairs
        that have an error in more than 1 anatomist.

        If the exclusions file is not found, then an empty set is returned.
        '''
        import neuropythy as ny
        if pseudo_path.find('normalized', 'logs', 'exclusions.json') is None: return frozenset([])
        dat = ny.load(pseudo_path.local_path('normalized', 'logs', 'exclusions.json'), to=None)
        dat = [(anat,int(sid),h) for (anat,sid,h) in dat]
        tmp = {}
        for (anat,sid,h) in dat:
            tup = (sid,h)
            if tup in tmp: tmp[tup] += 1
            else: tmp[tup] = 1
        dat = set(dat)
        for (k,v) in six.iteritems(tmp):
            if v > 1: dat.add(('mean',) + k)
        return frozenset(dat)
    @pimms.value
    def subject_affines(raw_data):
        '''
        subject_affines is a pimms lazy-map whose keys are subject IDs and whose values are each
        maps of {'lh': lh_affine, 'rh': rh_affine}. The affines themselves align the subjects'
        native FreeSurfer spherical surfaces such that the resulting (x,y) coordinates are suitable
        for orthographic projection. Such a projection matches the pixels given in subject_points.
        These affines are suitable for construction of a map_projection object.

        Notes:
          * The affines transform into *pixel*-space, so the y-value is decreasing from top to
            bottom.
          * However, the x-value (affine row/col 0) corresponds to image columns and the y-value 
            (affine row/col 1) corresponds to rows; this is reflected in the raw data itself, which
            encodes (cols,rows) as (x,y).
        '''
        sids = HCPLinesDataset.subject_list
        def affs(sid):
            d = next((a[sid] for a in six.itervalues(raw_data) if sid in a), None)
            if d is None or d.get('affines') is None:
                raise ValueError('No raw data found for subject %s' % str(sid))
            return d['affines']
        return pimms.lazy_map({sid:curry(affs, sid) for sid in sids})
    @pimms.value
    def subject_map_projections(subject_affines, _cached_data):
        '''
        subject_map_projections is a pimms lazy-map whose keys are the subject IDs and whose values
        are each maps of {'lh': lh_map_proj, 'rh': rh_map_proj}. The map projections given are those
        used to construct the maps that match up to the raw line data drawn by anatomists.
        '''
        from neuropythy import map_projection
        sids = HCPLinesDataset.subject_list
        def mps_from_cache(sid):
            dat = _cached_data.get('raw_path_traces')
            if dat is None: return (None,None)
            (lmp,rmp) = (None,None)
            for (anat,adat) in six.iteritems(dat):
                sdat = adat.get(sid)
                if sdat is None: continue
                (ldat,rdat) = [sdat.get(h) for h in ('lh','rh')]
                (lmp,rmp) = [
                    (mp   if mp is not None else
                     None if h  is     None else
                     next((x.map_projection
                           for g in six.itervalues(h) if g is not None
                           for x in six.itervalues(g)),
                          None))
                    for (mp,h) in zip([lmp,rmp],[ldat,rdat])]
                if lmp is not None and rmp is not None: break
            return (lmp,rmp)
        def make_mp(sid):
            (lmp,rmp) = mps_from_cache(sid)
            if lmp is None:
                affs = subject_affines[sid]
                lmp = map_projection(affs['lh'], 'lh', method='orthographic')
            else: affs = None
            if rmp is None:
                if affs is None: affs = subject_affines[sid]
                rmp = map_projection(affs['rh'], 'rh', method='orthographic')
            return pyr.m(lh=lmp, rh=rmp)
        return pimms.lazy_map({sid:curry(make_mp, sid) for sid in sids})
    @staticmethod
    def _calc_raw_traces(raw_data, anat, sid):
        '''
        _calc_raw_traces(raw_data, subject_map_projections anat_index, anat, sub_index, sub)
          calculates and yields a nested-map-structure of the raw path-traces for the given
          anatomist and subject based on the given map projections and raw data.
        '''
        from neuropythy import (path_trace, map_projection)
        dat = raw_data.get(anat,{}).get(sid)
        if dat is None: return None
        else: (affines, traces) = (dat['affines'], dat['traces'])
        r = {}
        for (h,htr) in six.iteritems(traces):
            mp = map_projection(affines[h], h, method='orthographic')
            rr = {}
            for ang in ['iso_angle','iso_eccen']:
                rrr = {}
                for (k,v) in six.iteritems(htr):
                    if   len(v) == 0:                          continue
                    elif (ang == 'iso_eccen') != ('ecc' in k): continue
                    elif k == 'ecc_0pt5':                      k = '0.5'
                    elif k.startswith('ecc_'):                 k = k[4:]
                    mdat = {'subject_id':sid, 'anatomist':anat, 'hemi':h, 'line_name':k}
                    rrr[k] = path_trace(mp, v, meta_data=mdat)
                if len(rrr) > 0: rr[ang] = pyr.pmap(rrr)
            if len(rr) > 0: r[h] = pyr.pmap(rr)
        return pyr.pmap(r) if len(r) > 0 else None
    @pimms.value
    def raw_path_traces(raw_data, _cached_data):
        '''
        raw_path_traces is a nested lazy-map structure of path-trace objects; the first layer of the
        map is the anatomist; the second layer is the subject ID; the third layer is the hemisphere;
        and the final layers are 'iso_angle' or iso_eccen' and finally the line value (e.g.,
        'V1_ventral' or '2' (for eccen = 2).
        '''
        anatomists = HCPLinesDataset.anatomist_list
        subjects   = HCPLinesDataset.subject_list
        m = pyr.pmap(
            {anat:pimms.lazy_map({sid: curry(HCPLinesDataset._calc_raw_traces, raw_data, anat, sid)
                                  for sid in subjects})
             for anat in anatomists})
        return mapsmerge(_cached_data.get('raw_path_traces', {}), m)
    @staticmethod
    def _clean_raw_traces(raw_path_traces, anat, sid, exclusions, trust_exclusions):
        '''
        _clean_raw_traces(raw_path_traces, anat, sid, e, te) yields a cleand ('native') set of path
          traces for the given anatomist and subject using the given raw_path_traces data. The final
          argument te (trust exclusions) can be False to force caluclation of lines that are in the
          penultimate exclusions (e) argument.
        '''
        from neuropythy.util import curve_intersection, close_curves, curve_spline
        # okay, we generate it anew
        ptrs   = raw_path_traces[anat][sid]
        def points_close(a,b,d=0.5):
            if d == 0: return np.isclose(a, b).all()
            else:      return (np.linalg.norm(np.asarray(a) - b) < d)
        def hcat(*args): return np.hstack([np.reshape(x,(2,-1)) for x in args])
        # we use this grid to avoid situations where both ventral and dorsal lines intersect an
        # iso-eccentricity line at two places
        grid = np.linspace(0.05, 0.95, 20)
        if ptrs is None:
            warnings.warn('Incomplete set of lines for %s / %s / %s' % (anat,sid,'lh'))
            warnings.warn('Incomplete set of lines for %s / %s / %s' % (anat,sid,'rh'))
            return None
        r = {}
        for h in six.iterkeys(ptrs):
            if trust_exclusions and (anat,sid,h) in exclusions: continue
            (rang,recc) = ({},{})
            (borders,mids) = ({},{})
            fail = False
            # for the angle lines, they go from their foveal intersection to the 7 eccen line
            (angptrs,eccptrs) = [ptrs[h].get(k) for k in ('iso_angle', 'iso_eccen')]
            if angptrs is None or len(angptrs) < 7 or eccptrs is None or len(eccptrs) < 5:
                warnings.warn('Incomplete set of lines for %s / %s / %s' % (anat,sid,h))
                continue
            if angptrs is None or eccptrs is None: continue
            # note that if this function is called with non-raw data we will have extra keys; go
            # ahead and clean them out here:
            angptrs = {k:v for (k,v) in six.iteritems(angptrs)
                       if k in HCPLinesDataset.raw_angle_list}
            eccptrs = {k:v for (k,v) in six.iteritems(eccptrs)
                       if k in HCPLinesDataset.raw_eccen_list}
            # start by processing the eccentricity lines into halves; for this, we need the v3 outer
            # lines and the v1 midline; but the v3 outer lines need to be processed using the foveal
            # eccentricity line, so start there...
            sorted_eptrs = sorted(list(six.iteritems(eccptrs)), key=lambda x:float(x[0]))
            fov = eccptrs[sorted_eptrs[0][0]].curve
            per = eccptrs[sorted_eptrs[-1][0]].curve
            v3vnt = angptrs.get('V3_ventral')
            v3drs = angptrs.get('V3_dorsal')
            v1mid = angptrs.get('V1_mid')
            # start by joining ventral and dorsal into a single VM line (note that all this code is
            # similar to that below in the loop for fixing eccentricity lines
            (t_vp,t_pv) = curve_intersection(v3vnt.curve, per, grid=grid)
            (t_dp,t_pd) = curve_intersection(v3drs.curve, per, grid=grid)
            (t_vd,t_dv) = curve_intersection(v3vnt.curve, v3drs.curve, grid=grid)
            vcrv = v3vnt.curve.subcurve(t_vp, t_vd)
            dcrv = v3drs.curve.subcurve(t_dv, t_dp)
            tmp = hcat(per(t_pv), vcrv.coordinates[:,1:], dcrv.coordinates[:,1:-1], per(t_pd))
            v3brd = v3vnt.copy(points=tmp)
            clen = fov.curve_length()
            t_bv = curve_intersection(v3brd.curve, fov.subcurve(0, clen/2), grid=40)[0]
            t_bd = curve_intersection(v3brd.curve, fov.subcurve(clen/2, clen), grid=40)[0]
            t_bm = 0.5*(t_bd + t_bv)
            v3mid = v3brd.curve(t_bm)
            v3vnt = v3brd.curve.subcurve(0, t_bm)
            v3drs = v3brd.curve.subcurve(t_bm, v3brd.curve.curve_length())
            # possibly these are upside-down; we know that the y-pixels for the dorsal will be
            # lower than for the ventral (remember the image-rows/y-axis flip), we can check this
            if np.mean(v3vnt.coordinates[1]) < np.mean(v3drs.coordinates[1]):
                (v3drs,v3vnt) = (v3vnt,v3drs)
            # now use the foveal curve to split this into two halves
            # we clear and re-find these below in order to get other related data on them
            (fov_k, per_k) = (None, None)
            if v1mid is None or v3vnt is None or v3drs is None: continue
            for (k,eptr) in sorted_eptrs:
                nn = eptr.curve.coordinates.shape[1] * 4
                (t_ev, t_ve) = curve_intersection(eptr.curve, v3vnt, grid=nn)
                (t_ed, t_de) = curve_intersection(eptr.curve, v3drs, grid=nn)
                (t_em, t_me) = curve_intersection(eptr.curve, v1mid.curve, grid=nn)
                (x_em, x_me) = (eptr.curve(t_em), v1mid.curve(t_me))
                # we allow a tolerance of 10 pixels here
                if not points_close(x_em, x_me, 10):
                    warnings.warn('No %s / mid intersection for %s / %s / %s' % (k,anat,sid,h))
                    continue
                elif not points_close(x_em, x_me):
                    warnings.warn('Closing %s / mid intersection for %s / %s / %s' % (k,anat,sid,h))
                x_me = x_em
                for (nm,t_eo,t_oe,ocrv) in zip(['ventral','dorsal'], [t_ev,t_ed],
                                               [t_ve,t_de], [v3vnt,v3drs]):
                    crv = eptr.curve.subcurve(t_em, t_eo)
                    (x_eo,x_oe) = (crv.coordinates[:,-1], ocrv(t_oe))
                    # we allow a tolerance of 10 pixels here
                    if not points_close(x_eo, x_oe, 10):
                        warnings.warn('No %s / %s intersection for %s / %s / %s'%(k,nm,anat,sid,h))
                        continue
                    elif not points_close(x_eo, x_oe):
                        warnings.warn(
                            'Closing %s / %s intersection for %s / %s / %s' % (k,nm,anat,sid,h))
                    x_eo = x_oe
                    x = hcat(x_oe, np.fliplr(crv.coordinates[:,1:-1]), x_me)
                    recc['%s_%s'%(k,nm)] = eptr.copy(points=x)
                (vpts, dpts) = (recc[k + '_ventral'].points, recc[k + '_dorsal'].points)
                if np.array_equal(vpts.shape, dpts.shape) and np.allclose(vpts, dpts):
                    s = 'Could not separate ventral/dorsal %s line for %s/%s/%s' % (k,anat,sid,h)
                    warnings.warn(s)
                    fail = True
                if fail: break
                (vx,dx) = (recc['%s_ventral'%k].points, recc['%s_dorsal'%k].points)
                recc[k] = eptr.copy(points=hcat(vx, np.fliplr(dx[:,:-1])))
                if fov_k is None:
                    (fov_k, fov_vnt, fov_drs) = (k, recc[k+'_ventral'], recc[k+'_dorsal'])
                per_k = k
            # we need to make sure that all the eccetricity lines fail to intersect each other:
            eptrks = [k for (k,eptr) in sorted_eptrs]
            for (ii,k1) in enumerate(eptrks):
                if k1 not in recc: continue
                crv1 = recc[k1].curve
                cl1 = crv1.curve_length()
                for k2 in eptrks[(ii+1):]:
                    if k2 not in recc: continue
                    crv2 = recc[k2].curve
                    cl2 = crv2.curve_length()
                    (t_12,t_21) = curve_intersection(crv1, crv2, grid=grid)
                    (x_12,x_21) = (crv1(t_12), crv2(t_21))
                    if (np.isfinite(t_12) and np.isfinite(t_21) and np.allclose(x_12, x_21) and
                        t_12 > 0 and t_12 < cl1 and t_21 > 0 and t_21 < cl2):
                        s = (k1,k2,anat,sid,h)
                        warnings.warn('eccen curves %s and %s intersect for %s / %s / %s' % s)
            if fail or per_k is None or per_k == fov_k: continue
            (per,fov) = (recc[per_k], recc[fov_k])
            (per_vnt,per_drs) = (recc[per_k+'_ventral'], recc[per_k+'_dorsal'])
            # That's al of the eccentricity lines processed; while doing that, we grabbed the foveal
            # and peripheral eccentricity keys and intersection data
            for area in ('V1', 'V2', 'V3'):
                # start by joining ventral and dorsal into a single VM line;
                (vnt,drs) = [angptrs[area+tag] for tag in ('_ventral', '_dorsal')]
                (t_vp,t_pv) = curve_intersection(vnt.curve, per.curve, grid=grid)
                (t_dp,t_pd) = curve_intersection(drs.curve, per.curve, grid=grid)
                (t_vd,t_dv) = curve_intersection(vnt.curve, drs.curve, grid=grid)
                vcrv = vnt.curve.subcurve(t_vp, t_vd)
                dcrv = drs.curve.subcurve(t_dv, t_dp)
                tmp = hcat(per.curve(t_pv), vcrv.coordinates[:,1:],
                           dcrv.coordinates[:,1:-1], per.curve(t_pd))
                brd = vnt.copy(points=tmp)
                blen = brd.curve.curve_length()
                # note that the border starts with the first drs point and ends with the last vnt 
                rang[area + '_outer'] = brd
                # if this is V1, then we find the midpoint as the intersection of V1_mid and this
                # outer boundary; otherwise we don't split the foveal region into ventral/dorsal
                if area == 'V1':
                    mid = angptrs['V1_mid']
                    (t_bm,t_mb) = curve_intersection(brd.curve, mid.curve)
                    (x_bm,x_mb) = (brd.curve(t_bm), mid.curve(t_mb))
                    (t_pm,t_mp) = curve_intersection(per.curve, mid.curve)
                    (x_pm,x_mp) = (per.curve(t_pm), mid.curve(t_mp))
                    if not points_close(x_bm,x_mb):
                        warnings.warn('No midline/V1 intersection for %s / %s / %s' % (anat,sid,h))
                        fail = True
                    else: x_bm = x_mb
                    if not points_close(x_pm,x_mp):
                        warnings.warn('No V1-mid/per intersection for %s / %s / %s' % (anat,sid,h))
                        fail = True
                    else: x_pm = x_mp
                    if fail: break
                    crv = mid.curve.subcurve(t_mb, t_mp)
                    x = hcat(x_bm, crv.coordinates[:,1:-1], x_pm)
                    rang['V1_mid'] = mid.copy(points=x)
                    last_mid = np.mean([x_bm,x_mb], axis=0)
                else:
                    vlen = vcrv.curve_length()
                    dlen = dcrv.curve_length()
                    t_bm = vlen + (blen - vlen - dlen)*0.5
                # split into dorsal and ventral parts; in case the dorsal and ventral lines don't
                # actually intersect, we use pieces of the brd line
                dcrv = brd.curve.subcurve(t_bm, blen)
                vcrv = brd.curve.subcurve(0, t_bm)
                rang[area+'_dorsal']  = brd.copy(points=dcrv.coordinates)
                rang[area+'_ventral'] = brd.copy(points=vcrv.coordinates)
            if fail: continue
            # We now want to do some extra checks of the topology. The main requirrements are that
            # the intersections lie in the correct order along each of the curves.
            # First, calculate all the intersections
            angkeys = [k for k in ['V3_ventral','V2_ventral','V1_ventral','V1_mid',
                                   'V1_dorsal','V2_dorsal','V3_dorsal']
                       if k in rang]
            ecckeys = [k for k in ['0.5','1','2','4','7'] if k in recc]
            isects = {(ak,ek): curve_intersection(rang[akk].curve, recc[ekk].curve, grid=grid)
                      for ak in angkeys
                      for sfx0 in ['_' + ak.split('_')[-1]]
                      for akk in [ak[:3] + 'outer']
                      for ek in ecckeys for ekk in [ek+'_ventral' if sfx0 == '_mid' else ek+sfx0]}
            # Now, check the iso-angle lines:
            for ak in angkeys:
                ixs = [isects[(ak,ek)][0] for ek in ecckeys]
                if 'ventral' in ak or 'mid' in ak: ixs = list(reversed(ixs))
                cl = rang[ak].curve.curve_length()
                if ixs != sorted(ixs):
                    tup = (anat,sid,h,ak)
                    warnings.warn('Bad ordering of eccen lines for %s / %s / %s / %s' % tup)
                    fail = True
                    break
            if fail: continue
            # Now, check the is-eccen lines:
            for ek in ecckeys:
                for (sfx,flipq) in zip(['_ventral','_dorsal'], [False,True]):
                    ixs = [isects[(ak,ek)][1] for ak in angkeys if sfx in ak]
                    if flipq:
                        ixs = list(reversed(ixs))
                    if ixs != sorted(ixs):
                        tup = (anat,sid,h,ek)
                        warnings.warn('Bad ordering of angle lines for %s / %s / %s / %s' % tup)
                        fail = True
                        break
                if fail: break
            if fail: continue
            # That's all there is to do for this hemisphere
            rr = {}
            if len(rang) > 0: rr['iso_angle'] = pyr.pmap(rang)
            if len(recc) > 0: rr['iso_eccen'] = pyr.pmap(recc)
            if len(rr) > 0: r[h] = rr
        return pyr.pmap(r) if len(r) > 0 else None
    @staticmethod
    def _calc_mean_subject_lines(data0, sid, excls):
        '''
        _calc_mean_subject_lines(data, sid, excls) calculates the mean set of trace lines across the
          anatomists represented in the given dataset of lines for the subject with the given id.
          The argument excls must give the set of exclusions, and is used to prevent inclusion of
          anatomist data that should not be used.
        '''
        res  = HCPLinesDataset.mean_sampling_resolution
        r = {}
        for (anat,adat) in six.iteritems(data0):
            if adat is None: continue
            sdat = adat.get(sid)
            if sdat is None: continue
            for h in six.iterkeys(sdat):
                if (anat,sid,h) in excls: continue
                hdat = sdat[h]
                if hdat is None: continue
                rr = r.get(h, {})
                for (ang,angdat) in six.iteritems(hdat):
                    if angdat is None: continue
                    rrr = rr.get(ang, {})
                    for (k,ptr) in six.iteritems(angdat):
                        if ptr is None: continue
                        if k not in rrr: rrr[k] = [ptr]
                        rrr[k].append(ptr.curve.linspace(res))
                    rr[ang] = rrr
                r[h] = rr
        ptrs = {h:{ang:{k:crds[0].copy(points=np.mean(crds[1:], axis=0))
                        for (k,crds) in six.iteritems(angdat)}
                   for (ang,angdat) in six.iteritems(hdat)}
                for (h,hdat) in six.iteritems(r)}
        # pass these through cleanup
        tmp = {HCPLinesDataset.mean_anatomist_name: {sid: ptrs}}
        return HCPLinesDataset._clean_raw_traces(tmp, 'mean', sid, frozenset([]), False)
    @pimms.param
    def trust_exclusions(te):
        '''
        trust_exclusions is an optional parameter of the HCPLinesDataset; if True (the default),
        then the dataset will not attempt to recalculate line datasets that are in the exclusions
        set (which is loaded from the exclusions.json file). If false, the dataset will attempt
        recalculation of these line data.
        '''
        return bool(te)
    @pimms.value
    def native_path_traces(raw_path_traces, _cached_data, exclusions, trust_exclusions):
        '''
        native_path_traces is a lazy-map structure identical to raw_path_traces with the exception
        that the path traces have been extended or shortened such that they start and end at the
        appropriate intersections between lines. Failure to discover such intersections results in
        an error being raised (albeit at the time of they lazy evaluation rather than at generation.
        '''
        # for each subject, we want to load from cached data if found otherwise generate:
        # That function does all the actual work; we now just process it into a lazy map:
        data0 = pyr.pmap(
            {anat:pimms.lazy_map(
                {sid:curry(HCPLinesDataset._clean_raw_traces,
                           raw_path_traces, anat, sid, exclusions, trust_exclusions)
                 for sid in six.iterkeys(adat)})
             for (anat,adat) in six.iteritems(raw_path_traces)})
        data = mapsmerge(_cached_data.get('native_path_traces', {}),
                         data0)
        def f(dat, sid, excl): return HCPLinesDataset._calc_mean_subject_lines(dat, sid, excl)
        # okay, the one thing we want to add is a mean anatomist
        excl = exclusions if trust_exclusions else frozenset([])
        mnlns = pimms.lazy_map({sid:curry(f, data, sid, excl)
                                for sid in HCPLinesDataset.subject_list})
        return mapsmerge(data, {HCPLinesDataset.mean_anatomist_name:mnlns})
    @staticmethod
    def _traces_to_paths(traces, name, anat, sid):
        '''
        _traces_to_paths(traces, name, anat, sid) yields the result of converting the path-trace
        nested-map data-structure sdat[anat][sid] into paths using the HCP subject with the given
        sid.
        '''
        from neuropythy.geometry import Path
        dat = traces
        for k in [anat, sid]:
            if dat is not None:
                dat = dat.get(k, None)
        if dat is None: return None
        sub = hcp_subject(sid)
        r = {}
        for (h,hdat) in six.iteritems(dat):
            rr = {}
            fmap = None
            for (ang,angdat) in six.iteritems(hdat):
                rrr = {}
                for (k,pt) in six.iteritems(angdat):
                    if pt is None: continue
                    elif fmap is None: fmap = pt.map_projection(sub.hemis[h])
                    try:
                        p = pt.to_path(sub.hemis[h], flatmap=fmap)
                        # for some reason, some of the transfers to fsaverage get screwy, probably
                        # because the fsaverage warping is sufficiently weird to invert triangles,
                        # but unknown yet; a simple fix is to remove the points with nan addresses
                        (fs,xs) = (p.addresses['faces'],p.addresses['coordinates'])
                        ii = np.where(np.isfinite(xs[0]))[0]
                        if len(ii) < 2: raise ValueError('problematic trace addresses')
                        if len(ii) < xs.shape[1]:
                            (fs,xs) = [u[:,ii] for u in (fs,xs)]
                            p = Path(sub.hemis[h], {'faces': fs, 'coordinates': xs})
                        rrr[k] = p
                    except Exception as e:
                        warnings.warn('failed to render %s path %s/%s/%s/%s: %s'
                                      % (name, anat, sid, h, k, str(e)))
                if len(rrr) > 0: rr[ang] = pyr.pmap(rrr)
            if len(rr) > 0: r[h] = pyr.pmap(rr)
        return pyr.pmap(r) if len(r) > 0 else None
    @staticmethod
    def _all_traces_to_paths(dat, name):
        '''
        Converts all anatomists and subjects using the above function, but in a lazy map at the
        subject-level.
        '''
        return pyr.pmap(
            {anat: pimms.lmap({s: curry(HCPLinesDataset._traces_to_paths, dat, name, anat, s)
                               for s in six.iterkeys(adat)})
             for (anat,adat) in six.iteritems(dat)})
    @pimms.value
    def raw_paths(_cached_data, raw_path_traces):
        '''
        raw_paths is identical to raw_path_traces except that it represents the raw paths on the
        HCP subject's native cortical surface.
        '''
        return mapsmerge(_cached_data.get('raw_paths', {}),
                         HCPLinesDataset._all_traces_to_paths(raw_path_traces, 'raw'))
    @pimms.value
    def native_paths(_cached_data, native_path_traces):
        '''
        native_paths is identical to native_path_traces except that it represents the raw paths on
        the HCP subject's native cortical surface.
        '''
        return mapsmerge(_cached_data.get('native_paths', {}),
                         HCPLinesDataset._all_traces_to_paths(native_path_traces, 'native'))
    @staticmethod
    def _native_paths_to_fsaverage_traces(native_paths, anat, sid):
        '''
        _native_paths_to_fsaverage_traces(native_paths, anat, sid) converts the native path data for
          the given anatomist and subject into the fsaverage space and yields a hierarchical map
          structure of those data as path-traces.
        '''
        from neuropythy import (map_projection, path_trace)
        fsmps = {h: map_projection('occipital_pole', h) for h in ('lh','rh')}
        sdat = native_paths[anat][sid]
        if sdat is None: return None
        sub = hcp_subject(sid)
        r = {}
        for h in six.iterkeys(sdat):
            hemi = sub.hemis[h]
            rr = {}
            for (dr, ddat) in six.iteritems(sdat[h]):
                rrr = {}
                for (lnm, pth) in six.iteritems(ddat):
                    fs_pts = hemi.registrations['fsaverage'].unaddress(pth.addresses)
                    fsmap_pts = fsmps[h](fs_pts)
                    # check that these don't get too close to each other
                    rrr[lnm] = path_trace(fsmps[h], fsmap_pts, closed=False).persist()
                if len(rrr) > 0: rr[dr] = pyr.pmap(rrr)
            if len(rr) > 0: r[h] = pyr.pmap(rr)
        return pyr.pmap(r) if len(r) > 0 else None
    @pimms.value
    def fsaverage_path_traces(native_paths, _cached_data):
        '''
        fsaverage_path_traces are roughly equivalent to native_path_traces, except that they have
        been projected onto the fsaverage cortical surface.
        '''
        # we use the native paths to generate these so that we have all the line/edge intersects
        f = HCPLinesDataset._native_paths_to_fsaverage_traces
        return mapsmerge(
            _cached_data.get('fsaverage_path_traces', {}),
            pyr.pmap({a: pimms.lazy_map({sid: curry(f, native_paths, a, sid)
                                         for sid in six.iterkeys(d)})
                      for (a,d) in six.iteritems(native_paths)}))
    @staticmethod
    def average_anatomist_traces(fsaverage500_path_traces, anat=None):
        '''
        average_anatomist_traces(trs) yields a set of average lines across all subjects drawn by all
          anatomists using the given fsaverage500_path_traces data trs.
        average_anatomist_traces(trs, anat) uses only the given anatomist or list of anatomists.
        '''
        import neuropythy as ny
        res = {h:{'iso_angle':ny.auto_dict(None,[]), 'iso_eccen':ny.auto_dict(None,[])}
               for h in ['lh','rh']}
        anat = (HCPLinesDataset.anatomist_list if anat is None          else
                anat                           if pimms.is_vector(anat) else
                [anat])
        mp = {}
        for aa in anat:
            lns = fsaverage500_path_traces.get(aa, None)
            if lns is None: continue
            for (s,sdat) in six.iteritems(lns):
                if sdat is None: continue
                for (h,hdat) in six.iteritems(sdat):
                    if hdat is None: continue
                    for (ang,adat) in six.iteritems(hdat):
                        if adat is None: continue
                        for (k,v) in six.iteritems(adat):
                            if v is None: continue
                            if h not in mp: mp[h] = v.map_projection
                            res[h][ang][k].append(v.points)
        res = {h: {a: {k: ny.path_trace(mp[h], mu, closed=False,
                                        meta_data={'all':v, 'std':sd, 'median':md})
                       for (k,v) in six.iteritems(adat)
                       for mu in [np.mean(v, 0)]
                       for ds in [np.sqrt(np.sum((v - mu)**2, 1))]
                       for sd in [np.std(ds, 0)]
                       for md in [np.mean(ds, 0)]}
                   for (a,adat) in six.iteritems(hdat)}
               for (h,hdat) in six.iteritems(res)}
        return pimms.persist(res)
    @staticmethod
    def _fsaverage_to_fsaverage500_traces(fsaverage_path_traces, anat, sid):
        '''
        _fsaverage_to_fsaverage500_traces(fsaverage_path_traces, anat, sid) makes the fsaverage500
          path-traces from the given fsaverage path data for the given anatomist and subject and
          yields these data as a hierarchical map structure.
        '''
        from neuropythy import (map_projection, path_trace)
        fsmps = {h: map_projection('occipital_pole', h) for h in ('lh','rh')}
        sdat = fsaverage_path_traces[anat][sid]
        if sdat is None: return None
        sub = hcp_subject(sid)
        r = {}
        for h in six.iterkeys(sdat):
            hemi = sub.hemis[h]
            rr = {}
            for (dr, ddat) in six.iteritems(sdat[h]):
                rrr = {}
                for (lnm, ptr) in six.iteritems(ddat):
                    fs500_pts = ptr.curve.linspace(500)
                    rrr[lnm] = path_trace(fsmps[h], fs500_pts, closed=False).persist()
                if len(rrr) > 0: rr[dr] = pyr.pmap(rrr)
            if len(rr) > 0: r[h] = pyr.pmap(rr)
        return pyr.pmap(r) if len(r) > 0 else None
    @pimms.value
    def fsaverage500_path_traces(fsaverage_path_traces, _cached_data):
        '''
        fsaverage500_path_traces are roughly equivalent to native_path_traces, except that they have
        been projected onto the fsaverage cortical surface and resampled to consist of  exactly 500
        points in the trace to enable easy averaging and comparison.
        '''
        # we use the native paths to generate these so that we have all the line/edge intersects
        f = HCPLinesDataset._fsaverage_to_fsaverage500_traces
        res = {}
        mnan = HCPLinesDataset.mean_anatomist_name
        mnsb = HCPLinesDataset.mean_subject_name
        for (a,adat) in six.iteritems(fsaverage_path_traces):
            lm = {s: safe_curry(f, fsaverage_path_traces, a, s) for s in six.iterkeys(adat)}
            res[a] = pimms.lazy_map(lm)
        trs0 = pyr.pmap(res)
        # each anatomist gets a mean subject:
        trs = {a: adat.set(mnsb, safe_curry(HCPLinesDataset.average_anatomist_traces, trs0, a))
               for (a,adat) in six.iteritems(trs0)}
        return mapsmerge(_cached_data.get('fsaverage500_path_traces', {}), trs)
    @pimms.value
    def fsaverage_paths(fsaverage_path_traces, _cached_data):
        '''
        fsaverage_paths is identical to fsaverage_path_traces except that it represents the raw
        paths on FreeSurfer's fsaverage cortical surface.
        '''
        return mapsmerge(_cached_data.get('fsaverage_paths', {}),
                         HCPLinesDataset._all_traces_to_paths(fsaverage_path_traces, 'fsaverage'))
    # Sectors and ROIs
    sector_paths = pyr.pmap(
        # These are specified for a map that's not mirror reversed
        {'V1v0': ('V1_mid',                   'V1_outer', '0.5_ventral'),
         'V1d0': ('V1_outer',                 'V1_mid',   '0.5_dorsal'),
         'V1v1': ('V1_mid',    '0.5_ventral', 'V1_outer', '1_ventral'),
         'V1d1': ('V1_outer',  '0.5_dorsal',  'V1_mid',   '1_dorsal'),
         'V1v2': ('V1_mid',    '1_ventral',   'V1_outer', '2_ventral'),
         'V1d2': ('V1_outer',  '1_dorsal',    'V1_mid',   '2_dorsal'),
         'V1v3': ('V1_mid',    '2_ventral',   'V1_outer', '4_ventral'),
         'V1d3': ('V1_outer',  '2_dorsal',    'V1_mid',   '4_dorsal'),
         'V1v4': ('V1_mid',    '4_ventral',   'V1_outer', '7_ventral'),
         'V1d4': ('V1_outer',  '4_dorsal',    'V1_mid',   '7_dorsal'),
         'V2v1': ('V1_outer',  '0.5_ventral', 'V2_outer', '1_ventral'),
         'V2d1': ('V2_outer',  '0.5_dorsal',  'V1_outer', '1_dorsal'),
         'V2v2': ('V1_outer',  '1_ventral',   'V2_outer', '2_ventral'),
         'V2d2': ('V2_outer',  '1_dorsal',    'V1_outer', '2_dorsal'),
         'V2v3': ('V1_outer',  '2_ventral',   'V2_outer', '4_ventral'),
         'V2d3': ('V2_outer',  '2_dorsal',    'V1_outer', '4_dorsal'),
         'V2v4': ('V1_outer',  '4_ventral',   'V2_outer', '7_ventral'),
         'V2d4': ('V2_outer',  '4_dorsal',    'V1_outer', '7_dorsal'),
         'V3v1': ('V2_outer',  '0.5_ventral', 'V3_outer', '1_ventral'),
         'V3d1': ('V3_outer',  '0.5_dorsal',  'V2_outer', '1_dorsal'),
         'V3v2': ('V2_outer',  '1_ventral',   'V3_outer', '2_ventral'),
         'V3d2': ('V3_outer',  '1_dorsal',    'V2_outer', '2_dorsal'),
         'V3v3': ('V2_outer',  '2_ventral',   'V3_outer', '4_ventral'),
         'V3d3': ('V3_outer',  '2_dorsal',    'V2_outer', '4_dorsal'),
         'V3v4': ('V2_outer',  '4_ventral',   'V3_outer', '7_ventral'),
         'V3d4': ('V3_outer',  '4_dorsal',    'V2_outer', '7_dorsal')})
    sector_labels = (None,) + tuple(sorted(sector_paths.keys()))
    sector_label_index = pyr.pmap({lbl:k for (k,lbl) in enumerate(sector_labels)})
    area_paths = pyr.pmap(
        {'V1':     ('V1_outer',  '7_ventral',   '7_dorsal'),
         'V2':     ('V2_dorsal', 'V2_ventral',  '7_ventral', 'V1_ventral', 'V1_dorsal', '7_dorsal'),
         'V3':     ('V3_dorsal', 'V3_ventral',  '7_ventral', 'V2_ventral', 'V2_dorsal', '7_dorsal'),
         'V1v':    ('V1_outer',  '7_ventral',   'V1_mid',    '0.5_ventral'),
         'V1d':    ('V1_outer',  '0.5_dorsal',  'V1_mid',    '7_dorsal'),
         'V2v':    ('V2_outer',  '7_ventral',   'V1_outer',  '0.5_ventral'),
         'V2d':    ('V2_outer',  '0.5_dorsal',  'V1_outer',  '7_dorsal'),
         'V3v':    ('V3_outer',  '7_ventral',   'V2_outer',  '0.5_ventral'),
         'V3d':    ('V3_outer',  '0.5_dorsal',  'V2_outer',  '7_dorsal'),
         'V1fov':  ('V1_outer',  '0.5_ventral', '0.5_dorsal'),
         'V2fov':  ('V1_outer',  '0.5_dorsal',  'V2_outer',  '0.5_ventral'),
         'V3fov':  ('V2_outer',  '0.5_dorsal',  'V3_outer',  '0.5_ventral'),
         'foveal': ('V3_outer',  '0.5_ventral', '0.5_dorsal')})
    area_labels = (None,'V1','V2','V3')
    area_label_index = pyr.pmap({lbl:k for (k,lbl) in enumerate(area_labels)})
    @staticmethod
    def _calculate_sectors(path_traces, anat, sid):
        '''
        _calculate_sectors(path_traces, anat, sid) calculates the set of sectors for the given path
          traces, anatomist, and subject, and yeilds these data as a nested map structure.
        '''
        from neuropythy import close_path_traces
        # we need to close the paths for both hemispheres...
        dat = path_traces[anat][sid]
        if dat is None: return None
        r = {}
        for h in six.iterkeys(dat):
            hdat = pimms.merge(dat[h]['iso_angle'], dat[h]['iso_eccen'])
            rr = {}
            for (snm, sparts) in six.iteritems(HCPLinesDataset.sector_paths):
                parts = [hdat.get(sp) for sp in sparts]
                if any(x is None for x in parts): continue
                if h == 'rh': parts = list(reversed(parts))
                try: rr[snm] = close_path_traces(*parts).persist()
                except Exception:
                    warnings.warn(
                        'Could not close path traces for %s / %s / %s / %s' % (anat,sid,h,snm))
            if len(rr) > 0: r[h] = pyr.pmap(rr)
        return pyr.pmap(r)
    @pimms.value
    def native_sector_traces(native_path_traces, _cached_data):
        '''
        native_sector_traces is a mapping of the sectors for each anatomist and subject.
        '''
        f = HCPLinesDataset._calculate_sectors
        return mapsmerge(_cached_data.get('native_sector_traces',{}),
                         pyr.pmap({a: pimms.lazy_map({s: curry(f, native_path_traces, a, s)
                                                      for s in six.iterkeys(adat)})
                                   for (a,adat) in six.iteritems(native_path_traces)}))
    @staticmethod
    def _calculate_areas(path_traces, anat, sid):
        '''
        _calculate_areas(path_traces, anat, sid) calculates the set of visual areas for the given
          traces, anatomist, and subject, and yeilds these data as a nested map structure.
        '''
        from neuropythy import close_path_traces
        # we need to close the paths for both hemispheres...
        dat = path_traces[anat][sid]
        if dat is None: return None
        r = {}
        for h in six.iterkeys(dat):
            hdat = pimms.merge(dat[h]['iso_angle'], dat[h]['iso_eccen'])
            rr = {}
            for (snm, sparts) in six.iteritems(HCPLinesDataset.area_paths):
                parts = [hdat.get(sp) for sp in sparts]
                if any(x is None for x in parts): continue
                if h == 'rh': parts = list(reversed(parts))
                try: rr[snm] = close_path_traces(*parts)
                except Exception:
                    msg = 'Could not close path traces for %s / %s / %s / %s' % (anat, sid, h, snm)
                    warnings.warn(msg)
            if len(rr) > 0: r[h] = pyr.pmap(rr)
        return pyr.pmap(r)
    @pimms.value
    def native_area_traces(native_path_traces, _cached_data):
        '''
        native_area_traces is a mapping of the visual areas (V1/2/3) for each anatomist and subject.
        '''
        f = HCPLinesDataset._calculate_areas
        return mapsmerge(_cached_data.get('area_path_traces',{}),
                         pyr.pmap({a: pimms.lazy_map({s: curry(f, native_path_traces, a, s)
                                                      for s in six.iterkeys(adat)})
                                   for (a,adat) in six.iteritems(native_path_traces)}))
    @staticmethod
    def _loop_traces_to_paths(name, anat, sdat, sid):
        '''
        _loop_traces_to_paths(sdat, sid) yields the result of converting the path-trace nested-map
        data-structure sdat[sid] into paths using the HCP subject with the given sid. The sdat data
        should be for sectors or ROI's and not for line paths.
        '''
        sdat = None if sdat is None else sdat.get(sid)
        if sdat is None: return None
        sub = hcp_subject(sid)
        r = {}
        for (h,hdat) in six.iteritems(sdat):
            rr = {}
            fmap = None
            for (k,pt) in six.iteritems(hdat):
                if pt is None: continue
                elif fmap is None: fmap = pt.map_projection(sub.hemis[h])
                try: rr[k] = pt.to_path(sub.hemis[h], flatmap=fmap)
                except Exception as e:
                    warnings.warn('failed to render %s loop path %s/%s/%s/%s: %s'
                                  % (name, anat, sid, h, k, str(e)))
            if len(rr) > 0: r[h] = pyr.pmap(rr)
        return pyr.pmap(r) if len(r) > 0 else None
    @staticmethod
    def _all_loop_traces_to_paths(name, dat):
        '''
        Converts all anatomists and subjects using the function above, but in a lazy map at the
        subject-level.
        '''
        f = HCPLinesDataset._loop_traces_to_paths
        return pyr.pmap(
            {anat: pimms.lazy_map({sid: curry(f, name, anat, adat, sid)
                                   for sid in six.iterkeys(adat)})
             for (anat,adat) in six.iteritems(dat)})
    @pimms.value
    def native_sectors(native_sector_traces, _cached_data):
        '''
        native_sectors represents the same data as native_sector_traces but after conversion to
        path objects by combination with the appropriate subject hemisphere.
        '''
        return mapsmerge(_cached_data.get('sector_paths', {}),
                         HCPLinesDataset._all_loop_traces_to_paths('sectors', native_sector_traces))
    @pimms.value
    def native_areas(native_area_traces, _cached_data):
        '''
        native_areas represents the same data as native_area_traces but after conversion to
        path objects by combination with the appropriate subject hemisphere.
        '''
        return mapsmerge(_cached_data.get('area_paths', {}),
                         HCPLinesDataset._all_loop_traces_to_paths('areas', native_area_traces))
    @staticmethod
    def _calculate_subject_labels(areas, sectors, anat, sid):
        '''
        _calculate_subject_labels(areas, sectors, anat, sid) calculates the labels for the given
          anatomist and subject using the given areas and sectors data; these data are yielded in 
          a nested map structure.
        '''
        sub = hcp_subject(sid)
        areadat = areas.get(anat, {}).get(sid)
        sectdat = sectors.get(anat, {}).get(sid)
        if areadat is None: areadat = {}
        if sectdat is None: sectdat = {}
        r = {}
        for h in ['lh','rh']:
            rr = {}
            hemi = sub.hemis[h]
            # areas first
            hdat = areadat.get(h, {})
            rrr = np.zeros(hemi.vertex_count, dtype=np.int32)
            for (k,lbl) in six.iteritems(HCPLinesDataset.area_label_index):
                if k is None: continue
                p = hdat.get(k)
                if p is None: continue
                if np.sum(p.label > 0.5) > np.sum(p.label < 0.5):
                    warnings.warn('inverted area label: %s' % ((anat,sid,h,k),))
                else: rrr[p.label >= 0.5] = lbl
            if np.sum(rrr) > 0:
                rrr.setflags(write=False)
                rr['visual_area'] = rrr
            # then sectors
            hdat = sectdat.get(h, {})
            rrr = np.zeros(hemi.vertex_count, dtype=np.int32)
            for (k,lbl) in six.iteritems(HCPLinesDataset.sector_label_index):
                if k is None: continue
                p = hdat.get(k)
                if p is None: continue
                if np.sum(p.label > 0.5) > np.sum(p.label < 0.5):
                    warnings.warn('inverted sector label: %s' % ((anat,sid,h,k),))
                else: rrr[p.label >= 0.5] = lbl
            if np.sum(rrr) > 0:
                rrr.setflags(write=False)
                rr['visual_sector'] = rrr
            if len(rr) > 0: r[h] = rr
        return None if len(r) == 0 else pimms.persist(r)
    @pimms.value
    def subject_labels(native_areas, native_sectors, _cached_data):
        '''
        subject_labels is a dict-structure of the labels granted to each subject; for visual areas,
        this is <anatomist>_visal_area or just visual_area for the mean, with 1, 2, and 3 labeled.
        For sectors, this is according the HCPLinesDataset.sector_labels and sector_label_index.
        '''
        anatomists = HCPLinesDataset.full_anatomist_list
        f = HCPLinesDataset._calculate_subject_labels
        return mapsmerge(
            _cached_data.get('labels', {}),
            pyr.pmap({anat: pimms.lazy_map({sid:curry(f, native_areas, native_sectors, anat, sid)
                                            for sid in HCPLinesDataset.subject_list})
                      for anat in anatomists}))
    @staticmethod
    def _calculate_subject_distances(paths, anat, sid):
        '''
        _calculate_subject_distances(paths, anat, sid) calculates the boundary distances for the
          given anatomist and subject using the given path data; these data are yielded in a
          nested map structure. All distances are calculated on the subject's midgray surface.
        '''
        surf = 'midgray'
        r = {}
        nptrs = paths.get(anat,{}).get(sid)
        if nptrs is None: return None
        sub = hcp_subject(sid)
        for h in ('lh','rh'):
            hemi = sub.hemis[h]
            nptr = nptrs.get(h,{})
            rr = {}
            for (ang,angdat) in six.iteritems(nptr):
                rrr = {k:p.estimated_distances[surf] for (k,p) in six.iteritems(angdat)}
                if len(rrr) > 0: rr[ang] = pyr.pmap(rrr)
            if len(rr) > 0: r[h] = pyr.pmap(rr)
        return pyr.pmap(r) if len(r) > 0 else None
    @pimms.value
    def subject_boundary_distances(native_paths, anat_distances, _cached_data):
        '''
        subject_boundary_distances is a measurement at each vertex of the distance to the nearest
        polar angle boundary; these distances are an estiate based on the neuropythy path distance
        estimates.
        '''
        f = HCPLinesDataset._calculate_subject_distances
        anats = native_paths.keys() if anat_distances else (HCPLinesDataset.mean_anatomist_name,)
        return mapsmerge(
            _cached_data.get('distances', {}),
            pyr.pmap({anat: pimms.lazy_map({sid:curry(f, native_paths, anat, sid)
                                            for sid in HCPLinesDataset.subject_list})
                      for anat in anats}))
    @staticmethod
    def calculate_clean_retinotopy(hemi, labels):
        '''
        Given an HCP-subject's hemi, a set of retinotopy data, and labels, calculates a cleaned set
        of retinotopy data and return a map of the polar angle and eccentricity for these.
        '''
        import neuropythy as ny
        (ang,ecc) = ny.vision.clean_retinotopy(hemi, 'prf_', visual_area=labels)
        if ang is None: return None
        return pimms.persist({'polar_angle':ang, 'eccentricity':ecc})
    @pimms.value
    def clean_retinotopic_maps(subject_labels, _cached_data):
        '''
        clean_retinotopic_maps is a map of the cleaned retinotopic maps for each of the HCP subjects
        in the HCP-lines dataset. The first level of the clean_retinotopic_maps map-structure is the
        subject ID while the second level is the hemisphere name. Below that is a property map for
        clean_polar_angle and clean_eccentricity.
        '''
        def clean_rmap(anat, sid):
            sdat = subject_labels.get(anat, {})
            if sdat is None: return None
            sdat = sdat.get(sid)
            if sdat is None: return None
            sub = hcp_subject(sid)
            r = {}
            for h in ('lh','rh'):
                ps = sdat.get(h)
                if ps is None: continue
                hemi = sub.hemis[h]
                rmap = HCPLinesDataset.calculate_clean_retinotopy(hemi, ps['visual_area'])
                (ang,ecc) = [rmap[k] for k in ['polar_angle', 'eccentricity']]
                if ang is None or ecc is None: continue
                ps = dict(clean_polar_angle=ang, clean_eccentricity=ecc)
                r[h] = ps
            if len(r) == 0: return None
            else: return pimms.persist(r)
        return mapsmerge(
            _cached_data.get('clean', {}),
            pyr.pmap({anat: pimms.lazy_map({sid:curry(clean_rmap, anat, sid)
                                            for sid in HCPLinesDataset.subject_list})
                      for anat in HCPLinesDataset.full_anatomist_list}))
    @staticmethod
    def calculate_cortical_magnification(anat,sid, hemi, rdat, labels):
        '''
        Given an HCP-subject's hemi, a retinotopic dataset, and a set of labels, calculates the
        cortical magnification for that hemisphere and yields it. The labels must a list of 
        values per vertex with 1/2/3 for V1/V2/V3.
        '''
        from neuropythy.vision import areal_cmag, retinotopy_data, as_retinotopy
        cm = np.zeros(hemi.vertex_count)
        for lbl in (1,2,3):
            (x,y) = as_retinotopy(rdat, 'geographical')
            msk = np.where((labels == lbl) & np.isfinite(x))[0]
            if len(msk) == 0:
                qq = (anat,sid,hemi.chirality,lbl)
                warnings.warn('cmag calc: subject hemi has empty label: %s/%d/%s/%d' % qq)
                continue
            f = areal_cmag(hemi, rdat, mask=msk)
            cm[msk] = f(x[msk], y[msk])
        cm.setflags(write=False)
        #return pimms.quant(cm, 'mm*mm / degree*degree')
        return cm
    @pimms.value
    def subject_cortical_magnifications(subject_labels, clean_retinotopic_maps, _cached_data):
        '''
        subject_cortical_magnifications is a map of the cortical magnification estimates at each
        vertex in the V1-V3 region, as defined by each particular anatomist.
        '''
        from functools import reduce
        mapfind = lambda m,k: None if m is None or k not in m else m[k]
        def calc_cmag(anat, sid):
            sub = hcp_subject(sid)
            r = {}
            for h in ('lh','rh'):
                cmr = reduce(mapfind, [anat, sid, h], clean_retinotopic_maps)
                if cmr is None: continue
                lbl = reduce(mapfind, [anat, sid, h, 'visual_area'], subject_labels)
                if lbl is None: continue
                hemi = sub.hemis[h]
                cmr = {k:cmr['clean_'+k] for k in ['polar_angle', 'eccentricity']}
                cmr['visual_area'] = lbl
                try: ve = hemi.prop('prf_variance_explained')
                except Exception: ve = hemi.prop('lowres-prf_variance_explained')
                cmr['variance_explained'] = ve
                cm = HCPLinesDataset.calculate_cortical_magnification(anat, sid, hemi, cmr, lbl)
                r[h] = cm
            if len(r) == 0: return None
            else: return pimms.persist(r)
        return mapsmerge(
            _cached_data.get('cmag', {}),
            pyr.pmap({anat: pimms.lazy_map({sid:curry(calc_cmag, anat, sid)
                                            for sid in HCPLinesDataset.subject_list})
                      for anat in HCPLinesDataset.full_anatomist_list}))
    @pimms.value
    def subjects(subject_labels, subject_boundary_distances, clean_retinotopic_maps,
                 subject_cortical_magnifications):
        '''
        subjects is a map of the HCP subjects that are part of the HCP-lines dataset; The
        hemispheres of the subjects contain additional properties for the visual areas and
        the sectors. These are named '<anatomist>_visual_area' and '<anatomist>_visual_sector'. The
        mean across anatomists is just 'visual_area' and 'visual_sector'.
        '''
        from functools import reduce
        mapfind = lambda m,k: None if m is None or k not in m else m[k]
        lookup = lambda m,*ks: reduce(mapfind, ks, m)
        meananat = HCPLinesDataset.mean_anatomist_name
        def makesub(sid):
            sdat = lookup(subject_labels, meananat, sid)
            if sdat is None: return None
            sub = hcp_subject(sid)
            r = {}
            for h in ('lh','rh'):
                hemi = sub.hemis[h]
                # get the data we're going to put on it...
                lbl = sdat.get(h)
                sbd = (None if subject_boundary_distances is None else
                       lookup(subject_boundary_distances, meananat, sid, h))
                cmg = lookup(subject_cortical_magnifications, meananat, sid, h)
                crm = lookup(clean_retinotopic_maps, meananat, sid, h)
                # bundle it up into one map...
                props = {k:v for m in [lbl,crm] if m is not None for (k,v) in six.iteritems(m)}
                if cmg is not None: props['areal_cmag'] = cmg
                for adat in six.itervalues({} if sbd is None else sbd):
                    for (k,v) in six.iteritems({} if adat is None else adat):
                        props[k + '_distance'] = v
                hemi = hemi.with_prop(**props)
                r[h] = hemi
            if len(r) == 0: return None
            else: return sub.with_hemi(**r)
        return pimms.lazy_map({sid: curry(makesub, sid) for sid in HCPLinesDataset.subject_list})
    @pimms.value
    def subject_tables(subjects, exclusions, _cached_data):
        '''
        subject_tables is a map of the HCP subjects in the HCP-lines projects to PANDAS dataframe
        objects of the relevant retinotopy data. The subject_tables track the mean data (of the
        mean anatomist) only and do not attempt to track individual anatomist lines/data.
        '''
        from neuropythy.util import (auto_dict, to_dataframe)
        def maketbl(sid):
            # we build the table...
            tbl = auto_dict(None, [])
            for h in ('lh','rh'):
                if ('mean',sid,h) in exclusions: continue
                sub = subjects[sid]
                if sub is None: return None
                hemi = sub.hemis[h]
                if hemi is None: continue
                nans = np.full(hemi.vertex_count, np.nan)
                lbls = hemi.prop('visual_area')
                ii = np.where(lbls > 0)[0]
                if len(ii) == 0: raise ValueError('Subject %d with no labels' % sid)
                for pp in ('visual_area','visual_sector','prf_polar_angle','prf_eccentricity',
                           'prf_radius','prf_variance_explained', 'clean_polar_angle',
                           'clean_eccentricity', 'areal_cmag', 'white_surface_area',
                           'midgray_surface_area','pial_surface_area','V3_ventral_distance',
                           'V2_ventral_distance', 'V1_ventral_distance', 'V3_dorsal_distance',
                           'V2_dorsal_distance', 'V1_dorsal_distance', 'V1_mid_distance',
                           '0.5_distance', '1_distance', '2_distance','4_distance', '7_distance'):
                    uu = nans if pp not in hemi.properties else hemi.prop(pp)
                    tbl[pp].append(uu[ii])
                for (k,v) in zip(['hemi','subject'], [h,sid]):
                    tbl[k].append(np.full(len(ii), v))
            if len(tbl) == 0: return None
            # join them into a table
            return to_dataframe({k:np.concatenate(v) for (k,v) in six.iteritems(tbl)})
        return mapsmerge(
            _cached_data.get('subject_tables', {}),
            pimms.lazy_map({sid:curry(maketbl,sid) for sid in HCPLinesDataset.subject_list}))
    @pimms.value
    def dataframe(subject_tables, exclusions, _cached_data):
        '''
        dataframe is a PANDAS dataframe of all the relevant pRF data.
        '''
        import pandas
        if _cached_data is None: _cached_data = {}
        table = _cached_data.get('dataframe')
        if table is not None: return table
        tables = []
        for sid in HCPLinesDataset.subject_list:
            #print(sid)
            # if either hemi is bad, skip this subject
            df = subject_tables.get(sid)
            if df is None: continue
            ldf = df.loc[df['hemi'] == 'lh']
            rdf = df.loc[df['hemi'] == 'rh']
            for (h,hdf) in zip(['lh','rh'], [ldf, rdf]):
                if ('mean',sid,h) in exclusions:
                    continue
                elif any(np.sum(hdf['visual_area'] == ll) == 0 for ll in (1,2,3)):
                    warnings.warn('Subject %d missing a visual area for %s' % (sid,h))
                else:
                    tables.append(hdf)
        return pandas.concat(tables)
    @staticmethod
    def _calc_visual_sulcus_area(sid, h, visual_sulcus_label):
        '''
        calculates the areas (pial, white, midgray) for the visual sulcus for the given subject
        and hemisphere.
        '''
        import neuropythy as ny
        sub = ny.hcp_subject(sid)
        def _f():
            hem = sub.hemis[h]
            fsa = ny.freesurfer_subject('fsaverage').hemis[h]
            ii = fsa.interpolate(hem, visual_sulcus_label[h], method='nearest')
            return ii
        ii = pimms.lazy_map({0: _f})
        def _g(k):
            hem = sub.hemis[h]
            return np.sum(hem.prop(k+'_surface_area')[ii[0]])
        return pimms.lazy_map({k: curry(_g, k) for k in ['white', 'midgray', 'pial']})
    @staticmethod
    def _calc_hemisphere_area(sid, h):
        '''
        calculates the areas (pial, white, midgray) for the entire cortical surface, excluding the
          corpus callosum.
        '''
        import neuropythy as ny
        def _f(srf):
            hem = ny.hcp_subject(sid).hemis[h]
            return np.sum(hem.prop(srf+'_surface_area')[hem.prop('atlas_label')])
        return pimms.lazy_map({k: curry(_f, k) for k in ['white', 'midgray', 'pial']})
    @staticmethod
    def _calc_surface_areas(paths, exclusions, anat, sid):
        '''
        _calc_surface_area(native_sectors, anat, sid, vsl) yields the surface area data for the
        given subject ID and either sector or area paths; vsl must be the visual_sulcus_label
        data.
        '''
        from functools import reduce
        npts = paths.get(anat, None)
        if npts is None: return None
        npts = npts.get(sid, None)
        if npts is None: return None
        def disc(rd):
            def getsa(k):
                u = rd.surface_area[k]
                # fix a common error: the path is encoded clockwise instead of counterclockwise
                tot = np.nansum(rd.surface.prop(k+'_surface_area'))
                if u > tot - u: u = tot - u
                return u
            return pimms.lazy_map({k: try_curry(getsa, np.nan, k)
                                   for k in ['midgray','white','pial']})
        r = {h: {roi: curry(disc, rd) for (roi,rd) in six.iteritems(hd)}
             for (h,hd) in six.iteritems(npts)
             if hd is not None
             if (anat,sid,h) not in exclusions}
        # If we have a V1/V2/V3 and a V1fov/V2fov/V3fov, we want to make sure to add non-foveated
        # ROI surface areas here as well
        def nonfov(rh, ll, srf):
            try:
                full = rh['V%d' % ll][srf]
                fov  = rh['V%dfov' % ll][srf]
                return full - fov
            except Exception:
                return None
        for h in six.iterkeys(r):
            rh = pimms.lazy_map(r[h])
            for ll in [1,2,3]:
                if ('V%d' % ll) in rh and ('V%dfov' % ll) in rh:
                    lm = {k: curry(nonfov, rh, ll, k) for k in ['midgray','white','pial']}
                    rh = pimms.assoc(rh, 'V%dnonfov' % ll, pimms.lmap(lm))
            r[h] = rh
        # That should be all; just return.
        return pyr.pmap(r)
    @staticmethod
    def _calc_label_surface_areas(sid, visual_sulcus_label):
        '''
        _calc_label_surface_areas(sid, vissulc) yields the surface area data for "label" ROIs for 
        the given subject. Label ROIs include the whole cortex, the visual sulcus label, and any
        FreeSurfer- or anatomically-defined labels such as Brodmann areas or Benson14 maps.
        '''
        import neuropythy as ny
        saprops = ['white', 'midgray', 'pial']
        # We'll build up a map for each hemisphere; for now we leave them as dicts, but we'll
        # persist them at the end
        lbls = {'lh': {}, 'rh': {}}
        def _cortex_sarea(h, srf):
            hem = ny.hcp_subject(sid).hemis[h]
            return np.sum(hem.prop(srf+'_surface_area')[hem.prop('atlas_label')])
        def _fslbl_sarea(h, ll, srf):
            hem = ny.hcp_subject(sid).hemis[h]
            return np.sum(hem.prop(srf+'_surface_area')[hem.prop(ll + '_label')])
        b14_maps = {}
        def _b14_sarea(h, prop, srf):
            hem = ny.hcp_subject(sid).hemis[h]
            if h not in b14_maps:
                pred = ny.vision.predict_retinotopy(hem)
                va = pred['varea']
                ec = pred['eccen']
                va[ec > 7] = 0
                va[ec < 0.5] = 0
                b14_maps[h] = va
            return np.sum(hem.prop(srf+'_surface_area')[b14_maps[h] == prop])
        for h in ['lh','rh']:
            # Cortex ROI first:
            lbls[h]['H'] = pimms.lazy_map({k: curry(_cortex_sarea, h, k) for k in saprops})
            # Calcarine sulcus:
            lbls[h]['Calc'] = HCPLinesDataset._calc_visual_sulcus_area(sid, h, visual_sulcus_label)
            # The Brodmann areas and FreeSurfer labels:
            for ll in ['V1', 'V2', 'MT', 'BA44', 'BA45', 'BA3b']:
                lbls[h][ll] = pimms.lazy_map({k: curry(_fslbl_sarea, h, ll, k) for k in saprops})
            # The Benson14 maps
            for ll in [1,2,3]:
                lbls[h]['B14V%d' % ll] = pimms.lazy_map({k: curry(_b14_sarea, h, ll, k)
                                                         for k in saprops})
        # That's all of them!
        return pimms.persist(lbls)
    @pimms.value
    def visual_sulcus_label(pseudo_path):
        '''
        visual_sulcus_label is a map whose keys are 'lh' and 'rh' and whose values are the ROI label
        for the fsaverage visual sulcus (i.e., the Calcarine sulcus) as defined by Kendrick Kay.
        Note that the visual sulcus area is labeled 'calcarine' in the surface area data structures
        and dataframe.
        '''
        import neuropythy as ny
        r = {}
        for h in ['lh','rh']:
            if not pseudo_path.find('visual_sulc', h+'.visualsulc.mgz'):
                warnings.warn('could not find %s visual sulc MGZ file' % h)
                continue
            fl = pseudo_path.local_path('visual_sulc', h+'.visualsulc.mgz')
            lbl = (ny.load(fl) == 2)
            lbl.setflags(write=False)
            r[h] = lbl
        return pyr.pmap(r)
    @pimms.value
    def area_surface_areas(native_areas, exclusions, visual_sulcus_label, _cached_data):
        '''
        area_surface_areas is a data-structure of the cortical surface area for each anatomist,
        subject, hemisphere, and visual area ROI.
        '''
        vsl = visual_sulcus_label
        f = HCPLinesDataset._calc_surface_areas
        r = pyr.pmap({anat: pimms.lazy_map({sid: curry(f, native_areas, exclusions, anat, sid)
                                            for sid in six.iterkeys(anatdat)})
                      for (anat,anatdat) in six.iteritems(native_areas)})
        # we want to add the visual sulc data also
        return mapsmerge(_cached_data.get('area_surface_areas', {}), r)
    @pimms.value
    def sector_surface_areas(native_sectors, exclusions, _cached_data):
        '''
        sector_surface_areas is a data-structure of the cortical surface area for each anatomist,
        subject, hemisphere, and visual sector ROI.
        '''
        f = HCPLinesDataset._calc_surface_areas
        r = pyr.pmap({anat: pimms.lazy_map({sid: curry(f, native_sectors, exclusions, anat, sid)
                                            for sid in six.iterkeys(anatdat)})
                      for (anat,anatdat) in six.iteritems(native_sectors)})
        return mapsmerge(_cached_data.get('sector_surface_areas', {}), r)
    @pimms.value
    def label_surface_areas(visual_sulcus_label, _cached_data):
        '''
        label_surface_areas is a data-structure of the cortical surface areas for each subject and
        hemisphere and a variety of anatomically-defined labels. These labels include  the whole
        cortex, the visual sulcus label, and any FreeSurfer- or anatomically-defined labels such as
        Brodmann areas or Benson14 maps.
        '''
        f = HCPLinesDataset._calc_label_surface_areas
        r = pimms.lazy_map({sid: curry(f, sid, visual_sulcus_label)
                            for sid in HCPLinesDataset.full_subject_list})
        return mapsmerge(_cached_data.get('label_surface_areas', {}), r)
    @pimms.value
    def surface_area_dataframe(area_surface_areas, sector_surface_areas, label_surface_areas,
                               exclusions, trust_exclusions, _cached_data):
        '''
        surface_area_dataframe is a dataframe object containing the various data about surface areas
        of both the visual areas and the visual sectors. If a subject+anatomist+hemisphere is in the
        exclusions list (and trust_exclusions is True), then all relevant entries for that
        hemisphere will be given a NaN value.

        The columns of the dataframe are as follows:
          * sid, anatomist: the subject ID and anatomist (which may be 'mean')
          * roi<X><tag>: a visual-area surface area; these include V1-V3 as well as partial ROIs
            such as V1-ventral ('V1v') and V2-fovea ('V2fov'); any ROI derived from the annotated
            lines that is not a sector is an ROI; additionally, foveal sectors are considered ROIs
            since the V1 foveal region is not a sector. The <X> is always L or R for LH or RH and
            the tag is the ROI name, e.g., V1d, V2, V3fov.
          * sct<X><tag>: any sector except for V2 and V3 foveal sectors, which are considered ROIs
            instead (for consistency with V1). <X> is always L or R for LH or RH while tags always
            follow the pattern V<id><vd><ec>: <id> is 1, 2, or 3 for V1-V3; <vd> is either v or d
            for ventral or dorsal, and <ec> is 0, 1, 2, 3, or 3--these represent the eccentricity
            rings (0-0.5, 0.5-1, 1-2, 2-4, 4-7). Note that V2 and V3 do not have ec values of 0
            as the foveal regions are considered ROIs.
          * lbl<X><tag>: an fsaverage or fs_LR-derived label or atlas region. These include the
            full cortex area (lbl<X>H), the Calcarine-sulcus ROI (lbl<X>Calc), the Benson 14
            atlas rois (equivalent regions to the traced ROIs), and a few FreeSurfer ROIs:
            V1, V2, MT, BA44, BA45, BA3b.
        '''
        import neuropythy as ny
        # Check the cache first:
        c = _cached_data.get('surface_area_dataframe', None)
        if c is not None: return c
        df = ny.auto_dict(None, [])
        roi_keys = list(HCPLinesDataset.area_paths.keys()) + ['V1nonfov', 'V2nonfov', 'V3nonfov']
        sct_keys = list(HCPLinesDataset.sector_paths.keys())
        lbl_keys = ['H', 'Calc', 'V1','V2','MT', 'BA44','BA45', 'B14V1','B14V2','B14V3']
        kk = (['anatomist','sid'] +
              ['lbl%s%s' % (h,k) for h in ['L','R'] for k in lbl_keys] +
              ['roi%s%s' % (h,k) for h in ['L','R'] for k in roi_keys] +
              ['sct%s%s' % (h,k) for h in ['L','R'] for k in sct_keys])
        meananat = HCPLinesDataset.mean_anatomist_name
        for sid in HCPLinesDataset.subject_list:
            #print(sid)
            for anat in HCPLinesDataset.full_anatomist_list:
                #print('  - ', anat)
                df['sid'].append(sid)
                df['anatomist'].append(anat)
                sub = ny.hcp_subject(sid)
                for (dat,tag,ks) in zip([area_surface_areas, sector_surface_areas,
                                         label_surface_areas],
                                        ['roi', 'sct', 'lbl'], [roi_keys, sct_keys, lbl_keys]):
                    # label areas should appear in all rows but are only stored for the mean anat
                    q = dat if tag == 'lbl' else dat.get(anat, None)
                    q = None if q is None else q.get(sid, None)
                    if q is None:
                        for h in [tag+'L',tag+'R']:
                            for k in ks:
                                df[h + k].append(np.nan)
                    else:
                        for (h,hd) in six.iteritems(q):
                            hh = h[0].upper()
                            # Make sure there's not an exclusion here
                            if tag != 'lbl' and trust_exclusions and (anat,sid,h) in exclusions:
                                for (lbl,ld) in six.iteritems(hd):
                                    df[tag + hh + lbl].append(np.nan)
                            else:
                                for (lbl,ld) in six.iteritems(hd):
                                    df[tag + hh + lbl].append(ld['midgray'])
                # make sure we didn't miss any fields...
                n = len(df['sid'])
                for k in kk:
                    v = df[k]
                    if len(v) < n:
                        v.append(np.nan)
        # That's basically it...
        return ny.to_dataframe(df)
    def save_normalized(self, subject_list=Ellipsis,
                        overwrite=False, create_directories=True, create_mode=0o755,
                        save_traces=True, save_paths=True, save_sectors=True,
                        save_properties=True, save_dataframe=True, save_surface_areas=True,
                        logger=None, forget=True):
        '''
        save_normalized() saves the data for all subjects and anatomists into an easily-loadible
        combination of JSON- and HDF5-formatted files in the directory 'normalized'. By default,
        this will not overwrite existing files, but it will create any directories if they do not
        exist.
        '''
        from neuropythy import save
        from neuropythy.hcp import forget_subject
        sids = (HCPLinesDataset.full_subject_list if subject_list is Ellipsis else
                []                                if subject_list is None     else
                subject_list)
        anatomists = HCPLinesDataset.full_anatomist_list
        meananat = HCPLinesDataset.mean_anatomist_name
        meansub = HCPLinesDataset.mean_subject_name
        if logger is None: logger = lambda s:None
        if pimms.is_int(sids): sids = [sids]
        # We want to organize this whole set of exports around the subject so that we can forget
        # the subject at the end of the loop if requested to do so
        for sid in sids:
            logger(' * Subject %s' % str(sid))
            for anat in anatomists:
                logger('    * Anatomist %s' % anat)
                # (1) All the Path Traces
                if save_traces:
                    for name in ('raw', 'native', 'fsaverage', 'fsaverage500', 'area', 'sector'):
                        logger('       * Saving %s path traces...' % name)
                        try:
                            self.save_traces(anat, sid, name, create_directories=create_directories,
                                             create_mode=create_mode, overwrite=overwrite)
                        except Exception:
                            msg = 'Save failure for %s_path_traces: %s / %s' % (name,anat,str(sid))
                            warnings.warn(msg)
                            logger('       - ' + msg)
                            raise
                # (2) All the Paths
                if save_paths and sid != meansub:
                    for name in ('raw', 'native', 'fsaverage', 'area', 'sector'):
                        logger('       * Saving %s paths...' % name)
                        try:
                            self.save_paths(anat, sid, name, create_directories=create_directories,
                                            create_mode=create_mode, overwrite=overwrite)
                        except Exception:
                            msg = 'Save failure for %s_paths: %s / %s' % (name, anat, str(sid))
                            warnings.warn(msg)
                            logger('       - ' + msg)
                            raise
                # (3) All the properties
                if save_properties is not None and save_properties is not False and sid != meansub:
                    if pimms.is_str(save_properties): lst = [save_properties]
                    elif save_properties is True:     lst = ['labels','distances','clean','cmag']
                    elif save_properties == 'all':    lst = ['labels','distances','clean','cmag']
                    else:                             lst = save_properties
                    for name in lst:
                        logger('       * Saving %s...' % name)
                        try:
                            self.save_properties(anat, sid, name,
                                                 create_directories=create_directories,
                                                 create_mode=create_mode, overwrite=overwrite)
                        except Exception:
                            msg = 'Save failure for %s: %s / %s' % (name, anat, str(sid))
                            warnings.warn(msg)
                            logger('       - ' + msg)
                # (4) The Subject Dataframe
                if save_dataframe and sid != meansub:
                    fp = HCPLinesDataset.find_path(self.pseudo_path,'mean','%d.dataframe.hdf5'%sid)
                    if not fp or overwrite:
                        logger('       * Saving dataframe...')
                        tbl = self.subject_tables[sid]
                        if tbl is not None:
                            flnm = HCPLinesDataset.cache_path(self.pseudo_path,
                                                              'mean','%d.dataframe.hdf5' % sid,
                                                              create_directories=create_directories,
                                                              create_mode=create_mode)
                            tbl.to_hdf(flnm, 'dataframe')
                # (5) The Surface Areas
                if save_surface_areas and sid != meansub:
                    for name in ['roi', 'sct', 'lbl']:
                        # labels don't have an anatomist so we only save on meananat.
                        if name == 'lbl' and anat != meananat: continue
                        logger('       * Saving %s surface_areas...' % name)
                        try:
                            self.save_surface_areas(anat, sid, name,
                                                    create_directories=create_directories,
                                                    create_mode=create_mode, overwrite=overwrite)
                        except Exception:
                            msg = 'Save failure for %s_surface_areas: %s / %s' % (name,anat,sid)
                            warnings.warn(msg)
                            logger('       - ' + msg)
            # (6) If we need to forget the subject, do so
            if forget and sid != meansub: forget_subject(sid)
        # Finally, the dataframe
        if save_dataframe and (len(sids) == 0 or subject_list is Ellipsis):
            logger('* Saving Dataframe...')
            flnm = HCPLinesDataset.cache_path(self.pseudo_path, 'dataframe.hdf5',
                                              create_directories=create_directories,
                                              create_mode=create_mode)
            if overwrite or not os.path.isfile(flnm): self.dataframe.to_hdf(flnm, 'dataframe')
        if save_surface_areas and (len(sids) == 0 or subject_list is Ellipsis):
            logger('* Saving surface_areas Dataframe...')
            flnm = HCPLinesDataset.cache_path(self.pseudo_path, 'surface_areas.hdf5',
                                              create_directories=create_directories,
                                              create_mode=create_mode)
            if overwrite or not os.path.isfile(flnm):
                self.surface_area_dataframe.to_hdf(flnm, 'dataframe')
        # that's it!
        return None

    # Plotting tools that go with the data
    default_line_styles = pyr.pmap(
        {'V1_ventral': pyr.m(linestyle='-', linewidth=0.5, color=(0,0,1)),
         'V1_dorsal':  pyr.m(linestyle='-', linewidth=0.5, color=(1,0,0)),
         'V1_mid':     pyr.m(linestyle='-', linewidth=0.5, color=(0,1,0)),
         'V2_ventral': pyr.m(linestyle='-', linewidth=0.5, color=(0,0,1)),
         'V2_dorsal':  pyr.m(linestyle='-', linewidth=0.5, color=(1,0,0)),
         'V3_ventral': pyr.m(linestyle='-', linewidth=0.5, color=(0,0,1)),
         'V3_dorsal':  pyr.m(linestyle='-', linewidth=0.5, color=(1,0,0)),
         '0.5':        pyr.m(linestyle='-', linewidth=0.5, color=(0,0,0)),
         '1':          pyr.m(linestyle='-', linewidth=0.5, color=(1,0,1)),
         '2':          pyr.m(linestyle='-', linewidth=0.5, color=(1,1,0)),
         '4':          pyr.m(linestyle='-', linewidth=0.5, color=(0,1,1)),   
         '7':          pyr.m(linestyle='-', linewidth=0.5, color=(0.5,0.5,0.5))})
    def lines_plot(self, sid, h, anatomist='mean', mesh=None, lines='native',
                   axes=None, styles=None):
        '''
        data.lines_plot(sid, h) plots the mean lines (across anatomists) for the subject id and
          hemisphere that is given.

        The following options may be given:
          * anatomist (default: 'mean') may specify the anatomist whose lines should be plotted.
          * mesh (default: None) may specify a flatmap or 3D mesh on which the lines should be
            drawn; if None, then the path traces are drawn on the subject's native projection (i.e,
            the projection found in data.map_projections); otherwise, the paths are interpolated
            onto the mesh given and are plotted there. For 3D meshes, the lines are plotted slightly
            (0.05 mm) above the surface.
          * lines (default: 'native') specifies which set of lines should be used; may be 'native'
            or 'raw'.
          * axes (default: None) specifies the axes to use; for a 3D mesh, this argument is ignored.
            If axes is None, then matplotlib.pyplot.gca() is used.
          * styles (default: None) may specify plot styles for the various lines by their names; by
            default, the value None is replaced with HCPLinesDataset.default_line_styles. The styles
            option may be a dictionary like default_line_styles or it may be a list/tuple in which
            the values are effectively merged with right-most values overwriting left-most values.
            The first element may be None to indicate that the second dictionary overwrites the
            values in the first dictionary. A style of None indicates that the earlier instructions
            should be forgotten and that the line in question should not be drawn.
        '''
        import neuropythy as ny
        # first, let's process some arguments
        sub = ny.hcp_subject(sid)
        hem = sub.hemis[h]
        if mesh is None: 
            mp = self.subject_map_projections[sid][h]
            mesh = mp(hem)
            reproj = False
        elif pimms.is_str(mesh):
            mesh = mesh.lower()
            if mesh == 'sphere': mesh = 'native'
            try: mesh = hem.surfaces[mesh]
            except Exception: mesh = hem.registrations[mesh]
            reproj = True
        else: reproj = True
        styles0 = HCPLinesDataset.default_line_styles
        if pimms.is_tuple(styles) or pimms.is_list(styles):
            iimx = 0
            for (ii,sty) in enumerate(styles):
                if sty is None: iimx = ii
            styles = styles[iimx:]
            if len(styles) == 0: styles = None
            elif styles[0] is None: styles = pimms.merge(styles0, *styles[1:])
            else: styles = pimms.merge(*styles)
        if styles is None: styles = styles0
        if reproj:
            if   lines == 'raw':    lines = self.raw_paths
            elif lines == 'native': lines = self.native_paths
            else: raise ValueError('lines must be raw or native')
            datlns = lines[anatomist][sid][h]
            # if we have to reproject the lines, let's get the coordinates now
            # note that, if the mesh is 3D, we want to put the points slightly outside of it
            if mesh.coordinates.shape[0] == 2:
                lines = {k:mesh.unaddress(v) for angdat in six.itervalues(datlns)
                         for (k,v) in six.iteritems(angdat) if k in styles}
            else:
                lines = {}
                for (k,v) in (y for x in six.itervalues(datlns) for y in six.iteritems(x)):
                    if k not in styles: continue
                    # get the address data
                    fids  = mesh.tess.index[v.addresses['faces']]
                    nrms  = mesh.face_normals[:,fids]
                    lines[k] = mesh.unaddress(v.addresses) + 0.1*nrms
        else:
            if   lines == 'raw':    lines = self.raw_path_traces
            elif lines == 'native': lines = self.native_path_traces
            else: raise ValueError('lines must be raw or native')
            lines = {k:v.points for angdat in six.itervalues(lines[anatomist][sid][h])
                     for (k,v) in six.iteritems(angdat)}
        # draw the lines...
        pp = {}
        if mesh.coordinates.shape[0] == 2:
            # use pyplot
            import matplotlib as mpl, matplotlib.pyplot as plt
            if axes is None: axes = plt.gca()
            for (k,ln) in six.iteritems(lines):
                if k not in styles: continue
                sty = styles[k]
                if pimms.is_str(sty):   pp[k] = axes.plot(ln[0], ln[1], sty)
                elif pimms.is_map(sty): pp[k] = axes.plot(ln[0], ln[1], **sty)
                elif pimms.is_tuple(sty) or pimms.is_list(sty):
                    if pimms.is_map(ln[-1]): pp[k] = axes.plot(ln[0], ln[1], *sty[:-1], **sty[-1])
                    else:                    pp[k] = axes.plot(ln[0], ln[1], *sty)
                else: raise ValueError('cannot interpret style for %s: %s' % (k, sty))
        else:
            # use iyvolume
            import ipyvolume as ipv
            for (k,ln) in six.iteritems(lines):
                if k not in styles: continue
                sty = styles[k]
                if pimms.is_str(sty):   pp[k] = ipv.plot(ln[0], ln[1], ln[2], sty)
                elif pimms.is_map(sty): pp[k] = ipv.plot(ln[0], ln[1], ln[2], **sty)
                elif pimms.is_tuple(sty) or pimms.is_list(sty):
                    if pimms.is_map(ln[-1]): pp[k] = ipv.plot(ln[0], ln[1], ln[2],
                                                              *sty[:-1], **sty[-1])
                    else:                    pp[k] = ipv.plot(ln[0], ln[1], ln[2], *sty)
                else: raise ValueError('cannot interpret style for %s: %s' % (k, sty))
        # that's it!
        return pp
    # A handy function for resampling a subject's sectors to a different estimateed set of sectors
    v123_sector_key = pyr.pmap({1:  (90,  180, 0,   0.5), # V1d0
                                2:  (90,  180, 0.5, 1),   # V1d1
                                3:  (90,  180, 1,   2),   # V1d2
                                4:  (90,  180, 2,   4),   # V1d3
                                5:  (90,  180, 4,   7),   # V1d4
                                6:  (0,    90, 0,   0.5), # V1v0
                                7:  (0,    90, 0.5, 1),   # V1v1
                                8:  (0,    90, 1,   2),   # V1v2
                                9:  (0,    90, 2,   4),   # V1v3
                                10: (0,    90, 4,   7),   # V1v4
                                11: (180, 270, 0.5, 1),   # V2d1
                                12: (180, 270, 1,   2),   # V2d2
                                13: (180, 270, 2,   4),   # V2d3
                                14: (180, 270, 4,   7),   # V2d4
                                15: (-90,   0, 0.5, 1),   # V2v1
                                16: (-90,   0, 1,   2),   # V2v2
                                17: (-90,   0, 2,   4),   # V2v3
                                18: (-90,   0, 4,   7),   # V2v4
                                19: (270, 360, 0.5, 1),   # V3d1
                                20: (270, 360, 1,   2),   # V3d2
                                21: (270, 360, 2,   4),   # V3d3
                                22: (270, 360, 4,   7),   # V3d4
                                23: (-180,-90, 0.5, 1),   # V3v1
                                24: (-180,-90, 1,   2),   # V3v2
                                25: (-180,-90, 2,   4),   # V3v3
                                26: (-180,-90, 4,   7)})  # V3v4
    def refit_sectors(self, sid, h, outangs, outeccs):
        '''
        ny.data['hcp_lines'].refit_sectors(sid, h, outangles, outeccens) yields a resampled set of
          sectors using neuropythy's ny.vision.refit_sectors() function. The result is a tuple for
          each visual area, (v1, v2, v3) of the sectors formed by outangles and outeccens.
        '''
        import neuropythy as ny
        sub = self.subjects[sid]
        hem = sub.hemis[h]
        rdat = ny.retinotopy_data(hem, 'prf_')
        (ang,ecc) = ny.as_retinotopy(rdat, 'visual')
        lbl = hem.prop('visual_sector')
        if h == 'rh': ang = -ang
        ang = np.mod(ang + 90, 360) - 90
        # Fix the angles to match the ranges in the sector key.
        for (s,bounds) in self.v123_sector_key.items():
            ii = lbl == s
            b1 = bounds[1]
            if b1 == 270:
                ang[ii] = 360 - ang[ii]
            elif b1 == 360:
                ang[ii] = 180 + ang[ii]
            elif b1 == 0:
                ang[ii] = -ang[ii]
            elif b1 == -90:
                ang[ii] = -180 + ang[ii]
        scts = ny.vision.labels_to_sectors(self.v123_sector_key, lbl)
        surf = hem.surface('midgray').with_prop(prf_polar_angle=ang)
        # Fix the out-angles to use the v2 and v3 values as well.
        outangs = np.unique([x for a in outangs for x in (a, a+180, a-180)])
        rfs = ny.vision.refit_sectors(surf, scts, outangs, outeccs,
                                      retinotopy=(ang,ecc))
        # Sort and translate the results into v1, v2, and v3 sections.
        res = ({}, {}, {})
        for (k,ii) in rfs.items():
            b1 = k[1]
            if b1 <= 270 and b1 > 180:
                (va,a1,a2) = (1, 360 - k[1], 360 - k[0])
            elif b1 > 270:
                (va,a1,a2) = (2, k[0] - 180, k[1] - 180)
            elif b1 > -90 and b1 <= 0:
                (va,a1,a2) = (1, -k[1], -k[0])
            elif b1 <= -90:
                (va,a1,a2) = (2, 180 + k[0], 180 + k[1])
            else:
                (va,a1,a2) = (0, k[0], k[1])
            res[va][(a1,a2,k[2],k[3])] = ii
        return res

# Add the neuropythy hook for the dataset:
add_dataset('hcp_lines', lambda:HCPLinesDataset().persist())


