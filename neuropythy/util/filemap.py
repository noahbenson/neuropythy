####################################################################################################
# neuropythy/util/filemap.py
# Utility for presenting a directory with a particular format as a data structure.
# By Noah C. Benson

import os, warnings, six, tarfile, atexit, shutil, posixpath, json, pimms
import numpy          as np
import pyrsistent     as pyr
from   posixpath  import join as urljoin, split as urlsplit, normpath as urlnormpath
from   six.moves  import urllib
from   .core      import (library_path, curry, ObjectWithMetaData, AutoDict, data_struct, tmpdir,
                          is_tuple, is_list)
from   .conf      import to_credentials

# Not required, but try to load it anyway:
try:              import s3fs
except Exception: s3fs = None

def is_url(url):
    '''
    is_url(p) yields True if p is a valid URL and False otherwise.
    '''
    try: return bool(urllib.request.urlopen(url))
    except Exception: return False
def url_download(url, topath=None, create_dirs=True):
    '''
    url_download(url) yields the contents of the given url as a byte-string.
    url_download(url, topath) downloads the given url to the given path, topath and yields that path
      on success.

    The option create_dirs (default: True) may be set to False to prevent the topath directory from
    being created.
    '''
    # ensure directory exists
    if topath: topath = os.path.expanduser(os.path.expandvars(topath))
    if create_dirs and topath:
        dnm = os.path.dirname(topath)
        if not os.path.isdir(dnm): os.makedirs(os.path.abspath(dnm), 0o755)
    if six.PY2:
        response = urllib.request.urlopen(url)
        if topath is None: topath = response.read()
        else:
            with open(topath, 'wb') as fl:
                shutil.copyfileobj(response, fl)
    else:
        with urllib.request.urlopen(url) as response:
            if topath is None: topath = response.read()
            else:
                with open(topath, 'wb') as fl:
                    shutil.copyfileobj(response, fl)
    return topath
def is_s3_path(path):
    '''
    is_s3_path(path) yields True if path is a valid Amazon S3 path and False otherwise.
    '''
    return pimms.is_str(path) and path.lower().startswith('s3:')
def is_osf_path(path):
    '''
    is_osf_path(path) yields True if path starts with 'osf:' and False otherwise.
    '''
    return pimms.is_str(path) and path.lower().startswith('osf:')
tarball_endings = tuple([('.tar' + s) for s in ('','.gz','.bz2','.lzma')])
def split_tarball_path(path):
    '''
    split_tarball_path(path) yields a tuple (tarball, p) in which tarball is the path to the tarball
      referenced by path and p is the internal path that followed that tarball. If no tarball is
      included in the path, then (None, path) is returned. If no internal path is found following
      the tarball, then (path, '') is returned.
    '''
    lpath = path.lower()
    for e in tarball_endings:
        if lpath.endswith(e): return (path, '')
        ee = e + ':'
        if ee not in lpath: continue
        spl = path.split(ee)
        tarball = spl[0] + e
        p = ee.join(spl[1:])
        return (tarball, p)
    return (None, path)
def is_tarball_path(path):
    '''
    is_tarball_path(p) yields True if p is either a valid path of a tarball file or is such a path
      followed by a colon then any additional path.
    '''
    (tb,p) = split_tarball_path(path)
    return tb is not None
osf_basepath = 'https://api.osf.io/v2/nodes/%s/files/%s/'
def _osf_tree(proj, path=None, base='osfstorage'):
    if path is None: path = (osf_basepath % (proj, base))
    else:            path = (osf_basepath % (proj, base)) + path.lstrip('/')
    dat = json.loads(url_download(path, None))
    if 'data' not in dat: raise ValueError('Cannot detect kind of url for ' + path)
    dat = dat['data']
    if pimms.is_map(dat): return dat['links']['download']
    res = {r['name']:(u['links']['download'] if r['kind'] == 'file' else
                      curry(lambda r: _osf_tree(proj, r, base), r['path']))
           for u in dat for r in [u['attributes']]}
    return pimms.lazy_map(res)
def osf_crawl(k, *pths, **kw):
    '''
    osf_crawl(k) crawls the osf repository k and returns a lazy nested map structure of the
      repository's files. Folders have values that are maps of their contents while files have
      values that are their download links.
    osf_crawl(k1, k2, ...) is equivalent to osf_crawl(posixpath.join(k1, k2...)).

    The optional named argument base (default: 'osfstorage') may be specified to search in a
    non-standard storage position in the OSF URL; e.g. in the github storage.
    '''
    from six.moves import reduce
    base = kw.pop('base', 'osfstorage')
    root = kw.pop('root', None)
    if len(kw) > 0: raise ValueError('Unknown optional parameters: %s' % (list(kw.keys()),))
    if k.lower().startswith('osf:'): k = k[4:]
    k = k.lstrip('/')
    pths = [p.lstrip('/') for p in (k.split('/') + list(pths))]
    (bpth, pths) = (pths[0], pths[1:])
    if root is None: root = _osf_tree(bpth, base=base)
    return reduce(lambda m,k: m[k], pths, root)
    
@pimms.immutable
class PseudoDir(ObjectWithMetaData):
    '''
    The PseudoDir class represents either directories themselves, tarballs, or URLs as if they were
    directories.
    '''
    def __init__(self, source_path, cache_path=None, delete=Ellipsis, credentials=None,
                 meta_data=None):
        ObjectWithMetaData.__init__(self, meta_data=meta_data)
        self.source_path = source_path
        self.cache_path = cache_path
        self.delete = delete
        self.credentials = credentials
    @pimms.param
    def source_path(sp):
        '''
        pseudo_dir.source_path is the source path of the the given pseudo-dir object.
        '''
        if sp is None: return os.path.join('/')
        if not pimms.is_str(sp): raise ValueError('source_path must be a string/path')
        if is_url(sp) or is_s3_path(sp): return sp
        return os.path.expanduser(os.path.expandvars(sp))
    @pimms.param
    def cache_path(cp):
        '''
        pseudo_dir.cache_path is the optionally provided cache path; this is the same as the
        storage path unless this is None.
        '''
        if cp is None: return None
        if not pimms.is_str(cp): raise ValueError('cache_path must be a string')
        return os.path.expanduser(os.path.expandvars(cp))
    @pimms.param
    def delete(d):
        '''
        pseudo_dir.delete is True if the pseudo_dir self-deletes on Python exit and False otherwise;
        if this is Ellipsis, then self-deletes only when the cache-directory is created by the
        PseudoDir class and is a temporary directory (i.e., not explicitly provided).
        '''
        if d in (True, False, Ellipsis): return d
        else: raise ValueError('delete must be True, False, or Ellipsis')
    @pimms.param
    def credentials(c):
        '''
        pdir.credentials is a tuple (key, secret) for authentication with Amazon S3.
        '''
        if c is None: return None
        else: return to_credentials(c)
    @staticmethod
    def _url_to_ospath(path):
        #if os.sep == posixpath.sep: return path
        path = urlnormpath(path)
        ps = []
        while path != '':
            (path,fl) = posixpath.split(path)
            ps.append(fl)
        return os.path.join(*reversed(ps))
    @staticmethod
    def _url_exists(urlbase, cache_path, path):
        cpath = os.path.join(cache_path, PseudoDir._url_to_ospath(path))
        if os.path.exists(cpath): return True
        else: return is_url(urljoin(urlbase, path))
    @staticmethod
    def _url_getpath(urlbase, cache_path, path):
        cpath = os.path.join(cache_path, PseudoDir._url_to_ospath(path))
        if os.path.exists(cpath): return cpath
        url = urljoin(urlbase, path)
        return url_download(url, cpath)
    @staticmethod
    def _osf_exists(fls, osfbase, cache_path, path):
        cpath = os.path.join(cache_path, PseudoDir._url_to_ospath(path))
        if os.path.exists(cpath): return cpath
        fl = fls
        for pp in path.split('/'):
            if   pimms.is_str(fl): return False
            elif pp in fl:         fl = fl[pp]
            else:                  return False
        return True
    @staticmethod
    def _osf_getpath(fls, osfbase, cache_path, path):
        cpath = os.path.join(cache_path, PseudoDir._url_to_ospath(path))
        if os.path.exists(cpath): return cpath
        fl = fls
        for pp in path.split('/'): fl = fl[pp]
        return url_download(fl, cpath)
    @staticmethod
    def _s3_exists(fs, urlbase, cache_path, path):
        cpath = os.path.join(cache_path, PseudoDir._url_to_ospath(path))
        if os.path.exists(cpath): return cpath
        url = urljoin(urlbase, path)
        return fs.exists(url)
    @staticmethod
    def _s3_getpath(fs, urlbase, cache_path, path):
        cpath = os.path.join(cache_path, PseudoDir._url_to_ospath(path))
        if os.path.exists(cpath): return cpath
        url = urljoin(urlbase, path)
        fs.get(url, cpath)
        return cpath
    @staticmethod
    def _tar_exists(tarpath, cache_path, path):
        cpath = os.path.join(cache_path, path)
        if os.path.exists(cpath): return True
        with tarfile.open(tarpath, 'r') as tfl:
            try: return bool(tfl.getmember(path))
            except Exception: return False
    @staticmethod
    def _tar_getpath(tarpath, cache_path, path):
        cpath = os.path.join(cache_path, path)
        if os.path.exists(cpath): return cpath
        with tarfile.open(tarpath, 'r') as tfl: tfl.extract(path, cache_path)
        return cpath
    @pimms.value
    def _path_data(source_path, cache_path, delete, credentials):
        need_cache = True
        rpr = source_path
        # Okay, it might be a directory, an Amazon S3 URL, a different URL, or a tarball
        if os.path.isdir(source_path):
            def exists_fn(p):  return os.path.exists(os.path.join(source_path, p))
            def getpath_fn(p): return os.path.join(source_path, p)
            rpr = os.path.normpath(source_path)
            join_fn = os.path.join
            need_cache = False
            pathmod = os.path
        elif is_tuple(source_path):
            (el0,pp) = (source_path[0], source_path[1:])
            if s3fs is not None and isinstance(el0, s3fs.S3FileSystem):
                rpr = 'S3:/' + urljoin(*pp)
                def ujn(p): return urljoin(*(pp + (p,)))
                def exists_fn(p):  return PseudoDir._s3_exists(el0,  cache_path, ujn(p))
                def getpath_fn(p): return PseudoDir._s3_getpath(el0, cache_path, ujn(p))
                pathmod = posixpath
            elif isinstance(el0, PseudoDir):
                pd = el0._path_data
                # we can use this dir's cache directory
                need_cache = False
                if cache_path is None: cache_path = el0.cache_path
                (pathmod,efn,gfn,rpr) = [pd[k] for k in ('pathmod','exists','getpath','repr')]
                def exists_fn(p):  return efn(pathmod.join(*(pp + (p,))))
                def getpath_fn(p): return gfn(pathmod.join(*(pp + (p,))))
                rpr = pathmod.join(*((rpr,) + pp))
        elif is_s3_path(source_path):
            if s3fs is None: raise ValueError('s3fs module is not installed')
            elif credentials is None: fs = s3fs.S3FileSystem(anon=True)
            else: fs = s3fs.S3FileSystem(key=credentials[0], secret=credentials[1])
            def exists_fn(p):  return PseudoDir._s3_exists(fs, source_path, cache_path, p)
            def getpath_fn(p): return PseudoDir._s3_getpath(fs, source_path, cache_path, p)
            pathmod = posixpath
        elif is_osf_path(source_path):
            fs = osf_crawl(source_path)
            def exists_fn(p):  return PseudoDir._osf_exists(fs, source_path, cache_path, p)
            def getpath_fn(p): return PseudoDir._osf_getpath(fs, source_path, cache_path, p)
            pathmod = posixpath
        elif is_url(source_path):
            def exists_fn(pth):  return PseudoDir._url_exists(source_path,  cache_path, pth)
            def getpath_fn(pth): return PseudoDir._url_getpath(source_path, cache_path, pth)
            pathmod = posixpath
        # Check if it's a "<tarball>:path", like subject10.tar.gz:subject10/"
        elif is_tarball_path(source_path):
            (tb,ip) = split_tarball_path(source_path)
            if ip is None:
                # tarball by itself
                def exists_fn(p):  return PseudoDir._tar_exists(source_path,  cache_path, p)
                def getpath_fn(p): return PseudoDir._tar_getpath(source_path, cache_path, p)
            else:
                # tarball with internal path
                def exists_fn(p):  return PseudoDir._tar_exists(tb,  cache_path, os.path.join(ip,p))
                def getpath_fn(p): return PseudoDir._tar_getpath(tb, cache_path, os.path.join(ip,p))
            pathmod = os.path
        # ok, don't know what it is...
        else: raise ValueError('Could not interpret source path: %s' % source_path)
        cache_path = tmpdir(delete=(True if delete is Ellipsis else delete)) if need_cache else None
        # one final layer on the exist and getpath functions: we want to automatically interpret
        # and expand internal tarball files as we go...
        tarballs = {}
        def tar_pdir(tb, path):
            if tb in tarballs: return tarballs[tb]
            ostb = tb if pathmod.sep == os.sep else PseudoDir._url_to_ospath(tb)
            pd = PseudoDir(tb, cache_path=os.path.join(cache_path, '.extracted_tarballs', tb),
                           delete=False, meta_data={'container_path':source_path})
            tarballs[tb] = pd
            return pd
        def exists_tar_fn(p):
            x = exists_fn(p)
            if x: return True
            # see if x has a tarball in it
            (tb,pth) = split_tarball_path(p)
            if tb is None or len(path) == 0: return False
            if pathmod.sep != os.sep: pth = PseudoDir._url_to_ospath(pth)
            # there is a tarball: we auto-extract it into a new pseudo-dir
            tb = tar_pdir(tb, pth)
            return tb._path_data['exists'](pth)
        def getpath_tar_fn(p):
            if exists_fn(p): return getpath_fn(p)
            # see if x has a tarball in it
            (tb,pth) = split_tarball_path(p)
            if tb is None or len(path) == 0: return getpath_fn(p)
            if pathmod.sep != os.sep: pth = PseudoDir._url_to_ospath(pth)
            # there is a tarball: we auto-extract it into a new pseudo-dir
            tb = tar_pdir(tb, pth)
            return tb._path_data['getpath'](pth)
        return pyr.pmap({'repr':    rpr,
                         'exists':  exists_tar_fn,
                         'getpath': getpath_tar_fn,
                         'cache':   cache_path,
                         'pathmod': pathmod})

    @pimms.value
    def actual_cache_path(_path_data):
        '''
        pdir.actual_cache_path is the cache path being used by the pseudo-dir pdir; this may differ
          from the pdir.cache_path if the cache_path provided was None yet a temporary cache path
          was needed.
        '''
        return _path_data['cache']
    def __repr__(self):
        p = self._path_data['repr']
        return "pseudo_dir('%s')" % p
    def join(self, *args):
        '''
        pdir.join(args...) is equivalent to os.path.join(args...) but always appropriate for the
          kind of path represented by the pseudo-dir pdir.
        '''
        join = self._path_data['pathmod'].join
        return join(*args)
    def find(self, *args):
        '''
        pdir.find(paths...) is similar to to os.path.join(paths...) but it only yields the joined
          relative path if it can be found inside pdir; otherwise None is yielded. Note that this
          does not extract or download the path--it merely ensures that it exists.
        '''
        data = self._path_data
        exfn = data['exists']
        join = data['pathmod'].join
        path = join(*args)
        return path if exfn(path) else None
    def local_path(self, *args):
        '''
        pdir.local_path(paths...) is similar to os.path.join(pdir, paths...) except that it
          additionally ensures that the path being requested is found in the pseudo-dir pdir then
          ensures that this path can be found in a local directory by downloading or extracting it
          if necessary. The local path is yielded.
        '''
        data = self._path_data
        gtfn = data['getpath']
        join = data['pathmod'].join
        path = join(*args)
        return gtfn(path)
    def local_cache_path(self, *args):
        '''
        pdir.local_cache_path(paths...) is similar to os.path.join(pdir, paths...) except that it
          yields a local version of the given path, much like pdir.local_path(paths...). The 
          local_cache_path function differs from the local_path function in that, if no existing
          file is found at the given destination, no error is raised and the path is still returned.
        '''
        # if the file exists in the pseudo-dir, just return the local path
        if self.find(*args) is not None: return self.local_path(*args)
        cp = self._path_data['cache']
        if cp is None: cp = self.source_path
        return os.path.join(cp, *args)
def pseudo_dir(source_path, cache_path=None, delete=Ellipsis, credentials=None, meta_data=None):
    '''
    pseudo_dir(source_path) yields a pseudo-directory object that represents files in the given
      source path.

    Pseudo-dir objects act as an interface for loading data from abstract sources. The given source
    path may be either a directory, a (possibly zipped) tarball, or a URL. In all cases but the
    local directory, the pseudo-dir object will quietly extract/download the requested files to a
    cache directory as their paths are requested. This is managed through two methods:
      * find(args...) joins the argument list as in os.path.join, then, if the resulting file is
        found in the source_path, this (relative) path-name is returned; otherwise None is returned.
      * local_path(args...) joins the argument list as in os.path.join, then, if the resulting file
        is found in the source_path, it is extracted/downloaded to the local cache directory if
        necessary, and this path (or the original path when no cache directory is used) is returned.

    The following optional arguments may be given:
      * cache_path (default: None) specifies the cache directory in which to put any extracted or
        downloaded contents. If None, then a temporary directory is created and used. If the source
        path is a local directory, then the cache path is not needed and is instead ignored. Note
        that if the cache path is not deleted, it can be reused across sessions--the pseudo-dir will
        always check for files in the cache path before extracting or downloading them.
      * delete (default: Ellipsis) may be set to True or False to declare that the cache directory
        should be deleted at system exit (assuming a normal Python system exit). If Ellipsis, then
        the cache_path is deleted only if it is created by the pseudo-dir object--given cache paths
        are never deleted.
      * credentials (default: None) may be set to a valid set of Amazon S3 credentials for use if
        the source path is an S3 path. The contents are passed through the to_credentials function.
      * meta_data (default: None) specifies an optional map of meta-data for the pseudo-dir.
    '''
    return PseudoDir(source_path, cache_path=cache_path, delete=delete, credentials=credentials,
                     meta_data=meta_data)
    
@pimms.immutable
class FileMap(ObjectWithMetaData):
    '''
    The FileMap class is a pimms immutable class that tracks a set of FileMap format instructions
    with a valid path containing data of that format.
    '''
    def __init__(self, path, instructions, path_parameters=None, data_hierarchy=None,
                 cache_path=None, cache_delete=Ellipsis, load_function=None, meta_data=None, **kw):
        ObjectWithMetaData.__init__(self, meta_data=meta_data)
        self.path = path
        self.instructions = instructions
        self.data_hierarchy = data_hierarchy
        self.supplemental_paths = kw
        self.path_parameters = path_parameters
        self.load_function = load_function
        self.cache_path = cache_path
        self.cache_delete = cache_delete
    _tarball_endings = tuple([('.tar' + s) for s in ('','.gz','.bz2','.lzma')])
    @staticmethod
    def valid_path(p):
        '''
        FileMap.valid_path(path) yields os.path.abspath(path) if path is either a directory or a
          tarball file; yields path if path is a URL or s3 path, and otherwise yields None. If the
          path is a tarball path with a trailing inner-path, then the abspath of the tarball with
          the inner path appended is yielded.
        '''
        if   os.path.isdir(p):   return os.path.abspath(p)
        elif is_s3_path(p):      return p
        elif is_url(p):          return p
        # could still be a tarball path
        (tb,p) = split_tarball_path(path)
        if   tb is None: return None
        elif p  == '':   return os.path.abspath(tb)
        else:            return os.path.abspath(tb) + ':' + p
    @pimms.param
    def load_function(lf):
        '''
        filemap.load_function is the function used to load data by the filemap. It must accept
        exactly two arguments: filename and filedata. The file-data object is a merged map of both
        the path_parameters, meta_data, and file instruction, left-to-right in that order.
        '''
        if lf is None:
            from ..io import load
            return lambda fl,ii: load(fl)
        else: return lf
    @pimms.param
    def cache_path(cp):
        '''
        filemap.cache_path is the path into which the filemap's data are cached (if necessary). This
        parameter represents the provided cache_path argument, but may not be the actual used cache
        path, which can be found in the actual_cache_path member.
        '''
        if cp is None: return None
        cp = os.path.expanduser(os.path.expandvars(cp))
        if not os.path.isdir(cp): raise ValueError('cache_path must be a directory or None')
        return cp
    @pimms.param
    def cache_delete(d):
        '''
        filemap.cache_delete is the setting used to decide whether to delete the cache directory at
        system exit. If Ellipsis (the default), then deletes any temporary directory created;
        otherwise does as this variable instructs.
        '''
        if d in [True,False,Ellipsis]: return d
        raise ValueError('cache_delete must be True, False, or Ellipsis')
    @pimms.param
    def path(p):
        '''
        filemap.path is the root path of the filemap object. 
        '''
        p = FileMap.valid_path(p)
        if p is None: raise ValueError('Path must be a directory or a tarball')
        else: return p
    @pimms.param
    def instructions(inst):
        '''
        filemap.instructions is the map of load/save instructions for the given filemap.
        '''
        if not pimms.is_map(inst) and not isinstance(inst, list):
            raise ValueError('instructions must be a map or a list')
        return pimms.persist(inst)
    @pimms.param
    def data_hierarchy(h):
        '''
        filemap.data_hierarchy is the initial data hierarchy provided to the filemap object.
        '''
        return pimms.persist(h)
    @pimms.param
    def supplemental_paths(sp):
        '''
        filemap.supplemental_paths is a map of additional paths provided to the filemap object.
        '''
        if not pimms.is_map(sp): raise ValueError('supplemental_paths must be a map')
        rr = {}
        for (nm,pth) in six.iteritems(sp):
            pth = FileMap.valid_path(pth)
            if pth is None: raise ValueError('supplemental paths must be directories or tarballs')
            rr[nm] = pth
        return pimms.persist(rr)
    @pimms.param
    def path_parameters(pp):
        '''
        filemap.path_parameters is a map of parameters for the filemap's path.
        '''
        if pp is None: return pyr.m()
        elif not pimms.is_map(pp): raise ValueError('path perameters must be a mapping')
        else: return pimms.persist(pp)
    @staticmethod
    def parse_instructions(inst, hierarchy=None):
        '''
        FileMap.parse_instructions(inst) yields the tuple (data_files, data_tree); data_files is a
          map whose keys are relative filenames and whose values are the instruction data for the
          given file; data_tree is a lazy/nested map structure of the instruction data using 'type'
          as the first-level keys.

        The optional argument hierarchy specifies the hierarchy of the data to return in the
        data_tree. For example, if hierarchy is ['hemi', 'surface', 'roi'] then a file with the
        instructions {'type':'property', 'hemi':'lh', 'surface':'white', 'roi':'V1', 'name':'x'}
        would appear at data_tree['hemi']['lh']['surface']['white']['roi']['V1']['property']['x']
        whereas if hierarchy were ['roi', 'hemi', 'surface'] it would appear at
        data_tree['roi']['V1']['surface']['white']['hemi']['lh']['property']['x']. By default the
        ordering is undefined.
        '''
        dirstack = []
        data_tree = {}
        data_files = {}
        hierarchies = hierarchy if hierarchy else []
        if len(hierarchies) > 0 and pimms.is_str(hierarchies[0]): hierarchies = [hierarchies]
        known_filekeys = ('load','filt','when','then','miss')
        hierarchies = list(hierarchies)
        def handle_file(inst):
            # If it's a tuple, we just do each of them
            if isinstance(inst, tuple):
                for ii in inst: handle_file(ii)
                return None
            # first, walk the hierarchies; if we find one that matches, we use it; otherwise we make
            # one up
            dat = None
            for hrow in hierarchies:
                if not all(h in inst for h in hrow): continue
                dat = data_tree
                for h in hrow:
                    v = inst[h]
                    if h not in dat: dat[h] = {}
                    dat = dat[h]
                    if v not in dat: dat[v] = {}
                    dat = dat[v]
                break
            if dat is None:
                # we're gonna make up a hierarchy
                hh = []
                dat = data_tree
                for (k,v) in six.iteritems(inst):
                    if k in known_filekeys: continue
                    hh.append(k)
                    if k not in dat: dat[k] = {}
                    dat = dat[k]
                    if v not in dat: dat[v] = {}
                    dat = dat[v]
                # append this new hierarchy to the hierarchies
                hierarchies.append(hh)
            # Okay, we have the data, get the filename
            flnm = os.path.join(*dirstack)
            # add this data ot the data tree
            dat[flnm] = pyr.pmap(inst).set('_relpath', flnm)
            data_files[flnm] = inst
            return None
        def handle_dir(inst):
            # iterate over members
            dnm = None
            for k in inst:
                if dnm: dnm = handle_inst(k, dnm)
                elif pimms.is_str(k): dnm = k
                elif not isinstance(k, tuple) or len(k) != 2: raise ValueError('Bad dir content')
                else: dnm = handle_inst(k, dnm)
            return None
        def handle_inst(inst, k=None):
            if k: dirstack.append(k)
            if pimms.is_map(inst): handle_file(inst)
            elif isinstance(inst, (list, tuple)):
                if len(inst) == 0 or not pimms.is_map(inst[0]): handle_dir(inst)
                else: handle_file(inst)
            else: raise ValueError('Illegal instruction type: %s' % (inst,))
            if k: dirstack.pop()
            return None
        handle_inst(inst)
        return (data_files, data_tree)
    @pimms.value
    def _parsed_instructions(instructions, data_hierarchy):
        return pimms.persist(FileMap.parse_instructions(instructions, data_hierarchy))
    @pimms.value
    def actual_cache_path(cache_path, cache_delete, path, supplemental_paths):
        '''
        filemap.actual_cache_path is the cache path used by the filemap, if needed.
        '''
        if cache_path is not None: return cache_path
        if (is_url(path) or is_s3_path(path) or is_tarball_path(path) or
            any(is_url(s) or is_s3_path(s) or is_tarball_path(s)
                for s in six.itervalues(supplemental_paths))):
            # we need a cache path...
            return tmpdir(delete=(True if cache_delete is Ellipsis else cache_delete))
        else: return None
    @pimms.require
    def setup_cache_deletion(cache_path, actual_cache_path, cache_delete):
        '''
        The filemap.setup_cache_deletion requirement ensures that the cache directory will be
        deleted at system exit, if required.
        '''
        if cache_delete is True and cache_path is actual_cache_path:
            atexit.register(shutil.rmtree, cache_path)
        return True
    @staticmethod
    def _load(pdir, flnm, loadfn, *argmaps, **kwargs):
        try:
            lpth = pdir.local_path(flnm)
            args = pimms.merge(*argmaps, **kwargs)
            loadfn = inst['load'] if 'load' in args else loadfn
            filtfn = inst['filt'] if 'filt' in args else lambda x,y:x
            dat = loadfn(lpth, args)
            dat = filtfn(dat, args)
        except Exception: dat = None
        # check for miss instructions if needed
        if dat is None and 'miss' in args:
            miss = args['miss']
        elif pimms.is_str(miss) and miss.lower() in ('error','raise','exception'):
            raise ValueError('File %s failed to load' % flnm)
        elif miss is not None:
            dat = miss(flnm, args)
        return dat
    @staticmethod
    def _parse_path(flnm, path, spaths, path_parameters, inst):
        flnm = flnm.format(**pimms.merge(path_parameters, inst))
        p0 = None
        for k in six.iterkeys(spaths):
            if flnm.startswith(k + ':'):
                (flnm, p0) = (flnm[(len(k)+1):], k)
                break
        return (p0, flnm)
    @pimms.value
    def psuedo_dirs(path, supplemental_paths, actual_cache_path):
        '''
        fmap.pseudo_dirs is a mapping of pseduo-dirs in the file-map fmap. The primary path's
        pseudo-dir is mapped to the key None.
        '''
        # we need to make cache paths for some of these...
        spaths = {}
        cp = None
        n = 0
        for (s,p) in six.iteritems(supplemental_paths):
            if actual_cache_path:
                cp = os.path.join(actual_cache_path, 'supp', s)
                if not os.path.isdir(cp): os.makedirs(os.path.abspath(cp), 0o755)
                n += 1
            spaths[s] = pseudo_dir(p, delete=False, cache_path=cp)
        if actual_cache_path:
            if n > 0: cp = os.path.join(actual_cache_path, 'main')
            else:     cp = actual_cache_path
            if not os.path.isdir(cp): os.makedirs(os.path.abspath(cp), 0o755)
        spaths[None] = pseudo_dir(path, delete=False, cache_path=cp)
        return pyr.pmap(spaths)
    @pimms.value
    def data_files(pseudo_dirs, path_parameters, load_function, meta_data, _parsed_instructions):
        '''
        filemap.data_files is a lazy map whose keys are filenames and whose values are the loaded
        files.
        '''
        (data_files, data_tree) = _parsed_instructions
        res = {}
        for (flnm, inst) in six.iteritems(data_files):
            (pathnm, fn) = FileMap._parse_path(flnm, path, pseudo_dirs, path_parameters, inst)
            res[fn] = curry(FileMap._load,
                            pseudo_dirs[pathnm], flnm, load_function,
                            path_parameters, meta_data, inst)
        return pimms.lazy_map(res)
    @pimms.value
    def data_tree(_parsed_instructions, path, supplemental_paths, path_parameters, data_files):
        '''
        filemap.data_tree is a lazy data-structure of the data loaded by the filemap's instructions.
        '''
        data_tree = _parsed_instructions[1]
        def visit_data(d):
            d = {k:visit_maps(v) for (k,v) in six.iteritems(d)}
            return data_struct(d)
        def visit_maps(m):
            r = {}
            anylazy = False
            for (k,v) in six.iteritems(m):
                kk = k if isinstance(k, tuple) else [k]
                for k in kk:
                    if len(v) > 0 and '_relpath' in next(six.itervalues(v)):
                        (flnm,inst) = next(six.iteritems(v))
                        flnm = FileMap._deduce_filename(flnm, path, supplemental_paths,
                                                        path_parameters, inst)
                        r[k] = curry(lambda flnm:data_files[flnm], flnm)
                        anylazy = True
                    else: r[k] = visit_data(v)
            return pimms.lazy_map(r) if anylazy else pyr.pmap(r)
        return visit_data(data_tree)
def file_map(path, instructions, **kw):
    '''
    file_map(path, instructions) yields a file-map object for the given path and instruction-set.
    file_map(None, instructions) yields a lambda of exactly one argument that is equivalent to the
      following:  lambda p: file_map(p, instructions)
    
    File-map objects are pimms immutable objects that combine a format-spec for a directory 
    (instructions) with a directory to yield a lazily-loaded data object. The format-spec is not
    currently documented, but interested users should see the variable
    neuropythy.hcp.files.hcp_filemap_instructions.
    
    The following options can be given:
     * path_parameters (default: None) may be set to a map of parameters that are used to format the
       filenames in the instructions.
     * data_hierarchy (default: None) may specify how the data should be nested; see the variable
       neuropythy.hcp.files.hcp_filemap_data_hierarchy.
     * load_function (default: None) may specify the function that is used to load filenames; if
       None then neuropythy.io.load is used.
     * meta_data (default: None) may be passed on to the FileMap object.

    Any additional keyword arguments given to the file_map function will be used as supplemental
    paths.
    '''
    if path: return FileMap(path, instructions, **kw)
    else:    return lambda path:file_map(path, instructions, **kw)
