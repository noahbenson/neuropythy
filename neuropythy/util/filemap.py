####################################################################################################
# neuropythy/util/filemap.py
# Utility for presenting a directory with a particular format as a data structure.
# By Noah C. Benson

import os, sys, warnings, logging, six, tarfile, atexit, shutil, posixpath, json, copy, pimms
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
    try: (tb,p) = split_tarball_path(path)
    except Exception: return False
    return tb is not None
def is_tarball_file(path):
    '''
    is_tarball_file(p) yields True if p is a valid path of a tarball file. Trailing :paths are not
      allowed (see is_tarball_path).
    '''
    (tb,p) = split_tarball_path(path)
    return tb is not None and p == ''
osf_basepath = 'https://api.osf.io/v2/nodes/%s/files/%s/'
def _osf_tree(proj, path=None, base='osfstorage'):
    if path is None: path = (osf_basepath % (proj, base))
    else:            path = (osf_basepath % (proj, base)) + path.lstrip('/')
    dat = json.loads(url_download(path, None))
    if 'data' not in dat: raise ValueError('Cannot detect kind of url for ' + path)
    res = {}
    if pimms.is_map(dat['data']): return dat['data']['links']['download']
    while dat is not None:
        links = dat.get('links', {})
        dat = dat['data']
        for u in dat:
            for r in [u['attributes']]:
                res[r['name']] = (u['links']['download'] if r['kind'] == 'file' else
                                  curry(lambda r: _osf_tree(proj, r, base), r['path']))
        nxt = links.get('next', None)
        if nxt is not None: dat = json.loads(url_download(nxt, None))
        else: dat = None
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
    (bpth, pths) = (pths[0].strip('/'), [p for p in pths[1:] if p != ''])
    if root is None: root = _osf_tree(bpth, base=base)
    return reduce(lambda m,k: m[k], pths, root)

class BasicPath(object):
    '''
    BasicPath is the core path type that has utilities for handling various kinds of path
    operations through the pseudo-path interface.
    '''
    def __init__(self, base_path, pathmod=posixpath, cache_path=None):
        object.__setattr__(self, 'base_path', base_path)
        object.__setattr__(self, 'pathmod', pathmod)
        object.__setattr__(self, 'sep', pathmod.sep)
        object.__setattr__(self, 'cache_path', cache_path)
    def __setattr__(self, k, v): raise TypeError('path objects are immutable')
    def join(self, *args):
        args = [a for a in args if a != '']
        if len(args) == 0: return ''
        else: return self.pathmod.join(*args)
    def osjoin(self, *args):
        args = [a for a in args if a != '']
        if len(args) == 0: return ''
        else: return os.path.join(*args)
    def posixjoin(self, *args):
        args = [a for a in args if a != '']
        if len(args) == 0: return ''
        else: return posixpath.join(*args)
    def split(self, *args): return self.pathmod.split(*args)
    def to_ospath(self, path):
        path = self.pathmod.normpath(path)
        ps = []
        while path != '':
            (path,fl) = self.pathmod.split(path)
            ps.append(fl)
        return os.path.join(*reversed(ps))
    def base_find(self, rpath):
        '''
        Checks to see if the given path exists/can be found at the given base source; does not check
        the local cache and instead just checks the base source. If the file is not found, yields
        None; otherwise yields the relative path from the base_path.
        '''
        # this base version of the function assumes that pathmod.exists is sufficient:
        flnm = self.join(self.base_path, rpath)
        if self.pathmod.exists(flnm): return rpath
        else: return None
    def ensure_path(self, rpath, cpath):
        '''
        Called whenever a cache-path is supplied to getpath and the file is not already found there;
        must return a file path on the local filesystem.
        '''
        # for the base, we have no action here; for other types we may need to copy/download/extract
        # something then return that path.
        flnm = self.join(self.base_path, rpath)
        if os.path.exists(flnm): return flnm
        else: raise ValueError('ensure_path called for non-existant file: %s' % flnm)
    def _make_cache_path(self): return tmpdir(delete=True)
    def _cache_tarball(self, rpath, base_path=''):
        lpath = rpath[len(self.base_path):] if rpath.startswith(self.base_path) else rpath
        if self.cache_path is None: object.__setattr__(self, 'cache_path', self._make_cache_path())
        cpath = self.osjoin(self.cache_path, self.to_ospath(lpath))
        if os.path.isdir(cpath):
            (tbloc, tbinternal) = split_tarball_path(self.base_path)
            if tbloc is None: (tbloc, cpath) = (cpath, cpath)
            else:
                if self.base_path is None or self.base_path == '':
                    raise ValueError('attempting to cache unknown base file')
                tbname = self.split(tbloc)[-1]
                tbloc = cpath = self.osjoin(cpath, tbname)
                base_path = self.osjoin(base_path, tbinternal)
                if not os.path.exists(cpath): tbloc = self.ensure_path(rpath, cpath)
        else: tbloc = cpath
        tp = TARPath(tbloc, base_path, cache_path=cpath)
        if not os.path.exists(tp.tarball_path):
            self.ensure_path(lpath, tp.tarball_path)
        return tp
    def _check_tarball(self, *path_parts):
        rpath = self.join(*path_parts)
        # start by checking our base path:
        if self.base_path is not None and self.base_path != '':
            (tbloc, tbinternal) = split_tarball_path(self.base_path)
            if tbloc is not None:
                if tbinternal == '':
                    # we're fine; we just need to cache the current file...
                    return (self._cache_tarball(''), rpath)
                else:
                    # We copy ourselves to handle this base-path
                    tmp = copy.copy(self)
                    object.__setattr__(tmp, 'base_path', tbloc)
                    rpath = self.join(tbinternal, rpath)
                    # we defer to this path object with the new relative path:
                    return (tmp._cache_tarball(''), rpath)
        # okay, next check the relative path
        fpath = self.join('' if self.base_path is None else self.base_path, rpath)
        (tbloc, tbinternal) = split_tarball_path(fpath)
        if tbloc is not None:
            tbp = self._cache_tarball(tbloc)
            if tbinternal == '':
                return (None, tbp.tarball_path)
            else:
                return (tbp, tbinternal)
        # otherwise, we have no tarball on the path and just need to return ourselves as we are:
        return (self, rpath)
    def find(self, *path_parts):
        yes = self.join(*path_parts)
        (pp, rpath) = self._check_tarball(*path_parts)
        # if pp is none, we are requesting a cached tarball path rpath
        if pp is None: return yes
        # if that gave us back a different path object, we defer to it:
        if pp is not self: return yes if pp.find(rpath) is not None else None
        # check the cache path first
        if self.cache_path is not None:
            cpath = self.osjoin(self.cache_path, self.to_ospath(rpath))
            if os.path.exists(cpath): return yes
        # okay, check the base
        return self.base_find(rpath)
    def exists(self, *path_parts):
        return self.find(*path_parts) is not None
    def getpath(self, *path_parts):
        (pp, rpath) = self._check_tarball(*path_parts)
        # if pp is none, we are requesting a cached tarball path rpath
        if pp is None: return rpath
        # if that gave us back a different path object, we defer to it:
        if pp is not self: return pp.getpath(rpath)
        # check the cache path first
        rp = pp.find(rpath)
        fpath = self.join(self.base_path, rpath)
        if rp is None: raise ValueError('getpath: path not found: %s' % fpath)
        if self.cache_path is not None:
            cpath = self.osjoin(self.cache_path, self.to_ospath(rp))
            if len(rp) == 0:
                # we point to a file and are caching it locally...
                flnm = self.split(fpath)[-1]
                cpath = os.path.join(cpath, flnm)
            if os.path.exists(cpath): return cpath
            return self.ensure_path(rp, cpath)
        else: return self.ensure_path(rp, None)
    def ls(self, *path_parts):
        (pp, rpath) = self._check_tarball(*path_parts)
        # if pp is None, we are requesting a cached tarball path rpath
        if pp is None:
            # We want a listing of the tarball's root.
            pp = self._cache_tarball(rpath)
            rpath = ''
        # Whether pp is or is not this object, we defer to it.
        return pp.listdir(rpath)
    def listdir(self, rpath):
        raise TypeError("type %s does not implement listdir" % (type(self).__name__,))

class OSPath(BasicPath):
    def __init__(self, base_path, cache_path=None):
        BasicPath.__init__(self, base_path, pathmod=os.path, cache_path=cache_path)
    def to_ospath(self, path): return path
    def listdir(self, path):
        return os.listdir(os.path.join(self.base_path, path))
class URLPath(BasicPath):
    def __init__(self, base_path, cache_path=None):
        BasicPath.__init__(self, base_path, pathmod=posixpath, cache_path=cache_path)
    def base_find(self, rpath):
        if is_url(self.join(self.base_path, rpath)): return rpath
        else: return None
    def ensure_path(self, rpath, cpath):
        url = self.join(self.base_path, rpath)
        cdir = os.path.split(cpath)[0]
        if not os.path.isdir(cdir): os.makedirs(cdir, mode=0o755)
        return url_download(url, cpath)
class OSFPath(BasicPath):
    def __init__(self, base_path, cache_path=None):
        BasicPath.__init__(self, base_path, pathmod=posixpath, cache_path=cache_path)
        # if there's a tarball on the base path, we need to grab it instead of doing a typical
        # osf_crawl
        object.__setattr__(self, 'osf_tree', Ellipsis)
    def _find_url(self, rpath):
        if self.osf_tree is Ellipsis:
            (tbloc, tbinternal) = split_tarball_path(self.base_path)
            if tbloc == None: tree = osf_crawl(self.base_path)
            else: tree = osf_crawl(tbloc)
            object.__setattr__(self, 'osf_tree', tree)
        fl = self.osf_tree
        parts = [s for s in rpath.split(self.sep) if s != '']
        # otherwise walk the tree...
        for pp in parts:
            if   pp == '':                           continue
            elif not pimms.is_str(fl) and pp in fl:  fl = fl[pp]
            else:                                    return None
        return fl
    def base_find(self, rpath):
        fl = self._find_url(rpath)
        if fl is None: return None
        else: return rpath
    def ensure_path(self, rpath, cpath):
        fl = self._find_url(rpath)
        if not pimms.is_str(fl):
            if not os.path.isdir(cpath): os.makedirs(cpath, mode=0o755)
            return cpath
        else:
            cdir = os.path.split(cpath)[0]
            if not os.path.isdir(cdir): os.makedirs(cdir, mode=0o755)
        return url_download(fl, cpath)
    def listdir(self, path):
        fl = self._find_url(path)
        if fl is None: raise FileNotFoundError(path)
        if pimms.is_str(fl): raise NotADirectoryError(path)
        return list(fl.keys())
class S3Path(BasicPath):
    def __init__(self, fs, base_path, cache_path=None):
        BasicPath.__init__(self, base_path, pathmod=posixpath, cache_path=cache_path)
        object.__setattr__(self, 's3fs', fs)
    def base_find(self, rpath):
        fpath = self.join(self.base_path, rpath)
        if self.s3fs.exists(fpath): return rpath
        else: return None
    def ensure_path(self, rpath, cpath):
        url = self.join(self.base_path, rpath)
        cdir = os.path.split(cpath)[0]
        if not os.path.isdir(cdir): os.makedirs(cdir, mode=0o755)
        self.s3fs.get(url, cpath)
        return cpath
    def listdir(self, path):
        fpath = self.join(self.base_path, path)
        if fpath.lower().startswith('s3://'): fpath = fpath[5:]
        fpath = fpath.strip('/')
        lst = self.s3fs.ls(fpath)
        if len(lst) == 0: raise FileNotFoundError(path)
        elif len(lst) == 1 and lst[0] == fpath: raise NotADirectoryError(path)
        n = len(fpath) + 1
        return [fl[n:] for fl in lst]
class TARPath(OSPath):
    def __init__(self, tarpath, basepath='', cache_path=None, tarball_name=None):
        OSPath.__init__(self, basepath, cache_path=cache_path)
        object.__setattr__(self, 'tarball_path', tarpath)
        object.__setattr__(self, 'base_path', basepath)
        if cache_path == tarpath:
            # we need to prep the path
            tarfl = os.path.split(tarpath)[1]
            cpath = os.path.join(cache_path, 'contents')
            tarpath = os.path.join(cache_path, tarfl)
            if os.path.isfile(cache_path):
                # we need to move things around
                td = tmpdir(delete=True)
                tmpfl = os.path.join(td,tarfl)
                shutil.move(cache_path, tmpfl)
                if not os.path.isdir(cpath): os.makedirs(cpath, mode=0o755)
                shutil.move(tmpfl, tarpath)
            object.__setattr__(self, 'tarball_path', tarpath)
            object.__setattr__(self, 'cache_path', cpath)
        # get the tarball 'name'
        flnm = os.path.split(self.tarball_path if tarball_name is None else tarball_name)[-1]
        tarball_name = flnm.split('.tar')[0]
        object.__setattr__(self, 'tarball_name', tarball_name)
    tarball_fileobjs = {}
    @staticmethod
    def tarball_fileobj(path):
        if path not in TARPath.tarball_fileobjs:
            TARPath.tarball_fileobjs[path] = tarfile.open(path, 'r')
        return TARPath.tarball_fileobjs[path]
    def base_find(self, rpath):
        rpath = self.join(self.base_path, rpath)
        # check the cache path for both this rpath and this rpath + tarball name:
        cpath = self.join(self.cache_path, rpath)
        if os.path.exists(cpath): return rpath
        rpalt = self.join(self.tarball_name, rpath)
        cpalt = self.join(self.cache_path, rpalt)
        if os.path.exists(cpath): return rpalt
        # okay, see if they're int he tarfile
        tfl = TARPath.tarball_fileobj(self.tarball_path)
        try: found = bool(tfl.getmember(rpath))
        except Exception: found = False # might be that we need to prepend the tarball-name path
        if found: return rpath
        try: found = bool(tfl.getmember(rpalt))
        except Exception: pass
        if found: return rpalt
        # could still have a ./ ...
        rpath = self.join('.', rpath)
        rpalt = self.join('.', rpalt)
        try: found = bool(tfl.getmember(rpath))
        except Exception: found = False # might be that we need to prepend the tarball-name path
        if found: return rpath
        try: found = bool(tfl.getmember(rpalt))
        except Exception: pass
        if found: return rpalt
        else: return None
    def ensure_path(self, rpath, cpath):
        # we ignore cpath in this case
        with tarfile.open(self.tarball_path, 'r') as tfl:
            tfl.extract(rpath, self.cache_path)
            return cpath
    def listdir(self, path):
        fpath = self.join(self.base_path, rpath)
        fpath = fpath.strip('/') + '/'
        fn = len(fpath)
        with tarfile.open(self.tarball_path, 'r') as tfl:
            allfiles = tfl.getmembers()
        mems = []
        isdir = None
        for fl in allfiles:
            if fl.name == fpath[:-1]:
                if fl.isdir(): isdir = True
                else: raise NotADirectoryError(fpath)
            elif fl.name.startswith(fpath):
                nm = fl.name[fn:]
                if len(nm.split('/')) > 1: continue
                mems.append(nm)
        if len(mems) == 0 and not isdir: raise FileNotFoundError(path)
        return mems

@pimms.immutable
class PseudoPath(ObjectWithMetaData):
    '''
    The PseudoPath class represents either directories themselves, tarballs, or URLs as if they were
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
        pseudo_path.source_path is the source path of the the given pseudo-path object.
        '''
        if sp is None: return os.path.join(os.path.sep)
        if is_tuple(sp) and isinstance(sp[0], s3fs.S3FileSystem): return sp
        if not pimms.is_str(sp): raise ValueError('source_path must be a string/path')
        if is_url(sp) or is_s3_path(sp): return sp
        return os.path.expanduser(os.path.expandvars(sp))
    @pimms.param
    def cache_path(cp):
        '''
        pseudo_path.cache_path is the optionally provided cache path; this is the same as the
        storage path unless this is None.
        '''
        if cp is None: return None
        if not pimms.is_str(cp): raise ValueError('cache_path must be a string')
        return os.path.expanduser(os.path.expandvars(cp))
    @pimms.param
    def delete(d):
        '''
        pseudo_path.delete is True if the pseudo_path self-deletes on Python exit and False
        otherwise; if this is Ellipsis, then self-deletes only when the cache-directory is created
        by the PseudoPath class and is a temporary directory (i.e., not explicitly provided).
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
    @pimms.value
    def _path_data(source_path, cache_path, delete, credentials):
        need_cache = True
        rpr = None
        delete = True if delete is Ellipsis else delete
        cp = {'cp':lambda:(tmpdir(delete=delete) if cache_path is None else cache_path)}
        cp = pimms.lazy_map(cp)
        # Okay, it might be a directory, an Amazon S3 URL, a different URL, or a tarball
        if is_tuple(source_path):
            (el0,pp) = (source_path[0], source_path[1:])
            if s3fs is not None and isinstance(el0, s3fs.S3FileSystem):
                base_path = urljoin(*pp)
                bpl = base_path.lower()
                if   bpl.startswith('s3://'): base_path = base_path[5:]
                elif bpl.startswith('s3:/'):  base_path = base_path[4:]
                elif bpl.startswith('s3:'):   base_path = base_path[3:]
                rpr = 's3://' + base_path
                pathmod = S3Path(el0, base_path, cache_path=cp['cp'])
            elif is_pseudo_path(el0):
                raise NotImplementedError('pseudo-path nests not yet supported')
            else: raise ValueError('source_path tuples must start with S3FileSystem or PseudoPath')
        elif os.path.isfile(source_path) and is_tarball_file(source_path):
            base_path = ''
            pathmod = TARPath(source_path, base_path, cache_path=cp['cp'])
            rpr = os.path.normpath(source_path)
        elif os.path.exists(source_path):
            rpr = os.path.normpath(source_path)
            base_path = source_path
            pathmod = OSPath(source_path, cache_path=cache_path)
        elif is_s3_path(source_path):
            if s3fs is None: raise ValueError('s3fs module is not installed')
            elif credentials is None: fs = s3fs.S3FileSystem(anon=True)
            else: fs = s3fs.S3FileSystem(key=credentials[0], secret=credentials[1])
            base_path = source_path
            pathmod = S3Path(fs, base_path, cache_path=cp['cp'])
            rpr = source_path
        elif is_osf_path(source_path):
            base_path = source_path
            pathmod = OSFPath(base_path, cache_path=cp['cp'])
            rpr = source_path
        elif is_url(source_path):
            base_path = source_path
            pathmod = URLPath(base_path, cache_path=cp['cp'])
            rpr = source_path
        elif is_tarball_path(source_path):
            # must be a a file starting with a tarball:
            (tbloc, tbinternal) = split_tarball_path(source_path)
            pathmod = TARPath(tbloc, tbinternal, cache_path=cp['cp'])
            rpr = os.path.normpath(source_path)
            base_path = ''
            # ok, don't know what it is...
        else: raise ValueError('Could not interpret source path: %s' % source_path)
        tmp = {'repr':rpr, 'pathmod':pathmod}
        if not cp.is_lazy('cp'):
            tmp['cache_path'] = cp['cp']
        return pyr.pmap(tmp)
    @pimms.require
    def check_path_data(_path_data):
        '''
        Ensures that _path_data is created without error.
        '''
        return ('pathmod' in _path_data)
    @pimms.value
    def actual_cache_path(_path_data):
        '''
        pdir.actual_cache_path is the cache path being used by the pseudo-path pdir; this may differ
          from the pdir.cache_path if the cache_path provided was None yet a temporary cache path
          was needed.
        '''
        return _path_data.get('cache_path', None)
    @pimms.value
    def actual_source_path(source_path):
        '''
        pdir.actual_source_path is identical to pdir.source_path except when the input source_path
        is a tuple (e.g. giving an s3fs object followed by a source path), in which case
        pdir.actual_source_path is a string representation of the source path.
        '''
        if source_path is None: return None
        if pimms.is_str(source_path): return source_path
        s = urljoin(*source_path[1:])
        if not s.lower().startswith('s3://'): s = 's3://' + s
        return s
    def __repr__(self):
        p = self._path_data['repr']
        return "pseudo_path('%s')" % p
    def join(self, *args):
        '''
        pdir.join(args...) is equivalent to os.path.join(args...) but always appropriate for the
          kind of path represented by the pseudo-path pdir.
        '''
        join = self._path_data['pathmod'].join
        return join(*args)
    def find(self, *args):
        '''
        pdir.find(paths...) is similar to to os.path.join(paths...) but it only yields the joined
          relative path if it can be found inside pdir; otherwise None is yielded. Note that this
          does not extract or download the path--it merely ensures that it exists.
        '''
        pmod = self._path_data['pathmod']
        return pmod.find(*args)
    def listpath(self, *args):
        '''
        pdir.listpath(paths...) returns a list of the files that are part of the subdirectory
           given by joining the paths. The pseudo-path pdir must support directory listing.
        '''
        pmod = self._path_data['pathmod']
        return pmod.ls(*args)
    def local_path(self, *args):
        '''
        pdir.local_path(paths...) is similar to os.path.join(pdir, paths...) except that it
          additionally ensures that the path being requested is found in the pseudo-path pdir then
          ensures that this path can be found in a local directory by downloading or extracting it
          if necessary. The local path is yielded.
        '''
        pmod = self._path_data['pathmod']
        return pmod.getpath(*args)
    def local_cache_path(self, *args):
        '''
        pdir.local_cache_path(paths...) is similar to os.path.join(pdir, paths...) except that it
          yields a local version of the given path, much like pdir.local_path(paths...). The 
          local_cache_path function differs from the local_path function in that, if no existing
          file is found at the given destination, no error is raised and the path is still returned.
        '''
        # if the file exists in the pseudo-path, just return the local path
        if self.find(*args) is not None: return self.local_path(*args)
        cp = self._path_data.get('cache_path', None)
        if cp is None: cp = self.source_path
        return os.path.join(cp, *args)
    def subpath(self, *args):
        '''
        pdir.subpath(paths...) returns a new pseudo-path object that points to the subdirectory
          implied by the given paths, which are joined. The new pseudo-path uses the matching
          subdirectory of pdir for its cache-path.
        '''
        cp = self.actual_cache_path
        sp = self.source_path
        if cp is not None:
            cp = os.path.join(cp, *args)
        sp = self.join(sp, *args)
        return PseudoPath(sp, cache_path=cp, delete=False, credentials=self.credentials,
                          meta_data={'superpath': self})
def is_pseudo_path(pdir):
    '''
    is_pseudo_path(pdir) yields True if the given object pdir is a pseudo-path object and False
      otherwise.
    '''
    return isinstance(pdir, PseudoPath)
def pseudo_path(source_path, cache_path=None, delete=Ellipsis, credentials=None, meta_data=None):
    '''
    pseudo_path(source_path) yields a pseudo-pathectory object that represents files in the given
      source path.

    pseudo-path objects act as an interface for loading data from abstract sources. The given source
    path may be either a directory, a (possibly zipped) tarball, or a URL. In all cases but the
    local directory, the pseudo-path object will quietly extract/download the requested files to a
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
        that if the cache path is not deleted, it can be reused across sessions--the pseudo-path
        will always check for files in the cache path before extracting or downloading them.
      * delete (default: Ellipsis) may be set to True or False to declare that the cache directory
        should be deleted at system exit (assuming a normal Python system exit). If Ellipsis, then
        the cache_path is deleted only if it is created by the pseudo-path object--given cache paths
        are never deleted.
      * credentials (default: None) may be set to a valid set of Amazon S3 credentials for use if
        the source path is an S3 path. The contents are passed through the to_credentials function.
      * meta_data (default: None) specifies an optional map of meta-data for the pseudo-path.
    '''
    return PseudoPath(source_path, cache_path=cache_path, delete=delete, credentials=credentials,
                     meta_data=meta_data)
def to_pseudo_path(obj):
    '''
    to_pseudo_path(obj) yields a pseudo-path object that has been coerced from the given obj or
      raises an exception. If the obj is a pseudo-path already, it is returned unchanged.
    '''
    if   is_pseudo_path(obj):   return obj
    elif pimms.is_str(obj):    return pseudo_path(obj)
    elif pimms.is_vector(obj):
        if len(obj) > 0 and pimms.is_map(obj[-1]): (obj,kw) = (obj[:-1],obj[-1])
        else: kw = {}
        return pseudo_path(*obj, **kw)
    else: raise ValueError('cannot coerce given object to a pseudo-path: %s' % obj)
    
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
        (tb,p) = split_tarball_path(p)
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
            return lambda flnm, meta: load(flnm, meta['format']) if 'format' in meta else load(flnm)
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
        p0 = 0
        if is_pseudo_path(p): return p
        p = FileMap.valid_path(p)
        if p is None: raise ValueError('Path must be a directory or a tarball: %s' % p0)
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
    def actual_path(path):
        '''
        filemap.actual_path is always a string path (even when filemap.path is a pseudo-path).
        '''
        if is_pseudo_path(path): return path.source_path
        else: return path
    @pimms.value
    def actual_cache_path(cache_path, cache_delete, actual_path, supplemental_paths):
        '''
        filemap.actual_cache_path is the cache path used by the filemap, if needed.
        '''
        path = actual_path
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
        inst = pimms.merge(*(argmaps + (kwargs,)))
        flnm = flnm.format(**inst)
        args = pimms.merge(*argmaps, **kwargs)
        #logging.info('FileMap: loading file "%s"...\n' % flnm) #debug
        try:
            lpth = pdir.local_path(flnm)
            #logging.info('     ... local path: %s\n' % lpth) #debug
            args = pimms.merge(*argmaps, **kwargs)
            loadfn = inst['load'] if 'load' in args else loadfn
            #filtfn = inst['filt'] if 'filt' in args else lambda x,y:x
            dat = loadfn(lpth, args)
            #dat = filtfn(dat, args)
        except Exception:
            dat = None
            #raise
        # check for miss instructions if needed
        if dat is None and 'miss' in argmaps: miss = args['miss']
        else: miss = None
        if pimms.is_str(miss) and miss.lower() in ('error','raise','exception'):
            raise ValueError('File %s failed to load' % flnm)
        elif miss is not None:
            dat = miss(flnm, args)
        return dat
    @staticmethod
    def _parse_path(flnm, spaths, path_parameters, inst):
        flnm = flnm.format(**pimms.merge(path_parameters, inst))
        p0 = None
        for k in six.iterkeys(spaths):
            if k is None: continue
            if flnm.startswith(k + ':'):
                (flnm, p0) = (flnm[(len(k)+1):], k)
                break
        return (p0, flnm)
    @pimms.value
    def pseudo_paths(path, supplemental_paths, actual_cache_path):
        '''
        fmap.pseudo_paths is a mapping of pseduo-dirs in the file-map fmap. The primary path's
        pseudo-path is mapped to the key None.
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
            spaths[s] = pseudo_path(p, delete=False, cache_path=cp)
        if actual_cache_path:
            if n > 0: cp = os.path.join(actual_cache_path, 'main')
            else:     cp = actual_cache_path
            if not os.path.isdir(cp): os.makedirs(os.path.abspath(cp), 0o755)
        spaths[None] = (path if is_pseudo_path(path) else
                        pseudo_path(path,delete=False,cache_path=cp))
        return pyr.pmap(spaths)
    @pimms.value
    def data_files(pseudo_paths, path_parameters, load_function, meta_data, _parsed_instructions):
        '''
        filemap.data_files is a lazy map whose keys are filenames and whose values are the loaded
        files.
        '''
        (data_files, data_tree) = _parsed_instructions
        res = {}
        for (flnm, inst) in six.iteritems(data_files):
            (pathnm, fn) = FileMap._parse_path(flnm, pseudo_paths, path_parameters, inst)
            res[fn] = curry(FileMap._load,
                            pseudo_paths[pathnm], flnm, load_function,
                            path_parameters, meta_data, inst)
        return pimms.lazy_map(res)
    @pimms.value
    def data_tree(_parsed_instructions, pseudo_paths, supplemental_paths, path_parameters,
                  data_files):
        '''
        filemap.data_tree is a lazy data-structure of the data loaded by the filemap's instructions.
        '''
        data_tree = _parsed_instructions[1]
        def visit_data(d):
            d = {k:visit_maps(v) for (k,v) in six.iteritems(d)}
            return data_struct(d)
        def lookup(flnm, inst):
            val = data_files[flnm]
            if 'filt' in inst: val = inst['filt'](val)
            return val
        def visit_maps(m):
            r = {}
            anylazy = False
            for (k,v) in six.iteritems(m):
                kk = k if isinstance(k, tuple) else [k]
                for k in kk:
                    if len(v) > 0 and '_relpath' in next(six.itervalues(v)):
                        (flnm,inst) = next(six.iteritems(v))
                        flnm = FileMap._parse_path(flnm, pseudo_paths, path_parameters, inst)[1]
                        r[k] = curry(lookup, flnm, inst)
                        anylazy = True
                    else: r[k] = visit_data(v)
            return pimms.lazy_map(r) if anylazy else pyr.pmap(r)
        return visit_data(data_tree)
def is_file_map(fmap):
    '''
    if_file_map(fmap) yields True if the given object fmap is a file map object and False otherwise.
    '''
    return isinstance(fmap, FileMap)
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
