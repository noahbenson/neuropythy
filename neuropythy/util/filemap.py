####################################################################################################
# neuropythy/util/filemap.py
# Utility for presenting a directory with a particular format as a data structure.
# By Noah C. Benson

import os, warnings, six, tarfile, pimms
import numpy          as np
import pyrsistent     as pyr
from   .core      import (library_path, curry, ObjectWithMetaData, AutoDict, data_struct)

@pimms.immutable
class FileMap(ObjectWithMetaData):
    '''
    The FileMap class is a pimms immutable class that tracks a set of FileMap format instructions
    with a valid path containing data of that format.
    '''
    def __init__(self, path, instructions, path_parameters=None, data_hierarchy=None,
                 load_function=None, meta_data=None, **kw):
        ObjectWithMetaData.__init__(self, meta_data=meta_data)
        self.path = path
        self.instructions = instructions
        self.data_hierarchy = data_hierarchy
        self.supplemental_paths = kw
        self.path_parameters = path_parameters
        self.load_function = load_function
    @staticmethod
    def valid_path(p):
        '''
        FileMap.valid_path(path) yields os.path.abspath(path) if path is either a directory or a
          tarball file; otherwise yields None.
        '''
        return (os.path.abspath(p) if os.path.isdir(p)      else
                None               if not os.path.exists(p) else
                os.path.abspath(p) if p.endswith('.tar')    else
                os.path.abspath(p) if p.endswith('.tar.gz') else
                None)
    @pimms.param
    def load_function(lf):
        '''
        filemap.load_function is the function used to load data by the filemap.
        '''
        if lf is None:
            from ..io import load
            return load
        else: return lf
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
        known_filekeys = ('load','filt')
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
    @staticmethod
    def _load_file(flnm, inst, loadfn):
        loadfn = inst['load'] if 'load' in inst else loadfn
        dat = loadfn(flnm)
        return inst['filt'](dat) if 'filt' in inst else dat
    @staticmethod
    def _deduce_filename(flnm, path, supplemental_paths, path_parameters):
        p0 = path
        if ':' in flnm:
            s = flnm.split(':')[0].format(path_parameters)
            if s in supplemental_paths: p0 = supplemental_paths[s]
        flnm = os.path.join(p0, flnm)
        return flnm.format(path_parameters)
    @pimms.value
    def data_files(_parsed_instructions, path, supplemental_paths, path_parameters, load_function):
        '''
        filemap.data_files is a lazy map whose keys are filenames and whose values are the loaded
        files.
        '''
        (data_files, data_tree) = _parsed_instructions
        res = {}
        for (flnm, inst) in six.iteritems(data_files):
            flnm = FileMap._deduce_filename(flnm, path, supplemental_paths, path_parameters)
            res[flnm] = curry(FileMap._load_file, flnm, inst, load_function)
        return pimms.lazy_map(res)
    @pimms.value
    def data_tree(_parsed_instructions, path, supplemental_paths, path_parameters, data_files):
        '''
        filemap.data_tree is a lazy data-structure of the data loaded by the filemap's instructions.
        '''
        data_tree = _parsed_instructions[1]
        class _tmp:
            ident = 0
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
                                                        path_parameters)
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
     * load_function (default: None) may specify the function that is used to load filenames; if None
       then neuropythy.io.load is used.
     * meta_data (default: None) may be passed on to the FileMap object.

    Any additional keyword arguments given to the file_map function will be used as supplemental
    paths.
    '''
    if path: return FileMap(path, instructions, **kw)
    else:    return lambda path:file_map(path, instructions, **kw)
