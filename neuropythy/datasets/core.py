####################################################################################################
# neuropythy/datasets/core.py
# Implementation of builtin datasets for Neuropythy
# by Noah C. Benson

import os, six, shutil, tempfile, atexit, pimms
import numpy as np

from ..util import (config, ObjectWithMetaData)
from ..freesurfer import subject as freesurfer_subject

# We declare a configuration variable, data_cache_root -- where to put the data that is downloaded.
# If this is None / unset, then we'll use a temporary directory and auto-delete it on exit.
def _check_cache_root(path):
    if path is None: return None
    path = os.path.expanduser(os.path.expandvars(path))
    if not os.path.isdir(path): return None
    return os.path.abspath(path)
config.declare('data_cache_root', filter=_check_cache_root)

@pimms.immutable
class Dataset(ObjectWithMetaData):
    '''
    The Dataset class is a simple immutable class that should be implemented by all neuropythy
    datasets. The design is such that neuropythy.data[name] should always (lazily) yield a Dataset
    object specific to the dataset given by name, if it exists and can be loaded.
    
    One reason to require (by convention) that all datasets are distinct classes is that it should
    thus be easy to evaluate help(ny.data[name]) to see help on the given dataset. If you overload
    this class, be sure to overload the documentation.
    '''
    def __init__(self, name, meta_data=None, custom_directory=None,
                 create_directories=True, create_mode=0o755):
        ObjectWithMetaData.__init__(self, meta_data)
        self.custom_directory = custom_directory
        self.name = name
        self.create_directories = create_directories
        self.create_mode = create_mode
    def __repr__(self): return self.repr
    @pimms.value
    def repr(name):
        '''
        dataset.repr is the representation string used for the given dataset.
        '''
        return ("Dataset('%s')" % name) if pimms.is_str(name) else ("Dataset%s" % (name,))
    @staticmethod
    def to_name(nm):
        '''
        Dataset.to_name(name) yields a valid dataset name equivalent to the given name or raises an
          error if name is not valid. In order to be valid, a name must be either strings or a tuple
          of number and strings that start with a string.
        '''
        if pimms.is_str(nm): return nm
        if not pimms.is_vector(nm): raise ValueError('name must be a string or tuple')
        if len(nm) < 1: raise ValueError('names that are tuples must have at least one element')
        if not pimms.is_str(nm): raise ValueError('names that are tuples must begin with a string')
        if not all(pimms.is_str(x) or pimms.is_number(x) for x in nm):
            raise ValueError('dataset names that are tuples must contain only strings and numbers')
        return tuple(nm)
    @pimms.param
    def custom_directory(d):
        '''
        dataset.custom_directory is None if no custom directory was provided for the given dataset;
          otherwise it is the provided custom directory.
        '''
        if d is None: return None
        if not pimms.is_str(d): raise ValueError('custom_directory must be a string')
        else: return d
    @pimms.param
    def create_directories(c):
        '''
        dataset.create_directories is True if the dataset was instructed to create its cache
        directory, should it be found to not exist, and is otherwise False.
        '''
        return bool(c)
    @pimms.param
    def create_mode(c):
        '''
        dataset.create_mode is the octal permision mode used to create the cache directory for the
        given dataset, if the dataset had to create its directory at all.
        '''
        return c
    @pimms.param
    def name(nm):
        '''
        dataset.name is either a string or a tuple of strings and numbers that identifies the given
        dataset. If dataset.name is a tuple, then the first element must be a string.
        '''
        return Dataset.to_name(nm)
    @pimms.value
    def cache_root(custom_directory):
        '''
        dataset.cache_root is the root directory in which the given dataset has been cached.
        '''
        if custom_directory is not None: return None
        elif config['data_cache_root'] is None:
            # we create a data-cache in a temporary directory
            path = tempfile.mkdtemp(prefix='npythy_data_cache_')
            if not os.path.isdir(path): raise ValueError('Could not find or create cache directory')
            config['data_cache_root'] = path
            atexit.register(shutil.rmtree, path)
        return config['data_cache_root']
    @pimms.value
    def cache_directory(cache_root, name, custom_directory):
        '''
        dataset.cache_directory is the directory in which the given dataset is cached.
        '''
        if custom_directory is not None: return custom_directory
        return os.path.join(cache_root, (name    if pimms.is_str(name) else
                                         name[0] if len(name) == 1     else
                                         '%s_%x' % (name[0], hash(name[1:]))))
    @pimms.require
    def ensure_cache_directory(cache_directory, create_directories, create_mode):
        '''
        ensure_cache_directory requires that a dataset's cache directory exists and raises an error
        if it cannot be found.
        '''
        if os.path.isdir(cache_directory): return True
        if not create_directories: raise ValueError('dataset cache directory not found: %s')
        os.makedirs(cache_directory, create_mode)
        return True
# We create the dataset repository: this is a lazy map; to add a dataset to it, use the function
# add_dataset(), immediately below
data = pimms.lazy_map({})
def add_dataset(dset, fn=None):
    '''
    add_dataset(dset) adds the given dataset to the neuropythy.data map.
    add_dataset(name, fn) adds a dataset with the given name; fn must be a function of zero
      arguments that yields the dataset.

    add_dataset() always yeilds None or raises an error.
    '''
    global data
    if fn is None:
        if not isinstance(dset, Dataset):
            raise ValueError('Cannot add non-Dataset object to neuropythy datasets')
        nm = dset.name
        data = data.set(nm, dset)
    else:
        nm = Dataset.to_name(dset)
        def _load_dset():
            x = fn()
            if not isinstance(x, Dataset):
                raise ValueError('Loader for dataset %s failed to return a dataset' % nm)
            return x
        data = data.set(nm, _load_dset)
    return None

    
            
        
