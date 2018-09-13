####################################################################################################
# neuropythy/util/conf.py
# Contains configuration code for setting up the environment.
# By Noah C. Benson

import os, sys, six, json, warnings, pimms

def loadrc(filename):
    '''
    loadrc(filename) yields the dict object encoded in JSON in the given filename; if the filename
      does not exist or does not contain a valid JSON dict, then an error is raised.
    '''
    filename = os.path.expanduser(os.path.expandvars(filename))
    if not os.path.isfile(filename): raise ValueError('Filename %s does not exist' % filename)
    with open(filename, 'r') as fl:
        dat = json.load(fl)
    try: dat = dict(dat)
    except: raise ValueError('Given file %s does not contain a dictionary' % filename)
    return dat
def saverc(filename, dat, overwrite=False):
    '''
    saverc(filename, d) saves the given configuration dictionary d to the given filename in JSON
      format. If d is not a dictionary or if filename already exists or cannot be created, an error
      is raised. This funciton does not create directories.

    The optional argument overwrite (default: False) may be passed as True to overwrite files that
    already exist.
    '''
    filename = os.path.expanduser(os.path.expandvars(filename))
    if not overwrite and os.path.isfile(filename):
        raise ValueError('Given filename %s already exists' % filename)
    if not pimms.is_map(dat):
        try: dat = dict(dat)
        except: raise ValueError('Given config data must be a dictionary')
    with open(filename, 'w') as fl:
        json.dump(dat, fl, sort_keys=True)
    return filename

# the private class that handles all the details...
class ConfigMeta(type):
    def __getitem__(cls,name):
        return cls._getitem(cls,name)
    def __setitem__(cls,name,val):
        return cls._setitem(cls,name,val)
    def __len__(cls):
        return cls._len(cls)
    def __iter__(cls):
        return cls._iter(cls)
        
@six.add_metaclass(ConfigMeta)
class config(object):
    '''
    neuropythy.util.conf.config is a class that manages configuration items for neuropythy and their
    interface. This class reads in the user's neuropythy-rc file at startup, which by default is in
    the user's home directory named "~/.npythyrc" though it may be altered by setting the
    environment variable NPYTHYRC. Environment variables that are associated with configurable
    variables always override the values in the RC file, and any direct set

    To declare a configurable variable in Neuropythy, you must use config.declare().
    '''
    _rc = None
    @staticmethod
    def rc():
        '''
        config.rc() yields the data imported from the Neuropythy rc file, if any.
        '''
        if config._rc is None:
            # First: We check to see if we have been given a custom nptyhrc file:
            npythyrc_path = os.path.expanduser('~/.npythyrc')
            if 'NPYTHRC' in os.environ:
                npythyrc_path = os.path.expanduser(os.path.expandvars(os.environ['NPYTHYRC']))
            # the default config:
            if os.path.isfile(npythyrc_path):
                try:
                    config._rc = loadrc(npythyrc_path)
                    config._rc['npythyrc_loaded'] = True
                except Exception as err:
                    warnings.warn('Could not load neuropythy RC file: %s' % npythyrc_path)
                    config._rc = {'npythyrc_loaded':False,
                                  'npythyrc_error': err}
            else:
                config._rc = {'npythyrc_loaded':False}
            config._rc['npythyrc'] = npythyrc_path
        return config._rc
    _vars = {}
    @staticmethod
    def declare(name, rc_name=None, environ_name=None, filter=None, default_value=None):
        '''
        config.declare(name) registers a configurable variable with the given name to the neuropythy
          configuration system. This allows the variable to be looked up in the neuropythy RC-file
          and the neuropythy environment variables.

        By default, the variable will be assumed to have the identical name in the neuropythy
        RC-file and the environment variable ('NPYTHY_' + name.upper()) is searched for in the
        environment. The environment variable will always overwrite the RC-file value if both are
        provided. Note that all inputs from the environment or the RC-file are parsed as JSON inputs
        prior to being put in the config object itself.

        The following optional arguments may be provided:
          * rc_name (default: None) specifies the name of the variable in the neuropythy RC-file; if
            None, then uses the name.lower().
          * environ_name (default: None) specifies the name of the variable to look for in the
            os.environ dictionary; if None, then uses ('NPYTHY_' + name.upper()).
          * filter (default: None) specifies a function f that is passed the provided value u; the 
            resulting config value is f(u).
          * default_value (default: None) specifies the default value the configuration item should
            take if not provided.
        '''
        if rc_name is None: rc_name = name.lower()
        if environ_name is None: environ_name = 'NPYTHY_' + name.upper()
        # okay, see if the name exists
        if name in config._vars: raise ValueError('Multiple config items declared for %s' % name)
        config._vars[name] = (rc_name, environ_name, filter, default_value)
        return True
    _vals = {}
    @staticmethod
    def _getitem(self, name):
        if name not in config._vars: raise KeyError(name)
        if name not in config._vals:
            (rcname, envname, fltfn, dval) = config._vars[name]
            val = dval
            rcdat = config.rc()
            # see if it's in the rc-file first, then the environment
            if rcname  in rcdat: val = rcdat[rcname]
            if envname in os.environ:
                val = os.environ[envname]
                try: val = json.loads(val)
                except: pass # it's a string if it can't be json'ed
            # if there's a filter, run it
            if fltfn is not None:
                val = fltfn(val)
                try: True
                except: val = dval # failure--reset to default
            config._vals[name] = val
        return config._vals[name]
    @staticmethod
    def _setitem(self, name, val):
        if name not in config._vars:
            raise ValueError('Configurable neuropythy key "%s" not declared' % name)
        (rcname, envname, fltfn, dval) = config._vars[name]
        self._vals[name] = val if fltfn is None else fltfn(val)
    @staticmethod
    def _iter(self): return six.iterkeys(self._vars)
    @staticmethod
    def _len(self): return len(self._vars)
    @staticmethod
    def keys(): return config._vars.keys()
    @staticmethod
    def values(): return map(lambda k:config[k], config.keys())
    @staticmethod
    def items(): return map(lambda k:(k,config[k]), config.keys())
    @staticmethod
    def todict(): return {k:config[k] for k in config.keys()}
    
