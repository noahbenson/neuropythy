####################################################################################################
# neuropythy/util/conf.py
# Contains configuration code for setting up the environment.
# By Noah C. Benson

import os, sys, six, types, json, warnings, pimms


if six.PY2:
    (_tuple_type, _list_type) = (types.TupleType, types.ListType)
    import ConfigParser as confparse
else:
    (_tuple_type, _list_type) = (tuple, list)
    import configparser as confparse

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
    except Exception: raise ValueError('Given file %s does not contain a dictionary' % filename)
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
        except Exception: raise ValueError('Given config data must be a dictionary')
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
    def __repr__(cls):
        return 'config(' + repr({k:cls[k] for k in cls.keys()}) + ')'
        
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
    @staticmethod
    def declare_credentials(name, rc_name=None, environ_name=None, default_value=None,
                            extra_environ=None, filenames=None, aws_profile_name=None):
        '''
        config.declare_credentials(name) registers a configurable variable with the given name to
          the neuropythy configuration system (see also config.declare()). The variable is
          specifically created as a credentials-type variable such that if the config variable is 
          not set, a call to detect_credentials is used to attempt to find the credentials; this
          effectively allows for additional filenames and environment variables to be searched while
          ensuring that strings, files, and json representations all get coerced to (key, secret)
          if possible.

        Note that if you set default_value to something that is neither None nor a valid credentials
        representation, then the config item will still end up as None; this is because the
        default value must fall through the to_credentials() function.

        The following optional arguments from detect_credentials are accepted and passed along to it
        if necessary:
          * extra_environ
          * filenames
          * aws_profile_name
          * default_value (Ellipsis is not allowed via this interface).
        The following optional arguments from config.declare() are accepted and passed along to it:
          * rc_name
          * environ_name

        See also help(config.declare) and help(detect_credentials).
        '''
        if default_value is Ellipsis:
            raise ValueError('Ellipsis is not a valid credentials default_value when calling the'
                             ' declare_credentials() method.')
        def _check_creds(creds):
            if creds is None:
                return detect_credentials(None,
                                          extra_environ=extra_environ,
                                          filenames=filenames,
                                          aws_profile_name=aws_profile_name,
                                          default=default_value)
            else: return to_credentials(creds)
        return config.declare(name,
                              rc_name=rc_name,
                              environ_name=environ_name,
                              filter=_check_creds,
                              default_value=None)
    @staticmethod
    def declare_path(name, 
                     rc_name=None, environ_name=None, filter=None, default_value=None,
                     fail_value=None):
        '''
        config.declare_path(...) is equivalent to config.declare(...) except that it requires in
          addition that the value found for the config item be an existing file or dir on the local
          filesystem. Only if this is true is the value passed to any provided filter.

        This function (similar to config.declare_file and config.declare_dir) is generally expected
        to return some data--the contents of the path--not the path-name itself. For this reason
        the handling of the default_value argument is expanded relative to that of config.declare().
        If no value is provided or if all provided values are either not paths or fail the filter,
        then the default_value is checked (as a path); if this also fails then the fail_value is
        used as the config value. By default this is None, but note that by using a filter function
        that loads the provided path (which will be guaranteed to exist) and yields the data along
        with a fail_value that resembles 'default' data, one can effectively guarantee that data is
        either loaded from a source on the filesystem or is in a coherent default state.

        See also config.declare_dir and config.declare_file.
        '''
        def _check_path(path):
            if path is None:
                # check the default_value...
                if default_value is not None:
                    try: return _check_path(default_value)
                    except Exception: pass
                return None
            path = os.path.expanduser(os.path.expandvars(path))
            if not os.path.exists(path): raise ValueError('Provided file not found: %s' % path)
            path = os.path.abspath(path)
            try:              return path if filter is None else filter(path)
            except Exception: return fail_value
        return config.declare(name, rc_name=rc_name, environ_name=environ_name, filter=_check_path,
                              default_value=None)
    @staticmethod
    def declare_dir(name, 
                    rc_name=None, environ_name=None, filter=None,
                    default_value=None, fail_value=None, use_temp=False):
        '''
        config.declare_dir(...) is equivalent to config.declare_path(...) except that it
          additionally requires that the path be a directory.
        '''
        def _check_dir(path):
            if not os.path.isdir(path):
                raise ValueError('Path exists but is not a directory: %s' % path)
            return path if filter is None else filter(path)
        return config.declare_path(name, rc_name=rc_name, environ_name=environ_name,
                                   filter=_check_dir, default_value=default_value,
                                   fail_value=fail_value)
    @staticmethod
    def declare_file(name, 
                     rc_name=None, environ_name=None, filter=None,
                     default_value=None, fail_value=None):
        '''
        config.declare_file(...) is equivalent to config.declare_path(...) except that it
          additionally requires that the path be a file (i.e., not a directory).
        '''
        def _check_file(path):
            if not os.path.isfile(path):
                raise ValueError('Path exists but is not a file: %s' % path)
            return path if filter is None else filter(path)
        return config.declare_path(name, rc_name=rc_name, environ_name=environ_name,
                                   filter=_check_file, default_value=default_value,
                                   fail_value=fail_value)
    @staticmethod
    def declare_json(name,
                     rc_name=None, environ_name=None, filter=None,
                     default_value=None, fail_value=None):
        '''
        config.declare_json(...) is equivalent to config.declare_file(...) except that it
          additionally requires that the file be a json file and it loads the file then passes the
          json object through the given filter function, if any.
        '''
        def _check_json(path):
            with open(path, 'r') as fl: dat = json.load(fl)
            return dat if filter is None else filter(dat)
        return config.declare_file(name, rc_name=rc_name, environ_name=environ_name,
                                   filter=_check_hson, default_value=default_value,
                                   fail_value=fail_value)
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
                except Exception: pass # it's a string if it can't be json'ed
            # if there's a filter, run it
            if fltfn is not None:
                try: val = fltfn(val)
                except Exception: val = dval # failure--reset to default
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
    

# These are general utility functions that we use for datasets; managing credentials is one of them
def str_to_credentials(s):
    '''
    str_to_credentials(s) yields (key, secret) if the given string is a valid representation of a
      set of credentials. Valid representations include '<key>:<secret>' and '<key>\n<secret>'. All
      initial and trailing whitespace is always stripped from both key and scret. If a newline
      appears in the string, then this character always takes precedense as the separator over a
      colon character. The given string may also be a json object, in which case it is parsed and
      considered valid if it is either a 
    '''
    if not pimms.is_str(s): raise ValueError('str_to_credentials requires a string argument')
    s = s.strip()
    # First try a json object:
    try:
        js = json.loads(s)
        return to_credentials(s)
    except Exception: pass
    # must be '<key>\n<secret>' or '<key>:<secret>'
    dat = s.split('\n')
    if len(dat) == 1: dat = s.split(':')
    if len(dat) != 2: raise ValueError('String "%s" does not appear to be a credentials file' % s)
    return tuple([q.strip() for q in dat])
def load_credentials(flnm):
    '''
    load_credentials(filename) yields the credentials stored in the given file as a tuple
      (key, secret). The file must contain <key>:<secret> on a single line. If the file does not
      contain valid credentials, then an exception is raised. Yields (key, secret).

    Optionally, if the file contains exactly two lines, then the key is taken as the first line and
    the secret is taken as the second line. This can be used if ':' is a valid character in either
    the key or secret for a particular service.

    Note that if the key/secret are more than 8kb in size, this function's behaviour is undefined.
    '''
    flnm = os.path.expanduser(os.path.expandvars(flnm))
    with open(flnm, 'r') as fl: dat = fl.read(1024 * 8)
    # see if its a 2-line file:
    try:              return str_to_credentials(dat)
    except Exception: raise ValueError('File %s does not contain a valid credential string' % flnm)
def to_credentials(arg):
    '''
    to_credentials(arg) converts arg into a pair (key, secret) if arg can be coerced into such a
      pair and otherwise raises an error.
    
    Possible inputs include:
      * A tuple (key, secret)
      * A mapping with the keys 'key' and 'secret'
      * The name of a file that can load credentials via the load_credentials() function
      * A string that separates the key and secret by ':', e.g., 'mykey:mysecret'
      * A string that separates the key and secret by a "\n", e.g., "mykey\nmysecret"
    '''
    if pimms.is_str(arg):
        try: return load_credentials(arg)
        except Exception: pass
        try: return str_to_credentials(arg)
        except Exception:
            raise ValueError('String "%s" is neither a file containing credentials nor a valid'
                             ' credentials string itself.' % arg)
    elif pimms.is_map(arg) and 'key' in arg and 'secret' in arg: return (arg['key'], arg['secret'])
    elif pimms.is_vector(arg, str) and len(arg) == 2: return tuple(arg)
    else: raise ValueError('given argument cannot be coerced to credentials: %s' % arg)
def detect_credentials(config_name, extra_environ=None, filenames=None,
                       aws_profile_name=None, default_value=Ellipsis):
    '''
    detect_credentials(config_name) attempts to locate Amazon S3 Bucket credentials from the given
      configuration item config_name.

    The following optional arguments are accepted:
      * extra_environ (default: None) may specify a string or a tuple (key_name, secret_name) or a
        list of strings or tuples; strings are treated as an additional environment variable that
        should be checked for credentials while tuples are treated as paired varialbes: if both are
        defined, then they are checked as separate holders of a key/secret pair. Note that a list
        of strings is considered a pair of solo environment varialbes while a tuple of strings is
        considered a single (key_name, secret_name) pair.
      * filenames (default: None) may specify a list of filenames that are checked in order for
        credentials.
      * aws_profile_name (default: None) may specify a profile name that appears in the
        ~/.aws/credentials file that will be checked for aws_access_key_id and aws_secret_access_key
        values. The files ~/.amazon/credentials and ~/.credentials are also checked. Note that this
        may be a list of profiles to check.
      * default_value (default: Ellipsis) may specify a value to return when no credentials are
        found; if this value is None, then it is always returned; otherwise, the value is passed
        through to_credentials() and any errors are allowed to propagate out of
        detect_credentials(). If default_value is Ellipsis then an error is simply raised stating
        that no credentials could be found.

    The detect_credentials() function looks at the following locations in the following order,
    assuming that it has been provided with the relevant information:
      * first, if the Neuropythy configuration variable config_name is set via either the npythyrc
        file or the associated environment variable, then it is coerced into credentials;
      * next, if the environment contains both the variables key_name and secret_name (from the 
        optional argument key_secret_environ), then these values are used;
      * next, if the filenames argument is given, then all files it refers to are checked for
        credentials; these files are expanded with both os.expanduser and os.expandvars.
      * finally, if no credentials were detected, an error is raised.
    '''
    # Check the config first:
    if config_name is not None and config[config_name] is not None: return config[config_name]
    # Okay, not found there; check the key/secret environment variables
    if   extra_environ is None: extra_environ = []
    elif pimms.is_str(extra_environ): extra_environ = [extra_environ]
    elif pimms.is_vector(extra_environ):
        if pimms.is_vector(extra_environ, str):
            if len(extra_environ) == 2 and isinstance(extra_environ, _tuple_type):
                extra_environ = [extra_environ]
    elif not pimms.is_matrix(extra_environ, str):
        raise ValueError('extra_environ must be a string, tuple of strings, or list of these')
    for ee in extra_environ:
        if pimms.is_str(ee):
            if ee in os.environ:
                try:              return to_credentials(q)
                except Exception: pass
        elif pimms.is_vector(ee, str) and len(ee) == 2:
            if ee[0] in os.environ and ee[1] in os.environ:
                (k,s) = [os.environ[q] for q in ee]
                if len(k) > 0 and len(s) > 0: continue
                return (k,s)
        else: raise ValueError('cannot interpret extra_environ argument: %s' % ee)
    # Okay, next we check the filenames
    if filenames is None: filenames = []
    elif pimms.is_str(filenames): filenames = [filenames]
    for flnm in filenames:
        flnm = os.expanduser(os.expandvars(flnm))
        if os.path.isfile(flnm):
            try:              return to_credentials(flnm)
            except Exception: pass
    # okay... let's check the AWS credentials file, if it exists
    if pimms.is_str(aws_profile_name): aws_profile_name = [aws_profile_name]
    elif aws_profile_name is None or len(aws_profile_name) == 0: aws_profile_name = None
    elif not pimms.is_vector(aws_profile_name, str):
        raise ValueError('Invalid aws_profile_name value: %s' % aws_profile_name)
    if aws_profile_name is not None:
        try:
            cc = confparse.ConfigParser()
            cc.read([os.expanduser(os.path.join('~', '.aws', 'credentials')),
                     os.expanduser(os.path.join('~', '.amazon', 'credentials')),
                     os.expanduser(os.path.join('~', '.credentials'))])
            for awsprof in aws_profile_names:
                try:
                    aws_access_key_id     = cc.get(awsprof, 'aws_access_key_id')
                    aws_secret_access_key = cc.get(awsprof, 'aws_secret_access_key')
                    return (aws_access_key_id, aws_secret_access_key)
                except Exception: pass
        except Exception: pass
    # no match!
    if default_value is None:
        return None
    elif default_value is Ellipsis:
        if config_name is None: raise ValueError('No valid credentials were detected')
        else: raise ValueError('No valid credentials (%s) were detected' % config_name)
    else: return to_credentials(default_value)
