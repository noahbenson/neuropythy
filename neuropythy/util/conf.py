# -*- coding: utf-8 -*-
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
    """Loads a JSON-format file with the given filename or raises an error.

    `loadrc(filename)` returns a dict object decoded from the given `filename`,
    which must represent a JSON-format file. If the filename does not exist or
    does not contain a valid JSON dict, then an error is raised.

    Parameters
    ----------
    filename : str
        The name of the file to be loaded; may include variable and user
        expansion codes.

    Returns
    -------
    dict
        A dictionary of the JSON contents of the file.

    Raises
    ------
    ValueError
        If the given `filename` does not exist or does not contain a JSON dict.
    """
    filename = os.path.expanduser(os.path.expandvars(filename))
    if not os.path.isfile(filename): raise ValueError('Filename %s does not exist' % filename)
    with open(filename, 'r') as fl:
        dat = json.load(fl)
    try: dat = dict(dat)
    except Exception: dat = None
    if dat is None: raise ValueError('Given file %s does not contain a dictionary' % filename)
    return dat
def saverc(filename, dat, overwrite=False):
    """Saves the given configuration object to a file in JSON format.

    `saverc(filename, dat)` saves the given configuration dictionary `dat` to
    the given `filename` in JSON format. If `dat` is not a dictionary or if
    `filename` already exists or cannot be created, an error is raised. This
    funciton does not create directories.

    Parameters
    ----------
    filename : str
        The path to the file that should be saved; may contain user or variable
        expansion codes.
    dat : JSON-compatible dict
        The configuration dictionary that is to be saved.
    overwrite : boolean, optional
        Whether to overwrite the file if it already exists (default: `False`).

    Returns
    -------
    str
        The full path of the file that was saved.

    Raises
    ------
    ValueError
        If the given `filename` already exists and `overwrite` is `False` or if
        the `dat` object is not a JSON-compatible dictionary
    """
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
    """Configuration dictionary class for Neuropythy.

    `neuropythy.util.conf.config` is a class that manages configuration items
    for neuropythy and the interfaces for those items.  This class reads in the
    user's neuropythy-rc file at startup, which by default is in the user's home
    directory named `"~/.npythyrc"` (though it may be altered by setting the
    environment variable `NPYTHYRC`). Environment variables that are associated
    with configurable variables always override the values in the RC file, and
    any direct set action overrides any previous value.

    To declare a configurable variable in Neuropythy, you must use config.declare().
    """
    _rc = None
    @staticmethod
    def rc():
        """Returns the data imported from the Neuropythy RC file, if any.

        Returns
        -------
        dict or None
            A dictionary object of the loaded RC data or `None` if nothing could
            be loaded.
        """
        if config._rc is None:
            # First: We check to see if we have been given a custom nptyhrc file:
            npythyrc_path = os.path.expanduser('~/.npythyrc')
            if 'NPYTHYRC' in os.environ:
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
    def declare(name, rc_name=None, environ_name=None, filter=None, merge=None, default_value=None):
        """Registers a neuropythy configuration variable with the given name.

        `config.declare(name)` registers a configurable variable with the given
         name to the neuropythy configuration system. This allows the variable
         to be looked up in the neuropythy RC-file and the neuropythy
         environment variables.

        By default, the variable will be assumed to have the identical name in
        the neuropythy RC-file and the environment variable (`'NPYTHY_' +
        name.upper()`) is searched for in the environment. The environment
        variable will always overwrite the RC-file value if both are
        provided. Note that all inputs from the environment or the RC-file are
        parsed as JSON inputs prior to being put in the config object itself.

        Parameters
        ----------
        name : str
            The name for the configuration variable that should be used to look
            up its value in the `config` dict.
        rc_name : str or None, optional
            The name that should be used in the RC file for the variable. The
            default value (`None`) indicates that `name.lower()` should be
            used.
        environ_name : str or None, optional
            The name of the environment variable that should represent the
            configuration variable. The default value (`None`) indicates that the 
            environment name should be `'NPYTHY_' + name.upper()`.
        filter : function or None
            A function `f` that is passed the provided or loaded value of the
            configuration variable whenever the variable is changed; the new
            value that is then applied to the variable is `f(x)` instead of `x`.
            The default value (`None`) indicates that the filter is `f(x) = x`.
        merge : function or None
            A function that specifies how the environment variables should be
            merged with the value found in the RC file if both are found. If the
            `merge` parameter is `None` (the default) or `False`, then the
            environment variable value always overwrites the RC-file value;
            otherwise, `merge` must be a function `f` such that `f(rc_value,
            environ_value)` returns the value that should be applied to the
            configuration variable.
        default_value : object
            The default value that the configuration item should take if not
            provided in either the RC-file or the environment.

        Raises
        ------
        ValueError
            If multiple configuration items with the same name are declared.
        """
        if rc_name is None: rc_name = name.lower()
        if environ_name is None: environ_name = 'NPYTHY_' + name.upper()
        # okay, see if the name exists
        if name in config._vars: raise ValueError('Multiple config items declared for %s' % name)
        if merge is False: merge = None
        config._vars[name] = (rc_name, environ_name, filter, default_value, merge)
        return True
    @staticmethod
    def declare_credentials(name, rc_name=None, environ_name=None, merge=None, default_value=None,
                            extra_environ=None, filenames=None, aws_profile_name=None):
        """Declares a config variable that loads credentials from a file.

        `config.declare_credentials(name)` registers a configurable variable
        with the given name to the neuropythy configuration system (see also
        `config.declare()`). The variable is specifically created as a
        credentials-type variable such that if the config variable is not set, a
        call to `detect_credentials()` is used to attempt to find the
        credentials; this effectively allows for additional filenames and
        environment variables to be searched while ensuring that strings, files,
        and json representations all get coerced to (key, secret) if possible.

        Note that if you set default_value to something that is neither `None`
        nor a valid credentials representation, then the config item will still
        end up as `None`; this is because the default value must fall through
        the `to_credentials()` filter function.

        Parameters
        ----------
        name : str
            The name for the configuration variable that should be used to look
            up its value in the `config` dict. See also `config.declare()`.
        rc_name : str or None, optional
            The name that should be used in the RC file for the variable. The
            default value (`None`) indicates that `name.lower()` should be
            used. See also `config.declare()`.
        environ_name : str or None, optional
            The name of the environment variable that should represent the
            configuration variable. The default value (`None`) indicates that the 
            environment name should be `'NPYTHY_' + name.upper()`.  See also
            `config.declare()`.
        merge : function or None, optional
            A function that specifies how the environment variables should be
            merged with the value found in the RC file if both are found. If the
            `merge` parameter is `None` (the default) or `False`, then the
            environment variable value always overwrites the RC-file value;
            otherwise, `merge` must be a function `f` such that `f(rc_value,
            environ_value)` returns the value that should be applied to the
            configuration variable. See also `config.declare()`.
        default_value : object, optional
            The default value that the configuration item should take if not
            provided in either the RC-file or the environment. See also
            `config.declare()`.
        extra_environ : str or iterable or None, optional
            Specifies how to read the credentials from the environment. The
            `extra_environ` value may be one of the following:
              1. `None`, indicating no extra environment instructions;
              2. a string
              3. a tuple `(key_env_name, secret_env_name)`, indicating that the
                 key and the secret should be extracted from a pair of
                 environment variables;
              4. a list whose entries fall into category 2 or 3 above,
                 indicating that each element of the list is one of many
                 alternative valid environment variables in which to find the
                 credentials (earlier entries are given precedence).
        filenames : str or iterable of str or None, optional
            A filename or list of filenames that are checked in order for valid
            credentials. The default value `None` is equivalent to `[]`.
        aws_profile_name : str or iterable of str or None, optional
            Optionally may specify a profile name or list of profile names that
            may appear in the user's `~/.aws/credentials` file that and will be
            checked for `aws_access_key_id` and `aws_secret_access_key`
            values. The files `~/.amazon/credentials` and `~/.credentials` are
            also checked. Note that this may be a list of profiles to check.
            The default value of `None` indicates that no AWS files should be
            checked.

        Raises
        ------
        ValueError
            If multiple configuration items with the same name are declared or
            if the `default_value` is `Ellipsis`.
        """
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
                              merge=merge,
                              default_value=None)
    @staticmethod
    def declare_path(name, 
                     rc_name=None, environ_name=None, filter=None, default_value=None, merge=None,
                     fail_value=None):
        """Declares a configuration variable that must be a filesystem path.

        `config.declare_path(...)` is equivalent to `config.declare(...)` except
        that it requires in addition that the value found for the config item be
        an existing file or directory on the local filesystem. Only when this is
        true is the value passed to any provided filter function.

        This function (similar to config.declare_file and config.declare_dir) is
        generally expected to return some data--the contents of the path--not
        the path-name itself. For this reason the handling of the default_value
        argument is expanded relative to that of `config.declare()`.  If no
        value is provided or if all provided values are either not paths or fail
        the filter, then the `default_value` is checked (as a path); if this
        also fails then the `fail_value` is used as the config value. By default
        this is `None`, but note that by using a filter function that loads the
        provided path (which will be guaranteed to exist) and returnss those
        loaded data along with a `fail_value` that resembles 'default' data, one
        can effectively guarantee that data is either loaded from a source on
        the filesystem or is in a coherent default state.

        Parameters
        ----------
        name : str
            The name for the configuration variable that should be used to look
            up its value in the `config` dict. See also `config.declare()`.
        rc_name : str or None, optional
            The name that should be used in the RC file for the variable. The
            default value (`None`) indicates that `name.lower()` should be
            used. See also `config.declare()`.
        environ_name : str or None, optional
            The name of the environment variable that should represent the
            configuration variable. The default value (`None`) indicates that the 
            environment name should be `'NPYTHY_' + name.upper()`.  See also
            `config.declare()`.
        merge : function or None, optional
            A function that specifies how the environment variables should be
            merged with the value found in the RC file if both are found. If the
            `merge` parameter is `None` (the default) or `False`, then the
            environment variable value always overwrites the RC-file value;
            otherwise, `merge` must be a function `f` such that `f(rc_value,
            environ_value)` returns the value that should be applied to the
            configuration variable. See also `config.declare()`.
        default_value : object, optional
            The default value that the configuration item should take if not
            provided in either the RC-file or the environment. Unlike
            `fail_value` this value should contain a filename that is the
            default filename. See also `config.declare()`.
        fail_value : object, optional
            The value that should be used as the actual configuration value if
            no files are loaded and `default_value` is either not an existing
            file or fails to load properly.

        Raises
        ------
        ValueError
            If multiple configuration items with the same name are declared or
            if the `default_value` is `Ellipsis`.

        See also `config.declare_dir()` and `config.declare_file()`.
        """
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
                              default_value=None, merge=merge)
    @staticmethod
    def declare_dir(name, 
                    rc_name=None, environ_name=None, filter=None, merge=None,
                    default_value=None, fail_value=None, use_temp=False):
        """Declares a configuration variable that must contain a directory name.

        `config.declare_dir(...)` is equivalent to `config.declare_path(...)`
        except that it additionally requires that the path be a directory.
        """
        def _check_dir(path):
            if not os.path.isdir(path):
                raise ValueError('Path exists but is not a directory: %s' % path)
            return path if filter is None else filter(path)
        return config.declare_path(name, rc_name=rc_name, environ_name=environ_name,
                                   filter=_check_dir, default_value=default_value, merge=merge,
                                   fail_value=fail_value)
    @staticmethod
    def declare_file(name, 
                     rc_name=None, environ_name=None, filter=None, merg=None,
                     default_value=None, fail_value=None):
        """Declares a configuration variable that must contain a filename.

        `config.declare_file(...)` is equivalent to `config.declare_path(...)`
        except that it additionally requires that the path be a file (i.e., not
        a directory).
        """
        def _check_file(path):
            if not os.path.isfile(path):
                raise ValueError('Path exists but is not a file: %s' % path)
            return path if filter is None else filter(path)
        return config.declare_path(name, rc_name=rc_name, environ_name=environ_name,
                                   filter=_check_file, default_value=default_value, merge=merge,
                                   fail_value=fail_value)
    @staticmethod
    def declare_json(name,
                     rc_name=None, environ_name=None, filter=None, merge=None,
                     default_value=None, fail_value=None):
        """Declares a configuration variable that must contain a JSON filename.
        
        `config.declare_json(...)` is equivalent to `config.declare_file(...)`
        except that it additionally requires that the file be a json file and it
        loads the file then passes the JSON object through the given `filter`
        function, if any.
        """
        def _check_json(path):
            with open(path, 'r') as fl: dat = json.load(fl)
            return dat if filter is None else filter(dat)
        return config.declare_file(name, rc_name=rc_name, environ_name=environ_name,
                                   filter=_check_hson, default_value=default_value, merge=merge,
                                   fail_value=fail_value)
    _vals = {}
    @staticmethod
    def _getitem(self, name):
        if name not in config._vars: raise KeyError(name)
        if name not in config._vals:
            (rcname, envname, fltfn, dval, merge) = config._vars[name]
            val = dval
            rcdat = config.rc()
            # see where it's defined:
            if envname in os.environ:
                val = os.environ[envname]
                try: val = json.loads(val)
                except Exception: pass # it's a string if it can't be json'ed
                # it could be in both env and rc: use merge if needed
                if merge is not None and rcname in rcdat: val = merge(rcdat[rcname], val)
            elif rcname in rcdat: val = rcdat[rcname]
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
        (rcname, envname, fltfn, dval, merge) = config._vars[name]
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
    """Convert a string into a credentials tuple `(key, secret)`.

    `str_to_credentials(s)` returns `(key, secret)` if the given string `s` is a
    valid representation of a set of credentials. Valid representations include
    `'<key>:<secret>'` and `'<key>\n<secret>'`. All initial and trailing
    whitespace is always stripped from both key and scret. If a newline appears
    in the string, then this character always takes precedense as the separator
    over a colon character. The given string may also be a JSON string, in which
    case it is parsed and considered valid if its JSON representation can be
    converted into a credentials object via the `to_credentials()` function.

    Parameters
    ----------
    s : str
        A string that should be converted into a credentials tuple.

    Returns
    -------
        A tuple `(key, secret)` containing two strings.

    Raises
    ------
    ValueError
        If `s` is not a string or if it cannot be parsed as credentials.
    """
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
def load_credentials(filename):
    """Loads a credentials tuple from a file.

    `load_credentials(filename)` returns the credentials stored in the given
    file as a tuple `(key, secret)`. The file must contain a string such as
    `<key>:<secret>` that can be parsed by the `str_to_credentials()`
    function. If the file does not contain valid credentials, then an exception
    is raised.

    Note that if the key/secret are more than 8kb in size, this function's
    behaviour is undefined.

    Parameters
    ----------
    filename : str
        The path of a file on the local filesystem, which may contain user or
        variable expansion codes.

    Returns
    -------
    tuple of str
        A tuple of the `(key, secret)` strings that were parsed from the file
        with the given `filename`.

    Raises
    ------
    ValueError
        When the given file does not contain a valid set of credentials.
    FileNotFountError
        If the given file does not exist.
    """
    flnm = os.path.expanduser(os.path.expandvars(filename))
    with open(flnm, 'r') as fl: dat = fl.read(1024 * 8)
    # see if its a 2-line file:
    try:              return str_to_credentials(dat)
    except Exception: pass
    raise ValueError('File %s does not contain a valid credential string' % flnm)
def to_credentials(arg):
    """Converts an object into a credentials tuple.

    `to_credentials(arg)` converts the given `arg` into a tuple `(key, secret)`
    assuming that `arg` can be reinterpreted as a credentials tuple and
    otherwise raises an error.

    The following arguments can be converted successfully into a credentials
    tuple:
      1. A tuple of strings `(key, secret)` (already a valid credentials tuple).
      2. A dict with the keys `'key'` and `'secret'`.
      3. The name of a file that can be passed to the `load_credentials()`
         function successfully.
      4. A string that separates the key and secret by `':'`, e.g.,
         `'mykey:mysecret'`.
      5. A string that separates the key and secret by a `"\n"`, e.g., 
         `"mykey\nmysecret"`.
    
    Parameters
    ----------
    arg : object
        An object to be converted into a credentials tuple.

    Returns
    ------
    tuple of str
        A tuple of the `(key, secret)` strings that were parsed from the file
        with the given `filename`.

    Raises
    ------
    ValueError
        When the given argument cannot be reinterpreted as a valid set of
        credentials.
    """
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
    """Attempts to locate Amazon S3 Bucket credentials from a config variaable.

    `detect_credentials(config_name)` attempts to locate Amazon S3 Bucket
    credentials from the given configuration item `config_name`.

   The `detect_credentials()` function looks at the following locations in the
   following order, assuming that it has been provided with the relevant
   information:
     * first, if the Neuropythy configuration variable `config_name` is set via
       either the Neuropythy RC file or the associated environment variable,
       then it is coerced into credentials;
     * next, if the environment contains the credentials, as specified in the
       `extra_environ` parameter, these are used;
     * next, if the filenames argument is given, then all files it refers to are
       checked for credentials; these files are expanded with both
       `os.expanduser` and `os.expandvars` and are loaded via the
       `load_credentials()` function.
     * finally, if no credentials were detected, an error is raised.
 
    Parameters
    ----------
    extra_environ : str or iterable or None, optional
        Specifies how to read the credentials from the environment. The
        `extra_environ` value may be one of the following:
          1. `None`, indicating no extra environment instructions;
          2. a string
          3. a tuple `(key_env_name, secret_env_name)`, indicating that the key
             and the secret should be extracted from a pair of environment
             variables;
          4. a list whose entries fall into category 2 or 3 above, indicating
             that each element of the list is one of many alternative valid
             environment variables in which to find the credentials (earlier
             entries are given precedence).
    filenames : str or iterable of str or None, optional
        A filename or list of filenames that are checked in order for valid
        credentials. The default value `None` is equivalent to `[]`.
    aws_profile_name : str or iterable of str or None, optional
        Optionally may specify a profile name or list of profile names that may
        appear in the user's `~/.aws/credentials` file that and will be checked
        for `aws_access_key_id` and `aws_secret_access_key` values. The files
        `~/.amazon/credentials` and `~/.credentials` are also checked. Note that
        this may be a list of profiles to check.  The default value of `None`
        indicates that no AWS files should be checked.
    default_value : object, optional
        The default value that the configuration item should take if not provided
        in either the RC-file or the environment. See also `config.declare()`.

    Returns
    ------
    tuple of str
        A tuple of the `(key, secret)` strings that were parsed from the file
        with the given `filename`.

    Raises
    ------
    ValueError
        When valid credentials cannot be found.
    """
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
