####################################################################################################
# neuropythy/immutable/__init__.py
# Simple class to inherit for immutable lazily-loading classes.
# By Noah C. Benson

import itertools
import collections
from pysistence import make_dict
import pysistence

class Immutable:
    '''
    The Immutable class can be overloaded by any class that wishes to be an immutable lazily-loading
    class---Immutable facilitates that. It does so by overloading the standard dictionary functions
    and accepting a list of lazy loading instructions in its constructor. The constructor for the
    Immutable class expects three arguments plus any number of optional key-val assignments. The 
    optional key-val assignments are placed into the settable member 'options'. The first three 
    arguments are: Immutable(settable_vals, const_vals, lazy_vals).
      * settable_vals must be a dict of settable value names (strings) mapped to a function which
        takes as an argument the self object and the value being assigned and returns the value
        that should actually be assigned.
      * const_vals must be a dict of constant value names (strings) mapped to constant values; these
        will be constant member values of the object.
      * lazy_vals must be a dict of lazy value names (strings) mapped to a 2-tuple of (1) a tuple of
        argument names and (2) a function that accepts these arguments and yields the lazy value;
        the argument names may be any settable, const, or lazy value name.
    '''

    # Calculate lazy dependencies between values for this object
    def __lazy_deps(self):
        # we build up this dependency tree:
        dep_tree = dict()
        for (sname, sdata) in self._settable_vals.iteritems():
            deps = {sname}
            # sweep through lazy vals, eliminating those that depend on name; do this until none are
            # removed on a sweep
            go = True
            while go:
                tmp = [lnm for (lnm, (ldeps, lfn)) in self._lazy_vals.iteritems()
                           if lnm not in deps and any(dep in deps for dep in ldeps)]
                go = len(tmp) > 0
                deps.update(tmp)
            dep_tree[sname] = (deps - {sname})
        return dep_tree

    # This function will clear the lazily-evaluated members when a given value is changed
    def __update_values(self, name):
        for nm in self._lazy_deps[name]:
            if nm in self.__dict__:
                del self.__dict__[nm]

    # This is the most important function, given the encapsulation of this class:
    def __setattr__(self, name, val):
        if name in self._settable_vals:
            fn = self._settable_vals[name]
            self.__dict__[name] = fn(self, val)
            self.__update_values(name)
        elif name in self._lazy_vals:
            raise ValueError('The member %s is a lazy value and cannot be set' % name)
        else:
            raise ValueError('Unrecognized Immutable member: %s' % name)

    # The getattr method makes sure that lazy members are computed when requested
    def __getattr__(self, name):
        if name[0] == '_':
            return self.__dict__[name]
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self._lazy_vals:
            (deps, fn) = self._lazy_vals[name]
            tmp = fn(*map(lambda x: getattr(self, x), deps))
            self.__dict__[name] = tmp
            return tmp
        elif name in self._const_vals:
            return self._const_vals[name];
        else:
            raise ValueError('Unrecognized member of Immutable: %s' % name)

    def __init__(self, settable_vals, const_vals, lazy_vals, **opts):
        self.__dict__['_settable_vals'] = make_dict(settable_vals).using(
            options=lambda s,o: make_dict(o))
        self.__dict__['_const_vals'] = make_dict(const_vals)
        self.__dict__['_lazy_vals'] = lazy_vals
        self.__dict__['_lazy_deps'] = self.__lazy_deps()
        self.options = opts

    def __repr__(self):
        return 'Immutable(<' + '>, <'.join([k for k,v in self._settable_vals.iteritems()]) + '>)'
