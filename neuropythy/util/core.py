####################################################################################################
# neuropythy/util/core.py
# This file implements the command-line tools that are available as part of neuropythy as well as
# a number of other random utilities.

import numpy                        as np
import scipy.sparse                 as sps
import pyrsistent                   as pyr
import nibabel                      as nib
import nibabel.freesurfer.mghformat as fsmgh
import types, inspect, pimms

@pimms.immutable
class CommandLineParser(object):
    '''
    CommandLineParser(instructions) yields a command line parser object, which acts as a function,
    that is capable of parsing command line options into a dictionary of opts and a list of args,
    as defined by the list of instructions.
    The instructions should be a list of lists (or tuples), each of which should have three or four
    entries: [character, word, entry, default] where the default is optional.
    The character and word are the -x and --xxx versions of the argument; the entry is the name of
    the entry in the opts dictionary that is yielded for that command line option, and the default
    is the value inserted into the dictionary if the command line option is not found; if no
    default value is given, then the entry will appear in the dictionary only if it appears in the
    command line arguments.
    If the default value given is either True or False, then the option is understood to be a flag;
    i.e., the option does not take an argument (and single letter flags can appear together, such
    as -it instead of -i -t), and the appearance of the flag toggles the default to the opposite
    value.
    
    Example:
      parser = CommandLineParser(
        [('a', 'flag-a', 'aval', False),
         ('b', 'flag-b', 'bval', True),
         ('c', 'flag-c', 'cval', True),
         ('d', 'flag-d', 'dval', False),
         ('e', 'opt-e',  'eval', None),
         ('f', 'opt-f',  'fval', None),
         ('g', 'opt-g',  'gval'),
         ('h', 'opt-h',  'hval')])
      cmd_line = ['-ab', 'arg1', '--flag-d', '-etestE', '--opt-f=123', '-htestH', 'arg2']
      parser(cmd_line) == parser(*cmd_line)  # identical calls
      parser(cmd_line)
      # ==> (['arg1', 'arg2'],
      # ==>  {'a':True, 'b':False, 'c':True, 'd':True, 'e':'testE', 'f':'123', 'h':'testH'})
    '''

    def __init__(self, instructions):
        'See help(CommandLineParser).'
        wflags = {}
        cflags = {}
        wargs = {}
        cargs = {}
        defaults = {}
        for row in instructions:
            if not hasattr(row, '__iter__') or len(row) < 3 or len(row) > 4 or \
               any(x is not None and not isinstance(x, basestring) for x in row[:3]):
                raise ValueError('Invalid instruction row: %s ' % row)
            (c, w, var, dflt) = row if len(row) == 4 else (list(row) + [None])
            defaults[var] = dflt
            if dflt is True or dflt is False:
                if c is not None: cflags[c] = var
                if w is not None: wflags[w] = var
            else:
                if c is not None: cargs[c] = var
                if w is not None: wargs[w] = var
        self.default_values = pyr.pmap(defaults)
        self.flag_words = pyr.pmap(wflags)
        self.flag_characters = pyr.pmap(cflags)
        self.option_words = pyr.pmap(wargs)
        self.option_characters = pyr.pmap(cargs)
    @pimms.param
    def default_values(dv):
        '''
        clp.default_values yields the persistent map of default values for the given command-line
          parser clp.
        '''
        if pimms.is_pmap(dv): return dv
        elif pimms.is_map(dv): return pyr.pmap(dv)
        else: raise ValueError('default_value must be a mapping')
    @pimms.param
    def flag_words(u):
        '''
        clp.flag_words yields the persistent map of optional flag words recognized by the given
          command-line parser clp.
        '''
        if pimms.is_pmap(u): return u
        elif pimms.is_map(u): return pyr.pmap(u)
        else: raise ValueError('flag_words must be a mapping')
    @pimms.param
    def flag_characters(u):
        '''
        clp.flag_characters yields the persistent map of the flag characters recognized by the given
          command-line parser clp.
        '''
        if pimms.is_pmap(u): return u
        elif pimms.is_map(u): return pyr.pmap(u)
        else: raise ValueError('flag_characters must be a mapping')
    @pimms.param
    def option_words(u):
        '''
        clp.option_words yields the persistent map of optional words recognized by the given
          command-line parser clp.
        '''
        if pimms.is_pmap(u): return u
        elif pimms.is_map(u): return pyr.pmap(u)
        else: raise ValueError('option_words must be a mapping')
    @pimms.param
    def option_characters(u):
        '''
        clp.option_characters yields the persistent map of optional characters recognized by the
          given command-line parser clp.
        '''
        if pimms.is_pmap(u): return u
        elif pimms.is_map(u): return pyr.pmap(u)
        else: raise ValueError('option_characters must be a mapping')
        
    def __call__(self, *args):
        if len(args) > 0 and not isinstance(args[0], basestring) and \
           isinstance(args[0], (types.ListType, types.TupleType)):
            args = list(args)
            return self.__call__(*(list(args[0]) + args[1:]))
        parse_state = None
        more_opts = True
        remaining_args = []
        opts = dict(self.default_values)
        wflags = self.flag_words
        cflags = self.flag_characters
        wargs  = self.option_words
        cargs  = self.option_characters
        dflts  = self.default_values
        for arg in args:
            larg = arg.lower()
            if parse_state is not None:
                opts[parse_state] = arg
                parse_state = None
            else:
                if arg == '': pass
                elif more_opts and arg[0] == '-':
                    if len(arg) == 1:
                        remaining_args.append(arg)
                    elif arg[1] == '-':
                        trimmed = arg[2:]
                        if trimmed == '':     more_opts = False
                        if trimmed in wflags: opts[wflags[trimmed]] = not dflts[wflags[trimmed]]
                        else:
                            parts = trimmed.split('=')
                            if len(parts) == 1:
                                if trimmed not in wargs:
                                    raise ValueError('Unrecognized flag/option: %s' % trimmed)
                                # the next argument specifies this one
                                parse_state = wargs[trimmed]
                            else:
                                k = parts[0]
                                if k not in wargs:
                                    raise ValueError('Unrecognized option: %s' % k)
                                opts[wargs[k]] = trimmed[(len(k) + 1):]
                    else:
                        trimmed = arg[1:]
                        for (k,c) in enumerate(trimmed):
                            if c in cflags: opts[cflags[c]] = not dflts[cflags[c]]
                            elif c in cargs:
                                remainder = trimmed[(k+1):]
                                if len(remainder) > 0: opts[cargs[c]] = remainder
                                else:
                                    # next argument...
                                    parse_state = cargs[c]
                                break
                else:
                    remaining_args.append(arg)
        if parse_state is not None:
            raise ValueError('Ran out of arguments while awaiting value for %s' % parse_state)
        # that's done; all args are parsed
        return (remaining_args, opts)

@pimms.immutable
class ObjectWithMetaData(object):
    '''
    ObjectWithMetaData is a class that stores a few useful utilities and the param meta_data, all of
    which assist in tracking a persistent map of meta-data with an object.
    '''
    def __init__(self, meta_data=None):
        if meta_data is None:
            self.meta_data = pyr.m()
        else:
            self.meta_data = meta_data
    @pimms.option(pyr.m())
    def meta_data(md):
        '''
        obj.meta_data is a persistent map of meta-data provided to the given object, obj.
        '''
        if md is None: return pyr.m()
        return md if pimms.is_pmap(md) else pyr.pmap(md)
    def meta(self, name):
        '''
        obj.meta(x) is equivalent to obj.meta_data.get(name, None).
        '''
        return self.meta_data.get(name, None)
    def with_meta(self, *args, **kwargs):
        '''
        obj.with_meta(...) collapses the given arguments with pimms.merge into the object's current
        meta_data map and yields a new object with the new meta-data.
        '''
        md = pimms.merge(self.meta_data, *(args + (kwargs,)))
        if md is self.meta_data: return self
        else: return self.copy(meta_data=md)
    def wout_meta(self, *args, **kwargs):
        '''
        obj.wout_meta(...) removes the given arguments (keys) from the object's current meta_data
        map and yields a new object with the new meta-data.
        '''
        md = self.meta_data
        for a in args:
            if pimms.is_vector(a):
                for u in a:
                    md = md.discard(u)
            else:
                md = md.discard(a)
        return self if md is self.meta_data else self.copy(meta_data=md)

def to_affine(aff, dims=None):
    '''
    to_affine(None) yields None.
    to_affine(data) yields an affine transformation matrix equivalent to that given in data. Such a
      matrix may be specified either as (matrix, offset_vector), as an (n+1)x(n+1) matrix, or, as an
      n x (n+1) matrix.
    to_affine(data, dims) additionally requires that the dimensionality of the data be dims; meaning
      that the returned matrix will be of size (dims+1) x (dims+1).
    '''
    if aff is None: return None
    if isinstance(aff, types.TupleType):
        # allowed to be (mtx, offset)
        if (len(aff) != 2                       or
            not pimms.is_matrix(aff[0], 'real') or
            not pimms.is_vector(aff[1], 'real')):
            raise ValueError('affine transforms must be matrices or (mtx,offset) tuples')
        mtx = np.asarray(aff[0])
        off = np.asarray(aff[1])
        if dims is not None:
            if mtx.shape[0] != dims or mtx.shape[1] != dims:
                raise ValueError('%dD affine matrix must be %d x %d' % (dims,dims,dims))
            if off.shape[0] != dims:
                raise ValueError('%dD affine offset must have length %d' % (dims,dims))
        else:
            dims = off.shape[0]
            if mtx.shape[0] != dims or mtx.shape[1] != dims:
                raise ValueError('with offset size=%d, matrix must be %d x %d' % (dims,dims,dims))
        aff = np.zeros((dims+1,dims+1), dtype=np.float)
        aff[dims,dims] = 1
        aff[0:dims,0:dims] = mtx
        aff[0:dims,dims] = off
        return pimms.imm_array(aff)
    if not pimms.is_matrix(aff, 'real'):
        raise ValueError('affine transforms must be matrices or (mtx, offset) tuples')
    aff = np.asarray(aff)
    if dims is None:
        dims = aff.shape[1] - 1
    if aff.shape[0] == dims:
        lastrow = np.zeros((1,dims+1))
        lastrow[0,-1] = 1
        aff = np.concatenate((aff, lastrow))
    if aff.shape[1] != dims+1 or aff.shape[0] != dims+1:
        arg = (dims, dims,dims+1, dims+1,dims+1)
        raise ValueError('%dD affine matrix must be %dx%d or %dx%d' % args)
    return aff
        
def simplex_summation_matrix(simplices, weight=None, inverse=False):
    '''
    simplex_summation_matrix(mtx) yields a scipy sparse array matrix that, when dotted with a
      column vector of length m (where m is the number of simplices described in the simplex matrix,
      mtx), yields a vector of length n (where n is the number of vertices in the simplex mesh); the
      returned vetor is the sum over each vertex, of the faces to which it belongs.

    The matrix mtx must be oriented such that the first dimension (rows) corresponds to the vertices
    of the simplices and the second dimension (columns) corresponds to simplices themselves.

    The optional argument weight may specify a weight for each face, in which case the summation is
    a weighted sum instead of a flat sum.

    The optional argument inverse=True may be given to indicate that the inverse summation matrix
    (summation of the vertices onto the simplices) should be returned.
    '''
    simplices = np.asarray(simplices)
    n = np.max(simplices) + 1
    (d,m) = simplices.shape
    rng = range(m)
    if inverse:
        if weight is None: f = sps.csr_matrix
        else:
            nrng = range(n)
            ww = sps.csr_matrix((weight, (nrng, nrng)), shape=(n,n), dtype=np.float)
            f = lambda *args,**kwargs: ww.dot(sps.csc_matrix(*args,**kwargs))
        s = f((np.ones(d*m, dtype=np.int),
               (np.concatenate([rng for _ in range(d)]), np.concatenate(simplices))),
              shape=(m,n),
              dtype=np.int)
    else:
        s = sps.csr_matrix(
            (np.ones(d*m, dtype=np.int),
             (np.concatenate(simplices), np.concatenate([rng for _ in range(d)]))),
            shape=(n,m),
            dtype=np.int)
        if weight is not None:
            s = s.dot(sps.csc_matrix((weight, (rng, rng)), shape=(m,m), dtype=np.float))
    return s
def simplex_averaging_matrix(simplices, weight=None, inverse=False):
    '''
    Simplex_averaging_matrix(mtx) is equivalent to simplex_simmation_matrix, except that each row of
      the matrix is subsequently normalized such that all rows sum to 1.
    
    The optional argument inverse=True may be passed to indicate that the inverse averaging matrix
    (of vertices onto simplices) should be returned.
    '''
    m = simplex_summation_matrix(simplices, weight=weight, inverse=inverse)
    rs = np.asarray(m.sum(axis=1), dtype=np.float)[:,0]
    invrs = zinv(rs)
    rng = range(m.shape[0])
    diag = sps.csr_matrix((invrs, (rng, rng)), dtype=np.float)
    return diag.dot(sps.csc_matrix(m, dtype=np.float))

def zinv(x):
    '''
    zinv(x) yields 1/x if x is not close to 0 and 0 otherwise. Automatically threads over arrays.
    '''
    z = np.isclose(x, 0)
    return np.logical_not(z) / (x + z)
def zdiv(a, b, null=0):
    '''
    zdiv(a,b) yields a/b if b is not close to 0 and 0 if b is close to 0; automatically threads over
      lists.
    '''
    z = np.isclose(b, 0)
    iz = np.logical_not(z)
    res = iz*a / (b + z)
    res[z] = null
    return res
