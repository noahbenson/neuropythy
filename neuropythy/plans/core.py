####################################################################################################
# neuropythy/plans/core.py
# Core utilities for PIMMS Calculation Plan algorithms/workflows.
# By Noah C. Benson

import numpy as np
import pyrsistent as pyr
import os, sys, gzip, six, types, pimms
from ..util import (is_tuple, is_list)
from ..math import (pytorch, totensor, astensor)
from .. import math


def limit_param(param, min=-1, max=1):
    '''
    limit_param(u) yields the limited parameter x, which must always be in the
      range -1 to 1, from its unlimited value u, which may take on any real
      value.
    limit_param(u, min, max) uses the given minimum and maximum parameters in
      place of -1 and 1.

    The opposite of x = limit_param(u) is u = unlimit_param(x). The intended use
    of these functions is that, during a continuous optimization in which a
    parameter should be restricted to a particular range, the parameter over
    which the optimization occurs is in fact the unlimited parameeter, which is
    allowed to take on any real value, and which gets limited to its range
    prior to its use in any calculation.
    '''
    return min + (max - min) * (math.arctan(param)/np.pi + 0.5)
def unlimit_param(param, min=-1, max=1):
    '''
    unlimit_param(x) yields the unlimited parameter u, which may take on any
      real value, from its limited value x, which must be between -1 and 1.
    unlimit_param(x, min, max) uses the given min and max values instead of the
      default values -1 and 1.

    The opposite of u = unlimit_param(x) is x = limit_param(u). The intended use
    of these functions is that, during a continuous optimization in which a
    parameter should be restricted to a particular range, the parameter over
    which the optimization occurs is in fact the unlimited parameeter, which is
    allowed to take on any real value, and which gets limited to its range
    prior to its use in any calculation.
    '''
    return math.tan(np.pi * ((param - min) / (max - min) - 0.5))
def imap_forget(imap, ks):
    '''
    imap_forget(imap, k) yields imap after clearing the cacne for key k. This
      can also be accomplished with: del imap[k].
    imap_forget(imap, [k1, k2, ...]) clears the cache for all the given keys.
    '''
    if pimms.is_str(ks): ks = [ks]
    for k in ks:
        del imap[k]
    return imap
def imap_efferents(imap, ks):
    '''
    imap_efferents(imap, k) yields the key names of any efferent of the key k
      in imap.
    imap_efferents(imap, [k1, k2, ...]) yields the key names of the any
      efferent of any of the given keys.
    '''
    if pimms.is_str(ks): ks = [ks]
    effs = set([])
    for (k,deps) in imap.plan.dependencies.items():
        if k in ks: continue
        for d in deps:
            if d in ks:
                effs.add(k)
                break
    return list(effs)
