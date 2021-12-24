####################################################################################################
# neuropythy/optimize/core.py
# Code for optimizing functions; made with the intention of optimizing functions on the cortical
# surface using gradient descent or a gradient-aware method.
# By Noah C. Benson

import os, gzip, types, six, abc, pimms
import numpy                 as np
import numpy.linalg          as npla
import scipy                 as sp
import scipy.sparse          as sps
import scipy.optimize        as spopt
import pyrsistent            as pyr
from   functools         import reduce
from   ..                import geometry as geo
from   ..                import mri      as mri
from   ..geometry        import (triangle_area)

# Functions that used to live in neuropythy.util.core ##############################################
def numel(x):
    '''
    numel(x) yields the number of elements in x: the product of the shape of x.
    '''
    return int(np.prod(np.shape(x)))
def rows(x):
    '''
    rows(x) yields the number of rows in x; if x is a scalar, this is still 1.
    '''
    s = np.shape(x)
    return s[0] if len(s) > 0 else 1
def unbroadcast(a, b):
    '''
    unbroadcast(a, b) yields a tuple (aa, bb) that is equivalent to (a, b) except that aa and bb
      have been reshaped such that arithmetic numpy operations such as aa * bb will result in
      row-wise operation instead of column-wise broadcasting.
    '''
    # they could be sparse:
    spa = sps.issparse(a)
    spb = sps.issparse(b)
    if   spa and spb: return (a,b)
    elif spa or  spb:
        def fix(sp,nm):
            nm = np.asarray(nm)
            dnm = len(nm.shape)
            nnm = np.prod(nm.shape)
            # if we have (sparse matrix) * (high-dim array), unbroadcast the dense array
            if   dnm == 0: return (sp, np.reshape(nm, (1,   1)))
            elif dnm == 1: return (sp, np.reshape(nm, (nnm, 1)))
            elif dnm == 2: return (sp, nm)
            else:          return unbroadcast(sp.toarray(), nm)
        return fix(a, b) if spa else tuple(reversed(fix(b, a)))
    # okay, no sparse matrices found:
    a = np.asarray(a)
    b = np.asarray(b)
    da = len(a.shape)
    db = len(b.shape)
    if   da > db: return (a, np.reshape(b, b.shape + tuple(np.ones(da-db, dtype=np.int))))
    elif da < db: return (np.reshape(a, a.shape + tuple(np.ones(db-da, dtype=np.int))), b)
    else:         return (a, b)
def cplus(*args):
    '''
    cplus(a, b...) returns the sum of all the values as a numpy array object. Like numpy's add
      function or a+b syntax, plus will thread over the latest dimension possible.

    Additionally, cplus works correctly with sparse arrays.
    '''
    n = len(args)
    if   n == 0: return np.asarray(0)
    elif n == 1: return np.asarray(args[0])
    elif n >  2: return reduce(plus, args)
    (a,b) = args
    if sps.issparse(a):
        if not sps.issparse(b):
            b = np.asarray(b)
            if len(b.shape) == 0: b = np.reshape(b, (1,1))
    elif sps.issparse(b):
        a = np.asarray(a)
        if len(a.shape) == 0: a = np.reshape(a, (1,1))
    else:
        a = np.asarray(a)
        b = np.asarray(b)
    return a + b
def plus(*args):
    '''
    plus(a, b...) returns the sum of all the values as a numpy array object. Unlike numpy's add
      function or a+b syntax, plus will thread over the earliest dimension possible; thus if a.shape
      a.shape is (4,2) and b.shape is 4, plus(a,b) is a equivalent to
      [ai+bi for (ai,bi) in zip(a,b)].
    '''
    n = len(args)
    if   n == 0: return np.asarray(0)
    elif n == 1: return np.asarray(args[0])
    elif n >  2: return reduce(plus, args)
    (a,b) = unbroadcast(*args)
    return a + b
def cminus(a, b):
    '''
    cminus(a, b) returns the difference a - b as a numpy array object. Like numpy's subtract
      function or a - b syntax, minus will thread over the latest dimension possible.
    '''
    # adding/subtracting a constant to/from a sparse array is an error...
    spa = sps.issparse(a)
    spb = sps.issparse(b)
    if not spa: a = np.asarray(a)
    if not spb: b = np.asarray(b)
    if   spa: b = np.reshape(b, (1,1)) if len(np.shape(b)) == 0 else b
    elif spb: a = np.reshape(a, (1,1)) if len(np.shape(a)) == 0 else a
    return a - b
def minus(a, b):
    '''
    minus(a, b) returns the difference a - b as a numpy array object. Unlike numpy's subtract
      function or a - b syntax, minus will thread over the earliest dimension possible; thus if
      a.shape is (4,2) and b.shape is 4, a - b is a equivalent to [ai-bi for (ai,bi) in zip(a,b)].
    '''
    (a,b) = unbroadcast(a,b)
    return a - b
def ctimes(*args):
    '''
    ctimes(a, b...) returns the product of all the values as a numpy array object. Like numpy's
      multiply function or a*b syntax, times will thread over the latest dimension possible; thus
      if a.shape is (4,2) and b.shape is 2, times(a,b) is a equivalent to a * b.

    Unlike numpy's multiply function, ctimes works with sparse matrices and will reify them.
    '''
    n = len(args)
    if   n == 0: return np.asarray(0)
    elif n == 1: return np.asarray(args[0])
    elif n >  2: return reduce(plus, args)
    (a,b) = args
    if   sps.issparse(a): return a.multiply(b)
    elif sps.issparse(b): return b.multiply(a)
    else:                 return np.asarray(a) * b
def times(*args):
    '''
    times(a, b...) returns the product of all the values as a numpy array object. Unlike numpy's
      multiply function or a*b syntax, times will thread over the earliest dimension possible; thus
      if a.shape is (4,2) and b.shape is 4, times(a,b) is a equivalent to
      [ai*bi for (ai,bi) in zip(a,b)].
    '''
    n = len(args)
    if   n == 0: return np.asarray(0)
    elif n == 1: return np.asarray(args[0])
    elif n >  2: return reduce(plus, args)
    (a,b) = unbroadcast(*args)
    if   sps.issparse(a): return a.multiply(b)
    elif sps.issparse(b): return b.multiply(a)
    else:                 return a * b
def inv(x):
    '''
    inv(x) yields the inverse of x, 1/x.

    Note that inv supports sparse matrices, but it is forced to reify them. Additionally, because
    inv raises an error on divide-by-zero, they are unlikely to work. For better sparse-matrix
    support, see zinv.
    '''
    if sps.issparse(x): return 1.0 / x.toarray()        
    else:               return 1.0 / np.asarray(x)
def zinv(x, null=0):
    '''
    zinv(x) yields 1/x if x is not close to 0 and 0 otherwise. Automatically threads over arrays and
      supports sparse-arrays.

    The optional argument null (default: 0) may be given to specify that zeros in the arary x should
    instead be replaced with the given value. Note that if this value is not equal to 0, then any
    sparse array passed to zinv must be reified.

    The zinv function never raises an error due to divide-by-zero; if you desire this behavior, use
    the inv function instead.
    '''
    if sps.issparse(x):
        if null != 0: return zinv(x.toarray(), null=null)
        x = x.copy()
        x.data = zinv(x.data)
        try: x.eliminate_zeros()
        except Exception: pass
        return x
    else:
        x = np.asarray(x)
        z = np.isclose(x, 0)
        r = np.logical_not(z) / (x + z)
        if null == 0: return r
        r[z] = null
        return r
def cdivide(a, b):
    '''
    cdivide(a, b) returns the quotient a / b as a numpy array object. Like numpy's divide function
      or a/b syntax, divide will thread over the latest dimension possible. Unlike numpy's divide,
      cdivide works with sparse matrices.

    Note that warnings/errors are raised by this function when divide-by-zero occurs, so it is
    usually not useful to use cdivide() with sparse matrices--see czdivide instead.
    '''
    if   sps.issparse(a): return a.multiply(inv(b))
    elif sps.issparse(b): return np.asarray(a) / b.toarray()
    else:                 return np.asarray(a) / np.asarray(b)
def divide(a, b):
    '''
    divide(a, b) returns the quotient a / b as a numpy array object. Unlike numpy's divide function
      or a/b syntax, divide will thread over the earliest dimension possible; thus if a.shape is
      (4,2) and b.shape is 4, divide(a,b) is a equivalent to [ai/bi for (ai,bi) in zip(a,b)].

    Note that divide(a,b) supports sparse array arguments, but if b is a sparse matrix, then it will
    be reified. Additionally, errors are raised by this function when divide-by-zero occurs, so it
    is usually not useful to use divide() with sparse matrices--see zdivide instead.
    '''
    (a,b) = unbroadcast(a,b)
    return cdivide(a,b)
def czdivide(a, b, null=0):
    '''
    czdivide(a, b) returns the quotient a / b as a numpy array object. Like numpy's divide function
      or a/b syntax, czdivide will thread over the latest dimension possible. Unlike numpy's divide,
      czdivide works with sparse matrices. Additionally, czdivide multiplies a by the zinv of b, so
      divide-by-zero entries are replaced with 0 in the result.

    The optional argument null (default: 0) may be given to specify that zeros in the arary b should
    instead be replaced with the given value in the result. Note that if this value is not equal to
    0, then any sparse array passed as argument b must be reified.

    The czdivide function never raises an error due to divide-by-zero; if you desire this behavior,
    use the cdivide function instead.
    '''
    if null == 0:         return a.multiply(zinv(b)) if sps.issparse(a) else a * zinv(b)
    elif sps.issparse(b): b = b.toarray()
    else:                 b = np.asarray(b)
    z = np.isclose(b, 0)
    q = np.logical_not(z)
    zi = q / (b + z)
    if sps.issparse(a):
        r = a.multiply(zi).tocsr()
    else:
        r = np.asarray(a) * zi
    r[np.ones(a.shape, dtype=np.bool)*z] = null
    return r
def zdivide(a, b, null=0):
    '''
    zdivide(a, b) returns the quotient a / b as a numpy array object. Unlike numpy's divide function
      or a/b syntax, zdivide will thread over the earliest dimension possible; thus if a.shape is
      (4,2) and b.shape is 4, zdivide(a,b) is a equivalent to [ai*zinv(bi) for (ai,bi) in zip(a,b)].

    The optional argument null (default: 0) may be given to specify that zeros in the arary b should
    instead be replaced with the given value in the result. Note that if this value is not equal to
    0, then any sparse array passed as argument b must be reified.

    The zdivide function never raises an error due to divide-by-zero; if you desire this behavior,
    use the divide function instead.

    Note that zdivide(a,b, null=z) is not quite equivalent to a*zinv(b, null=z) unless z is 0; if z
    is not zero, then the same elements that are zet to z in zinv(b, null=z) are set to z in the
    result of zdivide(a,b, null=z) rather than the equivalent element of a times z.
    '''
    (a,b) = unbroadcast(a,b)
    return czdivide(a,b, null=null)
def cpower(a,b):
    '''
    cpower(a,b) is equivalent to a**b except that it also operates over sparse arrays; though it
    must reify them to do so.
    '''
    if sps.issparse(a): a = a.toarray()
    if sps.issparse(b): b = b.toarray()
    return a ** b
hpi    = np.pi / 2
tau    = 2 * np.pi
negpi  = -np.pi
neghpi = -hpi
negtau = -tau
def power(a,b):
    '''
    power(a,b) is equivalent to a**b except that, like the neuropythy.util.times function, it
      threads over the earliest dimension possible rather than the latest, as numpy's power function
      and ** syntax do. The power() function also works with sparse arrays; though it must reify
      them during the process.
    '''
    (a,b) = unbroadcast(a,b)
    return cpower(a,b)
def inner(a,b):
    '''
    inner(a,b) yields the dot product of a and b, doing so in a fashion that respects sparse
      matrices when encountered. This does not error check for bad dimensionality.

    If a or b are constants, then the result is just the a*b; if a and b are both vectors or both
    matrices, then the inner product is dot(a,b); if a is a vector and b is a matrix, this is
    equivalent to as if a were a matrix with 1 row; and if a is a matrix and b a vector, this is
    equivalent to as if b were a matrix with 1 column.
    '''
    if   sps.issparse(a): return a.dot(b)
    else: a = np.asarray(a)
    if len(a.shape) == 0: return a*b
    if sps.issparse(b):
        if len(a.shape) == 1: return b.T.dot(a)
        else:                 return b.T.dot(a.T).T
    else: b = np.asarray(b)
    if len(b.shape) == 0: return a*b
    if len(a.shape) == 1 and len(b.shape) == 2: return np.dot(b.T, a)
    else: return np.dot(a,b)
def sine(x):
    '''
    sine(x) is equivalent to sin(x) except that it also works on sparse arrays.
    '''
    if sps.issparse(x):
        x = x.copy()
        x.data = np.sine(x.data)
        return x
    else: return np.sin(x)
def cosine(x):
    '''
    cosine(x) is equivalent to cos(x) except that it also works on sparse arrays.
    '''
    # cos(0) = 1 so no point in keeping these sparse
    if sps.issparse(x): x = x.toarray(x)
    return np.cos(x)
def tangent(x, null=(-np.inf, np.inf), rtol=default_rtol, atol=default_atol):
    '''
    tangent(x) is equivalent to tan(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x == -pi/2 or -pi/2. If only one number is given, then it is
    used for both values; otherwise the first value corresponds to -pi/2 and the second to pi/2.
    A value of x is considered to be equal to one of these valids based on numpy.isclose. The
    optional arguments rtol and atol are passed along to isclose. If null is None, then no
    replacement is performed.
    '''
    if sps.issparse(x):
        x = x.copy()
        x.data = tangent(x.data, null=null, rtol=rtol, atol=atol)
        return x
    else: x = np.asarray(x)
    if rtol is None: rtol = default_rtol
    if atol is None: atol = default_atol
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    x = np.mod(x + pi, tau) - pi
    ii = None if nln is None else np.where(np.isclose(x, neghpi, rtol=rtol, atol=atol))
    jj = None if nlp is None else np.where(np.isclose(x, hpi,    rtol=rtol, atol=atol))
    x = np.tan(x)
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def cotangent(x, null=(-np.inf, np.inf), rtol=default_rtol, atol=default_atol):
    '''
    cotangent(x) is equivalent to cot(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x == 0 or pi. If only one number is given, then it is used for
    both values; otherwise the first value corresponds to 0 and the second to pi.  A value of x is
    considered to be equal to one of these valids based on numpy.isclose. The optional arguments
    rtol and atol are passed along to isclose. If null is None, then no replacement is performed.
    '''
    if sps.issparse(x): x = x.toarray()
    else:               x = np.asarray(x)
    if rtol is None: rtol = default_rtol
    if atol is None: atol = default_atol
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    x = np.mod(x + hpi, tau) - hpi
    ii = None if nln is None else np.where(np.isclose(x, 0,  rtol=rtol, atol=atol))
    jj = None if nlp is None else np.where(np.isclose(x, pi, rtol=rtol, atol=atol))
    x = np.tan(x)
    if ii: x[ii] = 1
    if jj: x[jj] = 1
    x = 1.0 / x
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def secant(x, null=(-np.inf, np.inf), rtol=default_rtol, atol=default_atol):
    '''
    secant(x) is equivalent to 1/sin(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x == -pi/2 or -pi/2. If only one number is given, then it is
    used for both values; otherwise the first value corresponds to -pi/2 and the second to pi/2.
    A value of x is considered to be equal to one of these valids based on numpy.isclose. The
    optional arguments rtol and atol are passed along to isclose. If null is None, then an error is
    raised when -pi/2 or pi/2 is encountered.
    '''
    if sps.issparse(x): x = x.toarray()
    else:               x = np.asarray(x)
    if rtol is None: rtol = default_rtol
    if atol is None: atol = default_atol
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    x = np.mod(x + pi, tau) - pi
    ii = None if nln is None else np.where(np.isclose(x, neghpi, rtol=rtol, atol=atol))
    jj = None if nlp is None else np.where(np.isclose(x, hpi,    rtol=rtol, atol=atol))
    x = np.cos(x)
    if ii: x[ii] = 1.0
    if jj: x[jj] = 1.0
    x = 1.0/x
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def cosecant(x, null=(-np.inf, np.inf), rtol=default_rtol, atol=default_atol):
    '''
    cosecant(x) is equivalent to 1/sin(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x == 0 or pi. If only one number is given, then it is used for
    both values; otherwise the first value corresponds to 0 and the second to pi. A value x is
    considered to be equal to one of these valids based on numpy.isclose. The optional arguments
    rtol and atol are passed along to isclose. If null is None, then an error is raised when -pi/2
    or pi/2 is encountered.
    '''
    if sps.issparse(x): x = x.toarray()
    else:               x = np.asarray(x)
    if rtol is None: rtol = default_rtol
    if atol is None: atol = default_atol
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    x = np.mod(x + hpi, tau) - hpi # center on pi/2 so that 0 and pi are easy to detect
    ii = None if nln is None else np.where(np.isclose(x, 0,  rtol=rtol, atol=atol))
    jj = None if nlp is None else np.where(np.isclose(x, pi, rtol=rtol, atol=atol))
    x = np.sin(x)
    if ii: x[ii] = 1.0
    if jj: x[jj] = 1.0
    x = 1.0/x
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def arcsine(x, null=(-np.inf, np.inf)):
    '''
    arcsine(x) is equivalent to asin(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x < -1 or x > 1. If only one number is given, then it is used
    for both values; otherwise the first value corresponds to <-1 and the second to >1.  If null is
    None, then an error is raised when invalid values are encountered.
    '''
    if sps.issparse(x):
        x = x.copy()
        x.data = arcsine(x.data, null=null, rtol=rtol, atol=atol)
        return x
    else: x = np.asarray(x)
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    ii = None if nln is None else np.where(x < -1)
    jj = None if nlp is None else np.where(x > 1)
    if ii: x[ii] = 0
    if jj: x[jj] = 0
    x = np.arcsin(x)
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def arccosine(x, null=(-np.inf, np.inf)):
    '''
    arccosine(x) is equivalent to acos(x) except that it also works on sparse arrays.

    The optional argument null (default, (-numpy.inf, numpy.inf)) may be specified to indicate what
    value(s) should be assigned when x < -1 or x > 1. If only one number is given, then it is used
    for both values; otherwise the first value corresponds to <-1 and the second to >1.  If null is
    None, then an error is raised when invalid values are encountered.
    '''
    if sps.issparse(x): x = x.toarray()
    else:               x = np.asarray(x)
    try:    (nln,nlp) = null
    except Exception: (nln,nlp) = (null,null)
    ii = None if nln is None else np.where(x < -1)
    jj = None if nlp is None else np.where(x > 1)
    if ii: x[ii] = 0
    if jj: x[jj] = 0
    x = np.arccos(x)
    if ii: x[ii] = nln
    if jj: x[jj] = nlp
    return x
def arctangent(y, x=None, null=0, broadcast=False, rtol=default_rtol, atol=default_atol):
    '''
    arctangent(x) is equivalent to atan(x) except that it also works on sparse arrays.
    arctangent(y,x) is equivalent to atan2(y,x) except that it also works on sparse arrays.

    The optional argument null (default: 0) specifies the result found when y and x both equal 0. If
    null is None, then an error is raised on this condition. Note that if null is not 0, then it is
    more likely that sparse arrays will have to be reified. If null is set to None, then no attempt
    is made to detect null values.

    The optional argument broadcast (default: False) specifies whether numpy-like (True) or
    Mathematica-like (False) broadcasting should be used. Broadcasting resolves ambiguous calls to
    arctangent, such as artangent([a,b,c], [[d,e,f],[g,h,i],[j,k,l]]). If broadcasting is True, 
    arctangent(y,x) behaves like numpy.arctan2(y,x), so [a,b,c] is interpreted like [[a,b,c],
    [a,b,c], [a,b,c]]. If broadcasting is False, [a,b,c] is interpreted like [[a,a,a], [b,b,b],
    [c,c,c]].
    '''
    if sps.issparse(y):
        if x is None:
            y = y.copy()
            y.data = np.arctan(y.data)
            return y
        elif null is not None and null != 0:
            # we need to reify anyway...
            y = y.toarray()
            if sps.issparse(x): x = x.toarray()
        else:
            # anywhere that y is zero must have an arctan of 0 or null (which is 0), so we only have
            # to look at those values that are non-zero in y
            (yr,yc,yv) = sps.find(y)
            xv = np.asarray(x[rr,rc].flat)
            res = y.copy()
            res.data = arctangent(yv, xv, null=null)
            res.eliminate_zeros()
            return res
    elif sps.issparse(x): x = x.toarray()
    # we should start by broadcasting if need be...
    if x is None: res = np.arctan(y)
    else:
        if not broadcast: (y,x) = unbroadcast(y,x)
        res = np.arctan2(y, x)
        # find the zeros, if need-be
        if null is not None:
            if rtol is None: rtol = default_rtol
            if atol is None: atol = default_atol
            # even if null is none, we do this because the rtol and atol may be more lenient than
            # the tolerance used by arctan2.
            z = np.isclose(y, 0, rtol=rtol, atol=atol) & np.isclose(x, 0, rtol=rtol, atol=atol)
            res[z] = null
    return res
def flattest(x):
    '''
    flattest(x) yields a 1D numpy vector equivalent to a flattened version of x. Unline
      np.asarray(x).flatten, flattest(x) works with sparse matrices. It does not, however, work with
      ragged arrays.
    '''
    x = x.toarray().flat if sps.issparse(x) else np.asarray(x).flat
    return np.array(x)
def flatter(x, k=1):
    '''
    flatter(x) yields a numpy array equivalent to x but whose first dimension has been flattened.
    flatter(x, k) yields a numpy array whose first k dimensions have been flattened; if k is
      negative, the last k dimensions are flattened. If np.inf or -np.inf is passed, then this is
      equivalent to flattest(x). Note that flatter(x) is equivalent to flatter(x,1).
    flatter(x, 0) yields x.
    '''
    if k == 0: return x
    x = x.toarray() if sps.issparse(x) else np.asarray(x)
    if len(x.shape) - abs(k) < 2: return x.flatten()
    k += np.sign(k)
    if k > 0: return np.reshape(x, (-1,) + x.shape[k:])
    else:     return np.reshape(x, x.shape[:k] + (-1,))
def part(x, *args):
    '''
    part(x, ii, jj...) is equivalent to x[ii, jj...] if x is a sparse matrix or numpy array and is
      equivalent to np.asarray(x)[ii][:, jj][...] if x is not. If only one argument is passed and
      it is a tuple, then it is passed like x[ii] alone.

    The part function is comparible with slices (though the must be entered using the slice(...)
    rather than the : syntax) and Ellipsis.
    '''
    n = len(args)
    sl = slice(None)
    if sps.issparse(x):
        if n == 1: return x[args[0]]
        elif n > 2: raise ValueError('Too many indices for sparse matrix')
        (ii,jj) = args
        if   ii is Ellipsis: ii = sl
        elif jj is Ellipsis: jj = sl
        ni = pimms.is_number(ii)
        nj = pimms.is_number(jj)
        if   ni and nj: return x[ii,jj]
        elif ni:        return x[ii,jj].toarray()[0]
        elif nj:        return x[ii,jj].toarray()[:,0]
        else:           return x[ii][:,jj]
    else:
        x = np.asarray(x)
        if n == 1: return x[args[0]]
        i0 = []
        for (k,arg) in enumerate(args):
            if arg is Ellipsis:
                # special case...
                #if Ellipsis in args[ii+1:]: raise ValueError('only one ellipsis allowed per part')
                left = n - k - 1
                i0 = [sl for _ in range(len(x.shape) - left)]
            else:
                x = x[tuple(i0 + [arg])]
                if not pimms.is_number(arg): i0.append(sl)
        return x
def hstack(tup):
    '''
    hstack(x) is equivalent to numpy.hstack(x) or scipy.sparse.hstack(x) except that it works
      correctly with both sparse and dense arrays (if any inputs are dense, it converts all inputs
      to dense arrays).
    '''
    if all([sps.issparse(u) for u in tup]): return sps.hstack(tup, format=tup[0].format)
    else: return np.hstack([u.toarray() if sps.issparse(u) else u for u in tup])
def vstack(tup):
    '''
    vstack(x) is equivalent to numpy.vstack(x) or scipy.sparse.vstack(x) except that it works
      correctly with both sparse and dense arrays (if any inputs are dense, it converts all inputs
      to dense arrays).
    '''
    if all([sps.issparse(u) for u in tup]): return sps.vstack(tup, format=tup[0].format)
    else: return np.vstack([u.toarray() if sps.issparse(u) else u for u in tup])
def repmat(x, r, c):
    '''
    repmat(x, r, c) is equivalent to numpy.matlib.repmat(x, r, c) except that it works correctly for
      sparse matrices.
    '''
    if sps.issparse(x):
        row = sps.hstack([x for _ in range(c)])
        return sps.vstack([row for _ in range(r)], format=x.format)
    else: return np.matlib.repmat(x, r, c)
    
def replace_close(x, xhat, rtol=default_rtol, atol=default_atol, copy=True):
    '''
    replace_close(x, xhat) yields x if x is not close to xhat and xhat otherwise. Closeness is
      determined by numpy's isclose(), and the atol and rtol options are passed along.

    The x and xhat arguments may be lists/arrays.

    The optional argument copy may also be set to False to chop x in-place.
    '''
    if rtol is None: rtol = default_rtol
    if atol is None: atol = default_atol
    x = np.array(x) if copy else np.asarray(x)
    w = np.isclose(x, xhat, rtol=rtol, atol=atol)
    x[w] = np.asarray(xhat)[w]
    return x
def chop(x, rtol=default_rtol, atol=default_atol, copy=True):
    '''
    chop(x) yields x if x is not close to round(x) and round(x) otherwise. Closeness is determined
      by numpy's isclose(), and the atol and rtol options are passed along.

    The x and xhat arguments may be lists/arrays.

    The optional argument copy may also be set to False to chop x in-place.
    '''
    return replace_close(x, np.round(x), rtol=rtol, atol=atol, copy=copy)

def nan_compare(f, x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nan_compare(f, x, y) is equivalent to f(x, y), which is assumed to be a boolean function that
      broadcasts over x and y (such as numpy.less), except that NaN values in either x or y result
      in a value of False instead of being run through f.

    The argument f must be a numpy comparison function such as numpy.less that accepts the optional
    arguments where and out.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to f(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to f(nan, non_nan).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to f(non_nan, nan).
    '''
    #TODO: This should work with sparse matrices as well
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    xii = np.isnan(x)
    yii = np.isnan(y)
    if not xii.any() and not yii.any(): return f(x, y)
    ii  = (~xii) & (~yii)
    out = np.zeros(ii.shape, dtype=np.bool)
    if nan_nan == nan_val and nan_val == val_nan:
        # All the nan-result values are the same; we can simplify a little...
        if nan_nan: out[~ii] = nan_nan
    else:
        if nan_nan: out[   xii &    yii] = nan_nan
        if nan_val: out[   xii & (~yii)] = nan_val
        if val_nan: out[(~xii) &    yii] = val_nan
    return f(x, y, out=out, where=ii)
def naneq(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    naneq(x, y) is equivalent to (x == y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to naneq(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to naneq(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to naneq(nan, 0).
    '''
    return nan_compare(np.equal, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nanne(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nanne(x, y) is equivalent to (x != y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanne(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanne(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanne(nan, 0).
    '''
    return nan_compare(np.not_equal, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nanlt(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nanlt(x, y) is equivalent to (x < y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanlt(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanlt(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nan;t(nan, 0).
    '''
    return nan_compare(np.less, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nanle(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nanle(x, y) is equivalent to (x <= y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanle(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanle(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nanle(nan, 0).
    '''
    return nan_compare(np.less_equal, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nangt(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nangt(x, y) is equivalent to (x > y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nangt(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to nangt(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nangt(nan, 0).
    '''
    return nan_compare(np.greater, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nange(x, y, nan_nan=False, nan_val=False, val_nan=False):
    '''
    nange(x, y) is equivalent to (x >= y) except that NaN values in either x or y result in False.

    The following optional arguments may be provided:
      * nan_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nange(nan, nan).
      * nan_val (default: False) specifies the return value (True or False) for comparisons
        equivalent to nange(nan, 0).
      * val_nan (default: False) specifies the return value (True or False) for comparisons
        equivalent to nange(nan, 0).
    '''
    return nan_compare(np.greater_equal, x, y, nan_nan=nan_nan, nan_val=nan_val, val_nan=val_nan)
def nanlog(x, null=np.nan):
    '''
    nanlog(x) is equivalent to numpy.log(x) except that it avoids calling log on 0 and non-finie
      values; in place of these values, it returns the value null (which is nan by default).
    '''
    x = np.asarray(x)
    ii0 = np.where(np.isfinite(x))
    ii  = np.where(x[ii0] > 0)[0]
    if len(ii) == numel(x): return np.log(x)
    res = np.full(x.shape, null)
    ii = tuple([u[ii] for u in ii0])
    res[ii] = np.log(x[ii])
    return res    


# Helper Functions #################################################################################
def fapply(f, x, tz=False):
    '''
    fapply(f,x) yields the result of applying f either to x, if x is a normal value or array, or to
      x.data if x is a sparse matrix. Does not modify x (unless f modifiex x).

    The optional argument tz (default: False) may be set to True to specify that, if x is a sparse 
    matrix that contains at least 1 element that is a sparse-zero, then f(0) should replace all the
    sparse-zeros in x (unless f(0) == 0).
    '''
    if sps.issparse(x):
        y = x.copy()
        y.data = f(x.data)
        if tz and y.getnnz() < np.prod(y.shape):
            z = f(np.array(0))
            if z != 0:
                y = y.toarray()
                y[y == 0] = z
        return y
    else: return f(x)
def finto(x, ii, n, null=0):
    '''
    finto(x,ii,n) yields a vector u of length n such that u[ii] = x.

    Notes:
      * The ii index may be a tuple (as can be passed to numpy arrays' getitem method) in order to
        specify that the specific elements of a multidimensional output be set. In this case, the
        argument n should be a tuple of sizes (a single integer is taken to be a square/cube/etc).
      * x may be a sparse-array, but in it will be reified by this function.

    The following optional arguments are allowed:
      * null (defaut: 0) specifies the value that should appear in the elements of u that are not
        set.
    '''
    x  = x.toarray() if sps.issparse(x) else np.asarray(x)
    shx = x.shape
    if isinstance(ii, tuple):
        if not pimms.is_vector(n): n = tuple([n for _ in ii])
        if len(n) != len(ii): raise ValueError('%d-dim index but %d-dim output' % (len(ii),len(n)))
        sh = n + shx[1:]
    elif pimms.is_int(ii): sh = (n,) + shx
    else:                  sh = (n,) + shx[1:]
    u = np.zeros(sh, dtype=x.dtype) if null == 0 else np.full(sh, null, dtype=x.dtype)
    u[ii] = x
    return u

# Potential Functions ##############################################################################
@six.add_metaclass(abc.ABCMeta)
class PotentialFunction(object):
    '''
    The PotentialFunction class is intended as the base-class for all potential functions that can
    be minimized by neuropythy. PotentialFunction is effectively an abstract class that requires its
    subclasses to implement the method __call__(), which must take one argument: a numpy vector of
    parameters. The method must return a tuple of (z,dz) where z is  the potential value for the
    given paramters and dz is the Jacobian of the function at the parameters. Note that if the
    potential z returned is a scalar, then dz must be a vector of length len(params); if z is a
    vector, then dz must be a matrix of size (len(z) x len(params))
    '''
    # The value() function should return the potential alone
    @abc.abstractmethod
    def value(self, params):
        '''
        pf.value(params) yields the potential function value at the given parameters params. Values
          must always be vectors; if the function returns a scalar, it must return a 1-element
          vector instead.
        '''
        raise RuntimeError('The value() method was not overloaded for object %s' % self)
    @abc.abstractmethod
    def jacobian(self, params, into=None):
        '''
        pf.jacobian(params) yields the potential function jacobian matrix at the given parameters
          params. The jacobian must always be a (possibly sparse) matrix; if there is a scalar
          output value for the potential function, then Jacobian should be (1 x n); if there is a
          scalar input parameter, the Jacobian should be (n x 1).

        If the optional matrix into is provided then the returned jacobian may optionally be added
        directly into this matrix and returned.
        '''
        raise RuntimeError('The gradient() method was not overloaded for object %s' % self)
    # The __call__ function is how one generally calls a potential function
    def __call__(self, params):
        '''
        pf(params) yields the tuple (z, dz) where z is the potential value at the given parameters
          vector, params, and dz is the vector of the potential gradient.
        '''
        z  = self.value(params)
        dz = self.jacobian(params)
        if sps.issparse(dz): dz = dz.toarray()
        z  = np.squeeze(z)
        dz = np.squeeze(dz)
        return (z,dz)
    def fun(self):
        '''
        pf.fun() yields a value calculation function for the given potential function pf that is
          appropriate for passing to a minimizer.
        '''
        return lambda x: np.squeeze(self.value(x))
    def jac(self):
        '''
        pf.jac() yields a jacobian calculation function for the given potential function pf that is
          appropritate for passing to a minimizer.
        '''
        def _jacobian(x):
            dz = self.jacobian(x)
            if sps.issparse(dz): dz = dz.toarray()
            dz = np.asarray(dz)
            return np.squeeze(dz)
        return _jacobian
    def minimize(self, x0, **kwargs):
        '''
        pf.minimize(x0) minimizes the given potential function starting at the given point x0; any
          additional options are passed along to scipy.optimize.minimize.
        '''
        x0 = np.asarray(x0)
        kwargs = pimms.merge({'jac':self.jac(), 'method':'CG'}, kwargs)
        res = spopt.minimize(self.fun(), x0.flatten(), **kwargs)
        res.x = np.reshape(res.x, x0.shape)
        return res
    def argmin(self, x0, **kwargs):
        '''
        pf.argmin(x0) is equivalent to pf.minimize(x0).x.
        '''
        return self.minimize(x0, **kwargs).x
    def maximize(self, x0, **kwargs):
        '''
        pf.maximize(x0) is equivalent to (-pf).minimize(x0).
        '''
        return (-self).minimize(x0, **kwargs)
    def argmax(self, x0, **kwargs):
        '''
        pf.argmax(x0) is equivalent to pf.maximize(x0).x.
        '''
        return self.maximize(x0, **kwargs).x
    # Arithmetic Operators #########################################################################
    def __getitem__(self, ii):
        return part(self, ii)
    def __neg__(self):
        if is_const_potential(self): return const_potential(-self.c)
        else: return PotentialTimesConstant(self, -1)
    def __add__(self, x):
        x = to_potential(x)
        if is_const_potential(x):
            if   np.isclose(x.c, 0).all(): return self
            elif is_const_potential(self): return const_potential(self.c + x.c)
            else:                          return PotentialPlusConstant(self, x.c)
        elif is_const_potential(self):     return PotentialPlusConstant(x.c, self)
        else:                              return PotentialPlusPotential(self, x)
    def __radd__(self, x):
        return self.__add__(x)
    def __sub__(self, x):
        return self.__add__(-x)
    def __rsub__(self, x):
        return PotentialPlusConstant(-self, x)
    def __mul__(self, x):
        x = to_potential(x)
        if is_const_potential(x):
            if   np.isclose(x.c, 1).all(): return self
            elif np.isclose(x.c, 0).all(): return x
            elif is_const_potential(self): return const_potential(self.c * x.c)
            else:                          return PotentialTimesConstant(self, x.c)
        elif is_const_potential(self):     return PotentialTimesConstant(x, self.c)
        else:                              return PotentialTimesPotential(self, x)
    def __rmul__(self, x):
        return (self * x)
    def __div__(self, x):
        x = to_potential(x)
        if is_const_potential(x):
            if   np.isclose(x.c, 1).all(): return self
            elif is_const_potential(self): return const_potential(self.c / x.c)
            else:                          return PotentialTimesConstant(self, 1.0/x.c)
        else:                              return PotentialTimesPotential(self, 1/x)
    def __rdiv__(self, x):
        x = to_potential(x)
        if is_const_potential(self): return x / self.c
        else:                        return x * PotentialPowerConstant(self, -1)
    def __truediv__(self, x):
        return self.__div__(x)
    def __rtruediv__(self, x):
        return self.__rdiv__(x)
    def __pow__(self, x):
        x = to_potential(x)
        if is_const_potential(x):
            if   is_const_potential(self): return const_potential(power(self.c, x.c))
            else:                          return PotentialPowerConstant(self, x.c)
        elif is_const_potential(self):     return ConstantPowerPotential(self, x)
        else:                              return PotentialPowerPotential(self, x)
    def __rpow__(self, x):
        return to_potential(x).__pow__(self)
    def compose(self, f):
        return compose(self, f)
def is_potential(f):
    '''
    is_potential(f) yields True if f is a potential function and False otherwise.
    '''
    return isinstance(f, PotentialFunction)
def safe_into(into, term):
    if into is None: return term
    into0 = into
    into += term
    if into is into0: return into
    else: return term
@pimms.immutable
class PotentialIdentity(PotentialFunction):
    '''
    PotentialIdentity is a potential function that represents the arguments given to it as outputs.
    '''
    def __init__(self): pass
    def value(self, params): return np.asarray(params)
    def jacobian(self, params, into=None):
        return safe_into(into, sps.eye(numel(params)))
identity = PotentialIdentity()
def is_identity_potential(f):
    '''
    is_identity_potential(f) yields True if f is a potential function that merely yields its
      parameters (f(x) = x); otherwise yields False.
    '''
    return isinstance(f, PotentialIdentity)
@pimms.immutable
class PotentialLambda(PotentialFunction):
    '''
    PotentialLambda is a PotentialFunction type that takes as initialization arguments two functions
    g and h which must return the value and the jacobian, respectively.
    '''
    def __init__(self, g, h):
        self.valfn = g
        self.jacfn = h
    @pimms.param
    def valfn(v): return v
    @pimms.param
    def jacfn(j): return j
    def value(self, x): return self.valfn(x)
    def jacobian(self, x, into=None):
        try:              return self.jacfn(x, into)
        except Exception: return self.jacfn(x)
@pimms.immutable
class PotentialConstant(PotentialFunction):
    def __init__(self, c):
        self.c = c
    @pimms.param
    def c(c0):
        c0 = c0.toarray() if sps.issparse(c0) else np.array(c0)
        if len(c0.shape) > 1: c0 = c0.flatten()
        c0.setflags(write=False)
        return c0
    def value(self, params):
        return self.c
    def jacobian(self, params, into=None):
        c = self.c
        d = 1 if len(c.shape) == 0 else c.shape[0]
        return sps.csr_matrix(([], [[],[]]), shape=(d, len(params))) if into is None else into
def is_const_potential(f):
    '''
    is_const_potential(f) yields True if f is a constant potential function and False otherwise.
    '''
    return isinstance(f, PotentialConstant)
def const_potential(f):
    '''
    const_potential(f) yields f if f is a constant potential function; if f is a constant, yields
      a potential function that always yields f; otherwise raises an error.
    '''
    if is_const_potential(f): return f
    elif pimms.is_array(f, 'number'): return PotentialConstant(f)
    else: raise ValueError('Could not convert given value to potential constant: %s' % f)
def const(f):
    '''
    const(f) yields f.c if f is a constant potential function; if f is a constant, it yields f or
      the equivalent numpy array object; if f is a potential function that is not const, or is not
      a valid potential function constant, yields None.
    '''
    if is_const_potential(f): return f.c
    elif pimms.is_array(f, 'number'):
        if sps.issparse(f): return f
        else: return np.asarray(f)
    else: return None
def to_potential(f):
    '''
    to_potential(f) yields f if f is a potential function; if f is not, but f can be converted to
      a potential function, that conversion is performed then the result is yielded.
    to_potential(Ellipsis) yields a potential function whose output is simply its input (i.e., the
      identity function).
    to_potential(None) is equivalent to to_potential(0).

    The following can be converted into potential functions:
      * Anything for which pimms.is_array(x, 'number') yields True (i.e., arrays of constants).
      * Any tuple (g, h) where g(x) yields a potential value and h(x) yields a jacobian matrix for
        the parameter vector x.
    '''
    if   is_potential(f): return f
    elif f is Ellipsis:   return identity
    elif pimms.is_array(f, 'number'): return const_potential(f)
    elif isinstance(f, tuple) and len(f) == 2: return PotentialLambda(f[0], f[1])
    else: raise ValueError('Could not convert object of type %s to potential function' % type(f))
@pimms.immutable
class PotentialComposition(PotentialFunction):
    def __init__(self, g, h):
        self.g = g
        self.h = h
    @pimms.param
    def g(g0): return g0
    @pimms.param
    def h(h0): return h0
    def value(self, params):
        return self.g.value(self.h.value(params))
    def jacobian(self, params, into=None):
        zh  = self.h.value(params)
        dzh = self.h.jacobian(params)
        zg  = self.g.value(zh)
        dzg = self.g.jacobian(zh)
        return safe_into(into, inner(dzg, dzh))
def compose(*args):
    '''
    compose(g, h...) yields a potential function f that is the result of composing together all the
      arguments g, h, etc. after calling to_potential() on each. The result is defined such that
      f(x) is equivalent to g(h(...(x))).
    '''
    return reduce(lambda h,g: PotentialComposition(g,h), reversed(list(map(to_potential, args))))
@pimms.immutable
class PotentialPart(PotentialFunction):
    def __init__(self, ii, input_len=None):
        self.output_indices = ii
        self.input_len = input_len
    @pimms.param
    def output_indices(ii):
        ii = flattest(ii)
        if (np.issubdtype(ii.dtype, np.dtype('bool').type)): ii = np.where(ii)[0]
        return pimms.imm_array(ii)
    @pimms.param
    def input_len(m):
        if m is None: return m
        assert(pimms.is_int(m) and m > 0)
        return int(m)
    @pimms.value
    def jacobian_matrix(output_indices, input_len):
        m = (np.max(output_indices) + 1) if input_len is None else input_len
        n = len(output_indices)
        return sps.csr_matrix((np.ones(n), (np.arange(n), output_indices)), shape=(n,m))
    def value(self, params):
        ii = self.output_indices
        return flattest(params)[ii]
    def jacobian(self, params, into=None):
        params = flattest(params)
        jm = self.jacobian_matrix
        if jm.shape[1] != len(params):
            jm = jm.copy()
            jm.resize((jm.shape[0], len(params)))
        return safe_into(into, jm)
def part(f, ii=None, input_len=None):
    '''
    part(u, ii) for constant or constant potential u yields a constant-potential form of u[ii].
    part(f, ii) for potential function f yields a potential function g(x) that is equivalent to
      f(x)[ii].
    part(ii) is equivalent to part(identity, ii); i.e., pat of the input parameters to the function.
    '''
    if ii is None: return PotentialPart(f, input_len=input_len)
    f = to_potential(f)
    if is_const_potential(f): return PotentialConstant(f.c[ii])
    else:                     return compose(PotentialPart(ii, input_len=input_len), f)
@pimms.immutable
class PotentialPlusPotential(PotentialFunction):
    def __init__(self, g, h):
        self.g = g
        self.h = h
    @pimms.param
    def g(g0): return g0
    @pimms.param
    def h(h0): return h0
    def value(self, params):
        return self.g.value(params) + self.h.value(params)
    def jacobian(self, params, into=None):
        dg = self.g.jacobian(params, into=into)
        dh = self.h.jacobian(params, into=dg)
        if dh is dg: return dh
        else:        return dh + dg
@pimms.immutable
class PotentialPlusConstant(PotentialFunction):
    def __init__(self, f, c):
        self.f = f
        self.c = c
    @pimms.param
    def f(f0): return f0
    @pimms.param
    def c(c0): return c0
    def value(self,params):
        return self.f.value(params) + self.c
    def jacobian(self, params, into=None):
        return self.f.jacobian(params, into=into)
@pimms.immutable
class PotentialTimesPotential(PotentialFunction):
    def __init__(self, g, h):
        self.g = g
        self.h = h
    @pimms.param
    def g(g0): return g0
    @pimms.param
    def h(h0): return h0
    def value(self, params):
        g = self.g.value(params)
        h = self.h.value(params)
        return g * h
    def jacobian(self, params, into=None):
        g  = self.g.value(params)
        dg = self.g.jacobian(params)
        h  = self.h.value(params)
        dh = self.h.jacobian(params)
        return safe_into(into, cplus(times(dg, h), times(dh, g)))
    def __call__(self, params):
        g  = self.g.value(params)
        dg = self.g.jacobian(params)
        h  = self.h.value(params)
        dh = self.h.jacobian(params)
        return (g*h, cplus(times(dg, h), times(dh, g)))
@pimms.immutable
class PotentialTimesConstant(PotentialFunction):
    def __init__(self, f, c):
        self.f = f
        self.c = c
    @pimms.param
    def f(f0): return f0
    @pimms.param
    def c(c0): return c0
    def value(self, params):
        z = self.f.value(params)
        return z * self.c
    def jacobian(self, params, into=None):
        dz = self.f.jacobian(params)
        return safe_into(into, times(dz, self.c))
@pimms.immutable
class PotentialPowerConstant(PotentialFunction):
    def __init__(self, f, c):
        self.f = f
        self.c = c
    @pimms.param
    def f(f0): return f0
    @pimms.param
    def c(c0): return c0
    def value(self, params):
        z = self.f.value(params)
        return z**self.c
    def jacobian(self, params, into=None):
        z  = self.f.value(params)
        dz = self.f.jacobian(params)
        c  = self.c
        cc = self.c - 1
        if cc <= 0:
            cc = -cc
            z = zinv(z)
        return safe_into(into, times(dz, c * z**cc))
    def __call__(self, params):
        z  = self.f.value(params)
        dz = self.f.jacobian(params)
        c  = self.c
        dz = times(dz, c * z**(c-1))
        return (z, dz)
@pimms.immutable
class ConstantPowerPotential(PotentialFunction):
    def __init__(self, c, f):
        self.f = f
        self.c = c
    @pimms.param
    def f(f0): return f0
    @pimms.param
    def c(c0):
        if np.isclose(c0, 0).any(): raise ValueError('in c**f(x) c cannot be zero')
        return c0
    @pimms.value
    def log_c(c): return np.log(c)
    def value(self, params):
        z = self.f.value(params)
        return self.c**z
    def jacobian(self, params, into=None):
        ctoz = self.value(params)
        dz = self.f.jacobian(params)
        return safe_into(into, times(dz, self.log_c * ctoz))
def exp(x):
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(np.exp(x.c))
    else:                     return ConstantPowerPotential(np.e, x)
def exp2(x):
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(np.exp2(x.c))
    else:                     return ConstantPowerPotential(2.0, x)
@pimms.immutable
class PotentialPowerPotential(PotentialFunction):
    def __init__(self, g, h):
        self.g = g
        self.h = h
    @pimms.param
    def g(g0): return g0
    @pimms.param
    def h(h0): return h0
    def value(self, params):
        zg = self.g.value(params)
        zh = self.h.value(params)
        return zg ** zh
    def jacobian(self, params, into=None):
        zg  = self.g.value(params)
        dzg = self.g.jacobian(params)
        zh  = self.h.value(params)
        dzh = self.h.jacobian(params)
        z   = zg ** zh
        return safe_into(into, times(plus(times(dzg, zh, inv(zg)), times(dzh, np.log(zg))), z))
def power(x,y):
    x = to_potential(x)
    y = to_potential(y)
    xc = is_const_potential(x)
    yc = is_const_potential(y)
    if xc and yc: return PotentialPowerPotential(x,   y)
    elif xc:      return ConstantPowerPotential( x.c, y)
    elif yc:      return PotentialPowerConstant( x,   y.c)
    else:         return PotentialConstant(power(x.c, y.c))
def sqrt(x): return power(x, 0.5)
def cbrt(x): return power(x, 1.0/3.0)
@pimms.immutable
class PotentialLog(PotentialFunction):
    def __init__(self, f, base=None):
        self.f = f
        self.base = base
    @pimms.param
    def f(f0): return to_potential(f0)
    @pimms.param
    def base(b): return None if b is None else to_potential(b)
    def value(self, params):
        z = self.f.value(params)
        if self.base is None: return np.log(z)
        b = self.base.value(params)
        return np.log(z)/np.log(b)
    def jacobian(self, params, into=None):
        z  = self.f.value(params)
        dz = self.f.jacobian(params)
        if self.base is None:
            dz = divide(dz, z)
        else:
            b = self.base.value(params)
            db = self.base.jacobian(params)
            logb = np.log(b)
            dz = dz / logb - times(np.log(z), db) / (b * logb * logb)
        return safe_into(into, dz)
def log(x, base=None):
    x = to_potential(x)
    xc = is_const_potential(x)
    if base is None:
        if xc: return PotentialConstant(np.log(x.c))
        else:  return PotentialLog(x)
    base = to_potential(base)
    bc = is_const_potential(base)
    if xc and bc: return PotentialConstant(np.log(x.c, bc.c))
    else:         return PotentialLog(x, base)
def log2(x):  return log(x, 2)
def log10(x): return log(x, 10)
@pimms.immutable
class PotentialSum(PotentialFunction):
    def __init__(self, f, weights=None):
        self.f = f
        self.weights = None
    @pimms.param
    def f(f0): return to_potential(f0)
    @pimms.param
    def weights(w): return None if w is None else pimms.imm_array(w)
    def value(self, params):
        z = self.f.value(params)
        w = self.weights
        if w is None: return np.sum(z)
        else: return np.dot(z, w)
    def jacobian(self, params, into=None):
        dz = self.f.jacobian(params)
        w = self.weights
        if w is None: q = dz.sum(axis=0)
        else:         q = times(dz, w).sum(axis=0)
        return safe_into(into, q)
def sum(x, weights=None):
    '''
    sum(x) yields either a potential-sum object if x is a potential function or the sum of x if x
      is not. If x is not a potential-field then it must be a vector.
    sum(x, weights=w) uses the given weights to produce a weighted sum.
    '''
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(np.sum(x.c))
    else:                     return PotentialSum(x, weights=weights)
@pimms.immutable
class DotPotential(PotentialFunction):
    '''
    DotPotential is a potential function that represents the dot product of two functions.
    '''
    def __init__(self, g, h, g_shape=None, h_shape=None):
        self.g = g
        self.h = h
        self.g_shape = g_shape
        self.h_shape = h_shape
    @pimms.param
    def g(g0): return to_potential(g0)
    @pimms.param
    def h(h0): return to_potential(h0)
    @pimms.param
    def g_shape(gs):
        if gs is None: return None
        gs = tuple(gs)
        if   len(gs) < 2:  return None
        #elif len(gs) == 2: return gs
        else: raise ValueError('dot supports only scalars, vectors, and (soon) matrices')
    @pimms.param
    def h_shape(hs):
        if hs is None: return None
        hs = tuple(hs)
        if   len(hs) < 2:  return None
        #elif len(hs) == 2: return hs
        else: raise ValueError('dot supports only scalars, vectors, and (soon) matrices')
    def value(self, params):
        g = self.g.value(params)
        h = self.h.value(params)
        g = np.reshape(g, self.g_shape) if self.g_shape else flattest(g)
        h = np.reshape(h, self.h_shape) if self.h_shape else flattest(h)
        return flattest(inner(g, h))
    def jacobian(self, params, into=None):
        g = self.g.value(params)
        h = self.h.value(params)
        g = np.reshape(g, self.g_shape) if self.g_shape else flattest(g)
        h = np.reshape(h, self.h_shape) if self.h_shape else flattest(h)
        dg = self.g.jacobian(params)
        dh = self.h.jacobian(params)
        gvec = self.g_shape is None
        hvec = self.h_shape is None
        if gvec == hvec:
            if gvec: return np.sum(g*dh + h*dg)
        # one or both are matrices
        raise NotImplementedError('matrix x matrix dot products not yet supported')
def dot(a, b, ashape=None, bshape=None):
    '''
    dot(a,b) yields a potential function that represents the dot product of a and b.

    Currently only vector and scalars are allowed.
    '''
    a = to_potential(a)
    b = to_potential(b)
    if is_const_potential(a) and is_const_potential(b): return PotentialConstant(np.dot(a.c, b.c))
    else: return DotPotential(a, b, g_shape=ashape, h_shape=bshape)
@pimms.immutable
class CosPotential(PotentialFunction):
    '''
    CosPotential is a potential function that represents cos(x).
    '''
    def __init__(self): pass
    def value(self, x): return cosine(x)
    def jacobian(self, x, into=None):
        x = flattest(x)
        z = sps.diags(-sine(x))
        return safe_into(into, z)
@pimms.immutable
class SinPotential(PotentialFunction):
    '''
    SinPotential is a potential function that represents sin(x).
    '''
    def __init__(self): pass
    def value(self, x): return sine(x)
    def jacobian(self, x, into=None):
        x = flattest(x)
        z = sps.diags(cosine(x))
        return safe_into(into, z)
@pimms.immutable
class TanPotential(PotentialFunction):
    '''
    TanPotential is a potential function that represents tan(x).
    '''
    def __init__(self): pass
    def value(self, x): return tangent(x)
    def jacobian(self, x, into=None):
        x = flattest(x)
        z = sps.diags(secant(x)**2)
        return safe_into(into, z)
@pimms.immutable
class SecPotential(PotentialFunction):
    '''
    SecPotential is a potential function that represents sec(x).
    '''
    def __init__(self): pass
    def value(self, x): return secant(x)
    def jacobian(self, x, into=None):
        x = flattest(x)
        z = sps.diags(secant(x)*tangent(x))
        return safe_into(into, z)
@pimms.immutable
class CscPotential(PotentialFunction):
    '''
    CscPotential is a potential function that represents csc(x).
    '''
    def __init__(self): pass
    def value(self, x): return cosecant(x)
    def jacobian(self, x, into=None):
        x = flattest(x)
        z = sps.diags(-cosecant(x)*cotangent(x))
        return safe_into(into, z)
@pimms.immutable
class CotPotential(PotentialFunction):
    '''
    CotPotential is a potential function that represents cot(x).
    '''
    def __init__(self): pass
    def value(self, x): return cotangent(x)
    def jacobian(self, x, into=None):
        x = flattest(x)
        return safe_into(into, -cosecant(x)**2)
def cos(x):
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(cosine(x.c))
    elif x is identity:       return CosPotential()
    else:                     return compose(CosPotential(), x)
def sin(x):
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(sine(x.c))
    elif x is identity:       return SinPotential()
    else:                     return compose(SinPotential(), x)
def tan(x):
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(tangent(x.c))
    elif x is identity:       return TanPotential()
    else:                     return compose(TanPotential(), x)
def sec(x):
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(secant(x.c))
    elif x is identity:       return SecPotential()
    else:                     return compose(SecPotential(), x)
def csc(x):
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(cosecant(x.c))
    elif x is identity:       return CscPotential()
    else:                     return compose(CscPotential(), x)
def cot(x):
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(cotangent(x.c))
    elif x is identity:       return CotPotential()
    else:                     return compose(CotPotential(), x)
@pimms.immutable
class ArcSinPotential(PotentialFunction):
    '''
    ArcSinPotential is a potential function that represents asin(x).
    '''
    def __init__(self): pass
    def value(self, x): return arcsine(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        z = 1.0 / np.sqrt(1.0 - x**2)
        z = sps.diags(z)
        return safe_into(into, z)
@pimms.immutable
class ArcCosPotential(PotentialFunction):
    '''
    ArcCosPotential is a potential function that represents acos(x).
    '''
    def __init__(self): pass
    def value(self, x): return arccosine(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        z = -1.0 / np.sqrt(1.0 - x**2)
        z = sps.diags(z)
        return safe_into(into, z)
@pimms.immutable
class ArcTanPotential(PotentialFunction):
    '''
    ArcTanPotential is a potential function that represents atan(x).
    '''
    def __init__(self): pass
    def value(self, x): return arctangent(x)
    def jacobian(self, x, into=None):
        x = flattest(x)[None]
        z = 1.0 / (1.0 + x**2)
        z = sps.diags(z)
        return safe_into(into, z)
@pimms.immutable
class ArcTan2Potential(PotentialFunction):
    '''
    ArcTan2Potential is a potential function that represents atan2(y,x).
    '''
    def __init__(self, y, x):
        self.y = y
        self.x = x
    @pimms.param
    def y(y0): return to_potential(y0)
    @pimms.param
    def x(x0): return to_potential(x0)
    def value(self, params):
        y = self.y.value(params)
        x = self.x.value(params)
        return arctangent(y, x)
    def jacobian(self, params, into=None):
        y  = self.y.value(params)
        x  = self.x.value(params)
        dy = self.y.jacobian(params)
        dx = self.x.jacobian(params)
        if   dy.shape[0] == 1 and dx.shape[0] > 1: dy = repmat(dy, dx.shape[0], 1)
        elif dx.shape[0] == 1 and dy.shape[0] > 1: dx = repmat(dx, dy.shape[0], 1)
        dz = zdivide(times(dy, x) - times(dx, y), x**2 + y**2)
        return safe_into(into, dz)
def asin(x):
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(arcsine(x.c))
    elif x is identity:       return ArcSinPotential()
    else:                     return compose(ArcSinPotential(), x)
def acos(x):
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(arccosine(x.c))
    elif x is identity:       return ArcCosPotential()
    else:                     return compose(ArcCosPotential(), x)
def atan(x):
    x = to_potential(x)
    if is_const_potential(x): return PotentialConstant(arctangent(x.c))
    elif x is identity:       return ArcTanPotential()
    else:                     return compose(ArcTanPotential(), x)
def atan2(y,x): return ArcTan2Potential(y, x)
def row_norms(ii, f=Ellipsis, squared=False):
    '''
    row_norms(ii) yields a potential function h(x) that calculates the vector norms of the rows of
      the matrix formed by [x[i] for i in ii] (ii is a matrix of parameter indices).
    row_norms(ii, f) yield a potential function h(x) equivalent to compose(row_norms(ii), f).
    '''
    try:
        (n,m) = ii
        # matrix shape given
        ii = np.reshape(np.arange(n*m), (n,m))
    except Exception: ii = np.asarray(ii)
    f = to_potential(f)
    if is_const_potential(f):
        q = flattest(f.c)
        q = np.sum([q[i]**2 for i in ii.T], axis=0)
        return PotentialConstant(q if squared else np.sqrt(q))
    F = reduce(lambda a,b: a + b, [part(Ellipsis, col)**2 for col in ii.T])
    F = compose(F, f)
    if not squared: F = sqrt(F)
    return F
def col_norms(ii, f=Ellipsis, squared=False):
    '''
    col_norms(ii) yields a potential function h(x) that calculates the vector norms of the columns
      of the matrix formed by [x[i] for i in ii] (ii is a matrix of parameter indices).
    col_norms(ii, f) yield a potential function h(x) equivalent to compose(col_norms(ii), f).
    '''
    try:
        (n,m) = ii
        # matrix shape given
        ii = np.reshape(np.arange(n*m), (n,m))
    except Exception: ii = np.asarray(ii)
    f = to_potential(f)
    if is_const_potential(f):
        q = flattest(f.c)
        q = np.sum([q[i]**2 for i in ii], axis=0)
        return PotentialConstant(q if squared else np.sqrt(q))
    F = reduce(lambda a,b: a + b, [part(Ellipsis, col)**2 for col in ii])
    F = compose(F, f)
    if not squared: F = sqrt(F)
    return F
def distances(a, b, shape, squared=False, axis=1):
    '''
    distances(a, b, (n,d)) yields a potential function whose output is equivalent to the row-norms
      of reshape(a(x), (n,d)) - reshape(b(x), (n,d)).
    
    The shape argument (n,m) may alternately be a matrix of parameter indices, as can be passed to
    row_norms and col_norms.

    The following optional arguments are accepted:
      * squared (default: False) specifies whether the output should be the square distance or the
        distance.
      * axis (default: 1) specifies whether the rows (axis = 1) or columns (axis = 0) are treated
        as the vectors between which the distances should be calculated.
    '''
    a = to_potential(a)
    b = to_potential(b)
    if axis == 1: return row_norms(shape, a - b, squared=squared)
    else:         return col_norms(shape, a - b, squared=squared)
@pimms.immutable
class PotentialPiecewise(PotentialFunction):
    def __init__(self, dflt, *args):
        self.default = dflt
        self.pieces = args
    @pimms.param
    def default(d): return to_potential(d)
    @pimms.param
    def pieces(ps):
        r = []
        for p in ps:
            try:              ((mn, mx), f) = p
            except Exception: (mn,  mx,  f) = p
            if mx < mn: raise ValueError('given piece has mn > mx: %s' % p)
            f = to_potential(f)
            r.append(((mn,mx),f))
        r = sorted(r, key=lambda x:x[0])
        for (((lmn,lmx),_),((umn,umx),_)) in zip(r[:-1],r[1:]):
            if lmx > umn: raise ValueError('pieces contain overlapping ranges')
        return tuple(r)
    @pimms.value
    def pieces_with_default(pieces, default):
        ps = list(pieces)
        ps.append(((-np.inf,np.inf), default))
        return tuple(ps)
    def value(self, params):
        params = flattest(params)
        n = len(params)
        ii = np.arange(n)
        res = np.zeros(n)
        for ((mn,mx), f) in self.pieces_with_default:
            if len(ii) == 0: break
            k = np.where((params >= mn) & (params <= mx))[0]
            if len(k) == 0: continue
            kk = ii[k]
            res[kk] = f.value(params[k])
            ii = np.delete(ii, k)
            params = np.delete(params, k)
        return res
    def jacobian(self, params, into=None):
        params = flattest(params)
        n = len(params)
        ii = np.arange(n)
        (rs,cs,zs) = ([],[],[])
        for ((mn,mx), f) in self.pieces_with_default:
            if len(ii) == 0: break
            k = np.where((params >= mn) & (params <= mx))[0]
            if len(k) == 0: continue
            kk = ii[k]
            j = f.jacobian(params[k])
            if j.shape[0] == 1 and j.shape[1] > 1: j = repmat(j, j.shape[1], 1)
            (rj,cj,vj) = sps.find(j)
            rs.append(kk[rj])
            cs.append(kk[cj])
            zs.append(vj)
            ii = np.delete(ii, k)
            params = np.delete(params, k)
        (rs,cs,zs) = [np.concatenate(us) if len(us) > 0 else [] for us in (rs,cs,zs)]
        dz = sps.csr_matrix((zs, (rs,cs)), shape=(n,n))
        return safe_into(into, dz)
def piecewise(dflt, *spec):
    '''
    piecewise(g, ((mn1, mx1), f1), ((mn2, mx2), f2), ...) yields a potential function f(x) that, for
      each value x[i] in x, calculate y[i] = f1(x[i]) if mn1 <= x[i] <= mx1 else f2(x[i]) if mn2 <=
      x[i] <= mx2 else ... else g(x[i]).

    The ((mn,mx), f) may alternately be specified (mn,mx,f).
    '''
    return PotentialPiecewise(dflt, *spec)
def cos_well(f=Ellipsis, width=np.pi/2, offset=0, scale=1):
    '''
    cos_well() yields a potential function g(x) that calculates 0.5*(1 - cos(x)) for -pi/2 <= x
      <= pi/2 and is 1 outside of that range.
    
    The full formulat of the cosine well is, including optional arguments:
      scale / 2 * (1 - cos((x - offset) / (width/pi)))

    The following optional arguments may be given:
      * width (default: pi) specifies that the frequency of the cos-curve should be pi/width; the
        width is the distance between the points on the cos-curve with the value of 1.
      * offset (default: 0) specifies the offset of the minimum value of the coine curve on the
        x-axis.
      * scale (default: 1) specifies the height of the cosine well.
    '''
    f = to_potential(f)
    freq = np.pi/width*2
    (xmn,xmx) = (offset - width/2, offset + width/2)
    F = piecewise(scale, ((xmn,xmx), scale/2 * (1 - cos(freq * (identity - offset)))))
    if   is_const_potential(f):    return const_potential(F.value(f.c))
    elif is_identity_potential(f): return F
    else:                          return compose(F, f)
def cos_edge(f=Ellipsis, width=np.pi, offset=0, scale=1):
    '''
    cos_edge() yields a potential function g(x) that calculates 0 for x < pi/2, 1 for x > pi/2, and
      0.5*(1 + cos(pi/2*(1 - x))) for x between -pi/2 and pi/2.
    
    The full formulat of the cosine well is, including optional arguments:
      scale/2 * (1 + cos(pi*(0.5 - (x - offset)/width)

    The following optional arguments may be given:
      * width (default: pi) specifies that the frequency of the cos-curve should be pi/width; the
        width is the distance between the points on the cos-curve with the value of 1.
      * offset (default: 0) specifies the offset of the minimum value of the coine curve on the
        x-axis.
      * scale (default: 1) specifies the height of the cosine well.
    '''
    f = to_potential(f)
    freq = np.pi/2
    (xmn,xmx) = (offset - width/2, offset + width/2)
    F = piecewise(scale,
                  ((-np.inf, xmn), 0),
                  ((xmn,xmx), scale/2 * (1 + cos(np.pi*(0.5 - (identity - offset)/width)))))
    if   is_const_potential(f):    return const_potential(F.value(f.c))
    elif is_identity_potential(f): return F
    else:                          return compose(F, f)
def gaussian(f=Ellipsis, mu=0, sigma=1, scale=1, invert=False, normalize=False):
    '''
    gaussian() yields a potential function f(x) that calculates a Gaussian function over x; the
      formula used is given below.
    gaussian(g) yields a function h(x) such that, if f(x) is yielded by gaussian(), h(x) = f(g(x)).

    The formula employed by the Gaussian function is as follows, with mu, sigma, and scale all being
    parameters that one can provide via optional arguments:
      scale * exp(0.5 * ((x - mu) / sigma)**2)
    
    The following optional arguments may be given:
      * mu (default: 0) specifies the mean of the Gaussian.
      * sigma (default: 1) specifies the standard deviation (sigma) parameger of the Gaussian.
      * scale (default: 1) specifies the scale to use.
      * invert (default: False) specifies whether the Gaussian should be inverted. If inverted, then
        the formula, scale * exp(...), is replaced with scale * (1 - exp(...)).
      * normalize (default: False) specifies whether the result should be multiplied by the inverse
        of the area under the uninverted and unscaled curve; i.e., if normalize is True, the entire
        result is multiplied by 1/sqrt(2*pi*sigma**2).
    '''
    f = to_potential(f)
    F = exp(-0.5 * ((f - mu) / sigma)**2)
    if invert: F = 1 - F
    F = F * scale
    if normalize: F = F / (np.sqrt(2.0*np.pi) * sigma)
    return F
@pimms.immutable
class ErfPotential(PotentialFunction):
    '''
    ErfPotential is a potential function that represents the error function.
    '''
    coef = 2.0 / np.sqrt(np.pi)
    def __init__(self): pass
    def value(self, x): return np.erf(flattest(x))
    def jacobian(self, x, into=None):
        x = flattest(x)
        z = ErfPotential.coef * np.exp(-x**2)
        z = sps.diags(z)
        return safe_into(into, z)
def erf(f=Ellipsis):
    '''
    erf(x) yields a potential function that calculates the error function over the input x. If x is
      a constant, yields a constant potential function.
    erf() is equivalent to erf(...), which is just the error function, calculated over its inputs.
    '''
    f = to_potential(f)
    if is_const_potential(f): return const_potential(np.erf(f.c))
    elif is_identity_potential(f): return ErfPotential()
    else: return compose(ErfPotential(), f)
def sigmoid(f=Ellipsis, mu=0, sigma=1, scale=1, invert=False, normalize=False):
    '''
    sigmoid() yields a potential function that is equivalent to the integral of gaussian(), i.e.,
      the error function, but scaled to match gaussian().
    sigmoid(f) is equivalent to compose(sigmoid(), f).

    All options that are accepted by the gaussian() function are accepted by sigmoid() with the same
    default values and are handled in an equivalent manner with the exception of the invert option;
    when a sigmoid is inverted, the function approaches its maximum value at -inf and approaches 0
    at inf.

    Note that because sigmoid() explicitly matches gaussian(), the base formula used is as follows:
      f(x) = scale * sigma * sqrt(pi/2) * erf((x - mu) / (sqrt(2) * sigma))
      k*sig*Sqrt[Pi/2] Erf[(x - mu)/sig/Sqrt[2]]
    '''
    f = to_potential(f)
    F = erf((f - mu) / (sigma * np.sqrt(2.0)))
    if invert: F = 1 - F
    F = np.sqrt(np.pi / 2) * scale * F
    if normalize: F = F / (np.sqrt(2.0*np.pi) * sigma)
    return F
@pimms.immutable
class AbsPotential(PotentialFunction):
    '''
    AbsPotential is a potential function that represents the absolute value function.
    '''
    def __init__(self): pass
    def value(self, x): return np.abs(flattest(x))
    def jacobian(self, x, into=None):
        x = flattest(x)
        z = np.sign(x)
        z = sps.diags(z)
        return safe_into(into, z)
def abs(f=Ellipsis):
    '''
    abs() yields a potential function equivalent to the absolute value of the input.
    abs(f) yields the absolute value of the potential function f.

    Note that abs has a derivative of 0 at 0; this is not mathematically correct, but it is useful
    for the purposes of numerical methods. If you want traditional behavior, it is suggested that
    one instead employ sqrt(f**2).
    '''
    f = to_potential(f)
    if is_const_potential(f): return const_potential(np.abs(f.c))
    elif is_identity_potential(f): return AbsPotential()
    else: return compose(AbsPotential(), f)
@pimms.immutable
class SignPotential(PotentialFunction):
    '''
    SignPotential is a potential function that represents the sign function.
    '''
    def __init__(self): pass
    def value(self, x): return np.sign(flattest(x))
    def jacobian(self, x, into=None):
        n = len(flattest(x))
        return sps.csr_matrix(([], [[],[]]), shape=(n,n)) if into is None else into
def sign(f=Ellipsis):
    '''
    sign() yields a potential function equivalent to the sign of the input.
    sign(f) yields the sign of the potential function f.

    Note that sign has a derivative of 0 at all points; this is not mathematically correct, but it is
    useful for the purposes of numerical methods. If you want traditional behavior, it is suggested
    that one instead employ f/sqrt(f**2).
    '''
    f = to_potential(f)
    if is_const_potential(f): return const_potential(np.sign(f.c))
    elif is_identity_potential(f): return SignPotential()
    else: return compose(SignPotential(), f)
    
@pimms.immutable
class TriangleSignedArea2DPotential(PotentialFunction):
    '''
    TriangleSignedArea2DPotential(n) yields a potential function that tracks the signed area of
    the given face count n embedded in 2 dimensions.
    The signed area is positive if the triangle is counter-clockwise and negative if the triangle is
    clockwise.
    '''
    def value(self, p):
        # transpose to be 3 x 2 x n
        p = np.transpose(np.reshape(p, (-1, 3, 2)), (1,2,0))
        # First, get the two legs...
        (dx_ab, dy_ab) = p[1] - p[0]
        (dx_ac, dy_ac) = p[2] - p[0]
        (dx_bc, dx_bc) = p[2] - p[1]
        # now, the area is half the z-value of the cross-product...
        sarea = 0.5 * (dx_ab*dy_ac - dx_ac*dy_ab)
        return sarea
    def jacobian(self, p, into=None):
        p = np.transpose(np.reshape(p, (-1, 3, 2)), (1,2,0))
        (dx_ab, dy_ab) = p[1] - p[0]
        (dx_ac, dy_ac) = p[2] - p[0]
        (dx_bc, dy_bc) = p[2] - p[1]
        z = 0.5 * np.transpose([[-dy_bc,dx_bc], [dy_ac,-dx_ac], [-dy_ab,dx_ab]], (2,0,1))
        m = numel(p)
        n = p.shape[2]
        ii = (np.arange(n) * np.ones([6, n])).T.flatten()
        z = sps.csr_matrix((z.flatten(), (ii, np.arange(len(ii)))), shape=(n, m))
        return safe_into(into, z)
def signed_face_areas(faces, axis=1):
    '''
    signed_face_areas(faces) yields a potential function f(x) that calculates the signed area of
      each face represented by the simplices matrix faces.

    If faces is None, then the parameters must arrive in the form of a flattened (n x 3 x 2) matrix
    where n is the number of triangles. Otherwise, the faces matrix must be either (n x 3) or (n x 3
    x s); if the former, each row must list the vertex indices for the faces where the vertex matrix
    is presumed to be shaped (V x 2). Alternately, faces may be a full (n x 3 x 2) simplex array of
    the indices into the parameters.

    The optional argument axis (default: 1) may be set to 0 if the faces argument is a matrix but
    the coordinate matrix will be (2 x V) instead of (V x 2).
    '''
    faces = np.asarray(faces)
    if len(faces.shape) == 2:
        if faces.shape[1] != 3: faces = faces.T
        n = 2 * (np.max(faces) + 1)
        if axis == 0: tmp = np.reshape(np.arange(n), (2,-1)).T
        else:         tmp = np.reshape(np.arange(n), (-1,2))
        faces = np.reshape(tmp[faces.flat], (-1,3,2))
    faces = faces.flatten()
    return compose(TriangleSignedArea2DPotential(), part(Ellipsis, faces))
@pimms.immutable
class TriangleArea2DPotential(PotentialFunction):
    '''
    TriangleArea2DPotential(n) yields a potential function that tracks the unsigned area of the
    given number of faces.
    '''
    def value(self, p):
        # transpose to be 3 x 2 x n
        p = np.transpose(np.reshape(p, (-1, 3, 2)), (1,2,0))
        # First, get the two legs...
        (dx_ab, dy_ab) = p[1] - p[0]
        (dx_ac, dy_ac) = p[2] - p[0]
        (dx_bc, dy_bc) = p[2] - p[1]
        # now, the area is half the z-value of the cross-product...
        sarea0 = 0.5 * (dx_ab*dy_ac - dx_ac*dy_ab)
        # but we want to abs it
        return np.abs(sarea0)
    def jacobian(self, p, into=None):
        # transpose to be 3 x 2 x n
        p = np.transpose(np.reshape(p, (-1, 3, 2)), (1,2,0))
        # First, get the two legs...
        (dx_ab, dy_ab) = p[1] - p[0]
        (dx_ac, dy_ac) = p[2] - p[0]
        (dx_bc, dy_bc) = p[2] - p[1]
        # now, the area is half the z-value of the cross-product...
        sarea0 = 0.5 * (dx_ab*dy_ac - dx_ac*dy_ab)
        # but we want to abs it
        dsarea0 = np.sign(sarea0)
        z = np.transpose([[-dy_bc,dx_bc], [dy_ac,-dx_ac], [-dy_ab,dx_ab]], (2,0,1))
        z = times(0.5*dsarea0, z)
        m = numel(p)
        n = p.shape[2]
        ii = (np.arange(n) * np.ones([6, n])).T.flatten()
        z = sps.csr_matrix((z.flatten(), (ii, np.arange(len(ii)))), shape=(n, m))
        return safe_into(into, z)
def face_areas(faces, axis=1):
    '''
    face_areas(faces) yields a potential function f(x) that calculates the unsigned area of each
      faces represented by the simplices matrix faces.

    If faces is None, then the parameters must arrive in the form of a flattened (n x 3 x 2) matrix
    where n is the number of triangles. Otherwise, the faces matrix must be either (n x 3) or (n x 3
    x s); if the former, each row must list the vertex indices for the faces where the vertex matrix
    is presumed to be shaped (V x 2). Alternately, faces may be a full (n x 3 x 2) simplex array of
    the indices into the parameters.

    The optional argument axis (default: 1) may be set to 0 if the faces argument is a matrix but
    the coordinate matrix will be (2 x V) instead of (V x 2).
    '''
    faces = np.asarray(faces)
    if len(faces.shape) == 2:
        if faces.shape[1] != 3: faces = faces.T
        n = 2 * (np.max(faces) + 1)
        if axis == 0: tmp = np.reshape(np.arange(n), (2,-1)).T
        else:         tmp = np.reshape(np.arange(n), (-1,2))
        faces = np.reshape(tmp[faces.flat], (-1,3,2))
    faces = faces.flatten()
    return compose(TriangleArea2DPotential(), part(Ellipsis, faces))
