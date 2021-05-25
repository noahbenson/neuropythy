####################################################################################################
# neuropythy/math/core.py
# This file contains useful math tools for neuropythy largely built around interfacing seamlessly
# with the PyTorch library.
# by Noah C. Benson

import numpy        as np
import scipy        as sp
import scipy.sparse as sps

# Constants ########################################################################################
pi = np.pi
half_pi = pi / 2
quarter_pi = pi / 4
tau = 2 * pi
inf = np.inf
nan = np.nan
radperdeg = pi / 180
degperrad = 180 / pi

# Importing PyTorch ################################################################################
# We want to work with pytorch but not to require it. Thus we use this function to get it.
def pytorch():
    '''
    Yields the pytorch module or raises an ImportError.
    '''
    global pytorch
    try:
        import torch
        def pytorch():
            '''
            Yields the pytorch module or raises an ImportError.
            '''
            return torch
        return torch
    except Exception: pass
    raise ImportError("failed to import torch: PyTorch may not be installed")

# General Utility Functions ########################################################################
# These are functions that are typically available in both numpy and pytorch; the function here
# provides an interface that works for both numpy and pytorch types.
def to_torchdtype(dtype):
    '''
    to_torchdtype(d) yields d if d is a PyTorch dtype, otherwise yields a dtype
      object that is equivalent to d. If d cannot be converted to a dtype object
      then an exception is raised.

    The argument d may be a numpy dtype object, a numeric type, or a string that
    names a PyTorch dtype.
    '''
    import pimms
    torch = pytorch()
    if isinstance(dtype, torch.dtype):
        return dtype
    elif pimms.is_str(dtype):
        return getattr(torch, dtype)
    elif np.issubdtype(dtype, np.generic):
        return getattr(torch, dtype.__name__)
    else:
        raise ValueError("Cannot convert to pytorch dtype: %s" % (dtype,))
def torchdtype_to_numpydtype(dtype):
    '''
    torchdtype_to_numpydtype(dtype) yields a numpy dtype equivalent to the given
      torch dtype. If dtype is None or if no matching dtype is known, None is
      returned.
    '''
    torch = pytorch()
    dtype = to_torchdtype(dtype)
    return (np.float32    if dtype == torch.float32    else 
            np.float64    if dtype == torch.float64    else
            np.int32      if dtype == torch.int32      else
            np.int64      if dtype == torch.int64      else
            np.float16    if dtype == torch.float16    else
            np.int16      if dtype == torch.int16      else
            np.int8       if dtype == torch.int8       else
            np.bool       if dtype == torch.bool       else
            np.uint8      if dtype == torch.uint8      else
            np.complex64  if dtype == torch.complex64  else
            np.complex128 if dtype == torch.complex128 else
           #np.uint64     if dtype == torch.uint64     else
           #np.uint32     if dtype == torch.uint32     else
           #np.uint16     if dtype == torch.uint16     else
            None)
def isarray(u):
    '''
    isarray(u) yields True if u is either a NumPy array or is a SciPy sparse
      matrix and yields False otherwise.

    See also: istensor(), issparse(), isdense()
    '''
    if   sps.issparse(u):           return True
    elif isinstance(u, np.ndarray): return True
    else: return False
def istensor(u):
    '''
    istensor(u) yields True if u is a PyTorch tensor and False otherwise.
    '''
    try:
        torch = pytorch()
        return torch.is_tensor(u)
    except ImportError: pass
    return False
def issparse(u):
    '''
    issparse(u) yields True if u is either a SciPy sparse matrix or if u is a
      PyTorch sparse tensor; otherwise yields False.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(u):
            return u.is_sparse
    except ImportError: pass
    return sps.issparse(u)
def isdense(u):
    '''
    isdense(u) yields True if u is either a dense NumPy array or a dense PyTorch
      tensor and False otherwise.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(u):
            return not u.is_sparse
    except ImportError: pass
    if sps.issparse(u): return False
    try: u = np.asarray(u)
    except Exception: return False
    return True
def indices_to_integers(shape, ii):
    '''
    indices_to_integers(shape, ii) yields an integer array whose values
      represent the columns of the index matrix ii (e.g., as returned by
      numpy.where or torch.where).

    The return value is always a numpy array.

    See also integers_to_indices().
    '''
    tr = np.roll(np.cumprod(shape), 1)
    tr[0] = 1
    return np.dot(tr, asarray(ii))
def integers_to_indices(shape, ii):
    '''
    integers_to_indices(shape, ii) yields an indices matrix whose columns
      represent the values of the integer vector ii (e.g., as returned by
      indices_to_integers).

    The return value is always a numpy array.

    See also integers_to_indices().
    '''
    tr = np.roll(np.cumprod(shape), 1)
    tr[0] = 1
    rr = np.zeros((len(shape), len(ii)), dtype=np.int)
    for (rrrow,shval) in zip(reversed(rr), reversed(tr)):
        rrrow[:] = np.floor_divide(ii, shval)
        ii = np.mod(ii, shval)
    return rr
def find_indices(ii, k):
    '''
    find_index(ii, k) yields a NumPy array of the indices at which the
      (integer) values of k occur in the (integer) array ii.
    '''
    n = len(ii)
    zs = np.zeros(n, dtype=np.int)
    ri = sps.csr_matrix((np.arange(n), [k, zs]))
    return np.asarray(ri[ii, zs]).flatten()
def dense_cmp_dense(f, a, b, nannan=False, nanval=False):
    '''
    dense_cmp_dense(f, a, b) yields a dense boolean numpy array of values in the
      elementwise comparison array/tensor represented by the arguments a and b.
      The value of the comparison must be f(a,b) where f is a numpy function
      such as np.equal or np.less_equal.

    The following options may be given:
      * nanval (default: False) the value returned when comparing NaN against a
        valid value.
      * nannan (default: False) the value returned when comparing NaN against
        another NaN (i.e., is NaN == NaN?).
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(a): a = a.detach().numpy()
        if torch.is_tensor(b): b = b.detach().numpy()
    except ImportError: pass
    a = np.asarray(a)
    b = np.asarray(b)
    # This is pretty straightforward!
    r = np.asarray(f(a, b), dtype=np.bool)
    # Check for NaNs
    if nannan or nanval:
        iia = np.isnan(a)
        iib = np.isnan(b)
        if nannan:
            nn = (iia & iib)
            r[nn] = nannan
        if nanval:
            nv = (iia | iib)
            r[nv] = nanval
    # That's it!
    return r
def sparse_cmp_dense(f, ij, dat, x, nannan=False, nanval=False):
    '''
    sparse_cmp_dense(f, ij, dat, x_dense) yields a dense numpy array of the
      elementwise comparison array/tensor representd by the indices and data
      values given and the dense array/tensor x_dense. The ij and dat pairs are
      the reesult of calling either the scipy.sparse.find() function (which
      returns (ij[0], ij[1], dat)) or the indices() (ij) and values() (dat)
      methods on a PyTorch sparse tensor. The value of the comparison must be
      f(a,b) where f is a numpy function such as np.equal or np.less_equal.

    The following options may be given:
      * nanval (default: False) the value returned when comparing NaN against a
        valid value.
      * nannan (default: False) the value returned when comparing NaN against
        another NaN (i.e., is NaN == NaN?).
    '''
    try:
        torch = pytorch()
        if torch.is_tensor( ij1):  ij1 =  ij1.detach().numpy()
        if torch.is_tensor(dat1): dat1 = dat1.detach().numpy()
        if torch.is_tensor(   x):    x =    x.detach().numpy()
    except ImportError: pass
    ij1 = tuple(np.asarray(ij1))
    dat1 = np.asarray(dat1)
    x = np.asarray(x)
    # We start by making a matrix that compares the sparse value to x.
    r = dense_cmp_dense(f, 0, x, nannan=nannan, nanval=nanval)
    # Next, we grab the specific elements with non-zero comparisons to be made.
    xij = x[ij]
    # Compare these...
    rij = dense_cmp_dense(f, dat1, xij, nannan=nannan, nanval=nanval)
    # Add them into the result array.
    r[ij] = rij
    # That's all that is needed.
    return r
def dense_cmp_sparse(f, x, ij, dat, nannan=False, nanval=False):
    '''
    dense_cmp_sparse(f, x_dense, ij, dat) yields a dense numpy array of the
      elementwise comparison array/tensor representd by the indices and data
      values given and the dense array/tensor x_dense. The ij and dat pairs are
      the reesult of calling either the scipy.sparse.find() function (which
      returns (ij[0], ij[1], dat)) or the indices() (ij) and values() (dat)
      methods on a PyTorch sparse tensor. The value of the comparison must be
      f(a,b) where f is a numpy function such as np.equal or np.less_equal.

    The following options may be given:
      * nanval (default: False) the value returned when comparing NaN against a
        valid value.
      * nannan (default: False) the value returned when comparing NaN against
        another NaN (i.e., is NaN == NaN?).
    '''
    try:
        torch = pytorch()
        if torch.is_tensor( ij1):  ij1 =  ij1.detach().numpy()
        if torch.is_tensor(dat1): dat1 = dat1.detach().numpy()
        if torch.is_tensor(   x):    x =    x.detach().numpy()
    except ImportError: pass
    ij1 = tuple(np.asarray(ij1))
    dat1 = np.asarray(dat1)
    x = np.asarray(x)
    # We start by making a matrix that compares the sparse value to x.
    r = dense_cmp_dense(f, x, 0, nannan=nannan, nanval=nanval)
    # Next, we grab the specific elements with non-zero comparisons to be made.
    xij = x[ij]
    # Compare these... (the dense_cmp_dense function handles the NaNs).
    rij = dense_cmp_dense(f, xij, dat1, nannan=nannan, nanval=nanval)
    # Add them into the result array.
    r[ij] = rij
    # That's all that is needed.
    return r
def sparse_cmp_sparse(f, ij1, dat1, ij2, dat2, shape,
                      nannan=False, nanval=False, backend=sps):
    '''
    sparse_cmp_sparse(cmp ij1, dat1, ij2, dat2) yields an array of comparisons
      in the elementwise equality array/tensor representd by the indices and
      data values given. The ij and dat pairs are the reesult of calling either
      the scipy.sparse.find() function (which returns (ij[0], ij[1], dat)) or
      the indices() (ij) and values() (dat) methods on a PyTorch sparse tensor.
      The value of the comparison must be f(a,b) where f is a numpy function
      such as np.equal or np.less_equal.

    The following options may be given:
      * nanval (default: False) the value returned when comparing NaN against a
        valid value.
      * nannan (default: False) the value returned when comparing NaN against
        another NaN (i.e., is NaN == NaN?).
      * backend (default: None) the backend library to use for the resulting
        comparison array; should be either torch (PyTorch) or scipy.sparse.
    '''
    # There's no reason not to use numpy arrays; we want to detach torch arrays
    # anyway.
    try:
        torch = pytorch()
        if torch.is_tensor( ij1):  ij1 =  ij1.detach().numpy()
        if torch.is_tensor(dat1): dat1 = dat1.detach().numpy()
        if torch.is_tensor( ij2):  ij2 =  ij2.detach().numpy()
        if torch.is_tensor(dat2): dat2 = dat2.detach().numpy()
    except ImportError: pass
    ij1  = np.asarray(ij1)
    dat1 = np.asarray(dat1)
    ij2  = np.asarray(ij2)
    dat2 = np.asarray(dat2)
    if ij1.shape[0] != ij2.shape[0]:
        raise ValueError('sparse arrays must have equal dimensionality to be compared')
    # Convert the indices into integers by linearizing the cartesian indices.
    tr = np.roll(np.cumprod(shape), 1)
    tr[0] = 1
    k1 = np.dot(tr, ij1)
    k2 = np.dot(tr, ij2)
    # We also need two sparse matrices that reverse these indices.
    ri1 = sps.csr_matrix((np.arange(len(k1)), [k1, np.zeros(len(k1), dtype=np.int)]))
    ri2 = sps.csr_matrix((np.arange(len(k2)), [k2, np.zeros(len(k2), dtype=np.int)]))
    # Figure out which keys get compared with each other versus with 0.
    k_cmp12 = np.intersect1d(k1, k2)
    k_cmp10 = np.setdiff1d(k1, k_cmp12)
    k_cmp02 = np.setdiff1d(k2, k_cmp12)
    # Make the comparisons; regarding NaNs: we pass these values down to the dense_cmp_dense
    # function, so it handles the NaNs for us.
    if len(k_cmp12) > 0:
        zs = np.zeros(len(k_cmp12), dtype=np.int)
        ii1 = np.asarray(ri1[k_cmp12, zs]).flatten()
        ii2 = np.asarray(ri2[k_cmp12, zs]).flatten()
        r_cmp12 = dense_cmp_dense(f, dat1[ii1], dat2[ii2], nannan=nannan, nanval=nanval)
    else:
        r_cmp12 = []
    if len(k_cmp10) > 0:
        ii1 = np.asarray(ri1[k_cmp10, np.zeros(len(k_cmp10), dtype=np.int)]).flatten()
        r_cmp10 = dense_cmp_dense(f, dat1[ii1], 0, nannan=nannan, nanval=nanval)
    else:
        r_cmp10 = []
    if len(k_cmp02) > 0:
        ii2 = np.asarray(ri2[k_cmp02, np.zeros(len(k_cmp02), dtype=np.int)]).flatten()
        r_cmp02 = dense_cmp_dense(f, 0, dat2[ii2], nannan=nannan, nanval=nanval)
    else:
        r_cmp02 = []
    # Create the output values.
    k = np.concatenate([k_cmp12, k_cmp10, k_cmp02])
    ij = np.zeros((len(shape), len(k)), dtype=np.int)
    for (ijrow,shval) in zip(reversed(ij), reversed(tr)):
        ijrow[:] = np.floor_divide(k, shval)
        k = np.mod(k, shval)
    v = np.concatenate([rr for rr in [r_cmp12, r_cmp10, r_cmp02] if len(rr) > 0])
    # We will return a dense or sparse matrix depending on how the sparse values compare.
    if bool(f(0, 0)):
        # We need to return a bunch of ones with the tallied indices set.
        if backend is sps:
            r = np.ones(shape, dtype=np.bool)
            r[tuple(ij)] = v
        else:
            r = torch.ones(shape, dtype=torch.bool)
            r[tuple(ij)] = torch.tensor(v, dtype=torch.bool)
        return r
    else:
        # We return a sparse array of the appropriate type; we can ignore False values (recall that
        # v is a boolean mask already).
        if backend is sps:
            return sps.csr_matrix((v[v], ij[:,v]), shape=shape, dtype=np.bool)
        else:
            return backend.sparse_coo_tensor(ik[:,v], v[v], shape, dtype=backend.bool)
def cmp(f, a, b, nannan=False, nanval=False):
    '''
    cmp(f, a, b) yields f(a, b) as an elementwise comparison of a and b. The
      function f must be a predicate function that works on NumPy arrays, such
      as np.equal or np.less_equal.

    The cmp function works on all NumPy arrays, NumPy-compatible objects (like
    lists and numbers), PyTorch tensors, and SciPy sparse matrices. Note that
    in the case of PyTorch tensors, the result of a cmp call will always be
    detached from the inputs.

    The following options may be given:
      * nanval (default: False) the value returned when comparing NaN against a
        valid value.
      * nannan (default: False) the value returned when comparing NaN against
        another NaN (i.e., is NaN == NaN?).
    '''
    (ash,bsh) = (np.shape(a), np.shape(b))
    shape = ash if len(ash) > len(bsh) else bsh
    try:
        torch = pytorch()
        r = None
        if torch.is_tensor(a):
            if a.is_sparse:
                if not a.is_coalesced(): a = a.coalesce()
                if torch.is_tensor(b):
                    if b.is_sparse:
                        if not b.is_coalesced(): b = b.coalesce()
                        return sparse_cmp_sparse(f, a.indices(), a.values(),
                                                 b.indices(), b.values(), shape,
                                                 nannan=nannan, nanval=nanval,
                                                 backend=torch)
                    else:
                        r = sparse_cmp_dense(f, a.indices(), a.values(), b,
                                             nannan=nannan, nanval=nanval)
                elif sps.issparse(b):
                    (iib,jjb,datb) = sps.find(b)
                    return sparse_cmp_sparse(f, a.indices(), a.values(),
                                             (iib,jjb), datb, shape,
                                             nannan=nannan, nanval=nanval,
                                             backend=torch)
                else:
                    r = sparse_cmp_dense(f, a.indices(), a.values(), b,
                                         nannan=nannan, nanval=nanval)
            elif b.is_sparse:
                if not b.is_coalesced(): b = b.coalesce()
                r = dense_cmp_sparse(f, a, b.indices(), b.values(),
                                     nannan=nannan, nanval=nanval)
            else:
                r = dense_cmp_dense(f, a, b, nannan=nannan, nanval=nanval)
        elif torch.is_tensor(b):
            if b.is_sparse:
                if not b.is_coalesced(): b = b.coalesce()
                if sps.issparse(a):
                    (iia,jja,data) = sps.find(a)
                    return sparse_cmp_sparse(f, (iia,jja), data,
                                             b.indices(), b.values(), shape,
                                             nannan=nannan, nanval=nanval,
                                             backend=torch)
                else:
                    r = dense_cmp_sparse(f, a, b.indices(), b.values(),
                                         nannan=nannan, nanval=nanval)
            elif sps.issparse(a):
                (iia,jja,data) = sps.find(a)
                r = sparse_cmp_dense(f, (iia,jja), data, b,
                                     nannan=nannan, nanval=nanval)
            else:
                r = dense_cmp_dense(f, a, b, nannan=nannan, nanval=nanval)
        if r is not None: return torch.tensor(r)
    except ImportError: pass
    if sps.issparse(a):
        (iia,jja,data) = sps.find(a)
        if sps.issparse(b):
            (iib,jjb,datb) = sps.find(b)
            return sparse_cmp_sparse(f, (iia,jja), data, (iib,jjb), datb, shape,
                                     nannan=nannan, nanval=nanval,
                                     backend=sps)
        else:
            return sparse_cmp_dense(f, (iia,jja), data, b,
                                     nannan=nannan, nanval=nanval)
    elif sps.issparse(b):
        (iib,jjb,datb) = sps.find(b)
        return dense_cmp_sparse(f, a, (iib,jjb), datb,
                                nannan=nannan, nanval=nanval)
    else:
        return dense_cmp_dense(f, a, b, nannan=nannan, nanval=nanval)        
def eq(a, b, nannan=False, nanval=False):
    '''
    eq(a, b) yields (a == b) as an elementwise comparison of a and b.

    The arguments a and b can be NumPy-compatible arrays/numbers/lists, SciPy
    sparse matrices, or PyTorch tensors.

    The following options may be given:
      * nanval (default: False) the value returned when comparing NaN against a
        valid value.
      * nannan (default: False) the value returned when comparing NaN against
        another NaN (i.e., is NaN == NaN?).
    '''
    return cmp(np.equal, a, b, nannan=nannan, nanval=nanval)
def ne(a, b, nannan=True, nanval=True):
    '''
    ne(a, b) yields (a != b) as an elementwise comparison of a and b.

    The arguments a and b can be NumPy-compatible arrays/numbers/lists, SciPy
    sparse matrices, or PyTorch tensors.

    The following options may be given:
      * nanval (default: True) the value returned when comparing NaN against a
        valid value.
      * nannan (default: True) the value returned when comparing NaN against
        another NaN (i.e., is NaN == NaN?).
    '''
    return cmp(np.not_equal, a, b, nannan=nannan, nanval=nanval)
def lt(a, b, nannan=False, nanval=False):
    '''
    lt(a, b) yields (a < b) as an elementwise comparison of a and b.

    The arguments a and b can be NumPy-compatible arrays/numbers/lists, SciPy
    sparse matrices, or PyTorch tensors.

    The following options may be given:
      * nanval (default: False) the value returned when comparing NaN against a
        valid value.
      * nannan (default: False) the value returned when comparing NaN against
        another NaN (i.e., is NaN == NaN?).
    '''
    return cmp(np.less, a, b, nannan=nannan, nanval=nanval)
def gt(a, b, nannan=False, nanval=False):
    '''
    gt(a, b) yields (a > b) as an elementwise comparison of a and b.

    The arguments a and b can be NumPy-compatible arrays/numbers/lists, SciPy
    sparse matrices, or PyTorch tensors.

    The following options may be given:
      * nanval (default: False) the value returned when comparing NaN against a
        valid value.
      * nannan (default: False) the value returned when comparing NaN against
        another NaN (i.e., is NaN == NaN?).
    '''
    return cmp(np.greater, a, b, nannan=nannan, nanval=nanval)
def le(a, b, nannan=False, nanval=False):
    '''
    le(a, b) yields (a <= b) as an elementwise comparison of a and b.

    The arguments a and b can be NumPy-compatible arrays/numbers/lists, SciPy
    sparse matrices, or PyTorch tensors.

    The following options may be given:
      * nanval (default: False) the value returned when comparing NaN against a
        valid value.
      * nannan (default: False) the value returned when comparing NaN against
        another NaN (i.e., is NaN == NaN?).
    '''
    return cmp(np.less_equal, a, b, nannan=nannan, nanval=nanval)
def ge(a, b, nannan=False, nanval=False):
    '''
    ge(a, b) yields (a >= b) as an elementwise comparison of a and b.

    The arguments a and b can be NumPy-compatible arrays/numbers/lists, SciPy
    sparse matrices, or PyTorch tensors.

    The following options may be given:
      * nanval (default: False) the value returned when comparing NaN against a
        valid value.
      * nannan (default: False) the value returned when comparing NaN against
        another NaN (i.e., is NaN == NaN?).
    '''
    return cmp(np.greater_equal, a, b, nannan=nannan, nanval=nanval)

# Copying and converting tensors/arrays between types ##############################################
def clone(u, **kw):
    '''
    clone(u) yields a copy of u. If u is a pytorch tensor, then this copy is
      not detached.

    If the device keyword argument is passed, then the result is always a
    PyTorch tensor.
    '''
    hasdev = 'device' in kw
    try:
        torch = pytorch()
        if torch.is_tensor(u) or hasdev:
            return totensor(u, **kw)
    except ImportError: pass
    if hasdev:
        raise ValueError("device given but failed to import PyTorch--make sure it is installed")
    if sps.issparse(u):
        return u.copy(**kw)
    else:
        return np.array(u, **kw)
def totensor(u, **kw):
    '''
    totensor(u) yields a copy of u as a PyTorch tensor. Keyword arguments to the
      tensor function may be passed to totensor(). If u is a tensor, a copy of u
      is always returned. Note that u may be a scipy sparse array, in which case
      it is converted into an uncoalesced sparse COO tensor.

    The tensors returned by totensor() are always detached.

    All optional keywords are passed to the torch.tensor() function; however
    the dtype argument is handled specially: if dtype is a string, then the
    dtype that is passed to tensor is getattr(torch, dtype) instead of dtype
    ifself.
    '''
    torch = pytorch()
    dtype = kw.pop('dtype', None)
    if dtype is None:
        npdtype = None
    else:
        dtype = to_torchdtype(dtype)
        npdtype = torchdtype_to_numpydtype(dtype)
    if torch.is_tensor(u):
        if len(kw) == 0 and (dtype is None or dtype == u.dtype):
            return u.clone().detach()
        if u.is_sparse:
            u = u.coalesce()
            # We just detach the indices and values
            v = torch.sparse_coo_tensor(u.indices().clone().detach(), u.values().clone().detach(),
                                        u.shape, dtype=dtype, **kw)
            return v
        # The easiest way to proceed with a dense pytorch tensor is to just start with the
        # numpy array and proceed as usual.
        u = u.detach().numpy()
    # If we have a scipy sparse tensor, we treat this all differently
    if sps.issparse(u):
        (ii,jj,xx) = sps.find(u)
        if npdtype is not None: xx = np.asarray(xx, dtype=npdtype)
        return torch.sparse_coo_tensor([ii,jj], xx, dtype=dtype, **kw)
    else:
        u = np.array(u, dtype=npdtype)
        return torch.tensor(u, dtype=dtype, **kw)
def toarray(u, **kw):
    '''
    toarray(u) yields a copy of u that is either a NumPy array or a SciPy sparse
      matrix that is equivalent to the given array u. This function always
      makes a copy of the given object.
    '''
    dt = kw.pop('dtype', None)
    if dt is not None: kw['dtype'] = to_torchdtype(dt)
    try:
        torch = pytorch()
        if torch.is_tensor(u):
            if u.is_sparse:
                if not u.is_coalesced(): u = u.coalesce()
                ((ii,jj), xx) = (u.indices().detach().numpy(), u.values())
                xx = np.array(xx.detach().numpy(), dtype=dt)
                return sps.coo_matrix((xx, (ii,jj)), **kw)
            u = u.detach().numpy()
    except ImportError: pass
    if sps.issparse(u):
        if len(kw) == 0: return u.copy()
        (ii,jj,xx) = sps.find(u)
        xx = np.array(xx.detach().numpy(), dtype=dt)
        return sps.coo_matrix((xx, (ii,jj)), **kw)
    else:
        return np.array(u, **kw)        
def astensor(u, **kw):
    '''
    astensor(u) yields u if u is a PyTorch tensor and yields a PyTorch-tensor
      version of u otherwise. This is similar to totensor() except that totensor()
      always makes a copy, while astensor() only makes a copy if it must.

    Any keyword arguments for the torch.tensor() function are respected and can
    result in a copy of the tensor being made if they differ from those of the
    torch.tensor() function.
    '''
    torch = pytorch()
    if torch.is_tensor(u):
        # We return u if it is a tensor that has an appropriate dtype and device
        if len(kw) == 0: return u
        if 'dtype' in kw:
            dt = kw['dtype']
            if dt is not None: kw['dtype'] = to_torchdtype(dt)
        return torch.as_tensor(u, **kw)
    else:
        return totensor(u, **kw)
def asarray(u, **kw):
    '''
    asarray(u) yields u if u is a Numpy array; otherwise yields a version of u
      as a Numpy array. A copy of u will not be made unless such a copy is
      required.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(u):
            if u.is_sparse:
                if not u.is_coalesced(): u = u.coalesce()
                (ij,uu) = (u.indices(), u.values())
                uu = uu.detach().numpy()
                if len(kw) > 0: uu = np.asarray(uu.detach().numpy(), **kw)
                return sps.coo_matrix((uu, ij.detach().numpy()), u.shape)
            return u.detach().numpy()
    except ImportError: pass
    if sps.issparse(u):
        (ii,jj,uu) = sps.find(u)
        if len(kw) > 0: uu = np.asarray(uu, **kw)
        return torch.sparse_coo_matrix([ii,jj], uu)
    else:
        return np.asarray(u, **kw)
def asdense(u):
    '''
    asdense(u) yields u if u is a dense (non-sparse) array or tensor, asarray(u)
      if u is not an array, scipy matrix, or tensor, and a dense representation
      of u if it is sparse.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(u):
            return u.to_dense() if u.is_sparse else u
    except ImportError: pass
    if sps.issparse(u): return u.toarray()
    else: return np.asarray(u)
def todtype(u, dtype):
    '''
    todtype(u, dtype) yields a a copy of PyTorch tensor or NumPy array u with
      the new dtype. The argument u may also be a SciPy sparse array.

    A detached copy of u is always returned. To yield the array itself in cases
    where this is possible, use asdtype(u, dtype).
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(u):
            if u.is_sparse:
                u = u.coalesce()
                # We just detach the indices and values
                vals = todtype(u.values(), dtype)
                return torch.sparse_coo_tensor(u.indices().clone().detach(), vals,
                                               u.shape, dtype=dtype, device=u.device)
            else:
                dtype = to_torchdtype(dtype)
                npdtype = torchdtype_to_numpydtype(dtype)
                return torch.tensor(todtype(u.detach().numpy(), npdtype),
                                    dtype=dtype, device=u.device)
    except ImportError: pass
    if sps.issparse(u):
        (ii,jj,xx) = sps.find(u)
        xx = np.array(xx.detach().numpy(), dtype=dtype)
        return sps.coo_matrix((xx, (ii,jj)), **kw)
    else:
        return np.array(u, dtype=dtype)
def asdtype(u, dtype):
    '''
    asdtype(u, dtype) yields either a copy of the PyTorch tensor or NumPy array
      u with the new dtype, if the dtype is not equivalent to u.dtype, or yields
      u itself if the dtype already matches. The argument u may also be a SciPy
      sparse array.

    A copy of u is made only if necessary, and it is not detached unless
    necessary. To always obtain a detached copy, you can use the todtype()
    function.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(u):
            dtype = to_torchdtype(dtype)
            if u.dtype == dtype: return u
            if u.is_sparse:
                u = u.coalesce()
                # We just detach the indices and values
                vals = u.values().type(dtype)
                return torch.sparse_coo_tensor(u.indices().clone().detach(), vals,
                                               u.shape, dtype=dtype, device=u.device)
            else:
                npdtype = torchdtype_to_numpydtype(dtype)
                return torch.tensor(asdtype(u.detach().numpy(), npdtype),
                                    dtype=dtype, device=u.device)
    except ImportError: pass
    dtype = np.dtype(dtype)
    if sps.issparse(u):
        if u.dtype == dtype: return u
        (ii,jj,xx) = sps.find(u)
        xx = np.array(xx.detach().numpy(), dtype=dtype)
        return sps.coo_matrix((xx, (ii,jj)), **kw)
    elif np.dtype(u) == dtype:
        return u
    else:
        return np.asarray(u).astype(dtype)
def promote(*args, **kw):
    '''
    promote(a, b) yields a tuple (A, B) where A and B are both the same
      type of array. If either a or b is a PyTorch tensor, then both A and B
      will be PyTorch tensors. Otherwise, they will both be numpy arrays.
    promote(a, b, c...) yields a tuple of all the arguments promoted to
      the appropriate class.

    The optional argument copy (default: False) may be set to true to indicate
    that copies should be made. In this case, copies are made of all arguments
    even if nothing is promoted.
    '''
    copy = kw.pop('copy', False)
    # First, figure out what the promotion type is:
    try:
        torch = pytorch()
        if any(torch.is_tensor(u) for u in args):
            (tof, asf) = (totensor, astensor)
        else:
            (tof, asf) = (toarray, asarray)
    except ImportError:
        (tof, asf) = (toarray, asarray)
    if copy: return tuple([tof(u, **kw) for u in args])
    else:    return tuple([asf(u, **kw) for u in args])
def reshape(u, shape):
    '''
    reshape(u, new_shape) is equivalent to NumPy or PyTorch's reshape functions
      and works on both NumPy arrays and PyTorch tensors as well as SciPy sparse
      matrices.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(u): return torch.reshape(u, shape)
    except ImportError: pass
    if sps.issparse(u):
        return u.reshape(shape)
    else:
        return np.reshape(u, shape)

# Access Functions #################################################################################
def extract(u, ii):
    '''
    extract(u, ii) yields an array or tensor of the elements indexed by the
      integer, vector, or matrix ii from the NumPy array, SciPy matrix, or
      PyTorch tensor u.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(u):
            if u.is_sparse:
                u = u.coalesce()
                ki = indices_to_integers(u.shape, ii)
                ku = indices_to_integers(u.shape, u.indices())
                kk = np.intersect1d(ki, ku)
                r = torch.zeros(len(ki), dtype=u.dtype, device=u.device)
                r[find_index(ki, kk)] = u.values()[find_index(ku, kk)]
                return r
            else:
                return u[tuple(ii)]
        if torch.is_tensor(ii): ii = ii.detach().numpy()
    except ImportError: pass
    if sps.issparse(u):
        return np.asarray(u[tuple(ii)]).flatten()
    else:
        return np.asarray(u)[tuple(ii)]

# Mathematical Functions ###########################################################################
# These two functions make it easier to handle both PyTorch and Numpy arrays in either unary or
# binary math functioins.
def unary_handle(torch_fn, numpy_fn, zeroval, u, **kw):
    '''
    Handles the unary dispatch of the function represented by torch_fn and
    numpy_Fn for the given object u and the keyword arguments, which are
    passed along to the associated functions.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(u):
            if u.is_sparse:
                if not u.is_coalesced(): u = u.coalesce()
                ((ii,jj), uu) = (u.indices(), u.values())
                if zeroval == 0:
                    vv = torch_fn(uu, **kw)
                    return torch.sparse_coo_matrix((ii,jj), vv, dtype=u.dtype, device=u.device)
                else:
                    r = torch.full(u.shape, zeroval, dtype=u.dtype, device=u.device)
                    r[ii,jj] = torch_fn(uu, **kw)
                    return r
            else:
                return torch_fn(u, **kw)
    except ImportError: pass
    if sps.issparse(u):
        if zeroval == 0:
            r = u.copy()
            r.data[:] = numpy_fn(r.data, **kw)
        else:
            r = np.full(u.shap, zeroval, dtype=u.dtype)
            (ii,jj,uu) = sps.find(u)
            r[ii,jj] = numpy_fn(uu, **kw)
        return r
    else:
        return numpy_fn(u, **kw)
def binary_handle(torch_fn, numpy_fn, a, b, **kw):
    '''
    Handles binary operators.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(a):
            if a.is_sparse: a = a.to_dense()
            if torch.is_tensor(b):
                if b.is_sparse: b = b.to_dense()
            else:
                b = totensor(b)
            return torch_fn(a, b, **kw)
        elif torch.is_tensor(b):
            a = totensor(a)
            return torch_fn(a, b, **kw)
    except ImportError: pass
    if sps.issparse(a): a = a.toarray()
    if sps.issparse(b): b = b.toarray()
    return numpy_fn(a, b, **kw)
def unary_mathfn(numpy_fn):
    '''
    @unary_mathfn(numpy_fn) is a decorator that allows the PyTorch-specific
    function that follows it to be compatible with PyTorch sprase arrays,
    scipy sparse arrays, numpy arrays, python lists, and numbers.
    '''
    import functools, warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zeroval = numpy_fn(0.0)
    def wrap1(torch_fn):
        @functools.wraps(torch_fn)
        def wrapper(u, **kw):
            return unary_handle(torch_fn, numpy_fn, zeroval, u, **kw)
        return wrapper
    return wrap1
def binary_mathfn(numpy_fn):
    '''
    @binary_mathfn(numpy_fn) is a decorator that allows the PyTorch-specific
    function that follows it to be compatible with PyTorch sprase arrays,
    scipy sparse arrays, numpy arrays, python lists, and numbers.
    '''
    import functools
    def wrap1(torch_fn):
        @functools.wraps(torch_fn)
        def wrapper(u, **kw):
            return binary_handle(torch_fn, numpy_fn, u, **kw)
        return wrapper
    return wrap1
def summate(a, b):
    '''
    summate(a, b) yields a + b and works correctly for NumPy arrays, PyTorch
      tensors (sparse and dense), SciPy sparse matrices, and and general numbers
      and lists.

    Generally speaking, the add() function is preferred.
    '''
    (a, b) = promote(a, b)
    try:
        torch = pytorch()
        if torch.is_tensor(a):
            if a.is_sparse:
                if b.is_sparse:
                    return a + b
                else:
                    (b, a) = (a, b)
            if b.is_sparse:
                (ijb, xb) = (tuple(b.indices()), a.values())
                a = a + torch.zeros((), dtype=b.dtype, device=b.device)
                a[ijb] += xb
                return a
            else:
                return a + b
    except ImportError: pass
    if sps.issparse(a) or sps.issparse(b):
        return np.asarray(a + b)
    else:
        return a + b
def add(*args):
    '''
    add(a, b, ...) yields the sum of all the given values. This is equivalent to
      (a + b + ...) except that it works for NumPy arrays, SciPy sparse
      matrices, and PyTorch tensors (including sparse tensors).
    '''
    from functools import reduce
    if   len(args) == 0: return 0
    elif len(args) == 1: return args[0]
    else:                return reduce(summate, *args)
def sub(a, b):
    '''
    sub(a, b) yields the a minus b. This is equivalent to (a - b) except
      that it works for NumPy arrays, SciPy sparse matrices, and PyTorch tensors
      (including sparse tensors).
    '''
    (a, b) = promote(a, b)
    try:
        torch = pytorch()
        if torch.is_tensor(a):
            if a.is_sparse and not b.is_sparse: a = a.to_dense()
            return a - b
    except ImportError: pass
    if sps.issparse(a) or sps.issparse(b):
        return np.asarray(a - b)
    else:
        return a - b
def multiply(a, b):
    '''
    multiply(a, b) yields a * b and works correctly for NumPy arrays, PyTorch
      tensors (sparse and dense), SciPy sparse matrices, and and general numbers
      and lists.
    '''
    (a, b) = promote(a, b)
    try:
        torch = pytorch()
        if torch.is_tensor(a):
            if a.is_sparse:
                if b.is_sparse:
                    return a * b
                else:
                    (ija, xa) = (a.indices(), a.values())
                    xb = b[tuple(ija)]
                    xa = xa * xb
                    nans = torch.where(~torch.isfinite(b))
                    if len(nans[0]) > 0:
                        ija = torch.stack([torch.cat(ij) for ij in zip(ija, nans)])
                        nans = torch.full(xa.shape, np.nan, dtype=xa.dtype, device=xa.device)
                        xa = torch.cat((xa, nans))
                    return torch.sparse_coo_tensor(ija, xa, dtype=xa.dtype, device=xa.device)
            elif b.is_sparse:
                (ijb, xb) = (b.indices(), a.values())
                xa = a[tuple(ijb)]
                xb = xa * xb
                nans = torch.where(~torch.isfinite(a))
                if len(nans[0]) > 0:
                    ijb = torch.stack([torch.cat(ij) for ij in zip(ijb, nans)])
                    nans = torch.full(xa.shape, np.nan, dtype=xb.dtype, device=xb.device)
                    xa = torch.cat((xb, nans))
                return torch.sparse_coo_tensor(ijb, xb, dtype=xb.dtype, device=xb.device)
            else:
                return a * b
    except ImportError: pass
    if sps.issparse(a):
        return a.multiply(b)
    elif sps.issparse(b):
        return b.multiply(a)
    else:
        return a * b
def mul(*args):
    '''
    mul(a, b, ...) yields the product of all the given values. This is
      equivalent to (a * b * ...) except that it works for NumPy arrays, SciPy
      sparse matrices, and PyTorch tensors (including sparse tensors).
    '''
    from functools import reduce
    if   len(args) == 0: return 1
    elif len(args) == 1: return args[0]
    else:                return reduce(multiply, *args)
def div(a, b):
    '''
    div(a, b) yields the a divided by b. This is equivalent to (a / b) except
      that it works for NumPy arrays, SciPy sparse matrices, and PyTorch tensors
      (including sparse tensors).
    '''
    (a, b) = promote(a, b)
    try:
        torch = pytorch()
        if torch.is_tensor(a):
            if a.is_sparse: a = a.to_dense()
            if b.is_sparse: b = b.to_dense()
            return a / b
    except ImportError: pass
    if sps.issparse(a):
        return np.asarray(a / b)
    elif sps.issparse(b):
        return (a / b.toarray())
    else:
        return a / b
def mod(a, b):
    '''
    mod(a, b) yields a mod b. The arguments may be PyTorch tensors, numpy
      arrays, SciPy sparse matrices, or primitives. The PyTorch function used is
      remainder.
    '''
    (a,b) = promote(a, b)
    try:
        torch = pytorch()
        if torch.is_tensor(a):
            if a.is_sparse: a = a.to_dense()
            if b.is_sparse: b = b.to_dense()
            return torch.remainder(a, b)
    except ImportError: pass
    if sps.issparse(a): a = asdense(a)
    if sps.issparse(b): b = asdense(b)
    return np.mod(a, b)
def _numpy_safesqrt(x):
    u = np.array(x)
    ii = u > 0
    u[u < 0] = x
    s = u[ii]
    s[:] = np.sqrt(s)
    return u
@unary_mathfn(_numpy_safesqrt)
def safesqrt(u, x=0):
    '''
    safesqrt(u, x) is equivalent to torch.sqrt(u) but only operates on values
      that are greater than or equal to 0; elsewhere the values returned is x.
    safesqrt(u) is equivalent to safesqrt(u, 0).
    '''
    torch = pytorch()
    r = torch.full(u.shape, x, dtype=u.dtype, device=u.device)
    ii = u > 0
    r[ii] = torch.sqrt(u[ii])
    return r
@unary_mathfn(np.sqrt)
def sqrt(u):
    '''
    sqrt(u) is equivalent to torch.sqrt(u) or np.sqrt(u) depending on the type
      of u.
    '''
    torch = pytorch()
    return torch.sqrt(u)
@unary_mathfn(np.exp)
def exp(x):
    '''
    exp(x) yields the exponential of x, and works for either PyTorch tensors or
      for numpy arrays.
    '''
    torch = pytorch()
    return torch.exp(x)
@unary_mathfn(np.log)
def log(x):
    '''
    log(x) yields the logarithm of x, and works for either PyTorch tensors or
      for numpy arrays.
    '''
    torch = pytorch()
    return torch.log(x)
@unary_mathfn(np.log10)
def log10(x):
    '''
    log10(x) yields the base-10 logarithm of x, and works for either PyTorch
      tensors or for numpy arrays.
    '''
    torch = pytorch()
    return torch.log10(x)
@unary_mathfn(np.log2)
def log2(x):
    '''
    log2(x) yields the base-2 logarithm of x, and works for either PyTorch
      tensors or for numpy arrays.
    '''
    torch = pytorch()
    return torch.log2(x)
@unary_mathfn(np.abs)
def abs(x):
    '''
    abs(x) yields the absolute value of x, and works for either PyTorch tensors
      or for numpy arrays.
    '''
    torch = pytorch()
    return torch.abs(x)
@unary_mathfn(np.arcsin)
def arcsin(x):
    '''
    arcsin(x) is equivalent to torch.asin(x) or numpy.arcsin(x) depending on
      whether x is a pytorch tensor or not.
    '''
    torch = pytorch()
    return torch.asin(x)
@unary_mathfn(np.arccos)
def arccos(x):
    '''
    arccos(x) is equivalent to torch.arccos(x) or numpy.arccos(x) depending on
      whether x is a pytorch tensor or not.
    '''
    torch = pytorch()
    return torch.acos(x)
@unary_mathfn(np.sin)
def sin(x):
    '''
    sin(x) is equivalent to torch.sin(x) or numpy.sin(x) depending on whether x
      is a pytorch tensor or not.
    '''
    torch = pytorch()
    return torch.sin(x)
@unary_mathfn(np.cos)
def cos(x):
    '''
    cos(x) is equivalent to torch.cos(x) or numpy.cos(x) depending on whether x
      is a pytorch tensor or not.
    '''
    torch = pytorch()
    return torch.cos(x)
@unary_mathfn(np.cos)
def tan(x):
    '''
    tan(x) is equivalent to torch.tan(x) or numpy.tan(x) depending on whether x
      is a pytorch tensor or not.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(x): return torch.tan(x)
    except ImportError: pass
    return np.tan(x)
@unary_mathfn(np.cos)
def lgamma(x):
    '''
    lgamma(x) yields the log-gamma of x and is equivalent to torch.lgamma(x) or
      scipy.special.loggamma(x) depending on whether x is a pytorch tensor or
      not.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(x): return torch.lgamma(x)
    except ImportError: pass
    return sp.special.loggamma(x)
@unary_mathfn(np.arctan)
def atan(y):
    '''
    atan(u) is equivalent to np.arctan(u) or torch.atan(u) depending on the type of u.
    '''
    torch = pytorch()
    if y.is_sparse:
        (ij,x) = (y.indices(), y.values())
        return torch.sparse_coo_tensor(ij, torch.atan(x), dtype=y.dtype, device=y.device)
    else:
        return torch.atan(y)
def atan2(y, x):
    '''
    atan2(y, x) is equivalent to np.arctan2(y,x) or torch.atan2(y,x) depending on the
      types of y and x.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(y) or torch.is_tensor(x):
            if not torch.is_tensor(y): y = astensor(y)
            if not torch.is_tensor(x): x = astensor(x)
            if y.is_sparse:
                ((ii,jj),yy) = (y.indices(), y.values())
                xx = x[ii,jj]
                return torch.sparse_coo_tensor((ii,jj), torch.atan2(yy,xx),
                                               dtype=y.dtype, device=y.device)
            elif x.is_sparse:
                return torch.atan2(y, x.to_dense())
            else:
                return torch.atan2(y, x)
    except ImportError: pass
    if sps.issparse(y):
        (ii,jj,yy) = sps.find(y)
        if sps.issparse(x): xx = np.asarray(x[ii,jj]).flatten()
        else:
            x = np.asarray(x)
            nsh = len(x.shape)
            if   nsh == 0: xx = x
            elif nsh == 1: xx = x[jj]
            else:          xx = x[...,ii,jj]
        r = y.copy()
        r[ii,jj] = np.arctan(yy, xx)
        return r
    elif sps.issparse(x):
        y = np.asarray(y)
        nsh = len(y.shape)
        r = np.arctan(y, x.dtype.type(0))
        (ii,jj,xx) = sps.find(x)
        if   nsh == 0: yy = y
        elif nsh == 1: yy = y[jj]
        else:          yy = y[...,ii,jj]
        r[ii,jj] = np.arctan(yy, xx)
        return r
    else:
        return np.arctan2(y, x)
def arctan(y, x=None):
    '''
    arctan(y, x) is equivalent to torch.atan2(y, x) or numpy.arctan2(y, x)
      depending on whether y and x are pytorch tensors or not.
    arctan(y) is equvalent to torch.atan(y) or numpy.arctan(y).
    '''
    if x is None: return atan(y)
    else:         return atan2(y, x)
def pow(a, b):
    '''
    pow(a, b) yields a**b and works whether a and b are PyTorch tensors, NumPy
      arrays (or compatible objects like numbers), or SciPy sparse arrays.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(a):
            b = astensor(b)
            if b.is_sparse: b = b.to_dense()
            if a.is_sparse: a = a.to_dense()
            return a ** b
        elif torch.is_tensor(b):
            a = astensor(a)
            return torch.sqrt(a**2 + b**2)
    except ImportError: pass
    if sps.issparse(a): a = a.toarray()
    else: a = np.asarray(a)
    if sps.issparse(b): b = b.toarray()
    else: b = np.asarray(b)
    return a ** b
def hypot(a, b):
    '''
    hypot(a, b) yields sqrt(a**2 + b**2) and works whether a and b are PyTorch
      tensors or numpy arrays.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(a):
            b = astensor(b)
            return torch.sqrt(a**2 + b**2)
        elif torch.is_tensor(b):
            a = astensor(a)
            return torch.sqrt(a**2 + b**2)
    except ImportError: pass
    if sps.issparse(a):
        a2 = a.multiply(a)
        if sps.issparse(b):
            b2 = b.multiply(b)
            return np.sqrt(a2 + b2)
        else:
            b = asarray(b)
            return np.sqrt(a2 + b**2)
    elif sps.issparse(b):
        b2 = b.multiply(b)
        a = asarray(a)
        return np.sqrt(a**2 + b2)
    else:
        (a, b) = (np.asarray(a), np.asarray(b))
        return np.sqrt(a**2 + b**2)
def hypot2(a, b):
    '''
    hypot2(a, b) yields a**2 + b**2 and works whether a and b are PyTorch
      tensors or numpy arrays.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(a):
            b = astensor(b)
            return (a**2 + b**2)
        elif torch.is_tensor(b):
            a = astensor(a)
            return (a**2 + b**2)
    except ImportError: pass
    if sps.issparse(a):
        a2 = a.multiply(a)
        if sps.issparse(b):
            b2 = b.multiply(b)
            return (a2 + b2)
        else:
            b = asarray(b)
            return (a2 + b**2)
    elif sps.issparse(b):
        b2 = b.multiply(b)
        a = asarray(a)
        return (a**2 + b2)
    else:
        (a, b) = (np.asarray(a), np.asarray(b))
        return (a**2 + b**2)
def triarea(a,b,c):
    '''
    triarea(a,b,c) yields the area of the triangle whose sides have lengths
      a, b, and c. The arguments may be PyTorch tensors, numpy arrays, or
      Python primitives.
    '''
    hp = 0.5 * add(a, b, c)
    return mul(hp, sub(hp, a), sub(hp, b), sub(hp, c))
def eudist2(a, b):
    '''
    eudist2(a, b) yields the square distance between points a and b. The points
      may be vectors of the same length or matrices with one row per dimension.
      PyTorch tensors, numpy arrays, and primitives are allowed.
    '''
    (a, b) = promote(a, b)
    a = asdense(a)
    b = asdense(b)
    tot = (b[0] - a[0])**2
    for ii in range(1, a.shape[0]):
        tot += (b[ii] - a[ii])**2
    return tot
def eudist(a, b):
    '''
    eudist(a, b) yields the distance between points a and b. The points may be
      vectors of the same length or matrices with one row per dimension. PyTorch
      tensors, numpy arrays, and primitives are allowed.
    '''
    return sqrt(eudist2(a, b))
def trisides2(a, b, c):
    '''
    trisides2(a, b, c) yields the squared lengths of the sides of the triangles
      formed by the given coordinates. The arguemnts a, b, and c may be vectors
      or matrices with rows representing dimensions. PyTorch tensors, numpy
      arrays, and primitives are allowed.
    '''
    return (dist2(b, c), dist2(c, a), dist2(a, b))
def trisides(a, b, c):
    '''
    trisides(a, b, c) yields the lengths of the sides of the triangles formed by
      the given coordinates. The arguemnts a, b, and c may be vectors or
      matrices with rows representing dimensions. PyTorch tensors, numpy arrays,
      and primitives are allowed.
    '''
    (a2,b2,c2) = trisides2(a, b, c)
    return (sqrt(a2), sqrt(b2), sqrt(c2))
def trialtitudes(sides, area=None):
    '''
    trialtitudes((a,b,c)) yields (ha, hb, hc), the altitudes of each
      side associated with the side-lengths, a,b,c.
    trialtitudes((a,b,c), area) uses the given triangle area (which then doesn't
      have to be calculated).
    '''
    (a,b,c) = asdense(sides)
    if area is None: area = triarea(a, b, c)
    return (2*area/a, 2*area/b, 2*area/c)
def rangemod(u, min, max):
    '''
    rangemod(u, min, max) yields mod(u + min, max - min) - min.
    '''
    (u, min, max) = [asdense(x) for x in promote(u, min, max)]
    rng = max - min
    return mod(u + min, rng) - min
def radmod(u):
    '''
    radmod(u) yields mod(u + pi, 2 pi) - pi.
    '''
    return rangemod(u, -pi, pi)
def degmod(u):
    '''
    degmod(u) yields mod(u + 180, 360) - 180.
    '''
    return rangemod(u, -180, 180)
def branch(iftensor, thentensor, elsetensor=None):
    '''
    branch(q, t, e) yields, elementwise for the given tensors, t if q else e.
    branch(q, t) or branch(q, t, None) yields, elementwise, t if q else 0.
    branch(q, None, e) yields, elementwise, 0 if q else e.
    
    The output tensor will always have the same shape as q. The values for t
    and e may be constants or tensors the same shape as q.

    This function should be safe to use in optimization, i.e., for gradient
    calculatioins.
    '''
    if thentensor is None: thentensor = 0.0
    if elsetensor is None: elsetensor = 0.0
    (q, t, e) = promote(iftensor, thentensor, elsetensor)
    q = asdense(q)
    res = clone(q, dtype=t.dtype)
    q = (q != 0) # to convert to boolean
    res[:] = e
    if t.shape == ():        res[q] = t
    elif t.shape == q.shape: res[q] = t[q]
    else:
        ii = np.where(q)
        res[ii] = t[ii[-len(t.shape):]]
    return res
def zinv(x):
    '''
    zinv(x) yields 0 if x == 0 and 1/x otherwise. This is done in a way that
      is safe for torch gradients; i.e., the gradient for any element of x that
      is equal to 0 will also be 0. PyTorch tensors and numpy arrays are
      allowed.
    '''
    try:
        torch = pytorch()
        if torch.is_tensor(x):
            if x.is_sparse:
                (ij, vals) = (x.indices(), x.values())
                vals = zinv(vals)
                return torch.sparse_coo_tensor(ij, vals, dtype=x.dtype, device=x.device)
            ii = (x != 0)
            r = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
            r[ii] = 1 / x[ii]
            return r
    except ImportError: pass
    if sps.issparse(x):
        r = x.copy()
        r.data[:] = zinv(r.data)
        return r
    else:
        x = np.asarray(x)
        ii = (x != 0)
        r = np.zeros(x.shape, dtype=x.dtype)
        r[ii] = 1 / x[ii]
        return r

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
def check_sparsity(x, fraction=0.6):
    '''
    check_sparsity(x) yields either x or an array equivalent to x with a different sparsity based on
      a heuristic: if x is a sparse array with more than 60% of its elements specified, it is made
      dense; otherwise, it is left alone.

    The optional argument fraction (default 0.6) specifies the fraction of elements that must be
    specified in the array for it to be un-sparsified.
    '''
    if sps.issparse(x):
        n = numel(x)
        if n == 0: return x
        elif len(x.data) / float(n) > 0.6: return x.toarray()
        else: return x
    else:
        try:
            torch = pytorch()
            if torch.is_tensor(x) and x.is_sparse:
                n = numel(x)
                if len(x.indices()) / float(n) > 0.6: return x.to_dense()
        except ImportError: pass
        return x
def unbroadcast(a, b):
    '''
    unbroadcast(a, b) yields a tuple (aa, bb) that is equivalent to (a, b) except that aa and bb
      have been reshaped such that arithmetic numpy operations such as aa * bb will result in
      row-wise operation instead of column-wise broadcasting.
    '''
    # First: promote the arrays.
    (a,b) = [asdense(x) for x in promote(a, b)]
    # See if they're tensors first
    try:
        torch = pytorch()
        if torch.is_tensor(a):
            da = len(a.shape)
            db = len(b.shape)
            if   da > db: return (a, torch.reshape(b, b.shape + (1,)*(da - db)))
            elif da < db: return (torch.reshape(a, a.shape + (1,)*(db - da)), b)
            else:         return (a, b)
    except ImportError: pass
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
    da = len(a.shape)
    db = len(b.shape)
    if   da > db: return (a, np.reshape(b, b.shape + (1,)*(da - db)))
    elif da < db: return (np.reshape(a, a.shape + (1,)*(db - da)), b)
    else:         return (a, b)

# Accumulators #####################################################################################
@unary_mathfn(np.sum)
def sum(x, *args, **kw):
    '''
    sum(x) yields the sum of the elements in x.
    sum(x, axis) yields the sum along the given axis.
    '''
    return pytorch().sum(x, *args, **kw)
@unary_mathfn(np.prod)
def prod(x, *args, **kw):
    '''
    prod(x) yields the product of the elements in x.
    prod(x, axis) yields the product along the given axis.
    '''
    return pytorch().prod(x, *args, **kw)
@unary_mathfn(np.mean)
def mean(x, *args, **kw):
    '''
    mean(x) yields the mean of the elements in x.
    mean(x, axis) yields the mean along the given axis.
    '''
    return pytorch().mean(x, *args, **kw)
@unary_mathfn(np.var)
def var(x, *args, **kw):
    '''
    var(x) yields the variance of the elements in x.
    var(x, axis) yields the variance along the given axis.
    '''
    return pytorch().var(x, *args, **kw)
@unary_mathfn(np.std)
def std(x, *args, **kw):
    '''
    std(x) yields the standard deviation of the elements in x.
    std(x, axis) yields the standard deviation along the given axis.
    '''
    return pytorch().std(x, *args, **kw)
@unary_mathfn(np.median)
def median(x, *args, **kw):
    '''
    median(x) yields the median of the elements in x.
    median(x, axis) yields the median along the given axis.
    '''
    return pytorch().median(x, *args, **kw)
@unary_mathfn(np.min)
def min(x, *args, **kw):
    '''
    min(x) yields the minimum of the elements in x.
    min(x, axis) yields the minimum along the given axis.
    '''
    return pytorch().min(x, *args, **kw)[0]
@unary_mathfn(np.max)
def max(x, *args, **kw):
    '''
    max(x) yields the maximum of the elements in x.
    max(x, axis) yields the maximum along the given axis.
    '''
    return pytorch().max(x, *args, **kw)[0]
@unary_mathfn(np.argmin)
def argmin(x, *args, **kw):
    '''
    argmin(x) yields the inidex of the minimum of the element in x.
    argmin(x, axis) yields the indices of the minima along the given axis.
    '''
    return pytorch().min(x, *args, **kw)[1]
@unary_mathfn(np.argmax)
def argmax(x, *args, **kw):
    '''
    argmax(x) yields the inidex of the maximum of the element in x.
    argmax(x, axis) yields the indices of the maxima along the given axis.
    '''
    return pytorch().max(x, *args, **kw)[1]
@unary_mathfn(np.all)
def all(x, *args, **kw):
    '''
    min(x) yields the minimum of the elements in x.
    min(x, axis) yields the minimum along the given axis.
    '''
    return pytorch().all(x, *args, **kw)


# Distributions ####################################################################################
def beta_log_prob(x, mu, scale):
    '''
    beta_log_prob(mu, scale, x) yields the log probability density of the beta
      distribution parameterized using the mean mu (which must be between 0 and
      1) and scale (which may be any real number) at the value x.
      
    The traditional beta distribution uses parameters a and b. The
    reparameterizatoin of mu and scale here is as follows:
      a = (mu * (2 - b) - 1) / (mu - 1)
      b = (2 - 1/mu) + exp(scale)
    '''
    (x, mu, scale) = promote(x, mu, scale)
    try:
        torch = pytorch()
        if torch.is_tensor(x):
            from torch.distributions import Beta
            b = torch.exp(scale)
            a = (mu * (b - 2) + 1) / (1 - mu)
            dist = Beta(a, b)
            return dist.log_prob(x)
    except ImportError: pass
    from scipy.stats import beta
    b = np.exp(scale)
    a = (mu * (b - 2) + 1) / (1 - mu)
    return beta.logpdf(x, a, b)    
def beta_prob(x, mu, scale):
    '''
    beta_prob(x, mu, scale) yields the probability density function of the beta
      distribution parrameterized using the mean mu (which must be between 0 and
      1) and scale (which may be any real number) at the value x.
    
    The traditional beta distribution uses parameters a and b. The
    reparameterizatoin of mu and scale here is as follows:
      a = (mu * (2 - b) - 1) / (mu - 1)
      b = (2 - 1/mu) + exp(scale)
    '''
    (x, mu, scale) = promote(x, mu, scale)
    try:
        torch = pytorch()
        if torch.is_tensor(x):
            return torch.exp(beta_log_prob(x, mu, scale))
    except ImportError: pass
    from scipy.stats import beta
    b = np.exp(scale)
    a = (mu * (b - 2) + 1) / (1 - mu)
    return beta.pdf(x, a, b)
def normal_log_prob(t, center=0, width=1):
    '''
    normal_log_prob(t) yields the log-probability of the normal 
      distribution: -(t^2 + log(2 pi)) / 2.
    normal_log_prob(t, w) is equivalent to normal_log_prob(t) with a mean of w;
      equal to normal_log_prob(t - w).
    normal_log_prob(t, w, s) uses a standard deviation of s, equivalent to
      normal_log_prob(t - w, s) - log(w)
    '''
    (t, center, width) = promote(t, center, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            from torch.distributions import Normal
            dist = Normal(center, width)
            return dist.log_prob(t)
    except ImportError: pass
    from scipy.stats import norm
    return norm.logpdf(t, center, width)
def normal_prob(t, center=0, width=1):
    '''
    normal_prob(t) yields the probability density of the normal distribution at
      t: exp(-t^2 / 2) / sqrt(2 pi).
    normal_prob(t, w) is equivalent to normal_log_prob(t) with a mean of w;
      equal to normal_prob(t - w).
    normal_prob(t, w, s) uses a standard deviation of s, equivalent to
      normal_prob(t - w, s) / s.
    '''
    (t, center, width) = promote(t, center, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            return torch.exp(normal_log_prob(t, center, width))
    except ImportError: pass
    from scipy.stats import norm
    return norm.pdf(t, center, width)
def cauchy_log_prob(t, center=0, width=1):
    '''
    cauchy_log_prob(t) yields the log-probability of the Cauchy
      distribution: -log(pi (1 + t^2)).
    cauchy_log_prob(t, t0) is equivalent to cauchy_log_prob(t-t0).
    cauchy_log_prob(t, t0, s) is equivalent to cauchy_log_prob(t - t0) using
      a distribution with a width of s, equal to: -log(pi w (1 + (t/w)^2)).
    '''
    (t, center, width) = promote(t, center, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            from torch.distributions import Cauchy
            dist = Cauchy(center, width)
            return dist.log_prob(t)
    except ImportError: pass
    from scipy.stats import cauchy
    return cauchy.logpdf(t, center, width)
def cauchy_prob(t, center=0, width=1):
    '''
    cauchy_prob(t) yields the probability density of the Cauchy distribution at
      t with a scale of 1 and a center of 0.
    cauchy_prob(t, t0) is equivalent to cauchy_prob(t-t0).
    cauchy_prob(t, t0, s) is equivalent to cauchy_prob(t - t0) using
      a distribution with a width of s.
    '''
    (t, center, width) = promote(t, center, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            return torch.exp(cauchy_log_prob(t, center, width))
    except ImportError: pass
    from scipy.stats import cauchy
    return cauchy.pdf(t, center, width)
def halfcauchy_log_prob(t, width=1):
    '''
    halfcauchy_log_prob(t) yields the log-probability of the half-Cauchy
      distribution.
    halfcauchy_log_prob(t, s) uses a distribution with a width of s, equal to.
    '''
    (t, width) = promote(t, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            from torch.distributions import HalfCauchy
            dist = HalfCauchy(width)
            return dist.log_prob(t)
    except ImportError: pass
    from scipy.stats import halfcauchy
    return halfcauchy.logpdf(t, 0, width)
def halfcauchy_prob(t, width=1):
    '''
    halfcauchy_prob(t) yields the probability density of the half-Cauchy
      distribution at t with a scale of 1.
    halfcauchy_prob(t, s) uses a scale of s.
    '''
    (t, width) = promote(t, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            return torch.exp(halfcauchy_log_prob(t, width))
    except ImportError: pass
    from scipy.stats import halfcauchy
    return halfcauchy.pdf(t, 0, width)
def laplace_log_prob(t, width=1, center=0):
    '''
    laplace_log_prob(t) yields the log-probability of the Laplace
      distribution: -(|t| + log(2))
    laplace_log_prob(t, t0) is equivalent to laplace_log_prob(t - t0).
    laplace_log_prob(t, t0, s) uses a center of t0 and a scale of s.
    '''
    (t, center, width) = promote(t, center, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            from torch.distributions import Laplace
            dist = Laplace(center, width)
            return dist.log_prob(t)
    except ImportError: pass
    from scipy.stats import laplace
    return laplace.logpdf(t, center, width)
def laplace_prob(t, width=1, center=0):
    '''
    laplace_prob(t) yields the probability density of the Laplace distribution
      at t.
    laplace_prob(t, t0) is equivalent to laplace_prob(t - t0).
    laplace_prob(t, t0, s) uses a center of t0 and a scale of s.
    '''
    (t, center, width) = promote(t, center, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            return torch.exp(laplace_log_prob(t, center, width))
    except ImportError: pass
    from scipy.stats import laplace
    return laplace.pdf(t, center, width)
def exp_log_prob(t, width=1):
    '''
    exp_log_prob(t) yields the log-probability of the exponential distribution.
    exp_log_prob(t, s) uses a distribution with a width of s.
    '''
    (t, width) = promote(t, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            from torch.distributions import Exponential
            dist = Exponential(1 / width)
            return dist.log_prob(t)
    except ImportError: pass
    from scipy.stats import expon
    return expon.logpdf(t, 0, width)
def exp_prob(t, width=1):
    '''
    exp_prob(t) yields the probability density of the exponential distribution
      at t with a scale of 1.
    exp_prob(t, s) uses a scale of s.
    '''
    (t, width) = promote(t, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            return torch.exp(exp_log_prob(t, width))
    except ImportError: pass
    from scipy.stats import expon
    return expon.pdf(t, 0, width)
def gennorm_log_prob(t, q, center=0, width=1):
    '''
    gennorm_log_prob(t, q) yields the log-probability of the generalized error
      distribution, whose PDF is given by q / (2 gamma(1 / q)) exp(-|t|^q).
    genorm_log_prob(t, q, t0) is equivalent to gennorm_log_prob(t - t0, q).
    genorm_log_prob(t, q, t0, s) uses the scale s.
    '''
    (t, q, center, width) = promote(t, q, center, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            const = q / (2 * lgamma(1 / q)) / width
            return torch.log(const) - abs((t - t0) / width)**q
    except ImportError: pass
    from scipy.stats import gennorm
    return gennorm.logpdf(t, q, center, width)
def gennorm_prob(t, q, center=0, width=1):
    '''
    gennorm_prob(t, q) yields the probability density of the generalized error
      distribution, whose PDF is given by q / (2 gamma(1 / q)) exp(-|t|^q).
    genorm_prob(t, q, t0) is equivalent to gennorm_prob(t - t0, q).
    genorm_prob(t, q, t0, s) uses the scale s.
    '''
    (t, q, center, width) = promote(t, q, center, width)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            const = q / (2 * lgamma(1 / q)) / width
            return const * exp(-abs((t - t0) / width)**q)
    except ImportError: pass
    from scipy.stats import gennorm
    return gennorm.pdf(t, q, center, width)
def gumbel_log_prob(t, center=0, lw=1, rw=None):
    '''
    gumbel_log_prob(t) yields the log-probability of the standard Gumbel
      distribution whose PDF is given by exp(-(t + exp(-t))).
    gumbel_log_prob(t, w) yields the log-probability of the standard
      Gumbel distribution with a width of w.
    gumbel_log_prob(t, lw, rw) uses the lw and rw for the widths of the
      left-hand and right-hand side of the Gumbel distribution; i.e.:
      -(t/rw + exp(-t)/lw).
      
    Note that this is not a typical Gumbel distribution definition, but it is a
    very similar distribution nonetheless. Whereas the Gumbel PDF is usually
    defined as being proportional to exp(-((t-t0)/w + exp(-(t-t0)/w))), this
    particular parameterization allows for the linear part of the exponential
    B(T) = t to be scaled separately from the exponential part of the
    exponential A(t) = exp(-t). In a traditional Gumbel distribution,
    the PDF(t) = exp(-[A([t-t0]/w) + B([t-t0]/w)]). In the parameterization
    here, PDF(t) = exp(-[A([t-t0]/a) + B([t-t0]/b)]).
    '''
    if rw is None: rw = lw
    (t, center, lw, rw) = promote(t, center, lw, rw)
    t = (t - center)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            const = torch.log(1.0 / (lw * torch.lgamma(lw/rw)))
            return const - t/rw - torch.exp(-t/lw)
    except ImportError: pass
    const = np.log(1.0 / (lw * lgamma(lw/rw)))
    return const - t/rw - np.exp(-t/lw)
def gumbel_prob(t, center=0, lw=1, rw=None):
    '''
    gumbel_prob(t) yields the probability density of the standard Gumbel
      distribution whose PDF is given by exp(-(t + exp(-t))).
    gumbel_prob(t, w) yields the probability of the standard Gumbel
      distribution with a width of w.
    gumbel_prob(t, lw, rw) uses the lw and rw for the widths of the
      left-hand and right-hand side of the Gumbel distribution; i.e.: the
      probability distribution:
      exp(-(t/rw + exp(-t)/lw)) / (lw^(1/(lw*rw)) gamma(1/(lw*rw))).
      
    Note that this is not a typical Gumbel distribution definition, but it is a
    very similar distribution nonetheless. Whereas the Gumbel PDF is usually
    defined as being proportional to exp(-((t-t0)/w + exp(-(t-t0)/w))), this
    particular parameterization allows for the linear part of the exponential
    B(T) = t to be scaled separately from the exponential part of the
    exponential A(t) = exp(-t). In a traditional Gumbel distribution,
    the PDF(t) = exp(-[A([t-t0]/w) + B([t-t0]/w)]). In the parameterization
    here, PDF(t) = exp(-[A([t-t0]/a) + B([t-t0]/b)]).
    '''
    if rw is None: rw = lw
    (t, center, lw, rw) = promote(t, center, lw, rw)
    t = (t - center)
    try:
        torch = pytorch()
        if torch.is_tensor(t):
            const = 1.0 / (lw * torch.lgamma(lw/rw))
            return const * torch.exp(-t/rw - torch.exp(-t/lw))
    except ImportError: pass
    const = 1.0 / (lw * lgamma(lw/rw))
    return const * np.exp(-t/rw - np.exp(-t/lw))


