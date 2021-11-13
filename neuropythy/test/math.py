####################################################################################################
# neuropythy/test/math.py
# Tests for the neuropythy library's util.math module.
# By Noah C. Benson

import unittest, os, sys, six, warnings, logging, pimms, torch
import numpy        as np
import scipy.sparse as sps
import pyrsistent   as pyr
import neuropythy   as ny

if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls

class TestNeuropythyMath(unittest.TestCase):
    
    def test_info_fns(self):
        from neuropythy.math import (to_torchdtype, torchdtype_to_numpydtype, isarray, istensor,
                                     issparse, isdense)
        dts = {'int8': (torch.int8, np.int8),
               'int16': (torch.int16, np.int16),
               'int32': (torch.int32, np.int32),
               'int64': (torch.int64, np.int64),
               'float64': (torch.float64, np.float64),
               'complex128': (torch.complex128, np.complex128),
               'bool': (torch.bool, np.bool)}
        for (k, (vt,vn)) in six.iteritems(dts):
            self.assertEqual(to_torchdtype(k), vt)
            self.assertEqual(to_torchdtype(vt), vt)
            self.assertEqual(to_torchdtype(vn), vt)
            self.assertEqual(torchdtype_to_numpydtype(vt), vn)
        
        x = [1.0, 4.4, 5.1]
        y = [6,5,4,3]
        xn = np.array(x)
        yn = np.array(y)
        xt = torch.tensor(x)
        yt = torch.tensor(y)
        xs = sps.coo_matrix((x, ([0,1,2], [3,2,1])), (4,4))
        ys = sps.coo_matrix((y, ([0,1,2,3], [2,3,2,1])), (4,4))
        xp = torch.sparse_coo_tensor(([0,1,2], [3,2,1]), x, (4,4))
        yp = torch.sparse_coo_tensor(([0,1,2,3], [2,3,2,1]), y, (4,4))

        self.assertFalse(isarray(x))
        self.assertFalse(isarray(y))
        self.assertTrue(isarray(xn))
        self.assertTrue(isarray(yn))
        self.assertFalse(isarray(xt))
        self.assertFalse(isarray(yt))
        self.assertTrue(isarray(xs))
        self.assertTrue(isarray(ys))
        self.assertFalse(isarray(xp))
        self.assertFalse(isarray(yp))

        self.assertFalse(istensor(x))
        self.assertFalse(istensor(y))
        self.assertFalse(istensor(xn))
        self.assertFalse(istensor(yn))
        self.assertTrue(istensor(xt))
        self.assertTrue(istensor(yt))
        self.assertFalse(istensor(xs))
        self.assertFalse(istensor(ys))
        self.assertTrue(istensor(xp))
        self.assertTrue(istensor(yp))

        self.assertFalse(issparse(x))
        self.assertFalse(issparse(y))
        self.assertFalse(issparse(xn))
        self.assertFalse(issparse(yn))
        self.assertFalse(issparse(xt))
        self.assertFalse(issparse(yt))
        self.assertTrue(issparse(xs))
        self.assertTrue(issparse(ys))
        self.assertTrue(issparse(xp))
        self.assertTrue(issparse(yp))

        self.assertTrue(isdense(x))
        self.assertTrue(isdense(y))
        self.assertTrue(isdense(xn))
        self.assertTrue(isdense(yn))
        self.assertTrue(isdense(xt))
        self.assertTrue(isdense(yt))
        self.assertFalse(isdense(xs))
        self.assertFalse(isdense(ys))
        self.assertFalse(isdense(xp))
        self.assertFalse(isdense(yp))

    def test_create_fns(self):
        from neuropythy.math import (clone, eq, all, astensor, totensor, isarray, istensor,
                                     asarray, toarray)

        x = [1.0, 4.4, 5.1]
        y = [6,5,4,3]
        xn = np.array(x)
        yn = np.array(y)
        xt = torch.tensor(x)
        yt = torch.tensor(y)
        xs = sps.coo_matrix((x, ([0,1,2], [3,2,1])), (4,4))
        ys = sps.coo_matrix((y, ([0,1,2,3], [2,3,2,1])), (4,4))
        xp = torch.sparse_coo_tensor(([0,1,2], [3,2,1]), x, (4,4))
        yp = torch.sparse_coo_tensor(([0,1,2,3], [2,3,2,1]), y, (4,4))

        for xx in [x, xn, xt, xs, xp]:
            u = clone(xx)
            self.assertFalse(u is xx)
            self.assertTrue(all(eq(u, xx)))
        for xx in [xp, yp, xt, yt]:
            u = astensor(xx)
            self.assertTrue(u is xx)
            self.assertTrue(torch.is_tensor(u))
            u = astensor(u, dtype='float32')
            self.assertTrue(u.dtype == torch.float32)
            self.assertTrue(torch.is_tensor(u))
            self.assertTrue(all(eq(u, xx)))
            u = totensor(xx)
            self.assertFalse(u is xx)
            self.assertTrue(torch.is_tensor(u))
            self.assertFalse(isarray(u))
            self.assertTrue(istensor(u))
            self.assertTrue(all(eq(u, xx)))
            u = asarray(xx)
            self.assertFalse(u is xx)
            self.assertTrue(isinstance(u, np.ndarray) or sps.issparse(u))
            self.assertTrue(isarray(u))
            self.assertFalse(istensor(u))
            self.assertTrue(all(eq(u, xx)))
            u = toarray(xx)
            self.assertFalse(u is xx)
            self.assertTrue(isinstance(u, np.ndarray) or sps.issparse(u))
            self.assertTrue(isarray(u))
            self.assertFalse(istensor(u))
            self.assertTrue(all(eq(u, xx)))
        
    def test_index_fns(self):
        from neuropythy.math import (reshape_indices, flatten_indices, unflatten_indices)
        import numpy as np
        shape = (13,19,7,2)
        arr1 = np.reshape(np.arange(np.prod(shape)), shape)
        ii1_y = np.where(np.mod(arr1) == 0)
        ii1_n = np.where(np.mod(arr1) == 1)
        newshape = (7*2, 13*19)
        arr2 = np.reshape(arr1, newshape)
        ii2_y = reshape_indices(shape, ii1_y, newshape)
        ii2_n = reshape_indices(shape, ii1_n, newshape)
        self.assertTrue(np.array_equal(arr1[ii1_y], arr2[ii2_y]))
        self.assertTrue(np.array_equal(arr1[ii1_n], arr2[ii2_n]))
        ff1_y = flatten_indices(shape, ii1_y)
        ff2_y = flatten_indices(newshape, ii2_y)
        self.assertTrue(np.array_equal(ff1_y, ff2_y))
        ff1_n = flatten_indices(shape, ii1_n)
        ff2_n = flatten_indices(newshape, ii2_n)
        self.assertTrue(np.array_equal(ff1_n, ff2_n))
        ushape = (13*2*19, -1)
        uu1_y = unflatten_indices(ff1_y, ushape)
        uu2_y = unflatten_indices(ff2_y, ushape)
        self.assertTrue(np.array_equal(uu1_y, uu2_y))
        uu1_n = unflatten_indices(ff1_n, ushape)
        uu2_n = unflatten_indices(ff2_n, ushape)
        self.assertTrue(np.array_equal(uu1_n, uu2_n))

    def test_arraylike(self):
        from neuropythy.math import arraylike
        import numpy as np, torch
        # A variety of tests!
        tests = [dict(data=[1,2,3], dtype='int', shape=(3,), lib='python'),
                 dict(data=[[1,2.0,3],[3.0,-1,0.0]], dtype='float', shape=(2,3), lib='python'),
                 dict(data=[(1,2),(3,2),(3,3)], dtype='int', shape=(3,2), lib='python'),
                 dict(data=np.array([1,2,3]), dtype='int', shape=(3,), lib='numpy'),
                 dict(data=torch.tensor([1,2,3]), dtype='int', shape=(3,), lib='torch')]
        for test in tests:
            dat = test.pop('data')
            args = dict(dtype=test['dtype'], shape=test['shape'])
            self.assertTrue(arraylike(dat, **args))
            for lib in ['python','numpy','torch']:
                args[lib] = True
                res = arraylike(dat, **args)
                del args[lib]
                if test['lib'] == lib: self.assertTrue(res)
                else: self.assertFalse(res)
            for dtype in ['float', 'int']:
                args['dtype'] = dtype
                res = arraylike(dat, **args)
                if dtype == test['dtype']: self.assertTrue(res)
                else: self.assertFalse(res)
            args['dtype'] = test['dtype']
        # Some straightforward tests.
        self.assertTrue(arraylike([1,2,3], dtype='int', ndims=1, python=True))
        self.assertTrue(arraylike([1,2,3], dtype='int', ndims=1))
        self.assertTrue(arraylike([1,2,3], python=True))
        self.assertTrue(arraylike([1,2,3], ndims=1))
        self.assertTrue(arraylike([1,2,3], dtype='int'))
        self.assertTrue(arraylike([1,2,3], shape=(3,)))
        self.assertTrue(arraylike([1,2,3], shape=(-1,)))
        self.assertTrue(arraylike([1,2,3], shape=(-1,...)))
        self.assertTrue(arraylike([1,2,3], shape=(...,3)))
        self.assertFalse(arraylike([1,2,3], dtype='float', ndims=1, python=True))
        self.assertFalse(arraylike([1,2,3], dtype='int', ndims=2))
        self.assertFalse(arraylike([1,2,3], dtype='int', python=False))
        self.assertFalse(arraylike([1,2,3], ndims=3))
        self.assertFalse(arraylike([1,2,3], dtype='bool'))
        self.assertFalse(arraylike([1,2,3], shape=(2,)))
        self.assertFalse(arraylike([1,2,3], shape=(-1,-1)))
        # Shape and dim matching tests.
        shapes = [(3,), (5,5), (4,6,2), (1,9), (2,3,4,5,1)]
        for sh in shapes:
            arr = np.zeros(sh)
            self.assertTrue(arraylike(arr, shape=sh))
            self.assertFalse(arraylike(arr, shape=sh[:-1]))
            self.assertTrue(arraylike(arr, shape=(sh[:-1] + (-1,))))
            self.assertTrue(arraylike(arr, shape=(sh[:-1] + (Ellipsis,))))
            self.assertTrue(arraylike(arr, ndims=len(sh)))
            self.assertTrue(arraylike(arr, ndims=(1,len(sh),10)))
            self.assertFalse(arraylike(arr, ndims=(len(sh)-1,len(sh)+1)))
        
    # #TODO: Add tests for:
    #  - arraylike
    #  - is_numeric

                        
