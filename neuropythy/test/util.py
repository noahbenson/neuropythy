####################################################################################################
# neuropythy/test/util.py
# Tests for the neuropythy library's util module.
# By Noah C. Benson

import unittest, os, sys, six, warnings, logging, pimms, torch
import numpy        as np
import scipy.sparse as sps
import pyrsistent   as pyr
import neuropythy   as ny
import neuropythy.math as nym

if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls

class TestNeuropythyUtil(unittest.TestCase):
    
    def test_config(self):
        import neuropythy.util.conf as cfg
        # Credentials functions.
        for ss in ["abc:def", "abc\ndef", "\nabc :  def \n ", '["abc", "def"]',
                   '{"key": "abc", "secret": "def"}']:
            (k,s) = cfg.str_to_credentials(ss)
            self.assertTrue(k == "abc")
            self.assertTrue(s == "def")
            # If it works for str_to_credentials it should also work for to_credentials!
            (k,s) = cfg.to_credentials(ss)
            self.assertTrue(k == "abc")
            self.assertTrue(s == "def")
        for ss in [('abc','def'), ['abc','def'], {'key': 'abc', 'secret': 'def'}]:
            (k,s) = cfg.to_credentials(ss)
            self.assertTrue(k == "abc")
            self.assertTrue(s == "def")
    # #TODO: Need tests of the util.conf code that deals with files; for now these are just skipped.

    def test_info_utils(self):
        from neuropythy.util import (is_hemi_str, like_hemi_str, to_hemi_str,
                                     is_cortical_depth, like_cortical_depth, to_cortical_depth)
        # to_hemi_str
        for k in ['lh', 'LH', 'l', 'LeFT']:
            self.assertTrue(to_hemi_str(k) == 'lh')
        for k in ['rh', 'RH', 'r', 'RigHT']:
            self.assertTrue(to_hemi_str(k) == 'rh')
        for k in ['lr', 'both', 'LR', 'xh', 'ALL', None, Ellipsis]:
            self.assertTrue(to_hemi_str(k) == 'lr')
        self.assertTrue(is_hemi_str('lh'))
        self.assertTrue(is_hemi_str('rh'))
        self.assertTrue(is_hemi_str('lr'))
        for k in ['LH','RH','LR','both','all',None,Ellipsis]:
            self.assertFalse(is_hemi_str(k))
        for k in ['LH','RH','LR','both','all',None,Ellipsis]:
            self.assertTrue(like_hemi_str(k))
        # cortical depths
        for k in [0.0, 0.1, 0.5, 1.0]:
            self.assertTrue(is_cortical_depth(k))
            self.assertTrue(k is to_cortical_depth(k))
        for k in [0, 1, 2, 'white', -0.1, 1.1]:
            self.assertFalse(is_cortical_depth(k))
        for (k,d) in [(0,0.0) (1,1.0), ('white',0.0), ('pial',1.0),
                      ('MIDGRAY',0.5)]:
            self.assertTrue(like_cortical_depth(k))
            self.assertTrue(to_cortical_depth(k) == d)
        s = {'white': [0,1,2], 'midgray': [1,2,3], 'pial': [2,3,4]}
        for tt in (pimms.pmap, pimms.lmap, dict):
            d = to_cortical_depth(tt(s))
            self.assertTrue(len(d) == 3)
            self.assertTrue(d[0.0] == s['white'])
            self.assertTrue(d[0.5] == s['midgray'])
            self.assertTrue(d[1.0] == s['pial'])
        # iterpolation methods
        for k in ['linear', 'cubic', 'nearest', 'heaviest']:
            self.assertTrue(is_interpolation_method(k))
            self.assertTrue(like_interpolation_method(k))
            self.assertTrue(k is to_interpolation_method(k))
        for k in ['trilin', 'nn', 'HEAVIEST', 'cub']:
            self.assertFalse(is_interpolation_method(k))
            self.assertTrue(like_interpolation_method(k))
        for (k,d) in [('trilin','linear') ('nn','nearest'), ('HEAVIEST','heavest'),
                      ('cub','cubic')]:
            self.assertTrue(like_interpolation_method(k))
            self.assertTrue(to_interpolation_method(k) == d)
    
    def test_normalize(self):
        objs = [10, "abc", 5+6.7j,
                (1, 2.5, "def"), [1, 2.5, "def"],
                set([4, "test", (1,2,3)]),
                {'key1': 'val1', 'key2': set(['val2'])}]
        for obj in objs:
            self.assertEqual(obj, denormalize(normalize(obj)))
    
    def test_affines(self):
        for sz in [2,3,4,5]:
            for torchq in [True,False]:
                n = np.random.randint(1, 250)
                if torchq:
                    affine = torch.eye(sz+1)
                    affine[:-1, :] += torch.randn(sz, sz+1)**2
                    coords = torch.randn(sz, n)
                else:
                    affine = np.eye(sz + 1)
                    affine[:-1, :] += np.random.randn(sz, sz+1)**2
                    coords = np.random.randn(sz, n)
                inverse = nym.inverse(affine)
                mtx = affine[:-1,:-1]
                off = affine[:-1, -1]
                for trq in [True,False]:
                    self.assertTrue(nym.equal(affine, ny.util.to_affine(affine)))
                    self.assertTrue(nym.equal(affine, ny.util.to_affine((mtx, off))))
                    self.assertTrue(nym.equal(affine, ny.util.to_affine(affine[:-1,:])))
                    self.assertTrue(nym.equal(affine, ny.util.to_affine(tuple(affine[:-1,:])))))
                    self.assertTrue(nym.equal(affine, ny.util.to_affine(tuple(affine))))
                    if trq: fcoords = nym.tr(coords)
                    else: fcoords = coords
                    fcoords = ny.util.apply_affine(affine,  fcoords)
                    rcoords = ny.util.apply_affine(inverse, fcoords)
                    if trq: rcoords = nym.tr(rcoords)
                    self.assertTrue(nym.all(nym.isclose(rcoords, coords)))

    # #TODO: add tests for AutoDict and auto_dict() (they are very simple)

    def test_dataframe(self):
        import pandas
        arg1 = {'a': [1,2,3], 'b':[3.3, 4.4, 5.5]}
        arg2 = [{'a':1, 'b':2}, {'a':2, 'b'4.4}, {'a':3, 'b':5.5}]
        self.assertFalse(ny.util.is_dataframe(arg1))
        self.assertFalse(ny.util.is_dataframe(arg2))
        df1 = ny.util.to_dataframe(arg1)
        df2 = ny.util.to_dataframe(arg2)
        self.assertTrue(ny.util.is_dataframe(df1))
        self.assertTrue(ny.util.is_dataframe(df2))
        self.assertTrue(nym.equal(df1.values, df2.values))
        s1 = ny.util.dataframe_select(df1, a=1)
        s2 = ny.util.dataframe_select(df2, a=1)
        self.assertTrue(len(s1) == 1)
        self.assertTrue(len(s1.columns) == 2)
        self.assertTrue(nym.equal(s1.values, s2.values))
        s1 = ny.util.dataframe_select(df1, b=set([3.3,5.5]))
        s2 = ny.util.dataframe_select(df2, b=set([3.3,5.5]))
        self.assertTrue(len(s1) == 2)
        self.assertTrue(len(s1.columns) == 2)
        self.assertTrue(nym.equal(s1.values, s2.values))
        s1 = ny.util.dataframe_select(df1, 'a', b=(3.0, 4.5))
        s2 = ny.util.dataframe_select(df2, 'a', b=(3.0, 4.5))
        self.assertTrue(len(s1) == 2)
        self.assertTrue(len(s1.columns) == 1)
        self.assertTrue(nym.equal(s1.values, s2.values))
        s1 = ny.util.dataframe_select(df1, 'a', 'b', a=[1,2,3], b=(0,10))
        s2 = ny.util.dataframe_select(df2, 'a', 'b', a=[1,2,3], b=(0,10))
        self.assertTrue(len(s1) == 3)
        self.assertTrue(len(s1.columns) == 2)
        self.assertTrue(nym.equal(s1.values, s2.values))

    # #TODO: add tests for addresses
    
