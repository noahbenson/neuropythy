####################################################################################################
# neuropythy/test/__init__.py
# Tests for the neuropythy library.
# By Noah C. Benson

import unittest, os, sys, six, logging, pimms
import numpy      as np
import pyrsistent as pyr
import neuropythy as ny

if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls

logging.getLogger().setLevel(logging.INFO)

class TestNeuropythy(unittest.TestCase):
    '''
    The TestNeuropythy class defines all the tests for the neuropythy library.
    '''

    def test_mesh(self):
        '''
        test_mesh() ensures that many general mesh properties and methods are working.
        '''
        import neuropythy.geometry as geo
        logging.info('neuropythy: Testing meshes and properties...')
        # get a random subject's mesh
        sub  = ny.data['benson_winawer_2018'].subjects['S1204']
        hem  = sub.hemis[('lh','rh')[np.random.randint(2)]]
        msh  = hem.white_surface
        # few simple things
        self.assertEqual(msh.coordinates.shape[0], 3)
        self.assertEqual(msh.tess.faces.shape[0], 3)
        self.assertEqual(msh.tess.edges.shape[0], 2)
        self.assertEqual(msh.vertex_count, msh.coordinates.shape[1])
        # face areas and edge lengths should all be non-negative
        self.assertGreaterEqual(np.min(msh.face_areas), 0)
        self.assertGreaterEqual(np.min(msh.edge_lengths), 0)
        # test the properties
        self.assertTrue('blerg' in msh.with_prop(blerg=msh.prop('curvature')).properties)
        self.assertFalse('curvature' in msh.wout_prop('curvature').properties)
        self.assertEqual(msh.properties.row_count, msh.vertex_count)
        self.assertLessEqual(np.abs(np.mean(msh.prop('curvature'))), 0.1)
        # use the property interface to grab a fancy masked property
        v123_areas = msh.property('midgray_surface_area',
                                  mask=('inf-prf_visual_area', (1,2,3)),
                                  null=0)
        v123_area = np.sum(v123_areas)
        self.assertLessEqual(v123_area, 15000)
        self.assertGreaterEqual(v123_area, 500)
        (v1_ecc, v1_rad) = msh.property(['prf_eccentricity','prf_radius'],
                                        mask=('inf-prf_visual_area', 1),
                                        weight='prf_variance_explained',
                                        weight_min=0.1,
                                        clipped=0,
                                        null=np.nan)
        wh = np.isfinite(v1_ecc) & np.isfinite(v1_rad)
        self.assertGreater(np.corrcoef(v1_ecc[wh], v1_rad[wh])[0,0], 0.5)

    def test_cmag(self):
        '''
        test_cmag() ensures that the neuropythy.vision cortical magnification function is working.
        '''
        import neuropythy.vision as vis
        logging.info('neuropythy: Testing areal cortical magnification...')
        dset = ny.data['benson_winawer_2018']
        sub = dset.subjects['S1202']
        hem = [sub.lh, sub.rh][np.random.randint(2)]
        cm = vis.areal_cmag(hem.midgray_surface, 'prf_',
                            mask=('inf-prf_visual_area', 1),
                            weight='prf_variance_explained')
        # cmag should get smaller in general
        ths = np.arange(0, 2*np.pi, np.pi/3)
        es = [0.5, 1, 2, 4]
        x = np.diff([np.mean(cm(e*np.cos(ths), e*np.sin(ths))) for e in es])
        self.assertTrue((x < 0).all())
    
    def test_interpolation(self):
        '''
        test_interpolation() performs a variety of high-level tests involving interpolation using
          neuropythy that should catch major errors to important components.
        '''
        logging.info('neuropythy: Testing interpolation...')
        def choose(coll, k): return np.random.choice(coll, k, False)
        # to do these tests, we use the builtin dataset from Benson and Winawer (2018); see also
        # help(ny.data['benson_winawer_2018']) for more information on this dataset.
        dset = ny.data['benson_winawer_2018']
        self.assertTrue(os.path.isdir(dset.cache_directory))
        # pick 1 of the subjects at random
        allsubs = [dset.subjects['S12%02d' % (s+1)] for s in range(8)]
        subs = choose(allsubs, 1)
        fsa = ny.freesurfer_subject('fsaverage')
        def check_dtypes(a,b):
            for tt in [np.integer, np.floating, np.bool_, np.complexfloating]:
                self.assertEqual(np.issubdtype(a.dtype, tt), np.issubdtype(b.dtype, tt))
        def calc_interp(hem, interhem, ps):
            for p in ps: self.assertEqual(np.sum(~np.isfinite(hem.prop(p))), 0)
            us = hem.interpolate(interhem, ps)
            for u in us: self.assertEqual(np.sum(~np.isfinite(u)), 0)
            vs = interhem.interpolate(hem, us)
            for v in vs: self.assertEqual(np.sum(~np.isfinite(v)), 0)
            return vs
        def check_interp(hem, ps, vs):
            for (p,v) in zip(ps,vs):
                logging.info('neuropythy:         * %s', p)
                p = hem.prop(p)
                self.assertEqual(len(p), len(v))
                self.assertLessEqual(np.min(p), np.min(v))
                self.assertGreaterEqual(np.max(p), np.max(v))
                check_dtypes(p, v)
                self.assertGreater(np.corrcoef(p, v)[0,0], 0.6)
        for sub in subs:
            logging.info('neuropythy: - Testing subject %s', sub.name)
            # left hemisphere should have a negative mean x-value, right a positive mean x-value
            self.assertTrue(np.mean(sub.lh.white_surface.coordinates, axis=1)[0] < 0)
            self.assertTrue(np.mean(sub.rh.pial_surface.coordinates, axis=1)[0] > 0)
            # some simple ideas: if we interpolate the properties from one subject to another and
            # then interpolate back, we should get approximately, if not exactly, the same thing
            # for this pick a couple random properties:
            ps = ['prf_variance_explained', 'inf-prf10_visual_area']
            intersub = choose(allsubs, 1)[0]
            logging.info('neuropythy:  - Testing properties %s via subject %s', ps, intersub.name)
            logging.info('neuropythy:    - Testing LH interpolation')
            vs = calc_interp(sub.lh, intersub.lh, ps)
            check_interp(sub.lh, ps, vs)
            logging.info('neuropythy:    - Testing RH interpolation')
            vs = calc_interp(sub.rh, intersub.rh, ps)
            check_interp(sub.rh, ps, vs)
        
if __name__ == '__main__':
    unittest.main()
