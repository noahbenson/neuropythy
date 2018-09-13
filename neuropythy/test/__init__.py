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

    def test_data(self):
        '''
        test_data() performs a variety of high-level tests using neuropythy that should catch major 
          errors to important components.
        '''
        def choose(coll, k): return np.random.choice(coll, k, False)
        # to do these tests, we use the builtin dataset from Benson and Winawer (2018); see also
        # help(ny.data['benson_winawer_2018']) for more information on this dataset.
        dset = ny.data['benson_winawer_2018']
        self.assertTrue(os.path.isdir(dset.cache_directory))
        # pick 3 of the subjects at random
        subs = [dset.subjects['S12%02d' % (s+1)] for s in choose(range(len(dset.subjects)), 3)]
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
                logging.info('                    * %s', p)
                p = hem.prop(p)
                self.assertEqual(len(p), len(v))
                self.assertLessEqual(np.min(p), np.min(v))
                self.assertGreaterEqual(np.max(p), np.max(v))
                check_dtypes(p, v)
                self.assertGreater(np.corrcoef(p, v)[0,0], 0.6)
        for sub in subs:
            logging.info('neuropythy: Testing subject %s', sub.name)
            # left hemisphere should have a negative mean x-value, right a positive mean x-value
            self.assertTrue(np.mean(sub.lh.white_surface.coordinates, axis=1)[0] < 0)
            self.assertTrue(np.mean(sub.rh.pial_surface.coordinates, axis=1)[0] > 0)
            # some simple ideas: if we interpolate the properties from one subject to another and
            # then interpolate back, we should get approximately, if not exactly, the same thing
            # for this pick a couple random properties:
            ps = choose(list(sub.lh.properties.keys()), 2)
            intersub = choose(subs, 1)[0]
            logging.info('             - Testing properties %s via subject %s', ps, intersub.name)
            logging.info('               - Testing LH interpolation')
            vs = calc_interp(sub.lh, intersub.lh, ps)
            check_interp(sub.lh, ps, vs)
            logging.info('               - Testing RH interpolation')
            vs = calc_interp(sub.rh, intersub.rh, ps)
            check_interp(sub.rh, ps, vs)
            
        
if __name__ == '__main__':
    unittest.main()
