####################################################################################################
# neuropythy/datasets/benson_winawer_2018.py
# The dataset from Benson and Winawer (2018); DOI: https://doi.org/10.1101/325597
# by Noah C. Benson

import os, six, shutil, tarfile, logging, warnings, pimms
import numpy as np
import pyrsistent as pyr

from six.moves import urllib

if six.PY3: from functools import reduce

from .core        import (Dataset, add_dataset)
from ..util       import (config, curry, AutoDict)
from ..vision     import as_retinotopy
from ..           import io      as nyio
from ..freesurfer import subject as freesurfer_subject

config.declare('benson_winawer_2018_path')

@pimms.immutable
class BensonWinawer2018Dataset(Dataset):
    '''
    neuropythy.data['benson_winawer_2018'] is a Dataset containing the publicly provided data
    from the following publication:

    Benson NC, Winawer J (2018) Bayesian analysis of retinotopic maps. BioRxiv.
      DOI:10.1101/325597

    These data include 8 FreeSurfer subjects each with a set of measured and inferred
    retinotopic maps. These data are provided as follows:
    
    dset = neuropythy.data['benson_winawer_2018']
    sorted(dset.subjects.keys())
    #=> ['S1201', 'S1202', 'S1203', 'S1204', 'S1205', 'S1206', 'S1207', 'S1208', 'fsaverage']

    dset.subjects['S1202']
    #=> Subject(<S1202>,
    #=>         <'/Users/nben/.cache/benson_winawer_2018/freesurfer_subjects/S1202'>)

    dset.subjects['S1202'].lh
    #=> Cortex(<LH>, <301348 faces>, <150676 vertices>)

    sorted(dset.subjects['S1201'].lh.properties.keys())
    #=> ['convexity', 'curvature', 'index', 'inf-prf00_eccentricity',
    #=>  'inf-prf00_polar_angle', 'inf-prf_radius', 'inf-prf_visual_area', 'label',
    #=>  'midgray_surface_area', 'pial_surface_area', 'prf00_eccentricity',
    #=>  'prf00_polar_angle', 'prf00_radius', 'prf00_variance_explained',
    #=>  'prf01_eccentricity', 'prf01_polar_angle', 'prf01_radius', 
    #=>  'prf01_variance_explained', 'prf02_eccentricity', 'prf02_polar_angle',
    #=>  'prf02_radius', 'prf02_variance_explained', 'prf03_eccentricity',
    #=>  'prf03_polar_angle', 'prf03_radius', 'prf03_variance_explained',
    #=>  'prf04_eccentricity', 'prf04_polar_angle', 'prf04_radius',
    #=>  'prf04_variance_explained', 'prf05_eccentricity', 'prf05_polar_angle',
    #=>  'prf05_radius', 'prf05_variance_explained', 'prf06_eccentricity',
    #=>  'prf06_polar_angle', 'prf06_radius', 'prf06_variance_explained',
    #=>  'prf07_eccentricity', 'prf07_polar_angle', 'prf07_radius',
    #=>  'prf07_variance_explained', 'prf08_eccentricity', 'prf08_polar_angle',
    #=>  'prf08_radius', 'prf08_variance_explained', 'prf09_eccentricity',
    #=>  'prf09_polar_angle', 'prf09_radius', 'prf09_variance_explained',
    #=>  'prf10_eccentricity', 'prf10_polar_angle', 'prf10_radius',
    #=>  'prf10_variance_explained', 'prf11_eccentricity', 'prf11_polar_angle',
    #=>  'prf11_radius', 'prf11_variance_explained', 'prf12_eccentricity',
    #=>  'prf12_polar_angle', 'prf12_radius', 'prf12_variance_explained',
    #=>  'prf13_eccentricity', 'prf13_polar_angle', 'prf13_radius',
    #=>  'prf13_variance_explained', 'prf14_eccentricity', 'prf14_polar_angle',
    #=>  'prf14_radius', 'prf14_variance_explained', 'prf15_eccentricity',
    #=>  'prf15_polar_angle', 'prf15_radius', 'prf15_variance_explained',
    #=>  'prf16_eccentricity', 'prf16_polar_angle', 'prf16_radius',
    #=>  'prf16_variance_explained', 'prf17_eccentricity', 'prf17_polar_angle',
    #=>  'prf17_radius', 'prf17_variance_explained', 'prf18_eccentricity',
    #=>  'prf18_polar_angle', 'prf18_radius', 'prf18_variance_explained',
    #=>  'prf19_eccentricity', 'prf19_polar_angle', 'prf19_radius',
    #=>  'prf19_variance_explained', 'prf20_eccentricity', 'prf20_polar_angle',
    #=>  'prf20_radius', 'prf20_variance_explained', 'prf21_eccentricity',
    #=>  'prf21_polar_angle', 'prf21_radius', 'prf21_variance_explained', 'prf_eccentricity',
    #=>  'prf_polar_angle', 'prf_radius', 'prf_variance_explained', 'prior-prf_eccentricity',
    #=>  'prior-prf_polar_angle', 'prior-prf_radius', 'prior-prf_visual_area', 'thickness',
    #=>  'volume', 'white_surface_area', 'wide-prf_eccentricity', 'wide-prf_polar_angle',
    #=>  'wide-prf_radius', 'wide-prf_variance_explained']
    dset.meta_data
    #=> ['prf', 'prf00', 'prf01', 'prf02', 'prf03', 'prf04', 'prf05', 'prf06', 'prf07',
    #=>  'prf08', 'prf09', 'prf10', 'prf11', 'prf12', 'prf13', 'prf14', 'prf15', 'prf16',
    #=>  'prf17', 'prf18', 'prf19', 'prf20', 'prf21']

    dset.meta_data['prf04']
    #=> {'scan_seconds': 576, 'name': 'training14', 'id': 14, 'scans': 3}

    # Note that the following lines will take awhile to calculate/load from cache due to the
    # size of the data; additionally, the generated cache file is ~1GB.

    dset.v123_table
    #=> itable(('inf_x', 'label', 'radius', 'eccentricity', 'inf_radius', 'hemi', 'x',
    #=>         'subject', 'y', 'inf_y', 'inf_eccentricity', 'midgray_surface_area',
    #=>         'inf_polar_angle', 'polar_angle', 'inf_visual_area', 'pial_surface_area',
    #=>         'dataset_id', 'dataset_name', 'variance_explained', 'white_surface_area'),
    #=>        <50278184 rows>)

    dset.v123_table[100] # (caching all rows is somewhat slow the first time you do this)
    #=> {'inf_x': 4.0969896, 'label': 100, 'radius': 2.096148, 'eccentricity': 3.3061967,
    #=>  'inf_radius': 1.5482311, 'hemi': 'lh', 'x': 0.032615896, 'subject': 'S1208', 
    #=>  'y': -3.30603, 'inf_y': -2.0219889, 'inf_eccentricity': 4.5687814,
    #=>  'midgray_surface_area': 1.072214, 'inf_polar_angle': 116.267746,
    #=>  'polar_angle': 179.43477, 'inf_visual_area': 3, 'pial_surface_area': 1.3856983,
    #=>  'dataset_id': 10, 'dataset_name': 'prf10', 'variance_explained': 0.054989286,
    #=>  'white_surface_area': 0.75872976}
    '''
    dataset_urls = {'analyses':              'https://osf.io/cpfa8/download',
                    'retinotopy':            'https://osf.io/m4k8q/download',
                    #'wang2015':              'https://osf.io/rx9ca/download',
                    'freesurfer_subjects':   'https://osf.io/pu9js/download'}
    prf_meta_data = pyr.m(prf00=pyr.m(id=0,  name='validation', scans=6,  scan_seconds=192*6),
                          prf01=pyr.m(id=1,  name='training01', scans=1,  scan_seconds=192),
                          prf02=pyr.m(id=2,  name='training02', scans=1,  scan_seconds=192),
                          prf03=pyr.m(id=3,  name='training03', scans=1,  scan_seconds=192),
                          prf04=pyr.m(id=4,  name='training04', scans=1,  scan_seconds=192),
                          prf05=pyr.m(id=5,  name='training05', scans=1,  scan_seconds=192),
                          prf06=pyr.m(id=6,  name='training06', scans=1,  scan_seconds=192),
                          prf07=pyr.m(id=7,  name='training07', scans=2,  scan_seconds=192*2),
                          prf08=pyr.m(id=8,  name='training08', scans=2,  scan_seconds=192*2),
                          prf09=pyr.m(id=9,  name='training09', scans=2,  scan_seconds=192*2),
                          prf10=pyr.m(id=10, name='training10', scans=2,  scan_seconds=192*2),
                          prf11=pyr.m(id=11, name='training11', scans=2,  scan_seconds=192*2),
                          prf12=pyr.m(id=12, name='training12', scans=3,  scan_seconds=192*3),
                          prf13=pyr.m(id=13, name='training13', scans=3,  scan_seconds=192*3),
                          prf14=pyr.m(id=14, name='training14', scans=3,  scan_seconds=192*3),
                          prf15=pyr.m(id=15, name='training15', scans=3,  scan_seconds=192*3),
                          prf16=pyr.m(id=16, name='training16', scans=4,  scan_seconds=192*4),
                          prf17=pyr.m(id=17, name='training17', scans=4,  scan_seconds=192*4),
                          prf18=pyr.m(id=18, name='training18', scans=4,  scan_seconds=192*4),
                          prf19=pyr.m(id=19, name='training19', scans=5,  scan_seconds=192*5),
                          prf20=pyr.m(id=20, name='training20', scans=5,  scan_seconds=192*5),
                          prf21=pyr.m(id=21, name='training21', scans=6,  scan_seconds=192*6),
                          prf  =pyr.m(id=99, name='full',       scans=12, scan_seconds=192*12))

    def __init__(self, meta_data=None, create_directories=True, create_mode=0o755):
        if meta_data is None: meta_data = BensonWinawer2018Dataset.prf_meta_data
        else: meta_data = pimms.merge(BensonWinawer2018Dataset.prf_meta_data, meta_data)
        Dataset.__init__(self, 'benson_winawer_2018',
                         meta_data=meta_data,
                         custom_directory=config['benson_winawer_2018_path'],
                         create_directories=create_directories,
                         create_mode=create_mode)
    @staticmethod
    def download(path, create_directories=True, mode=0o755, overwrite=False):
        '''
        BensonWinawer2018Dataset.download(path) downloads the Benson and Winawer (2018) dataset into
          the directory given by path. If the dataset is already found there, then it will not be
          overwritten.

        The following optional parameters may be provided:
          * create_directories (default: True) may be set to False to indicate that the path should
            not be created if it does not already exist.
          * mode (default: 0o755) specifies the permissions that should be used if the directory is
            created.
          * overwrite (default: False) may be set to True to indicate that the dataset should be
            overwritten if it is already found.
        '''
        dataset_urls = BensonWinawer2018Dataset.dataset_urls
        if not os.path.isdir(path):
            if not create_directories: raise ValueError('Path given to download() does not exist')
            elif not os.path.isdir(path): os.makedirs(os.path.abspath(path), mode)
        if not overwrite:
            if all(os.path.isdir(os.path.join(path, x)) for x in six.iterkeys(dataset_urls)):
                return path
            elif any(os.path.isdir(os.path.join(path, x)) for x in six.iterkeys(dataset_urls)):
                raise ValueError('some but not all of dataset already downloaded')
        # okay, go through the urls...
        logging.info('neuropythy: Downloading Benson and Winawer (2018) data from osf.io...')
        for (dirname, durl) in six.iteritems(dataset_urls):
            # download the url...
            tgz_file = os.path.join(path, dirname + '.tar.gz')
            logging.info('neuropythy: Fetching "%s"', tgz_file)
            if six.PY2:
                response = urllib.request.urlopen(durl)
                with open(tgz_file, 'wb') as fl:
                    shutil.copyfileobj(response, fl)
            else:
                with urllib.request.urlopen(durl) as response:
                    with open(tgz_file, 'wb') as fl:
                        shutil.copyfileobj(response, fl)
            if not tarfile.is_tarfile(tgz_file):
                raise ValueError('Error when downloading %s: not a tar file' % tgz_file)
            # now unzip it...
            logging.info('neuropythy: Extracting to "%s"', tgz_file)
            with tarfile.open(tgz_file, 'r:gz') as tar:
                tar.extractall(path)
                tar.close()
            # and delete the tar.gz file
            os.remove(tgz_file)
        # That's all!
        return path

    subject_properties = pimms.merge(
        # retinotopy data
        {('%s_%s' % (dset, pname)): os.path.join(
            'retinotopy', '{0[sub]}',
            ('{0[hemi]}' + 
             (('_%02d:%02d_' % (dsmeta['id'], dsmeta['scans'])) if dsmeta['id'] != 99 else '_') + 
             pname_file + '.mgz'))
         for (dset, dsmeta)     in six.iteritems(prf_meta_data)
         for (pname,pname_file) in six.iteritems({'polar_angle':'angle', 'eccentricity':'eccen',
                                                  'radius':'prfsz', 'variance_explained':'vexpl'})},
        # wide-field dataset (will be ignored for subjects other than S1201)
        {('wide-prf_%s' % pname): os.path.join(
            'retinotopy', '{0[sub]}',
            ('{0[hemi]}_widef_' + pname_file + '.mgz'))
         for (pname,pname_file) in six.iteritems({'polar_angle':'angle', 'eccentricity':'eccen',
                                                  'radius':'prfsz', 'variance_explained':'vexpl'})},
        # analyses data
        {('inf-%s_%s' % (dset, pname)): os.path.join(
            'analyses', '{0[sub]}',
            ('{0[hemi]}' + 
             ('.%02d.%s_steps=02500_scale=20.00_clip=12_prior=retinotopy.mgz' % (dsmeta['id'],
                                                                                 pname_file))))
         for (dset, dsmeta)     in six.iteritems(prf_meta_data)
         for (pname,pname_file) in six.iteritems({'polar_angle':'angle', 'eccentricity':'eccen',
                                                  'radius':'sigma', 'visual_area':'varea'})},
        # Benson14 data
        {('prior-prf_%s' % pname): os.path.join(
            'analyses', '{0[sub]}', ('{0[hemi]}.benson14_' + pname_file + '.mgz'))
         for (pname, pname_file) in six.iteritems({'polar_angle':'angle', 'eccentricity':'eccen',
                                                   'radius':'sigma', 'visual_area':'varea'})})
    fsaverage_properties = pimms.merge(
        # retinotopy data
        {('prf_%s' % pname): os.path.join(
            'retinotopy', '{0[sub]}', '{0[hemi]}_' + pname_file + '.mgz')
         for (pname,pname_file) in six.iteritems({'polar_angle':'angle', 'eccentricity':'eccen',
                                                  'radius':'prfsz', 'variance_explained':'vexpl'})},
        # analyses data
        {('prior_%s' % pname): os.path.join(
            'analyses', '{0[sub]}', '{0[hemi]}.anatomical-prior_' + pname_file + '.mgz')
         for (pname,pname_file) in six.iteritems({'polar_angle':'angle', 'eccentricity':'eccen',
                                                  'radius':'sigma', 'visual_area':'varea'})})
    subject_registrations = pyr.pmap(
        {('%s_retinotopy' % dset): os.path.join(
            'analyses', '{0[sub]}',
            ('{0[hemi]}' + 
             ('.%02d.retinotopy_steps=02500_scale=20.0_clip=12_prior=retinotopy.sphere.reg' %
                 dsmeta['id'])))
         for (dset, dsmeta) in six.iteritems(prf_meta_data)})
    fsaverage_registrations = pyr.pmap(
        {'retinotopy': os.path.join('analyses', '{0[sub]}', '{0[hemi]}.retinotopy.sphere.reg')})

    @staticmethod
    def load_subject(cache_directory, sid):
        '''
        BensonWinawer2018Dataset.load_subject(dir, subjid) loads the given subject ID from the given
          Benson and Winawer (2018) cache directory. This directory must contain the relevant
          freesurfer_subjects/, retinotopy/, and analyses/, directories (they should be
          auto-downloaded if accessed via the databases interface).
        '''
        if pimms.is_int(sid): sid = 'S12%02d' % sid
        sub = freesurfer_subject(os.path.join(cache_directory, 'freesurfer_subjects', sid))
        # okay, we need functions that will lazily extract a hemisphere then load the retinotopy,
        # analyses, and atlas data onto it (also lazily)
        def _load_ints(flnm):  return np.asarray(nyio.load(flnm), dtype=np.int)
        def _load_angle(flnm):
            dat = nyio.load(flnm)
            (d,n) = os.path.split(flnm)
            d = os.path.split(d)[0]
            return -dat if d.endswith('analyses') and n.startswith('rh') else dat
        def update_hemi(subname, hemis, hname):
            # get the original hemisphere...
            hemi = hemis[hname]
            stup = {'sub':subname, 'hemi':hname}
            pdat = {}
            sprops = (BensonWinawer2018Dataset.fsaverage_properties if subname == 'fsaverage' else
                      BensonWinawer2018Dataset.subject_properties)
            # okay, now we want to load a bunch of data; start with properties
            for (propname, filename) in six.iteritems(sprops):
                filename = os.path.join(cache_directory, filename.format(stup))
                if not os.path.isfile(filename): continue
                pdat[propname] = curry(
                    (_load_ints  if propname.endswith('visual_area') else
                     _load_angle if propname.endswith('polar_angle') else
                     nyio.load),
                    filename)
            # we can add this already...
            hemi = hemi.with_prop(pimms.lazy_map(pdat))
            # next, we want to grab the registrations...
            rdat = {}
            sregs = (BensonWinawer2018Dataset.fsaverage_registrations if subname == 'fsaverage' else
                     BensonWinawer2018Dataset.subject_registrations)
            for (rname, filename) in six.iteritems(sregs):
                filename = os.path.join(cache_directory, filename.format(stup))
                if not os.path.isfile(filename): continue
                rdat[rname] = curry(nyio.load, filename, 'freesurfer_geometry')
            hemi = hemi.copy(_registrations=pimms.merge(hemi.registrations, pimms.lazy_map(rdat)))
            # that's all
            return hemi
        # okay, update the hemi's map with a curried version of the above and return...
        hemis = reduce(lambda h,hname: h.set(hname, curry(update_hemi, sub.name, sub.hemis, hname)),
                       ['lh','rh'],
                       sub.hemis)
        return sub.copy(hemis=hemis)
    @pimms.value
    def subjects(cache_directory, create_directories, create_mode):
        '''
        dataset.subjects is a lazy persistent map of all the subjects that are part of the
        Benson and Winawer (2018) dataset.
        '''
        # make sure the data are downloaded
        BensonWinawer2018Dataset.download(cache_directory, create_directories=create_directories,
                                          mode=create_mode, overwrite=False)
        # okay, next we want to setup the subjects
        dat = {s:curry(BensonWinawer2018Dataset.load_subject, cache_directory, s)
               for s in [('S12%02d' % s) for s in range(1,9)]}
        dat['fsaverage'] = curry(BensonWinawer2018Dataset.load_subject,
                                 cache_directory, 'fsaverage')
        return pimms.lazy_map(dat)
    @pimms.value
    def v123_table(cache_directory, subjects):
        '''
        dataset.v123_table is a pimms ITable object for the BensonWinawer2018Dataset; the table
        contains all relevant pRF data for all cortical surface vertices in the 8 subjects included
        in the paper Benson and Winawer (2018).
        '''
        # First, see if there's a cache file
        cachefl = os.path.join(cache_directory, 'v123_table.p')
        if os.path.isfile(cachefl):
            try: return pimms.load(cachefl)
            except Exception:
                msg = 'neuropythy: Could not load existing v123_table cache file: %s' % cache_file
                warnings.warn(msg)
        # go through, building up arrays of arrays that we will concatenate at the end
        data = AutoDict()
        data.on_miss = lambda:[] # we want it to auto-produce lists...
        # non-retinotopy props we want to add to the data...
        props = ['midgray_surface_area', 'pial_surface_area', 'white_surface_area', 'label']
        for (sid,sub) in six.iteritems(subjects):
            for hname in ['lh','rh']:
                hemi = sub.hemis[hname]
                for (dskey,dsdata) in six.iteritems(BensonWinawer2018Dataset.prf_meta_data):
                    dsid = 99 if dskey == 'prf' else int(dskey[3:])
                    # okay, let's get the raw data we need to process...
                    ang = hemi.prop(dskey + '_polar_angle')
                    ecc = hemi.prop(dskey + '_eccentricity')
                    # and the inferred data...
                    iang = hemi.prop('inf-' + dskey + '_polar_angle')
                    iecc = hemi.prop('inf-' + dskey + '_eccentricity')
                    ilbl = hemi.prop('inf-' + dskey + '_visual_area')
                    # process both of these (get x/y basically)
                    (x, y)  = as_retinotopy({'polar_angle':ang,  'eccentricity':ecc},
                                            'geographical')
                    (ix,iy) = as_retinotopy({'polar_angle':iang, 'eccentricity':iecc},
                                            'geographical')
                    # find the relevant vertices
                    ii = np.where((iecc < 12) & np.sum([ilbl == k for k in (1,2,3)], axis=0))[0]
                    # now add the relevant properties...
                    for p in props: data[p].append(hemi.prop(p)[ii])
                    for p0 in ['polar_angle', 'eccentricity', 'radius', 'variance_explained']:
                        p = dskey + '_' + p0
                        data[p0].append(hemi.prop(p)[ii])
                    for (p,u) in zip(['x', 'y'], [x,y]):
                        data[p].append(u)
                    for p0 in ['_polar_angle', '_eccentricity', '_radius', '_visual_area']:
                        p = 'inf-' + dskey + p0
                        data['inf' + p0].append(hemi.prop(p)[ii])
                    for (p0,u) in zip(['inf_x', 'inf_y'], [ix,iy]):
                        data[p0].append(u)
                    # we also want repeated properties for some things
                    extras = {'subject':sid, 'hemi':hname, 'dataset_id':dsid, 'dataset_name':dskey}
                    for (p,v) in six.iteritems(extras): data[p].append(np.full(len(ii), v))
        # concatenate everything
        data = pimms.itable({k:np.concatenate(v) for (k,v) in six.iteritems(data)})
        if not os.path.isfile(cachefl):
            # try to write out the cache file
            try: pimms.save(cachefl, data)
            except Exception: pass
        return data
    
# we wrap this in a lambda so that it gets loaded when requested (in case the config changes between
# when this gets run and when the dataset gets requested)
add_dataset('benson_winawer_2018', lambda:BensonWinawer2018Dataset().persist())

