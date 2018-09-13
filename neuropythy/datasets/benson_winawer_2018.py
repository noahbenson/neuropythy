####################################################################################################
# neuropythy/datasets/benson_winawer_2018.py
# The dataset from Benson and Winawer (2018); DOI: https://doi.org/10.1101/325597
# by Noah C. Benson

import os, six, shutil, urllib, tarfile, logging, pimms
import numpy as np
import pyrsistent as pyr

if six.PY3: from functools import reduce

from .core        import (Dataset, add_dataset)
from ..util       import (config, curry)
from ..           import io      as nyio
from ..freesurfer import subject as freesurfer_subject

config.declare('benson_winawer_2018_path')

@pimms.immutable
class BensonWinawer2018Dataset(Dataset):
    '''
    neuropythy.data['benson_winawer_2018'] is a Dataset containing the publicly provided data from
    the following publication:

    Benson NC, Winawer J (2018) Bayesian analysis of retinotopic maps. BioRxiv. DOI:10.1101/325597

    These data include 8 FreeSurfer subjects each with a set of measured and inferred retinotopic
    maps. These data are provided as follows:
    
    dset = neuropythy.data['benson_winawer_2018']
    sorted(dset.subjects.keys())
    #=> ['S1201', 'S1202', 'S1203', 'S1204', 'S1205', 'S1206', 'S1207', 'S1208']
    sorted(dset.subjects['S1201'].lh.properties.keys())

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
            else: os.makedirs(path, mode)
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
            logging.info('            Fetching "%s"', tgz_file)
            with urllib.request.urlopen(durl) as response:
                with open(tgz_file, 'wb') as fl:
                    shutil.copyfileobj(response, fl)
            if not tarfile.is_tarfile(tgz_file):
                raise ValueError('Error when downloading %s: not a tar file' % tgz_file)
            # now unzip it...
            logging.info('            Extracting to "%s"', tgz_file)
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
    subject_registrations = pyr.pmap(
        {('%s_retinotopy' % dset): os.path.join(
            'analyses', '{0[sub]}',
            ('{0[hemi]}' + 
             ('.%02d.retinotopy_steps=02500_scale=20.0_clip=12_prior=retinotopy.sphere.reg' %
                 dsmeta['id'])))
         for (dset, dsmeta) in six.iteritems(prf_meta_data)})

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
        def update_hemi(subname, hemis, hname):
            # get the original hemisphere...
            hemi = hemis[hname]
            stup = {'sub':subname, 'hemi':hname}
            pdat = {}
            # okay, now we want to load a bunch of data; start with properties
            for (propname, filename) in six.iteritems(BensonWinawer2018Dataset.subject_properties):
                filename = os.path.join(cache_directory, filename.format(stup))
                if not os.path.isfile(filename): continue
                pdat[propname] = curry(nyio.load, filename)
            # we can add this already...
            hemi = hemi.with_prop(pimms.lazy_map(pdat))
            # next, we want to grab the registrations...
            rdat = {}
            for (rname, filename) in six.iteritems(BensonWinawer2018Dataset.subject_registrations):
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
        return pimms.lazy_map({s:curry(BensonWinawer2018Dataset.load_subject, cache_directory, s)
                               for s in [('S12%02d' % s) for s in range(1,9)]})
    
# we wrap this in a lambda so that it gets loaded when requested (in case the config changes between
# when this gets run and when the dataset gets requested)
add_dataset('benson_winawer_2018', lambda:BensonWinawer2018Dataset())

