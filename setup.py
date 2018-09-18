#! /usr/bin/env python

import os
from setuptools import setup

# Deduce the version from the __init__.py file:
version = None
with open(os.path.join(os.path.dirname(__file__), 'neuropythy', '__init__.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None: raise ValueError('No version found in neuropythy/__init__.py!')

setup(
    name='neuropythy',
    version=version,
    description='Toolbox for flexible cortical mesh analysis and registration',
    keywords='neuroscience mesh cortex registration',
    author='Noah C. Benson',
    author_email='nben@nyu.edu',
    maintainer_email='nben@nyu.edu',
    long_description='''
                     See the README.md file at the github repository for this package:
                     https://github.com/noahbenson/neuropythy
                     ''',
    url='https://github.com/noahbenson/neuropythy',
    download_url='https://github.com/noahbenson/neuropythy',
    license='GPLv3',
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
                 'Programming Language :: Python :: 2',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Topic :: Software Development',
                 'Topic :: Software Development :: Libraries',
                 'Topic :: Software Development :: Libraries :: Python Modules',
                 'Topic :: Scientific/Engineering',
                 'Topic :: Scientific/Engineering :: Information Analysis',
                 'Topic :: Scientific/Engineering :: Medical Science Apps.',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS'],
    packages=['neuropythy',
              'neuropythy.util',
              'neuropythy.java',
              'neuropythy.io',
              'neuropythy.geometry', 
              'neuropythy.mri',
              'neuropythy.freesurfer',
              'neuropythy.hcp',
              'neuropythy.registration',
              'neuropythy.vision',
              'neuropythy.graphics',
              'neuropythy.datasets',
              'neuropythy.commands',
              'neuropythy.test'],
    include_package_data=True,
    package_data={
        '': ['LICENSE.txt',
             'neuropythy/lib/nben/target/nben-standalone.jar',
             'neuropythy/lib/models/v123.fmm.gz',
             'neuropythy/lib/models/lh.benson17.fmm.gz',
             'neuropythy/lib/models/rh.benson17.fmm.gz',
             'neuropythy/lib/data/fsaverage/surf/lh.retinotopy_benson17.sphere.reg',
             'neuropythy/lib/data/fsaverage/surf/lh.benson17_angle',
             'neuropythy/lib/data/fsaverage/surf/lh.benson17_eccen',
             'neuropythy/lib/data/fsaverage/surf/lh.benson17_sigma',
             'neuropythy/lib/data/fsaverage/surf/lh.benson17_varea',
             'neuropythy/lib/data/fsaverage/surf/rh.retinotopy_benson17.sphere.reg',
             'neuropythy/lib/data/fsaverage/surf/rh.benson17_angle',
             'neuropythy/lib/data/fsaverage/surf/rh.benson17_eccen',
             'neuropythy/lib/data/fsaverage/surf/rh.benson17_sigma',
             'neuropythy/lib/data/fsaverage/surf/rh.benson17_varea',
             'neuropythy/lib/data/fsaverage_sym/surf/lh.retinotopy_benson14.sphere.reg',
             'neuropythy/lib/data/fsaverage_sym/surf/sym.benson14_angle',
             'neuropythy/lib/data/fsaverage_sym/surf/sym.benson14_eccen',
             'neuropythy/lib/data/fsaverage_sym/surf/sym.benson14_sigma',
             'neuropythy/lib/data/fsaverage_sym/surf/sym.benson14_varea']},
    install_requires=['numpy>=1.5',
                      'scipy>=0.9',
                      'nibabel>=2.0',
                      'pyrsistent>=0.11',
                      'pint>=0.7',
                      'pimms>=0.2.13',
                      'py4j>=0.10'],
    extras_require={
        'HCP': ['s3fs>=0.1.5', 'h5py>=2.8.0'],
        'graphics2D': ['matplotlib>=1.5.3'],
        'graphics3D': ['matplotlib>=1.5.3', 'ipyvolume>=0.5.1'],
        'all': ['s3fs>=0.1.5', 'h5py>=2.8.0', 'matplotlib>=1.5.3', 'ipyvolume>=0.5.1']})
