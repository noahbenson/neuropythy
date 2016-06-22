#! /usr/bin/env python

from setuptools import setup

setup(
    name='neuropythy',
    version='0.1.1',
    description='Toolbox for flexible cortical mesh analysis and registration',
    keywords='neuroscience mesh cortex registration',
    author='Noah C. Benson',
    author_email='nben@nyu.edu',
    url='https://github.com/noahbenson/neuropythy/',
    license='GPLv3',
    packages=['neuropythy',
              'neuropythy.geometry', 
              'neuropythy.immutable',
              'neuropythy.topology',
              'neuropythy.freesurfer',
              'neuropythy.cortex',
              'neuropythy.registration'],
    package_data={'': ['LICENSE.txt',
                       'lib/nben/target/nben-standalone.jar',
                       'lib/models/standard.ffm.gz']},
    install_requires=['numpy>=1.2',
                      'scipy>=0.7',
                      'nibabel>=2.0',
                      'pysistence>=0.4',
                      'py4j>=0.9',
                      'igraph>=0.7.1'])
