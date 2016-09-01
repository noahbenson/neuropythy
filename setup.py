#! /usr/bin/env python

from setuptools import setup

setup(
    name='neuropythy',
    version='0.1.4',
    description='Toolbox for flexible cortical mesh analysis and registration',
    keywords='neuroscience mesh cortex registration',
    author='Noah C. Benson',
    author_email='nben@nyu.edu',
    url='https://github.com/noahbenson/neuropythy/',
    license='GPLv3',
    packages=['neuropythy',
              'neuropythy.util',
              'neuropythy.java',
              'neuropythy.geometry', 
              'neuropythy.immutable',
              'neuropythy.topology',
              'neuropythy.freesurfer',
              'neuropythy.cortex',
              'neuropythy.registration',
              'neuropythy.vision',
              'neuropythy.commands'],
    package_data={'': ['LICENSE.txt',
                       'neuropythy/lib/nben/target/nben-standalone.jar',
                       'neuropythy/lib/models/standard.fmm.gz']},
    include_package_data=True,
    install_requires=['numpy>=1.2',
                      'scipy>=0.7',
                      'nibabel>=2.0',
                      'pysistence>=0.4',
                      'py4j>=0.9',
                      'python-igraph>=0.7.1'])
