####################################################################################################
# neuropythy/images/core.py
# This file implements the core code used to manipulate and import/export images in neuropythy.
# by Noah C. Benson

import types, inspect, pimms, os, six, sys
import numpy                            as np
import scipy.sparse                     as sps
import pyrsistent                       as pyr
import nibabel                          as nib
import nibabel.freesurfer.mghformat     as fsmgh
from   functools                    import reduce

from ..util import (to_affine, is_image, is_image_header, is_tuple, curry, to_hemi_str)

# conversion to nibabel images and nibabel image headers
def to_spatial_image_header(): pass
def to_minc1_image_header(): pass
def to_minc2_image_header(): pass
def to_nifti1_image_header(): pass
def to_nifti2_image_header(): pass
def to_parrec_image_header(): pass
def to_spm2analyze_image_header(): pass
def to_spm99analyze_image_header(): pass
def to_mgh_image_header():
    if True: raise NotImplementedError()
    else:
        # these are the struct array data for MGH header files
        structarr['version'] = 1
        structarr['dims'] = 1
        structarr['type'] = 3
        structarr['goodRASFlag'] = 1
        structarr['delta'] = 1
        structarr['Mdc'] = [[-1, 0, 0], [0, 0, 1], [0, -1, 0]]
def to_ecat_image_header(): pass
def to_image_header(obj, type='nifti1'):
    '''
    to_image_header(obj) attempts to coerce the given object to a nibabel Nifti1Header object and
      yields that object or raises an exception.
    to_image_header(obj, header_type) attempts to coerce the object into the given header type. For
      a list of possible header types, see to_image_header.types.

    The following objects can be coerced into header objects:
      * another header object (the identical object will be returned if appropriate)
      * an affine transformation
      * a dictionary or mapping containing keys appropriate to the header; usually these are
       'affine' and any other optional values.
      * a tuple (affine, options_mapping).
    '''
    raise NotImplementedError()

def to_spatial_image(): pass
def to_minc1_image(): pass
def to_minc2_image(): pass
def to_nifti1_image(): pass
def to_nifti2_image(): pass
def to_parrec_image(): pass
def to_spm2analyze_image(): pass
def to_spm99analyze_image(): pass
def to_mgh_image(): pass
def to_ecat_image(): pass
