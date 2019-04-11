####################################################################################################
# neuropythy/mri/images.py
# This file implements the core code used to manipulate and import/export images in neuropythy.
# by Noah C. Benson

import types, inspect, pimms, os, six, warnings, sys
import numpy                            as np
import scipy.sparse                     as sps
import pyrsistent                       as pyr
import nibabel                          as nib
import nibabel.freesurfer.mghformat     as fsmgh
from   functools                    import reduce

from ..util import (to_affine, is_image, is_image_header, is_tuple, curry, to_hemi_str)

# handy function for getting an image header
def to_image_header(img):
    '''
    to_image_header(img) yields img.header if img is a nibabel image object.
    to_image_header(hdr) yields hdr if hdr is a nibabel header object.
    to_image_header(obj) raises an error for other input types.
    '''
    if not img.__module__.startswith('nibabel.'):
        raise ValueError('to_image_header: only nibabel obejcts can be coerced to headers')
    if type(img).__name__.endswith('Header'): return img
    # if not a header given, must be an image given:
    try: return img.header
    except Exception:
        raise ValueError('to_image_header: can only convert nibabel image or header objects')

# ImageType Classes ################################################################################
# These classes define how images are interpreted and loaded; they are intended to be static.
class ImageType(object):
    '''
    The ImageType class defines how nibabel image types are handled by the neuropythy.mri.images
    subsystem.
    '''
    def __init__(self): raise NotImplementedError('ImageType is a static class')
    @classmethod
    def image_name(self): raise NotImplementedError('ImageType.image_name must be overloaded')
    @classmethod
    def image_type(self): raise NotImplementedError('ImageType.image_type must be overloaded')
    @classmethod
    def header_type(self): raise NotImplementedError('ImageType.header_type must be overloaded')
    @classmethod
    def meta_data(self, x): raise NotImplementedError('ImageType.meta_data must be overloaded')
    @classmethod
    def aliases(self): return ()
    @classmethod
    def default_type(self): return np.dtype('float32')
    @classmethod
    def default_affine(self): return np.eye(4)
    @classmethod
    def parse_type(self, hdat, dataobj=None):
        '''
        Parses the dtype out of the header data or the array, depending on which is given; if both,
          then the header-data overrides the array; if neither, then np.float32.
        '''
        try:    dataobj = dataobj.dataobj
        except Exception: pass
        dtype = np.asarray(dataobj).dtype if dataobj else self.default_type()
        if   hdat and 'type'  in hdat: dtype = np.dtype(hdat['type'])
        elif hdat and 'dtype' in hdat: dtype = np.dtype(hdat['dtype'])
        return dtype
    @classmethod
    def parse_affine(self, hdat, dataobj=None):
        '''
        Parses the affine out of the given header data and yields it.
        '''
        if 'affine' in hdat: return to_affine(hdat['affine'])
        else:                return to_affine(self.default_affine())
    @classmethod
    def parse_dataobj(self, dataobj, hdat={}):
        # first, see if we have a specified shape/size
        ish = next((hdat[k] for k in ('image_size', 'image_shape', 'shape') if k in hdat), None)
        if ish is Ellipsis: ish = None
        # make a numpy array of the appropriate dtype
        dtype = self.parse_type(hdat, dataobj=dataobj)
        try:    dataobj = dataobj.dataobj
        except Exception: pass
        if   dataobj: arr = np.asarray(dataobj).astype(dtype)
        elif ish:     arr = np.zeros(ish,       dtype=dtype)
        else:         arr = np.zeros([1,1,1,0], dtype=dtype)
        # reshape to the requested shape if need-be
        if ish and ish != arr.shape: arr = np.reshape(arr, ish)
        # then reshape to a valid (4D) shape
        sh = arr.shape
        if   len(sh) == 2: arr = np.reshape(arr, (sh[0], 1, 1, sh[1]))
        elif len(sh) == 1: arr = np.reshape(arr, (sh[0], 1, 1, 1))
        elif len(sh) == 3: arr = np.reshape(arr, sh + (1,))
        elif len(sh) != 4: raise ValueError('Cannot convert n-dimensional array to image if n > 4')
        # and return
        return arr
    @classmethod
    def parse_kwargs(self, arr, hdat={}):
        ext = hdat.get('extra', {})
        for (k,v) in six.iteritems(hdat):
            if k in ['header', 'extra', 'file_map']: continue
            ext[k] = v
        kw = {'extra': ext} if len(ext) > 0 else {}
        if 'header' in hdat: kw['header'] = hdat['header']
        if 'file_map' in hdat: kw['file_map'] = hdat['file_map']
        return kw
    @classmethod
    def to_image(self, arr, hdat={}):
        # reshape the data object or create it empty if None was given:
        arr = self.parse_dataobj(arr, hdat)
        # get the affine
        aff = self.parse_affine(hdat, dataobj=arr)
        # get the keyword arguments
        kw = self.parse_kwargs(arr, hdat)
        # create an image of the appropriate type
        cls = self.image_type()
        img = cls(arr, aff, **kw)
        # post-process the image
        return self.postprocess_image(img, hdat)
    @classmethod
    def postprocess_image(self, img, hdat={}): return img
    @classmethod
    def create(self, arr, meta_data={}, **kwargs):
        '''
        itype.create(dataobj) yields an image of the given image type itype that represents the
          given data object dataobj.
        itype.create(dataobj, meta_data) uses the given meta/header data to create the image.

        Any number of keyword arguments may also be appended to the call; these are merged into the
        meta_data argument.
        '''
        return self.to_image(arr, hdat=pimms.merge(meta_data, kwargs))
    @classmethod
    def meta_data(self, img):
        hdr = to_image_header(img)
        d = {}
        # basic stuff (most headers should have these)
        try: d['affine'] = hdr.get_best_affine()
        except Exception:
            try:    d['affine'] = hdr.get_affine()
            except Exception: pass
        try: d['voxel_size'] = hdr.get_zooms()
        except Exception: pass
        try: d['voxel_type'] = hdr.get_data_dtype()
        except Exception: pass
        try: d['image_shape'] = hdr.get_data_shape()
        except Exception: pass
        # less basic stuff (some have these)
        try: d['bytes_per_voxel'] = hdr.get_data_bytespervox()
        except Exception: pass
        try: d['image_bytes'] = hdr.get_data_size()
        except Exception: pass
        try: d['image_offset'] = hdr.get_data_offset()
        except Exception: pass
        try:
            (m,b) = hdr.get_slope_inter()
            d['data_slope']  = m
            d['data_offset'] = b
        except Exception: pass
        # that's it
        return d
class MGHImageType(ImageType):
    @classmethod
    def image_name(self): return 'mgh'
    @classmethod
    def image_type(self): return fsmgh.MGHImage
    @classmethod
    def header_type(self): return fsmgh.MGHHeader
    @classmethod
    def default_affine(self): return to_affine([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0]], 3)
    @classmethod
    def default_type(self): return np.dtype('float32')
    @classmethod
    def aliases(self): return ('mgh', 'mgz', 'mgh.gz', 'freesurfer_image', 'freesurfer')
    @classmethod
    def parse_type(self, hdat, dataobj=None):
        dtype = super(MGHImageType, self).parse_type(hdat, dataobj=dataobj)
        if   np.issubdtype(dtype, np.floating): dtype = np.float32
        elif np.issubdtype(dtype, np.int8):     dtype = np.int8
        elif np.issubdtype(dtype, np.int16):    dtype = np.int16
        elif np.issubdtype(dtype, np.integer):  dtype = np.int32
        else: raise ValueError('Could not deduce appropriate MGH type for dtype %s' % dtype)
        return dtype
    @classmethod
    def meta_data(self, img):
        d = super(self, MGHImageType).meta_data(img)
        hdr = to_image_header(img)
        try:    d['vox2ras'] = hdr.get_vox2ras()
        except Exception: pass
        try:    d['ras2vox'] = hdr.get_ras2vox()
        except Exception: pass
        try:    d['vox2ras_tkr'] = hdr.get_vox2ras_tkr()
        except Exception: pass
        try:    d['footer_offset'] = hdr.get_footer_offset()
        except Exception: pass
        try:
            zooms = hdr.get_zooms()
            (zooms, tr) = (zooms, None) if len(zooms) == 3 else (zooms[:3], zooms[3])
            d['voxel_size']     = pimms.quant(zooms, 'mm')
            d['slice_duration'] = pimms.quant(tr, 'ms') if tr is not None else None
        except Exception: pass
        return d
class Nifti1ImageType(ImageType):
    @classmethod
    def image_name(self): return 'nifti1'
    @classmethod
    def image_type(self): return nib.Nifti1Image
    @classmethod
    def header_type(self): return nib.Nifti1Header
    @classmethod
    def aliases(self): return ('nii', 'nii.gz', 'nifti')
    @classmethod
    def meta_data(self, img):
        from nibabel.nifti1 import (slice_order_codes, unit_codes)
        d = super(self, Nifti1ImageType).meta_data(img)
        hdr = to_image_header(img)
        try: d['dimension_information'] = hdr.get_dim_info()
        except Exception: pass
        try: d['intent'] = hdr.get_intent()
        except Exception: pass
        try: d['slice_count'] = hdr.get_n_slices()
        except Exception: pass
        try: (sunit, tunit) = hdr.get_xyzt_units()
        except Exception: (sunit, tunit) = ('unknown', 'unknown')
        if sunit != 'unknown':
            try: d['voxel_size'] = pimms.quant(d['voxel_size'], sunit)
            except Exception: pass
        try:
            sd = hdr.get_slice_duration()
            if tunit != 'unknown': sd = pimms.quant(sd, tunit)
            d['slice_duration'] = sd
        except Exception: pass
        try:
            (q,qc) = hdr.get_qform(True)
            d['qform_code'] = qc
            d['qform'] = q
        except Exception: pass
        try:
            (s,sc) = hdr.get_sform(True)
            d['sform_code'] = sc
            d['sform'] = s
        except Exception: pass
        try:
            sc = hdr['slice_code']
            if sc != 0: d['slice_order'] = slice_order_codes.label[sc]
        except Exception: pass
        try:
            ts = hdr.get_slice_times()
            ts = np.asarray([np.nan if t is None else t for t in ts])
            if tunit != 'unknown': ts = pimms.quant(ts, tunit)
            d['slice_times'] = ts
        except Exception: pass
        try:    d['header_size'] = hdr['sizeof_hdr']
        except Exception: pass
        try:    d['calibration'] = (hdr['cal_min'], hdr['cal_max'])
        except Exception: pass
        try:
            t0 = hdr['toffset']
            if tunits != 'unknown': t0 = pimms.quant(t0, tunits)
            d['time_offset'] = t0
        except Exception: pass
        try:    d['description'] = hdr['descrip']
        except Exception: pass
        try:    d['auxiliary_filename'] = hdr['aux_file']
        except Exception: pass
        return d
    @classmethod
    def unit_to_name(self, u):
        for name in ('meter', 'mm', 'sec', 'msec', 'usec'):
            if u == pimms.unit(name):
                return name
        # no current support for ppm designation
        if   u == pimms.unit('um'):     return 'micron'
        elif u == pimms.unit('1/sec'):  return 'hz'
        elif u == pimms.unit('Hz'):     return 'hz'
        elif u == pimms.unit('radian'): return 'rads'
        else: return 'unknown'
    @classmethod
    def postprocess_image(self, img, d):
        from nibabel.nifti1 import slice_order_codes
        hdr = img.header
        # dimension information:
        for k in ['dimension_information', 'dim_info', 'diminfo']:
            try:
                hdr.set_dim_info(*d[k])
                break
            except Exception: pass
        try:    hdr.set_intent(d['intent'])
        except Exception: pass
        # xyzt_units:
        try: sunit = self.unit_to_name(pimms.unit(d['voxel_size']))
        except Exception:
            try: sunit = self.unit_to_name(pimms.unit(d['voxel_unit']))
            except Exception: sunit = 'unknown'
        try: tunit = self.unit_to_name(pimms.unit(d['slice_duration']))
        except Exception:
            try: tunit = self.unit_to_name(pimms.unit(d['time_unit']))
            except Exception: tunit = 'unknown'
        try: hdr.set_xyzt_units(sunit, tunit)
        except Exception: pass
        # qform and sform
        try:
            try: q = to_affine(d['qform'])
            except Exception: q = hdr.get_best_affine()
            qc = d.get('qform_code', 'unknown')
            hdr.set_qform(q, qc)
        except Exception: pass
        try:
            try: s = to_affine(d['sform'])
            except Exception: s = hdr.get_best_affine()
            sc = d.get('sform_code', 'unknown')
            hdr.set_sform(s, sc)
        except Exception: pass
        # slice code
        try:    hdr['slice_code'] = slice_order_codes[d['slice_order']]
        except Exception: pass
        # slice duration
        try:
            dur = d['slice_duration']
            if pimms.is_quantity(dur):
                if tunit == 'unknown': dur = pimms.mag(dur)
                else: dur = pimms.mag(dur, tunit)
            hdr.set_slice_duration(dur)
        except Exception: pass
        # slice timing
        try:
            ts = d['slice_times']
            if pimms.is_quantity(ts):
                if tunit == 'unknown': ts = pimms.mag(ts)
                else: ts = pimms.mag(ts, tunit)
            hdr.set_slice_duration([None if np.isnan(t) else t for t in ts])
        except Exception: pass
        # slope / intercept
        try:    hdr.set_slope_inter(d.get('data_slope', None), d.get('data_offset', None))
        except Exception: pass
        # calibration
        try:
            (cmn, cmx) = d['calibration']
            hdr['cal_min'] = cmn
            hdr['cal_max'] = cmx
        except Exception: pass
        # time offset
        try:
            t0 = d['time_offset']
            if pimms.is_quantity(t0):
                if tunits != 'unknown': t0 = pimms.mag(t0, tunits)
                else: t0 = pimms.mag(t0)
            hdr['toffset'] = t0
        except Exception: pass
        # description
        try:    hdr['descrip'] = d['description']
        except Exception: pass
        # auxiliary filename
        try:    hdr['aux_file'] = d['auxiliary_filename']
        except Exception: pass
        return img
class Nifti2ImageType(Nifti1ImageType):
    @classmethod
    def image_name(self): return 'nifti2'
    @classmethod
    def image_type(self): return nib.Nifti2Image
    @classmethod
    def header_type(self): return nib.Nifti2Header
    @classmethod
    def aliases(self): return ('nii2', 'nii2.gz')
class Spm99AnalyzeImageType(ImageType):
    @classmethod
    def image_name(self): return 'spm99analyze'
    @classmethod
    def image_type(self): return nib.Spm99AnalyzeImage
    @classmethod
    def header_type(self): return nib.Spm99AnalyzeHeader
class Spm2AnalyzeImageType(Spm99AnalyzeImageType):
    @classmethod
    def image_name(self): return 'analyze'
    @classmethod
    def image_type(self): return nib.Spm2AnalyzeImage
    @classmethod
    def header_type(self): return nib.Spm2AnalyzeHeader
    @classmethod
    def aliases(self): return ('spm2analyze')
class Minc1ImageType(ImageType):
    @classmethod
    def image_name(self): return 'minc1'
    @classmethod
    def image_type(self): return nib.minc1.Minc1Image
    @classmethod
    def header_type(self): return nib.minc1.Minc1Header
    @classmethod
    def aliases(self): return ()
class Minc2ImageType(ImageType):
    @classmethod
    def image_name(self): return 'minc'
    @classmethod
    def image_type(self): return nib.minc2.Minc2Image
    @classmethod
    def header_type(self): return nib.minc2.Minc2Header
    @classmethod
    def aliases(self): return ('minc2')
class PARRECImageType(ImageType):
    @classmethod
    def image_name(self): return 'parrec'
    @classmethod
    def image_type(self): return nib.parrec.PARRECImage
    @classmethod
    def header_type(self): return nib.parrec.PARRECHeader
    @classmethod
    def aliases(self): return ()
    @classmethod
    def meta_data(self, img):
        d = super(self, PARRECImageType).meta_data(img)
        hdr = to_image_header(img)
        try:
            (bvals,bvec) = hdr.get_bvals_bvecs()
            d['b_values'] = bvals
            d['b_vectors'] = bvecs
        except Exception: pass
        try:
            (m,b) = hdr.get_data_scaling()
            d['data_slope'] = m
            d['data_offset'] = b
        except Exception: pass
        # get_def(name)?
        try:    d['echo_train_length'] = hdr.get_echo_train_length()
        except Exception: pass
        try:    d['record_shape'] = hdr.get_record_shape()
        except Exception: pass
        try:    d['slice_orientation'] = hdr.get_slice_orientation()
        except Exception: pass
        try:    d['sorted_slice_indices'] = hdr.get_sorted_slice_indices()
        except Exception: pass
        try:    d['volume_labels'] = hdr.get_volume_labels()
        except Exception: pass
        try:    d['water_fat_shift'] = hdr.get_water_fat_shift()
        except Exception: pass
        return d
class EcatImageType(ImageType):
    @classmethod
    def image_name(self): return 'ecat'
    @classmethod
    def image_type(self): return nib.ecat.EcatImage
    @classmethod
    def header_type(self): return nib.ecat.EcatHeader
    @classmethod
    def aliases(self): return ()
    @classmethod
    def meta_data(self, img):
        d = super(self, EcatImageType).meta_data(img)
        hdr = to_image_header(img)
        try: d['filetype'] = hdr.get_filetype()
        except Exception: pass
        try: d['patient_orientation'] = hdr.get_patient_orient()
        except Exception: pass
        return d

image_types = (Nifti1ImageType, Nifti2ImageType, MGHImageType,
               Spm99AnalyzeImageType, Spm2AnalyzeImageType,
               Minc1ImageType, Minc2ImageType,
               PARRECImageType, EcatImageType)
image_types_by_name        = pyr.pmap({it.image_name():it  for it in image_types})
image_types_by_image_type  = pyr.pmap({it.image_type():it  for it in image_types})
image_types_by_header_type = pyr.pmap({it.header_type():it for it in image_types})

def to_image_type(image_type):
    '''
    to_image_type(image_type) yields an image-type class equivalent to the given image_type
      argument, which may be a type name or alias or an image or header object or class.    
    '''
    if image_type is None: return None
    if isinstance(image_type, type) and issubclass(image_type, ImageType): return image_type
    if pimms.is_str(image_type):
        image_type = image_type.lower()
        if image_type in image_types_by_name: return image_types_by_name[image_type]
        for it in image_types:
            if image_type in it.aliases(): return it
        raise ValueError('"%s" is not a valid image-type name or alias' % image_type)
    for x in (image_type, type(image_type)):
        try:    return image_types_by_image_type[x]
        except Exception: pass
        try:    return image_types_by_header_type[x]
        except Exception: pass
    raise ValueError('Unsupported image type: %s' % image_type)
def to_image_meta_data(img):
    '''
    to_image_meta_data(img) yields a dictionary of meta-data for the given nibabel image object img.
    to_image_meta_data(hdr) yields the equivalent meta-data for the given nibabel image header.

    Note that obj may also be a mapping object, in which case it is returned verbatim.
    '''
    if pimms.is_map(img): return img
    try: hdr = img.header
    except Exception: hdr = img
    intype = to_image_type(hdr)
    return intype.meta_data(hdr)
def to_image(img, image_type=None, meta_data=None, **kwargs):
    '''
    to_image(array) yields a Nifti1Image of the given array with default meta-data.
    to_image(array, image_type) yields an image object of the given type; image_type may either be
      an image class or a class name (see supported types below).
    to_image((array, meta_data)) uses the given mapping of meta-data to fill in the image's
      meta-data; note that meta_data may simply be an affine transformation matrix.
    to_image((array, affine, meta_data)) uses the given affine specifically (the given affine
      overrides any affine included in the meta_data).

    Note that the array may optionally be an image itself, in which case its meta-data is used as a
    starting point for the new meta-data. Any meta-data passed as a tuple overwrites this meta-data,
    and any meta-data passed as an optional argument overwrites this meta-data in turn.

    The first optional argument, specifying image_type is as an image type if possible, but if a
    meta-data mapping is passed as the first argument it is used as such; otherwise, the optional
    third argument is named meta_data, and any additional keyword arguments passed to to_image are
    merged into this meta_data object left-to-right (i.e., keyword arguments overwrite the meta_data
    keys).
    '''
    # quick cleanup of args:
    meta_data = pimms.merge({} if meta_data is None else meta_data, kwargs)
    if image_type is None: image_type = 'nifti1'
    # deduce image type
    image_type = to_image_type(image_type)
    # okay, next, parse the image argument itself:
    if is_tuple(img):
        if   len(img) == 1: (img,aff,mdat) = (img[0], None, None)
        elif len(img) == 2: (img,aff,mdat) = (img[0], None, img[1])
        elif len(img) == 3: (img,aff,mdat) = img
        else: raise ValueError('cannot parse more than 3 elements from image tuple')
    else: (aff,mdat) = (None,None)
    # see if the img argument is an image object
    try: (img,aff0,mdat0) = (img.dataobj, img.affine, to_image_meta_data(img))
    except Exception: (aff0,mdat0) = (None, {})
    # check that the affine wasn't given as the meta-data (e.g. (img,aff) instead of (img,mdat))
    if aff is None and mdat is not None:
        try:    (aff, mdat) = (to_affine(mdat, 3), {})
        except Exception: pass
    # parse the meta-data that has been given
    mdat = dict(pimms.merge(mdat0, {} if mdat is None else mdat, meta_data))
    # if there is an explicit affine, we put it into mdat now
    if aff is not None: mdat['affine'] = to_affine(aff, 3)
    if aff0 is not None and 'affine' not in mdat: mdat['affine'] = to_affine(aff0, 3)
    # okay, we create the image now:
    return image_type.create(img, meta_data=mdat)
