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

from ..util import (to_affine, is_image, is_image_header, is_tuple, curry, to_hemi_str, zinv)

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
        try: dataobj = dataobj.dataobj
        except Exception: pass
        dtype = np.asarray(dataobj).dtype if dataobj is not None else self.default_type()
        if   hdat and hdat.get('type', None) not in [None,Ellipsis]: dtype = np.dtype(hdat['type'])
        elif hdat and hdat.get('dtype',None) not in [None,Ellipsis]: dtype = np.dtype(hdat['dtype'])
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
        if   dataobj is not None: arr = np.asarray(dataobj).astype(dtype)
        elif ish:                 arr = np.zeros(ish,       dtype=dtype)
        else:                     arr = np.zeros([1,1,1,0], dtype=dtype)
        # reshape to the requested shape if need-be
        if ish and ish != arr.shape: arr = np.reshape(arr, ish)
        # then reshape to a valid (4D) shape
        sh = arr.shape
        if   len(sh) == 2: arr = np.reshape(arr, (sh[0], 1, 1, sh[1]))
        elif len(sh) == 1: arr = np.reshape(arr, (sh[0], 1, 1))
        elif len(sh) == 3: arr = np.reshape(arr, sh)
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
        try: d['affine'] = img.affine
        except Exception:
            try:   d['affine'] = hdr.get_best_affine()
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
        d = super(MGHImageType, self).meta_data(img)
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
        d = super(Nifti1ImageType, self).meta_data(img)
        hdr = to_image_header(img)
        try: d['dimension_information'] = hdr.get_dim_info()
        except Exception: pass
        try: d['intent'] = hdr.get_intent()
        except Exception: pass
        try: d['slice_count'] = hdr.get_n_slices()
        except Exception: pass
        try: (sunit, tunit) = hdr.get_xyzt_units()
        except Exception: (sunit, tunit) = ('unknown', 'unknown')
        vsz = d['voxel_size']
        if vsz is not None and len(vsz) == 4:
            d['voxel_size'] = vsz[:3]
            if tunit == 'unknown': d['voxel_duration'] = vsz[3]
            else:
                try: d['voxel_duration'] = pimms.quant(vsz[3], tunit)
                except Exception: d['voxel_duration'] = vsz[3]
        else:
            d['voxel_duration'] = None
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
            except Exception: q = to_affine(d['affine'])
            qc = d.get('qform_code', None)
            hdr.set_qform(q, qc)
        except Exception: pass
        try:
            try: s = to_affine(d['sform'])
            except Exception: s = to_affine(d['affine'])
            sc = d.get('sform_code', None)
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
        d = super(PARRECImageType, self).meta_data(img)
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
        d = super(EcatImageType, self).meta_data(img)
        hdr = to_image_header(img)
        try: d['filetype'] = hdr.get_filetype()
        except Exception: pass
        try: d['patient_orientation'] = hdr.get_patient_orient()
        except Exception: pass
        return d
class Cifti2ImageType(ImageType):
    @classmethod
    def image_name(self): return 'cifti2'
    @classmethod
    def image_type(self): return nib.Cifti2Image
    @classmethod
    def header_type(self): return nib.Cifti2Header
    @classmethod
    def aliases(self): return ('cii', 'cii.gz', 'cifti')
    @classmethod
    def meta_data(self, img):
        from neuropythy.hcp import cifti_axis_spec
        d = {}
        axdat = cifti_axis_spec(img)
        d['axes_data'] = axdat
        d['image_shape'] = tuple([ax.get('size', None) for ax in axdat])
        #d['voxel_type'] = img.get_data_dtype()
        return pimms.persist(d)
    #@classmethod
    #def postprocess_image(self, img, d):
    #    return img


image_types = (Nifti1ImageType, Nifti2ImageType, MGHImageType,
               Spm99AnalyzeImageType, Spm2AnalyzeImageType,
               Minc1ImageType, Minc2ImageType,
               PARRECImageType, EcatImageType, Cifti2ImageType)
image_types_by_name        = pyr.pmap({it.image_name():it  for it in image_types})
image_types_by_image_type  = pyr.pmap({it.image_type():it  for it in image_types})
image_types_by_header_type = pyr.pmap({it.header_type():it for it in image_types})

# image-specs are allowed to store things using a few different keywords, so here are accessor
# functions based on some data structs:
imspec_aliases = {'image_shape': ['image_size', 'shape', 'image_dimensions', 'image_dims'],
                  'affine':      ['affine_transform', 'affine_matrix'],
                  'voxel_type':  ['dtype', 'type'],
                  'voxel_size':  ['pixel_size', 'voxel_dims', 'voxel_dimensions']}
def imspec_lookup(imspec, k, default=None):
    '''
    imspec_lookup(imspec, k) yields the value associated with the key k in the mapping imspec; if k
      is not in imspec, then imspec alises are checked and the appropriate value is returned;
      otherwise None is returned.
    imspec_lookup(imspec, k, default) yields default if neither k not an alias cannot be found.
    '''
    k = k.lower()
    if k in imspec: return imspec[k]
    aliases = imspec_aliases.get(k, None)
    if aliases is None:
        for q in six.itervalues(imspec_aliases):
            if k in q:
                aliases = q
                break
    if aliases is None: return default
    for kk in aliases:
        if kk in imspec: return imspec[kk]
    return default
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
def is_image_array(arr):
    '''
    is_image_array(arr) yields True if arr is a valid image array and False otherwise; in order to
      be a valid array, it must be a 1D, 2D, 3D or 4D array.
    '''
    return pimms.is_array(arr, None, (1,2,3,4))
def is_image_spec(imspec):
    '''
    is_image_spec(imspec) yields True if imspec is a map with the keys 'affine' and 'image_shape',
      otherwise yields False.
    '''
    return (pimms.is_map(imspec) and
            imspec_lookup(imspec, 'affine') is not None and
            imspec_lookup(imspec, 'image_shape') is not None)
def image_header_to_spec(hdr):
    '''
    image_header_to_spec(hdr) yields an image spec for the given image header object hdr.
    '''
    intype = to_image_type(hdr)
    return intype.meta_data(hdr)
def image_shape(arg):
    '''
    image_shape(im) yields the image shape for the given image im. The argument im may be an image,
      an array, an image header, or an image spec.
    '''
    if   is_image(arg):                                sh = arg.shape
    elif pimms.is_vector(arg, 'int') and len(arg) < 5: sh = tuple(arg)
    elif is_image_spec(arg):                           sh = imspec_lookup(arg, 'image_shape')
    elif is_image_header(arg):                         sh = image_header_to_spec(arg)['image_shape']
    elif is_image_array(arg):                          sh = np.shape(arg)
    else: raise VelueError('Bad argument of type %s given to image_shape()' % type(arg))
    sh = tuple(sh)
    if   len(sh) == 2: sh = (sh[0], 1, 1, sh[1])
    elif len(sh) == 1: sh = (sh[0], 1, 1)
    return sh
def image_array_to_spec(arr):
    '''
    image_array_to_spec(arr) yields an image-spec that is appropriate for the given array. The 
      default image spec for an array is a FreeSurfer-like affine transformation with a translation
      that puts the origin at the center of the array. The upper-right 3x3 matrix for this
      transformation is [[-1,0,0], [0,0,1], [0,-1,0]].
    image_array_to_spec((i,j,k)) uses (i,j,k) as the shape of the image array.
    image_array_to_spec(image) uses the array from the given image but not the affine matrix.
    image_array_to_spec(spec) uses the image shape from the given image spec but not the affine
      matrix.
    '''
    sh   = image_shape(arr)[:3]
    (i0,j0,k0) = np.asarray(sh) * 0.5
    ijk0 = (i0, -k0, j0)
    aff  = to_affine(([[-1,0,0],[0,0,1],[0,-1,0]], ijk0), 3)
    return {'image_shape':sh, 'affine':aff}
def image_to_spec(img):
    '''
    image_to_spec(img) yields an image-spec for the given image img. If img is not an image object,
      this will fail.
    '''
    return image_header_to_spec(img.header)
def image_spec_to_image(imspec, image_type=None, fill=0):
    '''
    image_spec_to_image(imspec) yields an empty with the given image-spec.
    image_spec_to_image(imspec, image_type) creates an image with the given type
    image_spec_to_image(imspec, image_type, fill) fills the resulting image with the given
      fill-value.
    '''
    imsh = imspec_lookup(imspec, 'image_shape')
    if fill == 0: imarr = np.zeros(imsh, dtype=imspec_lookup(imspec, 'voxel_type'))
    else:         imarr = np.full(imsh, fill, dtype=imspec_lookup(imspec, 'voxel_type'))
    # okay, we have the image array...
    image_type = to_image_type('nifti1' if image_type is None else image_type)
    return image_type.create(imarr, meta_data=imspec)
def to_image_spec(img, **kw):
    '''
    to_image_spec(img) yields a dictionary of meta-data for the given nibabel image object img.
    to_image_spec(hdr) yields the equivalent meta-data for the given nibabel image header.

    Note that obj may also be a mapping object, in which case it is returned verbatim.
    '''
    if pimms.is_vector(img,'int') and is_tuple(img) and len(img) < 5:
        r = image_array_to_spec(np.zeros(img))
    elif pimms.is_map(img):    r = img
    elif is_image_header(img): r = image_header_to_spec(img)
    elif is_image(img):        r = image_to_spec(img)
    elif is_image_array(img):  r = image_array_to_spec(img)
    else: raise ValueError('cannot convert object of type %s to image-spec' % type(img))
    if len(kw) > 0: r = {k:v for m in (r,kw) for (k,v) in six.iteritems(m)}
    # normalize the entries
    for (k,aliases) in six.iteritems(imspec_aliases):
        if k in r: continue
        for al in aliases:
            if al in r:
                val = r[al]
                r = pimms.assoc(pimms.dissoc(r, al), k, val)
                break
    return r
def to_image(img, image_type=None, spec=None, **kwargs):
    '''
    to_image(array) yields a Nifti1Image of the given array with default meta-data spec.
    to_image(array, image_type) yields an image object of the given type; image_type may either be
      an image class or a class name (see supported types below).
    to_image((array, spec)) uses the given mapping of meta-data (spec) to construct the image-spec
      note that spec may simply be an affine transformation matrix or may be an image.
    to_image((array, affine, spec)) uses the given affine specifically (the given affine
      overrides any affine included in the spec meta-data).
    to_image(imspec) constructs an image with the properties specified in the given imspec; the
      special optional argument fill (default: 0.0) can be set to something else to specify what the
      default cell value should be.

    Note that the array may optionally be an image itself, in which case its spec is used as a
    starting point for the new spec. Any spec-data passed as a tuple overwrites this spec-data,
    and any spec-data passed as an optional argument overwrites this spec-data in turn.

    The first optional argument, specifying image_type is as an image type if possible, but if a
    spec-data mapping or equivalent (e.g., an image header or affine) is passed as the first
    argument it is used as such; otherwise, the optional third argument is named spec, and any
    additional keyword arguments passed to to_image are merged into this spec object left-to-right
    (i.e., keyword arguments overwrite the spec keys).

    If no affine is given and the image object given is an array then a FreeSurfer-like transform
    that places the origin at the center of the image.
    '''
    # make sure we return unchanged if no change requested
    if is_image(img) and image_type is None and spec is None and len(kwargs) == 0: return img
    elif is_image_spec(img):
        fill = kwargs.pop('fill', 0.0)
        return to_image(image_spec_to_image(img, fill=fill),
                        image_type=image_type, spec=spec, **kwargs)
    # quick cleanup of args:
    # we have a variety of things that go into spec; in order (where later overwrites earlier):
    # (1) img spec, (2) image_type map (if not an image type) (3) spec, (4) kw args
    # see if image_type is actually an image type (might be a spec/image)...
    if pimms.is_str(image_type) or isinstance(image_type, type):
        (image_type, s2) = (to_image_type(image_type), {})
    else: 
        (image_type, s2) = (None, {} if image_type is None else to_image_spec(image_type))
    if image_type is None: image_type = image_types_by_name['nifti1']
    s3 = {} if spec is None else to_image_spec(spec)
    # okay, next, parse the image argument itself:
    if is_tuple(img):
        if   len(img) == 1: (img,aff,s1) = (img[0], None, {})
        elif len(img) == 2: (img,aff,s1) = (img[0], None, img[1])
        elif len(img) == 3: (img,aff,s1) = img
        else: raise ValueError('cannot parse more than 3 elements from image tuple')
        # check that the affine wasn't given as the meta-data (e.g. (img,aff) instead of (img,mdat))
        if aff is None and s1 is not None:
            try:    (aff, s1) = (to_affine(s1, 3), {})
            except Exception: pass
    else: (aff,s1) = (None, {})
    s0 = to_image_spec(img)
    spec = pimms.merge(s0, s1, s2, s3, kwargs)
    if aff is not None: spec = pimms.assoc(spec, affine=to_affine(aff, 3))
    # okay, we create the image now:
    return image_type.create(img, meta_data=spec)
def image_copy(img, dataobj=Ellipsis, affine=Ellipsis, image_type=None):
    '''
    image_copy(image) copies the given image and returns the new object; the affine, header, and
      dataobj members are duplicated so that changes will not change the original image.
    
    The following optional arguments may be given to overwrites part of the new image; in each case,
    the default value (Ellipsis) specifies that no update should be made.
      * dataobj (default: Ellipsis) may overwrite the new image's dataobj object.
      * affine (default: Ellipsis) may overwrite the new image's affine transformation matrix.
    '''
    dataobj = np.array(img.dataobj) if dataobj is Ellipsis else np.asanyarray(dataobj)
    imspec = to_image_spec(img)
    imtype = to_image_type(img if image_type is None else image_type)
    affine = imspec['affine'] if affine is Ellipsis else affine
    imspec['affine'] = affine
    return imtype.create(dataobj, imspec)
def image_clear(img, fill=0):
    '''
    image_clear(img) yields a duplicate of the given image img but with all voxel values set to 0.
    image_clear(img, fill) sets all voxels to the given fill value.
    '''
    img = image_copy(img)
    img.dataobj[...] = fill
    return img
def is_pimage(img):
    '''
    is_pimage(img) yields True if img is an image object and it contains a persistent dataobj (i.e.,
      the dataobj is a numpy array with the writeable flag set to False); otherwise yields False.
    '''
    if   not is_image(img):                      return False
    elif not pimms.is_nparray(img.dataobj):      return False
    elif img.dataobj.flags['WRITEABLE'] is True: return False
    else:                                        return True
def is_npimage(img):
    '''
    is_npimage(img) yields True if img is an image object and its dataobj member is a numpy array--
      i.e., img is not a pointer to an array-proxy object (e.g., when the image is cached on disk);
      yields False otherwise.
    '''
    if   not is_image(img):                      return False
    elif not pimms.is_nparray(img.dataobj):      return False
    else:                                        return True
def image_interpolate(img, points, affine=None, method=None, fill=0, dtype=None, weights=None):
    '''
    image_interpolate(img, points) yields the result of interpolating the given points in the given
      image.

    Generally, the provided image (img) would be a nibabel image object, in which case, an affine
    transformation is included; if img is just an array, the affine transform is assumed to a
    FreeSurfer-like transform that places the origin at the center of the array; see to_image().

    The following options may be used:
      * affine (default: None) may specify the affine transform that aligns the vertex coordinates
        with the image (vertex-to-voxel transform). If image is an MGHImage or a Nifti1Image or
        similar, then the affine transform included in the header will be used by default if None is
        given; this parameter overwrites whatever parameter is included in the image, however.
      * method (default: None) may specify either 'linear' or 'nearest'; if None, then the
        interpolation is linear when the image data is real and nearest otherwise.
      * fill (default: 0) values filled in when a vertex falls outside of the image.
      * affine (default: None) may optionally give a final transformation that converts from vertex
        positions to native subject orientation.
      * weights (default: None) may optionally provide an image whose voxels are weights to use
        during the interpolation; these weights are in addition to trilinear weights and are
        ignored in the case of nearest interpolation unless a voxel's weight is 0. The weights,
        whether an array or an image-object, but have the same shape as the input img--any affine
        is ignored.
    '''
    points = np.asarray(points)
    if len(points.shape) == 1:
        return image_interpolate(img, np.reshape(points,[3,1]), affine=affine, method=method,
                                 fill=fill, dtype=dtype, weights=weights)[0]
    if points.shape[0] != 3: points = points.T
    if pimms.is_str(img): image = load(img)
    img = to_image(img) if affine is None else to_image(img, affine=affine)
    imspec = to_image_spec(img)
    image = img.dataobj
    # we'll use the inverse affine on the points
    affine = np.linalg.inv(imspec['affine'])
    if method is not None: method = method.lower()
    if method is None or method in ['auto', 'automatic']:
        method = 'linear' if np.issubdtype(image.dtype, np.inexact) else 'nearest'
    if dtype is None: dtype = image.dtype
    # figure out the weights...
    if weights is not None: 
        if pimms.is_str(weights): weights = load(weights).dataobj
        elif is_image(weights): weights = weights.dataobj
        else: weights = np.asanyarray(weights)
        if not np.array_equal(weights.shape, image.shape[:3]):
            raise ValueError('weights and image must have the same shape')
    # okay, these are actually pretty simple; first transform the coordinates
    xyz = np.dot(affine, np.vstack([points, np.ones([1,points.shape[1]])]))[:3]
    # remember: this might be a 4d or higher-dim image...
    res = np.full((xyz.shape[1],) + image.shape[3:], fill, dtype=dtype)
    # now find the nearest voxel centers...
    # if we are doing nearest neighbor; we're basically done already:
    image = np.asarray(image)
    imsh = np.reshape(image.shape[:3], (3,1))
    if method == 'nearest':
        ijk = np.asarray(np.round(xyz), dtype=np.int)
        ok = np.all(ijk >= 0, axis=0) & np.all(ijk < imsh, axis=0)
        if weights is not None:
            ww = weights[tuple(ijk[:,ok])]
            ok[ok] &= ~np.isclose(ww, 0)
        res[ok] = image[tuple(ijk[:,ok])]
        return res
    # otherwise, we do linear interpolation; start by finding the 8 neighboring voxels
    mins = np.floor(xyz)
    maxs = np.ceil(xyz)
    ok = np.all(mins >= 0, axis=0) & np.all(maxs < imsh, axis=0)
    (mins,maxs,xyz) = [x[:,ok] for x in (mins,maxs,xyz)]
    voxs = np.asarray([mins,
                       [mins[0], mins[1], maxs[2]],
                       [mins[0], maxs[1], mins[2]],
                       [mins[0], maxs[1], maxs[2]],
                       [maxs[0], mins[1], mins[2]],
                       [maxs[0], mins[1], maxs[2]],                           
                       [maxs[0], maxs[1], mins[2]],
                       maxs],
                      dtype=np.int)
    vals = np.asarray([image[tuple(row)] for row in voxs])
    # trilinear weights
    wgts = np.asarray([np.prod(1 - np.abs(xyz - row), axis=0) for row in voxs])
    # weight-image weights
    if weights is not None: wgts = wgts * np.asarray([weights[tuple(row)] for row in voxs])
    winv = zinv(np.sum(wgts, axis=0))
    wgts *= winv
    ok2 = ~np.isclose(winv, 0)
    ok[ok] &= ok2
    res[ok] = np.sum(wgts * vals, axis=0)[ok2]
    return res
def image_apply(image, affine, post=True):
    '''
    image_apply(im, aff) applies the given affine transform to to_image(im) and yields an equivalent
      image with the new transform in place of the old one.

    The optional third argument post (default: True) may be set to False to specify that the affine
    in the returned image should be dot(im.affine, aff) instead of dot(aff, im.affine).
    '''
    im = to_image(image)
    af = to_affine(affine, 3)
    af = np.dot(aff, im.affine) if post else np.dot(im.affine, aff)
    return to_image(image, affine=affine)
def image_reslice(image, spec, method=None, fill=0, dtype=None, weights=None, image_type=None):
    '''
    image_reslice(image, spec) yields a duplicate of the given image resliced to have the voxels
      indicated by the given image spec. Note that spec may be an image itself.

    Optional arguments that can be passed to image_interpolate() (asside from affine) are allowed
    here and are passed through.
    '''
    if image_type is None and is_image(image): image_type = to_image_type(image)
    spec = to_image_spec(spec)
    image = to_image(image)
    # we make a big mesh and interpolate at these points...
    imsh = spec['image_shape']
    (args, kw) = ([np.arange(n) for n in imsh[:3]], {'indexing': 'ij'})
    ijk = np.asarray([u.flatten() for u in np.meshgrid(*args, **kw)])
    ijk = np.dot(spec['affine'], np.vstack([ijk, np.ones([1,ijk.shape[1]])]))[:3]
    # interpolate here...
    u = image_interpolate(image, ijk, method=method, fill=fill, dtype=dtype, weights=weights)
    return to_image((np.reshape(u, imsh), spec), image_type=image_type)

    
    
