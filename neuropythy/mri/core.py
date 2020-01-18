####################################################################################################
# neuropythy/mri/core.py
# Simple tools for dealing with the cortical objects and general MRImage data
# By Noah C. Benson

import numpy               as np
import numpy.linalg        as npla
import scipy               as sp
import scipy.sparse        as sps
import scipy.spatial       as spspace
import pyrsistent          as pyr
import collections         as colls
import os, sys, types, six, pimms

from .. import geometry as geo

from itertools import chain

from ..util import (ObjectWithMetaData, to_affine, apply_affine, is_image, is_address, is_tuple,
                    is_list, address_data, address_interpolate, curry, to_hemi_str, is_pseudo_path,
                    pseudo_path, to_pseudo_path)
from .images import (to_image, to_image_spec, is_image_spec, is_image_header, is_pimage, is_npimage,
                     image_copy)

@pimms.immutable
class Subject(ObjectWithMetaData):
    '''
    Subject is a class that tracks information about an individual subject. A Subject object keeps
    track of hemispheres (Cortex objects) as well as some information about the voxels (e.g., the
    gray_mask).

    When declaring a subject, the hemispheres argument (hemis) should always include at least the
    keys 'lh' and 'rh'.
    Images should, at a minimum, include the following in their images dictionary:
      * lh_gray_mask
      * rh_gray_mask
      * lh_white_mask
      * rh_white_mask
    Alternately, the values with the same names may be overloaded in a daughter class.

    Subject respects laziness in the hemis and images classes, and this mechanism is recommended
    as a way to lazily load subject data (see pimms.lazy_map).
    '''
    def __init__(self, name=None, pseudo_path=None, hemis=None, images=None, meta_data=None):
        self.name                   = name
        self.pseudo_path            = pseudo_path
        self.hemis                  = hemis
        self.images                 = images
        self.meta_data              = meta_data

    @pimms.param
    def name(nm):
        '''
        sub.name is the name of the subject sub.
        '''
        if nm is None or pimms.is_str(nm): return nm
        else: raise ValueError('subject names must be strings or None')
    @pimms.param
    def pseudo_path(pd):
        if pd is None: return None
        else: return to_pseudo_path(pd)
    @pimms.value
    def path(pseudo_path):
        '''
        sub.path is the path of the subject's data directory, if any.
        '''
        if pseudo_path is None: return None
        return pseudo_path.actual_source_path
    @pimms.param
    def hemis(h):
        '''
        sub.hemis is a persistent map of hemisphere names ('lh', 'rh', possibly others) for the
        given subject sub.
        '''
        if   h is None:        return pyr.m()
        elif pimms.is_pmap(h): return h
        elif pimms.is_map(h):  return pyr.pmap(h)
        else: raise ValueError('hemis must be a mapping')
    @pimms.param
    def images(imgs):
        '''
        sub.images is a persistent map of MRImages tracked by the given subject sub.
        '''
        if   imgs is None:        return pyr.m()
        elif pimms.is_pmap(imgs): return imgs
        elif pimms.is_map(imgs):  return pyr.pmap(imgs)
        else: raise ValueError('images must be a mapping')

    # Updater operations:
    def with_hemi(self, *ms, **kw):
        '''
        sub.with_hemi(name=hemi) adds the give Cortex object hemi to the subject's hemis map with
          the given name.
        sub.with_hemi(name1=hemi1, name2=hemi2...) adds all the given hemispheres.
        sub.with_hemi({name1: hemi1, name2:hemi2}) is equivalent to the previous line.
        
        Note that any number of maps may be passed followed by any number of keyword arguments.
        '''
        hh = pimms.merge(self.hemis, *(ms + (kw,)))
        if hh is self.hemis: return self
        return self.copy(hemis=hh)
    def with_image(self, *ms, **kw):
        '''
        sub.with_image(name=img) adds the give Cortex object hemi to the subject's hemis map with
          the given name.
        sub.with_image(name1=img1, name2=img2...) adds all the given hemispheres.
        sub.with_image({name1:img1, name2:img2}) is equivalent to the previous line.
        
        Note that any number of maps may be passed followed by any number of keyword arguments.
        '''
        ims = pimms.merge(self.images, *(ms + (kw,)))
        if ims is self.images: return self
        return self.copy(images=ims)
    def wout_hemi(self, *args):
        '''
        sub.wout_hemi(heminame) yields a clone of the given subject object sub but with the 
          hemisphere with the given name removed.

        Note that any number of names may be given.
        '''
        hh = self.hemis
        for a in args: hh = hh.discard(a)
        return self if hh is self.hemis else self.copy(hemis=hh)
    def wout_image(self, *args):
        '''
        sub.wout_image(imgname) yields a clone of the given subject object sub but with the image
          with the given name removed.

        Note that any number of names may be given.
        '''
        ims = self.images
        for a in args: ims = ims.discard(a)
        return self if ims is self.images else self.copy(images=ims)


    # Aliases for hemispheres
    @pimms.value
    def LH(hemis):
        '''
        sub.LH is an alias for sub.hemis['lh'].
        '''
        return hemis.get('lh', None)
    @pimms.value
    def RH(hemis):
        '''
        sub.RH is an alias for sub.hemis['rh'].
        '''
        return hemis.get('rh', None)
    @pimms.value
    def lh(hemis):
        '''
        sub.lh is an alias for sub.hemis['lh'].
        '''
        return hemis.get('lh', None)
    @pimms.value
    def rh(hemis):
        '''
        sub.rh is an alias for sub.hemis['rh'].
        '''
        return hemis.get('rh', None)
    def to_hemi(self, h):
        '''
        sub.to_hemi(arg) attempts to convert arg into a hemisphere of the given subject. If arg is
          a tuple or list of hemisphere objects or names, then each is converted and returned as
          a tuple.

        Note that the objects Ellipsis and None are converted into ('lh', 'rh'). If h is a string
        that refers to a suffix of a left and right hemisphere in sub, then (lh, rh) are returned
        for that particular hemisphere (e.g., with HCP subjects, 'LR32k' will return the 32k left
        and right hemispheres as a tuple).
        '''
        if isinstance(h, Cortex): return h
        elif pimms.is_str(h):
            if h in self.hemis: return self.hemis[h]
            elif h == 'lr' or len(h) == 0: return self.to_hemi(('lh', 'rh'))
            elif h.startswith('lr_'): return self.to_hemi(('lh'+h[2:], 'rh'+h[2:]))
            # see if there is a suffix for lh and rh:
            if ('lh%s' % h) in self.hemis and ('rh%s' % h) in self.hemis:
                return (self.hemis['lh%s' % h], self.hemis['rh%s' % h])
            elif h[0] != '_' and ('lh_%s' % h) in self.hemis and ('rh_%s' % h) in self.hemis:
                return (self.hemis['lh_%s' % h], self.hemis['rh_%s' % h])
            # try converting to a hemi string
            hh = to_hemi_str(h)
            if h == hh: raise ValueError('Could not convert "%s" into a hemisphere' % h)
            else: return self.to_hemi(hh)
        elif pimms.is_vector(h):
            return tuple([self.to_hemi(u) for u in h])
        else:
            # try to convert to a hemi-string
            return self.to_hemi(to_hemi_str(h))
        raise ValueError('Unrecognized hemisphere argument: %s' % (h,))

    # Aliases for images
    @pimms.value
    def lh_gray_mask(images):
        '''
        sub.lh_gray_mask is an alias for sub.images['lh_gray_mask'].
        '''
        return images.get('lh_gray_mask', None)
    @pimms.value
    def rh_gray_mask(images):
        '''
        sub.rh_gray_mask is an alias for sub.images['rh_gray_mask'].
        '''
        return images.get('rh_gray_mask', None)
    @pimms.value
    def lh_white_mask(images):
        '''
        sub.lh_white_mask is an alias for sub.images['lh_white_mask'].
        '''
        return images.get('lh_white_mask', None)
    @pimms.value
    def rh_white_mask(images):
        '''
        sub.rh_white_mask is an alias for sub.images['rh_white_mask'].
        '''
        return images.get('rh_white_mask', None)

    @pimms.value
    def lh_gray_indices(lh_gray_mask):
        '''
        sub.lh_gray_indices is equivalent to numpy.where(sub.lh_gray_mask).
        '''
        if lh_gray_mask is None: return None
        if is_image(lh_gray_mask): lh_gray_mask = lh_gray_mask.dataobj
        return tuple([pimms.imm_array(x) for x in np.where(lh_gray_mask)])
    @pimms.value
    def rh_gray_indices(rh_gray_mask):
        '''
        sub.rh_gray_indices is equivalent to numpy.where(sub.rh_gray_mask).
        '''
        if rh_gray_mask is None: return None
        if is_image(rh_gray_mask): rh_gray_mask = rh_gray_mask.dataobj
        return tuple([pimms.imm_array(x) for x in np.where(rh_gray_mask)])
    @pimms.value
    def gray_indices(lh_gray_indices, rh_gray_indices):
        '''
        sub.gray_indices is equivalent to numpy.where(sub.gray_mask).
        '''
        if lh_gray_indices is None and rh_gray_indices is None: return None
        elif lh_gray_indices is None: return rh_gray_indices
        elif rh_gray_indices is None: return lh_gray_indices
        else:
            gis = tuple([np.concatenate(pair) for pair in zip(lh_gray_indices, rh_gray_indices)])
            for row in gis: row.setflags(write=False)
            return gis
    @pimms.value
    def lh_white_indices(lh_white_mask):
        '''
        sub.lh_white_indices is a frozenset of the indices of the white voxels in the given
        subject's lh, represented as 3-tuples.
        '''
        if lh_white_mask is None: return None
        if is_image(lh_white_mask): lh_white_mask = lh_white_mask.dataobj
        idcs = np.transpose(np.where(lh_white_mask))
        return frozenset([tuple(row) for row in idcs])
    @pimms.value
    def rh_white_indices(rh_white_mask):
        '''
        sub.rh_white_indices is a frozenset of the indices of the white voxels in the given
        subject's rh, represented as 3-tuples.
        '''
        if rh_white_mask is None: return None
        if is_image(rh_white_mask): rh_white_mask = rh_white_mask.dataobj
        idcs = np.transpose(np.where(rh_white_mask))
        return frozenset([tuple(row) for row in idcs])
    @pimms.value
    def white_indices(lh_white_indices, rh_white_indices):
        '''
        sub.white_indices is a frozenset of the indices of the white voxels in the given subject
        represented as 3-tuples.
        '''
        if   lh_white_indices is None and rh_white_indices is None: return None
        elif lh_white_indices is None: return rh_white_indices
        elif rh_white_indices is None: return lh_white_indices
        else: return frozenset(lh_white_indices | rh_white_indices)
    @pimms.value
    def image_dimensions(images):
        '''
        sub.image_dimensions is a tuple of the default size of an anatomical image for the given
        subject.
        '''
        if images is None or len(images) == 0: return None
        if pimms.is_lazy_map(images):
            # look for an image that isn't lazy...
            key = next((k for k in images.iterkeys() if not images.is_lazy(k)), None)
            if key is None: key = next(images.iterkeys(), None)
        else:
            key = next(images.iterkeys(), None)
        img = images[key]
        if img is None: return None
        if is_image(img): img = img.dataobj
        return np.asarray(img).shape
    @pimms.value
    def repr(name, path):
        '''
        sub.repr is the representation string returned by sub.__repr__().
        '''
        if name is None and path is None:
            return 'Subject(<?>)'
        elif name is None:
            return 'Subject(<\'%s\'>)' % path
        elif path is None:
            return 'Subject(<%s>)' % name
        else:
            return 'Subject(<%s>, <\'%s\'>)' % (name, path)

    def __repr__(self):
        return self.repr
    def path_join(self, *args):
        '''
        sub.path_join(args...) is equivalent to sub.pseudo_path.join(sub.path, args...)
        '''
        return self.pseudo_path.join(self.path, *args)
    def load(self, filename, *args, **kw):
        '''
        sub.load(filename) attempts to load the given filename from the pseudo-path / directory
          represented by the given subject sub. Note that additional arguments and keyword arguments
          are passed verbatim to the neuropythy.load() function.

        Note that the given filename must be a relative path. It may, however, be a list or tuple of
        diretory names (as would be passed to os.path.join).
        '''
        from neuropythy.io import load
        if self.pseudo_path is None:
            raise ValueError('cannot load from subject without a pseudo_path')
        if not pimms.is_str(filename): filename = self.pseudo_path.join(*filename)
        flnm = self.pseudo_path.local_path(filename)
        return load(flnm, *args, **kw)
    def cortex_to_image(self, data, im,
                        hemi=None, method=None, fill=0, affine=Ellipsis, address=None,
                        # below are the property() args:
                        dtype=Ellipsis, outliers=None,    data_range=None,    clipped=np.inf,
                        weights=None,   weight_min=0,     null=np.nan,        transform=None,
                        mask=None,      valid_range=None, weight_transform=Ellipsis):
        '''
        sub.cortex_to_image(data, im) yields an MRImage object with the same image-spec as im but
          with the given data projected into the image. The argument im may be anything that can be
          converted to an image-spec using the to_image_spec() function. If im is an image whose
          dataobj member is not persistent and is not an array-proxy then the image itself will be
          written into.
        sub.cortex_to_image(data, im, hemi) projects the given cortical-surface data to the given
          subject's gray-matter voxels of the given hemisphere and returns the image object.
        sub.cortex_to_image((lh_data, rh_data), im) projects into both hemispheres and yields the
          object.

        Note that if no hemisphere is given and the input argument data is not a tuple like
        (lh_data, rh_data), then it must be a property name that is shared by both hemispheres. Even
        if both hemispheres have the same vertex_count, you cannot pass one vector unless a
        hemisphere name is given. By default the hemispheres used are 'lh' and 'rh'.
    
        The data argument may be a 2-tuple, in which case it is always interpreted as (lh-data,
        rh-data); the individual hemisphere data can be a vector of values or a property name. The
        data are always passed through the cortex.property() method and any option that can be given
        to property() can be given here and will be forwarded along. The following additional
        options may also be given:
          * method (default: None) specifies that a particular method should be used; valid options
            are 'linear' and 'nearest'. The 'linear' method uses linear interpolation within the
            prismatic columns of the cortex (see cortex.to_image() for more info); the nearest
            method uses the vertex on the cortex to which the given voxel is closest (using the
            lines that the vertices make through cortex). The default, None, specifies that linear
            should be used for all real/complex (inexact) data and nearest should be used for all
            other data (integers/strings/etc).
          * fill (default: 0) specifies the value to be assigned to all voxels not in the gray mask
            or voxels in the gray-mask that are missed by the interpolation method.
          * affine (default: Ellipsis) specifies the affine transformation that should be used to
            align the cortical surfaces with the voxels. If Ellipsis, then the affine transforms
            saved in the cortex objects. Note that this option overwrites these transforms if found.
          * address (default: None) may specify pre-calculated addresses for use in projecting the
            hemispheres into the image.
        '''
        # get our image:
        if pimms.is_nparray(im) and len(im.shape) >= 3: im = ny.to_image(im)
        if is_image(im) and (is_pimage(im) or not is_npimage(im)): im = image_copy(im)
        else: im = to_image(to_image_spec(im), fill=fill)
        # what hemisphere(s)?
        hemi = self.to_hemi(hemi)
        if pimms.is_vector(hemi) and len(hemi) == 1: hemi = hemi[0]
        if not pimms.is_vector(hemi):
            if   hemi.chirality == 'lh': hemi = (hemi, None)
            elif hemi.chirality == 'rh': hemi = (None, hemi)
            else:                        hemi = (hemi, hemi)
        elif len(hemi) != 2:
            raise ValueError('Exactly two hemispheres must be given')
        elif len(np.unique([None if h is None else h.chirality for h in hemi])) != 2:
            raise ValueError('One or two different chiralities must be given in hemi argument')
        # make sure they are in the right order:
        if   hemi[1] is not None and hemi[1].chirality == 'lh': hemi = (hemi[1], hemi[0])
        elif hemi[0] is not None and hemi[0].chirality == 'rh': hemi = (hemi[1], hemi[0])
        nhems = len([x for x in hemi if x])
        tr = None
        # Make the data match this format...
        if pimms.is_str(data): data = [None if h is None else data for h in hemi]
        elif nhems == 1 and len(data) == 1:
            if len(data[0]) != (hemi[0] if hemi[1] is None else hemi[1]).vertex_count:
                raise ValueError('hemi/property size mismatch')
            data = [None if h is None else data[0] for h in hemi]
        elif len(data) == 2:
            if not (data[0] is None and hemi[0] is None or len(data[0]) == hemi[0].vertex_count):
                raise ValueError('lh hemi/property size mismatch')
            if not (data[1] is None and hemi[1] is None or len(data[1]) == hemi[1].vertex_count):
                raise ValueError('rh hemi/property size mismatch')
            data = np.asarray(data)
        else:
            data = np.asarray(data)
            sh = data.shape
            if nhems == 1: data = tuple([None if k is None else data for k in hemi])
            else: raise ValueError('cannot given single array of data for multiple hemispheres')
        # gather up the property() keywords for the next step:
        kw = dict(affine=affine,         method=method,
                  dtype=dtype,           null=null,          outliers=outliers,
                  data_range=data_range, clipped=clipped,    weights=weights,
                  weight_min=weight_min, mask=mask,          valid_range=valid_range,
                  transform=transform,   weight_transform=weight_transform)
        if address is None: address = (None, None)
        elif not pimms.is_vector(address):
            if nhems == 1: address = [None if h is None else address for h in hemi]
            else: raise ValueError('Two hemispheres but only 1 address given')
        # okay, we just need to pass down to the to_image function of the hemispheres:
        for (dat,h,addr,ii) in zip(data, hemi, address, np.arange(nhems)):
            if h is None: dat = None
            if dat is None: continue
            if addr is not None: kw['address'] = addr
            elif 'address' in kw: del kw['address']
            im = h.to_image(dat, im, **kw)
        return im
    def image_to_cortex(self, image,
                        surface='midgray', hemi=None, affine=Ellipsis, method=None, fill=0,
                        dtype=None, weights=None):
        '''
        sub.image_to_cortex(image) is equivalent to the tuple
          (sub.lh.from_image(image), sub.rh.from_image(image)).
        sub.image_to_cortex(image, surface) uses the given surface (see also cortex.surface).
        '''
        if hemi is None: hemi = 'both'
        hemi = hemi.lower()
        if hemi in ['both', 'lr', 'all', 'auto']:
            return tuple(
                [self.image_to_cortex(image, surface=surface, hemi=h, affine=affine,
                                      method=method, fill=fill, dtype=dtype, weights=weights)
                 for h in ['lh', 'rh']])
        else:
            hemi = getattr(self, hemi)
            return hemi.from_image(image, surface=surface, affine=affine,
                                   method=method, fill=fill, dtype=dtype, weights=weights)
    def image_address(self, image, hemi='lr'):
        '''
        sub.image_address(image) yields the (lh, rh) image-addresses for the given image or
          image-spec.
        sub.image_address(image, h) uses the given hemisphere h, which may alternately be a tuple of
          hemisphere names. The default, 'lr', is equivalent to ('lh','rh').
        '''
        if is_tuple(hemi):
            sing = False
            hemi = [to_hemi_str(h) for h in hemi]
            if h == ['lr']: h = ['lh','rh']
        else:
            sing = True
            hemi = to_hemi_str(hemi)
            hemi = ['lh','rh'] if hemi == 'lr' else [hemi]
        r = [self.hemis[h].image_address(image)
             for h in hemi]
        if sing and len(r) == 1: return r[0]
        else: return tuple(r)
def is_subject(s):
    '''
    is_subject(s) yields True if s is a Subject object and False otherwise.
    '''
    return isinstance(s, Subject)

@pimms.immutable
class Cortex(geo.Topology):
    '''
    Cortex is a class that handles a single cortical hemisphere; cortex tracks both the spherical
    registrations that can be used for cross-subject interpolation as well as the various surfaces
    and cortical layers that can be produced from the combined white/pial surfaces.
    Cortex is the go-to class for performing interpolation between subjects as it is a Topology
    object and thus knows how to search for common registrations for interpolation and comparison.
    Cortex also holds the required methods for creating map projections from spherical
    registrations.

    Cortex(chirality, tess, surfaces, registrations) is typically used to initialize a cortex
    object. The chirality should be either 'lh' or 'rh'; tess must be the tesselation of the cortex
    object. The surfaces and registrations arguments should both be (possibly lazy) maps whose
    values are the appropriate mesh objects. The surfaces must include 'white' and 'pial'. If the
    registrations includes the key 'native' this is taken to be the default registration for the
    particular cortex object.
    '''
    def __init__(self, chirality, tess, surfaces, registrations,
                 properties=None, affine=None, meta_data=None):
        self.chirality = chirality
        self.surface_coordinates = surfaces
        self.meta_data = meta_data
        geo.Topology.__init__(self, tess, registrations, properties=properties)
        self.chirality = chirality
        self.surface_coordinates = surfaces
        self.affine = affine
        self.meta_data = meta_data
    @pimms.param
    def chirality(ch):
        '''
        cortex.chirality gives the chirality ('lh' or 'rh') for the given cortex.
        '''
        if ch is None: return None
        ch = ch.lower()
        if ch != 'lh' and ch != 'rh':
            raise ValueError('chirality must be \'lh\' or \'rh\'')
        return ch
    @pimms.param
    def affine(aff):
        '''
        cortex.affine is either None or an affine transformation that specifies how the coordinates
        of the surfaces in the given cortex should be rotated in order to align with an abstract
        'native' geometry. This field is used, e.g., by FreeSurfer subjects to indicate how the
        coordinates of the FreeSurfer surfaces should be transformed in order to be aligned with the
        native-space defined by the affine transforms of the FreeSurfer images.
        '''
        if aff is None: return None
        return to_affine(aff, 3)
    @pimms.param
    def surface_coordinates(surfs):
        '''
        cortex.surface_coordinates is a mapping of the surface coordinates of the given cortex; this
        must include the surfaces 'white' and 'pial'.
        '''
        if pimms.is_map(surfs):
            return pimms.persist(surfs)
        else:
            raise ValueError('surface_coordinates must be a mapping object')
    @pimms.value
    def surfaces(surface_coordinates, properties, tess, chirality):
        '''
        cortex.surfaces is a mapping of the surfaces of the given cortex; this must include the
        surfaces 'white' and 'pial'.
        '''
        def _make_mesh(name):
            def _lambda():
                val = surface_coordinates[name]
                if isinstance(val, geo.Mesh): val = val.coordinates
                m = geo.Mesh(tess, val, properties=properties, meta_data={'chirality':chirality})
                return m.persist()
            return _lambda
        return pimms.lazy_map({k:_make_mesh(k) for k in six.iterkeys(surface_coordinates)})
    @pimms.value
    def aligned_surfaces(surfaces, affine):
        '''
        cortex.aligned_surfaces is identical to cortex.surfaces except that the aligned surfaces
        have been transformed by cortex.affine.
        '''
        if affine is None: return surfaces
        def align_surf(s):
            srf = surfaces[s]
            return srf.copy(coordinates=apply_affine(affine, srf.coordinates))
        return pimms.lazy_map({k:curry(align_surf, k) for k in six.iterkeys(surfaces)})
    @pimms.require
    def validate_surfaces(surfaces):
        '''
        validate_surfaces requires that the surfaces map contain the keys 'white' and 'pial'.
        '''
        if 'white' in surfaces and 'pial' in surfaces: return True
        else: raise ValueError('surfaces parameter must contain both \'white\' and \'pial\'')

    @pimms.value
    def white_surface(surfaces):
        '''
        cortex.white_surface is the mesh representing the white-matter surface of the given cortex.
        '''
        return surfaces['white']
    @pimms.value
    def pial_surface(surfaces):
        '''
        cortex.pial_surface is the mesh representing the pial surface of the given cortex.
        '''
        return surfaces['pial']
    @pimms.value
    def midgray_surface(white_surface, pial_surface):
        '''
        cortex.midgray_surface is the mesh representing the midgray surface half way between the
        white and pial surfaces.
        '''
        midgray_coords = 0.5*(white_surface.coordinates + pial_surface.coordinates)
        return white_surface.copy(coordinates=midgray_coords)
    @pimms.value
    def white_to_pial_vectors(white_surface, pial_surface):
        '''
        cortex.white_to_pial_vectors is a (3 x n) matrix of the unit direction vectors that point
          from the n vertices in the cortex's white surface to their equivalent positions in the
          pial surface.
        '''
        u = pial_surface.coordinates - white_surface.coordinates
        d = np.sqrt(np.sum(u**2, axis=0))
        z = np.isclose(d, 0)
        return pimms.imm_array(np.logical_not(z) / (d + z))
    @pimms.value
    def repr(chirality, tess, vertex_count):
        '''
        cortex.repr is equivalent to repr(cortex).
        '''
        arg = ('XH' if chirality is None else chirality.upper(), tess.face_count, vertex_count)
        return 'Cortex(<%s>, <%d faces>, <%d vertices>)' % arg

    def __repr__(self):
        return self.repr
    def surface(self, name='white', aligned=False):
        '''
        cortex.surface() yields the white surface of the given cortex
        cortex.surface(name) yields the surface with the given name (e.g., 'white', 'pial',
          'inflated', 'midgray').
        cortex.surface(fraction) yields the surface that is <fraction> of the distance from white
          to pial (0 is equivalent to 'white'; 1 is equivalent to 'pial'). Layers outside of the
          range 0-1 may be returned by following the vectors between white and pial surfaces, but
          they may have odd appearances, and this should not be confused with surface inflation.
        cortex.surface([dist]) yields the layer that is the given distance from the white surface.

        The optional argument aligned (default: False) may be specified to extract meshes already
        aligned according to the cortex.affine matrix.
        '''
        srfs = self.aligned_surfaces if aligned else self.surfaces
        if pimms.is_str(name):
            if name.lower() == 'midgray' and 'midgray' not in srfs:
                return self.midgray_surface # in case it's been provided via overloading
            elif name in srfs: return srfs[name]
            else:
                m = geo.to_mesh((self, name))
                if not aligned or self.affine is None: return m
                return m.copy(coordinates=apply_affine(self.affine, m.coordinates))
        elif pimms.is_vector(name, 'real') and len(name) == 1:
            x0 = srfs['white'].coordinates
            x1 = srfs['pial'].coordinates
            q = name[0]
            return self.make_mesh(x0*(1 - q) + x1*q)
        elif pimms.is_real(name):
            x0 = srfs['white'].coordinates
            x1 = srfs['pial'].coordinates
            return self.make_mesh((1 - name)*x0 + name*x1)
        else: raise ValueError('could not understand surface layer: %s' % name)
    def from_image(self, image,
                   surface='midgray', affine=Ellipsis, method=None, fill=0, dtype=None,
                   weights=None):
        '''
        cortex.from_image(image) is equivalent to cortex.midgray_surface.from_image(image).
        cortex.from_image(image, surface) uses the given surface (see also cortex.surface).

        The optional argument affine (default: Ellipsis) can be used to override the cortex.affine
        transformation typically used in cortex-image alignment.
        '''
        mesh = geo.to_mesh((self, surface))
        if affine is Ellipsis: affine = self.affine
        return mesh.from_image(image, affine=affine, method=method, fill=fill, dtype=dtype,
                               weights=weights)
    def to_image(self, prop, image,
                 fill=0, image_type=None, method=None, address=None, affine=Ellipsis,
                 # the rest of the arguments are for property()
                 dtype=Ellipsis,   outliers=None,  data_range=None,           clipped=np.inf,
                 weights=None,     weight_min=0,   weight_transform=Ellipsis, mask=None,
                 valid_range=None, null=np.nan,    transform=None):
        '''
        cortex.to_image(property, ...) yields a 3D image of the given property.
        cortex.to_image(property, image) writes the data into the given image and returns it if the
          image contains a writeable numpy array; otherwise duplicates the image with a new
          read-only array that consists of the new data written over the old data and yields that
          image. Note that most images returned by nibabel use array proxies and thus are duplicated
          rather than written to by default (because the dataobj is an ArrayProxy and not a
          writeable numpy array).
        cortex.to_image(property, imspec) uses the given image-specification to construct the image.

        Additional keyword arguments accepted by cortex.property() are passed along when determining
        the property vector. The following additional optional arguments are also accepted:
          * fill (default: 0) also specifies the value given to the cells of a new image if one must
            be constructed (for example, if no image is passed or an image-spec is given).
          * image_type (default: None) specifies the image type that should be returned; by default
            uses the type of the given image or 'nifti1' if no image type can be inferred.
          * method (default: None) specifies the method of interpolation to use. By default, this
            is 'linear' if the property data are inexact numbers and 'nearest' if the data are
            anything else (strings, integers, etc.).
          * address (default: None) may optionally be used to provide the pre-calculated addresses
            rather than using cortex.image_address() to find them.
        '''
        # start by getting the propery:
        propkw = dict(dtype=dtype,           null=null,                         outliers=outliers,
                      data_range=data_range, clipped=clipped,                   weights=weights,
                      mask=mask,             valid_range=valid_range,           transform=transform,
                      weight_min=weight_min, weight_transform=weight_transform)
        if pimms.is_map(prop):
            ktr = lambda k:0 if k == 'white' else 1 if k == 'pial' else 0.5 if k == 'midgray' else k
            prop = {ktr(k):self.property(v, **propkw) for (k,v) in six.iteritems(prop)}
        else:
            prop = self.property(prop, **propkw)
            # we assume these are linearly-spaced between 0 and 1
            if pimms.is_vector(prop): prop = {0:prop, 1:prop}
            else: prop = {k:v for (k,v) in zip(np.linspace(0, 1, len(prop)), prop)}
        # next, figure out the image
        if   is_image_spec(image):   image = to_image(image,
                                                      fill=fill, image_type=image_type, dtype=dtype)
        elif is_image_header(image): image = to_image(image,
                                                      fill=fill, image_type=image_type, dtype=dtype)
        else:                        image = to_image(image, image_type=image_type, dtype=dtype)
        if is_pimage(image) or not pimms.is_nparray(image.dataobj):
            image = image_copy(image, image_type=image_type)
        if image_type is not None:
            image_type = to_image_type(image_type)
            if to_image_type(image) is not image_type:
                image = image_copy(image, image_type=image_type)
        # okay, now we get the projection of the cortex into the image:
        addr = self.image_address(image, affine=affine) if address is None else address
        # and use this to interpolate...
        dat = address_interpolate(addr, prop, method=method, null=null)
        # and put these into the image!
        image.dataobj[tuple(addr['voxel_indices'])] = dat
        return image
    def image_address(self, img, affine=Ellipsis):
        '''
        cortex.image_address(img) is like cortex.address() but works on images; note that
          cortex.address(img) will call cortex.address_image(img) when img is an image.

        The optional argument affine may be used to override the cortex.affine transform usually
        used in cortex-image alignment.

        Note that the algorithm employed by this method works by dividing each triangular "prism"
        (colum through the cortex defined by each face in the cortex tesselation) into a number of
        tetrahedrons; because this definition of cortex is not necessarily identical to other 
        definitions (such as that used by FreeSurfer itself), so depending the precise definition
        of what is inside the cortex in a particular image may vary around the edges.
        '''
        dat = to_image_spec(img)
        aff = dat['affine']
        imsh = np.reshape(dat['image_shape'], (3,1))
        if affine is Ellipsis: affine = self.affine
        if affine is not None: aff = np.dot(np.linalg.inv(affine), aff)
        aff = np.linalg.inv(aff)
        # transform coordinates into voxel-space:
        (wx,px) = (self.white_surface.coordinates, self.pial_surface.coordinates)
        (wx,px) = [np.dot(aff, np.vstack([x, np.ones([1,x.shape[1]])]))[:3] for x in (wx,px)]
        # now we find the addresses... start by getting the face coords:
        fs = self.tess.indexed_faces
        (wfx,pfx) = [np.transpose(x[:,fs], (1,0,2)) for x in (wx,px)]
        # okay, make a bounding box for each prism in the cortex:
        x = np.vstack([wfx,pfx])
        (mn,mx) = [f(x, axis=0) for f in (np.min,np.max)]
        mn[mn < 0] = 0
        for (ii,n) in enumerate(imsh): mx[ii,mx[ii] >= n] = n - 1
        # number of voxels in each bounding box:
        (mn, mx) = (np.ceil(mn), np.floor(mx) + 1)
        dims = mx - mn
        dims[dims < 0] = 0
        nvox = np.prod(dims, axis=0).astype('int') #dbg
        ii = np.where(nvox > 0)[0]
        # now we build up a big list of voxel indices and prisms to test against each other:
        i1 = np.cumsum(nvox[ii])
        i0 = np.concatenate([[0], i1[:-1]])
        n  = int(i1[-1])
        idcs = np.zeros((3,n),   dtype=np.int)
        wfxs = np.zeros((3,3,n), dtype=np.float)
        pfxs = np.zeros((3,3,n), dtype=np.float)
        iis  = np.zeros(n,       dtype=np.int)
        # we step along from i0 to i1 forgetting finished prisms along the way
        (kk, q, mn, nvox, dims) = (ii, 0, mn[:,ii], nvox[ii], dims[:,ii])
        while len(kk) > 0:
            wfxs[:,:,i0] = wfx[:,:,kk]
            pfxs[:,:,i0] = pfx[:,:,kk]
            iis[i0] = kk
            idcs[:,i0] = [(q // (dims[1]*dims[2]))      + mn[0],
                          np.mod(q // dims[2], dims[1]) + mn[1],
                          np.mod(q, dims[2])            + mn[2]]
            # next iteration:
            q += 1
            ki = np.where(nvox > q)[0]
            nvox = nvox[ki]
            kk = kk[ki]
            if len(kk) == 0: break
            mn = mn[:,ki]
            dims = dims[:,ki]
            i0 = i0[ki]
            i0 += 1
        # now test them all:
        bcs = geo.prism_barycentric_coordinates(wfxs, pfxs, idcs)
        ok = ~np.isclose(np.sum(bcs, axis=0), 0)
        # potentially-useful debug code (not currently used)
        #object.__setattr__(self, '_debug', {}) #dbg
        #self._debug['mn'] = np.array(mn) #dbg
        #self._debug['mx'] = np.array(mx) #dbg
        #self._debug['dims'] = np.array(dims) #dbg
        #self._debug['i0'] = i0 #dbg
        #self._debug['bcs'] = bcs
        #self._debug['ok'] = ok
        #self._debug['wfxs'] = wfxs
        #self._debug['pfxs'] = pfxs
        #self._debug['idcs'] = idcs
        idcs = idcs[:,ok]
        bcs = bcs[:,ok]
        ii = iis[ok]
        #(ii,kk) = np.unique(ii, return_index=True)
        #return {'coordinates':bcs[:,kk], 'faces':self.tess.faces[:,ii], 'voxel_indices':idcs[:,kk]}
        return {'coordinates':bcs, 'faces':self.tess.faces[:,ii], 'voxel_indices':idcs}
    def address(self, data, affine=Ellipsis):
        '''
        cortex.address(points) yields the barycentric coordinates of the given point or points; the
          return value is a dict whose keys are 'face_id' and 'coordinates'. The address may be used
          to interpolate or unaddress either from a surface mesh or from a cortex.
        cortex.address(image) is equivalent to cortex.address(image, mask) where mask is equivalent
          to (numpy.isfinite(image) & image.astype(numpy.bool)).

        The optional argument affine (default: Ellipsis) may be set to an affine transformation that
        should be applied prior to aligning the cortex with an image (if a point-set is given, then
        this parameter is ignored). Ellipsis indicates that cortex.affine should be used.
        '''
        if is_image(data): return self.image_address(data, affine=affine)
        data = np.asarray(data) # data must be a point matrix then
        if len(data.shape) > 2: raise ValueError('point or point matrix required')
        if len(data.shape) == 2: xyz = data.T if data.shape[0] == 3 else data
        else:                    xyz = np.asarray([data])
        if not xyz.flags['WRITEABLE']: xyz = np.array(xyz)
        # now, get the barycentric coordinates...
        n = len(xyz)
        fcount = self.tess.face_count
        # get some relevant structure data
        (wsrf, psrf) = (self.white_surface, self.pial_surface)
        (fwcoords,fpcoords) = (wsrf.face_coordinates, psrf.face_coordinates)
        fids = range(fcount)
        fids = np.concatenate((fids, fids))
        faces = np.concatenate((self.tess.indexed_faces, self.tess.indexed_faces), axis=-1)
        face_centers = np.hstack((wsrf.face_centers, psrf.face_centers))
        try:              shash = spspace.cKDTree(face_centers.T)
        except Exception: shash = spspace.KDTree(face_centers.T)
        (whsh, phsh) = [s.face_hash for s in (wsrf, psrf)]
        # we can define a max distance for when something is too far from a point to plausibly be
        # in a prism:
        wpdist = np.sqrt(np.sum((wsrf.coordinates - psrf.coordinates)**2, axis=0))
        max_dist = np.max(np.concatenate([wsrf.edge_lengths, psrf.edge_lengths, wpdist]))
        # Okay, for each voxel (xyz), we want to find the closest face centers; from those
        # centers, we find the ones for which the nearest point in the plane of the face to the
        # voxel lies inside the face, and of those we find the closest; this nearest point is
        # then used for trilinear interpolation of the points in the prism.
        (N, sofar) = (256, 0)
        # go ahead and make the results
        res_fs = np.full((3, n),     -1, dtype=np.int)
        res_xs = np.full((3, n), np.nan, dtype=np.float)
        # points tha lie outside the pial surface entirely we can eliminate off the bat:
        ii = [(mn <= ix) & (ix <= mx)
              for (px,ix) in zip(psrf.coordinates, xyz.T)
              for (mn,mx) in [(np.min(px), np.max(px))]]
        ii = np.where(ii[0] & ii[1] & ii[2])[0]
        # Okay, we look for those isect's within the triangles
        idcs = []
        for i in range(N):
            if len(ii) == 0: break
            if i >= sofar:
                sofar = max(4, 2*sofar)
                (ds,idcs) = shash.query(xyz[ii], sofar)
            # if dist is greater than max distance, we can skip those
            oks = np.where(ds[:,i] <= max_dist)[0]
            if len(oks) == 0: break
            elif len(oks) < len(ii):
                ii = ii[oks]
                idcs = idcs[oks]
                ds = ds[oks]
            col = fids[idcs[:,i]]
            bcs = geo.prism_barycentric_coordinates(fwcoords[:,:,col], fpcoords[:,:,col], xyz[ii].T)
            # figure out which ones were discovered to be in this prism
            outp = np.isclose(np.sum(bcs, axis=0), 0)
            if not np.all(outp):
                inp    = np.logical_not(outp)
                ii_inp = ii[inp]
                # for those in their prisms, we capture the face id's and coordinates
                res_fs[:, ii_inp] = faces[:, col[inp]]
                res_xs[:, ii_inp] = bcs[:, inp]
                # trim down those that matched so we don't keep looking for them
                ii = ii[outp]
                idcs = idcs[outp]
                ds = ds[outp]
            # And continue!
        # and return the data
        return {'faces': res_fs, 'coordinates': res_xs}
    def unaddress(self, data, surface=0.5):
        '''
        cortex.unaddress(address) yields the (3 x n) coordinate matrix of the given addresses (or,
          if address is singular, the 3D vector) in the given cortex. If the address is a 2D instead
          of a 3D address, then the mid-gray position is returned by default.

        The following options may be given:
          * surface (default: 0.5) specifies the surface to use for 2D addresses; this should be
            either 'white', 'pial', 'midgray', or a real number in the range [0,1] where 0 is the
            white surface and 1 is the pial surface.
        '''
        (faces, coords) = address_data(data, 3, surface=surface)
        (bc, ds) = (coords[:2], coords[2])
        faces = self.tess.index(faces)
        (wx, px) = (self.white_surface.coordinates, self.pial_surface.coordinates)
        if all(len(np.shape(x)) > 1 for x in (faces, coords)):
            (wtx, ptx) = [
                np.transpose([sx[:,ff] if ff[0] >= 0 else null for ff in faces.T], (2,1,0))
                for null in [np.full((3, wx.shape[0]), np.nan)]
                for sx   in (wx, px)]
        elif faces == -1:
            return np.full(selfx.shape[0], np.nan)
        else:
            (wtx, ptx) = [sx[:,faces].T for sx in (wx, px)]
        (wu, pu) = [geo.barycentric_to_cartesian(tx, bc) for tx in (wtx, ptx)]
        return wu*ds + pu*(1 - ds)
    def image_weight(self, img, affine=None):
        '''
        cortex.image_weight(img) yields an image with the spec given by img (which may be an image
          or an image-spec) and with each voxel containing the weight of the cortex in that voxel.
          The weight is equal to the fraction of the voxel's volume that overlaps with the cortex.

        Note: not yet implemented.
        '''
        raise NotImplementedError('image_weight is not yet implemented') #TODO

def is_cortex(c):
    '''
    is_cortex(c) yields True if c is a Cortex object and False otherwise.
    '''
    return isinstance(c, Cortex)
def to_cortex(c):
    '''
    to_cortex(c) yields a Cortex object if the argument c can be coerced to one and otherwise raises
      an error.

    An object can be coerced to a Cortex object if:
      * it is a cortex object
      * it is a tuple (subject, h) where subject is a subject object and h is a subject hemisphere.
    '''
    if is_cortex(c): return c
    elif pimms.is_vector(c) and len(c) == 2:
        (s,h) = c
        if is_subject(s) and pimms.is_str(h):
            if h in s.hemis: return s.hemis[h]
            else: raise ValueError('to_cortex: hemi %s not found in given subject' % h)
    raise ValueError('Could not coerce argument to Cortex object')

####################################################################################################
# These functions deal with cortex_to_image and image_to_cortex interpolation:
def _vertex_to_voxel_linear_interpolation(hemi, gray_indices, image_shape, voxel_to_vertex_matrix):
    if gray_indices is None: raise ValueError('gray indices cannot be None')
    n      = len(gray_indices[0])
    vcount = hemi.vertex_count
    # convert voxels to vertex-space
    xyz = voxel_to_vertex_matrix.dot(np.vstack((np.asarray(gray_indices), np.ones(n))))[0:3].T
    if not xyz.flags['WRITEABLE']: xyz = np.array(xyz)
    # get some relevant structure data
    (fwcoords,fpcoords)  = (hemi.white_surface.face_coordinates, hemi.pial_surface.face_coordinates)
    fids = np.concatenate((range(hemi.tess.face_count), range(hemi.tess.face_count)))
    faces = np.concatenate((hemi.tess.indexed_faces, hemi.tess.indexed_faces), axis=-1)
    face_centers = np.hstack((hemi.white_surface.face_centers, hemi.pial_surface.face_centers))
    try:              shash = spspace.cKDTree(face_centers.T)
    except Exception: shash = spspace.KDTree(face_centers.T)
    (whsh, phsh) = [getattr(hemi, '%s_surface' % s).face_hash for s in ['white', 'pial']]
    # Okay, for each voxel (xyz), we want to find the closest face centers; from those centers, we
    # find the ones for which the nearest point in the plane of the face to the voxel lies inside
    # the face, and of those we find the closest; this nearest point is then used for trilinear
    # interpolation of the points in the triangle.
    N = 256
    sofar = 0
    # go ahead and make our interp matrix
    interp = sps.lil_matrix((n, vcount), dtype=np.float)
    # Okay, we look for those isect's within the triangles
    ii = np.asarray(range(n)) # the subset not yet matched
    for i in range(N):
        if len(ii) == 0: break
        if i >= sofar:
            sofar = max(4, 2*sofar)
            idcs = shash.query(xyz[ii], sofar)[1].T
        # we look at just the i'th column of the indices
        col = fids[idcs[i]]
        bcs = geo.prism_barycentric_coordinates(fwcoords[:,:,col], fpcoords[:,:,col], xyz[ii].T)
        bcs = bcs[0:3] # since the layers are the same in this case...
        outp = np.isclose(np.sum(bcs, axis=0), 0)
        inp = np.logical_not(outp)
        # for those in their prisms, we linearly interpolate using the bc coordinates
        bcs = bcs[:,inp]
        ii_inp = ii[inp]
        fs = faces[:, col[inp]]
        for k in (0,1,2): interp[(ii_inp, fs[k])] = bcs[k]
        # trim down those that matched so we don't keep looking for them
        ii = ii[outp]
        idcs = idcs[:,outp]
        # And continue!
    # last, we normalize the rows
    rowsums = np.asarray(interp.sum(axis=1))[:,0]
    z = np.isclose(rowsums, 0)
    invrows = np.logical_not(z) / (rowsums + z)
    ii = np.asarray(range(n))
    return sps.csc_matrix((invrows, (ii,ii))).dot(interp)
    
def _vertex_to_voxel_lines_interpolation(hemi, gray_indices, image_shape, vertex_to_voxel_matrix):
    ijks = np.asarray(list(gray_indices) if isinstance(gray_indices, colls.Set) else gray_indices)
    ijks = ijks.T if ijks.shape[0] != 3 else ijks
    n = hemi.vertex_count
    # we also need the transformation from surface to voxel
    # given that the voxels assume 0-based indexing, this means that the center of each voxel
    # is at (i,j,k) for integer (i,j,k) > 0; we want the voxels to start and end at integer
    # values with respect to the vertex positions, so we add 1/2 to the vertex
    # positions, which should make the range 0-1, for example, cover the vertices in the first
    # voxel
    tmtx = vertex_to_voxel_matrix[0:3]
    (pialX, whiteX) = [np.dot(tmtx, np.vstack((mtx, np.ones(mtx.shape[1]))))
                       for mtx in [hemi.pial_surface.coordinates, hemi.white_surface.coordinates]]
    # get vectors along each axis
    u = pialX - whiteX
    # normalize these...
    lens = np.sqrt(np.sum(u**2, axis=0))
    z = np.isclose(lens, 0)
    inv_lens = np.logical_not(z) / (lens + z)
    u = inv_lens * u
    usign = np.sign(u)
    z = usign == 0
    u_inv = np.logical_not(z) / (u + z)
    # find smallest and largest voxel indices through which a particular line passes
    (mins, maxs) = [np.array(x) for x in [whiteX, pialX]]
    # extend these a little so that we don't miss any voxels
    tmp = u*(lens*0.05)
    mins -= tmp
    maxs += tmp
    # make a lookup table of gray indices (i,j,k) to the gray voxel's index
    # this is a trick to get this all pushed into c-modules:
    ijk_mult = (image_shape[1]*image_shape[2], image_shape[2], 1)
    def ijk_to_idx(ijk):
        return ijk_mult[0]*ijk[0] + ijk_mult[1]*ijk[1] + ijk[2]
    idcs = ijk_to_idx(ijks)
    index = sps.csr_matrix(
        (range(1, 1+len(idcs)), (np.zeros(len(idcs)), idcs)),
        shape=(1,np.prod(image_shape)),
        dtype=np.int)
    # make the interpolation matrix...
    interp = sps.lil_matrix((len(ijks[0]), n), dtype=np.float)
    # ends are the voxels in which the lines end
    ends = usign * np.ceil(usign*maxs)
    # Okay, we are going to walk along each of the lines...
    ii = np.asarray(range(n))
    while len(ii) > 0:
        # we know that the current min values make a voxel on the line, but the question is, which
        # voxel will be the next along the line?
        # we want the ceiling of the current minimum; however, if the direction is negative (usign)
        # then we need to reverse it, ceiling, then reverse it again:
        z = (mins == np.round(mins))
        uu = usign * np.ceil(usign * mins) # this is, for each dim, the nearest int boundary
        # the new end-point might be farther than max? if so we replace it with max
        uu = usign * np.min((uu, usign*ends), axis=0) # usign here undoes negatives
        uu[z] += usign[z]
        d3 = (uu - mins) * u_inv # the dist in each dim from the nearest integer boundary
        # whichever row of d3 is smallest (rows are x, y, or z) determines how far the next voxel is
        min_d = np.argsort(d3, axis=0)[0]
        addrs = (min_d, range(len(min_d)))
        d = d3[addrs]
        frac = d * inv_lens
        frac[inv_lens == 0] = 1
        # what is the start voxel?
        start = np.asarray(usign * np.floor(usign * mins), dtype=np.int)
        # we want to add these fractions into the interp matrix
        tmp = ijk_to_idx(start)
        oob = np.any(start < 0, axis=0) | (tmp >= index.shape[1])
        tmp[oob] = 0
        found_idcs = index[0, tmp].toarray()[0]
        gray_fis = (found_idcs > 0) & np.logical_not(oob)
        found_idcs = found_idcs[gray_fis] - 1
        interp[(found_idcs, ii[gray_fis])] = frac[gray_fis]
        # update the min values
        mins = mins + d*u
        # if the min is now equal to the max, then we are done with that line!
        keep = np.where(np.logical_not(np.isclose(np.sum((mins - maxs)**2, axis=0), 0))
                        & np.all((maxs - mins) * u_inv > 0, axis=0))[0]
        if len(keep) < len(ii):
            (inv_lens,ii) = [x[keep] for x in (inv_lens,ii)]
            (ends,mins,maxs,usign,u,u_inv) = [x[:,keep] for x in (ends,mins,maxs,usign,u,u_inv)]
    # now we want to scale the rows by their totals
    totals = np.asarray(interp.sum(axis=1))[:,0]
    zs = np.isclose(totals, 0)
    inv_totals = np.logical_not(zs) / (totals + zs)
    rng = range(len(inv_totals))
    interp = sps.csr_matrix((inv_totals, (rng, rng))).dot(interp.tocsc())
    # That's all we have to do!
    return interp

def _vertex_to_voxel_nearest_interpolation(hemi, gray_indices, voxel_to_vertex_matrix):
    if isinstance(hemi, Subject):       hemi   = (hemi.lh, hemi.rh)
    if isinstance(hemi, geo.VertexSet): vcount = hemi.vertex_count
    else:                               vcount = hemi[0].vertex_count + hemi[1].vertex_count
    n   = len(gray_indices[0])
    xyz = voxel_to_vertex_matrix.dot(np.vstack((np.asarray(gray_indices), np.ones(n))))[0:3].T
    if not xyz.flags['WRITEABLE']: xyz = np.array(xyz)
    # find the nearest to the white and pial layers
    if isinstance(hemi, Cortex):
        (wd, wn) = hemi.white_surface.vertex_hash.query(xyz, 1)
        (pd, pn) = hemi.pial_surface.vertex_hash.query(xyz, 1)
        oth = pd < wd
        wn[oth] = pn[oth]
    else:
        (lwd, lwn) = hemi[0].white_surface.vertex_hash.query(xyz, 1)
        (lpd, lpn) = hemi[0].pial_surface.vertex_hash.query(xyz, 1)
        (rwd, rwn) = hemi[1].white_surface.vertex_hash.query(xyz, 1)
        (rpd, rpn) = hemi[1].pial_surface.vertex_hash.query(xyz, 1)
        nn = hemi[0].vertex_count
        oth = lpd < lwd
        lwn[oth] = lpn[oth]
        lwd[oth] = lpd[oth]
        oth = rpd < lwd
        lwn[oth] = rpn[oth] + nn 
        lwd[oth] = rpd[oth]
        oth = rwd < lwd
        lwn[oth] = rwn[oth] + nn
        wn = lwn
    return sps.csr_matrix((np.ones(len(wn)), (range(len(wn)), wn)),
                          shape=(len(gray_indices[0]), vcount),
                          dtype=np.float)
