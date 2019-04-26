####################################################################################################
# neuropythy/mri/core.py
# Simple tools for dealing with the cortical objects and general MRImage data
# By Noah C. Benson

import numpy               as np
import numpy.linalg        as npla
import scipy               as sp
import scipy.sparse        as sps
import scipy.spatial       as spspace
import neuropythy.geometry as geo
import pyrsistent          as pyr
import collections         as colls
import os, sys, types, six, pimms

from itertools import chain

from ..util import (ObjectWithMetaData, to_affine, is_image, is_address, is_tuple, address_data,
                    curry, to_hemi_str)

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
    def __init__(self, name=None, path=None, hemis=None, images=None, meta_data=None,
                 voxel_to_vertex_matrix=None, voxel_to_native_matrix=None):
        self.name                   = name
        self.path                   = path
        self.hemis                  = hemis
        self.images                 = images
        self.voxel_to_vertex_matrix = voxel_to_vertex_matrix
        self.meta_data              = meta_data

    @pimms.param
    def name(nm):
        '''
        sub.name is the name of the subject sub.
        '''
        if nm is None or pimms.is_str(nm): return nm
        else: raise ValueError('subject names must be strings or None')
    @pimms.param
    def path(p):
        '''
        sub.path is the path of the subject's data directory, if any.
        '''
        if p is None or pimms.is_str(p): return p
        else: raise ValueError('subject paths must be strings or None')
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
    @pimms.value
    def native_to_vertex_matrix(native_to_voxel_matrix, voxel_to_vertex_matrix):
        '''
        sub.native_to_vertex_matrix is the affine transformation matrix that converts from the
        subject's 'native' orientation to the vertex orientation.
        '''
        return pimms.imm_array(np.dot(voxel_to_vertex_matrix, native_to_voxel_matrix))
    @pimms.value
    def vertex_to_native_matrix(native_to_vertex_matrix):
        '''
        sub.vertex_to_native_matrix is the inverse matrix of sub.native_to_vertex_matrix.
        '''
        return pimms.imm_array(npla.inv(native_to_vertex_matrix))
    @pimms.param
    def voxel_to_native_matrix(mtx):
        '''
        sub.voxel_to_vertex_matrix is the 4x4 affine transformation matrix that converts from a
        subject's (0-indexed) voxel indices to that subject's 'native' orientation; this is the
        orientation matrix used when exporting a subject's images, and should be the orientation
        encoded in the subject's image data.
        '''
        return pimms.imm_array(to_affine(mtx, 3))
    @pimms.value
    def native_to_voxel_matrix(voxel_to_native_matrix):
        '''
        sub.native_to_voxel_matrix is the inverse matrix of sub.voxel_to_native_matrix.
        '''
        return pimms.imm_array(npla.inv(voxel_to_native_matrix))
    @pimms.value
    def vertex_to_voxel_matrix(voxel_to_vertex_matrix):
        '''
        sub.vertex_to_voxel_matrix is the inverse matrix of sub.voxel_to_vertex_matrix.
        '''
        return pimms.imm_array(npla.inv(voxel_to_vertex_matrix))
    @pimms.param
    def voxel_to_vertex_matrix(mtx):
        '''
        sub.voxel_to_vertex_matrix is the 4x4 affine transformation matrix that converts from
        (i,j,k) indices in the subject's image/voxel space to (x,y,z) coordinates in the subject's
        cortical surface space.
        '''
        return pimms.imm_array(to_affine(mtx, 3))

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
        if is_image(lh_gray_mask): lh_gray_mask = lh_gray_mask.get_data()
        return tuple([pimms.imm_array(x) for x in np.where(lh_gray_mask)])
    @pimms.value
    def rh_gray_indices(rh_gray_mask):
        '''
        sub.rh_gray_indices is equivalent to numpy.where(sub.rh_gray_mask).
        '''
        if rh_gray_mask is None: return None
        if is_image(rh_gray_mask): rh_gray_mask = rh_gray_mask.get_data()
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
        if is_image(lh_white_mask): lh_white_mask = lh_white_mask.get_data()
        idcs = np.transpose(np.where(lh_white_mask))
        return frozenset([tuple(row) for row in idcs])
    @pimms.value
    def rh_white_indices(rh_white_mask):
        '''
        sub.rh_white_indices is a frozenset of the indices of the white voxels in the given
        subject's rh, represented as 3-tuples.
        '''
        if rh_white_mask is None: return None
        if is_image(rh_white_mask): rh_white_mask = rh_white_mask.get_data()
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
        if is_image(img): img = img.get_data()
        return np.asarray(img).shape

    @pimms.value
    def lh_vertex_to_voxel_linear_interpolation(lh_gray_indices, lh, image_dimensions,
                                                voxel_to_vertex_matrix):
        '''
        sub.lh_gray_vertex_to_voxel_linear_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.lh_gray_indices.
        The method works by projecting the vectors from the white surface vertices to the pial
        surface vertices into the the ribbon and weighting them by the fraction of the vector that
        lies in the voxel.
        '''
        return _vertex_to_voxel_linear_interpolation(lh, lh_gray_indices, image_dimensions,
                                                     voxel_to_vertex_matrix)
    @pimms.value
    def rh_vertex_to_voxel_linear_interpolation(rh_gray_indices, rh, image_dimensions,
                                                voxel_to_vertex_matrix):
        '''
        sub.rh_gray_vertex_to_voxel_linear_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.rh_gray_indices.
        The method works by projecting the vectors from the white surface vertices to the pial
        surface vertices into the the ribbon and weighting them by the fraction of the vector that
        lies in the voxel.
        '''
        return _vertex_to_voxel_linear_interpolation(rh, rh_gray_indices, image_dimensions,
                                                     voxel_to_vertex_matrix)
    @pimms.value
    def lh_vertex_to_voxel_heaviest_interpolation(lh_vertex_to_voxel_linear_interpolation):
        '''
        sub.lh_gray_vertex_to_voxel_heaviest_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.lh_gray_indices.
        The method works by projecting the vectors from the white surface vertices to the pial
        surface vertices into the the ribbon and weighting them by the fraction of the vector that
        lies in the voxel; the column in each row of the interpolation matrix with the highest
        weight is then given a value of 1 while all other rows are given values of 0. This is
        equivalent to performing nearest-neighbor interpolation while controlling for the depth of
        the voxel in the cortex.
        '''
        interp = lh_vertex_to_voxel_linear_interpolation
        (rs,cs) = interp.shape
        argmaxs = np.asarray(interp.argmax(axis=1))[:,0]
        return sps.csr_matrix((np.ones(rs, dtype=np.int), (range(rs), argmaxs)),
                              shape=interp.shape,
                              dtype=np.int)
    @pimms.value
    def rh_vertex_to_voxel_heaviest_interpolation(rh_vertex_to_voxel_linear_interpolation):
        '''
        sub.rh_gray_vertex_to_voxel_heaviest_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.rh_gray_indices.
        The method works by projecting the vectors from the white surface vertices to the pial
        surface vertices into the the ribbon and weighting them by the fraction of the vector that
        lies in the voxel; the column in each row of the interpolation matrix with the highest
        weight is then given a value of 1 while all other rows are given values of 0. This is
        equivalent to performing nearest-neighbor interpolation while controlling for the depth of
        the voxel in the cortex.
        '''
        interp = rh_vertex_to_voxel_linear_interpolation
        (rs,cs) = interp.shape
        argmaxs = np.asarray(interp.argmax(axis=1))[:,0]
        return sps.csr_matrix((np.ones(rs, dtype=np.int), (range(rs), argmaxs)),
                              shape=interp.shape,
                              dtype=np.int)
    @pimms.value
    def vertex_to_voxel_linear_interpolation(lh_vertex_to_voxel_linear_interpolation,
                                             rh_vertex_to_voxel_linear_interpolation):
        '''
        sub.rh_gray_vertex_to_voxel_linear_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.gray_indices. The vertex-values should be concatenated, LH
          values then RH values.
        The method works by projecting the vectors from the white surface vertices to the pial
        surface vertices into the the ribbon and weighting them by the fraction of the vector that
        lies in the voxel.
        '''
        (lm, rm) = (lh_vertex_to_voxel_linear_interpolation,rh_vertex_to_voxel_linear_interpolation)
        (ls, rs) = (lm.shape, rm.shape)
        (lels, rels) = (sps.find(lm), sps.find(rm))
        rels = (rels[0] + ls[0], rels[1] + ls[1], rels[2])
        (rows,cols,vals) = [np.concatenate(pair) for pair in zip(lels, rels)]
        return sps.csr_matrix((vals, (rows,cols)), shape=np.add(ls, rs))

    @pimms.value
    def lh_vertex_to_voxel_lines_interpolation(lh_gray_indices, lh, image_dimensions,
                                               vertex_to_voxel_matrix):
        '''
        sub.lh_gray_vertex_to_voxel_lines_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.lh_gray_indices.
        The method works by projecting the vectors from the white surface vertices to the pial
        surface vertices into the the ribbon and weighting them by the fraction of the vector that
        lies in the voxel.
        '''
        return _vertex_to_voxel_lines_interpolation(lh, lh_gray_indices, image_dimensions,
                                                    vertex_to_voxel_matrix)
    @pimms.value
    def rh_vertex_to_voxel_lines_interpolation(rh_gray_indices, rh, image_dimensions,
                                               vertex_to_voxel_matrix):
        '''
        sub.rh_gray_vertex_to_voxel_lines_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.rh_gray_indices.
        The method works by projecting the vectors from the white surface vertices to the pial
        surface vertices into the the ribbon and weighting them by the fraction of the vector that
        lies in the voxel.
        '''
        return _vertex_to_voxel_lines_interpolation(rh, rh_gray_indices, image_dimensions,
                                                    vertex_to_voxel_matrix)
    @pimms.value
    def vertex_to_voxel_lines_interpolation(lh_vertex_to_voxel_lines_interpolation,
                                            rh_vertex_to_voxel_lines_interpolation):
        '''
        sub.rh_gray_vertex_to_voxel_lines_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.gray_indices. The vertex-values should be concatenated, LH
          values then RH values.
        The method works by projecting the vectors from the white surface vertices to the pial
        surface vertices into the the ribbon and weighting them by the fraction of the vector that
        lies in the voxel.
        '''
        (lm, rm) = (lh_vertex_to_voxel_lines_interpolation, rh_vertex_to_voxel_lines_interpolation)
        (ls, rs) = (lm.shape, rm.shape)
        (lels, rels) = (sps.find(lm), sps.find(rm))
        rels = (rels[0] + ls[0], rels[1] + ls[1], rels[2])
        (rows,cols,vals) = [np.concatenate(pair) for pair in zip(lels, rels)]
        return sps.csr_matrix((vals, (rows,cols)), shape=np.add(ls, rs))

    @pimms.value
    def lh_vertex_to_voxel_nearest_interpolation(lh_gray_indices, lh, voxel_to_vertex_matrix):
        '''
        sub.lh_gray_vertex_to_voxel_nearest_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.lh_gray_indices.
        The method used is nearest-neighbors to either the closest pial or white surface vertex.
        '''
        return _vertex_to_voxel_nearest_interpolation(lh, lh_gray_indices, voxel_to_vertex_matrix)
    @pimms.value
    def rh_vertex_to_voxel_nearest_interpolation(rh_gray_indices, rh, voxel_to_vertex_matrix):
        '''
        sub.rh_gray_vertex_to_voxel_nearest_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.lh_gray_indices.
        The method used is nearest-neighbors to either the closest pial or white surface vertex.
        '''
        return _vertex_to_voxel_nearest_interpolation(rh, rh_gray_indices, voxel_to_vertex_matrix)
    @pimms.value
    def vertex_to_voxel_nearest_interpolation(lh_vertex_to_voxel_nearest_interpolation,
                                              rh_vertex_to_voxel_nearest_interpolation):
        '''
        sub.rh_gray_vertex_to_voxel_nearest_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.lh_gray_indices.
        The method used is nearest-neighbors to either the closest pial or white surface vertex.
        '''
        (lm, rm) = (lh_vertex_to_voxel_nearest_interpolation,
                    rh_vertex_to_voxel_nearest_interpolation)
        (ls, rs) = (lm.shape, rm.shape)
        (lels, rels) = (sps.find(lm), sps.find(rm))
        rels = (rels[0] + ls[0], rels[1] + ls[1], rels[2])
        (rows,cols,vals) = [np.concatenate(pair) for pair in zip(lels, rels)]
        return sps.csr_matrix((vals, (rows,cols)), shape=np.add(ls, rs))
    
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
        sub.path_join(args...) is equivalent to os.path.join(sub.path, args...).
        '''
        return os.path.join(sub.path, *args)
    def cortex_to_image(self, data,
                        hemi=None, method='linear', fill=0, dtype=None, affine=None, shape=None):
        '''
        sub.cortex_to_image(data, hemi) projects the given cortical-surface data to the given
          subject's gray-matter voxels of the given hemisphere and returns the resulting numpy
          array.
        sub.cortex_to_image((lh_data, rh_data)) projects into both hemispheres.
    
        The following options may be given:
          * method (default: 'linear') specifies that a particular method should be used; valid
            options are 'linear', 'heaviest', and 'nearest'. The 'linear' method uses the
            lh_vertex_to_voxel_linear_interpolation and rh_vertex_to_voxel_linear_interpolation 
            matrices while 'nearest' uses the nearest-neighbor interpolation. The 'heaviest' method
            uses the highest-valued weight in the 'linear' interpolation matrix, which is
            equivalent to using nearest-neighbor interpolation after controlling for the depth of
            the voxel with respect to the vertices. The 'linear' method is generally preferred
            unless your data is discreet, in which the 'heaviest' method is generally best.
          * fill (default: 0) specifies the value to be assigned to all voxels not in the gray mask
            or voxels in the gray-mask that are missed by the interpolation method.
          * dtype (default: None) specifies the data type that should be exported. If None, this
            will be automatically set to np.float32 for floating-point data and np.int32 for integer
            data.
          * affine (default: None) specifies the affine transformation that should be used to align
            the cortical surfaces with the voxels. If None, then the subject's vertex-to-voxel
            matrix will be used.
          * shape (default: None) specifies the dimensions of the output array; if None, then the
            subject's image_dimensions is used.
        '''
        # what hemisphere(s)?
        hemi = to_hemi_str(hemi)
        hemi = ('lh', None) if hemi == 'lh' else (None, 'rh') if hemi == 'rh' else ('lh','rh')
        nhems = len([x for x in hemi if x])
        tr = None
        # Make the data match this format...
        if   is_tuple(data) and len(data) == 2: pass
        elif pimms.is_map(data): data = ((None,       data['rh']) if hemi[0] is None else
                                         (data['lh'], None)       if hemi[1] is None else
                                         (data['lh'], data['rh']))
        elif pimms.is_str(data): data = (data, data)
        elif pimms.is_array(data):
            data = np.asarray(data)
            sh = data.shape
            if   nhems == 1: data = tuple([None if k is None else data for k in hemi])
            elif 2 in sh:
                ax = next([k for (k,d) in enumerate(sh) if d == 2])
                data = np.transpose(data, list(chain([ax], range(0, ax), range(ax+1, len(sh)))))
            else: raise ValueError('bad data shape for number of hemispheres')
        else: raise ValueError('cortex_to_image cannot deduce data structure')
        # Make sure we have all the data...
        data = list(data)
        for (dat,h,ii) in zip(data, hemi, [0,1]):
            if h is None: continue
            if dat is None: raise ValueError('hemisphere %s requested but data not provided' % h)
            hem = self.hemis[h]
            dat = hem.property(dat)
            if not pimms.is_array(dat, None, (1,2)):
                raise ValueError('data given for %s is neither a vector nor a matrix' % h)
            if pimms.is_matrix(dat) and dat.shape[0] != hem.vertex_count: dat = dat.T
            if dat.shape[0] != hem.vertex_count:
                raise ValueError('vertex data for %s does not match number of vertices' % h)
            data[ii] = dat
        # data can be a matrix, but both datas must be the same if so; also figure out the number
        # of frames in case matrices were provided:
        if data[0] is not None and data[1] is not None:
            if not pimms.is_matrix(data[0]):
                if pimms.is_matrix(data[1]):
                    raise ValueError('hemisphere data shapes must match in non-vertex dimension')
                else: frames = 1
            elif not pimms.is_matrix(data[1]):
                raise ValueError('hemisphere data shapes must match in non-vertex dimension')
            elif data[0].shape[1] != data[1].shape[1]:
                raise ValueError('hemisphere data shapes must match in non-vertex dimension')
            else: frames = data[0].shape[1]
        elif data[0] is None:
            frames = data[1].shape[1] if pimms.is_matrix(data[1]) else 1
        else:
            frames = data[0].shape[1] if pimms.is_matrix(data[0]) else 1
        # Figure out the dtype
        if dtype is None:
            # check the input...
            floatq = any(d is not None and pimms.is_array(d, np.inexact) for d in data)
            dtype = np.float32 if floatq else np.int32
        shape = self.image_dimensions if shape is None else shape
        # make our output array
        dims = shape + (frames,) if frames > 1 and len(shape) < 4 else shape
        arr = np.full(dims, fill, dtype=dtype)
        # if we are given a transform matrix, we have to build the interpolation
        # what method? specifically, what matrices to use?
        if pimms.is_str(method):
            method = 'auto' if method is None else method.lower()
            if method in ['auto', 'automatic']:
                method = 'linear' if np.issubdtype(dtype, np.inexact) else 'heaviest'
            # if there is no affine specified, we can use one of the pre-built
            # matrices for this subject
            if affine is None and np.array_equal(shape, self.image_dimensions):
                if nhems == 2:
                    interp  = getattr(self, 'vertex_to_voxel_%s_interpolation' % method)
                    indices = self.gray_indices
                else:
                    interp =  [getattr(self, '%s_vertex_to_voxel_%s_interpolation' % (h, method))
                               for h in hemi if h is not None][0]
                    indices = [getattr(self, '%s_gray_indices'%h) for h in hemi if h is not None][0]
            else:
                # we need to build a matrix
                tmp = [cortex_to_image_interpolation(h, mask=mask, affine=affine,
                                                     method=method, shape=shape)
                       for h in hemi if h is not None]
                indices = (tmp[0][0], tmp[1][0])
                interp = (tmp[0][1], tmp[1][1])
                interp  = interp[0]  if nhems == 1 else sps.hstack(interp)
                indices = indices[0] if nhems == 1 else tuple(np.hstack(indices))
        else:
            indices = mask
            interp = method
        data = [x for x in data if x is not None][0] if nhems == 1 else np.concatenate(data)
        # if the fill is non-zero, we need to note where the interp matrix misses
        misses = None if fill == 0 else np.where(np.abs(interp).sum(axis=1) == 0)
        arr[indices] = interp.dot(data)
        # note the misses if there are some
        if misses: arr[misses] = fill
        # That's everything!
        return arr
    def image_to_cortex(self, image,
                        surface='midgray', hemi=None, affine=None, method=None, fill=0, dtype=None,
                        weights=None):
        '''
        sub.image_to_cortex(image) is equivalent to the tuple
          (sub.lh.from_image(image), sub.rh.from_image(image)).
        sub.image_to_cortex(image, surface) uses the given surface (see also cortex.surface).
        '''
        if hemi is None: hemi = 'both'
        hemi = hemi.lower()
        if hemi in ['both', 'lr', 'all', 'auto']:
            return tuple([self.image_to_cortex(image, surface=surface, hemi=h, affine=affine,
                                               method=method, fill=fill, dtype=dtype, weights=weights)
                          for h in ['lh', 'rh']])
        else:
            hemi = getattr(self, hemi)
            return hemi.from_image(image, surface=surface, affine=affine,
                                   method=method, fill=fill, dtype=dtype, weights=weights,
                                   native_to_vertex_matrix=self.native_to_vertex_matrix)
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
    def __init__(self, chirality, tess, surfaces, registrations, properties=None, meta_data=None):
        self.chirality = chirality
        self.surface_coordinates = surfaces
        self.meta_data = meta_data
        geo.Topology.__init__(self, tess, registrations, properties=properties)
        self.chirality = chirality
        self.surface_coordinates = surfaces
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
    def surface(self, name='white'):
        '''
        cortex.surface() yields the white surface of the given cortex
        cortex.surface(name) yields the surface with the given name (e.g., 'white', 'pial',
          'inflated', 'midgray').
        cortex.surface(fraction) yields the surface that is <fraction> of the distance from white
          to pial (0 is equivalent to 'white'; 1 is equivalent to 'pial'). Layers outside of the
          range 0-1 may be returned by following the vectors between white and pial surfaces, but
          they may have odd appearances, and this should not be confused with surface inflation.
        cortex.surface([dist]) yields the layer that is the given distance from the white surface.
        '''
        if pimms.is_str(name):
            if name.lower() == 'midgray' and 'midgray' not in self.surfaces:
                return self.midgray_surface # in case it's been provided via overloading
            return self.surfaces[name]
        elif pimms.is_vector(name, 'real') and len(name) == 1:
            x0 = self.white_surface.coordinates
            dx = self.white_to_pial_vectors
            return self.make_mesh(x0 + name[0]*dx)
        elif pimms.is_real(name):
            x0 = self.white_surface.coordinates
            x1 = self.pial_surface.coordinates
            return self.make_mesh((1 - name)*x0 + name*x1)
        else:
            raise ValueError('could not understand surface layer: %s' % name)
    def from_image(self, image, surface='midgray', affine=None, method=None, fill=0, dtype=None,
                   native_to_vertex_matrix=None, weights=None):
        '''
        cortex.from_image(image) is equivalent to cortex.midgray_surface.from_image(image).
        cortex.from_image(image, surface) uses the given surface (see also cortex.surface).
        '''
        mesh = geo.to_mesh((self, surface))
        return mesh.from_image(image, affine=affine, method=method, fill=fill, dtype=dtype,
                               native_to_vertex_matrix=native_to_vertex_matrix, weights=weights)
    def address(self, data, indices=None, native_to_vertex_matrix=None):
        '''
        cortex.address(points) yields the barycentric coordinates of the given point or points; the
          return value is a dict whose keys are 'face_id' and 'coordinates'. The address may be used
          to interpolate or unaddress either from a surface mesh or from a cortex.
        cortex.address(image, idcs) yields the addresses of the voxels with the given indices in the
          given image. The indices should either be a boolean mask image the same size as data or be
          identical in format to the return value value of numpy.where().
        cortex.address(image) is equivalent to cortex.address(image, mask) where mask is equivalent
          to (numpy.isfinite(image) & image.astype(numpy.bool)).

        If an image is provided as input then the optional argument native_to_vertex_matrix may
        provide an affine-transformation from the image's native coordinate system to the cortex's
        vertex coordinate system.
        '''
        idcs = indices
        if is_image(data):
            arr = np.asarray(data.get_data())
            if idcs is None: idcs = np.isfinte(arr) & arr.astype(np.bool)
            if pimms.is_array(idcs, None, 3): idcs = np.where(idcs)
            aff = data.affine
            idcs = np.asarray(idcs)
            if idcs.shape[0] != 3: idcs = idcs.T
            if native_to_vertex_matrix is not None:
                aff = np.dot(native_to_vertex_matrix, aff)
            xyz = np.dot(
                aff,
                np.concatenate((idcs, np.ones(1 if len(idcs.shape) == 1 else (1, idcs.shape[1])))))
            return self.address(xyz[:3])
        else: data = np.asarray(data) # data must be a point matrix then
        if len(data.shape) > 2: raise ValueError('point or point matrix required')
        if len(data.shape) == 2: xyz = data.T if data.shape[0] == 3 else data
        else:                    xyz = np.asarray([data])
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
        # Okay, for each voxel (xyz), we want to find the closest face centers; from those
        # centers, we find the ones for which the nearest point in the plane of the face to the
        # voxel lies inside the face, and of those we find the closest; this nearest point is
        # then used for trilinear interpolation of the points in the prism.
        (N, sofar) = (256, 0)
        # go ahead and make the results
        res_fs = np.full((3, n),     -1, dtype=np.int)
        res_xs = np.full((3, n), np.nan, dtype=np.float)
        # Okay, we look for those isect's within the triangles
        ii = np.asarray(range(n)) # the subset not yet matched
        idcs = []
        for i in range(N):
            if len(ii) == 0: break
            if i >= sofar:
                sofar = max(4, 2*sofar)
                idcs = shash.query(xyz[ii], sofar)[1].T
            # we look at just the i'th column of the indices
            col = fids[idcs[i]]
            bcs = geo.prism_barycentric_coordinates(fwcoords[:,:,col], fpcoords[:,:,col], xyz[ii].T)
            # figure out which ones were discovered to be in this prism
            outp   = np.isclose(np.sum(bcs, axis=0), 0)
            inp    = np.logical_not(outp)
            ii_inp = ii[inp]
            # for those in their prisms, we capture the face id's and coordinates
            res_fs[:, ii_inp] = faces[:, col[inp]]
            res_xs[:, ii_inp] = bcs[:, inp]
            # trim down those that matched so we don't keep looking for them
            ii = ii[outp]
            idcs = idcs[:,outp]
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

def cortex_to_image_interpolation(obj, mask=None, affine=None, method='linear', shape=None):
    '''
    cortex_to_image_interpolation(obj) yields a tuple (indices, interp) where indices is a tuple of
      voxel indices and interp is an interpolation matrix that converts a vector of cortical
      surface vertex values into a vector of of voxel values with the same ordering as those given
      in indices.

    The argument obj may be either a subject or a cortex or a tuple of (lh, rh) cortices. If the obj
    specifies two cortices, then the interp matrix will be arranged such that the vertex-value
    vector with which interp should be multiplied must first list the LH values then the RH values;
    i.e., image[indices] = interp.dot(join(lh_values, rh_values)).

    The following options are accepted:
      * mask (default: None) specifies the mask of vertices that should be interpolated; if this is
        None, then this will attempt to use the gray_indices of the object if the object is a
        subject, otherwise this is equivalent to 'all'. The special value 'all' indicates that all
        vertices overlapping with the cortex should be interpolated. Mask may be given as a boolean
        mask or as a set or as a tuple/matrix of indices equivalent to numpy.where(binary_mask) or
        its transpose. The value indices that is returned will be identical to mask if mask is given
        in the same format.
      * affine (default: None) specifies the affine transform that is used to align the vertices
        with the voxels. In voxel-space, the voxel with index (i,j,k) is centered at at the point
        (i,j,k) and is 1 unit wide in every direction. If the value None is given, will attempt to
        use the object's vertex_to_voxel_matrix if object is a subject, otherwise will use a
        FreeSurfer-like orientation that places the vertex origin in the center of the image.
      * shape (default: None) specifies the size of the resulting image as a tuple. If None is given
        then the function attempts to deduce the correct shape; if obj is a subject, then its
        image_dimensions tuple is used. Otherwise, the size is deduced from the mask, if possible;
        if it cannot be deduced from the mask, then (256, 256, 256) is used.
      * method (default: 'linear') specifies the method to use. May be 'linear', 'heaviest', or
        'nearest'.
    '''
    # get the min/max values of the coordinates (for deducing sizes, if necessary)
    if mask is None:
        if shape is None and affine is None:
            # we have no way to deduce anything
            shape = (256,256,256)
            affine = ([[-1,0,0],[0,0,-1],[0,1,0]], [128,128,128])
        elif shape is None:
            shape = (256,256,256)
        elif affine is None:
            affine = ([[-1,0,0],[0,0,-1],[0,1,0]], np.asarray(shape) / 2)
        mask = obj.gray_indices if isinstance(obj, Subject) else 'all'
    # okay, having handled that no-arg case, lets parse the argument we have
    if pimms.is_matrix(mask):
        # we take this to be the list; we don't chante its order
        if not is_tuple(mask) or len(mask) != 3:
            mask = np.asarray(mask, dtype=np.int)
            mask = tuple(mask.T if mask.shape[0] != 3 else mask)
    elif isinstance(mask, colls.Set):
        # we have to convert this into a propert mask
        mask = np.asarray(list(mask)).T
        tmp = np.full(shape, False)
        tmp[tuple(mask)] = True
        mask = np.where(tmp)
    elif pimms.is_array(mask, dims=3):
        if shape is None: shape = mask.shape
        mask = np.where(mask)
    elif pimms.is_str(mask) and mask.lower() == 'all':
        if shape is None: shape = (256,256,256)
        mask = np.where(np.ones(shape))
    else:
        raise ValueError('Could not understand mask argument')
    # at this point, we can deduce shape from mask and affine from shape
    if shape is None:  shape  = np.asarray(np.ceil(np.max(mask, axis=0)), dtype=np.int)
    if affine is None:
        if isinstance(obj, Subject):
            affine = obj.vertex_to_voxel_matrix
        else:
            affine = ([[-1,0,0],[0,0,-1],[0,1,0]], np.asarray(shape) / 2)
    affine = to_affine(affine, 3)
    hems = (obj.lh, obj.rh) if isinstance(obj, Subject) else (obj,)
    # all arguments are basically pre-processed; we just need to make the interpolation
    method = 'auto' if method is None else method.lower()
    if method in ['linear', 'auto', 'automatic']:
        interp = [_vertex_to_voxel_linear_interpolation(h, mask, shape, affine) for h in hems]
        if len(interp) == 1: interp = interp[0]
        else: interp = sps.hstack(interp)
    elif method in ['lines', 'line']:
        interp = [_vertex_to_voxel_lines_interpolation(h, mask, shape, affine) for h in hems]
        if len(interp) == 1: interp = interp[0]
        else: interp = sps.hstack(interp)
    elif method in ['heaviest', 'heavy', 'weight', 'weightiest']:
        interp = [_vertex_to_voxel_linear_interpolation(h, mask, shape, affine) for h in hems]
        if len(interp) == 1: interp = interp[0]
        else: interp = sps.hstack(interp)
        # convert to binary matrix:
        (rs,cs) = interp.shape
        argmaxs = np.asarray(interp.argmax(axis=1))[:,0]
        return sps.csr_matrix((np.ones(rs, dtype=np.int), (range(rs), argmaxs)),
                              shape=interp.shape,
                              dtype=np.int)
    elif method in ['nearest', 'near', 'nearest-neighbor', 'nn']:
        aff = npla.inv(affine)
        interp = [_vertex_to_voxel_nearest_interpolation(h, mask, aff) for h in hems]
        if len(interp) == 1: interp = interp[0]
        else: interp = sps.hstack(interp)
    else:
        raise ValueError('unsupported method: %s' % method)
    # That's it; we have the interp matrix and the indices
    return (mask, interp)
