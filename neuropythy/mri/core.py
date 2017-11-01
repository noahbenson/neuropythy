####################################################################################################
# neuropythy/mri/core.py
# Simple tools for dealing with the cortical objects and general MRImage data
# By Noah C. Benson

import numpy               as np
import numpy.linalg        as npla
import scipy               as sp
import scipy.sparse        as sps
import neuropythy.geometry as geo
import pyrsistent          as pyr
import os, sys, types, pimms

from neuropythy.util import (ObjectWithMetaData, to_affine)

if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls

def _line_voxel_overlap(vx, xs, xe):
    '''
    _line_voxel_overlap((i,j,k), xs, xe) yields the fraction of the vector from xs to xe that
      overlaps with the voxel (i,j,k); i.e., the voxel that starts at (i,j,k) and ends at
      (i,j,k) + 1.
    '''
    # We want to consider the line segment from smaller to larger coordinate:
    seg = [(x0,x1) if x0 <= x1 else (x1,x0) for (x0,x1) in zip(xs,xe)]
    # First, figure out intersections of start and end values
    isect = [(min(seg_end, vox_end), max(seg_start, vox_start))
             if seg_end >= vox_start and seg_start <= vox_end else None
             for ((seg_start,seg_end), (vox_start,vox_end)) in zip(seg, [(x,x+1) for x in vx])]
    # If any of these are None, we can't possibly overlap
    if None in isect or any(a == b for (a,b) in isect): return 0.0
    return npla.norm([e - s for (s,e) in isect]) / npla.norm([e - s for (s,e) in zip(xs,xe)])

def _vertex_to_voxel_line_interpolation(hemi, gray_indices, vertex_to_voxel_matrix):
    # 1-based indexing is assumed:
    idcs = np.transpose(gray_indices) + 1
    # we also need the transformation from surface to voxel
    tmtx = vertex_to_voxel_matrix
    # given that the voxels assume 1-based indexing, this means that the center of each voxel
    # is at (i,j,k) for integer (i,j,k) > 0; we want the voxels to start and end at integer
    # values with respect to the vertex positions, so we subtract 1/2 from the vertex
    # positions, which should make the range 0-1, for example, cover the vertices in the first
    # voxel
    txcoord = lambda mtx: np.dot(tmtx[0:3], np.vstack((mtx, np.ones(mtx.shape[1])))) + 1.5
    # Okay; get the transformed coordinates for white and pial surfaces:
    pialX = txcoord(hemi.pial_surface.coordinates)
    whiteX = txcoord(hemi.white_surface.coordinates)
    # make a list of voxels through which each vector passes:
    min_idx = np.min(idcs, axis=1)
    max_idx = np.max(idcs, axis=1)
    vtx_voxels = [((i,j,k), (id, olap))
                  for (id, (xs, xe)) in enumerate(zip(whiteX.T, pialX.T))
                  for i in range(int(np.floor(min(xs[0], xe[0]))), int(np.ceil(max(xs[0], xe[0]))))
                  for j in range(int(np.floor(min(xs[1], xe[1]))), int(np.ceil(max(xs[1], xe[1]))))
                  for k in range(int(np.floor(min(xs[2], xe[2]))), int(np.ceil(max(xs[2], xe[2]))))
                  for olap in [_line_voxel_overlap((i,j,k), xs, xe)]
                  if olap > 0]
    # and accumulate these lists... first group by voxel index then sum across these
    first_fn = lambda x: x[0]
    vox_byidx = {vox: ([q[0] for q in dat], [q[1] for q in dat])
                 for (vox,xdat) in itertools.groupby(sorted(vtx_voxels,key=first_fn), key=first_fn)
                 for dat in [[q[1] for q in xdat]]}
    v2v_map = {tuple([i-1 for i in idx]): (ids, np.array(olaps) / np.sum(olaps))
               for idx in idcs
               if idx in vox_byidx
               for (ids, olaps) in [vox_byidx[idx]]}
    # we just need to put these into a matrix; we need a voxel-ijk-to-index translation
    ijk2idx = {tuple(ijk):idx for (idx,ijk) in enumerate(zip(*gray_indices))}
    # we now just need to put things in order:
    interp = sps.lil_matrix((len(gray_index[0]), hemi.vertex_count), dtype=np.float)
    for (ijk,dat) in six.iteritems(ijk2idx):
        ijk = ijk2idx[ijk]
        for (idx,olap) in zip(*dat):
            interp[idx, ijk] = olap
    return interp.tocsr()

def _vertex_to_voxel_nearest_interpolation(hemi, gray_indices, voxel_to_vertex_matrix):
    xyz = voxel_to_vertex_matrix.dot(np.vstack((np.asarray(gray_indices) + 1,
                                                np.ones(len(gray_indices[0]))))).T
    # find the nearest to the white and pial layers
    (dw, nw) = hemi.white_surface.vertex_hash.query(xyz, 1)
    (dp, np)  = hemi.pial_surface.vertex_hash.query(xyz, 1)
    oth = dp < dw
    nw[oth] = np[oth]
    interp = sps.lil_matrix((len(gray_index[0]), hemi.vertex_count), dtype=np.float)
    for (ii,n) in enumerate(nw):
        interp[ii,n] = 1
    return interp.tocsr()

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
                 voxel_to_vertex_matrix=None):
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
    @pimms.param
    def voxel_to_vertex_matrix(mtx):
        '''
        sub.voxel_to_vertex_matrix is the 4x4 affine transformation matrix that converts from
        (i,j,k) indices in the subject's image/voxel space to (x,y,z) coordinates in the subject's
        cortical surface space.
        '''
        return pimms.imm_array(to_affine(mtx, 3))
    @pimms.value
    def vertex_to_voxel_matrix(voxel_to_vertex_matrix):
        '''
        sub.vertex_to_voxel_matrix is the inverse matrix of sub.voxel_to_vertex_matrix.
        '''
        return npla.inv(voxel_to_vertex_matrix)

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
        return tuple([pimms.imm_array(x) for x in np.where(lh_gray_mask)])
    @pimms.value
    def rh_gray_indices(rh_gray_mask):
        '''
        sub.rh_gray_indices is equivalent to numpy.where(sub.rh_gray_mask).
        '''
        return tuple([pimms.imm_array(x) for x in np.where(rh_gray_mask)])
    @pimms.value
    def gray_indices(lh_gray_indices, rh_gray_indices):
        '''
        sub.gray_indices is a frozenset of the indices of the gray voxels in the given subject
        represented as 3-tuples.
        '''
        if   lh_gray_indices is None and rh_gray_indices is None: return None
        if lh_gray_indices is None: lh_gray_indices = frozenset([])
        else: lh_gray_indices = frozenset([tuple(idx) for idx in np.transpose(lh_gray_indices)])
        if rh_gray_indices is None: rh_gray_indices = frozenset([])
        else: rh_gray_indices = frozenset([tuple(idx) for idx in np.transpose(rh_gray_indices)])
        return frozenset(lh_gray_indices | rh_gray_indices)
    @pimms.value
    def lh_white_indices(lh_white_mask):
        '''
        sub.lh_white_indices is a frozenset of the indices of the white voxels in the given
        subject's lh, represented as 3-tuples.
        '''
        if lh_white_mask is None: return None
        idcs = np.transpose(np.where(lh_white_mask))
        return frozenset([tuple(row) for row in idcs])
    @pimms.value
    def rh_white_indices(rh_white_mask):
        '''
        sub.rh_white_indices is a frozenset of the indices of the white voxels in the given
        subject's rh, represented as 3-tuples.
        '''
        if rh_white_mask is None: return None
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
        sub.image_dimensions is a tuple of the size of an anatomical image for the given subject.
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
        return np.asarray(img).shape

    @pimms.value
    def lh_vertex_to_voxel_line_interpolation(lh_gray_indices, lh, vertex_to_voxel_matrix):
        '''
        sub.lh_gray_vertex_to_voxel_line_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.lh_gray_indices.
        The method works by projecting the vectors from the white surface vertices to the pial
        surface vertices into the the ribbon and weighting them by the fraction of the vector that
        lies in the voxel.
        '''
        return _vertex_to_voxel_line_interpolation(lh, lh_gray_indices, vertex_to_voxel_matrix)
    @pimms.value
    def rh_vertex_to_voxel_line_interpolation(rh_gray_indices, rh, vertex_to_voxel_matrix):
        '''
        sub.rh_gray_vertex_to_voxel_line_interpolation is a scipy sparse matrix representing the
          interpolation from the vertices into the voxels; the ordering of the voxels that is
          produced by the dot-product of this matrix with the vector of vertex-values is the same
          as the ordering used in sub.rh_gray_indices.
        The method works by projecting the vectors from the white surface vertices to the pial
        surface vertices into the the ribbon and weighting them by the fraction of the vector that
        lies in the voxel.
        '''
        return _vertex_to_voxel_line_interpolation(rh, rh_gray_indices, vertex_to_voxel_matrix)
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
    def cortex_to_image(data, hemi=None, method='lines', fill=0, dtype=None):
        '''
        sub.cortex_to_ribbon(data, hemi) projects the given cortical-surface data to the given
          subject's gray-matter voxels of the given hemisphere and returns the resulting numpy
          array.
        sub.cortex_to_ribbon((lh_data, rh_data)) projects into both hemispheres.
    
        The following options may be given:
          * method (default: 'lines') specifies that a particular method should be used; valid
            options are 'lines' and 'nearest'. The 'lines' method uses the
            sub.lh_vertex_to_voxel_line_interpolation and sub.rh_vertex_to_voxel_line_interpolation
            matrices while 'nearest' uses the nearest-neighbor interpolation. The 'lines' method is
            generally preferred.
          * fill (default: 0) specifies the value to be assigned to all voxels not in the gray mask.
          * dtype (default: None) specifies the data type that should be exported. If None, this
            will be automatically set to np.float32 for floating-point data and np.int32 for integer
            data.
        '''
        # what hemisphere(s)?
        hemi = 'both' if hemi is None else hemi.lower()
        if hemi == 'both': hemi = ('lh', 'rh')
        elif hemi == 'lh': hemi = ('lh', None)
        elif hemi == 'rh': hemi = (None, 'rh')
        else: raise ValueError('unrecognized hemi argument: %s' % hemi)
        # Make the data match this format...
        if pimms.is_map(data):
            if   hemi[0] is None: data = (None, data['rh'])
            elif hemi[1] is None: data = (data['lh'], None)
            else: data = (data['lh'], data['rh'])
        elif pimms.is_matrix(data):
            data = np.asarray(data)
            if data.shape[0] != 2: data = data.T
        else:
            if   hemi[0] is None: data = (data, None)
            elif hemi[1] is None: data = (None, data)
            else: raise ValueError('1 data vector but 2 hemispheres given')
        # Make sure we have all the data...
        for (dat,h,ii) in zip(data, hemi, [0,1]):
            if h is None: continue
            if dat is None: raise ValueError('hemisphere %s requested but data not provided' % h)
            dat = np.asarray(dat)
            hem = getattr(sub, h)
            if not pimms.is_matrix(dat) and not pimms.is_vector(dat):
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
            else:
                frames = data[0].shape[1]
        elif data[0] is None:
            frames = data[1].shape[1] if pimms.is_matrix(data[1]) else 1
        else:
            frames = data[0].shape[1] if pimms.is_matrix(data[0]) else 1
        # what method? specifically, what matrices to use?
        method = 'lines' if method is None else method.lower()
        attr_patt = '%s_vertex_to_voxel_%s_interpolation'
        interp = [None if h is None else getattr(self, attr_patt % (h, method)) for h in hemi]
        # we should also get the indices list
        indices = [None if h is None else getattr(self, '%s_gray_indices' % h) for h in hemi]
        # Figure out the dtype
        if dtype is None:
            # check the input...
            if all(d is None or not np.is_vector(d, 'inexact') for d in data):
                dtype = np.int32
            else: dtype = np.float32
        # make our output array
        dims = self.image_dimensions
        if frames > 1: dims = dims + (frames,)
        arr = np.full(dims, fill, dtype=dtype)
        for (mtx,idcs,dat) in zip(interp, indices, data):
            if dat is None: continue
            arr[idcs] = mtx.dot(dat)
        # That's everything!
        return arr

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
    def __init__(self, chirality, tess, surfaces, registrations, meta_data=None):
        Topology.__init__(self, tess, registrations)
        self.chirality = chirality
        self.surfaces = surfaces
        self.meta_data = meta_data

    @pimms.param
    def chirality(ch):
        '''
        cortex.chirality gives the chirality ('lh' or 'rh') for the given cortex.
        '''
        ch = ch.lower()
        if ch != 'lh' and ch != 'rh':
            raise ValueError('chirality must be \'lh\' or \'rh\'')
        return ch
    @pimms.param
    def surfaces(surfs):
        '''
        cortex.surfaces is a mapping of the surfaces of the given cortex; this must include the
        surfaces 'white' and 'pial'.
        '''
        if pimms.is_lazy_map(surfs) or pimms.is_pmap(surfs):
            return surfs
        elif pimms.is_map(surfs):
            return pyr.pmap(surfs)
        else:
            raise ValueError('surfaces must be a mapping object')
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
        return pimms.imm_array((~z) / (d + z))
    @pimms.value
    def repr(chirality, tess, vertex_count):
        '''
        cortex.repr is equivalent to repr(cortex).
        '''
        arg = (chirality.upper(), tess.face_count, vertex_count)
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
                return self.migray_surface # in case it's been provided via overloading
            return self.surfaces[name]
        elif pimms.is_vector(name, 'real') and len(name) == 1:
            x0 = self.white_surface
            dx = self.white_to_pial_vectors
            return self.make_mesh(x0 + name[0]*dx)
        elif pimms.is_real(name):
            x0 = self.white_surface
            x1 = self.pial_surface
            return self.make_mesh((1 - name)*x0 + name*x1)
        else:
            raise ValueError('could not understand surface layer: %s' % name)


####################################################################################################
# Below are various graphics-related functions; some are wrapped in a try block in case matplotlib
# is not installed.
try:
    import matplotlib, matplotlib.pyplot, matplotlib.tri
    
    _curv_cmap_dict = {
        name: ((0.0, 0.0, 0.5),
               (0.5, 0.5, 0.2),
               (1.0, 0.2, 0.0))
        for name in ['red', 'green', 'blue']}
    _curv_cmap = matplotlib.colors.LinearSegmentedColormap('curv', _curv_cmap_dict)
    _angle_cmap_withneg = matplotlib.colors.LinearSegmentedColormap(
        'polar_angle_full',
        {'red':   ((0.000, 1.0,   1.0),
                   (0.250, 0.5,   0.5),
                   (0.500, 0.0,   0.0),
                   (0.625, 0.0,   0.0),
                   (0.750, 0.0,   0.0),
                   (0.875, 0.833, 0.833),
                   (1.000, 1.0,   1.0)),
         'green': ((0.000, 0.0,   0.0),
                   (0.250, 0.0,   0.0),
                   (0.500, 0.0,   0.0),
                   (0.625, 0.833, 0.833),
                   (0.750, 0.667, 0.667),
                   (0.875, 0.833, 0.833),
                   (1.000, 0.0,   0.0)),
         'blue':  ((0.000, 0.0,   0.0),
                   (0.250, 0.5,   0.5),
                   (0.500, 1.0,   1.0),
                   (0.625, 0.833, 0.833),
                   (0.750, 0.0,   0.0),
                   (0.875, 0.0,   0.0),
                   (1.000, 0.0,   0.0))})
    _angle_cmap = matplotlib.colors.LinearSegmentedColormap(
        'polar_angle',
        {'red':   ((0.000, 1.0,   1.0),
                   (0.250, 0.5,   0.5),
                   (0.500, 0.0,   0.0),
                   (0.625, 0.0,   0.0),
                   (0.750, 0.0,   0.0),
                   (0.875, 0.833, 0.833),
                   (1.000, 1.0,   1.0)),
         'green': ((0.000, 0.0,   0.0),
                   (0.250, 0.0,   0.0),
                   (0.500, 0.0,   0.0),
                   (0.625, 0.833, 0.833),
                   (0.750, 0.667, 0.667),
                   (0.875, 0.833, 0.833),
                   (1.000, 0.0,   0.0)),
         'blue':  ((0.000, 0.0,   0.0),
                   (0.250, 0.5,   0.5),
                   (0.500, 1.0,   1.0),
                   (0.625, 0.833, 0.833),
                   (0.750, 0.0,   0.0),
                   (0.875, 0.0,   0.0),
                   (1.000, 0.0,   0.0))})
    _eccen_cmap = matplotlib.colors.LinearSegmentedColormap(
        'eccentricity',
        {'red':   ((0.0,       0.0, 0.0),
                   (2.5/90.0,  0.5, 0.5),
                   (5.0/90.0,  1.0, 1.0),
                   (10.0/90.0, 1.0, 1.0),
                   (20.0/90.0, 0.0, 0.0),
                   (40.0/90.0, 0.0, 0.0),
                   (90.0/90.0, 1.0, 1.0)),
         'green': ((0.0,       0.0, 0.0),
                   (2.5/90.0,  0.0, 0.0),
                   (5.0/90.0,  0.0, 0.0),
                   (10.0/90.0, 1.0, 1.0),
                   (20.0/90.0, 1.0, 1.0),
                   (40.0/90.0, 1.0, 1.0),
                   (90.0/90.0, 1.0, 1.0)),
         'blue':  ((0.0,       0.0, 0.0),
                   (2.5/90.0,  0.5, 0.5),
                   (5.0/90.0,  0.0, 0.0),
                   (10.0/90.0, 0.0, 0.0),
                   (20.0/90.0, 0.0, 0.0),
                   (40.0/90.0, 1.0, 1.0),
                   (90.0/90.0, 1.0, 1.0))})
    _vertex_angle_empirical_prefixes = ['prf_', 'measured_', 'empiirical_']
    _vertex_angle_model_prefixes = ['model_', 'predicted_', 'inferred_', 'template_', 'atlas_',
                                    'benson14_', 'benson17_']
    _vertex_angle_prefixes = ([''] + _vertex_angle_model_prefixes + _vertex_angle_model_prefixes)
except: pass

def vertex_curvature_color(m):
    return [0.2,0.2,0.2,1.0] if m['curvature'] > -0.025 else [0.7,0.7,0.7,1.0]
def vertex_weight(m):
    return next((m[k]
                 for name in ['weight', 'variance_explained']
                 for pref in ([''] + _vertex_angle_empirical_prefixes)
                 for k in [pref + name]
                 if k in m),
                1.0)
def vertex_angle(m):
    ang0 = next((m[kk] for k in _vertex_angle_prefixes for kk in [k+'polar_angle'] if kk in m),
                None)
    if ang0 is not None: return ang0
    ang0 = next((m[kk]
                 for name in ['angle', 'theta']
                 for k in _vertex_angle_prefixes for kk in [k + name]
                 if kk in m),
                None)
    if ang0 is not None:
        return np.mod(90.0 - 180.0/np.pi*ang0 + 180, 360) - 180
    return None
def vertex_eccen(m):
    ecc0 = next((m[k]
                 for kk in _vertex_angle_prefixes
                 for k in [kk + 'eccentricity']
                 if k in m),
                None)
    if ecc0 is not None: return ecc0
    ecc0 = next((m[k]
                 for kk in _vertex_angle_prefixes
                 for k in [kk + 'rho']
                 if k in m),
                None)
    if ecc0 is not None:
        return 180.0/np.pi*ecc0
    return None
def vertex_angle_color(m, weight_min=0.2, weighted=True, hemi=None, property_name=Ellipsis,
                       null_color='curvature'):
    global _angle_cmap_withneg
    if m is Ellipsis:
        return lambda x: vertex_angle_color(x, weight_min=0.2, weighted=weighted, hemi=hemi,
                                            property_name=property_name, null_color=null_color)
    if isinstance(null_color, basestring):
        null_color = null_color.lower()
        if null_color == 'curvature' or null_color == 'curv':
            nullColor = np.asarray(vertex_curvature_color(m))
        else:
            raise ValueError('bad null color: %s' % null_color)
    else:
        nullColor = np.asarray(null_color)
    if property_name is Ellipsis or property_name is None:
        ang = vertex_angle(m)
    else:
        ang = m[property_name]
    if ang is None: return nullColor
    if isinstance(hemi, basestring):
        hemi = hemi.lower()
        if hemi == 'lh' or hemi == 'left':
            ang = ang
        elif hemi == 'rh' or hemi == 'right':
            ang = -ang
        elif hemi == 'abs':
            ang = np.abs(ang)
        else: raise ValueError('bad hemi argument: %s' % hemi)
    w = vertex_weight(m)
    if weighted and (not pimms.is_number(w) or w < weight_min):
        return nullColor
    angColor = np.asarray(_angle_cmap_withneg((ang + 180.0) / 360.0))
    if weighted:
        return angColor*w + nullColor*(1-w)
    else:
        return angColor
def vertex_eccen_color(m, weight_min=0.2, weighted=True,
                       property_name=Ellipsis, null_color='curvature'):
    global _eccen_cmap
    if m is Ellipsis:
        return lambda x: vertex_eccen_color(x, weight_min=0.2, weighted=weighted,
                                            property_name=property_name, null_color=null_color)
    if isinstance(null_color, basestring):
        null_color = null_color.lower()
        if null_color == 'curvature' or null_color == 'curv':
            nullColor = np.asarray(vertex_curvature_color(m))
        else:
            raise ValueError('bad null color: %s' % null_color)
    else:
        nullColor = np.asarray(null_color)
    if property_name is Ellipsis or property_name is None:
        ecc = vertex_eccen(m)
    else:
        ecc = m[property_name]
    if ecc is None: return nullColor
    w = vertex_weight(m)
    if weighted and (not pimms.is_number(w) or w < weight_min):
        return nullColor
    eccColor = np.asarray(_eccen_cmap((ecc if 0 < ecc < 90 else 0 if ecc < 0 else 90)/90.0))
    if weighted:
        return eccColor*w + nullColor*(1-w)
    else:
        return eccColor
def curvature_colors(m):
    '''
    curvature_colors(m) yields an array of curvature colors for the vertices of the given
      property-bearing object m.
    '''
    return np.asarray(m.map(vertex_curvature_color))
def angle_colors(*args, **kwargs):
    '''
    angle_colors(obj) yields an array of colors for the polar angle map of the given
      property-bearing object (cortex, tesselation, mesh).
    angle_colors(dict) yields an array of the color for the particular vertex property
      mapping that is given as dict.
    angle_colors() yields a functor version of angle_colors that can be called with one of the
      above arguments; note that this is useful precisely because the returned function
      preserves the arguments passed; e.g. angle_colors(weighted=False)(mesh) is equivalent to
      angle_colors(mesh, weighted=False).

    The following options are accepted:
      * weighted (True) specifies whether to use weight as opacity.
      * weight_min (0.2) specifies that below this weight value, the curvature (or null color)
        should be plotted.
      * property_name (Ellipsis) specifies the specific property that should be used as the
        polar angle value; if Ellipsis, will attempt to auto-detect this value.
      * null_color ('curvature') specifies a color that should be used as the background.
    '''
    if len(args) == 0:
        def _angle_color_pass(*args, **new_kwargs):
            return angle_colors(*args, **{k:(new_kwargs[k] if k in new_kwargs else kwargs[k])
                                          for k in set(kwargs.keys() + new_kwargs.keys())})
        return _angle_color_pass
    elif len(args) > 1:
        raise ValueError('angle_colors accepts at most one argument')
    m = args[0]
    if isinstance(m, geo.VertexSet):
        return np.asarray(m.map(vertex_angle_color(**kwargs)))
    else:
        return vertex_angle_color(m, **kwargs)
def eccen_colors(*args, **kwargs):
    '''
    eccen_colors(obj) yields an array of colors for the eccentricity map of the given
      property-bearing object (cortex, tesselation, mesh).
    eccen_colors(dict) yields an array of the color for the particular vertex property mapping
      that is given as dict.
    eccen_colors() yields a functor version of eccen_colors that can be called with one of the
      above arguments; note that this is useful precisely because the returned function
      preserves the arguments passed; e.g. eccen_colors(weighted=False)(mesh) is equivalent to
      eccen_colors(mesh, weighted=False).

    The following options are accepted:
      * weighted (True) specifies whether to use weight as opacity.
      * weight_min (0.2) specifies that below this weight value, the curvature (or null color)
        should be plotted.
      * property_name (Ellipsis) specifies the specific property that should be used as the
        eccentricity value; if Ellipsis, will attempt to auto-detect this value.
      * null_color ('curvature') specifies a color that should be used as the background.
    '''
    if len(args) == 0:
        def _eccen_color_pass(*args, **new_kwargs):
            return eccen_colors(*args, **{k:(new_kwargs[k] if k in new_kwargs else kwargs[k])
                                          for k in set(kwargs.keys() + new_kwargs.keys())})
        return _eccen_color_pass
    elif len(args) > 1:
        raise ValueError('eccen_colors accepts at most one argument')
    m = args[0]
    if isinstance(m, geo.VertexSet):
        return np.asarray(m.map(vertex_eccen_color(**kwargs)))
    else:
        return vertex_eccen_color(m, **kwargs)
def colors_to_cmap(colors):
    colors = np.asarray(colors)
    if colors.shape[1] == 3:
        colors = np.hstack((colors, np.ones((len(colors),1))))
    steps = (0.5 + np.asarray(range(len(colors)-1), dtype=np.float))/(len(colors) - 1)
    return matplotlib.colors.LinearSegmentedColormap(
        'auto_cmap',
        {clrname: ([(0, col[0], col[0])] +
                   [(step, c0, c1) for (step,c0,c1) in zip(steps, col[:-1], col[1:])] +
                   [(1, col[-1], col[-1])])
         for (clridx,clrname) in enumerate(['red', 'green', 'blue', 'alpha'])
         for col in [colors[:,clridx]]},
        N=(len(colors)))

def cortex_plot(the_map, color=None, plotter=matplotlib.pyplot, weights=Ellipsis):
    '''
    cortex_plot(map) yields a plot of the given 2D cortical mesh, map. The following options are
    accepted:
      * color (default: None) specifies a function that, when passed a single argument, a dict
        of the properties of a single vertex, yields an RGBA list for that vertex. By default,
        uses the curvature colors.
      * weight (default: Ellipsis) specifies that the given weights should be used instead of
        the weights attached to the given map; note that Ellipsis indicates that the current
        map's weights should be used. If None or a single number is given, then all weights are
        considered to be 1. A string may be given to indicate that a property should be used.
      * plotter (default: matplotlib.pyplot) specifies a particular plotting object should be
        used. If plotter is None, then instead of attempting to render the plot, a tuple of
        (tri, zs, cmap) is returned; in this case, tri is a matplotlib.tri.Triangulation
        object for the given map and zs and cmap are an array and colormap (respectively) that
        will produce the correct colors. Without plotter equal to None, these would instead
        be rendered as plotter.tripcolor(tri, zs, cmap, shading='gouraud').
    '''
    tri = matplotlib.tri.Triangulation(the_map.coordinates[0],
                                       the_map.coordinates[1],
                                       triangles=the_map.indexed_faces.T)
    if weights is not Ellipsis:
        if weights is None or not hasattr(weights, '__iter__'):
            weights = np.ones(the_map.vertex_count)
        elif isinstance(weights, basestring):
            weights = the_map.prop(weights)
        the_map = the_map.with_prop(weight=weights)
    if isinstance(color, np.ndarray):
        colors = color
    else:
        if color is None or color == 'curv' or color == 'curvature':
            color = vertex_curvature_color
        elif color == 'angle' or color == 'polar_angle':
            color = vertex_angle_color
        elif color == 'eccen' or color == 'eccentricity':
            color = vertex_eccen_color
        colors = np.asarray(the_map.map(color))
    cmap = colors_to_cmap(colors)
    zs = np.asarray(range(the_map.vertex_count), dtype=np.float) / (the_map.vertex_count - 1)
    if plotter is None:
        return (tri, zs, cmap)
    else:
        return plotter.tripcolor(tri, zs, cmap=cmap, shading='gouraud')
