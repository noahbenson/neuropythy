####################################################################################################
# cmag.py
# Cortical magnification caclculation code and utilities.
# by Noah C. Benson

import pimms
import numpy               as np
import numpy.linalg        as npla

from .. import geometry as geo
from .. import mri      as mri

from   neuropythy.util     import (zinv, zdiv, simplex_summation_matrix)
from   .retinotopy         import (extract_retinotopy_argument, retinotopy_data, as_retinotopy)

def disk_projection_cmag(mesh, retinotopy='any', surface=None):
    '''
    disk_projection_cmag(mesh) yields the cortical magnification based on the projection of disks
      on the cortical surface into the visual field.
    '''
    # First, find the retino data
    if pimms.is_str(retinotopy):
        retino = retinotopy_data(mesh, retinotopy)
    else:
        retino = retinotopy
    # Convert from polar angle/eccen to longitude/latitude
    vcoords = np.asarray(as_retinotopy(retino, 'geographical'))
    # note the surface coordinates
    if surface is None:
        tess = mesh.tess
        scoords = mesh.coordinates
    elif pimms.is_str(surface):
        if not isinstance(mesh, mri.Cortex):
            raise ValueError('named surfaces can only be used with cortex objects')
        surface = surface.lower()
        if   surface in ['white']:                         mesh = mesh.white_surface
        elif surface in ['middle', 'midgray', 'mid-gray']: mesh = mesh.midgray_surface
        elif surface in ['pial']:                          mesh = mesh.pial_surface
        elif surface in mesh.surfaces:                     mesh = mesh.surfaces[surface]
        else: raise ValueError('Unrecognized surface: %s' % surface)
        scoords = mesh.coordinates
        tess = mesh.tess
    elif isinstance(surface, geo.Mesh):
        tess = mesh.tess
        scoords = surface.coordinates
        if scoords.shape[1] > tess.vertex_count:
            scoords = scoords[:, surface.tess.index(tess.labels)]
    else: raise ValueError('Could not understand surface option')
    faces = tess.indexed_faces
    # okay, we have the data organized into scoords, vcoords, and faces;
    # let's get sfx and vfx (surface face coords and visual face coords)
    (sfx,vfx) = [np.asarray([x[:,ii] for ii in faces]) for x in (scoords,vcoords)]
    # TODO
    raise ValueError('Not yet implemented')
    

@pimms.immutable
class FieldOfView(object):
    '''
    FieldOfView is a class that represents and calculates the field of view in a cortical area.
    '''
    def __init__(self, angle, eccen, sigma,
                 scale=None, weight=None, search_scale=3.0, bins=6):
        self.polar_angle = angle
        self.eccentricity = eccen
        self.sigma = sigma
        self.weight = weight
        self.bins = bins
        self.search_scale = search_scale
        self.scale = scale
    @pimms.param
    def polar_angle(pa):  return pimms.imm_array(pa)
    @pimms.param
    def eccentricity(ec): return pimms.imm_array(ec)
    @pimms.param
    def sigma(r):         return pimms.imm_array(r)
    @pimms.param
    def scale(s): return s
    @pimms.param
    def weight(w):
        if w is None: return None
        w = np.array(w)
        w /= np.sum(w)
        w.setflags(write=False)
        return w
    @pimms.param
    def bins(h): return h
    @pimms.param
    def search_scale(s): return s
    @pimms.value
    def theta(polar_angle):
        return pimms.imm_array(np.mod(np.pi/180.0*(90 - polar_angle) + np.pi, 2*np.pi) - np.pi)
    @pimms.value
    def coordinates(theta, eccentricity):
        x = eccentricity * np.cos(theta)
        y = eccentricity * np.sin(theta)
        return pimms.imm_array(np.transpose([x,y]))
    @pimms.value
    def _weight(weight, polar_angle):
        if weight is None: weight = np.ones(len(polar_angle))
        weight = weight / np.sum(weight)
        weight.setflags(write=False)
        return weight
    @pimms.value
    def sigma_bin_walls(sigma, bins):
        import scipy, scipy.cluster, scipy.cluster.vq as vq
        std = np.std(sigma)
        if np.isclose(std, 0): return pimms.imm_array([0, np.max(sigma)])
        cl = sorted(std * vq.kmeans(sigma/std, bins)[0])
        cl = np.mean([cl[:-1],cl[1:]], axis=0)
        return pimms.imm_array(np.concatenate(([0], cl, [np.max(sigma)])))
    @pimms.value
    def sigma_bins(sigma, sigma_bin_walls):
        bins = []
        for (mn,mx) in zip(sigma_bin_walls[:-1], sigma_bin_walls[1:]):
            ii = np.logical_and(mn < sigma, sigma <= mx)
            bins.append(pimms.imm_array(np.where(ii)[0]))
        return tuple(bins)
    @pimms.value
    def bin_query_distances(sigma_bins, sigma, search_scale):
        return tuple([np.max(sigma[ii])*search_scale for ii in sigma_bins])
    @pimms.value
    def spatial_hashes(coordinates, sigma_bins):
        import scipy, scipy.spatial
        try:    from scipy.spatial import cKDTree as shash
        except: from scipy.spatial import KDTree  as shash
        return tuple([shash(coordinates[ii]) for ii in sigma_bins])
    # Methods
    def __call__(self, x, y=None):
        if y is not None: x = (x,y)
        x = np.asarray(x)
        if len(x.shape) == 1: return self([x])[0]
        x = np.transpose(x) if x.shape[0] == 2 else x
        crd = self.coordinates
        sig = self.sigma
        wts = self._weight
        res = np.zeros(x.shape[0])
        c1 = 1.0 / np.sqrt(2.0 * np.pi)
        for (sh, qd, bi) in zip(self.spatial_hashes, self.bin_query_distances, self.sigma_bins):
            neis = sh.query_ball_point(x, qd)
            res += [
                np.sum(wts[ii] * c1/s * np.exp(-0.5 * d2/s**2))
                for (ni,pt) in zip(neis,x)
                for ii in [bi[ni]]
                for (s,d2) in [(sig[ii], np.sum((crd[ii] - pt)**2, axis=1))]]
        return res
def field_of_view(mesh, retinotopy='any', mask=None, search_scale=3.0, bins=6):
    '''
    field_of_view(obj) yields a field-of-view function for the given vertex-set object or mapping of
      retinotopy data obj.

    The field-of-view function is a measurement of how much total pRF weight there is at each point
    in the visual field; essentially it is the sum of all visual-field Gaussian pRFs, where the
    Gaussian's are normalized both by the pRF size (i.e., each pRF is a 2D normal distribution whose
    center comes from its polar angle and eccentricity and whose sigma parameter is the pRF radius)
    and the weight (if any weight such as the variance explained is included in the retinotopy data,
    then the normal distributions are multiplied by this weight, otherwise weights are considered to
    be uniform). Note that the weights are normalized by their sum, so the resulting field-of-view
    function should be a valid probability distribution over the visual field.

    The returned object fov = field_of_view(obj) is a special field-of-view object that can be
    called as fov(x, y) or fov(coord) where x and y may be lists or values and coord may be a 2D
    vector or an (n x 2) matrix.

    The following options may be provided (also in argument order):
      * retinotopy (default: 'any') specifies the retinotopy data to be used; if this is a string,
        then runs retinotopy_data(obj, retinotopy) and uses the result, otherwise runs
        retinotopy_data(retinotopy) and uses the result; note that this must have a radius parameter
        as well as polar angle and eccentricity or equivalent parameters that specify the pRF center
      * mask (default: None) specifies the mask that should be used; this is interpreted by a call
        to mesh.mask(mask)
      * search_scale (default: 3.0) specifies how many Gaussian standard deviations away from a pRF
        center should be included when calculating the sum of Gaussians at a point
      * bins (default: 6) specifies the number of bins to divide the pRFs into based on the radius
        value; this is used to prune the search of pRFs that overlap a point to just those
        reasonanly close to the point in question
    '''
    # First, find the retino data
    if pimms.is_str(retinotopy):
        retino = retinotopy_data(mesh, retinotopy)
    else:
        retino = retinotopy_data(retinotopy)
    # Convert from polar angle/eccen to longitude/latitude
    (ang,ecc) = as_retinotopy(retino, 'visual')
    if 'radius' not in retino:
        raise ValueError('Retinotopy data must contain a radius, sigma, or size value')
    sig = retino['radius']
    wgt = next((retino[q] for q in ('variance_explained', 'weight') if q in retino), None)
    # Get the indices we care about
    ii = geo.to_mask(mesh, mask, indices=True)
    return FieldOfView(ang[ii], ecc[ii], sig[ii], weight=wgt[ii],
                       search_scale=search_scale, bins=bins)

@pimms.immutable
class ArealCorticalMagnification(object):
    '''
    ArealCorticalMagnification is a class that represents and calculates coritcal magnification.
    '''
    def __init__(self, angle, eccen, sarea, weight=None, nnearest=200):
        self.polar_angle = angle
        self.eccentricity = eccen
        self.surface_area = sarea
        self.weight = weight
        self.nnearest = nnearest
    @pimms.param
    def polar_angle(pa):  return pimms.imm_array(pa)
    @pimms.param
    def eccentricity(ec): return pimms.imm_array(ec)
    @pimms.param
    def surface_area(sa): return pimms.imm_array(sa)
    @pimms.param
    def weight(w):
        if w is None: return None
        w = np.array(w)
        w /= np.sum(w)
        w.setflags(write=False)
        return w
    @pimms.param
    def nnearest(s): return int(s)
    @pimms.value
    def theta(polar_angle):
        return pimms.imm_array(np.mod(np.pi/180.0*(90 - polar_angle) + np.pi, 2*np.pi) - np.pi)
    @pimms.value
    def coordinates(theta, eccentricity):
        x = eccentricity * np.cos(theta)
        y = eccentricity * np.sin(theta)
        return pimms.imm_array(np.transpose([x,y]))
    @pimms.value
    def _weight(weight, surface_area):
        if weight is None: weight = np.ones(len(surface_area))
        weight = weight * surface_area
        weight /= np.sum(weight)
        weight.setflags(write=False)
        return weight
    @pimms.value
    def spatial_hash(coordinates):
        import scipy, scipy.spatial
        try:    from scipy.spatial import cKDTree as shash
        except: from scipy.spatial import KDTree  as shash
        return shash(coordinates)
    # Static helper function
    @staticmethod
    def _xy_to_matrix(x, y=None):
        if y is not None: x = (x,y)
        x = np.asarray(x)
        return x if len(x.shape) == 1 or x.shape[1] == 2 else x.T
    # Methods
    def nearest(self, x, y=None, n=Ellipsis):
        n = self.nnearest if n is Ellipsis else n
        x = ArealCorticalMagnification._xy_to_matrix(x,y)
        return self.spatial_hash.query(x, n)
    def __call__(self, x, y=None):
        (d,ii) = self.nearest(x, y)
        carea = self.surface_area[ii.flatten()]
        if len(d.shape) == 1:
            varea = np.pi * d[-1]**2
            carea = np.sum(carea)
        else:
            varea = np.pi * d[:,-1]**2
            carea = np.sum(np.split(carea, d.shape[0]), axis=1)
        return carea / varea
    def visual_pooling_area(self, x, y=None):
        (d,ii) = self.nearest(x, y)
        return d[-1] if len(d.shape) == 1 else d[:,-1]    
def areal_cmag(mesh, retinotopy='any', mask=None, surface_area=None, weight=None, nnearest=None):
    '''
    areal_cmag(obj) yields an areal cortical magnification function for the given vertex-set object
      or mapping of retinotopy data obj. If obj is a mapping of retinotopy data, then it must also
      contain the key 'surface_area' specifying the surface area of each vertex in mm^2.

    The areal cmag function is a measurement of how much surface area in a retinotopy dataset is
    found within a circle found in the visual field. The size of the circle is chosen automatically
    by looking up the <n> nearest neighbors of the point in question, summing their surface areas,
    and dividing by the area of the circle in the visual field that is big enough to hold them.

    The returned object cm = areal_cmag(obj) is a special cortical magnification object that can be
    called as cm(x, y) or cm(coord) where x and y may be lists or values and coord may be a 2D
    vector or an (n x 2) matrix. The yielded value(s) will always be in units mm^2 / deg^2.

    The following options may be provided (also in argument order):
      * retinotopy (default: 'any') specifies the retinotopy data to be used; if this is a string,
        then runs retinotopy_data(obj, retinotopy) and uses the result, otherwise runs
        retinotopy_data(retinotopy) and uses the result; note that this must have a radius parameter
        as well as polar angle and eccentricity or equivalent parameters that specify the pRF center
      * mask (default: None) specifies the mask that should be used; this is interpreted by a call
        to mesh.mask(mask)
      * surface_area (default: 'midgray_surface_area') specifies the vertex area property that
        should be used; this is interpreted by a call to mesh.property(surface_area) so may be
        either a property name or a list of values
      * weight (default: None) specifies additional weights that should be used; this is generally
        discouraged
      * nnearest (default: None) specifies the number of nearest neighbors to search for when
        calculating the magnification; if None, then uses int(numpy.ceil(numpy.sqrt(n) * np.log(n)))
        where n is the number of vertices included in the mask
    '''
    # First, find the retino data
    if pimms.is_str(retinotopy):
        retino = retinotopy_data(mesh, retinotopy)
    else:
        retino = retinotopy_data(retinotopy)
    # Convert from polar angle/eccen to longitude/latitude
    (ang,ecc) = as_retinotopy(retino, 'visual')
    # get the surface area
    if surface_area is None: surface_area = 'midgray_surface_area'
    if pimms.is_str(surface_area):
        if surface_area in retino: surface_area = retino[surface_area]
        elif isinstance(mesh, geo.VertexSet): surface_area = mesh.property(surface_area)
        else: surface_area = mesh[surface_area]
    # get the weight
    if weight is not None:
        if pimms.is_str(weight):
            if weight in retino: weight = retino[weight]
            elif isinstance(mesh, geo.VertexSet): weight = mesh.property(weight)
            else: weight = mesh[weight]
    wgt = next((retino[q] for q in ('variance_explained', 'weight') if q in retino), None)
    # Get the indices we care about
    ii = geo.to_mask(mesh, mask, indices=True)
    # get our nnearest
    if nnearest is None: nnearest = int(np.ceil(np.sqrt(len(ii)) * np.log(len(ii))))
    return ArealCorticalMagnification(ang[ii], ecc[ii], surface_area[ii],
                                      weight=(None if wgt is None else wgt[ii]),
                                      nnearest=nnearest)

# Three methods to calculate cortical magnification:
# (1) local projection of the triangle neighborhood then comparison of path across it in the visual
#     field versus on the cortical surface
# (2) examine cortical distance along a visual field path on the cortical surface
# (3) carve out Voronoi polygons in visual space and on the cortical surface; compare areas (can
#     also do this with individual mesh triangles)

def _cmag_coord_idcs(coordinates):
    return [i for (i,(x,y)) in enumerate(zip(*coordinates))
            if (np.issubdtype(type(x), np.float) or np.issubdtype(type(x), np.int))
            if (np.issubdtype(type(y), np.float) or np.issubdtype(type(y), np.int))
            if not np.isnan(x) and not np.isnan(y)]
def _cmag_fill_result(mesh, idcs, vals):
    idcs = {idx:i for (i,idx) in enumerate(idcs)}
    return [vals[idcx[i]] if i in idcs else None for i in mesh.vertex_count]

def cmag(mesh, retinotopy='any', surface=None, to='vertices'):
    '''
    cmag(mesh) yields the neighborhood-based cortical magnification for the given mesh.
    cmag(mesh, retinotopy) uses the given retinotopy argument; this must be interpretable by
      the as_retinotopy function, or should be the name of a source (such as 'empirical' or
      'any').

    The neighborhood-based cortical magnification data is yielded as a map whose keys are 'radial',
    'tangential', 'areal', and 'field_sign'; the units of 'radial' and 'tangential' magnifications 
    are cortical-distance/degree and the units on the 'areal' magnification is
    (cortical-distance/degree)^2; the field sign has no unit.

    Note that if the retinotopy source is not given, this function will by default search for any
    source using the retinotopy_data function.

    The option surface (default None) can be provided to indicate that while the retinotopy and
    results should be formatted for the given mesh (i.e., the result should have a value for each
    vertex in mesh), the surface coordinates used to calculate areas on the cortical surface should
    come from the given surface. The surface option may be a super-mesh of mesh.

    The option to='faces' or to='vertices' (the default) specifies whether the return-values should
    be for the vertices or the faces of the given mesh. Vertex data are calculated from the face
    data by summing and averaging.
    '''
    # First, find the retino data
    if pimms.is_str(retinotopy):
        retino = retinotopy_data(mesh, retinotopy)
    else:
        retino = retinotopy
    # If this is a topology, we want to change to white surface
    if isinstance(mesh, geo.Topology): mesh = mesh.white_surface
    # Convert from polar angle/eccen to longitude/latitude
    vcoords = np.asarray(as_retinotopy(retino, 'geographical'))
    # note the surface coordinates
    if surface is None: scoords = mesh.coordinates
    else:
        scoords = surface.coordinates
        if scoords.shape[1] > mesh.vertex_count:
            scoords = scoords[:, surface.index(mesh.labels)]
    faces = mesh.tess.indexed_faces
    sx = mesh.face_coordinates
    # to understand this calculation, see this stack exchange question:
    # https://math.stackexchange.com/questions/2431913/gradient-of-angle-between-scalar-fields
    # each face has a directional magnification; we need to start with the face side lengths
    (s0,s1,s2) = np.sqrt(np.sum((np.roll(sx, -1, axis=0) - sx)**2, axis=1))
    # we want a couple other handy values:
    (s0_2,s1_2,s2_2) = (s0**2, s1**2, s2**2)
    s0_inv = zinv(s0)
    b = 0.5 * (s0_2 - s1_2 + s2_2) * s0_inv
    h = 0.5 * np.sqrt(2*s0_2*(s1_2 + s2_2) - s0_2**2 - (s1_2 - s2_2)**2) * s0_inv
    h_inv = zinv(h)
    # get the visual coordinates at each face also
    vx = np.asarray([vcoords[:,f] for f in faces])
    # we already have enough data to calculate areal magnification
    s_areas = geo.triangle_area(*sx)
    v_areas = geo.triangle_area(*vx)
    arl_mag = s_areas * zinv(v_areas)
    # calculate the gradient at each triangle; this array is dimension 2 x 2 x m where m is the
    # number of triangles; the first dimension is (vx,vy) and the second dimension is (fx,fy); fx
    # and fy are the coordinates in an arbitrary coordinate system built for each face.
    # So, to reiterate, grad is ((d(vx0)/d(fx0), d(vx0)/d(fx1)) (d(vx1)/d(fx0), d(vx1)/d(fx1)))
    dvx0_dfx = (vx[2] - vx[1]) * s0_inv
    dvx1_dfx = (vx[0] - (vx[1] + b*dvx0_dfx)) * h_inv
    grad = np.asarray([dvx0_dfx, dvx1_dfx])
    # Okay, we want to know the field signs; this is just whether the cross product of the two grad
    # vectors (dvx0/dfx and dvx1/dfx) has a positive z
    fsgn = np.sign(grad[0,0]*grad[1,1] - grad[0,1]*grad[1,0])
    # We can calculate the angle too, which is just the arccos of the normalized dot-product
    grad_norms_2 = np.sum(grad**2, axis=1)
    grad_norms = np.sqrt(grad_norms_2)
    (dvx_norms_inv, dvy_norms_inv) = zinv(grad_norms)
    ngrad = grad * ((dvx_norms_inv, dvx_norms_inv), (dvy_norms_inv, dvy_norms_inv))
    dp = np.clip(np.sum(ngrad[0] * ngrad[1], axis=0), -1, 1)
    fang = fsgn * np.arccos(dp)
    # Great; now we can calculate the drad and dtan; we have dx and dy, so we just need to
    # calculate the jacobian of ((drad/dvx, drad/dvy), (dtan/dvx, dtan/dvy))
    vx_ctr = np.mean(vx, axis=0)
    (x0, y0) = vx_ctr
    den_inv = zinv(np.sqrt(x0**2 + y0**2))
    drad_dvx = np.asarray([x0,  y0]) * den_inv
    dtan_dvx = np.asarray([-y0, x0]) * den_inv
    # get dtan and drad
    drad_dfx = np.asarray([np.sum(drad_dvx[i]*grad[:,i], axis=0) for i in [0,1]])
    dtan_dfx = np.asarray([np.sum(dtan_dvx[i]*grad[:,i], axis=0) for i in [0,1]])
    # we can now turn these into the magnitudes plus the field sign
    rad_mag = zinv(np.sqrt(np.sum(drad_dfx**2, axis=0)))
    tan_mag = zinv(np.sqrt(np.sum(dtan_dfx**2, axis=0)))
    # this is the entire result if we are doing faces only
    if to == 'faces':
        return {'radial': rad_mag, 'tangential': tan_mag, 'areal': arl_mag, 'field_sign': fsgn}
    # okay, we need to do some averaging!
    mtx = simplex_summation_matrix(mesh.tess.indexed_faces)
    cols = np.asarray(mtx.sum(axis=1), dtype=np.float)[:,0]
    cols_inv = zinv(cols)
    # for areal magnification, we want to do summation over the s and v areas then divide
    s_areas = mtx.dot(s_areas)
    v_areas = mtx.dot(v_areas)
    arl_mag = s_areas * zinv(v_areas)
    # for the others, we just average
    (rad_mag, tan_mag, fsgn) = [cols_inv * mtx.dot(x) for x in (rad_mag, tan_mag, fsgn)]
    return {'radial': rad_mag, 'tangential': tan_mag, 'areal': arl_mag, 'field_sign': fsgn}

def neighborhood_cortical_magnification(mesh, coordinates):
    '''
    neighborhood_cortical_magnification(mesh, visual_coordinates) yields a list of neighborhood-
    based cortical magnification values for the vertices in the given mesh if their visual field
    coordinates are given by the visual_coordinates matrix (must be like [x_values, y_values]). If
    either x-value or y-value of a coordinate is either None or numpy.nan, then that cortical
    magnification value is numpy.nan.
    '''
    idcs = _cmag_coord_idcs(coordinates)
    neis = mesh.tess.indexed_neighborhoods
    coords_vis = np.asarray(coordinates if len(coordinates) == 2 else coordinates.T)
    coords_srf = mesh.coordinates
    res = np.full((mesh.vertex_count, 3), np.nan, dtype=np.float)
    res = np.array([row for row in [(np.nan,np.nan,np.nan)] for _ in range(mesh.vertex_count)],
                   dtype=np.float)
    for idx in idcs:
        nei = neis[idx]
        pts_vis = coords_vis[:,nei]
        pts_srf = coords_srf[:,nei]
        x0_vis = coords_vis[:,idx]
        x0_srf = coords_srf[:,idx]
        if any(u is None or np.isnan(u) for pt in pts_vis for u in pt): continue
        # find tangential, radial, and areal magnifications
        x0col_vis = np.asarray([x0_vis]).T
        x0col_srf = np.asarray([x0_srf]).T
        # areal is easy
        voronoi_vis = (pts_vis - x0col_vis) * 0.5 + x0col_vis
        voronoi_srf = (pts_srf - x0col_srf) * 0.5 + x0col_srf
        area_vis = np.sum([geo.triangle_area(x0_vis, a, b)
                           for (a,b) in zip(voronoi_vis.T, np.roll(voronoi_vis, 1, axis=1).T)])
        area_srf = np.sum([geo.triangle_area(x0_srf, a, b)
                           for (a,b) in zip(voronoi_srf.T, np.roll(voronoi_srf, 1, axis=1).T)])
        res[idx,2] = np.inf if np.isclose(area_vis, 0) else area_srf/area_vis
        # radial and tangentual we do together because they are very similar:
        # find the intersection lines then add up their distances along the cortex
        pts_vis = voronoi_vis
        pts_srf = voronoi_srf
        segs_srf = (pts_srf, np.roll(pts_srf, -1, axis=1))
        segs_vis = (pts_vis, np.roll(pts_vis, -1, axis=1))
        segs_vis_t = np.transpose(segs_vis, (2,0,1))
        segs_srf_t = np.transpose(segs_srf, (2,0,1))
        x0norm_vis = npla.norm(x0_vis)
        if not np.isclose(x0norm_vis, 0):
            dirvecs = x0_vis / x0norm_vis
            dirvecs = np.asarray([dirvecs, [-dirvecs[1], dirvecs[0]]])
            for dirno in [0,1]:
                dirvec = dirvecs[dirno]
                line = (x0_vis, x0_vis + dirvec)
                try:
                    isects_vis = np.asarray(geo.line_segment_intersection_2D(line, segs_vis))
                    # okay, these will all be nan but two of them; they are the points we care about
                    isect_idcs = np.unique(np.where(np.logical_not(np.isnan(isects_vis)))[1])
                except:
                    isect_idcs = []
                if len(isect_idcs) != 2:
                    res[idx,dirno] = np.nan
                    continue
                isects_vis = isects_vis[:,isect_idcs].T
                # we need the distance in visual space
                len_vis = npla.norm(isects_vis[0] - isects_vis[1])
                if np.isclose(len_vis, 0): res[idx,dirno] = np.inf
                else:
                    # we also need the distances on the surface: find the points by projection
                    fsegs_srf = segs_srf_t[isect_idcs]
                    fsegs_vis = segs_vis_t[isect_idcs]
                    s02lens_vis = npla.norm(fsegs_vis[:,0] - fsegs_vis[:,1], axis=1)
                    s01lens_vis = npla.norm(fsegs_vis[:,0] - isects_vis, axis=1)
                    vecs_srf = fsegs_srf[:,1] - fsegs_srf[:,0]
                    s02lens_srf = npla.norm(vecs_srf, axis=1)
                    isects_srf = np.transpose([(s01lens_vis/s02lens_vis)]) * vecs_srf \
                                 + fsegs_srf[:,0]
                    len_srf = np.sum(npla.norm(isects_srf - x0_srf, axis=1))
                    res[idx,dirno] = len_srf / len_vis
    # That's it!
    return res

def path_cortical_magnification(mesh, path, mask=None, return_all=False,
                                polar_angle='polar_angle', eccentricity='eccentricity'):
    '''
    path_cortical_magnification(mesh, path) yields the length of the given path along the cortical
      surface divided by the length of the given path in visual space. The path must be an n x 2
      matrix of (x,y) valuesin the visual field.

    The following options are accepted:
      * mask (default: None) may be a boolean mask of vertices to include in the calculation.
    '''
    ang = extract_retinotopy_argument(mesh, 'polar_angle', polar_angle, default='predicted')
    ecc = extract_retinotopy_argument(mesh, 'eccentricity', eccentricity, default='predicted')
    srf = mesh.coordinates.T
    ids = np.asarray(range(len(srf)))
    atol = 1e-6
    if mask is not None:
        msk = np.where(mask)[0]
        ids = ids[msk]
        ang = ang[msk]
        ecc = ecc[msk]
        srf = srf[msk]
    # edit out values we can't use
    okays = [k for (k,i,a,e) in zip(range(len(ids)), ids, ang, ecc)
             if np.issubdtype(type(a), np.number)
             if np.issubdtype(type(e), np.number)]
    if len(okays) != len(ids):
        ids = ids[okays]
        ang = ang[okays]
        ecc = ecc[okays]
        srf = srf[okays]
    # okay; now we have the subset we can use; lets get the appropriate triangles...
    okays = set(ids)
    tris = np.asarray([f for f in mesh.tess.indexed_faces.T if all(a in okays for a in f)]).T
    # in case anything wasn't connected by triangle:
    okays = set(np.unique(tris))
    idcs = [k for (k,i) in enumerate(ids) if i in okays]
    ids = ids[idcs]
    ang = ang[idcs]
    ecc = ecc[idcs]
    srf = srf[idcs]
    okays = {i:k for (k,i) in enumerate(ids)}
    # now we can recreate the triangles with proper id's
    tris = np.asarray([[okays[a] for a in f] for f in tris.T]).T
    # get the x/y coordinates in visual space
    vis_coords = ecc * np.asarray([np.cos(np.pi/180*(90-ang)), np.sin(np.pi/180*(90-ang))])
    # okay, setup the topology/registrations
    tess = geo.Tesselation(tris)
    srf_reg = tess.make_mesh(srf.T)
    vis_reg = tess.make_mesh(vis_coords)
    # Now the Great Work begins...
    # Find the triangle id's of the containers of the points first
    tids = vis_reg.container(path.T)
    # now step along the triangles starting at the first point...
    steps = []
    ss = []
    pth = path.T
    # to do this we want to have an index of edges to triangles that neighbor them
    edge_idx = {}
    for (tid,(a,b,c)) in enumerate(vis_reg.tess.faces.T):
        for edge in map(tuple, map(sorted, [(a,b), (c,a), (b,c)])):
            if edge not in edge_idx: edge_idx[edge] = set([])
            edge_idx[edge].add(tid)
    # we may need this to find arbitrary intersections:
    all_edges = edge_idx.keys()
    all_segs = np.asarray([vis_reg.coordinates[:,us] for us in np.transpose(all_edges)])
    for (tid,next_tid,pt,next_pt) in zip(tids, np.roll(tids,-1), pth, np.roll(pth,-1,axis=0)):
        # This could be the last point or there could be a break;
        # We handle breaks as separate paths
        if tid is None or (next_pt == pth[0]).all():
            if tid is not None: ss.append(pt)
            if len(ss) > 1: steps.append(ss)
            ss = []
            continue
        # goal is to travel through triangles on our way to next_pt
        pt0 = pt
        ss.append(pt)
        # here is the line segment we want to intersect with things
        seg = (pt, next_pt)
        while not geo.point_in_triangle(vis_reg.coordinates[:,vis_reg.tess.faces[:,tid]].T,next_pt):
            # otherwise, we need to find the next neighboring triangle:
            vtcs   = vis_reg.tess.faces[:,tid]
            tpts   = vis_reg.coordinates[:,vtcs]
            tsegs  = (tpts, np.roll(tpts, -1, axis=1))
            tedges = [tuple(sorted([u,v])) for (u,v) in zip(vtcs, np.roll(vtcs, -1))]
            tid0 = tid
            tid0_set = set([tid0])
            projdir = next_pt - pt
            isect = np.asarray(geo.segment_intersection_2D((pt, next_pt), tsegs))
            # there should be 1 or 2 intersections; 1 if this is the first/last step,
            # 2 if there is a triangle crossed by the step
            isect_idcs = np.where(np.isfinite(isect[0]))[0]
            isect = isect.T
            # take only the point closest to the destination that is not the source
            isect_idcs = sorted([(d,i) for (i,x) in enumerate(isect) for d in [np.dot(x-pt,projdir)]
                                 if d > 0 and not np.isclose(d, 0, atol=atol)],
                                key=lambda i: i[0])
            if len(isect_idcs) > 0:
                isect_idcs = isect_idcs[0][1]
                pt = isect[isect_idcs]
                isect_edge = tuple(sorted(vtcs[[isect_idcs, (isect_idcs + 1) % 3]]))
                # we have the old point and the new point
                tid = next((t for t in edge_idx[isect_edge] if t != tid), None)
                if tid is not None: ss.append(pt)
                # okay, it's possible that we've found a null triangle; in this case, we
                # want to continue at the next closest intersection point on the path to
                # the next_pt; to do this, we have to look for intersections among allll
                # of the segments in the mesh
            else:
                tid = None
            if tid is None:
                if len(ss) > 1: steps.append(ss)
                ss = []
                isect = np.asarray(geo.segment_intersection_2D((pt0, next_pt), all_segs)).T
                # we want to ignore intersections in the current triangle tid0
                isect_idcs = [idc for idc in np.where(np.logical_not(np.isnan(isect[:,0])))[0]
                              if edge_idx[all_edges[idc]] != tid0_set]
                if len(isect_idcs) == 0:
                    # nothing left but the last point
                    break
                isect = isect[isect_idcs]
                pts_by_nearness = sorted(
                    [(i,isct,d) for (i,isct) in enumerate(isect) for d in [np.dot(isct-pt, projdir)]
                     if d > 0 and not np.isclose(d, 0)],
                    key=lambda a: a[2])
                if len(pts_by_nearness) > 0:
                    pt_id = pts_by_nearness[0][0]
                    pt = isect[pt_id]
                    tid = list(edge_idx[tuple(sorted(all_edges[isect_idcs[pt_id]]))] - tid0_set)[0]
                if pt is None or tid is None or tid == tid0:
                    # This happens if the triangles are basically adjacent but not technically
                    # connected; 
                    pt = pt0
                    tid = next_tid
            # if tid is still None, something's basically gone wrong with the mesh here
            if tid is None:
                if len(ss) > 1: steps.append(ss)
                ss = []
                break
        # We exited out of the loop, so we've reached the triangle containing next_pt;
        # we don't need to add it to 
    # Okay, at this point we have a set of paths
    if len(steps) == 0: raise ValueError('No triangles found intersecting path!')
    # convert the visual path to the surface path
    vis_steps = []
    srf_steps = []
    for uss in steps:
        ss_srf = []
        ss_vis = []
        for x_vis in uss:
            try:
                addr  = vis_reg.address(x_vis)
                x_srf = srf_reg.unaddress(addr)
            except:
                if len(ss_vis) > 1:
                    vis_steps.append(ss_vis)
                    srf_steps.append(ss_srf)
                ss_vis = []
                ss_srf = []
            else:
                ss_vis.append(x_vis)
                ss_srf.append(x_srf)
        if len(ss_vis) > 1:
            vis_steps.append(ss_vis)
            srf_steps.append(ss_srf)
            ss_vis = []
            ss_srf = []

    # for each path calculate its length in both spaces
    srf_ds = [[npla.norm(x0 - x1) for (x0,x1) in zip(srf_path[:-1], srf_path[1:])]
              for srf_path in srf_steps]
    vis_ds = [[npla.norm(x0 - x1) for (x0,x1) in zip(vis_path[:-1], vis_path[1:])]
              for vis_path in vis_steps]
    srf_d = np.sum(np.hstack(srf_ds))
    vis_d = np.sum(np.hstack(vis_ds))
    #return (srf_ds, vis_ds)
    if return_all:
        return (srf_steps, vis_steps)
    else:
        return np.inf if np.isclose(vis_d, 0) else srf_d/vis_d
    

def isoangular_path(mesh, pathtype, val, mask=None, min_segment_length=4,
                    polar_angle='polar_angle', eccentricity='eccentricity'):
    '''
    isoangular_path(mesh, pathtype, val) yields a list of isoangular paths, each of whichs is given
      as a tuple (spath, vpath) of the points along the cortical surface (spath, n x 3) and the
      points in the visual field (vpath, n x 2).  The path must be specified as either 'angle' or
      'eccen' followed by a polar angle or eccentricity value.

    The following options are accepted:
      * mask (default: None) may be a boolean mask of vertices to include in the calculation.
      * min_segment_length (default: 4) the minimum number of faces that need to be included in a
        path segment in order to be included in the result.
    '''
    ang = extract_retinotopy_argument(mesh, 'polar_angle', polar_angle, default='predicted')
    ecc = extract_retinotopy_argument(mesh, 'eccentricity', eccentricity, default='predicted')
    srf = mesh.coordinates.T
    ids = np.asarray(range(len(srf)))
    if mask is not None:
        msk = np.where(mask)[0]
        ids = ids[msk]
        ang = ang[msk]
        ecc = ecc[msk]
        srf = srf[msk]
    # edit out values we can't use
    okays = [k for (k,i,a,e) in zip(range(len(ids)), ids, ang, ecc)
             if np.issubdtype(type(a), np.number)
             if np.issubdtype(type(e), np.number)]
    if len(okays) != len(ids):
        ids = ids[okays]
        ang = ang[okays]
        ecc = ecc[okays]
        srf = srf[okays]
    # okay; now we have the subset we can use; lets get the appropriate triangles...
    okays = set(ids)
    tris = np.asarray([f for f in mesh.tess.indexed_faces.T if all(a in okays for a in f)]).T
    # in case anything wasn't connected by triangle:
    okays = set(np.unique(tris))
    idcs = [k for (k,i) in enumerate(ids) if i in okays]
    ids = ids[idcs]
    ang = ang[idcs]
    ecc = ecc[idcs]
    srf = srf[idcs]
    okays = {i:k for (k,i) in enumerate(ids)}
    # now we can recreate the triangles with proper id's
    tris = np.asarray([[okays[a] for a in f] for f in tris.T]).T
    # get the x/y coordinates in visual space
    vis_coords = ecc * np.asarray([np.cos(np.pi/180*(90-ang)), np.sin(np.pi/180*(90-ang))])
    vis_coords = vis_coords.T
    # okay, setup the topology/registrations
    tess = geo.Tesselation(tris)
    srf_reg = tess.make_mesh(srf.T)
    vis_reg = tess.make_mesh(vis_coords)
    # now the Great Work begins...
    # Find all triangles that intersect this particular angle line
    pathtype = pathtype.lower()
    trisect = None
    if pathtype in ['angle', 'polar_angle', 'radial', 'rad']:
        vals = ang
    elif pathtype in ['eccen', 'eccentricity', 'tangential', 'tan']:
        vals = ecc
    else:
        raise ValueError('Unrecognized pathtype: %s' % pathtype)
    angsides = (np.sign(vals - val) + 1).astype(np.bool)
    angsides = angsides.astype(np.int)
    tris = tris.T
    trisides = np.sum([angsides[tt] for tt in tris.T], axis=0)
    trii = np.intersect1d(np.where(trisides > 0)[0], np.where(trisides < 3)[0])
    # Make an adjacency list of these intersecting triangles
    tadj = {}
    tmp = {}
    for (u,v,i) in zip(np.hstack((tris[trii,0], tris[trii,0], tris[trii,1])),
                       np.hstack((tris[trii,1], tris[trii,2], tris[trii,2])),
                       np.hstack((trii, trii, trii))):
        # if (u/v) doesn't cross the iso-line, ignore it
        if angsides[u] == angsides[v] or vals[u] == val or vals[v] == val:
            continue
        elif (v,u) in tmp:
            k = tmp[(v,u)]
            del tmp[(v,u)]
            tadj[(i,k)] = (u,v)
            tadj[(k,i)] = (u,v)
        else:
            tmp[(u,v)] = i
    # Okay, now we stitch these together into segments; this is basically a union-find problem
    seg = {ti:[ti] for ti in trii} # cluster end triangle
    cls = {ti:ti   for ti in trii}
    def _find(k):
        kc = cls[k]
        if k == kc or cls[kc] == kc: return kc
        kcc = _find(kc)
        cls[k] = kcc
        return kcc
    def _union(u, v):
        uc = _find(u)
        vc = _find(v)
        if uc == vc: return
        us = seg[uc]
        vs = seg[vc]
        (u0,ue) = (us[0],us[-1])
        (v0,ve) = (vs[0],vs[-1])
        # either u or v must be adjacent to the end of the other
        if   u0 == u and v0 == v: (a,b) = (reversed(us), vs)
        elif u0 == u and ve == v: (a,b) = (reversed(us), reversed(vs))
        elif ue == u and v0 == v: (a,b) = (us,           vs)
        elif ue == u and ve == v: (a,b) = (us,           reversed(vs))
        else: return # nothing joined; can't connect at intersection
        a = list(a)
        b = list(b)
        for bb in b: a.append(bb)
        seg[uc] = a
        seg[vc] = a
        cls[vc] = uc
        cls[u] = uc
        cls[v] = uc
    for (ti, tj) in tadj.iterkeys(): _union(ti, tj)
    for ti       in trii: _find(ti)
    # Okay, we should have the segments now...
    clarr = np.asarray([cls[ti] for ti in trii])
    seg_ids = np.unique(clarr)
    seg_vis = []
    seg_srf = []
    if min_segment_length < 2: min_segment_length = 2
    srf_coords = srf_reg.coordinates
    for sid in seg_ids:
        s = seg[sid]
        if len(s) < min_segment_length: continue
        # Okay, we want to walk through the triangles in order; make a list of the ordered edges
        # handle the first triangle/segment start point
        (u,v) = tadj[(s[0], s[1])]
        w = np.setdiff1d(tris[s[0]], (u,v))[0]
        if vals[w] == val:
            pts_vis = [vis_coords[w]]
            pts_srf = [srf_coords[w]]
        else:
            if angsides[w] != angsides[v]: u = v
            w_frac = (vals[w] - val) / (vals[w] - vals[u])
            u_frac = 1.0 - w_frac
            pts_vis = [w_frac*vis_coords[w] + u_frac*vis_coords[u]]
            pts_srf = [w_frac*srf_coords[w] + u_frac*srf_coords[u]]
        # Okay, walk along the adjacent edges
        for (s0,s1) in zip(s[:-1], s[1:]):
            (u,v) = tadj[(s0,s1)]
            v_frac = (vals[v] - val) / (vals[v] - vals[u])
            u_frac = 1.0 - v_frac
            pts_vis.append(v_frac*vis_coords[v] + u_frac*vis_coords[u])
            pts_srf.append(v_frac*srf_coords[v] + u_frac*srf_coords[u])
        # And finally handle the end-point
        (u,v) = tadj[(s[-2], s[-1])]
        w = np.setdiff1d(tris[s[-1]], (u,v))[0]
        if vals[w] == val:
            pts_vis.append(vis_coords[w])
            pts_srf.append(srf_coords[w])
        else:
            if angsides[w] != angsides[v]: u = v
            w_frac = (vals[w] - val) / (vals[w] - vals[u])
            u_frac = 1.0 - w_frac
            pts_vis.append(w_frac*vis_coords[w] + u_frac*vis_coords[u])
            pts_srf.append(w_frac*srf_coords[w] + u_frac*srf_coords[u])
        # Just append these points
        seg_vis.append(pts_vis)
        seg_srf.append(pts_srf)
    # That's it!
    return (seg_srf, seg_vis)
