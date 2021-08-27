####################################################################################################
# cmag.py
# Cortical magnification caclculation code and utilities.
# by Noah C. Benson

import os, sys, six, pimms
import numpy            as np
import scipy            as sp
import scipy.sparse     as sps
import numpy.linalg     as npla
import pyrsistent       as pyr

from .. import geometry as geo
from .. import mri      as mri

from ..util          import (zinv, simplex_summation_matrix, curry, to_hemi_str, flattest, zdivide)
from .retinotopy     import (extract_retinotopy_argument, retinotopy_data, as_retinotopy,
                             retinotopic_field_sign)
from ..geometry.util import (line_segment_intersection_2D, cartesian_to_barycentric_2D)

def mag_data(hemi, retinotopy='any', surface='midgray', mask=None,
             weights=Ellipsis, weight_min=0, weight_transform=Ellipsis,
             visual_area=None, visual_area_mask=Ellipsis,
             eccentricity_range=None, polar_angle_range=None):
    '''
    mag_data(hemi) yields a map of visual/cortical magnification data for the given hemisphere.
    mag_data(mesh) uses the given mesh.
    mag_data([arg1, arg2...]) maps over the given hemisphere or mesh arguments.
    mag_data(subject) is equivalent to mag_data([subject.lh, subject.rh]).
    mag_data(mdata) for a valid magnification data map mdata (i.e., is_mag_data(mdata) is True or
      mdata is a lazy map with integer keys) always yields mdata without considering any additional
      arguments.

    The data structure returned by magdata is a lazy map containing the keys:
      * 'surface_coordinates': a (2 x N) or (3 x N) matrix of the mesh coordinates in the mask
        (usually in mm).
      * 'visual_coordinates': a (2 x N) matrix of the (x,y) visual field coordinates (in degrees).
      * 'surface_areas': a length N vector of the surface areas of the faces in the mesh.
      * 'visual_areas': a length N vector of the areas of the faces in the visual field.
      * 'mesh': the full mesh from which the surface coordinates are obtained.
      * 'submesh': the submesh of mesh of just the vertices in the mask (may be identical to mesh).
      * 'mask': the mask used.
      * 'retinotopy_data': the full set of retinotopy_data from the hemi/mesh; note that this will
        include the key 'weights' of the weights actually used and 'visual_area' of the found or
        specified visual area.
      * 'masked_data': the subsampled retinotopy data from the hemi/mesh.
    Note that if a visual_area property is found or provided (see options below), instead of
    yielding a map of the above, a lazy map whose keys are the visual areas and whose values are the
    maps described above is yielded instead.

    The following named options are accepted (in order):
      * retinotopy ('any') specifies the value passed to the retinotopy_data function to obtain the
        retinotopic mapping data; this may be a map of such data.
      * surface ('midgray') specifies the surface to use.
      * mask (None) specifies the mask to use.
      * weights, weight_min, weight_transform (Ellipsis, 0, Ellipsis) are used as in the
        to_property() function  in neuropythy.geometry except weights, which, if equal to Ellipsis,
        attempts to use the weights found by retinotopy_data() if any.
      * visual_area (Ellipsis) specifies the property to use for the visual area label; Ellipsis is
        equivalent to whatever visual area label is found by the retinotopy_data() function if any.
      * visual_area_mask (Ellipsis) specifies which visual areas to include in the returned maps,
        assuming a visual_area property is found; Ellipsis is equivalent to everything but 0; None
        is equivalent to everything.
      * eccentricity_range (None) specifies the eccentricity range to include.
      * polar_angle_range (None) specifies the polar_angle_range to include.
    '''
    if is_mag_data(hemi): return hemi
    elif pimms.is_lazy_map(hemi) and pimms.is_vector(hemi.keys(), 'int'): return hemi
    if mri.is_subject(hemi): hemi = (hemi.lh. hemi.rh)
    if pimms.is_vector(hemi):
        return tuple([mag_data(h, retinotopy=retinotopy, surface=surface, mask=mask,
                               weights=weights, weight_min=weight_min,
                               weight_transform=weight_transform, visual_area=visual_area,
                               visual_area_mask=visual_area_mask,
                               eccentricity_range=eccentricity_range,
                               polar_angle_range=polar_angle_range)
                      for h in hemi])
    # get the mesh
    mesh = geo.to_mesh((hemi, surface))
    # First, find the retino data
    retino = retinotopy_data(hemi, retinotopy)
    retino = dict(retino)
    # we can process the rest the mask now, including weights and ranges
    if weights is Ellipsis: weights = retino.get('variance_explained', None)
    mask = hemi.indices if mask is None else hemi.mask(mask, indices=True)
    (arng,erng) = (polar_angle_range, eccentricity_range)
    (ang,ecc) = (retino['polar_angle'], retino['eccentricity'])
    if pimms.is_str(arng):
        tmp = to_hemi_str(arng)
        arng = (-180,0) if tmp == 'rh' else (0,180) if tmp == 'lh' else (-180,180)
    elif arng is None:
        tmp = ang[mask]
        tmp = tmp[np.isfinite(tmp)]
        arng = (np.min(tmp), np.max(tmp))
    if erng is None:
        tmp = ecc[mask]
        tmp = tmp[np.isfinite(tmp)]
        erng = (0, np.max(tmp))
    elif pimms.is_scalar(erng): erng = (0, erng)
    (ang,wgt) = hemi.property(retino['polar_angle'], weights=weights, weight_min=weight_min,
                              weight_transform=weight_transform, yield_weight=True)
    ecc = hemi.property(retino['eccentricity'], weights=weights, weight_min=weight_min,
                        weight_transform=weight_transform, data_range=erng)
    # apply angle range if given
    ((mn,mx),mid) = (arng, np.mean(arng))
    oks = mask[np.isfinite(ang[mask])]
    u = ang[oks]
    u = np.mod(u + 180 - mid, 360) - 180 + mid
    ang[oks[np.where((mn <= u) & (u < mx))[0]]] = np.inf
    # mark/unify the out-of-range ones
    bad = np.where(np.isinf(ang) | np.isinf(ecc))[0]
    ang[bad] = np.inf
    ecc[bad] = np.inf
    wgt[bad] = 0
    wgt *= zinv(np.sum(wgt[mask]))
    # get visual and surface coords
    vcoords = np.asarray(as_retinotopy(retino, 'geographical'))
    scoords = mesh.coordinates
    # now figure out the visual area so we can call down if we need to
    if visual_area is Ellipsis: visual_area = retino.get('visual_area', None)
    if visual_area is not None: retino['visual_area'] = visual_area
    if wgt is not None: retino['weights'] = wgt
    rdata = pimms.persist(retino)
    # calculate the range area
    (tmn,tmx) = [np.pi/180.0 * u for u in arng]
    if tmx - tmn >= 2*np.pi: (tmn,tmx) = (-np.pi,np.pi)
    (emn,emx) = erng
    rarea = 0.5 * (emx*emx - emn*emn) * (tmx - tmn)
    # okay, we have the data organized; we can do the calculation based on this, but we may have a
    # visual area mask to apply as well; here's how we do it regardless of mask
    def finish_mag_data(mask):
        if len(mask) == 0: return None
        # now that we have the mask, we can subsample
        if np.array_equal(mask, mesh.indices): submesh = mesh
        else:                                  submesh = mesh.submesh(mesh.labels[mask])
        mask = mesh.tess.index(submesh.labels)
        mdata = pyr.pmap({k:(v[mask]   if pimms.is_vector(v) else
                             v[:,mask] if pimms.is_matrix(v) else
                             None)
                          for (k,v) in six.iteritems(rdata)})
        fs = submesh.tess.indexed_faces
        (vx, sx)  = [x[:,mask]                        for x in (vcoords, scoords)]
        (vfx,sfx) = [np.asarray([x[:,f] for f in fs]) for x in (vx,      sx)]
        (va, sa)  = [geo.triangle_area(*x)            for x in (vfx, sfx)]
        return pyr.m(surface_coordinates=sx, visual_coordinates=vx,
                     surface_areas=sa,       visual_areas=va,
                     mesh=mesh,              submesh=submesh,
                     retinotopy_data=rdata,  masked_data=mdata,
                     mask=mask,              area_of_range=rarea)
    # if there's no visal area, we just use the mask as is
    if visual_area is None: return finish_mag_data(mask)
    # otherwise, we return a lazy map of the visual area mask values
    visual_area = hemi.property(visual_area, mask=mask, null=0, dtype=np.int)
    vam = (np.unique(visual_area)                    if visual_area_mask is None     else
           np.setdiff1d(np.unique(visual_area), [0]) if visual_area_mask is Ellipsis else
           np.unique(list(visual_area_mask)))
    return pimms.lazy_map({va: curry(finish_mag_data, mask[visual_area[mask] == va])
                           for va in vam})
def is_mag_data(mdat):
    '''
    is_mag_data(dat) yields True if the given data is a valid set of magnification data and False
      otherwise.

    Note that this does not return True for all valid return values of the mag_data() function:
    specifically, if the mag_data() function yields a list of mag-data maps or a lazy-map of the
    mag-data maps split out by visual area, then this will return False. This function only returns
    True for a map of mag data itself.
    '''
    if not pimms.is_map(mdat): return False
    for k in ['surface_coordinates', 'visual_coordinates', 'mesh', 'submesh', 'mask',
              'retinotopy_data', 'masked_data', 'surface_areas', 'visual_areas']:
        if k not in mdat: return False
    return True

def parse_toopt_facevertex(to):
    '''
    Parses the `to` optional-argument of the functions below; returns either 'vertex' or 'face',
    or raises an error.
    '''
    if to in [None,Ellipsis]: return 'vertex'
    if not pimms.is_str(to): raise ValueError('could not parse `to` argument: %s' % (to,))
    to = to.lower()
    if to in ['vertices','vertex','nodes','node','points','v','vtx','vtcs','pts']: return 'vertex'
    elif to in ['faces','f','triangles','t','tri','tris','face','triangle']: return 'face'
    else: raise ValueError('could not parse `to` argument: %s' % (to,))
    
def rtmag_potential(submesh, X0, mask=Ellipsis, fieldsign=None):
    '''
    rtmag_potential(mesh, viscoords, ...) yields the
      radial/tangential cortical magnification term of the potential field.

    This should generally not be called directly and instead should be obtained from the
    clean_retinotopy_potential() function instead.
    '''
    import neuropythy.optimize as op
    if fieldsign == 0: fieldsign = None
    # A few other handy pieces of data we can extract:
    sxyz = submesh.coordinates
    n = submesh.vertex_count
    (u,v) = submesh.tess.indexed_edges
    selen = submesh.edge_lengths
    sarea = submesh.face_areas
    m = submesh.tess.edge_count
    fs = submesh.tess.indexed_faces
    neis = submesh.tess.indexed_neighborhoods
    fangs = submesh.face_angles
    # we're adding r and t (radial and tangential visual magnification) pseudo-parameters to
    # each vertex; r and t are derived from the position of other vertices; our first step is
    # to derive these values; for this we start with the parameters themselves:
    (x,y) = [op.identity[np.arange(k, 2*n, 2)] for k in (0,1)]
    # okay, we need to setup a bunch of least-squares solutions, one for each vertex:
    nneis = np.asarray([len(nn) for nn in neis])
    maxneis = np.max(nneis)
    thts = op.atan2(y, x)
    eccs = op.compose(op.piecewise(op.identity, ((-1e-9, 1e-9), 1)),
                      op.sqrt(x**2 + y**2))
    coss = x/eccs
    sins = y/eccs
    # organize neighbors:
    # neis becomes a list of rows of 1st neighbor, second neighbor etc. with -1 indicating none
    neis = np.transpose([nei + (-1,)*(maxneis - len(nei)) for nei in neis])
    qnei = (neis > -1) # mark where there are actually neighbors
    neis[~qnei] = 0 # we want the -1s (now 0s) to behave okay when passed to a potential index
    # okay, walk through the neighbors setting up the least squares
    (r, t) = (None, None)
    for (k,q,nei) in zip(range(len(neis)), qnei.astype('float'), neis):
        xx = x[nei] - x
        yy = y[nei] - y
        sd = np.sum((sxyz[:,nei].T - sxyz[:,k])**2, axis=1)
        (xx, yy) = (xx*coss + yy*sins, yy*coss - xx*sins)
        xterm = (op.abs(xx) * q)
        yterm = (op.abs(yy) * q)
        r = xterm if r is None else (r + xterm)
        t = yterm if t is None else (t + yterm)
    (r, t) = [uu * zinv(nneis) for uu in (r, t)]
    # for neighboring edges, we want r and t to be similar to each other
    f_rtsmooth = op.sum((r[v]-r[u])**2 + (t[v]-t[u])**2) / m
    # we also want r and t to predict the radial and tangential magnification of the node, so
    # we want to make sure that edges are the right distances away from each other based on the
    # surface edge lengths and the distance around the vertex at the center
    # for this we'll want some constant info about the surface edges/angles
    # okay, in terms of the visual field coordinates of the parameters, we will want to know
    # the angular position of each node
    # organize face info
    mnden   = 0.0001
    (e,qs,qt) = np.transpose([(i,e[0],e[1]) for (i,e) in enumerate(submesh.tess.edge_faces)
                              if len(e) == 2 and selen[i] > mnden
                              if sarea[e[0]] > mnden and sarea[e[1]] > mnden])
    (fis,q) = np.unique(np.concatenate([qs,qt]), return_inverse=True)
    (qs,qt)   = np.reshape(q, (2,-1))
    o       = len(fis)
    faces   = fs[:,fis]
    fangs   = fangs[:,fis]
    varea   = op.signed_face_areas(faces)
    srfangmtx = sps.csr_matrix(
        (fangs.flatten(),
         (faces.flatten(), np.concatenate([np.arange(o), np.arange(o), np.arange(o)]))),
        (n, o))
    srfangtot = flattest(srfangmtx.sum(axis=1))
    # normalize this angle matrix by the total and put it back in the same order as faces
    srfangmtx = zdivide(srfangmtx, srfangtot / (np.pi*2)).tocsr().T
    nrmsrfang = np.array([sps.find(srfangmtx[k])[2][np.argsort(fs[:,k])] for k in range(o)]).T
    # okay, now compare these to the actual angles;
    # we also want to know, for each edge, the angle relative to the radial axis; let's start
    # by organizing the faces into the units we compute over:
    (fa,fb,fc) = [np.concatenate([faces[k], faces[(k+1)%3], faces[(k+2)%3]]) for k in range(3)]
    atht = thts[fa]
    # we only have to worry about the (a,b) and (a,c) edges now; from the perspective of a...
    bphi = op.atan2(y[fb] - y[fa], x[fb] - x[fa]) - atht
    cphi = op.atan2(y[fc] - y[fa], x[fc] - x[fa]) - atht
    ((bcos,bsin),(ccos,csin)) = bccssn = [(op.cos(q),op.sin(q)) for q in (bphi,cphi)]
    # the distance should be predicted by surface edge length times ellipse-magnification
    # prediction; we have made uphi and vphi so that radial axis is x axis and tan axis is y
    (ra,ta) = (op.abs(r[fa]), op.abs(t[fa]))
    bslen = np.sqrt(np.sum((sxyz[:,fb] - sxyz[:,fa])**2, axis=0))
    cslen = np.sqrt(np.sum((sxyz[:,fc] - sxyz[:,fa])**2, axis=0))
    bpre_x = bcos * ra * bslen
    bpre_y = bsin * ta * bslen
    cpre_x = ccos * ra * cslen
    cpre_y = csin * ta * cslen
    # if there's a global field sign, we want to invert these predictions when the measured
    # angle is the wrong sign
    if fieldsign is not None:
        varea_f = varea[np.concatenate([np.arange(o) for _ in range(3)])] * fieldsign
        fspos = (op.sign(varea_f) + 1)/2
        fsneg = 1 - fspos
        (bpre_x,bpre_y,cpre_x,cpre_y) = (
            bpre_x*fspos - cpre_x*fsneg, bpre_y*fspos - cpre_y*fsneg,
            cpre_x*fspos - bpre_x*fsneg, cpre_y*fspos - bpre_y*fsneg)
    (ax,ay,bx,by,cx,cy) = [x[fa],y[fa],x[fb],y[fb],x[fc],y[fc]]
    (cost,sint) = [op.cos(atht), op.sin(atht)]
    (bpre_x, bpre_y) = (bpre_x*cost - bpre_y*sint + ax, bpre_x*sint + bpre_y*cost + ay)
    (cpre_x, cpre_y) = (cpre_x*cost - cpre_y*sint + ax, cpre_x*sint + cpre_y*cost + ay)
    # okay, we can compare the positions now...
    f_rt = op.sum((bpre_x-bx)**2 + (bpre_y-by)**2 + (cpre_x-cx)**2 + (cpre_y-cy)**2) * 0.5/o
    f_vmag = f_rtsmooth # + f_rt #TODO: the rt part of this needs to be debugged
    object.__setattr__(f_vmag, 'meta_data', pyr.m(f_rtsmooth=f_rtsmooth, f_rt=f_rt))
    return f_vmag

def disk_vmag(hemi, retinotopy='any', yields='axes', min_cod=0, npoints=50, **kw):
    '''
    disk_vmag(mesh) yields the visual magnification based on the projection of disks on the cortical
      surface into the visual field.

    All options accepted by mag_data() are accepted by disk_vmag(). In addition, the parameters
    yields and min_cod may be provided. The min_cod parameter indicates the minimum coefficient of
    determination (r-squared), calculated between the fitted-ellipse and the vetex neighbor's raw
    positions, that is needed to be included in the returrn values. The yields option determines 
    what the return value should be. The default value is 'axes', but the following values are
    accepted:
      * 'axes' indicates that the return value should be an (n x 2 x 2) array where n is the number
        of vertices in the mesh or cortex; each 2x2 matrix is the [rad_x rad_y; tan_x tan_y] axes.
      * 'cod' indicates that only the coefficient of determination for the least-squares fit should
        be returned.
      * 'all' indicates that the return value should be (axes, cod).
    In all cases, nans indicate vertices that were not part of the retinotopy mask, that did not
    have CODs above the threshold, or that had too few neighbors to fit an ellipse.
    '''
    mdat = mag_data(hemi, retinotopy=retinotopy, **kw)
    if pimms.is_vector(mdat): return tuple([face_vmag(m, to=to) for m in mdat])
    elif pimms.is_vector(mdat.keys(), 'int'):
        return pimms.lazy_map({k: curry(lambda k: disk_vmag(mdat[k], to=to), k)
                               for k in six.iterkeys(mdat)})
    # for disk cmag we start by making sets of circular points around each vertex
    msh  = mdat['submesh']
    n    = msh.vertex_count
    N    = hemi.vertex_count
    vxy  = mdat['visual_coordinates'].T
    sxy  = msh.coordinates.T
    parts = msh.tess.neighborhood_face_partition
    emax = len(parts)
    mindist = np.full(n, np.inf)
    angtot = np.zeros(n)
    angs = []
    for (a,b,c) in parts:
        abvec = (sxy[b] - sxy[a])
        acvec = (sxy[c] - sxy[a])
        ablen = np.sqrt(np.sum(abvec**2, axis=1))
        aclen = np.sqrt(np.sum(acvec**2, axis=1))
        mindist[a] = np.min([mindist[a], ablen, aclen], axis=0)
        abvec /= ablen[:,None]
        acvec /= aclen[:,None]
        angs.append(np.arccos(np.sum(abvec*acvec, axis=1)))
        angtot[a] += angs[-1]
    ellipses = np.full((n,npoints,2), np.nan)
    angprog = np.zeros(n)
    ang_per_pt = angtot / npoints
    for (ang,part) in zip(angs, parts):
        (a,b,c) = part
        sabvec = (sxy[b] - sxy[a])
        sacvec = (sxy[c] - sxy[a])
        sablen = np.sqrt(np.sum(sabvec**2, axis=1))
        saclen = np.sqrt(np.sum(sacvec**2, axis=1))
        vabvec = (vxy[b] - vxy[a])
        vacvec = (vxy[c] - vxy[a])
        vablen = np.sqrt(np.sum(vabvec**2, axis=1))
        vaclen = np.sqrt(np.sum(vacvec**2, axis=1))
        uabvec = vabvec * (mindist[a] / sablen)[:,None]
        uacvec = vacvec * (mindist[a] / saclen)[:,None]
        uablen = np.sqrt(np.sum(uabvec**2, axis=1))
        uaclen = np.sqrt(np.sum(uacvec**2, axis=1))
        uabtht = np.arctan2(uabvec[:,1], uabvec[:,0])
        uactht = np.arctan2(uacvec[:,1], uacvec[:,0])
        angprog_start = np.array(angprog[a])
        p0 = np.round((angprog[a] / angtot[a]) * npoints).astype('int')
        ii = np.arange(len(a))
        aa = a
        while True:
            angprog_aa = angprog[aa]
            ii = ii[(angprog_aa - angprog_start[ii] < ang[ii]) &
                    (angprog_aa + ang_per_pt[aa] <= angtot[aa])]
            (aa,bb,cc) = (a[ii], b[ii], c[ii])
            if len(ii) == 0: break
            (rb,rc,thb,thc) = (uablen[ii], uaclen[ii], uabtht[ii], uactht[ii])
            angw = (angprog[aa] - angprog_start[ii]) / ang[ii]
            iangw = 1 - angw
            tht = iangw*thb + angw*thc
            r = iangw*rb + angw*rc
            atpt = np.round(angprog[aa] / angtot[aa] * npoints).astype('int')
            ellipses[aa, atpt, :] = np.transpose([np.cos(tht)*r, np.sin(tht)*r])
            angprog[aa] += ang_per_pt[aa]
    # At this point, ellipses is the set of ellipse points to fit for each neighborhood.
    # Its dimensions repesent (vertices, neighbors, x/y coords); we want to rotate the
    # points to be along their center's implied rad/tan axis.
    vrs  = np.sqrt(np.sum(vxy**2, axis=1))
    irs  = zinv(vrs)
    coss = vxy[:,0] * irs
    sins = vxy[:,1] * irs
    # rotating each ellipse by negative-theta gives us x=radial and y=tangential
    cels = (coss * ellipses.T)
    sels = (sins * ellipses.T)
    rots = np.transpose([cels[0] + sels[1], cels[1] - sels[0]], [1,2,0])
    rsrt = np.sqrt(np.sum(rots**2, axis=2)).T
    (csrt,snrt) = zinv(rsrt) * rots.T
    # At this point, csrt and snrt have dimensions that represent (vertices, neighbors)
    # ... (a*cos(rots))**2 + (b*sin(rots))**2 ~= r(rots) where a = radial vmag and b = tan vmag
    axes = []
    cods = []
    idxs = []
    for (r,c,s,irad,i) in zip(rsrt,csrt,snrt,zinv(mindist),range(n)):
        # if the center point is way outside the min/max, skip it
        (x,y) = (r*c, r*s)
        if len(np.unique(np.sign(x))) < 2 or len(np.unique(np.sign(y))) < 2: continue
        mudst = np.sqrt(np.sum(np.mean([x, y], axis=1)**2))
        #if mudst > np.min(r): continue
        # okay, fit an ellipse...
        fs = np.transpose([c,s])**2
        try:
            (ab,rss,rnk,svs) = np.linalg.lstsq(fs, r**2, rcond=None)
            # ab = np.sqrt(np.abs(ab)) # Why the sqrt?
            ab = np.abs(ab)
            if len(rss) == 0 or rnk < 2: continue # or np.min(svs/np.sum(svs)) < 0.01: continue
            cod = 1 - rss[0]*zinv(np.sum(r**2))
            if cod < min_cod: continue
            axes.append(ab * irad)
            cods.append(cod)
            idxs.append(i)
        except Exception as e: continue
    (axes,cods,idxs) = [np.asarray(u) for u in (axes, cods, idxs)]
    # make the return value; this is a 2x2 matrix for each vertex
    if yields != 'cod':
        raxes = np.full((N, 2), np.nan)
        raxes[idxs,:] = axes
        if yields == 'axes': return raxes
    rcods = np.full(N, np.nan)
    rcods[idxs] = cods
    if yields == 'cod': return rcods
    else: return (raxes, rcods)

def face_vmag(hemi, retinotopy='any', to=None, null=np.nan, **kw):
    '''
    face_vmag(mesh) yields the visual magnification based on the projection of individual faces on
      the cortical surface into the visual field.
    face_vmag(mdat) uses the given magnification data mdat (as returned from mag_data()); if valid
      magnification data is passed then all options related to the mag_data() function are ignored.

    All options accepted by mag_data() are accepted by face_vmag().

    The additional optional arguments are also accepted:
      * to (default: None) specifies that the resulting data should be transformed in some way;
        these transformations are:
          * 'faces': returns a property of the visual magnification value of each face;
          * 'vertices': returns a property of the visual magnification value of each vertex, as
            determined by averaging the magnification 
          * None (the detault) is equivalent to 'faces'
      * null (default: nan) specifies the value to use in the return-value for faces or
        vertices outside the mesh.
    '''
    to = parse_toopt_facevertex(to)
    if pimms.is_vector(hemi):
        return tuple([face_vmag(m, to=to) for m in mdat])
    if not is_mag_data(hemi):
        if pimms.is_map(hemi) and all(pimms.is_int(k) for k in six.iterkeys(hemi)):
            m = {k: curry(lambda k: face_vmag(mdat[k], to=to), k) for k in ks}
            return pimms.lmap(m)
        else:
           hemi = mag_data(hemi, retinotopy=retinotopy, **kw)
    mdat = hemi
    # Okay, we have a single mag-data dict; get the face visual and cortical areas:
    mesh = mdat['mesh']
    submesh = mdat['submesh']
    vismesh = submesh.copy(coordinates=mdat['visual_coordinates'])
    subface_vmags = zdivide(vismesh.face_areas, submesh.face_areas, null=0)
    face_vmags = np.full(mesh.tess.face_count, null)
    face_vmags[mesh.tess.index[submesh.tess.faces]] = subface_vmags
    # convert these if the to argument so-requires it:
    if to == 'face': return face_vmags
    # possible #TODO here: super-tesselate the mesh and calculate the actual visual vertex areas
    # then divide by the vertex cortical surface ares to do this properly.
    vmag = np.full(mesh.vertex_count, null)
    for (vtx,faces) in six.iteritems(mdat['submesh'].tess.vertex_face_index):
        tmp = face_vmags[list(faces)]
        ii = mesh.tess.index(vtx)
        jj = np.where(np.isfinite(tmp))[0]
        jj = jj[tmp[jj] > 0]
        if len(jj) > 0: vmag[ii] = np.mean(tmp[jj])
    return vmag

def face_rtcmag(hemi, retinotopy='any', to=None, **kw):
    '''
    face_rtcmag(mesh) yields the cortical magnification matrices of each face in
      the given mesh based on the projection of individual faces on the
      cortical surface into the visual field.
    face_rtcmag(mdat) uses the given magnification data mdat (as returned from
      mag_data()); if valid magnification data is passed then all options
      related to the mag_data() function are ignored.

    The return value of face_rtcmag() is an tuple of (rad_cmag, tan_cmag, theta)
    where rad_cmag and tan_cmag are both linear cortical magnifications in the
    associated directions, and theta is the angle between them on cortex.

    All options accepted by mag_data() are accepted by face_rtcmag().

    The additional optional arguments are also accepted:
      * to (default: None) specifies that the resulting data should be
        transformed in some way; these transformations are:
          * None or 'data': returns the full magnification data without 
            transformation;
          * 'faces': returns a property of the visual magnification value of
            each face;
          * 'vertices': returns a property of the visual magnification value of
            each vertex, as determined by averaging the magnification.
    '''
    if pimms.is_vector(hemi):
        return tuple([face_rtcmag(m, to=to) for m in mdat])
    if not is_mag_data(hemi):
        if (pimms.is_map(hemi) and 
            all(pimms.is_int(k) for k in six.iterkeys(hemi))):
            m = {k: curry(lambda k: face_rtcmag(mdat[k], to=to), k) for k in ks}
            return pimms.lmap(m)
        else:
            return face_rtcmag(mag_data(hemi, retinotopy=retinotopy, **kw), to=to)
    to = parse_toopt_facevertex(to)
    mdat = hemi
    # Okay, we have a single mag-data dict; get the face visual and cortical areas:
    mesh = mdat['mesh']
    submesh = mdat['submesh']
    vismesh = submesh.copy(coordinates=mdat['visual_coordinates'])
    face_vmags = zdivide(vismesh.face_areas, submesh.face_areas, null=0)
    # To determine radial and tangential, we start by drawing lines from each 
    # face center to whichever other triangle side in the radial and tangential
    # directions.
    fx0 = vismesh.face_centers
    (ax,bx,cx) = abcx = vismesh.face_coordinates
    ecc = np.sqrt(np.sum(fx0**2, axis=0))
    tht = np.arctan2(fx0[1], fx0[0])
    u_rad = fx0 * zinv(ecc, null=np.nan)
    u_tan = np.array([-u_rad[1], u_rad[0]])
    # Get the intersection of these u_rad and u_tan with the triangle sides
    radlines = [np.hstack([u,u,u]) for u in (fx0, fx0+u_rad)]
    tanlines = [np.hstack([u,u,u]) for u in (fx0, fx0+u_tan)]
    trisides = [np.hstack([ax,bx,cx]), np.hstack([bx,cx,ax])]
    isect_rad = line_segment_intersection_2D(radlines, trisides)
    isect_tan = line_segment_intersection_2D(tanlines, trisides)
    vis_segs = []
    for (isect_dat, u_dat) in zip([isect_rad, isect_tan], [u_rad, u_tan]):
        isects_wnan = np.array([u.T for u in np.reshape(np.transpose(isect_dat), (3,-1,2))])
        # There should be 2 intersections per column of these matrices.
        isects = np.zeros((2,) + isects_wnan.shape[1:])
        isects_fin = np.isfinite(isects_wnan[:,0,:])
        nisects = np.sum(isects_fin, axis=0)
        ii3 = (nisects == 3)
        if np.any(ii3):
            wh3 = np.where(ii3)[0]
            (iab, ibc, ica) = isects_wnan
            d2b = np.sum((iab[:,wh3] - ibc[:,wh3])**2, axis=0)
            d2c = np.sum((ibc[:,wh3] - ica[:,wh3])**2, axis=0)
            d2a = np.sum((ica[:,wh3] - iab[:,wh3])**2, axis=0)
            dists = [d2b, d2c, d2a]
            far_isects = [ica, iab, ibc]
            small = np.argmin(dists, axis=0)
            # The two with the smallest distance between them we average
            for s in (0,1,2):
                iis = np.where(small == s)[0]
                (far,cl1,cl2) = [far_isects[(s+k)%3][:, wh3[iis]] for k in (0,1,2)]
                cl = np.mean([cl1,cl2], axis=0)
                isects[0,:,wh3[iis]] = cl.T
                isects[1,:,wh3[iis]] = far.T
        # For the rest, we can just take the good two intersections.
        ii2 = ~ii3
        k = np.argmin(isects_fin, axis=0)[ii2]
        rng = np.arange(isects_wnan.shape[2])[ii2]
        isects[0,:,ii2] = isects_wnan[(k+1)%3, :, rng]
        isects[1,:,ii2] = isects_wnan[(k+2)%3, :, rng]
        # For all the intersections, we want to make sure that they are oriented
        # correctly w.r.t rad/tan (i.e., make we didn't flip the vectors).
        ivec = isects[1] - isects[0]
        sgn = np.sign(np.sum(ivec * u_dat, axis=0))
        ii = np.where(sgn == -1)[0]
        tmp = np.array(isects[1,:,ii])
        isects[1,:,ii] = isects[0,:,ii]
        isects[0,:,ii] = tmp
        vis_segs.append(isects)
    # That gives us all the intersections in visual space.
    # We now want them in cortical space as well, and we want to convert both
    # into distances.
    vis_segs = np.array(vis_segs)
    bc_coords = np.array(
        [[(a, b, 1.0 - a - b)
          for isect in isects
          for (a,b) in [cartesian_to_barycentric_2D(abcx, isect)]]
         for isects in vis_segs])
    srf_abcx = submesh.face_coordinates
    srf_segs = np.array(
        [[np.sum(srf_abcx * isect[:,None,:], axis=0)
          for isect in isects]
         for isects in bc_coords])
    # Okay, convert the segments into lengths.
    srf_rtlen = np.array([np.sqrt(np.sum((a - b)**2, axis=0)) for (a,b) in srf_segs])
    vis_rtlen = np.array([np.sqrt(np.sum((a - b)**2, axis=0)) for (a,b) in vis_segs])
    (rmag, tmag) = srf_rtlen / vis_rtlen
    # Those magnifications fail in consider the angle between the surface lines,
    # though; to correct for this, we just find it first:
    srf_vecs = np.array([b - a for (a,b) in srf_segs])
    srf_vecs *= zinv(np.sqrt(np.sum(srf_vecs**2, axis=1)))[:,None,:]
    theta = np.arccos(np.sum(srf_vecs[0] * srf_vecs[1], axis=0))
    # We also want to preserve +/- angle values to indicate whether there was a
    # flip (positive for positive fieldsign and negative for negative).
    fsign = retinotopic_field_sign(mesh, 'faces', retinotopy=mdat['retinotopy_data'])
    theta *= fsign[mesh.tess.index[submesh.tess.faces]]
    # Now accumulate and return.
    res = []
    ii = mesh.tess.index[submesh.tess.faces]
    for x in (rmag,tmag,theta):
        y = np.full(mesh.tess.face_count, np.nan)
        y[ii] = x
        res.append(y)
    return tuple(res)

@pimms.immutable
class FieldOfView(object):
    '''
    FieldOfView is a class that represents and calculates the field of view in a cortical area.
    '''
    def __init__(self, angle, eccen, sigma,
                 scale=None, weight=None, search_scale=3.0, bins=6,
                 normalize_weights=True, weights_method='height'):
        self.polar_angle = angle
        self.eccentricity = eccen
        self.sigma = sigma
        self.weight = weight
        self.bins = bins
        self.search_scale = search_scale
        self.scale = scale
        self.normalize_weights = normalize_weights
        self.weights_method = weights_method
    @pimms.param
    def polar_angle(pa):  return pimms.imm_array(pa)
    @pimms.param
    def eccentricity(ec): return pimms.imm_array(ec)
    @pimms.param
    def sigma(r):         return pimms.imm_array(r)
    @pimms.param
    def scale(s): return s
    @pimms.param
    def normalize_weights(nw): return bool(nw)
    @pimms.param
    def weights_method(wm):
        if wm is None or wm is Ellipsis: return 'height'
        wm = wm.lower()
        if wm in ['height', 'h', 'z']: return 'height'
        elif wm in ['volume', 'vol', 'v']: return 'volume'
        else: raise ValueError('unrecognized weights_method: %s' % (wm,))
    @pimms.param
    def weight(w):
        if w is None: return None
        w = np.array(w)
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
    def _weight(weight, polar_angle, normalize_weights, sigma, weights_method):
        if weight is None:
            weight = np.ones(len(polar_angle))
            normalize_weights = True
        if normalize_weights:
            weight = weight / np.sum(weight)
        if weights_method == 'volume':
            weight = weight / (2 * np.pi * sigma)
        return pimms.imm_array(weight)
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
        try:              from scipy.spatial import cKDTree as shash
        except Exception: from scipy.spatial import KDTree  as shash
        return tuple([shash(coordinates[ii]) for ii in sigma_bins])
    # Methods
    def __call__(self, x, y=None):
        if y is not None: x = (x,y)
        x = np.asarray(x)
        if len(x.shape) == 1: return self([x])[0]
        x = np.transpose(x) if x.shape[0] == 2 else x
        if not x.flags['WRITEABLE']: x = np.array(x)
        crd = self.coordinates
        sig = self.sigma
        wts = self._weight
        res = np.zeros(x.shape[0])
        for (sh, qd, bi) in zip(self.spatial_hashes, self.bin_query_distances, self.sigma_bins):
            neis = sh.query_ball_point(x, qd)
            res += [
                np.sum(w * np.exp(-0.5 * d2/s**2))
                for (ni,pt) in zip(neis,x)
                for ii in [bi[ni]]
                for (w,s,d2) in [(wts[ii], sig[ii], np.sum((crd[ii] - pt)**2, axis=1))]]
        return res
def field_of_view(mesh, retinotopy='any', mask=None, search_scale=3.0, bins=6, weights=Ellipsis,
                  normalize_weights=True, weights_method='height'):
    '''
    field_of_view(obj) yields a field-of-view function for the given vertex-set object or mapping of
      retinotopy data obj.

    The field-of-view function is a measurement of how much total pRF weight there is at each point
    in the visual field; essentially it is the sum of all visual-field Gaussian pRFs, where the
    Gaussians are normalized both by the pRF size (i.e., each pRF is a 2D normal distribution whose
    center comes from its polar angle and eccentricity and whose sigma parameter is the pRF radius)
    and the weight (if any weight such as the variance explained is included in the retinotopy data,
    then the normal distributions are multiplied by this weight, otherwise weights are considered to
    be uniform). Note that the weights are normalized by their sum, so the resulting field-of-view
    function should be a valid probability distribution over the visual field. To disable this,
    you can use the nomalize_weights=False option.

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
      * weights (default: Ellipsis) specifies the weights that should be used. Ellipsis inidicates
        that whatever weights are found in the retinotopy data should be used and none should be
        used otherwise.
      * normalize_weights (default: True) may be set to False to prevent normalization of the
        weights. The default value of True normalizes the weights to be equal to 1.
      * weights_method (default: 'height') specifies whether the weights should be considered the
        heights or the volumes of the Gaussians pRFs.
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
    if weights is Ellipsis or weights is True:
        wgt = next((retino[q] for q in ('variance_explained', 'weight') if q in retino), None)
    elif weights is None or weights is False:
        wgt = None
    else:
        wgt = mesh.property(weights)
    # Get the indices we care about
    ii = geo.to_mask(mesh, mask, indices=True)
    return FieldOfView(ang[ii], ecc[ii], sig[ii], weight=wgt[ii],
                       search_scale=search_scale, bins=bins,
                       normalize_weights=normalize_weights, weights_method=weights_method)

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
        try:              from scipy.spatial import cKDTree as shash
        except Exception: from scipy.spatial import KDTree  as shash
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
        if not x.flags['WRITEABLE']: x = np.array(x)
        return self.spatial_hash.query(x, n)
    def __call__(self, x, y=None):
        (d,ii) = self.nearest(x, y)
        n = self.spatial_hash.n
        bd = (ii == n)
        ii[bd] = 0
        carea = np.reshape(self.surface_area[ii.flatten()], ii.shape)
        carea[bd] = np.nan
        d[bd] = np.nan
        if len(d.shape) == 1:
            varea = np.pi * np.nanmax(d)**2
            carea = np.nansum(carea)
        else:
            varea = np.pi * np.nanmax(d, axis=1)**2
            carea = np.nansum(carea, axis=1)
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
      * retinotopy (default: 'any') specifies the retinotopy data to be used; this is passed to
        the retinotopy_data function with the mesh.
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
    retino = retinotopy_data(mesh, retinotopy)
    # Convert from polar angle/eccen to longitude/latitude
    (ang,ecc) = as_retinotopy(retino, 'visual')
    # get the surface area
    if surface_area is None: surface_area = 'midgray_surface_area'
    if pimms.is_str(surface_area):
        if surface_area in retino: surface_area = retino[surface_area]
        elif geo.is_vset(mesh):    surface_area = mesh.property(surface_area)
        else:                      surface_area = mesh[surface_area]
    # get the weight
    if weight is not None:
        if pimms.is_str(weight):
            if weight in retino:    weight = retino[weight]
            elif geo.is_vset(mesh): weight = mesh.property(weight)
            else:                   weight = mesh[weight]
    wgt = next((retino[q] for q in ('variance_explained', 'weight') if q in retino), None)
    # Get the indices we care about
    ii = geo.to_mask(mesh, mask, indices=True)
    ii = ii[np.isfinite(ang[ii]) & np.isfinite(ecc[ii])]
    # get our nnearest
    if nnearest is None: nnearest = int(np.ceil(np.sqrt(len(ii)) * np.log(len(ii))))
    return ArealCorticalMagnification(ang[ii], ecc[ii], surface_area[ii],
                                      weight=(None if wgt is None else wgt[ii]),
                                      nnearest=nnearest)

def isoline_vmag(hemi, isolines=None, surface='midgray', min_length=2, **kw):
    '''
    isoline_vmag(hemi) calculates the visual magnification function f using the default set of
      iso-lines (as returned by neuropythy.vision.visual_isolines()). The hemi argument may
      alternately be a mesh object.
    isoline_vmag(hemi, isolns) uses the given iso-lines rather than the default ones.
    
    The return value of this funciton is a dictionary whose keys are 'tangential', 'radial', and
    'areal', and whose values are the estimated visual magnification functions. These functions
    are of the form f(x,y) where x and y can be numbers or arrays in the visual field.
    '''
    from neuropythy.util import (curry, zinv)
    from neuropythy.mri import is_cortex
    from neuropythy.vision import visual_isolines
    from neuropythy.geometry import to_mesh
    # if there's no isolines, get them
    if isolines is None: isolines = visual_isolines(hemi, **kw)
    # see if the isolines is a lazy map of visual areas; if so return a lazy map recursing...
    if pimms.is_vector(isolines.keys(), 'int'):
        f = lambda k: isoline_vmag(isolines[k], surface=surface, min_length=min_length)
        return pimms.lazy_map({k:curry(f, k) for k in six.iterkeys(isolines)})
    mesh = to_mesh((hemi, surface))
    # filter by min length
    if min_length is not None:
        isolines = {k: {kk: {kkk: [vvv[ii] for ii in iis] for (kkk,vvv) in six.iteritems(vv)}
                        for (kk,vv) in six.iteritems(v)
                        for iis in [[ii for (ii,u) in enumerate(vv['polar_angles'])
                                     if len(u) >= min_length]]
                        if len(iis) > 0}
                    for (k,v) in six.iteritems(isolines)}
    (rlns,tlns) = [isolines[k] for k in ['eccentricity', 'polar_angle']]
    if len(rlns) < 2: raise ValueError('fewer than 2 iso-eccentricity lines found')
    if len(tlns) < 2: raise ValueError('fewer than 2 iso-angle lines found')
    # grab the visual/surface lines
    ((rvlns,tvlns),(rslns,tslns)) = [[[u for lns in six.itervalues(xlns) for u in lns[k]]
                                      for xlns in (rlns,tlns)]
                                     for k in ('visual_coordinates','surface_coordinates')]
    # calculate some distances
    (rslen,tslen) = [[np.sqrt(np.sum((sx[:,:-1] - sx[:,1:])**2, 0)) for sx in slns]
                     for slns in (rslns,tslns)]
    (rvlen,tvlen) = [[np.sqrt(np.sum((vx[:,:-1] - vx[:,1:])**2, 0)) for vx in vlns]
                     for vlns in (rvlns,tvlns)]
    (rvxy, tvxy)  = [[0.5*(vx[:,:-1] + vx[:,1:]) for vx in vlns] for vlns in (rvlns,tvlns)]
    (rvlen,tvlen,rslen,tslen) = [np.concatenate(u) for u in (rvlen,tvlen,rslen,tslen)]
    (rvxy,tvxy)   = [np.hstack(vxy) for vxy in (rvxy,tvxy)]
    (rvmag,tvmag) = [vlen * zinv(slen) for (vlen,slen) in zip([rvlen,tvlen],[rslen,tslen])]
    return {k: {'visual_coordinates':vxy, 'visual_magnification': vmag,
                'visual_lengths': vlen, 'surface_lengths': slen}
            for (k,vxy,vmag,vlen,slen) in zip(['radial','tangential'], [rvxy,tvxy],
                                              [rvmag,tvmag], [rvlen,tvlen], [rslen,tslen])}

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
                except Exception:
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
