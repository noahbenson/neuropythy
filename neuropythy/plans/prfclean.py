####################################################################################################
# neuropythy/plans/prfclean.py
# A PIMMS Calcuulation Plan for cleaning retinotopic maps within deduced/labeled visual areas.
# By Noah C. Benson

import numpy as np
import pyrsistent as pyr
import os, sys, gzip, six, types, pimms
from ..util import (is_tuple, is_list)
from .core  import (limit_param, unlimit_param)
from ..math import (totensor, astensor, pytorch, to_torchdtype)
from .. import math


# Initialization of Parameters #####################################################################
@pimms.calc('torch')
def init_pytorch_defaults(dtype='float', device='cpu'):
    '''
    This function provides default values for afferent parameters related to
    PyTorch and checks to ensure that PyTorch is installed.

    Afferent parameters:
     @ dtype May optionally specify the dtype for PyTorch to use in the
       optimization. This may be a PyTorch dtype object or it may be a string,
       in which case it is looked up as a member of the torch module (i.e.,
       'float' 
     @ device The pytorch device name to be used for all tensors (default:
       'cpu').

    Efferent values:
     @ torch The PyTorch module used in the optimization.
    '''
    # Make sure we can import pytorch.
    torch = pytorch()
    # Make sure that we have a valid dtype and device parameters.
    u = totensor(0, dtype=dtype, device=device)
    # If those didn't raise exceptions, it's good enough for now!
    return (torch,)
@pimms.calc('gradients')
def init_gradients(grad_coords=True, grad_cmag_rad=True, grad_cmag_tan=True, grad_ipsi=True):
    '''
    Initializes an empty (mutable!) dictionary for tracking the elements of the
    model that are part of the gradient descent. The gradients dictionary is
    used as a mutable dictionary in which any PyTorch tensor that is part of the
    gradient descent optimization can add itself. These data are needed for the
    PyTorch optimizer.

    Afferent parameters:
     @ grad_coords Specifies whether the tensors representing the coordinates of
       the PRF centers should track their gradients (default: True).
     @ grad_cmag_rad Specifies whether the tensors representing the radial 
       cortical magnification parameters of the retinotopy model should track
       their gradients (default: True).
     @ grad_cmag_tan Specifies whether the tensors representing the tangential
       cortical magnification parameters of the retinotopy model should track
       their gradients (default: True).
     @ grad_ipsi Whether the uvm_ipsi and lvm_ipsi parameters, representing the
       ipsilateral representation in each visual area, are part of the optimized
       parameters whose gradients are required.

    Efferent values:
     @ gradients A mutable dictiionary whose values are PyTorch tensors whose
       gradients are tracked. Any parameter of the optimization should be added
       to this dictionary during the model's initialization.
    '''
    return ({},)
init_params_plan = pimms.plan(gradients=init_gradients,
                              pytorch=init_pytorch_defaults)

# Initialization of Subject Data ###################################################################
@pimms.calc('meas_angle', 'meas_eccen', 'meas_theta', 'meas_prfsize', 'meas_cod',
            'meas_x', 'meas_y', 'meas_properties', 'hemifield_sign')
def init_meas(cortex, device, dtype, retinotopy='prf_'):
    '''
    Calculation:
      (cortex, retinotopy, dtype) => (meas_{angle,eccen,theta,prfsize,vexpl})
    Locates the measured retinotopy data.
    
    Afferent parameters:
     @ cortex The neuropythy cortex/hemisphere object on which the optimization
       occurs.
     @ retinotopy The argument for neuropythy's retinotopy_data function; this
       is typically a string prefix for retinotopy data on the cortex's
       properties ('prf_' for HCP subjects), or a map of the data itself.
    
    Efferent values:
     @ meas_angle The measured polar angle of each vertex on 'cortex' in
       neuropythy's 'visual' coordinate style (positive is clockwise, UVM is
       zero, units are degrees).
     @ meas_theta The measured polar angle of each vertex on 'cortex' in
       neuropythy's 'standard' coordinate style (positive is counter-clockwise,
       RHM is zero, units are radians).
     @ meas_eccen The measured eccentricity of each vertex on 'cotex' in
       visual degrees.
     @ meas_cod The measured variance-explained (COD) of each vertex on
       'cotex' represented as a real number between 0 and 1.
     @ meas_prfsize The measured pRF size of each vertex on 'cotex' in visual
       degrees.
     @ meas_x The measured x coordinate of each vertex.
     @ meas_y The measured y coordinate of each vertex.
     @ meas_properties A persistent map of all the measured properties.
    '''
    import torch, neuropythy as ny
    rd = ny.vision.retinotopy_data(cortex, retinotopy)
    ang = np.array(rd['polar_angle'])
    ecc = np.array(rd['eccentricity'])
    vxp = np.array(rd['variance_explained'])
    rfs = np.array(rd['radius'])
    tht = np.pi/180 * (np.mod(90 - ang + 180, 360) - 180)
    # NaNs need to be converted to 0's
    ang[~np.isfinite(ang)] = 0.0
    ecc[~np.isfinite(ecc)] = 0.0
    vxp[~np.isfinite(vxp)] = 0.0
    rfs[~np.isfinite(rfs)] = 0.0
    tht[~np.isfinite(tht)] = 0.0
    x = np.cos(tht) * ecc
    y = np.sin(tht) * ecc
    logecc = np.log(0.75 + ecc)
    logx = np.cos(tht) * logecc
    logy = np.sin(tht) * logecc
    ps = {'angle':   totensor(ang, dtype=dtype, device=device),
          'eccen':   totensor(ecc, dtype=dtype, device=device),
          'theta':   totensor(tht, dtype=dtype, device=device),
          'prfsize': totensor(rfs, dtype=dtype, device=device),
          'cod':     totensor(vxp, dtype=dtype, device=device),
          'x':       totensor(x, dtype=dtype, device=device),
          'y':       totensor(y, dtype=dtype, device=device)}
    ps = pyr.pmap(ps)
    retval = {('meas_' + k): v for (k,v) in six.iteritems(ps)}
    retval['meas_properties'] = ps
    # hemifiield sign
    retval['hemifield_sign'] = 1 if cortex.chirality == 'lh' else -1
    return retval
@pimms.calc('cortex_labels', 'visual_areas')
def init_labels(cortex, labels, device, torch):
    '''
    Calculation:
      (cortex, labels, label_key) => cortex_labels
    Initializes the visual area labels that are used in the optimization.

    Afferent parameters:
     @ labels Must be the labels used in the optimization. These must be integer
       labels (one per vertex of coftex) or the name of a property of cortex.
       Note that the labels must not use 0 as an ROI: 0 must always indicate
       vertices that should not be optimized.

    Efferent values:
     @ cortex_labels A numpy array of the labels on cortex. 
     @ visual_areas A tuple of the visual area labels that are included in thee
       optimization.
    '''
    if pimms.is_str(labels): labels = cortex.prop(labels)
    if torch.is_tensor(labels): labels = labels.detach().numpy()
    labels = np.array(labels, 'int')
    labels.setflags(write=False)
    if len(labels.shape) != 1 or labels.shape[0] != cortex.vertex_count:
        raise ValueError("labels parameter's length does not match cortex")
    vas = np.setdiff1d(np.unique(labels), [0])
    return (labels, tuple(vas))
@pimms.calc('full_mesh', 'mesh', 'roi_meshes', 'roi_meshindices', 'mesh_labels',
            'label_sort', 'label_indices')
def init_mesh(torch, cortex, meas_properties, cortex_labels, visual_areas):
    '''
    Calculation:
      (cortex, cortex_labels, visual_areas, meas_properties) 
        => (full_mesh, mesh, roi_meshes, labels)
    Loads the midgray mesh from the cortex object and limits it the vertices
    over which the minimization operates.
   
    Afferent parameters:
     @ cortex The neuropythy Cortex object (representing a hemisphere) to use.
    
    Efferent values:
     @ full_mesh The migray mesh for the cortex being modeled.
     @ mesh The midgray surface mesh on which the model is implemented,
       limited to those vertices whose labels is non-zero.
     @ roi_meshes One mesh per ROI in the visual_areas list, each of which is
       the midgray mesh limited to just the vertices in the given ROI. This
       value is a persistent dictionary whose keys are the visual area labels.
     @ roi_meshindices A persistent map whose keys are ROI labels in the mesh
       and whose vales are vectors of the indices in mesh for the vertices of
       the relevant ROIs.
     @ mesh_labels An array of the labels for eeach vertex of the mesh.
     @ label_sort For parameters of the optimization that are potentially part
       of the gradient, it is often desirable to store these in a vector, one
       per visual area. For this reason, we have a label_sort which gives us
       the labels in the order listed in such arrays. See also label_indices.
     @ label_indices For parameters of the optimization that are potentially
       part of the gradient, it is often desirable to store these in a vector,
       one per visual area. For this reason, we have a label_indices element,
       which is a persistent map that gives us the position (as values) when
       a visual area is looked up (as a key). See also label_sort.
    '''
    p = meas_properties.set('visual_areas', cortex_labels)
    p = {k: (v.detach().numpy() if torch.is_tensor(v) else v)
         for (k,v) in six.iteritems(p)}
    cortex = cortex.with_prop(**p)
    full_mesh = cortex.surface('midgray')
    mesh = full_mesh.submesh(cortex_labels != 0)
    roi_meshes = {k: full_mesh.submesh(cortex_labels == k)
                  for k in visual_areas}
    roi_meshes = pyr.pmap(roi_meshes)
    roi_meshindices = pyr.pmap({k: np.where(np.isin(mesh.labels, m.labels))[0]
                                for (k,m) in six.iteritems(roi_meshes)})
    for m in six.itervalues(roi_meshindices): m.setflags(write=False)
    lbls = cortex_labels[mesh.labels]
    lbl_sort = np.sort(np.unique(lbls))
    lbl_indices = pyr.pmap({k:v for (v,k) in enumerate(lbl_sort)})
    return (full_mesh, mesh, roi_meshes, roi_meshindices, lbls, lbl_sort, lbl_indices)
def cp_facearea(ax, bx, cx):
    '''
    cp_faceareas(ax, bx, cx) yields the signed face areas for the triangle
      or triangles with corners ax, bx, and cx (which may be 2 or 3 dimensional
      vectors, or 2 or 3 x N matrices of column-vectors). If the dimensionality
      is 3, the absolute value of the face area is always returned.
    
    Note that ax, bx, and cx must be given in counter-clockwise order, and they
    must all be either numpy arrays or PyTorch tensors.
    '''
    u = bx - ax
    v = cx - ax
    if ax.shape[0] == 2:
        return (u[0]*v[1] - u[1]*v[0]) / 2.0
    else:
        r2 = ((u[0]*v[1] - u[1]*v[0])**2 +
              (u[2]*v[1] - u[1]*v[2])**2 +
              (u[0]*v[2] - u[2]*v[0])**2)
        return math.safesqrt(r2) / 2.0
def vmag_vertex_areas(vmag_summation_matrices, faces, coords):
    '''
    vmag_vertex_areas(matrices, faces,, coords) yields the surface area
      associated with each vertex represented in the coordinate array, coords.
    
    The matrices and faces rguments are expected to be those returned
    by the init_geometry calculation. The coords must be a 2 x N or 3 x N matrix
    where N is the number of vertices. For 3D coordinates, the result is always
    positive; for 2D coordinates, the result is signed depending on the
    fieldsign.
    '''
    torch = pytorch()
    coords = astensor(coords)
    (amtx, bmtx, cmtx) = vmag_summation_matrices
    (a,b,c) = faces
    # The centers of the face and edges.
    (ax, bx, cx) = (coords[:,a], coords[:,b], coords[:,c])
    f0 = (ax + bx + cx) / 3.0
    ab0 = (ax + bx) / 2.0
    bc0 = (bx + cx) / 2.0
    ca0 = (cx + ax) / 2.0
    # Calculate the areas.
    aareas = cp_facearea(ax, ab0, f0) + cp_facearea(ax, f0, ca0)
    bareas = cp_facearea(bx, bc0, f0) + cp_facearea(bx, f0, ab0)
    careas = cp_facearea(cx, ca0, f0) + cp_facearea(cx, f0, bc0)
    # Transfer over to vertces.
    vertex_aareas = torch.sparse.mm(amtx, torch.reshape(aareas, (-1,1)))[:,0]
    vertex_bareas = torch.sparse.mm(bmtx, torch.reshape(bareas, (-1,1)))[:,0]
    vertex_careas = torch.sparse.mm(cmtx, torch.reshape(careas, (-1,1)))[:,0]
    return vertex_aareas + vertex_bareas + vertex_careas
@pimms.calc('faces', 'edges',
            'face_pairs', 'face_pair_edges', 'face_pair_opposites',
            'edge_lengths_mm', 'face_areas_mm2', 'vertex_areas_mm2',
            'vmag_matrices')
def init_geometry(mesh, dtype, device, torch):
    '''
    Calculation:
      (mesh) => (faces, edges, face_pairs, face_pair_opposites, edge_lenghts_mm,
                 face_areas_mm2, vertex_areas_mm2)
    Loads the midgray mesh from the cortex object and limits it using a mask
    formed by the vertices in a spotlight flatmap of the occipital pole.
   
    Efferent values:
     @ faces The 3 x n numpy arrary of the mesh vertex indices of each corner
       of its faces.
     @ edges The 2 x n numpy arrary of vertex indices in the mesh for each edge
       endpoint.
     @ face_pairs The 2 x n numpy array whose columns represent face indices
       are adjacent to each other. The indices of the edges that lie between
       each face pair is given by face_pair_edges.
     @ face_pair_edges The numpy array of edge indices that match the face
       pairs in the 'face_pairs' value. Each pair of faces that share an
       edge is given by the face inidices in face_pairs[k] and the edge 
       between them is given by the edge index in face_pair_edges[k].
     @ face_pair_opposites The 2 x n numpy array of mesh vetex indices of the
       vetices opposite each face in the face_pairs index. For a column k of
       face_pairs, the face indexed by face_pairs[k][0] is across the edge
       indexed by face_pair_edges[k] from the vertex indexed by
       face_pair_opposites[k][0]. The same is true for the face given by 
       ...[k][1].
     @ edge_lengths_mm The lengths of each edge in mesh in mm.
     @ face_areas_mm2 The cotical surface area of each face in mesh in square
       mm.
     @ vertex_areas_mm2 The cortical suface area of each vertex in mesh in
       square mm.
     @ vmag_matrices A 3-tuple of (n x N, n x M, n x M) PyTorch sparse
       matrices, each of whihch corresponds to its matching entry in the
       vmag_indices element. These matrices are designed such that if one
       multiplies them each by a vector of the surface areas (i.e., in the
       visual field) of the faces described in vmag_indices then sums the three
       resulting vectors, thee resulting n-length vector will be the surface
       areas corresponding to each of the n vertices in mesh.
    '''
    dtype = to_torchdtype(dtype)
    edges = mesh.tess.indexed_edges.T
    faces = mesh.tess.indexed_faces.T
    # Figure out the face-pairs.
    (fp1,fp2,fp_edges) = np.transpose(
        [(k[0],k[1],ii) for (ii,k) in enumerate(mesh.tess.edge_faces) if len(k) == 2])
    fps = np.array([fp1,fp2])
    face_pairs = fps.T
    f1other = np.squeeze([faces[f1, ~np.isin(faces[f1], faces[f2])]
                          for (f1,f2) in face_pairs])
    f2other = np.squeeze([faces[f2, ~np.isin(faces[f2], faces[f1])]
                          for (f1,f2) in face_pairs])
    fp_opps = np.array([f1other, f2other])
    fps = face_pairs.T
    # Get the edge lengths and triangle surface areas from the midgray surface.
    els = totensor(mesh.edge_lengths, dtype=dtype, device=device)
    fas = totensor(mesh.face_areas, dtype=dtype, device=device)
    # Calculate the vmag summation matrix.
    
    # The best way to do this is to split each face up into six faces. Each corner gets two of
    # these subfaces.
    (a,b,c) = mesh.tess.indexed_faces
    n = mesh.tess.face_count
    (amtx, bmtx, cmtx) = [
        torch.sparse_coo_tensor(
            torch.tensor(np.array([corner, np.arange(n)]), dtype=torch.long),
            torch.ones(n),
            size=(mesh.vertex_count, n),
            dtype=dtype,
            device=device)
        for corner in (a,b,c)]
    matrices = (amtx, bmtx, cmtx)
    # Go ahead and use these to return the vertex surface areas.
    faces = totensor(np.array(faces.T), dtype=torch.long)
    edges = totensor(np.array(edges.T), dtype=torch.long)
    vas = vmag_vertex_areas(matrices, faces, mesh.coordinates)
    vas = totensor(vas, dtype=dtype, device=device)
    # Return everything.
    return (faces, edges, fps, fp_edges, fp_opps, els, fas, vas, matrices)
init_geometry_plan = pimms.plan(geometry=init_geometry)
@pimms.calc('roi_geometries', 'roi_cortical_surface_areas', 'boundary_edges')
def init_geometries(mesh, mesh_labels, roi_meshes, label_sort, dtype, device, torch):
    '''
    Calculates an init_geometry() imap for each roi-mesh. See init_geometry for
    more information.

    Efferent values:
      @ roi_geometries A persistent map of geometry data for each ROI in the
        optimization. See init_geometry for more information.
      @ roi_cortical_surface_areas A PyTorch tonsor, in label-sort order, of the
        cortical surface areas of each ROI in square mm.
      @ boundary_edges A persistent map whose keys are (area1, area2) tuples
        containing two adjacent visual area labels and whose keys are 2 x N
        numpy arrays (u, v) where u and v are vectors of vertex indices for
        edges that cross the relevant boundary.
    '''
    r = {}
    sas = []
    for (k,msh) in six.iteritems(roi_meshes):
        r[k] = init_geometry_plan(mesh=msh, dtype=dtype, device=device, torch=torch)
        sas.append(np.sum(msh.face_areas))
    sas = totensor(np.array(sas), dtype=dtype, device=device)
    # Make the edge boundaries
    (u,v) = mesh.tess.indexed_edges
    (lu, lv) = (mesh_labels[u], mesh_labels[v])
    ii = np.where((lu == 1) & (lv == 3))[0]
    jj = np.where((lu == 3) & (lv == 1))[0]
    # We only care about edges without that cross label boundaries.
    ii = lu != lv
    (u,v,lu,lv) = (u[ii], v[ii], lu[ii], lv[ii])
    # We also want the lower vertex label to come first:
    ii = lu > lv
    tmp = np.array(lu[ii])
    lu[ii] = lv[ii]
    lv[ii] = tmp
    tmp = np.array(u[ii])
    u[ii] = v[ii]
    v[ii] = tmp
    ebs = {}
    for k1 in label_sort:
        for k2 in label_sort:
            if k1 >= k2: continue
            ii = (lu == k1) & (lv == k2)
            tmp = np.array([u[ii], v[ii]])
            if tmp.shape[1] == 0: continue
            tmp.setflags(write=False)
            ebs[(k1, k2)] = tmp
    # We also want edges with the outer mesh; we store these as (u,u) because they have no
    # neighbor in the mesh.
    outer_vs = [xx
                for (ii,k) in enumerate(mesh.tess.edge_faces)
                if len(k) == 1
                for xx in mesh.tess.indexed_edges[:,ii]]
    outer_vs = np.unique(outer_vs)
    outer_ls = mesh_labels[outer_vs]
    for lbl in label_sort:
        ii = outer_ls == lbl
        uu = outer_vs[ii]
        if len(uu) == 0: continue
        tmp = np.array([uu, uu])
        tmp.setflags(write=False)
        ebs[(0,lbl)] = tmp
    return (pyr.pmap(r), sas, pyr.pmap(ebs))
init_subject_plan = pimms.plan(
    {'meas':       init_meas,
     'labels':     init_labels,
     'mesh':       init_mesh,
     'geometries': init_geometries})

# Ipsilateral Parameters ###########################################################################
@pimms.calc('uvm_ipsi_tan', 'lvm_ipsi_tan')
def init_ipsi_tan_params(gradients, dtype, device, grad_ipsi, label_sort, torch,
                         prior_uvm_ipsi=0.09, prior_lvm_ipsi=0.17):
    '''
    Initializes the "free" versions of the ipsilateral representation parameters
    uvm_ipisi_tan and lvm_ipsi_tan. Both are the radians of ipsilateral
    representation from the respective boundary (i.e., positive values are
    always the number of radians into the ipsilateral visual field that is
    represented in the retinotopic map. The free values are in fact the tangent
    of the values used in the model, preventing very extreme values or values
    beyond +/- pi/2 at all. Note that negative values indicate that the data do
    not cover the entire contralateral hemifield.

    Afferent parameters:
     @ prior_uvm_ipsi The prior value of the upper-vertical-meridian's
       ipsilateral representation. This value is the number of radians into the
       ipsilateral visual field that the PRF representations of all visual areas
       extend for the upper vertical meridian.
     @ prior_lvm_ipsi The prior value of the lower-vertical-meridian's
       ipsilateral representation. This value is the number of radians into the
       ipsilateral visual field that the PRF representations of all visual areas
       extend for the lower vertical meridian.

    Efferent values:
     @ uvm_ipsi_tan The tangent of the ipsilateral relresentattion angle for the
       upper visual field.
     @ lvm_ipsi_tan The tangent of the ipsilateral relresentattion angle for the
       lower visual field.
    '''
    uvm = astensor(0.0 if prior_uvm_ipsi is None else prior_uvm_ipsi)
    lvm = astensor(0.0 if prior_lvm_ipsi is None else prior_lvm_ipsi)
    if uvm.shape == (): uvm = uvm * torch.ones(len(label_sort))
    if lvm.shape == (): lvm = lvm * torch.ones(len(label_sort))
    for u in uvm:
        if u > np.pi/2 or u < -np.pi/2:
            raise ValueError("prior UVM ipsilateral angle must be between + and - pi/2")
    for l in lvm:
        if l > np.pi/2 or l < -np.pi/2:
            raise ValueError("prior LVM ipsilateral angle must be between + and - pi/2")
    uvm = torch.tan(uvm)
    lvm = torch.tan(lvm)
    uvm = totensor(uvm, dtype=dtype, device=device, requires_grad=grad_ipsi)
    lvm = totensor(lvm, dtype=dtype, device=device, requires_grad=grad_ipsi)
    if grad_ipsi:
        gradients['uvm_ipsi_tan'] = uvm
        gradients['lvm_ipsi_tan'] = lvm
    return (uvm, lvm)
@pimms.calc('uvm_ipsi', 'lvm_ipsi')
def calc_ipsi_params(uvm_ipsi_tan, lvm_ipsi_tan, torch):
    '''
    Calculates the ipsilateral parameters from the tan-parameters (i.e. converts
    the "free" parameters to limited parameters between +/- pi/2).

    Efferent values:
     @ uvm_ipsi The angle of extent into the ipsilateral hemifield of the
       representation of the upper vertical meridian in the model.
     @ lvm_ipsi The angle of extent into the ipsilateral hemifield of the
       representation of the lower vertical meridian in the model.
    '''
    return (torch.atan(uvm_ipsi_tan), torch.atan(lvm_ipsi_tan))
init_ipsi_plan = pimms.plan(ipsi_ten_params=init_ipsi_tan_params,
                            ipsi_params=calc_ipsi_params)


# Optimized Visual Field Coordinates ###############################################################
@pimms.calc('theta', 'eccen')
def init_visual_coords(meas_theta, meas_eccen, mesh, cortex, gradients, grad_coords, dtype, device,
                       start_coords=None):
    '''
    Initializes the visual field coordinates that get minimized as part of the
    optimization scheme.

    Afferent parameters:
      @ start_coords May specify the (2 x N) coordinate matrix of measurements
        to use as the starting point in the optimization. The N must be the
        number of vertices in the cortex; though values are only needed wherever
        there are non-zero labels. If None, then the measurements are used as
        the starting coordinates. The first row of the start coordinates must be
        the polar angle theta given in counter-clockwise radians starting from
        the right horizontal meridian, and the second row must be the
        eccentricity of the vertices in visual degrees.

    Efferent values:
      @ theta The polar angle values over which the optimization takes place.
      @ eccen The eccentricity values over which the optimizationn takes place.
    '''
    if start_coords is None:
        # Use the measurements
        tht = np.array(meas_theta.detach().numpy()[mesh.labels])
        ecc = np.array(meas_eccen.detach().numpy()[mesh.labels])
    else:
        viscoords = math.asarray(start_coords)
        if len(viscoords.shape) != 2: raise ValueError('start_coords must be a 2 x N matrix')
        (rs,cs) = viscoords.shape
        if rs != 2 and rs == cortex.vertex_count and cs == 2:
            viscoords = viscoords.T
        elif rs != 2 or cs != cortex.vertex_count:
            raise ValueError('start_coords must be 2 x N where N is number of cortex vetices')
        (tht,ecc) = [np.array(u) for u in viscoords[:, mesh.labels]]
    # Make this into a PyTorch tensor.
    tht = totensor(tht, dtype=dtype, device=device, requires_grad=grad_coords)
    ecc = totensor(ecc, dtype=dtype, device=device, requires_grad=grad_coords)
    # Add this tensor to the gradients if it's one of them.
    if grad_coords: 
        gradients['theta'] = tht
        gradients['eccen'] = ecc
    return (tht, ecc)
@pimms.calc('x', 'y')
def calc_cartesian(theta, eccen, torch):
    '''
    Converts polar angle and eccentricity into x and y.

    Efferent values:
      @ x The x coordinate of each vertex in the visual field.
      @ y The y coordinate of each vertex in the visual field.
    '''
    return (torch.cos(theta) * eccen, torch.sin(theta) * eccen)
@pimms.calc('logeccen', 'logx', 'logy')
def calc_logcartesian(theta, eccen, torch):
    '''
    Converts polar angle and eccentricity into log-x and log-y.

    Efferent values:
      @ logx The x coordinate of each vertex in the visual field, scaled such
        that the eccentricity of each point is log(0.75 + eccen) - log(0.75).
      @ logy The y coordinate of each vertex in the visual field, scaled such
        that the eccentricity of each point is log(0.75 + eccen) - log(0.75).
    '''
    logecc = torch.log(0.75 + eccen) - log(0.75)
    return (torch.cos(theta) * logeccen, torch.sin(theta) * logeccen)
coords_plan = pimms.plan(visual_coords=init_visual_coords,
                         #logcartesian=calc_logcartesian,
                         cartesian=calc_cartesian)

# Cortical Magnification ###########################################################################
@pimms.calc('face_areas_deg2', 'vmag_faces')
def calc_vmag_faces(faces, face_areas_mm2, x, y):
    '''
    Calculation:
      (faces, edges, face_areas_mm2, x, y)
      ==> (face_areas_deg2, vmag_faces)
    
    Efferent values:
      @ face_areas_deg2 The surface area of each face in the visual field.
      @ vmag_faces The visual magnification of each face.
    '''
    (a,b,c) = faces
    (xa, ya) = (x[a], y[a])
    xab = x[b] - xa 
    yab = y[b] - ya
    xac = x[c] - xa 
    yac = y[c] - ya
    fareas = (xab*yac - yab*xac) / 2.0
    fvmags = fareas / face_areas_mm2
    return (fareas, fvmags)
@pimms.calc('vertex_areas_deg2', 'vmag')
def calc_vmag_vertices(faces, vertex_areas_mm2, vmag_matrices, x, y, torch):
    '''
    Calculation:
      (faces, vertex_areas_mm2, vmag_matrices, x, y)
      ==> (vertex_areas_deg2, vmag)
    
    Efferent values:
      @ vertex_areas_deg2 The surface area of each vertex in the visual field.
      @ vmag The visual magnification of each vertex.
    '''
    # Calculate the vertex areas.
    coords = torch.stack([x,y])
    vareas = vmag_vertex_areas(vmag_matrices, faces, coords)
    # The rest is easy.
    return (vareas, vareas / vertex_areas_mm2)
vmag_mesh_plan = pimms.plan(vmag_vertices=calc_vmag_vertices, vmag_faces=calc_vmag_faces)
@pimms.calc('roi_vmags')
def calc_roi_vmags(roi_meshindices, roi_geometries, x, y, torch):
    '''
    Calculates an imap of vmag data for each roi in the optimization.

    Efferent values:
      @ roi_vmags A persistent map whose keys are ROI labels and whose values
        are imaps obtained from running the vmag_mesh_plan on each ROI mesh.
    '''
    from neuropythy.util import curry
    runplan = lambda k: vmag_mesh_plan(
        faces=roi_geometries[k]['faces'],
        edges=roi_geometries[k]['edges'],
        face_areas_mm2=roi_geometries[k]['face_areas_mm2'],
        vertex_areas_mm2=roi_geometries[k]['vertex_areas_mm2'],
        vmag_matrices=roi_geometries[k]['vmag_matrices'],
        x=x[roi_meshindices[k].copy()],
        y=y[roi_meshindices[k].copy()],
        torch=torch)
    res = pimms.lmap({k: curry(runplan, k) for k in roi_geometries.keys()})
    return (res,)
vmag_plan = pimms.plan(roi_vmags=calc_roi_vmags)

# Cortical Magnification Model #####################################################################
def cmmdl_tanmult_rescale(theta, hemifield_sign, ui=0.09, li=0.17):
    '''
    Yields the theta parameter rescaled such that the upper vertical field ends
    ui radians into the ipsilateral visual field and such that the lower vertical
    field ends li radians into the ipsilateral visual field. The returned value
    is scaled such that a value of pi/2 + ui in the input theta has a value pi/2
    in the output theta, and equivalently with -pi/2 and li in the lower visual
    field. The x-value sign of the contralateral visual field is the second
    argument to this function (and the signs in the above explanation are
    reversed when this argument is -1).
    '''
    torch = pytorch()
    (hpi, tau) = (math.half_pi, math.tau)
    if hemifield_sign == -1: theta = torch.remainder(tau - theta, tau) - hpi
    th = torch.zeros(theta.shape)
    gt = theta > 0
    lt = ~gt
    th[gt] = theta[gt] / (hpi + ui) * hpi
    th[lt] = theta[lt] / (hpi + li) * hpi
    if hemifield_sign == -1: th = torch.remainder(tau - th, tau) - hpi
    return th
def cmmdl_tanmult_sinusoid(theta, hva=0.5, vma=0.5):
    '''
    Yields the angular cortical magnification multiplier based on the polar
      angle theta, which is measured in counter-clockwise radians from the RHM.
    
    The sinusoid version of this function yields a polar angle multiplier based
    on the sum of sinusoids.
    '''
    torch = pytorch()
    hvpart = hva * torch.cos(2 * theta)
    thsin  = torch.sin(theta)
    ulpart = vma * torch.sign(thsin) * thsin**2
    return 1.0 + 0.5*(hvpart - ulpart)
def cmmdl_tanmult_beta(theta, a=1.0, b=1.0):
    '''
    Yields the angular cortical magnification multiplier based on the polar
      angle theta, which is measured in counter-clockwise radians from the RHM.
    
    The beta-distribution version of this function uses a beta distribution, 
    spread across -pi to pi.
    '''
    torch = pytorch()
    thuvm = torch.remainder(math.half_pi - theta + np.pi, math.tau) - np.pi
    thuvm = torch.abs(thuvm)
    theta = torch.remainder(math.half_pi - thuvm + np.pi, math.tau) - np.pi
    theta = (theta + math.half_pi) / np.pi
    beta = torch.distributions.Beta(astensor(a), astensor(b))
    return torch.exp(beta.log_prob(theta))
def cmmdl_hhcmag(eccen, c1=17.3, c2=0.75):
    '''
    cmmdl_hhcmag(eccen) yields the linear radial cortical magnification using
      the Horton and Hoyt (1991) formula.
    '''
    return c1 / (c2 + eccen)
def cmmdl_hhcmag2(eccen, c1=17.3, c2=0.75):
    '''
    cmmdl_hhcmag2(eccen) yields the areal radial cortical magnification using
      the Horton and Hoyt (1991) formula.
    '''
    return cmmdl_hhcmag(eccen, c1=c1, c2=c2)**2
def cmmdl_cmag2_beta(theta, eccen,
                     c1=17.3, c2=0.75, a=1.0, b=1.0):
    '''
    Yields the areal cortical magnification prediction for the given polar
      angle and eccentricity using both the Horton and Hoyt (1991) equation
      as well as the tangential cortical magnification multiplier, based on a
      beta distribution.
    '''
    rmag = cmmdl_hhcmag2(eccen, c1=c1, c2=c2)
    tmlt = cmmdl_tanmult_beta(theta, a=a, b=b)
    return rmag * tmlt
def cmmdl_cmag2_sinusoid(theta, eccen,
                         c1=17.3, c2=0.75, hva=0.5, vma=0.5):
    '''
    Yields the areal cortical magnification prediction for the given polar
      angle and eccentricity using both the Horton and Hoyt (1991) equation
      as well as the tangential cortical magnification multiplier, base on a sum
      of sinusoids.
    '''
    rmag = cmmdl_hhcmag2(eccen, c1=c1, c2=c2)
    tmlt = cmmdl_tanmult_sinusoid(theta, hva=hva, vma=vma)
    return rmag * tmlt
def cmmdl_tan_inparams(method, params):
    torch = pytorch()
    if method is None:
        return params
    elif method == 'sinusoid':
        return torch.atan(astensor(params)) * 2 / np.pi
    elif method == 'beta':
        return torch.exp(torch.atan(astensor(params)) * 3 / math.half_pi)
    else:
        raise ValueError("unknown cmag2 tangential method: %s" % (method,))
def cmmdl_tan_defaultparams(method):
    if method is None:
        return None
    elif method == 'sinusoid':
        return (0.5, 0.5)
    elif method == 'beta':
        return (1.0, 1.0)
    else:
        raise ValueError("unknown cmag2 tangential method: %s" % (method,))
def cmmdl_tan_outparams(method, params):
    torch = pytorch()
    if method is None:
        return params
    elif method == 'sinusoid':
        return torch.tan(astensor(params) * math.half_pi)
    elif method == 'beta':
        return torch.tan(torch.log(astensor(params)) / 3 * math.half_pi)
    else:
        raise ValueError("unknown cmag2 tangential method: %s" % (method,))
def cmmdl_cmag2(theta, eccen, hemifield_sign, method='sinusoid',
                params=[0.0, 0.0], c1=17.3, c2=0.75, ui=0.09, li=0.17):
    '''
    Yields the areal cortical magnification using either cmmdl_cmag2_sinusoid,
    with parameters (hva, vma) and method name 'sinusoid', or cmmdl_cmag2_beta,
    with parameters (a, b). 
    '''
    #th = cmmdl_tanmult_rescale(theta, hemifield_sign, ui=ui, li=li)
    th = theta
    if method is None:
        return cmmdl_hhcmag2(eccen, c1=c1, c2=c2)
    elif method == 'sinusoid':
        (hva, vma) = params
        return cmmdl_cmag2_sinusoid(th, eccen, c1=c1, c2=c2, hva=hva, vma=vma)
    elif method == 'beta':
        torch = pytorch()
        (a, b) = params
        return cmmdl_cmag2_beta(th, eccen, c1=c1, c2=c2, a=a, b=b)
    else:
        raise ValueError("unknown cmag2 tangential method: %s" % (method,))
def cmmdl_c1(area, max_eccen, c2):
    '''
    Yields the value of c1 in the cmmdl that is appropriate for a visual area
    with the given surface-area, within the given max_eccen central degrees,
    and the parameter c2.
    '''
    torch = pytorch()
    c2_maxecc = c2 + max_eccen
    den = np.pi * (torch.log(c2_maxecc / c2) - max_eccen / c2_maxecc)
    return math.safesqrt(area / den) # #TODO is the safesqrt necessary here?
@pimms.calc('cmag_eccen_logoffsets', 'fieldsigns_tensor', 'cmag_tan_params_free')
def init_cmag_params(dtype, device, gradients, torch, label_sort, grad_cmag_rad, grad_cmag_tan,
                     fieldsigns=Ellipsis,
                     cmag_tan_method='sinusoid',
                     prior_cmag_eccen_offsets=0.75,
                     prior_cmag_tan_params=None):
    '''
    Initializes the model parameters required to calculate the cortical
    magnification of each vertex and/or visual field position. This
    calculation initializes the "free" versions of the cmag_eccen_offsets
    parameters--i.e., the version that can range from -inf to inf but that gets
    limited into a particular range for the actual model calculation.
    
    Afferent parameters:
     @ prior_cmag_eccen_offsets The initial value of the offset parameter c2 
       from Horton and Hoyt's (1991) equation cmag(ecc) = c1 / (c2 + ecc).
       This should be either a single number, which is used for all visual
       areas, or a dictionary of visual area labels mapped to values.
     @ fieldsigns The fieldsign values for each visual area. This may either be
       a dictionary whose keys are visual area labels and whose values are all
       either 1 or -1, or it can be Ellipsis or None. If the value is Ellipsis
       (the default), then the dictionary visual_area_field_signs from the
       neuropythy.vision package is used. If the value is None, then even
       labels are given fieldsigns of 1 and odd labels are given fieldsigns of
       -1.
     @ cmag_tan_method The method to use for calculating tangential cortical
       magnification. This may be either 'sinusoid' (the default), 'beta', or
       None, in which case tangential cortical magnification is modeled to be
       uniform across polar angle. For thee sinusoid, the parameters are tan_hva
       and tan_vma where the actual hva and vma are arctan(tan_hva)*2/pi and
       arctan(tan_vma)*2/pi, limiting them both to the +/- 1 range. For the
       'beta' method, the params are loga and logb, where the beta-distribution
       parameters that are used are simply a = exp(loga) and b = exp(logb).
     @ prior_cmag_tan_params The initial values of the tangential cortical
       magnificatiioin model parametters, whose interpretation depends on the
       value given for cmag_tan_method (see it for more details). Note that
       cmag_tan_method describes parameters such as loga, which are used in the
       model via the transformation a = exp(loga). The value that is given for
       the prior should *not* be the transformed param--i.e., provide a not loga
       as the prioir value.

    Efferent values:
     @ cmag_eccen_logoffsets The log of the cortical magnification offset
       parameter. As a free parameter, the value may be any real number; the
       version used in the cortical magnification model is the exp() of this
       value.
     @ fieldsigns_tensor A tensor of the fieldsigns of each visual area in the
       order provided by label_sort. Each of these values must be either 1 or
       -1.
     @ cmag_tan_params_free The free (untransformed) matrix of cmag tangential
       parameters. The matrix is N x D where N is the number of visual areas and
       D is the number of parameters (typically 2).
    '''
    # The c2 parameter.
    dtype = to_torchdtype(dtype)
    pri = prior_cmag_eccen_offsets
    log_de = totensor(np.zeros(len(label_sort)),
                      dtype=dtype, device=device, requires_grad=False)
    if pimms.is_map(pri):
        for (ii,k) in enumerate(label_sort):
            log_de[ii] = pri[k]
    elif torch.is_tensor(pri) or pimms.is_number(pri):
        log_de[:] = pri
    else:
        raise ValueError("Could not understand prior_cmag_eccen_offsets param")
    for x in log_de:
       if not torch.isfinite(x) or x <= 0:
           raise ValueError('prior_cmag_eccen_offsets values must be >= 0')
    log_de = torch.log(log_de)
    if grad_cmag_rad:
        log_de = log_de.clone().detach().requires_grad_(True)
        gradients['cmag_eccen_logoffsets'] = log_de
    # Fieldsigns.
    if fieldsigns is Ellipsis:
        from neuropythy.vision import visual_area_field_signs
        fieldsigns = visual_area_field_signs
    elif fieldsigns is None:
        fieldsigns = {k: (-1 if (k % 2) == 1 else 1) for k in label_sort}
    fieldsigns = [fieldsigns[k] for k in label_sort]
    for fs in fieldsigns:
        if fs != 1 and fs != -1:
            raise ValueError('fieldsign values must be either 1 or -1')
    fieldsigns = totensor(fieldsigns, dtype=dtype, device=device)
    # The tangential magnification parameters.
    if cmag_tan_method is None:
        tanparams = None
    else:
        if cmag_tan_method == 'beta':
            import warnings
            warnings.warn("cmag_tan_method='beta' is unfinished and frequently results in NaNs")
        tanparams = torch.zeros((len(label_sort), 2), dtype=dtype, device=device)
        pri = prior_cmag_tan_params
        if pri is None: pri = cmmdl_tan_defaultparams(cmag_tan_method)
        if pimms.is_map(pri):
            for (ii,k) in enumerate(label_sort):
                tanparams[ii,:] = pri[k]
        else:
            tanparams[:] = astensor(pri)
        tanparams = cmmdl_tan_outparams(cmag_tan_method, tanparams)
        if grad_cmag_tan:
            tanparams = tanparams.clone().detach().requires_grad_(True)
            gradients['cmag_tan_params_free'] = tanparams
    # Return.
    return (log_de, fieldsigns, tanparams)
@pimms.calc('cmag_eccen_offsets', 'cmag_tan_params')
def calc_cmag_params(torch, cmag_eccen_logoffsets, cmag_tan_method, cmag_tan_params_free):
    '''
    Calculates the model parameters required to calculate the cortical
    magnification of each vertex and/or visual field position. This
    calculation involves running the free versions of each pameter through the
    arctangent function in order to convert an infinite range to the range
    (0,1) then linearly expanding that to the parameter's limit.
    
    Efferent values:
     @ cmag_eccen_offsets The offset c2 in Horton and Hoyt's 1991 cortical
       magnification equation cmag(eccen) = c1/(c2 + eccen). This value is
       derived from the cmag_eccen_logoffset, which is the actual parameter
       over which minimization occurs. This is because any real value of the
       log-offset results in a valid value for the offset, preventing the offset
       from derailing the search.
     @ cmag_tan_params The "unfree" version of the cmag_tan_params_free. This is
       a matrix PyTorch-tensor of values where each row is a set of parameters
       for the tangential cortical magnification model (see cmag_tan_method for
       addititonal information).
    '''
    return (torch.exp(cmag_eccen_logoffsets),
            cmmdl_tan_inparams(cmag_tan_method, cmag_tan_params_free))
@pimms.calc('cmag_scale')
def calc_cmag_scale(eccen, label_sort, cmag_eccen_offsets,
                    max_eccen, roi_cortical_surface_areas, dtype):
    '''
    Calculates the 'cmag_scale' meta-parameter for each visual area.

    Afferent parameters:
     @ max_eccen The maximum eccentricity in the optimized maps. This should be
       the ecceentricity of the peripheral boundary of the labeled maps.
    
    Efferent values:
     @ cmag_scale The meta-parameter (c1) that can be calculated from the
       'cmag_eccen_offsets' (c2) parameter and the total visual area size,
       based on the fact that the cortical magnification used in the model
       conforms to Horton and Hoyt's (1991) equation:
       cmag2(ecc) = (c1 / (c2 + ecc))^2.
       The calculation of cmag_scale uses the surface area of the region of
       each visual area that is within max_stim_eccen to calculate the
       parameter value.
    '''
    r = cmmdl_c1(roi_cortical_surface_areas, max_eccen, cmag_eccen_offsets)
    return (r,)
@pimms.calc('model_vmags')
def calc_model_vmags(theta, eccen, fieldsigns_tensor, hemifield_sign,
                     cmag_eccen_offsets, cmag_scale, cmag_tan_method, cmag_tan_params,
                     uvm_ipsi, lvm_ipsi, roi_meshindices, label_sort,
                     dtype, device, torch):
    '''
    Calculates the per-visual-area visual magnification values as predicted by
    the cortical magnification model.
      
    Efferent values:
      @ model_vmags A persistent map whose keys are visual area labels and whose
        values are tensors of the visual magnification values predicted at the
        vertices of each ROI-mesh.
    '''
    c1s = cmag_scale
    c2s = cmag_eccen_offsets
    tps = cmag_tan_params
    vmags = {}
    for (ii,k) in enumerate(label_sort):
        roi_ii = np.array(roi_meshindices[k])
        tht = theta[roi_ii]
        ecc = eccen[roi_ii]
        cm = cmmdl_cmag2(tht, ecc, hemifield_sign,
                         c1=c1s[ii], c2=c2s[ii], params=tps[ii],
                         method=cmag_tan_method, ui=uvm_ipsi[ii], li=lvm_ipsi[ii])
        vmags[k] = fieldsigns_tensor[ii] / cm
    return (pyr.pmap(vmags),)
@pimms.calc('model_vmag_faces')
def calc_model_vmag_faces(theta, eccen, fieldsigns_tensor, hemifield_sign,
                          cmag_eccen_offsets, cmag_scale, cmag_tan_method, cmag_tan_params,
                          uvm_ipsi, lvm_ipsi, roi_meshindices, roi_meshes,
                          label_sort, dtype, device, torch):
    '''
    Calculates the per-visual-area visual magnification values as predicted by
    the cortical magnification model for the faces in the mesh.
      
    Efferent values:
      @ model_vmag_faces A persistent map whose keys are visual area labels and
        whose values are tensors of the visual magnification values predicted at
        the triangles of each ROI-mesh.
    '''
    c1s = cmag_scale
    c2s = cmag_eccen_offsets
    tps = cmag_tan_params
    vmags = {}
    for (ii,k) in enumerate(label_sort):
        roi_ii = np.array(roi_meshindices[k])
        tht = theta[roi_ii]
        ecc = eccen[roi_ii]
        (a,b,c) = np.array(roi_meshes[k].tess.indexed_faces)
        tht = (tht[a] + tht[b] + tht[c]) / 3.0
        ecc = (ecc[a] + ecc[b] + ecc[c]) / 3.0
        cm = cmmdl_cmag2(tht, ecc, hemifield_sign,
                         c1=c1s[ii], c2=c2s[ii], params=tps[ii],
                         method=cmag_tan_method, ui=uvm_ipsi, li=lvm_ipsi)
        vmags[k] = fieldsigns_tensor[ii] / cm
    return (pyr.pmap(vmags),)
model_plan = pimms.plan(init_cmag_params=init_cmag_params,
                        cmag_params=calc_cmag_params,
                        cmag_scale=calc_cmag_scale,
                        model_vmags=calc_model_vmags,
                        model_vmag_faces=calc_model_vmag_faces)

# Model Loss #######################################################################################
@pimms.calc('meas_stddev', 'meas_log_2stddev')
def calc_meas_stddev(torch, meas_eccen, meas_cod, meas_weight=None):
    '''
    Calculates the standard deviation of the measurement distributions that are
    used to calculate the likelihood of the measurements given the data.

    The standard deviation formula for a vertex u is as follows:
      stddev(u) = (0.75 + meas_eccen) / (0.25 + 3.75 * min_cod) / weight

    Afferent parameters:
     @ meas_weight A vector of weights, one per vertex in mesh, that specifies
       how strongly the model should believe that particular vertex's weight.
       Weights are the divisor for the meas_stddev. See calc_meas_stddev for
       more information about the standard deviation formula.
    '''
    if meas_weight is None: meas_weight = 1.0
    stddev = (0.75 + meas_eccen) / (0.25 + 3.75*meas_cod) / meas_weight
    return (stddev, torch.log(2.0 * stddev))
@pimms.calc('meas_likelihood', 'meas_likelihoods')
def calc_meas_likelihood(mesh, meas_x, meas_y, x, y, meas_stddev, meas_log_2stddev, torch):
    '''
    Calculates the measurement likelihood, given the retinotopy model, of the
    data. This is the sum over vertices of the log probabilities of a laplacian
    noise distribution around the predicted value. The standard devations of
    these distributions is meas_stddev.

    Efferent values:
     @ meas_likelihood The log likelihood of the PRF measurements given the PRF
       model.
     @ meas_likelihoods A PyTorch vector of log likelihood values, one per mesh
       vertex, of the measurements given the model.
    '''
    ii = mesh.labels.copy()
    meas_x = meas_x[ii]
    meas_y = meas_y[ii]
    meas_stddev = meas_stddev[ii]
    meas_log_2stddev = meas_log_2stddev[ii]
    dist = math.safesqrt((meas_x - x)**2 + (meas_y - y)**2)
    llhoods = -dist/meas_stddev # - meas_log_2stddev
    return (torch.sum(llhoods), llhoods)
@pimms.calc('vmag_likelihood', 'vmag_likelihoods')
def calc_vmag_likelihood(roi_vmags, model_vmag_faces, label_sort, 
                         dtype, device, torch):
    '''
    Calculates the log likelihood of the visual magnification model.

    Efferent values:
     @ vmag_likelihood The log-likelihood of the vmag component of the model.
     @ vmag_likelihoods A persistent map whose keys are visual area labels and
       whose values are PyTorch vectors of the log likelihood
    '''
    r = {}
    tot = totensor(0.0, dtype=dtype, device=device)
    for k in label_sort:
        #coord_vm = torch.log(0.5 + roi_vmags[k]['vmag'])
        #model_vm = torch.log(0.5 + model_vmags[k])
        coord_vm = roi_vmags[k]['vmag_faces']
        model_vm = model_vmag_faces[k]
        llhood = -(coord_vm - model_vm)**2
        r[k] = llhood
        tot = tot + torch.sum(llhood)
    return (tot, pyr.pmap(r))
visual_area_boundary_fns = pyr.m(
    VM=       lambda t,e,x,y,ui,li,mxec: math.branch(pytorch().sin(t) > 0, t - ui, t - li),
    HM=       lambda t,e,x,y,ui,li,mxec: y,
    periphery=lambda t,e,x,y,ui,li,mxec: e - mxec,
    fovea=    lambda t,e,x,y,ui,li,mxec: e)
visual_area_boundaries = pyr.pmap(
    {(1, 2): 'VM',
     (0, 1): 'periphery',
     (2, 3): 'HM',
     (0, 2): 'periphery',
     (0, 3): ('periphery', 'VM')})
def boundary_distances(lbl1, lbl2, theta, eccen, x, y, uvm_ipsi, lvm_ipsi, max_eccen,
                       boundaries=visual_area_boundaries,
                       boundary_fns=visual_area_boundary_fns):
    '''
    Calculates the square of the boundary distances for the given values.
    '''
    torch = pytorch()
    vab = boundaries.get((lbl1, lbl2), None)
    if vab is None: return None
    if is_tuple(vab):
        if len(vab) == 0: return None
        r = [boundary_fns[subvab](theta, eccen, x, y, uvm_ipsi, lvm_ipsi, max_eccen)**2
             for subvab in vab]
        return torch.min(*r)
    else:
        return boundary_fns[vab](theta, eccen, x, y, uvm_ipsi, lvm_ipsi, max_eccen)**2
@pimms.calc('boundary_likelihood', 'boundary_likelihoods')
def calc_boundary_likelihood(x, y, theta, eccen, boundary_edges, uvm_ipsi, lvm_ipsi, max_eccen,
                             hemifield_sign, label_sort, torch):
    '''
    Calculates the likelihood of the visual area boundaries given the model.

    Efferent values:
     @ boundary_likelihood The total log likelihood of all the boundary edges.
     @ boundary_likelihoods The log likelihood of each of the boundary edges.
    '''
    uvm_ipsi = math.half_pi + hemifield_sign*uvm_ipsi
    lvm_ipsi = -math.half_pi - hemifield_sign*lvm_ipsi
    llhs = {}
    llhtot = 0.0
    for (k, uv) in six.iteritems(boundary_edges):
        (u,v) = uv.copy()
        (k1,k2) = k
        llh = llh0 = 0
        for (kk,ii) in zip([k2] if k1 == 0 else [k1,k2],
                           [u]  if k1 == 0 else [u,v]):
            ii = ii.copy()
            kk = np.where(label_sort == kk)[0][0]
            bds = boundary_distances(k1, k2, theta[ii], eccen[ii], x[ii], y[ii],
                                     uvm_ipsi[kk], lvm_ipsi[kk], max_eccen)
            if bds is None: continue
            llh = llh - bds
        if k1 != 0:
            llh = llh - ((x[u] - x[v])**2 + (y[u] - y[v])**2)
        if llh is llh0: continue
        llhs[k] = llh
        llhtot += torch.sum(llh)
    return (llhtot, pyr.pmap(llhs))
@pimms.calc('likelihood')
def calc_likelihood(dtype, device, mesh, vmag_likelihood, meas_likelihood, boundary_likelihood,
                    model_knob=12, boundary_knob=2):
    '''
    Calculates the likelihood of the model given the measurements.

    Afferent parameters:
     @ model_knob The base-2 log of the constant weight that the vmag_likelihood
       is multiplied by prior to summation with meas_likelihood.
     @ boundary_knob The base-2 log of the constant weight that the
       boundary_likelihood is multiplied by prior to summation with
       meas_likelihood.

    Efferent values:
      @ likelihood The likelihood of the measurements given the model times the
        likelihood of the model.
    '''
    two = astensor(2.0, dtype=dtype, device=device)
    wv = 0 if model_knob is None else two ** model_knob
    wb = 0 if boundary_knob is None else two ** boundary_knob
    return ((meas_likelihood + wv*vmag_likelihood + wb*boundary_likelihood),)
likelihood_plan = pimms.plan(meas_stddev=calc_meas_stddev,
                             meas_likelihood=calc_meas_likelihood,
                             vmag_likelihood=calc_vmag_likelihood,
                             boundary_likelihood=calc_boundary_likelihood,
                             likelihood=calc_likelihood)

# Overall plan
prfclean_plan = pimms.plan(init_params_plan,
                           init_subject_plan,
                           init_ipsi_plan,
                           coords_plan,
                           vmag_plan,
                           model_plan,
                           likelihood_plan)

                           
