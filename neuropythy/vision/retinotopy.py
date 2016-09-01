####################################################################################################
# neuropythy/vision/retinotopy.py
# Tools for registering the cortical surface to a particular potential function
# By Noah C. Benson

import numpy                        as np
import scipy                        as sp
import nibabel.freesurfer.io        as fsio
import nibabel.freesurfer.mghformat as fsmgh

import os, sys, gzip

from numpy.linalg import norm
from math         import pi
from numbers      import Number
from pysistence   import make_dict

from neuropythy.cortex       import (CorticalMesh)
from neuropythy.freesurfer   import (freesurfer_subject, add_subject_path,
                                     cortex_to_ribbon, cortex_to_ribbon_map,
                                     Hemisphere, subject_paths)
from neuropythy.topology     import (Registration)
from neuropythy.registration import (mesh_register)

from .models import (RetinotopyModel, SchiraModel, RetinotopyMeshModel, RegisteredRetinotopyModel,
                     load_fmm_model)

# Tools for extracting retinotopy data from a subject:
_empirical_retinotopy_names = {
    'polar_angle':  ['prf_polar_angle',       'empirical_polar_angle',  'measured_polar_angle',
                     'training_polar_angle',  'polar_angle'],
    'eccentricity': ['prf_eccentricity',      'empirical_eccentricity', 'measured_eccentricity',
                     'training_eccentricity', 'eccentricity'],
    'weight':       ['prf_weight',       'prf_variance_explained',       'prf_vexpl',
                     'empirical_weight', 'empirical_variance_explained', 'empirical_vexpl',
                     'measured_weight',  'measured_variance_explained',  'measured_vexpl',
                     'training_weight',  'training_variance_explained',  'training_vexpl',
                     'weight',           'variance_explained',           'vexpl']}

# handy function for picking out properties automatically...
def empirical_retinotopy_data(hemi, retino_type):
    '''
    empirical_retinotopy_data(hemi, t) yields a numpy array of data for the given hemisphere object
    and retinotopy type t; it does this by looking at the properties in hemi and picking out any
    combination that is commonly used to denote empirical retinotopy data. These common names are
    stored in _empirical_retintopy_names, in order of preference, which may be modified.
    The argument t should be one of 'polar_angle', 'eccentricity', 'weight'.
    '''
    dat = _empirical_retinotopy_names[retino_type.lower()]
    hdat = {s.lower(): s for s in hemi.property_names}
    return next((hemi.prop(hdat[s]) for s in dat if s.lower() in hdat), None)

_predicted_retinotopy_names = {
    'polar_angle':  ['predicted_polar_angle',   'model_polar_angle',
                     'registered_polar_angle',  'template_polar_angle'],
    'eccentricity': ['predicted_eccentricity',  'model_eccentricity',
                     'registered_eccentricity', 'template_eccentricity'],
    'visual_area':  ['predicted_visual_area',   'model_visual_area',
                     'registered_visual_area',  'template_visual_area']}

def predicted_retinotopy_data(hemi, retino_type):
    '''
    predicted_retinotopy_data(hemi, t) yields a numpy array of data for the given hemisphere object
    and retinotopy type t; it does this by looking at the properties in hemi and picking out any
    combination that is commonly used to denote empirical retinotopy data. These common names are
    stored in _predicted_retintopy_names, in order of preference, which may be modified.
    The argument t should be one of 'polar_angle', 'eccentricity', 'visual_area'.
    '''
    dat = _predicted_retinotopy_names[retino_type.lower()]
    hdat = {s.lower(): s for s in hemi.property_names}
    return next((hemi.prop(hdat[s]) for s in dat if s.lower() in hdat), None)

_retinotopy_names = {
    'polar_angle':  set(['polar_angle']),
    'eccentricity': set(['eccentricity']),
    'visual_area':  set(['visual_area']),
    'weight':       set(['weight', 'variance_explained'])}

def retinotopy_data(hemi, retino_type):
    '''
    retinotopy_data(hemi, t) yields a numpy array of data for the given hemisphere object
    and retinotopy type t; it does this by looking at the properties in hemi and picking out any
    combination that is commonly used to denote empirical retinotopy data. These common names are
    stored in _predicted_retintopy_names, in order of preference, which may be modified.
    The argument t should be one of 'polar_angle', 'eccentricity', 'visual_area', or 'weight'.
    Unlike the related functions empirical_retinotopy_data and predicted_retinotopy_data, this
    function calls both of these (predicted first then empirical) in the case that it does not
    find a valid property.
    '''
    dat = _retinotopy_names[retino_type.lower()]
    val = next((hemi.prop(s) for s in hemi.property_names if s.lower() in dat), None)
    if val is None and retino_type.lower() != 'weight':
        val = predicted_retinotopy_data(hemi, retino_type)
    if val is None and retino_type.lower() != 'visual_area':
        val = empirical_retinotopy_data(hemi, retino_type)
    return val

def extract_retinotopy_argument(obj, retino_type, arg, default='any'):
    '''
    extract_retinotopy_argument(o, retino_type, argument) yields retinotopy data of the given
    retinotopy type (e.g., 'polar_angle', 'eccentricity', 'variance_explained', 'visual_area',
    'weight') from the given hemisphere or cortical mesh object o, according to the given
    argument. If the argument is a string, then it is considered a property name and that is
    returned regardless of its value. If the argument is an iterable, then it is returned. If
    the argument is None, then retinotopy will automatically be extracted, if found, by calling
    the retinotopy_data function.
    The option default (which, by default, is 'any') specifies which function should be used to
    extract retinotopy in the case that the argument is None. The value 'any' indicates that the
    function retinotopy_data should be used, while the values 'empirical' and 'predicted' specify
    that the empirical_retinotopy_data and predicted_retinotopy_data functions should be used,
    respectively.
    '''
    if isinstance(arg, basestring): values = obj.prop(arg)
    elif hasattr(arg, '__iter__'):  values = arg
    elif arg is not None:           raise ValueError('cannot interpret retinotopy arg: %s' % arg)
    elif default == 'predicted':    values = predicted_retinotopy_data(obj, retino_type)
    elif default == 'empirical':    values = empirical_retinotopy_data(obj, retino_type)
    elif default == 'any':          values = retinotopy_data(obj, retino_type)
    else:                           raise ValueError('bad default retinotopy: %s' % default)
    if values is None:
        raise RuntimeError('No %s retinotopy data found given argument: %s' % (retino_type, arg))
    n = obj.vertex_count
    if len(values) != n:
        raise RuntimeError('Given %s data has incorrect length (%s instead of %s)!' \
                           % (retino_type, len(values), n))
    return np.asarray(values)

# Tools for retinotopy model loading:
__loaded_V123_models = {}
_V123_model_paths = [
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'lib', 'models')]
def V123_model(name='standard', radius=pi/3.0, sphere_radius=100.0, search_paths=None, update=False):
    '''
    V123_model() yields a standard retinotopy model of V1, V2, and V3. The model itself is a set of
    meshes with values at the vertices that define the polar angle and eccentricity. These meshes
    are loaded from files in the neuropythy lib directory. The model's class is
    RegisteredRetinotopyModel, so the details of the model's 2D projection onto the cortical surface
    are included in the model.
    
    The following options may be given:
      * name (default: 'standard') indicates the name of the model to load; the standard model is
        included with the neuropythy library. If name is a filename, this file is loaded (must be a
        valid fmm file). Currently no other named models are supported.
      * radius, sphere_radius (defaults: pi/3 and 100, respectively) specify the radius of the
        projection (on the surface of the sphere) and the radius of the sphere (100 is the radius
        for Freesurfer spheres). See neuropythy.registration.load_fmm_model for mode details.
      * search_paths (default: None) specifies directories in which to look for fmm model files. No
        matter what is included in these files, the neuropythy library's folders are searched last.
    '''
    if name in __loaded_V123_models:
        return __loaded_V123_models[name]
    if os.path.isfile(name):
        fname = name
        name = None
    else:
        if len(name) > 4 and name[-4:] == '.fmm':
            fname = name
            name = name[:-4]
        elif len(name) > 7 and name[-7:] == '.fmm.gz':
            fname = name
            name = name[:-7]
        else:
            fname = name + '.fmm'
        # Find it in the search paths...
        spaths = (search_paths if search_paths is not None else []) + _V123_model_paths
        fname = next(
            (os.path.join(path, nm0)
             for path in spaths
             for nm0 in os.listdir(path)
             for nm in [nm0[:-4] if len(nm0) > 4 and nm0[-4:] == '.fmm'    else \
                        nm0[:-7] if len(nm0) > 7 and nm0[-7:] == '.fmm.gz' else \
                        None]
             if nm is not None and nm == name),
            None)
        if fname is None: raise ValueError('Cannot find an FFM file with the name %s' % name)
    # Okay, load the model...
    
    gz = True if fname[-3:] == '.gz' else False
    lines = None
    with (gzip.open(fname, 'rb') if gz else open(fname, 'r')) as f:
        lines = f.read().split('\n')
    if len(lines) < 3 or lines[0] != 'Flat Mesh Model Version: 1.0':
        raise ValueError('Given name does not correspond to a valid flat mesh model file')
    n = int(lines[1].split(':')[1].strip())
    m = int(lines[2].split(':')[1].strip())
    reg = lines[3].split(':')[1].strip()
    hemi = lines[4].split(':')[1].strip().upper()
    center = map(float, lines[5].split(':')[1].strip().split(','))
    onxaxis = map(float, lines[6].split(':')[1].strip().split(','))
    method = lines[7].split(':')[1].strip().lower()
    tx = np.asarray(
        [map(float, row.split(','))
         for row in lines[8].split(':')[1].strip(' \t[]').split(';')])
    crds = np.asarray([map(float, left.split(','))
                       for row in lines[9:(n+9)]
                       for (left,right) in [row.split(' :: ')]])
    vals = np.asarray([map(float, right.split(','))
                       for row in lines[9:(n+9)]
                       for (left,right) in [row.split(' :: ')]])
    tris = -1 + np.asarray(
        [map(int, row.split(','))
         for row in lines[(n+9):(n+m+9)]])
    mdl = RegisteredRetinotopyModel(
        RetinotopyMeshModel(tris, crds,
                            90 - 180/pi*vals[:,0], vals[:,1], vals[:,2],
                            transform=tx),
        registration=reg,
        center=center,
        center_right=onxaxis,
        method=method,
        radius=radius,
        sphere_radius=sphere_radius,
        chirality=hemi)
    __loaded_V123_models[name] = mdl
    return mdl

# Tools for retinotopy registration:
def _retinotopy_vectors_to_float(ang, ecc, wgt, weight_cutoff=0):
    (ang, ecc, wgt) = np.asarray(
        [(a,e,w) if all(isinstance(x, Number) for x in [a,e,w]) and w > weight_cutoff else (0,0,0)
         for (a,e,w) in zip(ang, ecc, wgt)]).T
    return (ang, ecc, wgt)

def retinotopy_anchors(mesh, mdl,
                       polar_angle=None, eccentricity=None,
                       weight=None, weight_cutoff=0.2,
                       scale=1,
                       shape='Gaussian', suffix=None,
                       sigma=[0.05, 1.0, 2.0],
                       select='close'):
    '''
    retinotopy_anchors(mesh, model) is intended for use with the mesh_register function and the
    V123_model() function and/or the RetinotopyModel class; it yields a description of the anchor
    points that tie relevant vertices the given mesh to points predicted by the given model object.
    Any instance of the RetinotopyModel class should work as a model argument; this includes
    SchiraModel objects as well as RetinotopyMeshModel objects such as those returned by the
    V123_model() function. If the model given is a string, then it is passed to the V123_model()
    function first.

    Options:
      * polar_angle (default None) specifies that the given data should be used in place of the
        'polar_angle' or 'PRF_polar_angle'  property values. The given argument must be numeric and
        the same length as the the number of vertices in the mesh. If None is given, then the
        property value of the mesh is used; if a list is given and any element is None, then the
        weight for that vertex is treated as a zero. If the option is a string, then the property
        value with the same name isused as the polar_angle data.
      * eccentricity (default None) specifies that the given data should be used in places of the
        'eccentricity' or 'PRF_eccentricity' property values. The eccentricity option is handled 
        virtually identically to the polar_angle option.
      * weight (default None) specifies that the weight or scale of the data; this is handled
        generally like the polar_angle and eccentricity options, but may also be 1, indicating that
        all vertices with polar_angle and eccentricity values defined will be given a weight of 1.
        If weight is left as None, then the function will check for 'weight',
        'variance_explained', 'PRF_variance_explained', and 'retinotopy_weight' values and will use
        the first found (in that order). If none of these is found, then a value of 1 is assumed.
      * weight_cutoff (default 0) specifies that the weight must be higher than the given value inn
        order to be included in the fit; vertices with weights below this value have their weights
        truncated to 0.
      * scale (default 1) specifies a constant by which to multiply all weights for all anchors; the
        value None is interpreted as 1.
      * shape (default 'Gaussian') specifies the shape of the potential function (see mesh_register)
      * suffix (default None) specifies any additional arguments that should be appended to the 
        potential function description list that is produced by this function; i.e., the 
        retinotopy_anchors function produces a list, and the contents of suffix, if given and not
        None, are appended to that list (see mesh_register).
      * select (default 'close') specifies a function that will be called with two arguments for
        every vertex given an anchor; the arguments are the vertex label and the matrix of anchors.
        The function should return a list of anchors to use for the label (None is equivalent to
        lambda id,anc: anc). The parameter may alternately be specified using the string 'close':
        select=['close', [k]] indicates that any anchor more than k times the average edge-length in
        the mesh should be excluded; a value of just ['close', k] on the other hand indicates that
        any anchor more than k distance from the vertex should be exlcuded. The default value,
        'close', is equivalent to ['close', [20]].
      * sigma (default [0.05, 1.0, 2.0]) specifies how the sigma parameter should be handled; if
        None, then no sigma value is specified; if a single number, then all sigma values are
        assigned that value; if a list of three numbers, then the first is the minimum sigma value,
        the second is the fraction of the minimum distance between paired anchor points, and the 
        last is the maximum sigma --- the idea with this form of the argument is that the ideal
        sigma value in many cases is approximately 0.25 to 0.5 times the distance between anchors
        to which a single vertex is attracted; for any anchor a to which a vertex u is attracted,
        the sigma of a is the middle sigma-argument value times the minimum distance from a to all
        other anchors to which u is attracted (clipped by the min and max sigma).

    Example:
     # The retinotopy_anchors function is intended for use with mesh_register, as follows:
     # Define our Schira Model:
     model = neuropythy.registration.SchiraModel()
     # Make sure our mesh has polar angle, eccentricity, and weight data:
     mesh.prop('polar_angle',  polar_angle_vertex_data);
     mesh.prop('eccentricity', eccentricity_vertex_data);
     mesh.prop('weight',       variance_explained_vertex_data);
     # register the mesh using the retinotopy and model:
     registered_mesh = neuropythy.registration.mesh_register(
        mesh,
        ['mesh', retinotopy_anchors(mesh, model)],
        max_step_size=0.05,
        max_steps=2000)
    '''
    if isinstance(mdl, basestring):
        mdl = V123_model(mdl)
    if not isinstance(mdl, RetinotopyModel):
        raise RuntimeError('given model is not a RetinotopyModel instance!')
    if not isinstance(mesh, CorticalMesh):
        raise RuntimeError('given mesh is not a CorticalMesh object!')
    n = mesh.vertex_count
    X = mesh.coordinates.T
    if weight_cutoff is None: weight_cutoff = 0
    # make sure we have our polar angle/eccen/weight values:
    # (weight is odd because it might be a single number, so handle that first)
    (polar_angle, eccentricity, weight) = [
        extract_retinotopy_argument(mesh, name, arg, default='empirical')
        for (name, arg) in [
                ('polar_angle', polar_angle),
                ('eccentricity', eccentricity),
                ('weight', [weight for i in range(n)] if isinstance(weight, Number) else weight)]]
    # Make sure they contain no None/invalid values
    (polar_angle, eccentricity, weight) = _retinotopy_vectors_to_float(
        polar_angle, eccentricity, weight,
        weight_cutoff=weight_cutoff)
    idcs = [i for (i,w) in enumerate(weight) if w > 0]
    # Interpret the select arg if necessary (but don't apply it yet)
    select = ['close', [20]] if select == 'close'   else \
             ['close', [20]] if select == ['close'] else \
             select
    if select is None:
        select = lambda a,b: b
    elif isinstance(select, list) and len(select) == 2 and select[0] == 'close':
        d = np.mean(mesh.edge_lengths)*select[1][0] if isinstance(select[1], list) else select[1]
        select = lambda idx,ancs: [a for a in ancs if a[0] is not None if norm(X[idx] - a) < d]
    # Okay, apply the model:
    res = mdl.angle_to_cortex(polar_angle[idcs], eccentricity[idcs])
    # Organize the data; trim out those not selected
    data = [[[i for dummy in r], r]
            for (i,r0) in zip(idcs, res)
            if r0[0] is not None
            for r in [select(i, r0)]
            if len(r) > 0]
    # Flatten out the data into arguments for Java
    idcs = [i for d in data for i in d[0]]
    ancs = np.asarray([pt for d in data for pt in d[1]]).T
    # Get just the relevant weights and the scale
    wgts = weight[idcs] * (1 if scale is None else scale)
    # Figure out the sigma parameter:
    if sigma is None: sigs = None
    elif isinstance(sigma, Number): sigs = sigma
    elif hasattr(sigma, '__iter__') and len(sigma) == 3:
        [minsig, mult, maxsig] = sigma
        sigs = np.clip(
            [mult*min([norm(a0 - a) for a in anchs if a is not a0]) if len(iii) > 1 else maxsig
             for (iii,anchs) in data
             for a0 in anchs],
            minsig, maxsig)
    else:
        raise ValueError('sigma must be a number or a list of 3 numbers')
    # okay, we've partially parsed the data that was given; now we can construct the final list of
    # instructions:
    return (['anchor', shape, idcs, ancs, 'scale', wgts]
            + ([] if sigs is None else ['sigma', sigs])
            + ([] if suffix is None else suffix))

def register_retinotopy_initialize(hemi,
                                   model='standard',
                                   polar_angle=None, eccentricity=None, weight=None,
                                   weight_cutoff=0.2,
                                   max_predicted_eccen=85,
                                   partial_voluming_correction=True,
                                   prior='retinotopy',
                                   resample='fsaverage_sym'):
    '''
    register_retinotopy_initialize(hemi, model) yields an fsaverage_sym LH hemisphere that has
    been prepared for retinotopic registration with the data on the given hemisphere, hemi. The
    options polar_angle, eccentricity, weight, and weight_cutoff are accepted, as are the
    prior and resample options; all are documented in help(register_retinotopy).
    The return value of this function is actually a dictionary with the element 'map' giving the
    resulting map projection, and additional entries giving other meta-data calculated along the
    way.
    '''
    # Step 1: get our properties straight
    prop_names = ['polar_angle', 'eccentricity', 'weight']
    data = {}
    (ang, ecc, wgt) = [
        extract_retinotopy_argument(hemi, name, arg, default='empirical')
        for (name, arg) in [
                ('polar_angle', polar_angle),
                ('eccentricity', eccentricity),
                ('weight', [weight for i in range(n)] if isinstance(weight, Number) else weight)]]
    ## we also want to make sure weight is 0 where there are none values
    (ang, ecc, wgt) = _retinotopy_vectors_to_float(ang, ecc, wgt, weight_cutoff=weight_cutoff)
    ## correct for partial voluming if necessary:
    if partial_voluming_correction is True: wgt *= (1.0 - np.asarray(hemi.partial_volume_factor()))
    ## note these in the result dictionary:
    data['sub_polar_angle'] = ang
    data['sub_eccentricity'] = ecc
    data['sub_weight'] = wgt
    # Step 2: do alignment, if required
    if isinstance(model, basestring): model = V123_model(model)
    if not isinstance(model, RegisteredRetinotopyModel):
        raise ValueError('model must be a RegisteredRetinotopyModel')
    data['model'] = model
    model_reg = model.projection_data['registration']
    model_reg = 'fsaverage_sym' if model_reg is None else model_reg
    model_chirality = model.projection_data['chirality']
    if model_reg == 'fsaverage_sym':
        useHemi = hemi if hemi.chirality == 'LH' else hemi.subject.RHX
    else:
        if model_chiraliry is not None and hemi.chirality != model_chiraliry:
            raise ValueError('Inverse-chirality hemisphere cannot be registered to model')
        useHemi = hemi
    ## make sure we are registered to the model space
    if model_reg not in useHemi.topology.registrations:
        raise ValueError('Hemisphere is not registered to the model registration: %s' % model_reg)
    data['sub_hemi'] = useHemi
    ## note the subject's registration to the model's registration:
    subreg = useHemi.topology.registrations[model_reg]
    ## if there's a prior, we should enforce it now:
    if prior is not None:
        if hemi.subject.id == model_reg:
            prior_subject = useHemi.subject
            prior_hemi = useHemi
        else:
            prior_subject = freesurfer_subject(model_reg)
            prior_hemi = prior_subject.__getattr__(useHemi.chirality)
        if prior not in prior_hemi.topology.registrations:
            raise ValueError('Prior registration %s not found in prior subject %s' \
                             % (prior, model_reg))
        if model_reg not in prior_hemi.topology.registrations:
            raise ValueError('Model registratio not found in prior subject: %s' % prior_subject)
        prior_reg0 = prior_hemi.topology.registrations[model_reg]
        prior_reg1 = prior_hemi.topology.registrations[prior]
        addr = prior_reg0.address(subreg.coordinates)
        data['address_in_prior'] = addr
        coords = prior_reg1.unaddress(addr)
    else:
        prior_hemi = None
        coords = subreg.coordinates
    prior_reg = Registration(useHemi.topology, coords)
    data['prior_registration'] = prior_reg
    data['prior_hemisphere'] = prior_hemi
    # Step 3: resample, if need be (for now we always resample to fsaverage_sym)
    data['resample'] = resample
    if resample is None:
        tohem = useHemi
        toreg = prior_reg
        data['initial_registration'] = prior_reg
        for (p,v) in zip(prop_names, [useHemi.prop(p) for p in prop_names]):
            data['initial_' + p] = v
        data['unresample_function'] = lambda rr: rr
    else:
        if resample == 'fsaverage_sym':
            tohem = freesurfer_subject('fsaverage_sym').LH
            toreg = tohem.topology.registrations['fsaverage_sym']
        elif resample == 'fsaverage':
            tohem = freesurfer_subject('fsaverage').__getattr__(model_chirality)
            toreg = tohem.topology.registrations['fsaverage']
        else:
            raise ValueError('resample argument must be fsaverage, fsaverage_sym, or None')
        data['resample_hemisphere'] = tohem
        resamp_addr = toreg.address(prior_reg.coordinates)
        data['resample_address'] = resamp_addr
        data['initial_registration'] = toreg
        for (p,v) in zip(prop_names,
                         _retinotopy_vectors_to_float(
                             *[toreg.interpolate_from(prior_reg, data['sub_' + p])
                               for p in prop_names])):
            data['initial_' + p] = v
        data['unresample_function'] = lambda rr: Registration(useHemi.topology,
                                                              rr.unaddress(resamp_addr))
    data['initial_mesh'] = tohem.registration_mesh(toreg)
    # Step 4: make the projection
    proj_data = model.projection_data
    m = proj_data['forward_function'](data['initial_mesh'])
    for p in prop_names:
        m.prop(p, data['initial_' + p][m.vertex_labels])
    data['map'] = m
    # Step 5: Annotate how we get back
    def __postproc_fn(reg):
        d = data.copy()
        d['registered_coordinates'] = reg
        # First, unproject the map
        reg_map_3dx = d['map'].unproject(reg).T
        reg_3dx = np.array(d['initial_registration'].coordinates, copy=True)
        reg_3dx[d['map'].vertex_labels] = reg_map_3dx
        final_reg = Registration(tohem.topology, reg_3dx)
        d['finished_registration'] = final_reg
        # Now, if need be, unresample the points:
        d['registration'] = d['unresample_function'](final_reg)
        # now convert the sub points into retinotopy points
        rmesh = useHemi.registration_mesh(d['registration'])
        pred = np.asarray(
            [(p,e,l) if rl > 0 and rl < 4 and e <= max_predicted_eccen else (0.0, 0.0, 0)
             for (p,e,l) in zip(*model.cortex_to_angle(rmesh))
             for rl in [round(l)]]).T
        pred = (np.asarray(pred[0], dtype=np.float32),
                np.asarray(pred[1], dtype=np.float32),
                np.asarray(pred[2], dtype=np.int32))
        for i in (0,1,2): pred[i].flags.writeable = False
        pred = make_dict({p:v
                          for (p,v) in zip(['polar_angle', 'eccentricity', 'V123_label'], pred)})
        d['prediction'] = pred
        rmesh.prop(pred)
        d['registered_mesh'] = rmesh
        return make_dict(d)
    data['postprocess_function'] = __postproc_fn
    return data

def register_retinotopy(hemi,
                        retinotopy_model='standard',
                        polar_angle=None, eccentricity=None, weight=None, weight_cutoff=0.2,
                        partial_voluming_correction=True,
                        edge_scale=1.0, angle_scale=1.0, functional_scale=1.0,
                        sigma=Ellipsis,
                        select='close',
                        prior='retinotopy',
                        resample='fsaverage_sym',
                        max_steps=2000, max_step_size=0.05,
                        max_predicted_eccen=85,
                        return_meta_data=False):
    '''
    register_retinotopy(hemi) yields the result of registering the given hemisphere's polar angle
    and eccentricity data to the SchiraModel, a registration in which the vertices are aligned with
    the given model of retinotopy. The registration is added to the hemisphere's topology unless
    the option registration_name is set to None.

    Options:
      * retinotopy_model specifies the instance of the retinotopy model to use; this must be an
        instance of the RegisteredRetinotopyModel class or a string that can be passed to the
        V123_model() function (default: 'standard').
      * polar_angle, eccentricity, and weight specify the property names for the respective
        quantities; these may alternately be lists or numpy arrays of values. If weight is not given
        or found, then unity weight for all vertices is assumed. By default, each will check the
        hemisphere's properties for properties with compatible names; it will prefer the properties
        PRF_polar_angle, PRF_ecentricity, and PRF_variance_explained if possible.
      * weight_cutoff specifies the minimum value a vertex must have in the weight property in order
        to be considered as retinotopically relevant.
      * partial_voluming_correction (default: True), if True, specifies that the value
        (1 - hemi.partial_volume_factor()) should be applied to all weight values (i.e., weights
        should be down-weighted when likely to be affected by a partial voluming error).
      * sigma specifies the standard deviation of the Gaussian shape for the Schira model anchors.
      * edge_scale, angle_scale, and functional_scale all specify the relative strengths of the
        various components of the potential field (functional_scale refers to the strength of the
        retinotopy model).
      * select specifies the select option that should be passed to retinotopy_anchors.
      * max_steps (default 30,000) specifies the maximum number of registration steps to run.
      * max_step_size (default 0.05) specifies the maxmim distance a single vertex is allowed to
        move in a single step of the minimization.
      * return_meta_data (default: False) specifies whether the return value should be the new
        Registration object or a dictionary of meta-data that was used during the registration
        calculations, in which the key 'registation' gives the registration object.
      * radius (default: pi/3) specifies the radius, in radians, of the included portion of the map
        projection (projected about the occipital pole).
      * sigma (default Ellipsis) specifies the sigma argument to be passed onto the 
        retinotopy_anchors function (see help(retinotopy_anchors)); the default value, Ellipsis,
        is interpreted as the default value of the retinotopy_anchors function's sigma option.
      * max_predicted_eccen (default: 85) specifies the maximum eccentricity that should appear in
        the predicted retinotopy values.
      * prior (default: 'retinotopy') specifies the prior that should be used, if found, in the 
        topology registrations for the subject associated with the retinotopy_model's registration.
      * resample (default: 'fsaverage_sym') specifies that the data should be resampled to one of
        the uniform meshes, 'fsaverage' or 'fsaverage_sym', prior to registration; if None then no
        resampling is performed.
    '''
    # Step 1: prep the map for registrationfigure out what properties we're using...
    retinotopy_model = \
        V123_model()                 if retinotopy_model is None                 else \
        V123_model(retinotopy_model) if isinstance(retinotopy_model, basestring) else \
        retinotopy_model
    data = register_retinotopy_initialize(hemi,
                                          model=retinotopy_model,
                                          polar_angle=polar_angle,
                                          eccentricity=eccentricity,
                                          weight=weight,
                                          weight_cutoff=weight_cutoff,
                                          partial_voluming_correction=partial_voluming_correction,
                                          max_predicted_eccen=max_predicted_eccen,
                                          prior=prior, resample=resample)
    # Step 2: run the mesh registration
    if max_steps == 0:
        r = data['map'].coordinates
    else:
        r = mesh_register(
            data['map'],
            [['edge', 'harmonic', 'scale', edge_scale],
             ['angle', 'infinite-well', 'scale', angle_scale],
             ['perimeter', 'harmonic'],
             retinotopy_anchors(data['map'], retinotopy_model,
                                polar_angle='polar_angle',
                                eccentricity='eccentricity',
                                weight='weight',
                                weight_cutoff=weight_cutoff,
                                scale=functional_scale,
                                select=select,
                                **({} if sigma is Ellipsis else {'sigma':sigma}))],
            max_steps=max_steps,
            max_step_size=max_step_size)
    # Step 3: run the post-processing function
    postproc = data['postprocess_function']
    ppr = postproc(r)
    return ppr if return_meta_data else ppr['registered_mesh']

# Tools for registration-free retinotopy prediction:
__benson14_templates = None
def benson14_retinotopy(sub):
    '''
    benson14_retinotopy(subject) yields a pair of dictionaries each with three keys: polar_angle,
    eccentricity, and v123roi; each of these keys maps to a numpy array with one entry per vertex.
    The first element of the yielded pair is the left hemisphere map and the second is the right
    hemisphere map. The values are obtained by resampling the Benson et al. 2014 anatomically
    defined template of retinotopy to the given subject.
    Note that the subject must have been registered to the fsaverage_sym subject prior to calling
    this function; this requires using the surfreg command (after the xhemireg command for the RH).
    Additionally, you must have the fsaverage_sym template files in your fsaverage_syn/surf
    directory; these files are sym.template_angle.mgz, sym.template_eccen.mgz, and 
    sym.template_areas.mgz.
    '''
    global __benson14_templates
    if __benson14_templates is None:
        # Find a sym template that has the right data:
        sym_path = next((os.path.join(path0, 'fsaverage_sym')
                         for path0 in subject_paths()
                         for path in [os.path.join(path0, 'fsaverage_sym', 'surf')]
                         if os.path.isfile(os.path.join(path, 'sym.template_angle.mgz'))     \
                            and os.path.isfile(os.path.join(path, 'sym.template_eccen.mgz')) \
                            and os.path.isfile(os.path.join(path, 'sym.template_areas.mgz'))),
                        None)
        if sym_path is None:
            raise ValueError('No fsaverage_sym subject found with surf/sym.template_*.mgz files!')
        sym = freesurfer_subject(sym_path).LH
        tmpl_path = os.path.join(sym_path, 'surf', 'sym.template_')
        # We need to load in the template data
        __benson14_templates = {
            'angle': fsmgh.load(tmpl_path + 'angle.mgz').get_data().flatten(),
            'eccen': fsmgh.load(tmpl_path + 'eccen.mgz').get_data().flatten(),
            'v123r': fsmgh.load(tmpl_path + 'areas.mgz').get_data().flatten()}
    # Okay, we just need to interpolate over to this subject
    sym = freesurfer_subject('fsaverage_sym').LH
    return (
        {'polar_angle':  sub.LH.interpolate(sym,  __benson14_templates['angle'], apply=False),
         'eccentricity': sub.LH.interpolate(sym,  __benson14_templates['eccen'], apply=False),
         'v123roi':      sub.LH.interpolate(sym,  __benson14_templates['v123r'], apply=False,
                                            method='nearest')},
        {'polar_angle':  sub.RHX.interpolate(sym, __benson14_templates['angle'], apply=False),
         'eccentricity': sub.RHX.interpolate(sym, __benson14_templates['eccen'], apply=False),
         'v123roi':      sub.RHX.interpolate(sym, __benson14_templates['v123r'], apply=False,
                                             method='nearest')})
        

