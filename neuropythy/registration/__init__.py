####################################################################################################
# registration.py
# Tools for registering the cortical surface to a particular potential function
# By Noah C. Benson

import numpy as np
import scipy as sp
import os, sys, gzip
from numpy.linalg import norm
from math import pi
from numbers import Number
from neuropythy.cortex import CorticalMesh, empirical_retinotopy_data
from neuropythy.freesurfer import freesurfer_subject
from neuropythy.topology import Registration
from neuropythy.java import (java_link, serialize_numpy,
                             to_java_doubles, to_java_ints, to_java_array)
from pysistence import make_dict
from array import array

from .models import (RetinotopyModel, SchiraModel, RetinotopyMeshModel, RegisteredRetinotopyModel)

from py4j.java_gateway import (launch_gateway, JavaGateway, GatewayParameters)


# These are dictionaries of all the details we have about each of the possible arguments to the
# mesh_register's field argument:
_parse_field_data_types = {
    'mesh': ['newStandardMeshPotential', ['edge_scale', 1.0], ['angle_scale', 1.0], 'F', 'X'],
    'edge': {
        'harmonic':      ['newHarmonicEdgePotential',   ['scale', 1.0], ['order', 2.0], 'F', 'X'],
        'lennard-jones': ['newLJEdgePotential',         ['scale', 1.0], ['order', 2.0], 'F', 'X']},
    'angle': {
        'harmonic':      ['newHarmonicAnglePotential',  ['scale', 1.0], ['order', 2.0], 'F', 'X'],
        'lennard-jones': ['newLJAnglePotential',        ['scale', 1.0], ['order', 2.0], 'F', 'X'],
        'infinite-well': ['newWellAnglePotential',      ['scale', 1.0], ['order', 2.0], 
                                                        ['min',   0.0], ['max',   pi],  'F', 'X']},
    'anchor': {
        'harmonic':      ['newHarmonicAnchorPotential', ['scale', 1.0], ['shape', 2.0], 0, 1, 'X'],
        'gaussian':      ['newGaussianAnchorPotential', ['scale', 1.0], ['shape', 2.0], 
                                                        ['sigma', 2.0], 0, 1, 'X']},
    'perimeter': {
        'harmonic':   ['newHarmonicPerimeterPotential', ['scale', 1.0], ['shape', 2.0], 'F', 'X']}};
        
def _parse_field_function_argument(argdat, args, faces, coords):
    # first, see if this is an easy one...
    if argdat == 'F':
        return faces
    elif argdat == 'X':
        return coords
    elif isinstance(argdat, (int, long)):
        return to_java_array(args[argdat])
    # okay, none of those; must be a list with a default arg
    argname = argdat[0]
    argdflt = argdat[1]
    # see if we can find such an arg...
    for i in range(len(args)):
        if isinstance(args[i], basestring) and args[i].lower() == argname.lower():
            return args[i+1] if isinstance(args[i+1], Number) else to_java_array(args[i+1])
    # did not find the arg; use the default:
    return argdflt

def _parse_field_argument(instruct, faces, coords):
    _java = java_link()
    if isinstance(instruct, basestring):
        insttype = instruct
        instargs = []
    elif type(instruct) in [list, tuple]:
        insttype = instruct[0]
        instargs = instruct[1:]
    else:
        raise RuntimeError('potential field instruction must be list/tuple or string')
    # look this type up in the types data:
    insttype = insttype.lower()
    if insttype not in _parse_field_data_types:
        raise RuntimeError('Unrecognized field data type: ' + insttype)
    instdata = _parse_field_data_types[insttype]
    # if the data is a dictionary, we must parse on the next arg
    if isinstance(instdata, dict):
        shape_name = instargs[0].lower()
        instargs = instargs[1:]
        if shape_name not in instdata:
            raise RuntimeError('Shape ' + shape_name + ' not supported for type ' + insttype)
        instdata = instdata[shape_name]
    # okay, we have a list of instructions... find the java method we are going to call...
    java_method = getattr(_java.jvm.nben.mesh.registration.Fields, instdata[0])
    # and parse the arguments into a list...
    java_args = [_parse_field_function_argument(a, instargs, faces, coords) for a in instdata[1:]]
    # and call the function...
    return java_method(*java_args)

# parse a field potential argument and return a java object that represents it
def _parse_field_arguments(arg, faces, coords):
    '''See mesh_register.'''
    if not isinstance(arg, list):
        raise RuntimeError('field argument must be a list of instructions')
    pot = [_parse_field_argument(instruct, faces, coords) for instruct in arg]
    # make a new Potential sum unless the length is 1
    if len(pot) <= 1:
        return pot[0]
    else:
        sp = java_link().jvm.nben.mesh.registration.Fields.newSum()
        for field in pot: sp.addField(field)
        return sp

# The mesh_register function
def mesh_register(mesh, field, max_steps=2000, max_step_size=0.05, max_pe_change=1, k=4):
    '''
    mesh_register(mesh, field) yields the mesh that results from registering the given mesh by
    minimizing the given potential field description over the position of the vertices in the
    mesh. The mesh argument must be a CorticalMesh (see neuropythy.cortex) such as can be read
    from FreeSurfer using the neuropythy.freesurfer.Subject class. The field argument must be
    a list of field names and arguments; with the exception of 'mesh' (or 'standard'), the 
    arguments must be a list, the first element of which is the field type name, the second
    element of which is the field shape name, and the final element of which is a dictionary of
    arguments accepted by the field shape.

    The following are valid field type names:
      * 'mesh' : the standard mesh potential, which includes an edge potential, an angle
        potential, and a perimeter potential. Accepts no arguments, and must be passed as a
        single string instead of a list.
      * 'edge': an edge potential field in which the potential is a function of the change in the
        edge length, summed over each edge in the mesh.
      * 'angle': an angle potential field in which the potential is a function of the change in
        the angle measure, summed over all angles in the mesh.
      * 'perimeter': a potential that depends on the vertices on the perimeter of a 2D mesh
        remaining in place; the potential changes as a function of the distance of each perimeter
        vertex from its reference position.
      * 'anchor': a potential that depends on the distance of a set of vertices from fixed points
        in space. After the shape name second argument, an anchor must be followed by a list of
        vertex ids then a list of fixed points to which the vertex ids are anchored:
        ['anchor', shape_name, vertex_ids, fixed_points, args...].

    The following are valid shape names:
      * 'harmonic': a harmonic function with the form (c/q) * abs(x - x0)^q.
        Parameters: 
          * 'scale', the scale parameter c; default: 1.
          * 'order', the order parameter q; default: 2.
      * 'Lennard-Jones': a Lennard-Jones function with the form c (1 + (r0/r)^q - 2(r0/r)^(q/2));
        Parameters:
          * 'scale': the scale parameter c; default: 1. 
          * 'order': the order parameter q; default: 2.
      * 'Gaussian': A Gaussian function with the form c (1 - exp(-0.5 abs((x - x0)/s)^q))
        Parameters:
          * 'scale': the scale parameter c; default: 1.
          * 'order': the order parameter q; default: 2.
          * 'sigma': the standard deviation parameter s; default: 1.
      * 'infinite-well': an infinite well function with the form 
        c ( (((x0 - m)/(x - m))^q - 1)^2 + (((M - x0)/(M - x))^q - 1)^2 )
        Parameters:
          * 'scale': the scale parameter c; default: 1.
          * 'order': the order parameter q; default: 0.5.
          * 'min': the minimum value m; default: 0.
          * 'max': the maximum value M; default: pi.

    Options: The following optional arguments are accepted.
      * max_steps (default: 25000) the maximum number of steps to minimize for.
      * max_step_size (default: 0.1) the maximum distance to allow a vertex to move in a single
        minimization step.
      * max_pe_change: the maximum fraction of the initial potential value that the minimizer
        should minimize away before returning; i.e., 0 indicates that no minimization should be
        allowed while 0.9 would indicate that the minimizer should minimize until the potential
        is 10% or less of the initial potential.
      * k (default: 4) the number of groups into which the gradient should be partitioned each step;
        this argument, if greater than 1, specifies that the minimizer should use the nimbleStep
        rather than the step function for performing gradient descent; the number of partitions is
        k in this case.

    Examples:
      registered_mesh = mesh_register(
         mesh,
         [['edge', 'harmonic', 'scale', 0.5], # slightly weak edge potential
          ['angle', 'infinite-well'], # default arguments for an infinite-well angle potential
          ['anchor', 'Gaussian', [1, 10, 50], [[0.0, 0.0], [1.1, 1.1], [2.2, 2.2]]]],
         max_step_size=0.05,
         max_steps=10000)
    '''
    # Sanity checking:
    # First, make sure that the arguments are all okay:
    if not isinstance(mesh, CorticalMesh):
        raise RuntimeError('mesh argument must be an instance of neuropythy.cortex.CorticalMesh')
    if not isinstance(max_steps, (int, long)) or max_steps < 1:
        raise RuntimeError('max_steps argument must be a positive integer')
    if not isinstance(max_steps, (float, int, long)) or max_step_size <= 0:
        raise RuntimeError('max_step_size must be a positive number')
    if not isinstance(max_pe_change, (float, int, long)) or max_pe_change <= 0 or max_pe_change > 1:
        raise RuntimeError('max_pe_change must be a number x such that 0 < x <= 1')
    if not isinstance(k, (int, long)) or k < 1:
        raise RuntimeError('k must be a positive integer')
    # Parse the field argument.
    faces  = to_java_ints([mesh.index[frow] for frow in mesh.faces])
    coords = to_java_doubles(mesh.coordinates)
    potential = _parse_field_arguments(field, faces, coords)
    # Okay, that's basically all we need to do the minimization...
    minimizer = java_link().jvm.nben.mesh.registration.Minimizer(potential, coords)
    if k == 1:
        minimizer.step(float(max_pe_change), int(max_steps), float(max_step_size))
    else:
        minimizer.nimbleStep(float(max_pe_change), int(max_steps), float(max_step_size), int(k))
    result = minimizer.getX()
    return np.asarray([[x for x in row] for row in result])

__loaded_V123_models = {}
def V123_model(name='standard',
               projection_parameters={'registration':'fsaverage_sym', 'chirality':'LH'}):
    '''
    V123_model(name) yields a model of retinotopy in V1-V3 with the given name; if not provided,
    this name 'standard' is used (currently, the only possible option). The model itself is a set of
    meshes with values at the vertices that define the polar angle and eccentricity. These meshes
    are loaded from files in the neuropythy lib directory.
    '''
    if name in __loaded_V123_models:
        return __loaded_V123_models[name]
    fname = os.path.join(os.path.dirname(__file__), '..', '..', 'lib', 'models', name + '.fmm')
    gz = False
    if not os.path.isfile(fname):
        fname = fname + '.gz'
        gz = True
        if not os.path.isfile(fname):
            raise ValueError('Mesh model named \'%s\' not found' % name)
    lines = None
    with (gzip.open(fname, 'rb') if gz else open(fname, 'r')) as f:
        lines = f.read().split('\n')
    if len(lines) < 3 or lines[0] != 'Flat Mesh Model Version: 1.0':
        raise ValueError('Given name does not correspond to a valid flat mesh model file')
    n = int(lines[1].split(':')[1].strip())
    m = int(lines[2].split(':')[1].strip())
    tx = np.asarray(
        [map(float, row.split(','))
         for row in lines[3].split(':')[1].strip(' \t[]').split(';')])
    crds = np.asarray([map(float, left.split(','))
                       for row in lines[4:(n+4)]
                       for (left,right) in [row.split(' :: ')]])
    vals = np.asarray([map(float, right.split(','))
                       for row in lines[4:(n+4)]
                       for (left,right) in [row.split(' :: ')]])
    tris = -1 + np.asarray(
        [map(int, row.split(','))
         for row in lines[(n+4):(n+m+4)]])
    mdl = RegisteredRetinotopyModel(
        RetinotopyMeshModel(tris, crds,
                            90 - 180/pi*vals[:,0], vals[:,1], vals[:,2],
                            transform=tx),
        **projection_parameters)
    __loaded_V123_models[name] = mdl
    return mdl

def retinotopy_anchors(mesh, mdl,
                       polar_angle=None, eccentricity=None,
                       weight=None, weight_cutoff=0.2,
                       scale=1,
                       shape='Gaussian', suffix=None,
                       sigma=[0.05, 0.3, 2.0],
                       select='close'):
    '''
    retinotopy_anchors(mesh, model) is intended for use with the mesh_register function and the
    V123_model() function and/or the RetinotopyModel class; it yields a description of the anchor
    points that tie relevant vertices the given mesh to points predicted by the given model object.
    Any instance of the RetinotopyModel class should work as a model argument; this includes
    SchiraModel objects as well as RetinotopyMeshModel objects such as those returned by the
    V123_model() function.

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
      * select (default None) specifies a function that will be called with two arguments for every
        vertex given an anchor; the arguments are the vertex label and the matrix of anchors. The
        function should return a list of anchors to use for the label (None is equivalent to
        lambda id,anc: anc).
      * sigma (default [0.05, 0.3, 2.0]) specifies how the sigma parameter should be handled; if
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
        ['mesh', retinotopy_anchors(mesh, model, weight_cutoff=0.2)],
        max_step_size=0.05,
        max_steps=2000)
    '''
    if not isinstance(mdl, RetinotopyModel):
        raise RuntimeError('given model is not a RetinotopyModel instance!')
    if not isinstance(mesh, CorticalMesh):
        raise RuntimeError('given mesh is not a CorticalMesh object!')
    n = len(mesh.vertex_labels)
    X = mesh.coordinates.T
    # make sure we have our polar angle/eccen/weight values:
    polar_angle = mesh.prop(polar_angle) if isinstance(polar_angle, basestring) else \
                  polar_angle if polar_angle is not None else \
                  empirical_retinotopy_data(mesh, 'polar_angle')
    if polar_angle is None:
        raise RuntimeError('No polar angle data given to schira_anchors!')
    if isinstance(polar_angle, dict):
        # a dictionary is okay, we just need to fix it to a list:
        tmp = polar_angle
        polar_angle = [tmp[i] if i in tmp else None for i in range(n)]
    if len(polar_angle) != n:
        raise RuntimeError('Polar angle data has incorrect length!')
    # Now Polar Angle...
    eccentricity = mesh.prop(eccentricity) if isinstance(eccentricity, basestring) else \
                   eccentricity if eccentricity is not None else \
                   empirical_retinotopy_data(mesh, 'eccentricity')
    if eccentricity is None:
        raise RuntimeError('No eccentricity data given to schira_anchors!')
    if isinstance(eccentricity, dict):
        tmp = eccentricity
        eccentricity = [tmp[i] if i in tmp else None for i in range(n)]
    if len(eccentricity) != n:
        raise RuntimeError('Eccentricity data has incorrect length!')
    # Now Weight...
    weight = mesh.prop(weight) if isinstance(weight, basestring) else \
             weight if weight is not None else \
             empirical_retinotopy_data(mesh, 'weight')
    if weight is None:
        weight = 1
    if isinstance(weight, dict):
        tmp = weight
        weight = [tmp[i] if i in tmp else None for i in range(n)]
    if isinstance(weight, Number):
        weight = [weight for i in range(n)]
    if len(weight) != n:
        raise RuntimeError('Weight data has incorrect length!')
    # Handle the select arg if necessary:
    select = ['close', [20]] if select == 'close'   else \
             ['close', [20]] if select == ['close'] else \
             select
    if select is None:
        select = lambda a,b: b
    elif isinstance(select, list) and len(select) == 2 and select[0] == 'close':
        d = np.mean(mesh.edge_lengths)*select[1][0] if isinstance(select[1], list) else select[1]
        select = lambda idx,ancs: [a for a in ancs if a[0] is not None if norm(X[idx] - a) < d]
    # let's go through and fix up the weights/polar angles/eccentricities into appropriate lists
    if weight_cutoff is None:
        idcs = [i
                for i in range(n)
                if (polar_angle[i] is not None and eccentricity[i] is not None 
                    and weight[i] is not None and weight[i] != 0)]
    else:
        idcs = [i
                for i in range(n)
                if (polar_angle[i] is not None and eccentricity[i] is not None 
                    and weight[i] is not None and weight[i] >= weight_cutoff)]
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
    wgts = np.asarray(weight)[idcs] * (1 if scale is None else scale)
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

def _retinotopy_vectors_to_float(ang, ecc, wgt):
    wgt = np.asarray(
        [(0.0 if w is None else w)*(0.0 if a is None else 1)*(0.0 if e is None else 1)
         for (a,e,w) in zip(ang, ecc, [1.0 for e in ecc] if wgt is None else wgt)])
    ang = np.asarray([0.0 if a is None else a for a in ang])
    ecc = np.asarray([0.0 if e is None else e for e in ecc])
    return (ang, ecc, wgt)


def register_retinotopy_initialize(hemi, model,
                                   polar_angle=None, eccentricity=None, weight=None,
                                   weight_cutoff=0.2,
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
        (hemi.prop(arg) if isinstance(arg, basestring) else
         arg            if hasattr(arg, '__iter__')    else
         None           if arg is not None             else
         empirical_retinotopy_data(hemi, argstr))
        for (arg, argstr) in zip((polar_angle, eccentricity, weight), prop_names)]
    if ang is None: raise ValueError('polar angle data not found')
    if ecc is None: raise ValueError('eccentricity data not found')
    ## we also want to make sure weight is 0 where there are none values
    (ang, ecc, wgt) = _retinotopy_vectors_to_float(ang, ecc, wgt)
    ## note these in the result dictionary:
    data['sub_polar_angle'] = ang
    data['sub_eccentricity'] = ecc
    data['sub_weight'] = wgt
    # Step 2: do alignment, if required
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
    ## note the position of the coordinates in the 
    subreg = useHemi.topology.registrations[model_reg]
    ## if there's a prior, we should enforce it now:
    if prior is not None:
        if hemi.subject.id == model_reg:
            prior_subject = useHemi.subject
            prior_hemi = None
        else:
            prior_subject = freesurfer_subject(model_reg)
            prior_hemi = prior_subject.__getattr__(model_chirality)
        if prior not in prior_hemi.topology.registrations:
            raise ValueError('Prior registration %s not found in prior subject %s' \
                             % (prior, model_reg))
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
        d['registered_mesh'] = useHemi.registration_mesh(d['registration'])
        d['prediction'] = make_dict({p: v for (p,v) in zip(
                                            ['polar_angle', 'eccentricity', 'visual_area'],
                                            model.cortex_to_angle(d['registration_mesh']))})
        return make_dict(d)
    data['postprocess_function'] = __postproc_fn
    return data

def register_retinotopy(hemi,
                        retinotopy_model=None,
                        polar_angle=None, eccentricity=None, weight=None, weight_cutoff=0.2,
                        edge_scale=1.0, angle_scale=1.0, functional_scale=1.0,
                        sigma=Ellipsis,
                        select='close',
                        prior='retinotopy',
                        resample='fsaverage_sym',
                        max_steps=2000, max_step_size=0.05,
                        return_meta_data=False):
    '''
    register_retinotopy(hemi) yields the result of registering the given hemisphere's polar angle
    and eccentricity data to the SchiraModel, a registration in which the vertices are aligned with
    the given model of retinotopy. The registration is added to the hemisphere's topology unless
    the option registration_name is set to None.

    Options:
      * retinotopy_model specifies the instance of the retinotopy model to use; this must be an
        instance of the RegisteredRetinotopyModel class (default: None, which is translated to
        V123_model()).
      * polar_angle, eccentricity, and weight specify the property names for the respective
        quantities; these may alternately be lists or numpy arrays of values. If weight is not given
        or found, then unity weight for all vertices is assumed. By default, each will check the
        hemisphere's properties for properties with compatible names; it will prefer the properties
        PRF_polar_angle, PRF_ecentricity, and PRF_variance_explained if possible.
      * weight_cutoff specifies the minimum value a vertex must have in the weight property in order
        to be considered as retinotopically relevant.
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
      * prior (default: 'retinotopy') specifies the prior that should be used, if found, in the 
        topology registrations for the subject associated with the retinotopy_model's registration.
      * resample (default: 'fsaverage_sym') specifies that the data should be resampled to one of
        the uniform meshes, 'fsaverage' or 'fsaverage_sym', prior to registration; if None then no
        resampling is performed.
    '''
    # Step 1: prep the map for registrationfigure out what properties we're using...
    retinotopy_model = V123_model() if retinotopy_model is None else retinotopy_model
    data = register_retinotopy_initialize(hemi, retinotopy_model,
                                          polar_angle=polar_angle,
                                          eccentricity=eccentricity,
                                          weight=weight,
                                          weight_cutoff=weight_cutoff,
                                          prior=prior, resample=resample)
    # Step 2: run the mesh registration
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
    return ppr if return_meta_data else ppr['registration']

def register_retinotopy_command(args):
    '''
    register_retinotopy_command(args) can be given a list of arguments, such as sys.argv[1:]; these
    arguments may include any options and must include one subject id and one hemisphere, all in any
    order. The subject whose id is given is registered to a retinotopy model, and the resulting
    registration, as well as the predictions made by the model in the registration, are exported.
    The following options are accepted:
      * --eccen=|-e<file>
        --angle=|-a<file>
        --weight=|-w<file>
        Each of these arguments specifies the name of a data file to load in as a representation of
        the subject's eccentricity, polar angle, or weight; these should be given the names of
        either an mgh/mgz files whose size is 1 x 1 x n, where n is the number of vertices in the
        hemisphere for the subject, or a FreeSurfer curv-style filename with n vertices. By default,
        files in the subject's surf directory that match a template are automatically loaded and
        used. This template is name <hemi>.<tag><name>, optionally ending with .mgz, where tag is
        one of (and in order of preference) 'prf_', 'empirical_', 'measured_', 'training_', or '', 
        and name is one of 'eccentricity'/'eccen', 'polar_angle'/'angle', or 
        'weight'/'variance_explained'/'vexpl'.
      * --cutoff=|-c<value>
        The cutoff value to use for the weight; 0.2 by default. Weights less than this will be
        truncated to 0.
      * --angle-radians|-r
        This flag specifies that the angle-file only is in radians instead of degrees.
      * --eccen-radians|-R
        This flag specifies that the eccen-file only is in radians instead of degrees.
      * --mathematical|-m
        This flag specifies that the angle file addresses the visual space in the way standard in
        geometry; i.e., with the right horizontal meridian represented as 0 and with the upper
        vertical meridian represented as 90 degrees or pi/4 instead of the convention in which the
        opper vertical meridian represented as 0 and the right horizontal meridian represented as 90
        degrees or pi/4 radians.
      * --edge-strength=|-e<weight>
        --angle-strength=|-a<weight>
        --functional-strength=|-f<weight>
        
     
    '''
    #TODO
    pass

# The topology and registration stuff is below:
class JavaTopology:
    '''
    JavaTopology(triangles, registrations) creates a topology object object with the given triangle
    mesh, defined by a 3xn matrix of triangle indices, and with the registration coordinate matrices
    given in the dictionary registrations. This class should only be instantiated by the neuropythy
    library and should generally not be constructed directly. See Hemisphere.topology objects to
    access a subject's topologies.
    '''
    def __init__(self, triangles, registrations):
        # First: make a java object for the topology:
        faces = serialize_numpy(triangles.T, 'i')
        topo = java_link().jvm.nben.geometry.spherical.MeshTopology.fromBytes(faces)
        # Okay, make our registration dictionary
        d = {k: topo.registerBytes(serialize_numpy(v, 'd'))
             for (k,v) in registrations.iteritems()}
        # That's all really
        self.__dict__['_java_object'] = topo
        self.__dict__['registrations'] = d
    def __getitem__(self, attribute):
        return self.registrations[attribute]
    def __setitem__(self, attribute, dat):
        self.registrations[attribute] = self._java_object.registerBytes(serialize_numpy(dat, 'd'))
    def keys(self):
        return self.registrations.keys()
    def iterkeys(self):
        return self.registrations.iterkeys()
    def values(self):
        return self.registrations.values()
    def itervalues(self):
        return self.registrations.itervalues()
    def items(self):
        return self.registrations.items()
    def iteritems(self):
        return self.registrations.iteritems()
    def __len__(self):
        return len(self.registrations)
    
    # These let us interpolate...
    def interpolate(fromtopo, data, order=2, fill=None):
        usable_keys = []
        for k in registrations.iterkeys():
            if k in fromtopo.registrations:
                usable_keys.append(k)
        if not usable_keys:
            raise RuntimeError('no registration found that links topologies')
        the_key = usable_keys[0]
        # Prep the data into java arrays
        jmask = serialize_numpy(np.asarray([1 if d is not None else 0 for d in data]), 'd')
        jdata = serialize_numpy(np.asarray([d if d is not None else 0 for d in data]), 'd')
        # okay, next step is to call out to the java...
        maskres = self._java_object.interpolateBytes(
            fromtopo.registrations[the_key],
            self.registrations[the_key].coordinates,
            order, jdata)
        datares = self._java_object.interpolateBytes(
            fromtopo.registrations[the_key],
            self.registrations[the_key].coordinates,
            order, jmask)
        # then interpret the results...
        return [datares[i] if maskres[i] == 1 else fill for i in range(len(maskres))]

