####################################################################################################
# registration/core.py
# Core tools for registering the cortical surface to a particular potential function
# By Noah C. Benson

import numpy as np
import scipy as sp
import os, sys, gzip
from numpy.linalg import norm
from math import pi
from numbers import Number
from neuropythy.cortex import (CorticalMesh)
from neuropythy.freesurfer import (freesurfer_subject, add_subject_path,
                                   cortex_to_ribbon, cortex_to_ribbon_map,
                                   Hemisphere, subject_paths)
from neuropythy.topology import Registration
from neuropythy.java import (java_link, serialize_numpy,
                             to_java_doubles, to_java_ints, to_java_array)
import nibabel.freesurfer.io as fsio
import nibabel.freesurfer.mghformat as fsmgh
from pysistence import make_dict
from array import array

from neuropythy.vision import (RetinotopyModel, SchiraModel,
                               RetinotopyMeshModel, RegisteredRetinotopyModel)

from py4j.java_gateway import (launch_gateway, JavaGateway, GatewayParameters)


# These are dictionaries of all the details we have about each of the possible arguments to the
# mesh_register's field argument:
_parse_field_data_types = {
    'mesh': ['newStandardMeshPotential', ['edge_scale', 1.0], ['angle_scale', 1.0], 'F', 'X'],
    'edge': {
        'harmonic':      ['newHarmonicEdgePotential',   ['scale', 1.0], ['order', 2.0], 'F', 'X'],
        'lennard-jones': ['newLJEdgePotential',         ['scale', 1.0], ['order', 2.0], 'F', 'X'],
        'infinite-well': ['newWellEdgePotential',       ['scale', 1.0], ['order', 0.5], 
                                                        ['min',   0.5], ['max',   3.0], 'E', 'X']},
    'angle': {
        'harmonic':      ['newHarmonicAnglePotential',  ['scale', 1.0], ['order', 2.0], 'F', 'X'],
        'lennard-jones': ['newLJAnglePotential',        ['scale', 1.0], ['order', 2.0], 'F', 'X'],
        'infinite-well': ['newWellAnglePotential',      ['scale', 1.0], ['order', 0.5], 
                                                        ['min',   0.0], ['max',   pi],  'F', 'X']},
    'anchor': {
        'harmonic':      ['newHarmonicAnchorPotential', ['scale', 1.0], ['shape', 2.0], 0, 1, 'X'],
        'gaussian':      ['newGaussianAnchorPotential', ['scale', 1.0], ['sigma', 2.0], 
                                                        ['shape', 2.0], 0, 1, 'X']},
    'mesh-field': {
        'harmonic':      ['newHarmonicMeshPotential',   ['scale', 1.0], ['order', 2.0], 0, 1, 2,
                                                        3, 4, 'X'],
        'gaussian':      ['newGaussianMeshPotential',   ['scale', 1.0], ['sigma', 0.5],
                                                        ['order', 2.0], 0, 1, 2, 3, 4, 'X']},
    'perimeter': {
        'harmonic':   ['newHarmonicPerimeterPotential', ['scale', 1.0], ['shape', 2.0], 'F', 'X']}};
        
def _parse_field_function_argument(argdat, args, faces, edges, coords):
    # first, see if this is an easy one...
    if argdat == 'F':
        return faces
    elif argdat == 'X':
        return coords
    elif argdat == 'E':
        return edges
    elif isinstance(argdat, (int, long)):
        return to_java_array(args[argdat])
    # okay, none of those; must be a list with a default arg
    argname = argdat[0]
    argdflt = argdat[1]
    # see if we can find such an arg...
    for i in range(len(args)):
        if isinstance(args[i], basestring) and args[i].lower() == argname.lower():
            return (args[i+1] if (isinstance(args[i+1], Number)
                                  or np.issubdtype(type(args[i+1]), np.float)) else
                    to_java_array(args[i+1]))
    # did not find the arg; use the default:
    return argdflt

def _parse_field_argument(instruct, faces, edges, coords):
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
    java_args = [_parse_field_function_argument(a, instargs, faces, edges, coords)
                 for a in instdata[1:]]
    # and call the function...
    return java_method(*java_args)

# parse a field potential argument and return a java object that represents it
def _parse_field_arguments(arg, faces, edges, coords):
    '''See mesh_register.'''
    if not isinstance(arg, list):
        raise RuntimeError('field argument must be a list of instructions')
    pot = [_parse_field_argument(instruct, faces, edges, coords) for instruct in arg]
    # make a new Potential sum unless the length is 1
    if len(pot) <= 1:
        return pot[0]
    else:
        sp = java_link().jvm.nben.mesh.registration.Fields.newSum()
        for field in pot: sp.addField(field)
        return sp

def java_potential_term(mesh, instructions):
    '''
    java_potential_term(mesh, instructions) yields a Java object that implements the potential field
      described in the given list of instructions. Generally, this should not be invoked directly
      and should only be called by mesh_register. Note: this expects a single term's description,
      not a series of descriptions.
    '''
    faces  = to_java_ints(mesh.indexed_faces)
    edges  = to_java_ints(mesh.indexed_edges)
    coords = to_java_doubles(mesh.coordinates)
    return _parse_field_arguments([instructions], faces, edges, coords)
    
# The mesh_register function
def mesh_register(mesh, field, max_steps=2000, max_step_size=0.05, max_pe_change=1,
                  method='random', return_report=False):
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
      * max_steps (default: 2000) the maximum number of steps to minimize for.
      * max_step_size (default: 0.1) the maximum distance to allow a vertex to move in a single
        minimization step.
      * max_pe_change: the maximum fraction of the initial potential value that the minimizer
        should minimize away before returning; i.e., 0 indicates that no minimization should be
        allowed while 0.9 would indicate that the minimizer should minimize until the potential
        is 10% or less of the initial potential.
      * return_report (default: False) indicates that instead of returning the registered data,
        mesh_register should instead return the Java Minimizer.Report object (for debugging).
      * method (default: 'random') specifies the search algorithm used; available options are 
        'random', 'nimble', and 'pure'. Generally all options will converge on a similar solution,
        but usually 'random' is fastest. The 'pure' option uses the nben library's step function,
        which performs straight-forward gradient descent. The 'nimble' option performs a gradient
        descent in which subsets of vertices in the mesh that have the highest gradients during the
        registration are updated more often than those vertices with small gradients; this can
        sometimes but not always increase the speed of the minimization. Note that instead of
        'nimble', one may alternately provide ('nimble', k) where k is the number of partitions that
        the vertices should be sorted into (by partition). 'nimble' by itself is equivalent to 
        ('nimble', 4). Note also that a single step of nimble minimization is equivalent to 2**k
        steps of 'pure' minimization. Finally, the 'random' option uses the nben library's
        randomStep function, which is a gradient descent algorithm that moves each vertex in the
        direction of its negative gradient during each step but which randomizes the length of the
        gradient at each individual vertex by drawing from an exponential distribution centered at
        the vertex's actual gradient length. In effect, this can prevent vertices with very large
        gradients from dominating the minimization and often results in the best results.

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
    if not isinstance(max_steps, (int, long)) or max_steps < 0:
        raise RuntimeError('max_steps argument must be a positive integer')
    if not isinstance(max_steps, (float, int, long)) or max_step_size <= 0:
        raise RuntimeError('max_step_size must be a positive number')
    if not isinstance(max_pe_change, (float, int, long)) or max_pe_change <= 0 or max_pe_change > 1:
        raise RuntimeError('max_pe_change must be a number x such that 0 < x <= 1')
    if isinstance(method, basestring):
        method = method.lower()
        if method == 'nimble': k = 4
        else:                  k = 0
    else:
        k = method[1]
        method = method[0].lower()
    # If steps is 0, we can skip most of this...
    if max_steps == 0:
        if return_report: return None
        else: return mesh.coordinates
    # Otherwise, we run at least some minimization
    max_pe_change = float(max_pe_change)
    max_steps = int(max_steps)
    max_step_size = float(max_step_size)
    # Parse the field argument.
    faces  = to_java_ints(mesh.indexed_faces)
    edges  = to_java_ints(mesh.indexed_edges)
    coords = to_java_doubles(mesh.coordinates)
    potential = _parse_field_arguments(field, faces, edges, coords)
    # Okay, that's basically all we need to do the minimization...
    minimizer = java_link().jvm.nben.mesh.registration.Minimizer(potential, coords)
    if method == 'pure':
        rep = minimizer.step(max_pe_change, max_steps, max_step_size)
    elif method == 'random':
        # if k is -1, we do the inverse version where we draw from the 1/mean distribution
        rep = minimizer.randomStep(max_pe_change, max_steps, max_step_size, k == -1)
    elif method == 'nimble':
        rep = minimizer.nimbleStep(max_pe_change, max_steps, max_step_size, int(k))
    else:
        raise ValueError('Unrecognized method: %s' % method)
    # Return the report if requested
    if return_report:
        return rep
    else:
        result = minimizer.getX()
        return np.asarray([[x for x in row] for row in result])

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

