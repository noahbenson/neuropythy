####################################################################################################
# neuropythy/topology/__init__.py
# Tools for transferring data from one cortical surface to another via spherical registered 
# topologies.
# By Noah C. Benson

import numpy as np
import numpy.linalg
import scipy as sp
import scipy.spatial as space
import os, math
from pysistence import make_dict

from neuropythy.immutable import Immutable
import neuropythy.geometry as geo

class Topology(object):
    '''
    Topology(triangles, registrations) constructs a topology object object with the given triangle
    mesh, defined by a 3xn matrix of triangle indices, and with the registration coordinate matrices
    given in the dictionary registrations. This class should only be instantiated by the neuropythy
    library and should generally not be constructed directly. See Hemisphere.topology objects to
    access a subject's topologies.
    '''

    def __init__(self, triangles, registrations):
        triangles = np.asarray(triangles)
        self.triangles = triangles if triangles.shape[1] == 3 else triangles.T
        self.vertex_count = np.max(triangles.flatten())
        self.registrations = make_dict({name: Registration(self, np.asarray(coords))
                                        for (name, coords) in registrations.iteritems()
                                        if coords is not None})
    def __repr__(self):
        return 'Topology(<%d triangles>, <%d vertices>)' % (self.triangles.shape[0],
                                                            self.vertex_count)
    def register(self, name, coords):
        '''
        topology.register(name, coordinates) adds the given registration to the given topology.
        '''
        coords = np.asarray(coords)
        self.registrations = self.registrations.using(
            name,
            Registration(self, coords if coords.shape[0] < 4 else coords.T))
        return self
    def interpolate_from(self, topo, data, mask=None, null=None, method='automatic', n_jobs=1):
        '''
        topology.interpolate_from(topo, data) yields a numpy array of the data interpolated from
        the given array, data, which must contain the same number of elements as there are points in
        the topology object topo, to the coordinates in the given Topology object topology. Note
        that in order for an interpolation to occur, the two topologies, topology and topo, must
        have a shared registration; i.e., a registration with the same name.
        
        The following options are accepted:
          * mask (default: None) indicates that the given True/False or 0/1 valued list/array should
            be used; any point whose nearest neighbor (see below) is in the given mask will, instead
            of an interpolated value, be set to the null value (see null option).
          * null (default: None) indicates the value that should be placed in the returned result if
            either a vertex does not lie in any triangle or a vertex is masked out via the mask
            option.
          * smoothing (default: 2) assuming that the method is 'interpolate' or 'automatic', this
            is the exponent used to smooth the interpolated surface; 1 is pure linear interpolation
            while 2 represents a slightly smoother version of this. Note that this is not an order
            of interpolation option.
          * method (default: 'automatic') specifies what method to use for interpolation. The only
            currently supported methods are 'automatic' or 'nearest'. The 'nearest' method does not
            actually perform a nearest-neighbor interpolation but rather assigns to a destination
            vertex the value of the source vertex whose veronoi-like polygon contains the
            destination vertex; note that the term 'veronoi-like' is used here because it uses the
            Voronoi diagram that corresponds to the triangle mesh and not the true delaunay
            triangulation. The 'automatic' checks every destination vertex and assigns it the
            'nearest' value if that value would not be a number, otherwise it interpolates linearly
            within the vertex's source triangle.
          * n_jobs (default: 1) is passed along to the cKDTree.query method, so may be set to an
            integer to specify how many processors to use, or may be -1 to specify all processors.
        '''
        # find a shared registration:
        reg_names = [k for k in topo.registrations.iterkeys() if k in self.registrations]
        if not reg_names:
            raise RuntimeError('Topologies do not share a matching registration!')
        res = []
        for reg_name in reg_names:
            try:
                res = self.registrations[reg_name].interpolate_from(
                    topo.registrations[reg_name], data,
                    mask=mask, null=null, method=method, n_jobs=n_jobs);
            except:
                pass
        if res is None:
            raise ValueError('All shared topologies raised errors during interpolation!')
        return res

class Registration(geo.Mesh):
    '''
    A Registration object represents a configuration of a topology object. Registrations represent
    various coordinate configurations of a hemisphere; e.g., in FreeSurfer, any given hemisphere has
    at least the subject's native configuration, the fsaverage configuration, and the fsaverage_sym
    configuration. A registration may be used to sample data between subjects. Generally speaking,
    a registration object should only be constructed via other parts of the neuropythy library and
    not via Registration() directly.
    '''

    def __init__(self, topology, coordinates):
        geo.Mesh.__init__(self, topology.triangles, coordinates)

    def __repr__(self):
        return 'Registration(<%d triangles>, <%d vertices>)' % (self.triangles.shape[0],
                                                                self.coordinates.shape[0])


    def interpolate_from(self, reg, data, 
                         smoothing=2, mask=None, null=None, method='automatic', n_jobs=1):
        '''
        registration.interpolate_from(reg, data) yields a numpy array of the data interpolated from
        the given array, data, which must contain the same number of elements as there are points in
        the Registration object reg, to the coordinates in the given Registration object,
        registration.
        
        The following options are accepted:
          * mask (default: None) indicates that the given True/False or 0/1 valued list/array should
            be used; any point whose nearest neighbor (see below) is in the given mask will, instead
            of an interpolated value, be set to the null value (see null option).
          * null (default: None) indicates the value that should be placed in the returned result if
            either a vertex does not lie in any triangle or a vertex is masked out via the mask
            option.
          * smoothing (default: 2) assuming that the method is 'interpolate' or 'automatic', this
            is the exponent used to smooth the interpolated surface; 1 is pure linear interpolation
            while 2 represents a slightly smoother version of this. Note that this is not an order
            of interpolation option.
          * method (default: 'automatic') specifies what method to use for interpolation. The only
            currently supported methods are 'automatic' or 'nearest'. The 'nearest' method does not
            actually perform a nearest-neighbor interpolation but rather assigns to a destination
            vertex the value of the source vertex whose veronoi-like polygon contains the
            destination vertex; note that the term 'veronoi-like' is used here because it uses the
            Voronoi diagram that corresponds to the triangle mesh and not the true delaunay
            triangulation. The 'automatic' checks every destination vertex and assigns it the
            'nearest' value if that value would not be a number, otherwise it interpolates linearly
            within the vertex's source triangle.
          * n_jobs (default: 1) is passed along to the cKDTree.query method, so may be set to an
            integer to specify how many processors to use, or may be -1 to specify all processors.
        '''
        return reg.interpolate(self.coordinates, data,
                               smoothing=smoothing, mask=mask, null=null,
                               method=method, n_jobs=n_jobs)

    
