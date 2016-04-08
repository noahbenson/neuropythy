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
import neuropythy.geometry as geo

class Topology:
    '''
    Topology(triangles, registrations) constructs a topology object object with the given triangle
    mesh, defined by a 3xn matrix of triangle indices, and with the registration coordinate matrices
    given in the dictionary registrations. This class should only be instantiated by the neuropythy
    library and should generally not be constructed directly. See Hemisphere.topology objects to
    access a subject's topologies.
    '''

    def __init__(self, triangles, registrations):
        self.triangles = triangles if triangles.shape[1] == 3 else triangles.T
        self.registrations = make_dict({name: Registration(self, coords) 
                                        for (name, coords) in registrations.iteritems()})
    def register(self, name, coords):
        '''
        topology.register(name, coordinates) adds the given registration to the given topology.
        '''
        self.registrations = self.registrations.using(name, Registration(coords))
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
        reg_name = next((k for k in topo.registrations.iterkeys() if k in self.registrations),
                        None)
        if reg_name is None:
            raise RuntimeError('Topologies do not share a matching registration!')
        return self.registrations[reg_name].interpolate_from(
            topo.registrations[reg_name], data,
            mask=mask, null=null, method=method, n_jobs=n_jobs);

class Registration:
    '''
    A Registration object represents a configuration of a topology object. Registrations represent
    various coordinate configurations of a hemisphere; e.g., in FreeSurfer, any given hemisphere has
    at least the subject's native configuration, the fsaverage configuration, and the fsaverage_sym
    configuration. A registration may be used to sample data between subjects. Generally speaking,
    a registration object should only be constructed via other parts of the neuropythy library and
    not via Registration() directly.
    '''

    def __init__(self, topology, coordinates):
        self.topology = topology
        coordinates = np.array(coordinates)
        self.coordinates = np.asarray(
            [x/numpy.linalg.norm(x) 
             for x in (coordinates if coordinates.shape[1] == 3 else coordinates.T)])
        # here, we build the spatial has...
        # We use the centers of the triangles as the points we will search by
        self.triangle_centers = np.asarray([np.mean(self.coordinates[tri], 0) 
                                            for tri in topology.triangles])
        self.triangle_hash = space.cKDTree(self.triangle_centers)
        self.vertex_hash = space.cKDTree(self.coordinates.T)


    # True if the point is in the triangle, otherwise False; tri_no is an index into the triangle
    def _point_in_triangle(self, tri_no, pt):
        tri = self.coordinates[self.topology.triangles[tri_no]]
        ab = np.dot(pt - tri[0], np.cross(tri[0], tri[1] - tri[0]))
        bc = np.dot(pt - tri[1], np.cross(tri[1], tri[2] - tri[1]))
        ca = np.dot(pt - tri[2], np.cross(tri[2], tri[0] - tri[2]))
        return (ab >= 0 and bc >= 0 and ca >= 0)
        
    def interpolate_from(self, reg, data, 
                         smoothing=2, mask=None, null=None, method='automatic', n_jobs=1):
        '''
        registration.interpolate_from(reg, data) yields a numpy array of the data interpolated from
        the given array, data, which must contain the same number of elements as there are points in
        the Registration object reg, to the coordinates in the given Registration object registration.
        
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
        # Okay, switch on method:
        if method == 'nearest':
            return self._interpolate_nearest(reg, data, mask, null, n_jobs)
        elif method == 'automatic':
            return self._interpolate_linear(reg, data, mask, null, smoothing, 12, n_jobs)

    # perform nearest-neighbor interpolation
    def _interpolate_nearest(self, reg, data, mask, null, n_jobs):
        # lookup the neighbors...
        (d, nei) = reg.vertex_hash.query(self.coordinates, k=1) #n_jobs fails? version problem?
        return [reg[i] if mask[i] == 1 else null for i in nei]
    # perform linear interpolation
    def _interpolate_linear(self, reg, data, mask, null, smoothing, check_no, n_jobs):
        # first, find the triangle containing each point...
        tris = reg.topology.triangles
        data = np.asarray(data)
        ## we only query the nearest check_no triangles; otherwise we don't fine the container
        (d, near) = reg.triangle_hash.query(self.coordinates, k=check_no) #n_jobs fails?
        containers = [
            next((k for k in tri_nos if reg._point_in_triangle(k, x)), None)
            for (x, tri_nos) in zip(self.coordinates, near)]
        print len([x for x in containers if x is not None])
        # Okay, now we interpolate for each triangle
        if mask is None:
            return [null if tri_no is None \
                        else reg._interpolate_triangle(x, data, tris[tri_no], smoothing)
                    for (x, tri_no) in zip(self.coordinates, containers)]
        else:
            return [null if tri_no is None or any(mask[u] == 0 for u in tris[tri_no]) \
                        else reg._interpolate_triangle(x, data, tris[tri_no], smoothing)
                    for (x, tri_no) in zip(self.coordinates, containers)]
    def _interpolate_triangle(self, x, data, tri_vertices, smoothing):
        # we'll want to project things down to 2 dimensions:
        mtx = geo.alignment_matrix_3D(x, [0,0,1])[0:2].T
        # Project out what we don't want
        corners = np.dot(self.coordinates[tri_vertices], mtx)
        x = np.asarray([0,0])
        # get the mini-triangles' areas
        a_area = geo.triangle_area(x, corners[1], corners[2]) ** smoothing
        b_area = geo.triangle_area(x, corners[2], corners[0]) ** smoothing
        c_area = geo.triangle_area(x, corners[0], corners[1]) ** smoothing
        # and do the interpolation:
        return np.dot([a_area, b_area, c_area], data[tri_vertices]) / (a_area + b_area + c_area)
        
            
