####################################################################################################
# neuropythy/geometry/mesh.ph
# Tools for interpolating from meshes.
# By Noah C. Benson

import numpy as np
import numpy.linalg
import scipy as sp
import scipy.spatial as space
import os, math
from pysistence import make_dict

from neuropythy.immutable import Immutable
from .util import (triangle_area, triangle_address, alignment_matrix_3D)

class Mesh(Immutable):
    '''
    A Mesh object represents a triangle mesh in either 2D or 3D space.
    '''

    @staticmethod
    def __calculate_triangle_centers(triangles, coords):
        coords = np.asarray(coords)
        coords = coords if coords.shape[0] < coords.shape[1] else coords.T
        triangles = np.asarray(triangles)
        triangles = triangles if triangles.shape[0] < triangles.shape[1] else triangles.T
        return (coords[:,triangles[0]] + coords[:,triangles[1]] + coords[:,triangles[2]]).T / 3
        
    
    def __init__(self, triangles, coordinates):
        coordinates = coordinates if coordinates.shape[0] < 4 else coordinates.T
        triangles = triangles if triangles.shape[0] == 3 else triangles.T
        Immutable.__init__(
            self,
            {},
            {'coordinates': coordinates.T, 'triangles': triangles.T},
            {'triangle_centers': (('triangles','coordinates'),
                                  lambda t,x: Mesh.__calculate_triangle_centers(t, x)),
             'triangle_hash':    (('triangle_centers',), lambda x: space.cKDTree(x)),
             'vertex_hash':      (('coordinates',), lambda x: space.cKDTree(x))})
    def __repr__(self):
        return 'Mesh(<%d triangles>, <%d vertices>)' % (self.triangles.shape[0],
                                                        self.coordinates.shape[0])


    # True if the point is in the triangle, otherwise False; tri_no is an index into the triangle
    def _point_in_triangle(self, tri_no, pt):
        tri = self.coordinates[self.triangles[tri_no]]
        if len(pt) == 2:
            tol = 1e-13
            a = triangle_area(tri[0], tri[1], tri[2])
            if a == 0: return False
            a = 0.5 / a
            s = a * (tri[0,1]*tri[2,0] - tri[0,0]*tri[2,1] + 
                     (tri[2,1] - tri[0,1])*pt[0] + (tri[0,0] - tri[2,1])*pt[1])
            if (s + tol) < 0 or (s - tol) >= 1: return False
            t = a * (tri[0,0]*tri[1,1] - tri[0,1]*tri[1,0] + 
                     (tri[0,1] - tri[1,1])*pt[0] + (tri[1,0] - tri[0,0])*pt[1])
            return False if (t + tol) < 0 or (s + t - tol) > 1 else True
        else:
            return (np.dot(pt - tri[0], np.cross(tri[0], tri[1] - tri[0])) >= 0 and
                    np.dot(pt - tri[1], np.cross(tri[1], tri[2] - tri[1])) >= 0 and
                    np.dot(pt - tri[2], np.cross(tri[2], tri[0] - tri[2])) >= 0)

    def _find_triangle_search(self, x, k=24, searched=set([])):
        # This gets called when a container triangle isn't found; the idea is that k should
        # gradually increase until we find the container triangle; if k passes the max, then
        # we give up and assume no triangle is the container
        if k > 288: return None
        (d,near) = self.triangle_hash.query(x, k=k)
        near = [n for n in near if n not in searched]
        searched = searched.union(near)
        tri_no = next((k for k in near if self._point_in_triangle(k, x)), None)
        return (tri_no if tri_no is not None
                else self._find_triangle_search(x, k=(2*k), searched=searched))
        
    
    def nearest_vertex(self, pt):
        '''
        mesh.nearest_vertex(pt) yields the id number of the nearest vertex in the given
        mesh to the given point pt. If pt is an (n x dims) matrix of points, an id is given
        for each column of pt.
        '''
        (d,near) = self.vertex_hash.query(pt, k=1)
        return near
    def container(self, pt, k=2, n_jobs=1):
        '''
        mesh.container(pt) yields the id number of the nearest triangle in the given
        mesh to the given point pt. If pt is an (n x dims) matrix of points, an id is given
        for each column of pt.
        '''
        pt = np.asarray(pt)
        (d, near) = self.triangle_hash.query(pt, k=k) #n_jobs fails?
        if len(pt.shape) == 1:
            tri_no = next((k for k in near if self._point_in_triangle(k, pt)), None)
            return (tri_no if tri_no is not None
                    else self._find_triangle_search(pt, k=(2*k), searched=set(near)))
        else:
            return [(tri_no if tri_no is not None
                     else self._find_triangle_search(x, k=(2*k), searched=set(near_i)))
                    for (x, near_i) in zip(pt, near)
                    for tri_no in [next((k for k in near_i if self._point_in_triangle(k, x)),
                                        None)]]

    def interpolate(self, x, data, 
                    smoothing=1, mask=None, null=None, method='automatic', n_jobs=1):
        '''
        mesh.interpolate(x, data) yields a numpy array of the data interpolated from the given
        array, data, which must contain the same number of elements as there are points in the Mesh
        object mesh, to the coordinates in the given point matrix x. Note that if x is a vector
        instead of a matrix, then just one value is returned.
        
        The following options are accepted:
          * mask (default: None) indicates that the given True/False or 0/1 valued list/array should
            be used; any point whose nearest neighbor (see below) is in the given mask will, instead
            of an interpolated value, be set to the null value (see null option).
          * null (default: None) indicates the value that should be placed in the returned result if
            either a vertex does not lie in any triangle or a vertex is masked out via the mask
            option.
          * smoothing (default: 1) assuming that the method is 'interpolate' or 'automatic', this
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
        x = np.asarray(x)
        if len(x.shape) == 2:
            x = x if x.shape[1] == 2 or x.shape[1] == 3 else x.T
            sz = 2
            dims = x.shape[1]
        elif len(x.shape) == 1:
            sz = 1
            dims = x.shape[0]
            x = np.asarray([x])
        else:
            raise ValueError('interpolation points must be a matrix or vector')
        if dims != self.coordinates.shape[1]:
            raise ValueError('interpolation points have wrong dimensionality for mesh')
        # Okay, switch on method:
        if method == 'nearest':
            return self._interpolate_nearest(x, data, mask, null, n_jobs)
        elif method == 'automatic':
            return self._interpolate_linear(x, data, mask, null, smoothing, 12, n_jobs)

    # perform nearest-neighbor interpolation
    def _interpolate_nearest(self, x, data, mask, null, n_jobs):
        # lookup the neighbors...
        (d, nei) = self.vertex_hash.query(x, k=1) #n_jobs fails? version problem?
        if mask is None:
            return [data[i] for i in nei]
        else:
            return [data[i] if mask[i] == 1 else null for i in nei]
    # perform linear interpolation
    def _interpolate_linear(self, coords, data, mask, null, smoothing, check_no, n_jobs):
        # first, find the triangle containing each point...
        tris = self.triangles
        data = np.asarray(data)
        ## we only query the nearest check_no triangles; otherwise we don't fine the container
        containers = self.container(coords, k=check_no, n_jobs=n_jobs)
        # Okay, now we interpolate for each triangle
        if mask is None:
            return [(null if tri_no is None
                     else self._interpolate_triangle(x, data, tris[tri_no], smoothing))
                    for (x, tri_no) in zip(coords, containers)]
        else:
            return [(null if tri_no is None or any(mask[u] == 0 for u in tris[tri_no])
                     else self._interpolate_triangle(x, data, tris[tri_no], smoothing))
                    for (x, tri_no) in zip(coords, containers)]
    def _interpolate_triangle(self, x, data, tri_vertices, smoothing):
        # we'll want to project things down to 2 dimensions:
        if len(x) == 3:
            mtx = alignment_matrix_3D(x, [0,0,1])[0:2].T
            # Project out what we don't want
            corners = np.dot(self.coordinates[tri_vertices], mtx)
            x = np.asarray([0,0])
        else:
            corners = self.coordinates[tri_vertices]
        # get the mini-triangles' areas
        a_area = triangle_area(x, corners[1], corners[2]) ** smoothing
        b_area = triangle_area(x, corners[2], corners[0]) ** smoothing
        c_area = triangle_area(x, corners[0], corners[1]) ** smoothing
        tot = a_area + b_area + c_area
        # and do the interpolation:
        if np.isclose(tot, 0):
            return None
        else:
            return np.dot([a_area, b_area, c_area], data[tri_vertices]) / tot
        

    def address(self, data):
        '''
        mesh.address(X) yields a dictionary containing the address or addresses of the point or
        points given in the vector or coordinate matrix X. Addresses specify a single unique 
        topological location on the mesh such that deformations of the mesh will address the same
        points differently. To convert a point from one mesh to another isomorphic mesh, you can
        address the point in the first mesh then unaddress it in the second mesh.
        '''
        # we have to have a topology and registration for this to work...
        data = np.asarray(data)
        if len(data.shape) == 1:
            face_id = self.container(data)
            (t, r) = triangle_address(self.coordinates[self.triangles[face_id]],
                                      data)
        else:
            data = data if data.shape[1] == 3 else data.T
            face_id = self.container(data)
            (t, r) = np.asarray([triangle_address(self.coordinates[tri], x)
                                 for (tri,x) in zip(self.triangles[face_id], data)]).T
        # And return the dictionary
        return {'face_id': face_id, 'angle_fraction': t, 'distance_fraction': r}

