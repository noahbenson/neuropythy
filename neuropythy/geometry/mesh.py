####################################################################################################
# neuropythy/geometry/mesh.ph
# Tools for interpolating from meshes.
# By Noah C. Benson

import numpy as np
import scipy as sp
import scipy.spatial as space
import os, math
from pysistence import make_dict
from numpy.linalg import norm

from neuropythy.immutable import Immutable
from .util import (triangle_area, triangle_address, alignment_matrix_3D,
                   cartesian_to_barycentric_3D, cartesian_to_barycentric_2D,
                   barycentric_to_cartesian, point_in_triangle)

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

    @staticmethod
    def __calculate_triangle_normals(triangles, coords):
        coords = np.asarray(coords)
        coords = coords if coords.shape[0] < coords.shape[1] else coords.T
        triangles = np.asarray(triangles)
        triangles = triangles if triangles.shape[0] < triangles.shape[1] else triangles.T
        tmp = coords[:, triangles[0]]
        u01 = coords[:, triangles[1]] - tmp
        u02 = coords[:, triangles[2]] - tmp
        xp = np.cross(u01, u02, axisa=0, axisb=0)
        xpnorms = np.sqrt(np.sum(xp**2, axis=1))
        zero = np.isclose(xpnorms, 0)
        zero_idcs = np.where(zero)
        xp[zeros_idcs,:] = 0
        return xp / (xpnorms + zero)
        
    
    def __init__(self, triangles, coordinates):
        coordinates = coordinates if coordinates.shape[0] < 4 else coordinates.T
        triangles = triangles if triangles.shape[0] == 3 else triangles.T
        Immutable.__init__(
            self,
            {},
            {'coordinates': coordinates.T, 'triangles': triangles.T},
            {'triangle_centers': (('triangles','coordinates'),
                                  lambda t,x: Mesh.__calculate_triangle_centers(t, x)),
             'triangle_normals': (('triangle','coordinates'),
                                  lambda t,x: Mesh.__calculate_triangle_normals(t, x)),
             'triangle_hash':    (('triangle_centers',), lambda x: space.cKDTree(x)),
             'vertex_hash':      (('coordinates',), lambda x: space.cKDTree(x))})
    def __repr__(self):
        return 'Mesh(<%d triangles>, <%d vertices>)' % (self.triangles.shape[0],
                                                        self.coordinates.shape[0])


    # True if the point is in the triangle, otherwise False; tri_no is an index into the triangle
    def _point_in_triangle(self, tri_no, pt):
        tri = self.coordinates[self.triangles[tri_no]]
        return point_in_triangle(tri, pt)

    def _find_triangle_search(self, x, k=24, searched=set([])):
        # This gets called when a container triangle isn't found; the idea is that k should
        # gradually increase until we find the container triangle; if k passes the max, then
        # we give up and assume no triangle is the container
        if k >= 288: return None
        (d,near) = self.triangle_hash.query(x, k=k)
        near = [n for n in near if n not in searched]
        searched = searched.union(near)
        tri_no = next((kk for kk in near if self._point_in_triangle(kk, x)), None)
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

    def point_in_plane(self, tri_no, pt):
        '''
        r.point_in_plane(id, pt) yields the distance from the plane of the id'th triangle in the
        registration r to the given pt and the point in that plane as a tuple (d, x).
        '''
        tx = self.coordinates[self.triangles[tri_no]]
        n = np.cross(tx[1] - tx[0], tx[2] - tx[0])
        n /= norm(n)
        d = np.dot(n, pt - tx[0])
        return (abs(d), pt - n*d)
    
    def nearest_data(self, pt, k=2, n_jobs=1):
        '''
        mesh.nearest_data(pt) yields a tuple (k, d, x) of the matrix x containing the point(s)
        nearest the given point(s) pt that is/are in the mesh; a vector d if the distances between
        the point(s) pt and x; and k, the face index/indices of the triangles containing the 
        point(s) in x.
        Note that this function and those of this class are made for spherical meshes and are not
        intended to work with other kinds of complex topologies; though they might work 
        heuristically.
        '''
        pt = np.asarray(pt, dtype=np.float32)
        if len(pt.shape) == 1:
            r = self.nearest_data([pt], k=k, n_jobs=n_jobs)[0];
            return (r[0][0], r[1][0], r[2][0])
        pt = pt if pt.shape[1] == 3 else pt.T
        (d, near) = self.triangle_hash.query(pt, k=k)
        ids = [tri_no if tri_no is not None else self._find_triangle_search(x, 2*k, set(near_i))
               for (x, near_i) in zip(pt, near)
               for tri_no in [next((k for k in near_i if self._point_in_triangle(k, x)), None)]]
        pips = [self.point_in_plane(i, p) if i is not None else (0, None)
                for (i,p) in zip(ids, pt)]
        return (np.asarray(ids),
                np.asarray([d[0] for d in pips]),
                np.asarray([d[1] for d in pips]))

    def nearest(self, pt, k=2, n_jobs=1):
        '''
        mesh.nearest(pt) yields the point in the given mesh nearest the given array of points pts.
        '''
        dat = self.nearest_data(pt)
        return dat[2]

    def distance(self, pt, k=2, n_jobs=1):
        '''
        mesh.distance(pt) yields the distance to the nearest point in the given mesh from the points
        in the given matrix pt.
        '''
        dat = self.nearest_data(pt)
        return dat[1]

    def container(self, pt, k=2, n_jobs=1):
        '''
        mesh.container(pt) yields the id number of the nearest triangle in the given
        mesh to the given point pt. If pt is an (n x dims) matrix of points, an id is given
        for each column of pt.
        '''
        pt = np.asarray(pt, dtype=np.float32)
        (d, near) = self.triangle_hash.query(pt, k=k) #n_jobs fails?
        if len(pt.shape) == 1:
            tri_no = next((kk for kk in near if self._point_in_triangle(kk, pt)), None)
            return (tri_no if tri_no is not None
                    else self._find_triangle_search(pt, k=(2*k), searched=set(near)))
        else:
            return [(tri_no if tri_no is not None
                     else self._find_triangle_search(x, k=(2*k), searched=set(near_i)))
                    for (x, near_i) in zip(pt, near)
                    for tri_no in [next((kk for kk in near_i if self._point_in_triangle(kk, x)),
                                        None)]]

    def interpolate(self, x, data, 
                    smoothing=1, mask=None, null=None, method='automatic', n_jobs=1,
                    container_ids=None):
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
        data = np.asarray(data)
        data_t = (data.shape[0] != self.coordinates.shape[0])
        if data_t: data = data.T
        # Okay, switch on method:
        if method == 'nearest':
            data = self._interpolate_nearest(x, data, mask, null, n_jobs)
        elif method == 'automatic':
            data = self._interpolate_linear(x, data, mask, null, smoothing, 12, n_jobs)
        data = np.asarray(data)
        return data.T if data_t else data

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
        ## we only query the nearest check_no triangles; otherwise we don't fine the container
        containers = self.container(coords, k=check_no, n_jobs=n_jobs)
        # Okay, now we interpolate for each triangle
        nulls = [null for _ in range(data.shape[1])] if len(data.shape) > 1 else null
        if mask is None:
            return [(nulls if tri_no is None
                     else self._interpolate_triangle(x, data, tris[tri_no], smoothing, nulls))
                    for (x, tri_no) in zip(coords, containers)]
        else:
            return [(nulls if tri_no is None or any(mask[u] == 0 for u in tris[tri_no])
                     else self._interpolate_triangle(x, data, tris[tri_no], smoothing, nulls))
                    for (x, tri_no) in zip(coords, containers)]
    def _interpolate_triangle(self, x, data, tri_vertices, smoothing, nulls):
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
            return nulls
        else:
            if len(data.shape) == 1:
                return np.dot([a_area, b_area, c_area], data[tri_vertices]) / tot
            else:
                return np.dot([[a_area, b_area, c_area]], data[tri_vertices])[0] / tot

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
            if face_id is None: return None
            tx = self.coordinates[self.triangles[face_id]]
        else:
            data = data if data.shape[1] == 3 or data.shape[1] == 2 else data.T
            face_id = self.container(data)
            faces = self.triangles[face_id]
            tx = np.transpose(np.asarray([self.coordinates[f] for f in faces]), (0,1,2))
        bc = cartesian_to_barycentric_3D(tx, data) if self.coordinates.shape[1] == 3 else \
             cartesian_to_barycentric_2D(tx, data)
        return {'face_id': face_id, 'coordinates': bc}

    def unaddress(self, data):
        '''
        mesh.unaddress(A) yields a coordinate matrix that is the result of unaddressing the given
        address dictionary A in the given mesh. See also mesh.address.
        '''
        if not isinstance(data, dict):
            raise ValueError('address data must be a dictionary')
        if 'face_id' not in data: raise ValueError('address must contain face_id')
        if 'coordinates' not in data: raise ValueError('address must contain coordinates')
        face_id = data['face_id']
        coords = data['coordinates']
        if all(hasattr(x, '__iter__') for x in (face_id, coords)):
            faces = self.triangles[face_id].T
            tx = np.transpose(np.asarray([self.coordinates[f] for f in faces]), (0,2,1))
        else:
            tx = np.asarray(self.coordinates[self.triangles[face_id]])
        return barycentric_to_cartesian(tx, coords)


