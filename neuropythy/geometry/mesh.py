####################################################################################################
# neuropythy/geometry/mesh.ph
# Tools for interpolating from meshes.
# By Noah C. Benson

import numpy as np
import numpy.matlib as npml
import scipy as sp
import scipy.spatial as space
import scipy.sparse  as sps
import os, math, itertools
import pyrsistent as pyr
import pimms
from numpy.linalg import norm
from .util import (triangle_area, triangle_address, alignment_matrix_3D,
                   cartesian_to_barycentric_3D, cartesian_to_barycentric_2D,
                   barycentric_to_cartesian, point_in_triangle)
if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls

@pimms.immutable
class TesselationIndex(object):
    '''
    TesselationIndex is an immutable helper-class for Tesselation. The TesselationIndex handles
    requests to obtain indices of vertices, edges, and faces in a tesselation object. Generally,
    this is done via the __getitem__ (index[item]) method. In the case that you wish to obtain the
    vertex indices for an edge or face but don't wish to obtain the index of the edge or face
    itself, the __call__ (index(item)) method can be used.
    '''

    def __init__(self, vertex_index, edge_index, face_index):
        self.vertex_index = vertex_index
        self.edge_index = edge_index
        self.face_index = face_index

    @pimms.param
    def vertex_index(vi):
        return vi
    @pimms.param
    def edge_index(ei):
        return ei
    @pimms.param
    def face_index(fi):
        return fi
    
    def __repr__(self):
            return "TesselationIndex(<%d vertices>)" % len(self.vertex_index)
    def __getitem__(self, index):
        if isinstance(index, tuple):
            if   len(index) == 3: return self.face_index.get(index, None)
            elif len(index) == 2: return self.edge_index.get(index, None)
            elif len(index) == 1: return self.vertex_index.get(index[0], None)
            else:                 raise ValueError('Unrecognized tesselation item: %s' % index)
        elif isinstance(index, colls.Set):
            return {k:self[k] for k in index}
        elif pimms.is_vector(index):
            vi = self.vertex_index
            return np.asarray([vi[k] for k in index])
        elif pimms.is_matrix(index):
            m = np.asarray(index)
            if m.shape[0] != 2 and m.shape[0] != 3: m = m.T
            idx = self.edge_index if m.shape[0] == 2 else self.face_index
            return pimms.imm_array([idx[k] for k in zip(*m)])
        else:
            return self.vertex_index[index]
    def __call__(self, index):
        vi = self.vertex_index
        if isinstance(index, tuple):
            return tuple([vi[i] for i in index])
        elif isinstance(index, colls.Set):
            return set([vi[k] for k in index])
        elif pimms.is_vector(index):
            return np.asarray([vi[k] for k in index])
        elif pimms.is_matrix(index):
            return np.asarray([[idx[k] for k in u] for u in index])
        else:
            return self.vertex_index[index]

@pimms.immutable
class Tesselation(object):
    '''
    A Tesselation object represents a triangle mesh with no particular coordinate embedding.
    '''
    def __init__(faces, properties=None, meta_data=None):
        self.faces = faces
        self._properties = properties
        self.meta_data = meta_data

    # The immutable parameters:
    @pimms.param
    def faces(tris):
        '''
        tess.faces is a read-only numpy integer matrix of the triangle indices that make-up; the
          given tesselation object; the matrix is (3 x m) where m is the number of triangles, and
          the cells are valid indices into the rows of the coordinates matrix.
        '''
        tris = pimms.imm_array(np.asarray(tris, dtype=np.int))
        if tris.shape[0] != 3:
            tris = tris.T
            if tris.shape[0] != 3:
                raise ValueError('faces must be a (3 x m) or (m x 3) matrix')
        return tris
    @pimms.param
    def _properties(props):
        '''
        tess.properties is a pimms ITable instance containing the vertex properties of the given
          tesselation object.
        '''
        if props is None: return pimms.itable()
        if pimms.is_itable(props): return props
        elif pimms.is_map(props): return pimms.itable(props)
        else: raise ValueError('tesselation _properties must be a mapping')
    @pimms.param
    def meta_data(md):
        '''
        tess.meta_data is a persistent map of meta-data provided to the given tesselation.
        '''
        return pyr.pmap(md)

    # The immutable values:
    @pimms.value
    def vertex_labels(faces):
        '''
        tess.vertex_labels is an array of the integer vertex labels; by default this is equivalent
        to range(tess.vertex_count), but sub-sampling of tesselations may cause this to be
        different.
        '''
        return np.unique(faces)
    @pimms.value
    def vertex_count(vertex_labels):
        '''
        tess.vertex_count is the number of vertices in the given tesselation.
        '''
        return len(vertex_labels)
    @pimms.value
    def face_count(faces):
        '''
        tess.face_count is the number of faces in the given tesselation.
        '''
        return faces.shape[1]
    @pimms.value
    def face_index(faces):
        '''
        tess.face_index is a mapping that indexes the faces by vertex labels (not vertex indices).
        '''
        idx = {}
        for (a,b,c,i) in zip(faces[0], faces[1], faces[2], range(faces.shape[1])):
            idx[(a,b,c)] = i
            idx[(b,c,a)] = i
            idx[(c,b,a)] = i
            idx[(a,c,b)] = i
            idx[(b,a,c)] = i
            idx[(c,a,b)] = i
        return pyr.pmap(idx)
    @pimms.value
    def edge_data(faces):
        '''
        tess.edge_data is a mapping of data relevant to the edges of the given tesselation.
        '''
        limit = np.max(faces) + 1
        edge2face = {}
        idx = {}
        edge_list = [None for i in range(3*faces.size)]
        k = 0
        rng = range(faces.shape[1])
        for (e,i) in zip(
            zip(np.concatenate((faces[0], faces[1], faces[2])),
                np.concatenate((faces[1], faces[2], faces[0]))),
            np.concatenate((rng, rng, rng))):
            e = tuple(sorted(e))
            if e in idx:
                edge2face[e].append(i)
            else:
                idx[e] = k
                idx[e[::-1]] = k
                edge_list[k] = e
                edge2face[e] = [i]
                k += 1
        edge2face = {k:tuple(v) for (k,v) in six.iteritems(edge2face)}
        for ((a,b),v) in six.iteritems(edge2face):
            edge2face[(b,a)] = v
        return pyr.m(edges=pimms.imm_array(np.transpose(edge_list[0:k])),
                     edge_index=pyr.pmap(idx),
                     edge_face_index=pyr.pmap(edge2face))
    @pimms.value
    def edges(edge_data):
        '''
        tess.edges is a (2 x p) numpy array containing the p edge pairs that are included in the
        given tesselation.
        '''
        return edge_data['edges']
    @pimms.value
    def edge_count(edges):
        '''
        tess.edge_count is the number of edges in the given tesselation.
        '''
        return edges.shape[1]
    @pimms.value
    def edge_index(edge_data):
        '''
        tess.edge_index is a mapping that indexes the edges by vertex labels (not vertex indices).
        '''
        return edge_data['edge_index']
    @pimms.value
    def edge_face_index(edge_data):
        '''
        tess.edge_face_index is a mapping that indexes the edges by vertex labels (not vertex
          indices) to a face index or pair of face indices. So for an edge from the vertex labeled
          u to the vertex labeled v, index.edge_face_index[(u,v)] is a tuple of the faces that are
          adjacent to the edge (u,v).
        '''
        return edge_data['edge_face_index']
    @pimms.value
    def vertex_index(vertex_labels):
        '''
        tess.vertex_index is an index of vertex-label to vertex index for the given tesselation.
        '''
        return pyr.pmap({v:i for (i,v) in enumerate(vertex_labels)})
    @pimms.value
    def index(vertex_index, edge_index, face_index):
        '''
        tess.index is a TesselationIndex object that indexed the faces, edges, and vertices in the
        given tesselation object. Vertex, edge, and face indices can be looked-up using the
        following syntax:
          # Note that in all of these, u, v, and w are vertex *labels*, not vertex indices:
          tess.index[u]       # for vertex-label u => vertex-index of u
          tess.index[(u,v)]   # for edge (u,v) => edge-index of (u,v)
          tess.index[(u,v,w)] # for face (u,v,w) => face-index of (u,v,w)
        Alternately, one may want to obtain the vertex indices of an object without obtaining the
        indices for it directly:
          # Assume ui,vi,wi are the vertex indices of the vertex labels u,v,w:
          tess.index(u)       # for vertex-label u => ui
          tess.index((u,v))   # for edge (u,v) => (ui,vi)
          tess.index((u,v,w)) # for face (u,v,w) => (ui,vi,wi)
        Finally, in addition to passing individual vertices or tuples, you may pass an appropriately
        sized vector (for vertices) or matrix (for edges and faces), and the result will be a list
        of the appropriate indices or an identically-sized array with the vertex indices.
        '''
        return TesselationIndex(vertex_index, edge_index, face_index)
    @pimms.value
    def indexed_edges(edges, vertex_index):
        '''
        tess.indexed_edges is identical to tess.edges except that each element has been indexed.
        '''
        return pimms.imm_array([[vertex_index[u] for u in row] for row in edges])
    @pimms.value
    def indexed_faces(faces, vertex_index):
        '''
        tess.indexed_faces is identical to tess.faces except that each element has been indexed.
        '''
        return pimms.imm_array([[vertex_index[u] for u in row] for row in faces])
    @pimms.value
    def vertex_edge_index(vertex_labels, edges):
        '''
        tess.vertex_edge_index is a map whose keys are vertices and whose values are tuples of the
        edge indices of the edges that contain the relevant vertex.
        '''
        d = {k:[] for _ in vertex_labels}
        for (i,(u,v)) in enumerate(edges.T):
            d[u].append(i)
            d[v].append(i)
        return pyr.pmap({k:tuple(v) for (k,v) in six.iteritems(d)})
    @pimms.value
    def vertex_face_index(vertex_labels, faces):
        '''
        tess.vertex_face_index is a map whose keys are vertices and whose values are tuples of the
        edge indices of the faces that contain the relevant vertex.
        '''
        d = {k:[] for _ in vertex_labels}
        for (i,(u,v,w)) in enumerate(faces.T):
            d[u].append(i)
            d[v].append(i)
            d[w].append(i)
        return pyr.pmap({k:tuple(v) for (k,v) in six.iteritems(d)})
    @pimms.value
    def vertex_faces(vertex_labels, vertex_face_index):
        '''
        tess.vertex_faces is a tuple whose elements are tuples of the edge indices of the faces
        that contain the relevant vertex.
        '''
        return tuple([vertex_face_index[u] for u in vertex_labels])
    @staticmethod
    def _order_neighborhood(edges):
        res = [edges[0][1]]
        for i in range(len(edges)):
            for e in edges:
                if e[0] == res[i]:
                    res.append(e[1])
                    break
        return tuple(res)
    @pimms.value
    def neighborhoods(vertex_labels, faces, vertex_faces):
        '''
        tess.neighborhoods is a tuple whose contents are the neighborhood of each vertex in the
        tesselation.
        '''
        faces = faces.T
        nedges = [[((f[0], f[1]) if f[2] == u else (f[1], f[2]) if f[0] == u else (f[2], f[0]))
                   for f in faces[fs]]
                  for (u, fs) in zip(vertex_labels, vertex_faces)]
        return tuple([Tesselation._order_neighborhood(nei) for nei in nedges])
    @pimms.value
    def indexed_neighborhoods(vertex_index, neighborhoods):
        '''
        tess.indexed_neighborhoods is a tuple whose contents are the neighborhood of each vertex in
        the given tesselation; this is identical to tess.neighborhoods except this gives the vertex
        indices where tess.neighborhoods gives the vertex labels.
        '''
        return tuple([tuple([vertex_index[u] for u in nei]) for nei in neighborhoods])
    @pimms.require
    def validate_properties(vertex_count, _properties):
        '''
        tess.validate_properties requres that all non-builtin properties have the same number of
          entries as the there are vertices in the tesselation.
        '''
        if len(_properties.column_names) == 0:
            return True
        if vertex_count != _properties.row_count:
            raise ValueError('_properties does not have the correct number of entries')
        return True

    # Normal Methods
    def __repr__(self):
        return 'Tesselation(<%d triangles>, <%d vertices>)' % (self.face_count, self.vertex_count)

    # functions for handling properties...
    def prop(self, name):
        '''
        tess.prop(name) yields the vertex property in the given tesselation with the given name.
        tess.prop(data) yields data if data is a valid vertex property list for the given tess.
        tess.prop([p1, p2...]) yields a (d x n) vector of properties where d is the number of
          properties given and n is tess.vertex_count.
        tess.prop(set([name1, name2...])) yields a mapping of the given names mapped to the
          appropriate property values.
        '''
        if pimms.is_str(name):
            return self.properties[name]
        elif isinstance(name, colls.Set):
            return pyr.pmap({nm:self.properties[nm] for nm in name})
        elif pimms.is_vector(name):
            if len(name) == self.vertex_count:
                return name
            else:
                return np.asarray([self[nm] for nm in name])
        else:
            raise ValueError('unrecognized property')
    def with_prop(self, *args, **kwargs):
        '''
        tess.with_prop(...) yields a duplicate of the given tesselation with the given properties
          added to it. The properties may be specified as a sequence of mapping objects followed by
          any number of keyword arguments, all of which are merged into a single dict left-to-right
          before application.
        '''
        pp = self._properties.merge(*args, **kwargs)
        return self if pp is self._properties else self.copy(_properties=pp)
    def wout_prop(self, *args):
        '''
        tess.wout_property(...) yields a duplicate of the given tesselation with the given
          properties removed from it. The properties may be specified as a sequence of column names
          or lists of column names.
        '''
        pp = self._properties
        for a in args:
            if pimms.is_vector(a):
                for u in a:
                    pp = pp.discard(u)
            else:
                pp = pp.discard(a)
        return self if pp is self._properties else self.copy(_properties=pp)

    def subtess(self, vertices, tag=None):
        '''
        tess.subtess(vertices) yields a sub-tesselation of the given tesselation object that only
          contains the given vertices, which may be specified as a boolean vector or as a list of
          vertex labels. Faces and edges are trimmed automatically, but the vertex labels for the
          new vertices remain the same as in the original graph.
        The optional argument tag may be set to True, in which case the new tesselation's meta-data
        will contain the key 'supertess' whose value is the original tesselation tess; alternately
        tag may be a string in which case it is used as the key name in place of 'supertess'.
        '''
        vertices = np.asarray(vertices)
        if len(vertices) != self.vertex_count or \
           not np.array_equal(vertices, np.asarray(vertices, np.bool)):
            tmp = self.index(vertices)
            vertices = np.zeros(self.vertex_count)
            vertices[tmp] = 1
        vidcs = np.asarray(range(self.vertex_count))[vertices]
        if len(vidcs) == len(vertices): return self
        fsum = np.sum([vertices[f] for f in self.indexed_faces], axis=0)
        fids = np.where(fsum == 3)[0]
        faces = self.faces[:,fids]
        props = self.properties[vidcs]
        md = self.meta_data.set(tag, self) if pimms.is_str(tag)   else \
             self.meta_data.set('supertess', self) if tag is True else \
             self.meta_data
        dat = {'faces': faces, '_properties': props}
        if md is not self.meta_data: dat['meta_data'] = md
        return self.copy(**opts)
    
@pimms.immutable
class Mesh(object):
    '''
    A Mesh object represents a triangle mesh in either 2D or 3D space.
    '''

    def __init__(self, faces, coordinates, meta_data=None, properties=None):
        self.coordinates = coordinates
        self.tess = faces
        self.meta_data = meta_data
        self._properties = properties

    # The immutable parameters:
    @pimms.param
    def coordinates(crds):
        '''
        mesh.coordinates is a read-only numpy array of size (d x n) where d is the number of
          dimensions and n is the number of vertices in the mesh.
        '''
        crds = pimms.imm_array(crds)
        if crds.shape[0] != 2 and crds.shape[0] != 3:
            crds = crds.T
            if crds.shape[0] != 2 and crds.shape[0] != 3:
                raise ValueError('coordinates must be a (d x n) or (n x d) array where d is 2 or 3')
        return crds
    @pimms.param
    def tess(tris):
        '''
        mesh.tess is the Tesselation object that represents the triangle tesselation of the given
        mesh object.
        '''
        if not isinstance(tris, Tesselation):
            try: tris = Tesselation(tris)
            except: raise ValueError('mesh.tess must be a Tesselation object')
        return tris
    @pimms.param
    def meta_data(md):
        '''
        mesh.meta_data is a persistent map of meta-data provided to the given mesh.
        '''
        return pyr.pmap(md)
    @pimms.param
    def _properties(props):
        '''
        mesh._properties is the properties argument that was given the the mesh constructor;
        see mesh.properties.
        '''
        if props is None: return pimms.itable()
        elif pimms.is_itable(props): return props
        elif pimms.is_map(props): return pimms.itable(props)
        else: raise ValueError('tesselation _properties must be a mapping')

    # The immutable values:
    @pimms.param
    def properties(tess, _properties):
        '''
        mesh.properties is the pimms Itable object of properties known to the given mesh.
        '''
        return pimms.merge(tess.properties, _properties)
    @pimms.value
    def edge_coordinates(tess, coordinates):
        '''
        mesh.edge_coordinates is the (2 x d x p) array of the coordinates that define each edge in
          the given mesh; d is the number of dimensions that define the vertex positions in the mesh
          and p is the number of edges in the mesh.
        '''
        return pimms.imm_array([coordinates[:,e] for e in tess.edges])
    @pimms.value
    def face_coordinates(tess, coordinates):
        '''
        mesh.face_coordinates is the (3 x d x m) array of the coordinates that define each face in
          the given mesh; d is the number of dimensions that define the vertex positions in the mesh
          and m is the number of triange faces in the mesh.
        '''
        return pimms.imm_array([coordinates[:,f] for f in tess.faces])
    @pimms.value
    def edge_centers(edge_coordinates):
        '''
        mesh.edge_centers is the (d x n) array of the centers of each edge in the given mesh.
        '''
        return pimms.imm_array(0.5 * np.sum(edge_coordinates, axis=0))
    @pimms.value
    def face_centers(face_coordinates):
        '''
        mesh.face_centers is the (d x n) array of the centers of each triangle in the given mesh.
        '''
        return pimms.imm_array(np.sum(face_coordinates, axis=0) / 3.0)
    @pimms.value
    def face_normals(face_coordinates):
        '''
        mesh.face_normals is the (3 x m) array of the outward-facing normal vectors of each
          triangle in the given mesh. If mesh is a 2D mesh, these are all either [0,0,1] or
          [0,0,-1].
        '''
        u01 = face_coordinates[1] - face_coordinates[0]
        u02 = face_coordinates[2] - face_coordinates[0]
        if len(u01) == 2:
            zz = np.zeros((1,tmp.shape[1]))
            u01 = np.concatenate((u01,zz))
            u02 = np.concatenate((u02,zz))
        xp = np.cross(u01, u02, axisa=0, axisb=0).T
        xpnorms = np.sqrt(np.sum(xp**2, axis=0))
        zero = np.isclose(xpnorms, 0)
        zero_idcs = np.where(zero)
        xp[:,zeros_idcs] = 0
        return pimms.imm_array(xp / (xpnorms + zero))

    @pimms.value
    def face_angle_cosines(face_coordinates):
        '''
        mesh.face_angle_cosines is the (3 x d x n) matrix of the cosines of the angles of each of
        the faces of the mesh; d is the number of dimensions of the mesh embedding and n is the
        number of faces in the mesh.
        '''
        X = face_coordinates
        X = np.asarray([x * (zs / (xl + (~zs)))
                        for x  in [X[1] - X[0], X[2] - X[1], X[0] - X[2]]
                        for xl in [np.sqrt(np.sum(x**2, axis=0))]
                        for zs in [np.isclose(xl, 0)]])
        dps = [(X[0] * (-X[2])).sum(0),
               (X[1] * (-X[0])).sum(0),
               (X[2] * (-X[1])).sum(0)]
        return np.asarray(dps)
    @pimms.value
    def face_angles(face_angle_cosines):
        '''
        mesh.face_angles is the (3 x d x n) matrix of the angles of each of the faces of the mesh;
        d is the number of dimensions of the mesh embedding and n is the number of faces in the
        mesh.
        '''
        return np.arccos(face_angle_cosines)
    @pimms.value
    def face_areas(face_coordinates):
        '''
        mesh.face_areas is the length-m numpy array of the area of each face in the given mesh.
        '''
        return pimms.imm_array(triangle_area(*face_coordinates))
    @pimms.value
    def edge_lengths(edge_coordinates):
        '''
        mesh.edge_lengths is a numpy array of the lengths of each edge in the given mesh.
        '''
        return pimms.imm_array(
            np.sqrt(np.sum((edge_coordinates[1] - edge_coordinates[0])**2, axis=0)))
    @pimms.value
    def face_hash(face_centers):
        '''
        mesh.face_hash yields the scipy spatial hash of triangle centers in the given mesh.
        '''
        try:
            return space.cKDTree(face_centers.T)
        except:
            return space.KDTree(face_centers.T)
    @pimms.value
    def vertex_hash(coordinates):
        '''
        mesh.vertex_hash yields the scipy spatial hash of the vertices of the given mesh.
        '''
        try:
            return space.cKDTree(coordinates.T)
        except:
            return space.KDTree(coordinates.T)

    # requirements/validators
    @pimms.require
    def validate_tess(tess, coordinates):
        '''
        mesh.validate_tess requires that all faces be valid indices into mesh.coordinates.
        '''
        (d,n) = coordinates.shape
        if tess.vertex_count != n:
            raise ValueError('mesh coordinate matrix size does not match vertex count')
        if d != 2 and d != 3:
            raise ValueError('Only 2D and 3D meshes are supported')
        return True

    # Normal Methods
    def __repr__(self):
        return 'Mesh(<%dD>, <%d faces>, <%d vertices>)' % (self.coordinates.shape[0],
                                                           self.tess.face_count,
                                                           self.tess.vertex_count)

    # functions for handling properties...
    # functions for handling properties...
    def prop(self, name):
        '''
        mesh.prop(name) yields the vertex property in the given mesh with the given name.
        mesh.prop(data) yields data if data is a valid vertex property list for the given mesh.
        mesh.prop([p1, p2...]) yields a (d x n) vector of properties where d is the number of
          properties given and n is mesh.vertex_count.
        mesh.prop(set([name1, name2...])) yields a mapping of the given names mapped to the
          appropriate property values.
        '''
        if pimms.is_str(name):
            return self.properties[name]
        elif isinstance(name, colls.Set):
            return pyr.pmap({nm:self.properties[nm] for nm in name})
        elif pimms.is_vector(name):
            if len(name) == self.vertex_count:
                return name
            else:
                return np.asarray([self[nm] for nm in name])
        else:
            raise ValueError('unrecognized property')
    def with_prop(self, *args, **kwargs):
        '''
        mesh.with_prop(...) yields a duplicate of the given mesh object with the given properties
          added to it. The properties may be specified as a sequence of mapping objects followed by
          any number of keyword arguments, all of which are merged into a single dict left-to-right
          before application.
        '''
        pp = self._properties.merge(*args, **kwargs)
        return self if pp is self._properties else self.copy(_properties=pp)
    def wout_prop(self, *args):
        '''
        mesh.wout_property(...) yields a duplicate of the given mesh with the given properties
          removed from it. The properties may be specified as a sequence of column names or lists of
          column names.
        '''
        pp = self._properties
        for a in args:
            if pimms.is_vector(a):
                for u in a:
                    pp = pp.discard(u)
            else:
                pp = pp.discard(a)
        return self if pp is self._properties else self.copy(_properties=pp)
    
    # True if the point is in the triangle, otherwise False; tri_no is an index into the faces
    def is_point_in_face(self, tri_no, pt):
        '''
        mesh.is_point_in_face(face_id, crd) yields True if the given coordinate crd is in the face
          of the given mesh with the given face id (i.e., mesh.faces[:,face_id] gives the vertex
          indices for the face in question).
        The face_id and crd values may be lists as well; either they may be lists the same length
        or one may be a list.
        '''
        pt = np.asarray(pt)
        tri_no = np.asarray(tri_no)
        if len(tri_no) == 0:
            tri = self.coordinates[:, self.tess.faces[:, tri_no]]
        else:
            tri = np.transpose([self.coordinates[:,t] for t in self.tess.faces[:,tri_no]], (1,2,0))
        return point_in_triangle(tri, pt)

    def _find_triangle_search(self, x, k=24, searched=set([])):
        # This gets called when a container triangle isn't found; the idea is that k should
        # gradually increase until we find the container triangle; if k passes the max, then
        # we give up and assume no triangle is the container
        if k >= 288: return None
        (d,near) = self.facee_hash.query(x, k=k)
        near = [n for n in near if n not in searched]
        searched = searched.union(near)
        tri_no = next((kk for kk in near if self.is_point_in_face(kk, x)), None)
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
        If id or pt or both are lists, then the elements d and x of the returned tuple are a vector
        and matrix, respectively, where the size of x is (dims x n).
        '''
        tri_no = np.asarray(tri_no)
        pt = np.asarray(pt)
        if tri_no.shape is () and len(pt.shape) == 1:
            tx = self.coordinates[:, self.tess.faces[:, tri_no]]
            n = self.face_normals[:, tri_no]
            d = np.dot(n, pt - tx[0])
            return (np.abs(d), pt - n*d)
        if len(pt.shape) == 1:
            pt = npml.repmat(pt, len(tri_no), 1).T
        else:
            pt = pt.T if pt.shape[0] != self.coordinates.shape[0] else pt
            tri_no = np.full(pt.shape[1], tri_no, dtype=np.int)
        tx0 = self.coordinates[:,  self.tess.faces[0,tri_no]]
        n   = self.face_normals[:, tri_no]
        d   = np.sum(n * (pt - tx0), axis=0)
        return (np.abs(d), pt - n*d)
    
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
        pt = pt.T if pt.shape[0] == self.coordinates.shape[0] else pt
        (d, near) = self.face_hash.query(pt, k=k)
        ids = [tri_no if tri_no is not None else self._find_triangle_search(x, 2*k, set(near_i))
               for (x, near_i) in zip(pt, near)
               for tri_no in [next((k for k in near_i if self.is_point_in_face(k, x)), None)]]
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

    def nearest_vertex(self, x, n_jobs=1):
        '''
        mesh.nearest_vertex(x) yields the vertex index or indices of the vertex or vertices nearest
          to the coordinate or coordinates given in x.
        '''
        x = np.asarray(x)
        if len(x.shape) == 1: return self.nearest_vertex([x], n_jobs=n_jobs)[0]
        if x.shape[0] == self.coordinates.shape[0]: x = x.T
        n = self.coordinates.shape[1]
        (_, nei) = self.vertex_hash.query(x, k=1) #n_jobs fails? version problem?
        return nei

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

        Implementation Note:
          This method will fail to find the container triangle of a point if you have a very odd
          geometry; the requirement for this condition is that, for a point p contained in a
          triangle t with triangle center x0, there are at least n triangles whose centers are
          closer to p than x0 is to p. The value n depends on your starting parameter k, but is
          approximately 256.
        '''
        pt = np.asarray(pt, dtype=np.float32)
        if len(pt.shape) == 1:
            (d, near) = self.face_hash.query(pt, k=k) #n_jobs fails?
            tri_no = next((kk for kk in near if self.is_point_in_face(kk, pt)), None)
            return (tri_no if tri_no is not None
                    else self._find_triangle_search(pt, k=(2*k), searched=set(near)))
        else:
            tcount = self.tess.faces.shape[1]
            max_k = 256 if tcount > 256 else tcount
            if k > tcount: k = tcount
            def try_nearest(sub_pts, cur_k=k, top_i=0, near=None):
                res = np.full(len(sub_pts), None, dtype=np.object)
                if k != cur_k and cur_k > max_k: return res
                if near is None:
                    near = self.face_hash.query(sub_pts, k=cur_k)[1]
                # we want to try the nearest then recurse on those that didn't match...
                guesses = near[:, top_i]
                in_tri_q = self.is_point_in_face(guesses, sub_pts)
                res[in_tri_q] = guesses[in_tri_q]
                if in_tri_q.all(): return res
                # recurse, trying the next nearest points
                out_tri_q = ~in_tri_q
                sub_pts = sub_pts[out_tri_q]
                top_i += 1
                res[out_tri_q] = (try_nearest(sub_pts, cur_k*2, top_i, None)
                                  if top_i == cur_k else
                                  try_nearest(sub_pts, cur_k, top_i, near[out_tri_q]))
                return res
            res = np.full(len(pt), None, dtype=np.object)
            # filter out points that aren't close enough to be in a triangle:
            (dmins, dmaxs) = [[f(x[np.isfinite(x)]) for x in self.coordinates.T]
                              for f in [np.min, np.max]]
            finpts = np.isfinite(np.sum(pt, axis=1))
            if finpts.sum() == 0:
                inside_q = reduce(np.logical_and,
                                  [(x >= mn)&(x <= mx) for (x,mn,mx) in zip(pt.T,dmins,dmaxs)])
            else:
                inside_q = np.full(len(pt), False, dtype=np.bool)
                inside_q[finpts] = reduce(
                    np.logical_and,
                    [(x >= mn)&(x <= mx) for (x,mn,mx) in zip(pt[finpts].T,dmins,dmaxs)])
            if not inside_q.any(): return res
            res[inside_q] = try_nearest(pt[inside_q])
            return res

    @staticmethod
    def scale_interpolation(interp, mask=None, weights=None):
        '''
        Mesh.scale_interpolation(interp) yields a duplicate of interp in which all rows of the
          interpolation matrix have a sum of 1 or contain a nan element; those rows that correspond
          to interpolation points that are either not in the mask or have zero weight contributing
          to them will have a nan value somewhere that indicates that the interpolated value should
          always be nan.
        Mesh.scale_interpolation(interp, mask) additionally applies the given mask to the
          interpolation matrix; any interpolation point whose nearest neighbor is a vertex outside
          of the given mask will be assigned a value of numpy.nan upon interpolation with the
          resulting interpolation matrix.
        Mesh.scale_interpolation(interp, mask, weights) additionally applies the given vertex 
          weights to the interpolation matrix.

        The value mask may be specified as either None (no interpolation), a boolean array with the
        same number of elements as there are columns in interp, or a list of indices. The values in
        the mask are considered valid while the values not in the mask are considered invalid. The
        string value 'all' is also a valid mask (equivalent to no mask, or all elements in the
        mask).

        The interp argument should be a scipy.sparse.*_matrix; the object will not be modified
        in-place except to run eliminate_zeros(), and the returned matrix will always be of the
        csr_matrix type. Typical usage would be:
        interp_matrix = scipy.sparse.lil_matrix((n, self.vertex_count))
        # create your interpolation matrix here...
        return Mesh.rescale_interpolation(interp_matrix, mask_arg, weights_arg)
        '''
        interp = interp.tocsr()
        interp.eliminate_zeros()
        (m,n) = interp.shape # n: no. of vertices in mesh; m: no. points being interpolated
        # We apply weights first, because they are fairly straightforward:
        if weights is not None:
            wmtx = sps.li_matrix((n,n))
            weights = np.array(weights)
            weights[~np.isfinite(weights)] = 0
            weights[weights < 0] = 0
            wmtx.setdiag(weights)
            interp = interp.dot(wmtx.tocsc())
        # we make a mask with 1 extra element, always out of the mask, as a way to flag vertices
        # that shouldn't appear in the mask for some other reason
        if mask is None or (pimms.is_str(mask) and mask.lower() == 'all'):
            mask = np.ones(n + 1)
            mask[n] = 0
        else:
            mask_mtx = sps.lil_matrix((n,n))
            diag = np.zeros(n + 1)
            diag[mask] = 1
            mask = diag
            mask_mtx.setdiag(diag[:-1])
            interp = interp.dot(mask_mtx.tocsc())
            interp.eliminate_zeros()
        # we may need to rescale the rows now:
        (closest, rowdivs) = np.transpose(
            [(r.indices[np.argsort(r.data)[-1]], 1/ss) if np.isfinite(ss) and ss > 0 else (n,0)
             for r in interp
             for ss in [r.data.sum()]])
        rescale_mtx = sps.lil_matrix((n,n))
        rescale_mtx.setdiag(rowdivs)
        interp = interp.dot(rescale_mtx.tocsc())
        # any row with no interpolation weights or that is nearest to a vertex not in the mesh
        # needs to be given a nan value upon interpolation
        bad_pts = ~mask[closest]
        if bad_pts.sum() > 0:
            interp = interp.tolil()
            interp[bad_pts, 0] = np.nan
            interp = interp.tocsr()
        return interp
    def nearest_interpolation(self, coords, n_jobs=1):
        '''
        mesh.nearest_interpolation(x) yields an interpolation matrix for the given coordinate or
          coordinate matrix x. An interpolation matrix is just a sparce array M such that for a
          column vector u with the same number of rows as there are vertices in mesh, M * u is the
          interpolated values of u at the coordinates in x.
        '''
        coords = np.asarray(coords)
        if coords.shape[0] == self.coordinates.shape[0]: coords = coords.T
        n = self.coordinates.shape[1]
        m = coords.shape[0]
        mtx = sps.lil_matrix((m, n), dtype=np.float)
        nv = self.nearest_vertex(x, n_jobs=n_jobs)
        for (ii,u) in enumerate(nv):
            mtx[ii,u] = 1
        return mtx.tocsr()
    def linear_interpolation(self, coords, n_jobs=1):
        '''
        mesh.linear_interpolation(x) yields an interpolation matrix for the given coordinate or 
          coordinate matrix x. An interpolation matrix is just a sparce array M such that for a
          column vector u with the same number of rows as there are vertices in mesh, M * u is the
          interpolated values of u at the coordinates in x.
        '''
        coords = np.asarray(coords)
        if coords.shape[0] == self.coordinates.shape[0]: coords = coords.T
        n = self.coordinates.shape[1]
        m = coords.shape[0]
        mtx = sps.lil_matrix((m, n), dtype=np.float)
        tris = self.tess.faces
        # first, find the triangle containing each point...
        containers = self.container(coords, n_jobs=n_jobs)
        # which points are in a triangle at all...
        contained_q = np.asarray([x is not None for x in containers], dtype=np.bool)
        contained_idcs = np.where(contained_q)[0]
        containers = containers[contained_idcs].astype(np.int)
        # interpolate for these points
        tris = tris[:,containers]
        corners = np.transpose(self.face_coordinates[:,:,containers], (0,2,1))
        coords = coords[contained_idcs]
        # get the mini-triangles' areas
        a_area = triangle_area(coords, corners[1], corners[2])
        b_area = triangle_area(coords, corners[2], corners[0])
        c_area = triangle_area(coords, corners[0], corners[1])
        tot = a_area + b_area + c_area
        for (x,ii,f,aa,ba,ca,tt) in zip(coords, contained_idcs, tris.T, a_area,b_area,c_area,tot):
            if np.isclose(tt, 0):
                (aa,ba,ca) = np.sqrt(np.sum((coords[:,f].T - x)**2, axis=1))
                # check if we can do line interpolation
                (zab,zbc,zca) = np.isclose((aa,ba,ca), (ba,ca,aa))
                (aa,ba,ca) = (1.0,   1.0,      1.0) if zab and zbc and zca else \
                             (ca,     ca,    aa+ba) if zab                 else \
                             (ba+ca,  aa,       aa) if zbc                 else \
                             (ba,     aa+ca,    ba)
                tt = aa + ba + ca
            mtx[ii, f] = (aa/tt, ba/tt, ca/tt)
        return mtx.tocsr()
    def apply_interpolation(self, interp, data, mask=None, weights=None):
        '''
        mesh.apply_interpolation(interp, data) yields the result of applying the given interpolation
          matrix (should be a scipy.sparse.csr_matrix), which can be obtained via
          mesh.nearest_interpolation or mesh.linear_interpolation, and applies it to the given data,
          which should be matched to the coordinates used to create the interpolation matrix.

        The data argument may be a list/vector of size m (where m is the number of columns in the
        matrix interp), a matrix of size (n x m) for any n, or a map whose values are lists of
        length m.

        The following options may be provided:
          * mask (default: None) specifies which elements should be in the mask.
          * weights (default: None) additional weights that should be applied to the vertices in the
            mesh during interpolation.
        '''
        # we can start by applying the mask to the interpolation
        if mask is not None or weights is not None:
            interp = Mesh.scale_interpolation(interp, mask=mask, weights=weights)
        (m,n) = interp.shape
        # if data is a map, we iterate over its columns:
        if pimms.is_str(data):
            return self.apply_interpolation(
                interp,
                self.properties if data.lower() == 'all' else self.properties[data])
        elif pimms.is_lazy_map(data):
            def _make_lambda(kk): return lambda:self.apply_interpolation(interp, data[kk])
            return pimms.lazy_map({k:_make_lambda(k) for k in data.iterkeys()})
        elif pimms.is_map(data):
            return pyr.pmap({k:self.apply_interpolation(interp, data[k]) for k in data.iterkeys()})
        elif pimms.is_matrix(data):
            data = np.asarray(data)
            if data.shape[0] == n:
                return np.asarray([self.apply_interpolation(interp, row) for row in data.T]).T
            else:
                return np.asarray([self.apply_interpolation(interp, row) for row in data])
        elif pimms.is_vector(data) and len(data) != n:
            return tuple([self.apply_interpolation(interp, d) for d in data])
        # If we've made it here, we have a single vector to interpolate;
        # we might have non-finite values in this array (additions to the mask), so let's check:
        data = np.asarray(data)
        # numeric arrays can be handled relatively easily:
        if pimms.is_vector(data, np.number):
            numer = np.isfinite(data)
            if np.sum(numer) < n:
                # just remove these elements from the mask
                data = np.array(data)
                data[~numer] = 0
                interp = Mesh.scale_interpolation(interp, mask=numer)
            return interp.dot(data)
        # not a numerical array; we just do nearest interpolation
        return np.asarray(
            [data[r.indices[np.argsort(r.data)[-1]]] if np.isfinite(ss) and ss > 0 else np.nan
             for r in interp
             for ss in [r.data.sum()]])

    def interpolate(self, x, data, mask=None, weights=None, method='automatic', n_jobs=1):
        '''
        mesh.interpolate(x, data) yields a numpy array of the data interpolated from the given
        array, data, which must contain the same number of elements as there are points in the Mesh
        object mesh, to the coordinates in the given point matrix x. Note that if x is a vector
        instead of a matrix, then just one value is returned.
        
        The following options are accepted:
          * mask (default: None) indicates that the given True/False or 0/1 valued list/array should
            be used; any point whose nearest neighbor (see below) is in the given mask will, instead
            of an interpolated value, be set to the null value (see null option).
          * method (default: 'automatic') specifies what method to use for interpolation. The only
            currently supported methods are 'automatic', 'linear', or 'nearest'. The 'nearest'
            method does not  actually perform a nearest-neighbor interpolation but rather assigns to
            a destination vertex the value of the source vertex whose veronoi-like polygon contains
            the destination vertex; note that the term 'veronoi-like' is used here because it uses
            the Voronoi diagram that corresponds to the triangle mesh and not the true delaunay
            triangulation. The 'linear' method uses linear interpolation; though if the given data
            is non-numerical, then nearest interpolation is used instead. The 'automatic' method
            uses linear interpolation for any floating-point data and nearest interpolation for any
            integral or non-numeric data.
          * n_jobs (default: 1) is passed along to the cKDTree.query method, so may be set to an
            integer to specify how many processors to use, or may be -1 to specify all processors.
        '''
        if method is None: method = 'auto'
        method = method.lower()
        if method == 'linear':
            interp = self.linear_interpolation(x, n_jobs=n_jobs),
        elif method == 'nearest':
            return self.apply_interpolation(self.nearest_interpolation(x, n_jobs=n_jobs),
                                            data, mask=mask, weights=weights)
        elif method == 'auto' or method == 'automatic':
            # unique challenge; we want to calculate the interpolation matrices but once:
            interps = pimms.lazy_map(
                {'nearest': lambda:Mesh.scale_interpolation(
                    self.nearest_interpolation(x, n_jobs=n_jobs),
                    mask=mask, weights=weights),
                 'linear': lambda:Mesh.scale_interpolation(
                    self.linear_interpolation(x, n_jobs=n_jobs),
                    mask=mask, weights=weights)})
            # we now need to look over data...
            def _apply_interp(dat):
                if pimms.is_str(dat):
                    return _apply_interp(self.properties[dat])
                elif pimms.is_vector(dat, np.integer) or not pimms.is_vector(dat, np.number):
                    return self.apply_interpolation(interps['nearest'], dat)
                else:
                    return self.apply_interpolation(interps['linear'], dat)
            if pimms.is_str(data) and data.lower() == 'all':
                data = self.properties
            if pimms.is_map(data):
                return pyr.pmap({k:_apply_interp(data[k]) for k in data.iterkeys()})
            elif pimms.is_matrix(data):
                data = np.asarray(data)
                if data.shape[0] == n:
                    return np.asarray([_apply_interp(interp, np.asarray(row)) for row in data.T]).T
                else:
                    return np.asarray([_apply_interp(interp, row) for row in data])
            elif pimms.is_vector(data, np.number) and len(data) == self.tess.vertex_count:
                return _apply_interp(data)
            elif pimms.is_vector(data):
                return tuple([_apply_interp(d) for d in data])
            else:
                return _apply_interp(data)
        else:
            raise ValueError('method argument must be linear, nearest, or automatic')
        return self.apply_interpolation(interp, data, mask=mask, weights=weights)

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
            tx = self.coordinates[:,self.tess.faces[:,face_id]].T
        else:
            data = data if data.shape[1] == 3 or data.shape[1] == 2 else data.T
            face_id = np.asarray(self.container(data))
            faces = self.tess.faces
            null = np.full((faces.shape[0], self.coordinates.shape[0]), np.nan)
            tx = np.asarray([self.coordinates[:,faces[:,f]].T if f else null
                             for f in face_id]))
        bc = cartesian_to_barycentric_3D(tx, data) if self.coordinates.shape[0] == 3 else \
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
        faces = self.tess.faces
        if all(hasattr(x, '__iter__') for x in (face_id, coords)):
            null = np.full((faces.shape[0], self.coordinates.shape[0]), np.nan)
            tx = np.asarray([self.coordinates[:,faces[:,f]].T if f else null for f in face_id])
        elif face_id is None:
            return np.full(self.coordinates.shape[0], np.nan)
        else:
            tx = np.asarray(self.coordinates[:,self.tess.faces[:,face_id]].T)
        return barycentric_to_cartesian(tx, coords)


