####################################################################################################
# cortex.py
# Simple tools for dealing with the cortical surface in Python
# By Noah C. Benson

import numpy as np
import scipy as sp
import neuropythy.geometry as geo
from neuropythy.immutable import Immutable
import nibabel.freesurfer.io as fsio
import os
import itertools
import collections
from pysistence import make_dict
import pysistence

class CorticalMesh(Immutable):
    '''CorticalMesh is a class that handles properties of the cortical surface mesh. 
       The cortical mesh representation contains only three pieces of concrete data and many pieces
       of derived data. All derived data is lazily computed when requested then saved. Any changes
       to the concrete data results in the relevant derived data being cleared (and thus 
       recalculated on next request). Note that this will not work if you simply modify the numpy
       arrays, such as mesh.coordinates[:,10] = [0,0,0]; in this case, the function mesh.flush()
       or mesh.flush('coordinates') should be called to make sure that all cache is flushed.

       The concrete data members of CorticalMesh are:
         * coordinates: the x,y,z coordinates of the vertices; this matrix must be 2 or 3 by n
           where n is the number of vertices in the mesh.
         * faces: the integer triples that form the triangles of the mesh
         * options: a dictionary of all options to CorticalMesh; note that one of these options is
           \'meta_data\', which may be anything, but should usually be a dictionary in which user-
           defined data may be placed.

       The derived (lazy) data members of CorticalMesh are:
         * edges: a 2 x m numpy array of the edges in the mesh
         * face_angles: a 3 x q vector of the face angles
         * face_normals: a 3 x q vector of the normal vectors to each face

       CorticalMesh supports the following methods:
         * vertex_edges, vertex_faces, edge_faces: these functions, when given a vertex, vertex, or
           edge, respectively, yield a list of the edges, faces, or faces that the given item is a
           part of; for example, mesh.vertex_edges(u) yields the list of edges containing u.
         * select: mesh.select(f) can be used to select only those vertices from the mesh such that
           f(v) yields true; a new mesh object is returned.
         * add_property, remove_property, property_value, prop: these methods deal with the mesh's
           properties.
         * map_vertices: this function maps over each vertex in the mesh and provides the given
           function with a dictionary of the properties, coordinate, and label of each vertex.'''


    ################################################################################################
    # Nested Classes

    class Index:
        def __init__(self, vidx, eidx, fidx):
            self.vertex_index = vidx
            self.edge_index = eidx
            self.face_index = fidx
        def __repr__(self):
            return "CorticalMesh.Index(<%d vertices>)" % len(self.vertex_index)
        def __getitem__(self, index):
            if isinstance(index, tuple):
                if len(index) == 3:   return self.face_index.get(index, None)
                elif len(index) == 2: return self.edge_index.get(index, None)
                elif len(index) == 1: return self.vertex_index.get(index[0], None)
                else:                 raise ValueError('Unrecognized mesh item: %s' % index)
            elif isinstance(index, (list, set)):
                if len(index) > 3 or not all(isinstance(i,int) for i in index):
                    return map(lambda x: self[x], index)
                else:
                    return [
                        lambda idx: [],
                        lambda idx: self.vertex_index.get(idx[0], None),
                        lambda idx: self.edge_index.get(tuple(idx), None),
                        lambda idx: self.face_index.get(tuple(idx), None)
                        ][len(index)](index)
            elif isinstance(index, np.ndarray):
                return self[index.tolist()]
            elif isinstance(index, int):
                return self.vertex_index.get(index, None)
            else:
                raise ValueError('Unrecognized mesh item: %s' % index)

                

    ################################################################################################
    # Static Functions
    # These are mostly for calculating instance members lazily from other members

    @staticmethod
    def calculate_vertex_data(faces, edges, coords, vlab):
        n = coords.shape[1]
        vertex_idx = {v:i for (i,v) in enumerate(vlab)}
        d = collections.defaultdict(list)
        for idx in itertools.product(*map(range, edges.shape)):
            d[edges[idx]].append(idx)
        edge_idx = [map(lambda q: q[1], d[i]) for i in range(n)]
        d = collections.defaultdict(list)
        for idx in itertools.product(*map(range, faces.shape)):
            d[faces[idx]].append(idx)
        face_idx = [map(lambda q: q[1], d[i]) for i in range(n)]
        return (vertex_idx, edge_idx, face_idx)

    @staticmethod
    def calculate_edge_data(faces):
        limit = max(faces.flatten()) + 1
        edge2face = {}#sp.sparse.lil_matrix((limit, limit), dtype='int')
        idx = {}#sp.sparse.lil_matrix((limit, limit), dtype='int')
        edge_list = [None for i in range(3*faces.size)]
        k = 0
        rng = range(faces.shape[1])
        for (e,i) in zip(
            zip(np.concatenate((faces[0], faces[1], faces[2])),
                np.concatenate((faces[1], faces[2], faces[0]))),
            np.concatenate((rng, rng, rng))):
            if e not in idx:
                idx[e] = k
                idx[e[::-1]] = k
                edge_list[k] = e
                edge2face[e] = i
                k += 1
            elif e in edge2face:
                edge2face[e[::-1]] = i
        for ((a,b),i) in edge2face.items():
            edge2face[(b,a)] = i
        return (np.array(edge_list[0:k]).transpose(), idx, edge2face)

    @staticmethod
    def calculate_face_data(faces):
        idx = {}
        for (a,b,c,i) in zip(faces[0], faces[1], faces[2], range(faces.shape[1])):
            idx[(a,b,c)] = i
            idx[(b,c,a)] = i
            idx[(c,b,a)] = i
            idx[(a,c,b)] = i
            idx[(b,a,c)] = i
            idx[(c,a,b)] = i
        return idx

    @staticmethod
    def calculate_face_angles(faces, coords):
        X = np.array([coords[:,faces[0]], coords[:,faces[1]], coords[:,faces[2]]])
        sides = [X[1] - X[0], X[2] - X[1], X[0] - X[2]]
        normed_sides = map(
            lambda side: side / np.repeat([np.sqrt((side**2).sum(0))], coords.shape[0], 0),
            sides)
        dps = [(normed_sides[0] * (-normed_sides[2])).sum(0),
               (normed_sides[1] * (-normed_sides[0])).sum(0),
               (normed_sides[2] * (-normed_sides[1])).sum(0)]
        return np.arccos(dps)

    @staticmethod
    def calculate_face_normals(faces, coords):
        if coords.shape[0] != 3:
            raise ValueError('Face normals cannot be computed for 2D meshes')
        X = np.array([coords[:,faces[0]], coords[:,faces[1]], coords[:,faces[2]]])
        s1 = X[1] - X[0]
        s2 = X[2] - X[0]
        crosses = np.array([s1[1]*s2[2] - s1[2]*s2[1],
                            s1[2]*s2[0] - s1[0]*s2[2],
                            s1[0]*s2[1] - s1[1]*s2[0]])
        norms = np.sqrt((crosses**2).sum(0))
        w = (norms == 0)
        norms[w] = 1.0
        return np.where(w, 0, crosses / np.repeat([norms], 3, 0))

    @staticmethod
    def calculate_vertex_normals(vfaces, fnorms):
        tmp = np.array(
            map(
                lambda fs: fnorms[:,fs].sum(1),
                vfaces)).transpose()
        norms = np.sqrt((tmp ** 2).sum(0))
        w = (norms == 0)
        norms[w] = 1.0
        return np.where(w, 0, tmp / np.repeat([norms], 3, 0))

    @staticmethod
    def calculate_index(vertex_index, edge_index, face_index):
        return CorticalMesh.Index(vertex_index, edge_index, face_index)

    @staticmethod
    def calculate_spherical_coordinates(coords):
        n = coords.shape[1]
        centroid = np.mean(coords, 1)
        coords = np.array(
            [coords[0] - centroid[0],
             coords[1] - centroid[1],
             coords[2] - centroid[2]])
        rho = np.sqrt((coords ** 2).sum(0))
        normed = coords / np.array([rho]).repeat(3, 0)
        theta = np.arctan2(normed[0], normed[1])
        phi = np.arcsin(normed[2])
        return np.array([theta, phi, rho])


    ################################################################################################
    # Lazy/Static Interface
    # This code handles the lazy interface; only three data are non-lazy and these are coordinates,
    # faces, and options; they may be set directly.

    # These functions and the following static variable allow one to set particular members of the
    # object directly:
    @staticmethod
    def _check_coordinates(self, val):
        # Coordinates must be stored as a np array
        if type(val) is np.ndarray: x = val
        elif type(val) is list: x = np.array(val)
        else: raise ValueError('coordinates must be a matrix of 2D or 3D points')
            # Coordinates must be 2D or 3D
        if len(x.shape) != 2:
            raise ValueError('coordinates must be a matrix of 2D or 3D points')
        elif x.shape[1] == 2 or x.shape[1] == 3:
            x = x.transpose()
        elif x.shape[0] != 2 and x.shape[0] != 3:
            raise ValueError('coordinates must be a matrix of 2D or 3D points')
        if not x.flags['WRITEABLE']:
            x = x.copy()
            x.setflags(write=False)
        return x
    @staticmethod
    def _check_faces(self, val):
        # Faces must be an array
        if type(val) is np.ndarray: x = val
        elif type(val) is list:
            x = np.array(val)
        else:
            raise ValueError('faces must be an integer metrix of faces')
        # Faces must be an integer array
        if not issubclass(x.dtype.type, np.integer):
            raise ValueError('faces must be an integer metrix of faces')
        # Faces must be n x 3 or 3 x n
        if x.shape[1] == 3:
            x = x.transpose()
        elif x.shape[0] != 3:
            raise ValueError('faces must be an integer metrix of faces')
        if not x.flags['WRITEABLE']:
            x = x.copy()
            x.setflags(write=False)
        return x
    @staticmethod
    def _check_vertex_labels(self, val):
        # Must be a 1D list
        if type(val) is np.ndarray: x = val
        elif type(val) is list: x = np.array(val)
        else: raise ValueError('vertex_label must be an integer list of vertex labels')
        if len(x.shape) != 1 or not issubclass(x.dtype.type, np.integer):
            raise ValueError('vertex_label must be an integer list of vertex labels')
        if x.shape[0] != self.coordinates.shape[1]:
            raise ValueError('vertex_label must match coordinates in size')
        if not x.flags['WRITEABLE']:
            x = x.copy()
            x.setflags(write=False)
        return x
    @staticmethod
    def _check_options(self, val):
        # Options just have to be a dictionary and are converted to an immutable one
        if not isinstance(val, dict):
            raise ValueError('options must be a dictionary')
        if type(val) is pysistence.persistent_dict.PDict:
            return val
        else:
            return make_dict(**val)
    @staticmethod
    def _check_property(self, name, val):
        if not isinstance(val, np.ndarray) and not isinstance(val, list):
            raise ValueError('property values must be lists or numpy arrays')
        elif len(val) != len(self.vertex_labels):
            raise ValueError('property values must be of equal length to vertex count')
        return True
    @staticmethod
    def _check_properties(self, val):
        if not isinstance(val, dict):
            raise ValueError('properties must be a dictionary')
        n = len(self.vertex_labels)
        for (k, v) in val.iteritems(): CorticalMesh._check_property(self, k, v)
        if type(val) is pysistence.persistent_dict.PDict:
            return val
        else:
            return make_dict(**val)
    __settable_members = {
        'coordinates': lambda m,v: CorticalMesh._check_coordinates(m,v),
        'faces': lambda m,v: CorticalMesh._check_faces(m,v),
        'vertex_labels': lambda m,v: CorticalMesh._check_vertex_labels(m,v),
        'properties': lambda m,v: CorticalMesh._check_properties(m,v),
        'options': lambda m,v: CorticalMesh._check_options(m,v)}

    # This static variable explains the dependency hierarchy in cached data
    __lazy_members = {
        'vertex_data': (
            ('faces', 'edges', 'coordinates', 'vertex_labels'),
            lambda F,E,X,L: CorticalMesh.calculate_vertex_data(F, E, X, L)),
        'vertex_index': (('vertex_data',), lambda VD: VD[0]),
        'vertex_edge_index': (('vertex_data',), lambda VD: VD[1]),
        'vertex_face_index': (('vertex_data',), lambda VD: VD[2]),

        'edge_data': (('faces',), lambda F: CorticalMesh.calculate_edge_data(F)),
        'edges': (('edge_data',), lambda ED: ED[0]),
        'edge_index': (('edge_data',), lambda ED: ED[1]),
        'indexed_edges': (('edges','vertex_index'), 
                          lambda E,VI: np.asarray([[VI[u] for u in E[0]], [VI[v] for v in E[1]]])),
        'edge_face_index': (('edge_data',), lambda ED: ED[2]),

        'face_index': (('faces',), lambda F: CorticalMesh.calculate_face_data(F)),
        'indexed_faces': (('faces','vertex_index'), 
                          lambda F,VI: np.asarray([[VI[a] for a in F[0]],
                                                   [VI[b] for b in F[1]],
                                                   [VI[c] for c in F[2]]])),

        'index': (
            ('vertex_index', 'edge_index', 'face_index'),
            lambda VI, EI, FI: CorticalMesh.calculate_index(VI, EI, FI)),

        'edge_coordinates': (('indexed_edges', 'coordinates'), 
                             lambda E,X: np.asarray([X[:,E[0]], X[:, E[1]]])),
        'face_coordinates': (('indexed_faces', 'coordinates'),
                             lambda F,X: np.asarray([X[:,F[0]], X[:, F[1]], X[:, F[2]]])),

        'edge_lengths': (('edge_coordinates',), 
                         lambda EX: np.sqrt(np.power(EX[0] - EX[1], 2).sum(0))),

        'face_angles': (
            ('indexed_faces', 'coordinates'),
            lambda F,X: CorticalMesh.calculate_face_angles(F,X)),
        'face_normals': (
            ('indexed_faces', 'coordinates'),
            lambda F,X: CorticalMesh.calculate_face_normals(F,X)),
        'vertex_normals': (
            ('vertex_face_index', 'face_normals'),
            lambda VF,FN: CorticalMesh.calculate_vertex_normals(VF, FN)),

        'meta_data': (
             ('options',),
             lambda opts: opts.get('meta_data', {})),

        'spherical_coordinates': (
            ('coordinates',),
            lambda X: CorticalMesh.calculate_spherical_coordinates(X))}
        
   
    
        
    ################################################################################################
    # Constructor

    def __init__(self, coords, faces, **args):
        Immutable.__init__(self, CorticalMesh.__settable_members, {}, CorticalMesh.__lazy_members)
        # Setup coordinates
        self.coordinates = coords
        # And faces...
        self.faces = faces
        # If vertex labels were provided, make sure to set these
        self.vertex_labels = args.pop('vertex_labels', range(self.coordinates.shape[1]))
        # Same with properties
        self.properties = args.pop('properties', make_dict())
        # Finally, set the remaining options...
        self.options = args


    ################################################################################################
    # String versions of the object

    def __repr__(self):
        return "CorticalMesh(<" + str(self.coordinates.shape[1]) + " vertices>, <" \
            + str(self.faces.shape[1]) + " faces>)"

    ################################################################################################
    # Methods

    def vertex_edges(self, U):
        '''mesh.vertex_edges(U) yields a list of the edges that contain vertex U. If U is a list of
           vertices, then this function will automatically thread over it.'''
        if isinstance(U, int):
            return self.vertex_edge_index[U]
        else:
            return map(lambda u: self.vertex_edges(u), U)
    def vertex_faces(self, U):
        '''mesh.vertex_faces(U) yields a list of the faces that contain vertex U. If U is a list of
           vertices, then this function will automatically thread over it.'''
        if isinstance(U, int):
            return self.vertex_face_index[U]
        else:
            return map(lambda u: self.vertex_faces(u), U)
    def edge_faces(self, E):
        '''mesh.edge_faces(E) yields a list of the faces that contain edge E. If E is a list of
           edges, then this function will automatically thread over it.'''
        if isinstance(E, tuple) and len(E) == 2:
            return self.edge_face_index[E]
        else:
            return map(lambda e: self.edge_faces(e), E)            

    def select(self, filt, filter_vertices=True, filter_edges=False, filter_faces=False):
        '''mesh.select(filt) yields a new CorticalMesh object that is identical to mesh except
           that only the vertices u for which filt(u) yields true will be retained. Three optional
           arguments, filter_vertices (true), filter_edges (false), and filter_faces (false) may
           also be specified. For any of these which is true, those elements will be filtered using
           the filt function. For any that is false, the elements will be excluded only if their
           subparts are excluded. For any that is neither true or false, it must be a function, in
           which case that function is used as a filter in place of the filt function.
           If filt is instead a list of vertex labels, then filter_vertices is not used and instead
           the given list is used as the initial vertex filter.'''
        vf = filt if filter_vertices is True \
            else filter_vertices if filter_vertices is not False \
            else None
        ef = filt if filter_edges is True \
            else filter_edges if filter_edges is not False \
            else None
        ff = filt if filter_faces is True \
            else filter_faces if filter_faces is not False \
            else None
        # Find the included vertices:
        if isinstance(vf, np.ndarray): vincl = vf
        elif isinstance(vf, list):     vincl = vf
        elif vf is not None:           vincl = [u for u in self.vertex_labels if vf(u)]
        else:                          vincl = self.vertex_labels
        vincl = set(self.index[vincl])
        # Find the included edges:
        es = self.indexed_edges
        edge_idcs = range(self.edges.shape[1])
        if isinstance(ef, (list, set)):
            ef = np.asarray(ef)
        if ef is None:
            eincl = edge_idcs
        elif isinstance(ef, np.ndarray): 
            if len(ef.shape) == 1:
                eincl = self.index[ef]
            else:
                eincl = self.index[ef.T if ef.shape[1] != 2 else ef]
        else:
            elst = self.edge_list.T
            if ef is None: eincl = edge_idcs
            else:          eincl = [e for e in edge_idcs if ef(tuple(elst[e]))]
        # filter by vertices
        eincl = set(map(int, eincl))
        eincl.intersection_update(
            np.union1d([], [l for idcs in self.vertex_edges(vincl)
                              for l in idcs
                              if es[0,l] in vincl and es[1,l] in vincl]))
        # Find the included faces
        face_idcs = range(self.faces.shape[1])
        fs = self.indexed_faces
        if isinstance(ff, (list, set)):
            ff = np.asarray(ff)
        if ff is None:
            fincl = face_idcs
        elif isinstance(ff, np.ndarray): 
            if len(ff.shape) == 1:
                fincl = ff.tolist()
            else:
                fincl = self.index[ff.T if ff.shape[1] != 3 else ff]
        else:
            if ff is None: fincl = face_idcs
            else:          fincl = [f for f in face_idcs if ff(tuple(self.faces[:,f]))]
        # filter by vertices
        fincl = set(map(int, fincl))
        fincl.intersection_update(
            np.union1d([], [l for idcs in self.vertex_faces(vincl)
                              for l in idcs
                              if fs[0,l] in vincl and fs[1,l] in vincl and fs[2,l] in vincl]))
        vincl = list(vincl)
        eincl = list(eincl)
        fincl = list(fincl)
        # Make the subsets
        I = self.index[vincl]
        X = self.coordinates[:, I]
        V = self.vertex_labels[I]
        F = self.faces[:, fincl]
        opts = self.options.copy()
        meta = opts.get('meta_data', {}).copy()
        opts = opts.without('meta_data')
        meta = meta.using(source_mesh=self)
        props = {name: prop[I] for (name,prop) in self.properties.iteritems()}
        return CorticalMesh(X, F, vertex_labels=V, meta_data=meta, properties=props, **opts)

    def orthographic_projection(self, center, radius, 
                                align_vertex_id=None, align_axis=None, scale=None):
        '''
        mesh.orthographic_projection((x,y,z), r) yields a 2D mesh made by projecting the vertices
        of the 3D mesh onto a 2D map orthographically for all vertices with a minor vector angle 
        with the given center (x,y,z) that is less than the given radius r. The optional parameters
        align_vertex_id and align_axis specify that the vertex with the given id (label) be aligned
        to the given 2D axis in the final map. The optional argument scale may also be specified to
        scale the resulting map; by default, the norm of the given center is used.
        '''
        # start by selecting only the vertices that are close enough to the center:
        n = len(self.vertex_labels)
        X = self.coordinates
        idcs = [i for i in range(n) if geo.vector_angle(center, X[:, i]) < radius]
        m = self.select(self.vertex_labels[idcs])
        # now, edit the coordinates so that they are 2D and appropriately centered/scaled
        if scale is None:
            scale = np.linalg.norm(center)
        # align the coordinates to be centered around the center...
        mtx2D = np.dot(
            geo.alignment_matrix_3D(center, (0, 0, 1))[0:2, :],
            m.coordinates)
        # if there is an alignment option, go ahead wiht it
        if align_vertex_id is not None:
            if align_axis is None:
                align_axis = (1, 0)
            mtx2D = np.dot(
                geo.alignment_matrix_2D(m.index[align_vertex_id], align_axis),
                mtx2D)
        # just set the coordinate matrix and return
        m.coordinates = scale * mtx2D
        return m

    def add_property(self, name, prop=Ellipsis):
        '''mesh.add_property(name, prop) adds (or overwrites) the given property with the given name
           in the given mesh. The name must be a valid dictionary key and the prop argument must be
           a list of numpy array of values, one per vertex.
           mesh.add_property(d) adds all of the properties in the given dictionary of properties, 
           d.
           Note that in either case, if the value of prop or the value in a dictionary item is
           None, then the item is removed from the property list instead of being added.'''
        n = len(self.vertex_labels)
        if prop is Ellipsis:
            if isinstance(name, dict):
                for (n,p) in name.iteritems():
                    self.add_property(n, p)
            else:
                raise ValueError('add_property must be called with a name and propery or a dict')
        else:
            if prop is None:
                self.remove_property(name)
            else:
                CorticalMesh._check_property(self, name, prop)
                self.__dict__['properties'] = self.properties.using(**{name: prop})

    def remove_property(self, name):
        '''mesh.remove_property(name) removes the property with the given name from the given mesh.
           The name argument may also be an iterable collection of names, in which case all are
           removed.'''
        if hasattr(name, '__iter__'):
            for n in name: self.remove_property(n)
        elif name in self.properties:
            self.__dict__['properties'] = self.properties.without(name)
    
    def property_value(self, name):
        '''mesh.property_value(name) yields a list of the property values with the given name. If
           name is an iterable, then property_value automatically threads across it. If no such
           property is found in the mesh, then None is returned.'''
        if hasattr(name, '__iter__'):
            return map(lambda n: self.property_value(n), name)
        else:
            return self.properties.get(name, None)
            
    def prop(self, name, arg=Ellipsis):
        '''mesh.prop(...) is a generic function for handling mesh properties. It may be called
           with a variety of arguments:
             * mesh.prop(name) is equivalent to mesh.property_value(name) if name is either not
               an iterable or is an iterable that is not a dictionary
             * mesh.prop(d) is equivalent to mesh.add_property(d) if d is a dictionary
             * mesh.prop(name, val) is equivalent to mesh.add_property(name, val) if val is not
               None
             * mesh.prop(name, None) is equivalent to mesh.remove_property(name)'''
        if arg is Ellipsis:
            if isinstance(name, dict):
                self.add_property(name)
            else:
                return self.property_value(name)
        else:
            self.add_property(name, arg)

    def map_vertices(self, f, merge=None):
        '''mesh.map_vertices(f) yields the result of mapping the function f over all vertices in
           the given mesh. For each vertex, f is called with a dictionary as the argument; the keys
           and values in the dictionary are the property names and property values for the
           vertices; additionally, the following properties are added to the dictionary:
             * \'vertex_label\': the label of the given vertex
             * \'coordinate\': the coordinate of the given vertex
           If map_vertices is called with the optional argument merge, a dictionary may be given;
           this dictionary must be a valid properties dictionary (see add_property) and the
           properties specified in the dictionary will be merged with the mesh's properties during
           the map operation.'''
        if merge is None: merge = {}
        extra = {'vertex_label':self.vertex_labels,
                 'coordinate': self.coordinates.transpose().tolist()}
        return map(
            f,
            [dict(zip(extra.keys() + self.properties.keys() + merge.keys(), vals)) \
                 for vals in zip(*(extra.values() + self.properties.values() + merge.values()))])

    def where(self, f):
        '''mesh.where(f) yields a boolean mask in which any vertex for which f(p) yields true
           is given a True value in the mask and all other vertives are given False values. The
           function f should operate on a dict p which is identical to that passed to the method
           mesh.map_vertices.'''
        return self.vertex_labels[np.array(self.map_vertices(f)) == True]

    def option(self, opt):
        '''mesh.option(x) yields the value of the option f in the given mesh. If x is not an option
           then None is returned.  If x is a list, then the list is automatically mapped.'''
        opts = self.options
        if isinstance(opt, list):
            return map(lambda o: self.option(o), opt)
        elif opt in opts:
            return opts[opt]
        else:
            return None

    def meta(self, arg):
        '''mesh.meta(x) yields the meta-data element with name x in the given mesh. Note that meta
           is stored as an option in meshes.'''
        if isinstance(arg, list):
            return map(lambda x: self.meta(x), arg)
        else:
            return self.options.get('meta_data', {}).get(arg, None)


    ################################################################################################
    # Importers
    # These are static import methods
    
    @staticmethod
    def load(file):
        (coords, faces) = fsio.read_geometry(file)
        return CorticalMesh(coords, faces, source_file=file)


####################################################################################################
# cortex_to_ribbon converter and related functions

def cortex_to_mrvolume(mesh, property):
    '''cortex_to_mrvolume(mesh, property) yields a volume (represented as a numpy array) with the
         given property projected into the volume along the surface.
       cortex_to_mrvolume([mesh1, mesh2...], property) yields the volume in which all of the given
         meshes are projected into the volume simultaneously.

       The argument property may be either the name of a property attached to all the given meshes
       or a list or numpy array of values with the same length as the vertices in all the given 
       meshes.

       The following options are accepted:
         * method (default: 'weighted-mean') specifies how a voxel's contained vertices determine
           the value assigned to the voxel. This may be a callable object f(x0, X, Y) where the
           first argument x0 is the center of the voxel, the second argument X is a 3 x n numpy
           matrix of the (x, y, z) coordinates of each vertex in the voxel, and Y is a numpy vector
           of the property values associated with the vertices in X. Alternately, method may be
           set to any of the following:
            * 'weighted-mean' uses a Gaussian-weighted mean with the radius of a voxel as the 
              standard deviation of the Gaussian.
            * 'mean' uses mean(Y) as the value.
            * 'nearest' uses the stritly nearest vertex to the center of the voxel.
         * dimensions (default: [256, 256, 256]) specifies the size of the volume to be used. The
           size may be given in three ways: (1) as an list of widths [w, h, d], which are taken to
           be the number of voxels in each dimension with the voxels spanning from -[w h d]/2 to
           [w h d]/2, colloquially; (2) as a list of min/max pairs [[wmin wmax], ...] in which case
           the voxel width is assumed to be 1; or (3) as a list of min/max/n triples in which n is
           the number of voxels along that dimension.
         * origin (default: 'automatic') specifies where the (0,0,0) coordinate of the given cortex
           lines up in the volume coordinate system. If the default, 'automatic', is used, then the
           origin or the mean of the coordinate system (if the origin is not in the system) is
           used.
         * orientation (default: 'identity') specifies how the volume is oriented relative to the
           cortex. The default, 'identity', results in an identity matrix being used. Specifically,
           this argument should be a 3x3 matrix whose inverse is applied to the (x,y,z) coordinates
           of the cortical surface vertices before their containing voxels are determined. For a
           FreeSurfer surface, this matrix is equivalent to [R_row, A_row, S_row] where R_row,
           A_row, and S_row are unit vectors pointing in the direction of the right, anterior, and
           superior directions in the directions implied by array-indexing order.
         * default (default: 0) specifies what the default value in the volume should be.'''
    # First, process the args...

    ## Process the mesh arg; make sure the meshes are safe for the transform and make sure they are
    #  in a list
    if isinstance(mesh, CorticalMesh) and mesh.coordinates.shape[0] == 3:
        meshes = [mesh]
    elif hasattr(mesh, '__len__') and hasattr(mesh, '__iter__') \
            and len(mesh) > 0  \
            and all(isinstance(m, CorticalMesh) for m in mesh):
        if all(3 == m.coordinates.shape[0] for m in mesh):
            meshes = mesh
        else:
            raise ValueError('meshes must be a list of 3D cortical meshes')
    else:
        raise ValueError('meshes must be a 3D cortical mesh or a list of such meshes')
    ## Process the property arg; make sure the property exists in at least one of the cortices
    #  and filter the mesh list down to just those that do have it.
    #  We also go ahead and make the master vertices and properties matrices since we have all
    #  of that data now.
    if isinstance(property, basestring):
        # This is a property name; filter the mesh list to those that have it
        meshes = [m for m in meshes if property in m.properties]
        if len(meshes) == 0:
            raise ValueError('given property was not found in any provided mesh')
        properties = [m.prop(property) for m in meshes]
    elif hasattr(property, '__len__') and hasattr(property, '__iter__'):
        n = len(property)
        if not all([n == len(m.vertex_labels) for m in meshes]):
             raise ValueError('property lists must be the same length as all mesh vertex labels')
        properties = [property for m in meshes]
    else:
        raise ValueError('property list must be either a string or a list')

    #here
    return True
