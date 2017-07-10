####################################################################################################
# cortex.py
# Simple tools for dealing with the cortical surface in Python
# By Noah C. Benson

import numpy                  as np
import scipy                  as sp
import scipy.spatial          as space
import scipy.optimize         as spopt
import nibabel.freesurfer.io  as fsio
import neuropythy.geometry    as geo
from neuropythy.immutable import Immutable
from scipy.sparse         import (lil_matrix, csr_matrix)
from numpy.linalg         import lstsq, norm
from numbers              import (Number, Integral)
from types                import DictType
from pysistence           import make_dict
import os, math, itertools, collections, pysistence, colorsys

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
                    if isinstance(index, list): return [self[x]    for x in index]
                    else:                       return {x: self[x] for x in index}
                else:
                    return [
                        lambda idx: [],
                        lambda idx: self.vertex_index.get(idx[0], None),
                        lambda idx: self.edge_index.get(tuple(idx), None),
                        lambda idx: self.face_index.get(tuple(idx), None)
                        ][len(index)](index)
            elif isinstance(index, np.ndarray):
                return self[index.tolist()]
            elif isinstance(index, Number) or np.issubdtype(type(index), np.float):
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
        tmp = np.array([fnorms[:,fs].sum(1) for fs in vfaces]).T
        norms = np.sqrt((tmp ** 2).sum(0))
        w = (norms == 0)
        norms[w] = 1.0
        return np.where(w, 0, tmp / np.repeat([norms], 3, 0))

    @staticmethod
    def _order_neighborhood(edges):
        res = [edges[0][1]]
        for i in range(len(edges)):
            for e in edges:
                if e[0] == res[i]:
                    res.append(e[1])
                    break
        return res
    @staticmethod
    def calculate_neighborhoods(vlab, faces, vfindex):
        nedges = [[((f[0], f[1]) if f[2] == u else
                    (f[1], f[2]) if f[0] == u else
                    (f[2], f[0]))
                   for fid in fs
                   for f in (faces[:,fid],)]
                  for (u, fs) in zip(vlab, vfindex)]
        return [CorticalMesh._order_neighborhood(nei) for nei in nedges]

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

    @staticmethod
    def construct_mesh_graph(V,E,L):
        try:
            import igraph
            g = igraph.Graph()
            g.add_vertices(V)
            g.add_edges(E.T)
            g.es['length'] = L
            return g
        except:
            raise ValueError('Graphs are not currently supported')


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
        'vertex_count': (('vertex_labels',), lambda L: len(L)),
        'vertex_data': (
            ('faces', 'edges', 'coordinates', 'vertex_labels'),
            lambda F,E,X,L: CorticalMesh.calculate_vertex_data(F, E, X, L)),
        'vertex_index': (('vertex_data',), lambda VD: VD[0]),
        'vertex_edge_index': (('vertex_data',), lambda VD: VD[1]),
        'vertex_face_index': (('vertex_data',), lambda VD: VD[2]),

        'edge_data':       (('faces',), lambda F: CorticalMesh.calculate_edge_data(F)),
        'edges':           (('edge_data',), lambda ED: ED[0]),
        'edge_index':      (('edge_data',), lambda ED: ED[1]),
        'indexed_edges':   (('edges','vertex_index'), 
                            lambda E,I: np.asarray([[I[u] for u in E[0]], [I[v] for v in E[1]]])),
        'edge_face_index': (('edge_data',), lambda ED: ED[2]),

        'face_index': (('faces',), lambda F: CorticalMesh.calculate_face_data(F)),
        'indexed_faces': (('faces','vertex_index'), 
                          lambda F,VI: np.asarray([[VI[a] for a in F[0]],
                                                   [VI[b] for b in F[1]],
                                                   [VI[c] for c in F[2]]])),

        'index': (
            ('vertex_index', 'edge_index', 'face_index'),
            lambda VI, EI, FI: CorticalMesh.calculate_index(VI, EI, FI)),

        'property_names': (('properties','hemisphere'),
                           lambda props,hemi: set(
                               (list(props.keys()) + list(hemi.property_names)
                                if hemi is not None else props.keys()))),
        
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

        'neighborhoods': (('vertex_labels','faces','vertex_face_index'),
                          lambda L, F, VFI: CorticalMesh.calculate_neighborhoods(L, F, VFI)),
        'indexed_neighborhoods': (('neighborhoods','index'),
                                  lambda n,idx: [[idx[i] for i in nrow] for nrow in n]),
        
        'vertex_spatial_hash': (('coordinates',), lambda X: space.cKDTree(X.T)),
        'face_spatial_hash': (('face_coordinates',), lambda FX: space.cKDTree(FX.mean(0).T)),

        'meta_data': (
             ('options',),
             lambda opts: opts.get('meta_data', {})),

        'spherical_coordinates': (
            ('coordinates',),
            lambda X: CorticalMesh.calculate_spherical_coordinates(X)),

        'graph': (
            ('vertex_labels','indexed_edges','edge_lengths'),
            lambda V,E,L: CorticalMesh.construct_mesh_graph(V,E,L))}
        
   
    
        
    ################################################################################################
    # Constructor

    def __init__(self, coords, faces, **args):
        Immutable.__init__(self,
                           CorticalMesh.__settable_members,
                           {'hemisphere': args.pop('hemisphere', None),
                            'subject':    args.pop('subject', None)},
                           CorticalMesh.__lazy_members)
        coords = np.asarray(coords)
        coords = coords.T if coords.shape[0] > 3 or coords.shape[0] < 2 else coords
        # Setup coordinates
        self.coordinates = coords
        # And faces...
        self.faces = faces
        # If vertex labels were provided, make sure to set these
        self.vertex_labels = args.pop('vertex_labels', range(coords.shape[1]))
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
        vincl = np.asarray([int(i) for i in sorted(list(vincl))])
        eincl = np.asarray([int(i) for i in sorted(list(eincl))])
        fincl = np.asarray([int(i) for i in sorted(list(fincl))])
        # Make the subsets
        I = self.index[vincl]
        X = self.coordinates[:, I]
        V = self.vertex_labels[I]
        F = self.faces[:, fincl]
        opts = self.options.copy()
        meta = opts.get('meta_data', {}).copy()
        opts = opts.without('meta_data')
        meta = meta.using(source_mesh=self)
        props = {}
        for p in self.property_names:
            v = None
            try:
                v = self.prop(p)
            except:
                pass
            if v is not None:
                props[p] = np.asarray(v)[I]
        return CorticalMesh(X, F, vertex_labels=V, meta_data=meta, properties=props, **opts)

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

    def has_property(self, name):
        '''mesh.has_property(name) yields True if the given mesh contains the property with the
           given name. Note that this checks the hemisphere properties as well as the mesh
           properties and should be used preferentially over (name in mesh.properties).'''
        if name in self.properties:
            return True
        elif self.hemisphere is not None:
            return self.hemisphere.has_property(name)
        else:
            return False

    def remove_property(self, name):
        '''mesh.remove_property(name) removes the property with the given name from the given mesh.
           The name argument may also be an iterable collection of names, in which case all are
           removed. Note that you cannot remove properties inherited from the Hemisphere object;
           these must be removed from the Hemisphere.'''
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
        elif name in self.properties:
            return self.properties[name]
        elif self.hemisphere is not None:
            return self.hemisphere.property_value(name)

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

    def reproject(self, X):
        '''mesh.reproject(X) yields a 2D coordinate matrix Xp with the same number of points as the
           3D coordinate matrix X such that the points have been projected identically as the
           projection used to create this 2D mesh. If this is not a mesh created with the
           CorticalMesh projection function, an error will be raised.'''
        projection_params = self.option('projection_parameters')
        if projection_params is None:
            raise ValueError('the given mesh was not created by projection from another mesh')
        if 'forward_function' not in projection_params:
            raise ValueError('the projection method used has not defined projection function')
        ffn = projection_params['forward_function']
        return ffn(X)
    def unproject(self, X):
        '''mesh.unproject(X) yields a 3D coordinate matrix Xu with the same number of points as the
           2D coordinate matrix X such that the points have been projected back from the 2D map to
           the 3D sphere in the opposite fashion as the projection that was used to create this 2D
           mesh. If this is not a mesh created with the CorticalMesh projection function, an error
           will be raised. This function is the inverse of reproject.'''
        projection_params = self.option('projection_parameters')
        if projection_params is None:
            raise ValueError('the given mesh was not created by projection from another mesh')
        if 'inverse_function' not in projection_params:
            raise ValueError('the projection method used has not defined inverse')
        ifn = projection_params['inverse_function']
        return ifn(X)

    def unaddress(self, data):
        '''mesh.unaddress(addr) yields a coordinate matrix of points on the given mesh that are
           located at the mesh addresses given in addr. The addr matrix should be generated from
           a mesh.address(points) method call; the two meshes must be topologically equivalent
           or an error may be raised. The resulting matrix will always be sized 2 or 3 by n unless
           the address is of a single point, in which case that point is returned.'''
        # addresses are dictionaries that contain two fields: 'face_id' and 'coordinates'
        # these may be numbers or a list and a matrix whose second dimension is the same lenght
        if self.coordinates.shape[0] == 2:
            # In this case, we unaddress on the sphere, then reproject...
            smesh = self.meta('source_mesh')
            if smesh is None: raise ValueError('2D mesh has no source mesh!')
            return self.reproject(smesh.unaddress(data))
        if not isinstance(data, dict):
            raise ValueError('address data must be a dictionary')
        if 'face_id' not in data: raise ValueError('address must contain face_id key')
        if 'coordinates' not in data: raise ValueError('address must contain coordinates key')
        face_id = data['face_id']
        coordinates = np.asarray(data['coordinates'])
        if hasattr(face_id, '__iter__'):
            if len(coordinates.shape) != 2:
                raise ValueError('if face_id is a list, then coordinates must be a matrix')
            if coordinates.shape[1] != len(face_id):
                coordinates = coordinates.T
            tx = np.asarray(self.coordinates[:, f] for f in self.faces[face_id])
            return geo.barycentric_to_cartesian(tx, coordinates)
        else:
            return geo.barycentric_to_cartesian(self.face_coordinates[:, :, face_id],
                                                coordinates)
    def address(self, data):
        '''
        mesh.address(X) yields a dictionary containing the address or addresses of the point or
        points given in the vector or coordinate matrix X. Addresses specify a single unique 
        topological location on the mesh such that deformations of the mesh will address the same
        points differently. To convert a point from one mesh to another isomorphic mesh, you can
        address the point in the first mesh then unaddress it in the second mesh. Note that an
        address may only be obtained from a spherical or 2D mesh.
        '''
        if self.coordinates.shape[0] == 2:
            smesh = self.meta('source_mesh')
            if smesh is None: raise ValueError('2D mesh has no source mesh!')
            return smesh.address(self.unproject(data))
        else:
            reg = self.meta('registration')
            if reg is None: raise ValueError('mesh has no registration!')
            return reg.address(data)

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


    def cortical_magnification(self, polar_angle=None, eccentricity=None, weight=None,
                               weight_cutoff=0, direction='all'):
        '''
        mesh.cortical_magnification() yields the cortical magnification factor of each vertex in the
        given mesh at which the polar angle and eccentricity weight is above the optional
        weight_cutoff argument. The option direction may be set to one of 'mean', 'radial', or
        'tangential' to specify that cortical magnification should be calculated in the radial or
        tangential direction; mean is the default and specifies that the mean change in mm per
        degree should be reported.
        '''
        angle = (
            polar_angle if hasattr(polar_angle, '__iter__') else
            self.prop(polar_angle) if isinstance(polar_angle, basestring) else
            retinotopy_data(self, 'polar_angle'))
        eccen = np.asarray(
            eccentricity if hasattr(eccentricity, '__iter__') else
            self.prop(eccentricity) if isinstance(eccentricity, basestring) else
            retinotopy_data(self, 'eccentricity'))
        weight = np.asarray(
            weight if hasattr(weight, '__iter__') else
            self.prop(weight) if isinstance(weight, basestring) else
            retinotopy_data(self, 'weight'))
        angle = np.asarray([(90 - ang)*math.pi/180 for ang in angle])
        point = np.asarray([[ecc * np.cos(ang), ecc * np.sin(ang)] if wgt > 0 else [0,0]
                            for (ang,ecc,wgt) in zip(angle, eccen, weight)])
        coord = self.coordinates.T
        direction = direction.lower()
        
        if direction == 'mean' or direction == 'all':
            if weight is None: weight = [1 if e is not None else 0 for e in eccen]
            relevant = [
                np.asarray(
                    [(np.sqrt(((point[u] - point[n]) ** 2).sum(0)),
                      np.sqrt(((coord[u] - coord[n]) ** 2).sum(0)),
                      weight[n])
                     for n in nei if weight[n] > 0]
                ).T if weight[u] > 0 else []
                for (u, nei) in enumerate(self.indexed_neighborhoods)]
            return [((np.dot(rel[1], rel[2])/np.sum(rel[2])) /
                     (np.dot(rel[0], rel[2]) / np.sum(rel[2]))
                     if len(rel) > 0 and np.sum(rel[2]) > 0 and np.dot(rel[0], rel[2]) > 0
                     else None)
                    for rel in relevant]
        elif direction == 'radial' or direction == 'eccentricity':
            if weight is None: weight = [1 if e is not None else 0 for e in eccen]
            data = eccen
        elif direction == 'tangential' or direction == 'polar_angle':
            if weight is None: weight = [1 if a is not None else 0 for a in angle]
            data = angle
        else:
            raise ValueError('Invalid direction given to cortical_magnification')
        # estimate the gradient of the data at each point on the cortex
        res = []
        for (x0,y0,w0,nei) in zip(coord, data, weight, self.neighborhoods):
            nei = [n for n in nei if weight[n] > weight_cutoff]
            if not nei or w0 <= weight_cutoff:
                res.append(None)
            else:
                m = norm(lstsq(coord[nei] - x0, data[nei] - y0)[0])
                res.append(None if m == 0 else 1.0/m)
        return res


    ################################################################################################
    # Importers
    # These are static import methods
    
    @staticmethod
    def load(file):
        (coords, faces) = fsio.read_geometry(file)
        return CorticalMesh(coords, faces, source_file=file)

# smooth a field on the cortical surface
def mesh_smooth(mesh, prop, smoothness=0.5, weights=None,
                outliers=None, mask=None, null=np.nan,
                data_range=None, match_distribution=None):
    '''
    mesh_smooth(mesh, prop) yields a numpy array of the values in the mesh property prop after they
      have been smoothed on the cortical surface. Smoothing is done by minimizing the square
      difference between the values in prop and the smoothed values simultaneously with the
      difference between values connected by edges. The prop argument may be either a property name
      or a list of property values.
    
    The following options are accepted:
      * weights (default: None) specifies the weight on each individual vertex that is in the mesh;
        this may be a property name or a list of weight values. Any weight that is <= 0 or None is
        considered outside the mask.
      * smoothness (default: 0.5) specifies how much the function should care about the smoothness
        versus the original values when optimizing the surface smoothness. A value of 0 would result
        in no smoothing performed while a value of 1 indicates that only the smoothing (and not the
        original values at all) matter in the solution.
      * outliers (default: None) specifies which vertices should be considered 'outliers' in that,
        when performing minimization, their values are counted in the smoothness term but not in the
        original-value term. This means that any vertex marked as an outlier will attempt to fit its
        value smoothly from its neighbors without regard to its original value. Outliers may be
        given as a boolean mask or a list of indices. Additionally, all vertices whose values are
        either infinite or outside the given data_range, if any, are always considered outliers
        even if the outliers argument is None.
      * mask (default: None) specifies which vertices should be included in the smoothing and which
        vertices should not. A mask of None includes all vertices by default; otherwise the mask may
        be a list of vertex indices of vertices to include or a boolean array in which True values
        indicate inclusion in the smoothing. Additionally, any vertex whose value is either NaN or
        None is always considered outside of the mask.
      * data_range (default: None) specifies the range of the data that should be accepted as input;
        any data outside the data range is considered an outlier. This may be given as (min, max),
        or just max, in which case 0 is always considered the min.
      * match_distribution (default: None) allows one to specify that the output values should be
        distributed according to a particular distribution. This distribution may be None (no
        transform performed), True (match the distribution of the input data to the output,
        excluding outliers), a collection of values whose distribution should be matched, or a 
        function that accepts a single real-valued argument between 0 and 1 (inclusive) and returns
        the appropriate value for that quantile.
      * null (default: numpy.nan) specifies what value should be placed in elements of the property
        that are not in the mask or that were NaN to begin with. By default, this is NaN, but 0 is
        often desirable.
    '''
    # Do some argument processing ##################################################################
    n = mesh.vertex_count
    all_vertices = np.asarray(range(n), dtype=np.int)
    # Parse the property data...
    prop = mesh.prop(prop) if isinstance(prop, basestring) else prop
    if not isinstance(prop, np.ndarray):
        prop = [np.nan if x is None else x for x in prop]
    prop = np.array(prop, dtype=np.float)
    # ...including the weights...
    if weights is None: weights = np.ones(len(prop), dtype=np.float)
    if not hasattr(weights, '__iter__') or len(weights) != len(prop):
        raise ValueError('weights must be None or an iterable with 1 entry per vertex')
    weights = np.asarray([0.0 if w is None or np.isnan(w) or w <= 0 else w for w in weights],
                         dtype=np.float)
    # First, find the mask; these are values that can be included theoretically
    where_inf = np.where(np.isinf(prop))[0]
    where_nan = np.where(np.isnan(prop))[0]
    where_bad = np.union1d(where_inf, where_nan)
    where_ok  = np.setdiff1d(all_vertices, where_bad)
    # Whittle down the mask to what we are sure is in the minimization:
    mask = np.union1d(
        np.setdiff1d(all_vertices if mask is None else all_vertices[mask], where_nan),
        np.where(np.isclose(weights, 0))[0])
    # Find the outliers: values specified as outliers or values with inf; will build this as we go
    outliers = [] if outliers is None else all_vertices[outliers]
    outliers = np.intersect1d(outliers, mask) # outliers not in the mask don't matter anyway
    # If there's a data range argument, deal with how it affects outliers
    if data_range is not None:
        if hasattr(data_range, '__iter__'):
            outliers = np.union1d(outliers, mask[np.where(prop[mask] < data_range[0])[0]])
            outliers = np.union1d(outliers, mask[np.where(prop[mask] > data_range[1])[0]])
        else:
            outliers = np.union1d(outliers, mask[np.where(prop[mask] < 0)[0]])
            outliers = np.union1d(outliers, mask[np.where(prop[mask] > data_range)[0]])
    # no matter what, trim out the infinite values (even if inf was in the data range)
    outliers = np.union1d(outliers, mask[np.where(np.isinf(prop[mask]))[0]])
    outliers = np.asarray(outliers, dtype=np.int)
    # here are the vertex sets we will use below
    tethered = np.setdiff1d(mask, outliers)
    tethered = np.asarray(tethered, dtype=np.int)
    mask = np.asarray(mask, dtype=np.int)
    maskset  = frozenset(mask)
    # Do the minimization ##########################################################################
    # start by looking at the edges
    el0 = mesh.indexed_edges
    # give all the outliers mean values
    prop[outliers] = np.mean(prop[tethered])
    # x0 are the values we care about; also the starting values in the minimization
    x0 = np.array(prop[mask])
    # since we are just looking at the mask, look up indices that we need in it
    mask_idx = {v:i for (i,v) in enumerate(mask)}
    mask_tethered = np.asarray([mask_idx[u] for u in tethered])
    el = np.asarray([(mask_idx[a], mask_idx[b]) for (a,b) in el0.T
                     if a in maskset and b in maskset])
    # These are the weights and objective function/gradient in the minimization
    (ks, ke) = (smoothness, 1.0 - smoothness)
    e2v = lil_matrix((len(x0), len(el)), dtype=np.int)
    for (i,(u,v)) in enumerate(el):
        e2v[u,i] = 1
        e2v[v,i] = -1
    e2v = csr_matrix(e2v)
    (us, vs) = el.T
    weights_tth = weights[tethered]
    def _f(x):
        rs = np.dot(weights_tth, (x0[mask_tethered] - x[mask_tethered])**2)
        re = np.sum((x[us] - x[vs])**2)
        return ks*rs + ke*re
    def _f_jac(x):
        df = 2*ke*e2v.dot(x[us] - x[vs])
        df[mask_tethered] += 2*ks*weights_tth*(x[mask_tethered] - x0[mask_tethered])
        return df
    sm_prop = spopt.minimize(_f, x0, jac=_f_jac, method='L-BFGS-B').x
    # Apply output re-distributing if requested ####################################################
    if match_distribution is not None:
        percentiles = 100.0 * np.argsort(np.argsort(sm_prop)) / (float(len(mask)) - 1.0)
        if match_distribution is True:
            sm_prop = np.percentile(x0[mask_tethered], percentiles)
        elif hasattr(match_distribution, '__iter__'):
            sm_prop = np.percentile(match_distribution, percentiles)
        elif hasattr(match_distribution, '__call__'):
            sm_prop = map(match_distribution, percentiles / 100.0)
        else:
            raise ValueError('Invalid match_distribution argument')
    result = np.full(len(prop), null, dtype=np.float)
    result[mask] = sm_prop
    return result

# Plotting and Coloring Meshes #####################################################################
# All of this requires matplotlib, so we try all and fail gracefully if we don't have it

try:
    import matplotlib, matplotlib.pyplot, matplotlib.tri
    _curv_cmap_dict = {
        name: ((0.0, 0.0, 0.5),
               (0.5, 0.5, 0.2),
               (1.0, 0.2, 0.0))
        for name in ['red', 'green', 'blue']}
    _curv_cmap = matplotlib.colors.LinearSegmentedColormap('curv', _curv_cmap_dict)

    def vertex_curvature_color(m):
        return [0.2,0.2,0.2,1.0] if m['curvature'] > -0.025 else [0.7,0.7,0.7,1.0]
    def vertex_weight(m):
        return m['weight']                 if 'weight'                 in m else \
               m['variance_explained']     if 'variance_explained'     in m else \
               m['PRF_variance_explained'] if 'PRF_variance_explained' in m else \
               1.0
    def vertex_angle_color(m, weight_cutoff=0.2, weighted=True):
        w = vertex_weight(m)
        curvColor = np.asarray(vertex_curvature_color(m))
        if weighted and w < weight_cutoff: return curvColor
        angColor = colorsys.hsv_to_rgb(0.666667*(1 - m['polar_angle']/180), 1, 1)
        angColor = np.asarray(angColor + (1,))
        if weighted:
            return angColor*w + curvColor*(1-w)
        else:
            return angColor
    _eccen_cmap = matplotlib.colors.LinearSegmentedColormap(
        'eccentricity',
        {'red':   ((0.0,       0.0, 0.0),
                   (2.5/90.0,  0.5, 0.5),
                   (5.0/90.0,  1.0, 1.0),
                   (10.0/90.0, 1.0, 1.0),
                   (20.0/90.0, 0.0, 0.0),
                   (40.0/90.0, 0.0, 0.0),
                   (90.0/90.0, 1.0, 1.0)),
         'green': ((0.0,       0.0, 0.0),
                   (2.5/90.0,  0.0, 0.0),
                   (5.0/90.0,  0.0, 0.0),
                   (10.0/90.0, 1.0, 1.0),
                   (20.0/90.0, 1.0, 1.0),
                   (40.0/90.0, 1.0, 1.0),
                   (90.0/90.0, 1.0, 1.0)),
         'blue':  ((0.0,       0.0, 0.0),
                   (2.5/90.0,  0.5, 0.5),
                   (5.0/90.0,  0.0, 0.0),
                   (10.0/90.0, 0.0, 0.0),
                   (20.0/90.0, 0.0, 0.0),
                   (40.0/90.0, 1.0, 1.0),
                   (90.0/90.0, 1.0, 1.0))})
    def vertex_eccen_color(m, weight_cutoff=0.2, weighted=True):
        global _eccen_cmap
        w = vertex_weight(m)
        curvColor = np.asarray(vertex_curvature_color(m))
        if weighted and w < weight_cutoff: return curvColor
        eccColor = np.asarray(_eccen_cmap(m['eccentricity']/90.0))
        if weighted:
            return eccColor*w + curvColor*(1-w)
        else:
            return eccColor
    def curvature_colors(m):
        return np.asarray(m.map_vertices(vertex_curvature_color))
    def angle_colors(m):
        return np.asarray(m.map_vertices(vertex_angle_color))
    def eccen_colors(m):
        return np.asarray(m.map_vertices(vertex_eccen_color))
    def colors_to_cmap(colors):
        colors = np.asarray(colors)
        if colors.shape[1] == 3:
            colors = np.hstack((colors, np.ones((len(colors),1))))
        steps = (0.5 + np.asarray(range(len(colors)-1), dtype=np.float))/(len(colors) - 1)
        return matplotlib.colors.LinearSegmentedColormap(
            'auto_cmap',
            {clrname: ([(0, col[0], col[0])] +
                       [(step, c0, c1) for (step,c0,c1) in zip(steps, col[:-1], col[1:])] +
                       [(1, col[-1], col[-1])])
             for (clridx,clrname) in enumerate(['red', 'green', 'blue', 'alpha'])
             for col in [colors[:,clridx]]},
            N=(len(colors)))
    def cortex_plot(the_map, color=None, plotter=matplotlib.pyplot, weights=Ellipsis):
        '''
        cortex_plot(map) yields a plot of the given 2D cortical mesh, map. The following options are
        accepted:
          * color (default: None) specifies a function that, when passed a single argument, a dict
            of the properties of a single vertex, yields an RGBA list for that vertex. By default,
            uses the curvature colors.
          * weight (default: Ellipsis) specifies that the given weights should be used instead of
            the weights attached to the given map; note that Ellipsis indicates that the current
            map's weights should be used. If None or a single number is given, then all weights are
            considered to be 1. A string may be given to indicate that a property should be used.
          * plotter (default: matplotlib.pyplot) specifies a particular plotting object should be
            used. If plotter is None, then instead of attempting to render the plot, a tuple of
            (tri, zs, cmap) is returned; in this case, tri is a matplotlib.tri.Triangulation
            object for the given map and zs and cmap are an array and colormap (respectively) that
            will produce the correct colors. Without plotter equal to None, these would instead
            be rendered as plotter.tripcolor(tri, zs, cmap, shading='gouraud').
        '''
        tri = matplotlib.tri.Triangulation(the_map.coordinates[0],
                                           the_map.coordinates[1],
                                           triangles=the_map.indexed_faces.T)
        if weights is not Ellipsis:
            if weights is None or not hasattr(weights, '__iter__'):
                weights = np.ones(the_map.vertex_count)
            elif isinstance(weights, basestring):
                weights = the_map.prop(weights)
            the_map = the_map.using(properties=the_map.properties.using(weight=weights))
        if isinstance(color, np.ndarray):
            colors = color
        else:
            if color is None or color == 'curv' or color == 'curvature':
                color = vertex_curvature_color
            elif color == 'angle' or color == 'polar_angle':
                color = vertex_angle_color
            elif color == 'eccen' or color == 'eccentricity':
                color = vertex_eccen_color
            colors = np.asarray(the_map.map_vertices(color))
        cmap = colors_to_cmap(colors)
        zs = np.asarray(range(the_map.vertex_count), dtype=np.float) / (the_map.vertex_count - 1)
        if plotter is None:
            return (tri, zs, cmap)
        else:
            return plotter.tripcolor(tri, zs, cmap=cmap, shading='gouraud')
except:
    pass

