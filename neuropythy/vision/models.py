####################################################################################################
# neuropythy/retinotopy/models.py
# Importing and interpreting of flat mesh models for registration.
# By Noah C. Benson

import numpy                 as     np
import numpy.linalg          as     npla
import scipy                 as     sp
import scipy.spatial         as     space
import pyrsistent            as     pyr
import os, gzip, types, six, pimms

from ..           import geometry as geo
from ..           import mri      as mri
from ..java       import (java_link, serialize_numpy,
                                     to_java_doubles, to_java_ints, to_java_array)
from ..util       import (to_affine, library_path, is_tuple, is_list)
from ..io         import importer

# These two variables are intended to provide default orderings to visual areas (but in general,
# visual areas should be referred to by name OR as a number paired with a model).
visual_area_names = (None,
                     'V1', 'V2', 'V3', 'hV4', 'VO1', 'VO2', 'LO1', 'LO2',
                     'TO1', 'TO2', 'V3b', 'V3a')
visual_area_numbers = pyr.pmap({v:k for (k,v) in enumerate(visual_area_names)})

@pimms.immutable
class RetinotopyModel(object):
    '''
    RetinotopyModel is a class designed to be inherited by other models of retinotopy; any class
    that inherits from RetinotopyModel must implement the following methods to work properly with
    the registration system of the neuropythy library.
    '''

    def __init__(self, area_name_to_id):
        self.area_name_to_id = area_name_to_id
    @pimms.param
    def area_name_to_id(vai):
        '''
        mdl.area_name_to_id is a persistent map whose keys are area names (such as 'V1' or 'hV4')
        and whose values are the area id (a number greater than 0) for that area.
        mdl.area_name_to_id is a parameter which may be provided as a lsit of area names, in which
        case the first is assumed to be area 1, the next area 2, etc.
        '''
        if vai is None: return None
        if not pimms.is_map(vai): return pyr.pmap({nm:(ii+1) for (ii,nm) in enumerate(vai)})
        elif pimms.is_pmap(vai): return vai
        else: return pyr.pmap(vai)
    @pimms.value
    def area_id_to_name(area_name_to_id):
        '''
        mdl.area_id_to_name is a persistent map whose keys are area id's and whose values are the
        associated area's name.
        '''
        if area_name_to_id is None: return None
        return pyr.pmap({v:k for (k,v) in six.iteritems(area_name_to_id)})
    # Methods that must be overloaded!
    def angle_to_cortex(self, theta, rho):
        '''
        model.angle_to_cortex(theta, rho) yields a (k x 2) matrix in which each row corresponds to
        an area map (e.g. V2 or MT) and the two columns represent x and y coordinates of the
        predicted location in a flattened cortical map at which one would find the visual angle
        values of theta (polar angle) and rho (eccentricity). Theta should be in units of degrees
        between 0 (upper vertical meridian) and 180 (lower vertical meridian).
        If theta and rhos are both vectors of length n, then the result is an (n x k x 2) matrix
        with one entry for each theta/rho pair.
        '''
        raise NotImplementedError(
            'Object with base class RetinotopyModel did not override angle_to_cortex')
    def cortex_to_angle(self, x, y):
        '''
        model.cortex_to_angle(x, y) yields a vector of (polar-angle, eccentricity, id) corresponding
        to the given (x,y) coordinates from a cortical map. The id that is returned is a positive
        integer corresponding to an ROI label (e.g., V1, MT).
        If x and y are vectors of length n then the return value is an (n x 3) matrix in which each
        row corresponds to one (x,y) pair.
        '''
        raise NotImplementedError(
            'Object with base class RetinotopyModel did not override cortex_to_angle')

# How we construct a Schira Model:
@pimms.immutable
class SchiraModel(RetinotopyModel):
    '''
    The SchiraModel class is a class that inherits from RetinotopyModel and acts as a Python wrapper
    around the Java nben.neuroscience.SchiraModel class; it handles conversion from visual field 
    angle to cortical surface coordinates and vice versa for the Banded Double-Sech model proposed
    in the following paper:
    Schira MM, Tyler CW, Spehar B, Breakspear M (2010) Modeling Magnification and Anisotropy in the
    Primate Foveal Confluence. PLOS Comput Biol 6(1):e1000651. doi:10.1371/journal.pcbi.1000651.
    '''

    # These are the accepted arguments to the model:
    default_parameters = pyr.pmap({'A': 1.05,
                                   'B': 90.0,
                                   'lam': 0.4,
                                   'psi': 0.15,
                                   'scale': (21.0, 21.0),
                                   'shear': ((1.0, 0.0), (0.0, 1.0)),
                                   'center': (-6.0, 0.0),
                                   'v1size': 1.2,
                                   'v2size': 0.6,
                                   'v3size': 0.4,
                                   'hv4size': 0.9,
                                   'v3asize': 0.9})
    visual_area_names = ('V1', 'V2', 'V3')

    def __init__(self, A=1.05, B=90.0, lam=0.4, psi=0.15, scale=(21.0, 21.0),
                 shear=((1.0,0.0),(0.0,1.0)), center=(-6.0, 0.0),
                 v1size=1.2, v2size=0.6, v3size=0.4, hv4size=0.9, v3asize=0.9):
        self.area_name_to_id = visual_area_names
        self.parameters = pyr.m(A=A, B=B, lam=lam, psi=psi, scale=scale, shear=shear,
                                center=center, v1size=v1size, v2size=v2size, v3size=v3size,
                                hv4size=hv4size, v3asize=v3asize)
    @pimms.param
    def parameters(params):
        '''
        mdl.parameters is a persistent map of the parameters for the given SchiraModel object mdl.
        '''
        if not pimms.is_pmap(params): params = pyr.pmap(params)
        # do the translations that we need...
        scale = params['scale']
        if pimms.is_number(scale):
            params = params.set('scale', (scale, scale))
        elif not is_tuple(scale):
            params = params.set('scale', tuple(scale))
        shear = params['shear']
        if pimms.is_number(shear) and np.isclose(shear, 0):
            params = params.set('shear', ((1, 0), (0, 1)))
        elif shear[0][0] != 1 or shear[1][1] != 1:
            raise RuntimeError('shear matrix diagonal elements must be 1!')
        elif not is_tuple(shear) or not all(is_tuple(s) for s in shear):
            params.set('shear', tuple([tuple(s) for s in shear]))
        center = params['center']
        if pimms.is_number(center) and np.isclose(center, 0):
            params = params.set('center', (0.0, 0.0))
        return pimms.persist(params, depth=None)

    @pimms.value
    def _java_object(parameters):
        '''
        mdl._java_object is the java representation of the SchiraModel object mdl.
        '''
        # Okay, let's construct the object...
        return java_link().jvm.nben.neuroscience.SchiraModel(
            parameters['A'],
            parameters['B'],
            parameters['lam'],
            parameters['psi'],
            parameters['v1size'],
            parameters['v2size'],
            parameters['v3size'],
            parameters['hv4size'],
            parameters['v3asize'],
            parameters['center'][0],
            parameters['center'][1],
            parameters['scale'][0],
            parameters['scale'][1],
            parameters['shear'][0][1],
            parameters['shear'][1][0])

    def angle_to_cortex(self, theta, rho):
        iterTheta = hasattr(theta, '__iter__')
        iterRho = hasattr(rho, '__iter__')
        jarr = None
        if iterTheta and iterRho:
            if len(theta) != len(rho):
                raise RuntimeError('Arguments theta and rho must be the same length!')
            jarr = self._java_object.angleToCortex(to_java_doubles(theta), to_java_doubles(rho))
        elif iterTheta:
            jarr = self._java_object.angleToCortex(to_java_doubles(theta),
                                                   to_java_doubles([rho for t in theta]))
        elif iterRho:
            jarr = self._java_object.angleToCortex(to_java_doubles([theta for r in rho]),
                                                   to_java_doubles(rho))
        else:
            return self._java_object.angleToCortex(theta, rho)
        return np.asarray([[c for c in r] for r in jarr])
    def cortex_to_angle(self, x, y):
        iterX = hasattr(x, '__iter__')
        iterY = hasattr(y, '__iter__')
        jarr = None
        if iterX and iterY:
            if len(x) != len(y):
                raise RuntimeError('Arguments x and y must be the same length!')
            jarr = self._java_object.cortexToAngle(to_java_doubles(x), to_java_doubles(y))
        elif iterX:
            jarr = self._java_object.cortexToAngle(to_java_doubles(x),
                                                   to_java_doubles([y for i in x]))
        elif iterY:
            jarr = self._java_object.cortexToAngle(to_java_doubles([x for i in y]),
                                                   to_java_doubles(y))
        else:
            return self._java_object.cortexToAngle(x, y)
        dat = np.asarray([[c for c in r] for r in jarr])
        a = dat[:,2]
        a = np.round(np.abs(a))
        a[a > 3] = 0
        dat[:,2] = a
        return dat

@pimms.immutable
class RetinotopyMeshModel(RetinotopyModel):
    '''
    RetinotopyMeshModel is a class that represents a retinotopic map or set of retinotopic maps on
    the flattened 2D cortex.
    RetinotopyMeshModel(tris, coords, polar_angle, eccen, areas) yields a retinotopy mesh model
    object in which the given triangle and coordinate matrices form the mesh and the polar_angle,
    eccen, and areas give the appropriate data for each vertex in coords. Note that the areas
    parameter should be 0 on any boundary vertex and an integer labelling the roi for any other
    vertex.
    '''

    def __init__(self, triangles, coordinates, angles, eccens, area_ids, transform=None,
                 area_name_to_id=None):
        self.faces = triangles
        self.cortical_coordinates = coordinates
        self.polar_angles = angles
        self.eccentricities = eccens
        self.visual_areas = area_ids
        self.transform = transform
        self.area_name_to_id = area_name_to_id
            

    @pimms.param
    def faces(tris):
        'mdl.faces is the triangle matrix for the given retinotopy mesh model mdl.'
        tris = np.asarray(tris, dtype=np.int)
        if tris.shape[0] != 3: tris = tris.T
        if tris.shape[0] != 3: raise ValueError('triangle matrix must have 3 rows or columns')
        return pimms.imm_array(tris)
    @pimms.param
    def cortical_coordinates(coords):
        '''
        mdl.cortical_coordinates is the coordinate matrix for the given retinotopy mesh model mdl's
        representation of the cortical surface.
        '''
        coords = np.asarray(coords)
        if coords.shape[0] != 2: coords = coords.T
        if coords.shape[0] != 2: raise ValueError('coordinate matrix must have 2 rows or columns')
        return pimms.imm_array(coords)
    @pimms.param
    def polar_angles(angs):
        'mdl.polar_angles is the vector of polar angle values for the given retinotopy mesh model.'
        return pimms.imm_array(angs)
    @pimms.param
    def eccentricities(eccs):
        'mdl.eccentrities is the vector of eccentricity values for the given retinotopy mesh model.'
        return pimms.imm_array(eccs)
    @pimms.param
    def visual_areas(labs):
        'mdl.visual_areas is the vector of visual area labels for the given retinotopy mesh model.'
        return pimms.imm_array(labs)
    @pimms.param
    def transform(tx):
        '''
        mdl.transform is the matrix for the affine transform applied to the coordinates on the
        cortical surface representation
        '''
        if tx is None: return None
        tx = to_affine(tx)
        if np.array_equal(tx, np.eye(3)): return None
        tx.setflags(write=False)
        return tx

    @pimms.value
    def inverse_transform(transform):
        '''
        mdl.inverse_transform is the inverse transform (see RetinotopyMeshModel.transform).
        '''
        if transform is None: return None
        return pimms.imm_array(npla.inv(transform))
    @pimms.value
    def visual_coordinates(polar_angles, eccentricities):
        '''
        mdl.cortical_coordinates is the coordinate matrix for the given retinotopy mesh model mdl's
        representation of the cortical surface.
        '''
        z = eccentricities * np.exp(1j * np.pi/180.0 * (90.0 - polar_angles))
        return pimms.imm_array([z.real, z.imag])
    @pimms.value
    def cleaned_visual_areas(visual_areas, faces):
        '''
        mdl.cleaned_visual_areas is the same as mdl.visual_areas except that vertices with visual
        area values of 0 (boundary values) are given the mode of their neighbors.
        '''
        area_ids = np.array(visual_areas)
        boundaryNeis = {}
        for (b,inside) in [(b, set(inside))
                           for t in faces.T
                           for (bound, inside) in [([i for i in t if area_ids[i] == 0],
                                                    [i for i in t if area_ids[i] != 0])]
                           if len(bound) > 0 and len(inside) > 0
                           for b in bound]:
            if b in boundaryNeis: boundaryNeis[b] |= inside
            else:                 boundaryNeis[b] =  inside
        for (b,neis) in six.iteritems(boundaryNeis):
            area_ids[b] = np.argmax(np.bincount(area_ids[list(neis)]))
        return pimms.imm_array(np.asarray(area_ids, dtype=np.int))
    @pimms.value
    def tess(faces, cortical_coordinates, visual_coordinates,
             polar_angles, eccentricities, cleaned_visual_areas):
        'mdl.tess is the tesselation object for mesh model.'
        props = pimms.itable({'polar_angle':  polar_angles,
                              'eccentricity': eccentricities,
                              'visual_area':  cleaned_visual_areas,
                              'cortical_coordinates': cortical_coordinates.T,
                              'visual_coordinates':   visual_coordinates.T})
        if isinstance(faces, geo.Tesselation): return faces.copy(properties=props)
        return geo.Tesselation(faces, properties=props).persist()
    @pimms.value
    def cortical_mesh(tess, cortical_coordinates):
        '''
        mdl.cortical_mesh is the mesh object that represents the 2D cortical surface of the model.
        '''
        return tess.make_mesh(cortical_coordinates).persist()
    @pimms.value
    def visual_meshes(tess, visual_coordinates, cleaned_visual_areas):
        '''
        mdl.visual_meshes is a map of meshes; the keys of the map are the unique visual area id's
        in the given retinotopy mesh model (mdl) and the values are the meshes that represent them.
        '''
        visual_areas = cleaned_visual_areas
        def _make_submesh(area_label):
            def _fn():
                idx = np.where(visual_areas == area_label)[0]
                st = tess.subtess(idx)
                return st.make_mesh(visual_coordinates[:, st.labels]).persist()
            return _fn
        return pimms.lazy_map({k:_make_submesh(k) for k in np.unique(visual_areas) if k != 0})

    
    def cortex_to_angle(self, x, y):
        'See RetinotopyModel.cortex_to_angle.'
        if not pimms.is_vector(x): return self.cortex_to_angle([x], [y])[0]
        # start by applying the transform to the points
        tx = self.inverse_transform
        xy = np.asarray([x,y]).T if tx is None else np.dot(tx, [x,y,np.ones(len(x))])[0:2].T
        # we only need to interpolate from the inverse mesh in this case
        interp_ae = self.cortical_mesh.interpolate(xy, [self.polar_angles, self.eccentricities],
                                                   method='linear')
        interp_id = self.cortical_mesh.interpolate(xy, self.visual_areas,
                                                   method='heaviest')
        interp = np.asarray([interp_ae[0], interp_ae[1], interp_id])
        bad = np.where(np.isnan(np.prod(interp, axis=0)))[0]
        interp[:,bad] = 0.0
        return interp
    def angle_to_cortex(self, theta, rho):
        'See help(neuropythy.registration.RetinotopyModel.angle_to_cortex).'
        #TODO: This should be made to work correctly with visual area boundaries: this could be done
        # by, for each area (e.g., V2) looking at its boundaries (with V1 and V3) and flipping the
        # adjacent triangles so that there is complete coverage of each hemifield, guaranteed.
        if not pimms.is_vector(theta): return self.angle_to_cortex([theta], [rho])[0]
        theta = np.asarray(theta)
        rho = np.asarray(rho)
        zs = np.asarray(
            rho * np.exp([np.complex(z) for z in 1j * ((90.0 - theta)/180.0*np.pi)]),
            dtype=np.complex)
        coords = np.asarray([zs.real, zs.imag]).T
        if coords.shape[0] == 0: return np.zeros((0, len(self.visual_meshes), 2))
        # we step through each area in the forward model and return the appropriate values
        tx = self.transform
        res = np.transpose(
            [self.visual_meshes[area].interpolate(coords, 'cortical_coordinates', method='linear')
             for area in sorted(self.visual_meshes.keys())],
            (1,0,2))
        if tx is not None:
            res = np.asarray(
                [np.dot(tx, np.vstack((area_xy.T, np.ones(len(area_xy)))))[0:2].T
                 for area_xy in res])
        return res

@pimms.immutable
class RegisteredRetinotopyModel(RetinotopyModel):
    '''
    RegisteredRetinotopyModel is a class that represents a retinotopic map or set of retinotopic
    maps on the flattened 2D cortex OR on the 3D cortical surface, via a registration and set of
    map projection parameters.
    RegisteredRetinotopyModel(model, projection_params) yields a retinotopy
    mesh model object in which the given RetinotopyModel object model describes the 2D 
    retinotopy that is predicted for the vertices that result from a map projection, defined using
    the given projection_params dictionary, of the given registration. In other words, the resulting
    RegisteredRetinotopyModel will, when given a hemisphere object to which to apply the model,
    will look up the appropriate registration (by the name registration_name) make a map projection
    of using the given projection_params dictionary, and apply the model to the resulting
    coordinates.
    '''

    def __init__(self, model, mapproj):
        '''
        RegisteredRetinotopyModel(retinotopy_model, map_projection) yields a retinotopy mesh model
        object in which the given RetinotopyModel object describes the 2D retinotopy that is
        predicted for the vertices that result from a map projection, defined using the given
        MapProjection object, of the given registration. In other words, the resulting
        RegisteredRetinotopyModel will, when given a cortex object to which to apply the model, will
        look up the appropriate registration (found in the map projection) make a map projection of
        the cortex using the projection, and apply the model to the resulting coordinates. See also
        neuropythy.geometry.MapProjection.
        '''
        self.model = model
        self.map_projection = mapproj
        self.area_name_to_id = model.area_name_to_id
    @pimms.param
    def model(mdl):
        '''
        rrm.model is the retinotopy model object for the RegisteredRetinotopyModel object rrm.
        '''
        if not isinstance(mdl, RetinotopyModel):
            raise ValueError('given parameter model must be a RetinotopyModel instance')
        return pimms.persist(mdl)
    @pimms.param
    def map_projection(mp):
        '''
        rrm.map_projection is the MapProjection object for the RegisteredRetinotopyModel object rrm.
        '''
        if not isinstance(mp, geo.MapProjection):
            raise ValueError('given parameter map_projection must be a MapProjection instance')
        return pimms.persist(mp)

    def save(self, f):
        '''
        model.save(filename) saves an FMM-formatted file to the given filename; the FMM format is
        used by neuropythy to save and load registered retinotopy models; it can be loaded with the
        load_fmm_model function.
        model.save(file) will write the text directly to the given file.
        '''
        if not isinstance(self.model, RetinotopyMeshModel):
            raise ValueError('Only RetinotopyMeshModels can be saved to an fmm file')
        if pimms.is_str(f):
            with open(f, 'w') as fl:
                self.save(fl)
            return f
        m = self.model
        x0 = self.map_projection.center
        x1 = self.map_projection.center_right
        tx = np.eye(3) if m.transform is None else m.transform
        chir = self.map_projection.chirality
        if chir is not None: chir = chir.upper()
        for ln in ['Flat Mesh Model Version: 1.0',
                   'Points: %d' % m.coordinates.shape[1],
                   'Triangles: %d' % m.faces.shape[1],
                   'Registration: %s' % self.map_projection.registration,
                   'Hemisphere: %s' % chir,
                   'Center: %f,%f,%f' % (x0[0], x0[1], x0[2]),
                   'OnXAxis: %f,%f,%f' % (x1[0], x1[1], x1[2]),
                   'Method: %s' % self.map_projection.method.capitalize(),
                   'Transform: [%f,%f,%f;%f,%f,%f;%f,%f,%f]' % tuple(tuple(x) for x in tx)]:
            f.write(ln + '\n')
        if self.area_name_to_id:
            lbls = [x for (_,x) in sorted(six.iteritems(self.area_name_to_id), key=lambda x:x[0])]
            f.write('AreaNames: [%s]\n' % ' '.join(lbls))
        (xs,ys) = m.coordinates
        for (x,y,t,r,a) in zip(xs, ys, m.polar_angles, m.eccentricities, m.visual_areas):
            f.write('%f,%f :: %f,%f,%f\n' % (x,y,t,r,a))
        for (a,b,c) in zip(**(m.faces + 1)):
            f.write('%d,%d,%d\n' % (a,b,c))
        return f
    def cortex_to_angle(self, *args):
        '''
        The cortex_to_angle method of the RegisteredRetinotopyModel class is identical to that
        of the RetinotopyModel class, but the method may be given a map, mesh, or cortex, in
        which case the result is applied to the coordinates after the appropriate transformation (if
        any) is first applied.
        '''
        if len(args) == 1:
            # see if we can cast it to something; first a mesh...
            try:              m = geo.to_mesh(args[0])
            except Exception: m = None
            if m is not None:
                if geo.is_flatmap(m):
                    (x,y) = m.coordinates
                    return self.model.cortex_to_angle(x, y)
                else:
                    fm = self.map_projection(m)
                    res = np.zeros((3, args[0].vertex_count))
                    c2a = np.asarray(self.cortex_to_angle(fm.coordinates))
                    res[:, fm.labels] = (c2a if len(c2a) == len(res) else c2a.T)
                    return res
            # next, try a cortex
            try:              c = mri.to_cortex(args[0])
            except Exception: c = None
            if c is not None:
                m = self.map_projection(c)
                res = np.zeros((3, c.vertex_count))
                c2a = np.asarray(self.cortex_to_angle(m.coordinates))
                res[:, m.labels] = (c2a if len(c2a) == len(res) else c2a.T)
                return res
            # finally, assume a coordinate matrix
            X = np.asarray(args[0])
            if len(X.shape) != 2: raise ValueError('given coordinate matrix must be rectangular')
            X = X if X.shape[0] == 2 or X.shape[0] == 3 else X.T
            if X.shape[0] == 2: return self.model.cortex_to_angle(X[0], X[1])
            elif X.shape[0] == 3:
                (x,y) = self.map_projection(X)
                return self.model.cortex_to_angle(x, y)
            else: raise ValueError('coordinate matrix must be 2 or 3 dimensional')
        else: return self.model.cortex_to_angle(*args)
    def angle_to_cortex(self, *args):
        '''
        The angle_to_cortex method of the RegisteredRetinotopyModel class is identical to that
        of the RetinotopyModel class, but the method may be given a map, mesh, or hemisphere, in
        which case the result is applied to the 'polar_angle' and 'eccentricity' properties.
        '''
        if len(args) == 1:
            if geo.is_vset(args[0]):
                ang = vis.retinotopy_data(args[0], 'polar_angle')
                ecc = vis.retinotopy_data(args[0], 'eccentricity')
                return self.model.angle_to_cortex(ang, ecc)
            else:
                tr = np.asarray(args)
                if tr.shape[1] == 2: tr = tr.T
                elif tr.shape[0] != 2: raise ValueError('cannot interpret argument')
                return self.model.angle_to_cortex(tr[0], tr[1])
        else: return self.model.angle_to_cortex(*args)

@importer('flatmap_model', ('fmm', 'fmm.gz'))
def load_fmm_model(filename, radius=np.pi/3.0, sphere_radius=100.0):
    '''
    load_fmm_model(filename) yields the fmm model indicated by the given file name. Fmm models are
    triangle meshes that define a field value at every vertex as well as the parameters of a
    projection to the 2D cortical surface via a registartion name and projection parameters.
    The following options may be given:
      * radius (default: pi/3) specifies the radius that should be assumed by the model; fmm models
        do not specify radius values in their projection parameters, but pi/3 is usually large for
        localized models.
      * sphere_radius (default: 100) specifies the radius of the sphere that should be assumed by
        the model. Note that in Freesurfer, spheres have a radius of 100.
    '''
    if not os.path.exists(filename):
        models_path = os.path.join(library_path(), 'models')
        # we look for it in a number of ways:
        fname = next((fnm
                      for beg in ['', models_path] for end in ['', '.fmm', '.fmm.gz']
                      for fnm0 in [filename + end]
                      for fnm in [fnm0 if beg is None else os.path.join(beg, fnm0)]
                      if os.path.exists(fnm)),
                     None)
        if fname is None:
            raise ValueError('Given model/file name does not exist: %s' % filename)
        filename = fname
    if not os.path.isfile(filename):
        raise ValueError('Given filename (%s) is not a file!' % filename)
    gz = True if len(filename) > 3 and filename[-3:] == '.gz' else False
    lines = None
    with (gzip.open(filename, 'rt') if gz else open(filename, 'rt')) as f:
        lines = f.read().split('\n')
    if len(lines) < 3 or lines[0] != 'Flat Mesh Model Version: 1.0':
        raise ValueError('Given file does not contain to a valid flat mesh model!')
    n = int(lines[1].split(':')[1].strip())
    m = int(lines[2].split(':')[1].strip())
    reg = lines[3].split(':')[1].strip()
    hemi = lines[4].split(':')[1].strip().upper()
    center = list(map(float, lines[5].split(':')[1].strip().split(',')))
    onxaxis = list(map(float, lines[6].split(':')[1].strip().split(',')))
    method = lines[7].split(':')[1].strip().lower()
    tx = np.asarray(
        [list(map(float, row.split(',')))
         for row in lines[8].split(':')[1].strip(' \t[]').split(';')])
    if lines[9].startswith('AreaNames: ['):
        # we load the area names
        s = lines[9][12:-1]
        area_names = tuple(s.split(' '))
        l0 = 10        
    else:
        area_names = None
        l0 = 9
    crds = []
    for row in lines[l0:(n+l0)]:
        (left,right) = row.split(' :: ')
        crds.append(list(map(float, left.split(','))))
    crds = np.asarray([list(map(float, left.split(',')))
                       for row in lines[l0:(n+l0)]
                       for (left,right) in [row.split(' :: ')]])
    vals = np.asarray([list(map(float, right.split(',')))
                       for row in lines[l0:(n+l0)]
                       for (left,right) in [row.split(' :: ')]])
    tris = -1 + np.asarray(
        [list(map(int, row.split(',')))
         for row in lines[(n+l0):(n+m+l0)]])
    return RegisteredRetinotopyModel(
        RetinotopyMeshModel(tris, crds,
                            90-180/np.pi*vals[:,0], vals[:,1], np.asarray(vals[:,2], dtype=np.int),
                            transform=tx,
                            area_name_to_id=area_names),
        geo.MapProjection(registration=reg,
                          center=center,
                          center_right=onxaxis,
                          method=method,
                          radius=radius,
                          sphere_radius=sphere_radius,
                          chirality=hemi))
