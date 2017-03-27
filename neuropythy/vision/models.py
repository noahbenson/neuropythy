####################################################################################################
# neuropythy/retinotopy/models.py
# Importing and interpreting of flat mesh models for registration.
# By Noah C. Benson

import numpy                as     np
import scipy                as     sp
import scipy.spatial        as     space
from   pysistence           import make_dict
from   numbers              import Number
import numpy.linalg, os, math, gzip

import neuropythy.geometry   as     geo
import neuropythy.freesurfer as     nfs
import neuropythy.cortex     as     ncx
from   neuropythy.immutable  import Immutable
from   neuropythy.java       import (java_link, serialize_numpy,
                                     to_java_doubles, to_java_ints, to_java_array)

class RetinotopyModel:
    '''
    RetinotopyModel is a class designed to be inherited by other models of retinotopy; any class
    that inherits from RetinotopyModel must implement the following methods to work properly with
    the registration system of the neuropythy library.
    '''
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
    default_parameters = {
        'A': 1.05,
        'B': 90.0,
        'lambda': 0.4,
        'psi': 0.15,
        'scale': [21.0, 21.0],
        'shear': [[1.0, 0.0], [0.0, 1.0]],
        'center': [-6.0, 0.0],
        'v1size': 1.2,
        'v2size': 0.6,
        'v3size': 0.4,
        'hv4size': 0.9,
        'v3asize': 0.9}
    # This function checks the given arguments to see if they are okay:
    def __check_parameters(self, parameters):
        # we don't care if there are extra parameters; we just make sure the given parameters make
        # sense, then return the full set of parameters
        opts = {
            k: parameters[k] if k in parameters else v
            for (k,v) in SchiraModel.default_parameters.iteritems()}
        return opts

    # This class is immutable: don't change the params to change the model; 
    # don't change the java object!
    def __setattr__(self, name, val):
        raise ValueError('The SchiraModel class is immutable; its objects cannot be edited')

    def __init__(self, **opts):
        # start by getting the proper parameters
        params = self.__check_parameters(opts)
        # Now, do the translations that we need...
        if isinstance(params['scale'], Number) or np.issubdtype(type(params['scale']), np.float):
            params['scale'] = [params['scale'], params['scale']]
        if (isinstance(params['shear'], Number) or np.issubdtype(type(params['shear']), np.float)) \
           and params['shear'] == 0:
            params['shear'] = [[1, 0], [0, 1]]
        elif params['shear'][0][0] != 1 or params['shear'][1][1] != 1:
            raise RuntimeError('shear matrix [0,0] elements and [1,1] elements must be 1!')
        if ((isinstance(params['center'], Number)
             or np.issubdtype(type(params['center']), np.float))
            and params['center'] == 0):
            params['center'] = [0.0, 0.0]
        self.__dict__['parameters'] = make_dict(params)
        # Okay, let's construct the object...
        self.__dict__['_java_object'] = java_link().jvm.nben.neuroscience.SchiraModel(
            params['A'],
            params['B'],
            params['lambda'],
            params['psi'],
            params['v1size'],
            params['v2size'],
            params['v3size'],
            params['hv4size'],
            params['v3asize'],
            params['center'][0],
            params['center'][1],
            params['scale'][0],
            params['scale'][1],
            params['shear'][0][1],
            params['shear'][1][0])
    
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
        return np.asarray([[c for c in r] for r in jarr])
        

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

    def __init__(self, triangles, coordinates, angles, eccens, area_ids, transform=None):
        triangles   = np.asarray(triangles)
        coordinates = np.asarray(coordinates)
        triangles   = triangles   if triangles.shape[1] == 3   else triangles.T
        coordinates = coordinates if coordinates.shape[1] == 2 else coordinates.T
        angles      = np.asarray(angles)
        eccens      = np.asarray(eccens)
        area_ids    = np.asarray(map(int, area_ids))
        # The forward model is the projection from cortical map -> visual angle
        self.forward = geo.Mesh(triangles, coordinates)
        # The inverse model is a set of meshes from visual field space to the cortical map
        xs = coordinates[:,0]
        ys = coordinates[:,1]
        zs = eccens * np.exp(1j * (90 - angles)/180*math.pi)
        coords = np.asarray([zs.real, zs.imag]).T
        self.inverse = {
            area: geo.Mesh(np.asarray(tris), coords)
            # Iterate over all the unique areas;
            for area in list(set(area_ids) - set([0]))
            # bind the triangles (0 area_ids indicate borders):
            for tris in [[t for t in triangles if (set(area_ids[t]) - set([0])) == set([area])]]}
        # Note the transform:
        self.transform = np.asarray(transform) if transform is not None else None
        self.itransform = numpy.linalg.inv(transform) if transform is not None else None
        # Save the data:
        self.data = {}
        self.data['x'] = xs
        self.data['y'] = ys
        self.data['polar_angle'] = angles
        self.data['eccentricity'] = eccens
        # we have to fix the area_ids to be the mean of their neighbors when on a boundary:
        boundaryNeis = {}
        for (b,inside) in [(b, set(inside))
                           for t in triangles
                           for (bound, inside) in [([i for i in t if area_ids[i] == 0],
                                                    [i for i in t if area_ids[i] != 0])]
                           if len(bound) > 0 and len(inside) > 0
                           for b in bound]:
            if b not in boundaryNeis: boundaryNeis[b] = inside
            else: boundaryNeis[b] |= inside
        for (b,neis) in boundaryNeis.iteritems():
            area_ids[b] = np.mean(area_ids[list(neis)])
        self.data['id'] = area_ids

    def cortex_to_angle(self, x, y):
        'See RetinotopyModel.cortex_to_angle.'
        if not hasattr(x, '__iter__'):
            return self.cortex_to_angle([x], [y])[0]
        # start by applying the transform to the points
        tx = self.itransform
        xy = np.asarray([x,y]).T if tx is None else np.dot(tx, [x,y,[1 for i in x]])[0:2].T
        # we only need to interpolate from the inverse mesh in this case
        interp_ae = self.forward.interpolate(
            xy,
            [self.data[tt] for tt in ['polar_angle', 'eccentricity']],
            method='automatic',
            null=np.nan)
        interp_id = self.forward.interpolate(
            xy,
            self.data['id'],
            method='nearest',
            null=np.nan)
        interp = np.asarray([interp_ae[0], interp_ae[1], interp_id])
        bad = np.where(np.isnan(np.prod(interp, axis=0)))[0]
        interp[:,bad] = 0.0
        return interp

    def angle_to_cortex(self, theta, rho):
        'See help(neuropythy.registration.RetinotopyModel.angle_to_cortex).'
        if not hasattr(theta, '__iter__'):
            return self.angle_to_cortex([theta], [rho])[0]
        theta = np.asarray(theta)
        rho = np.asarray(rho)
        zs = np.asarray(
            rho * np.exp([np.complex(z) for z in 1j * ((90.0 - theta)/180.0*math.pi)]),
            dtype=np.complex)
        coords = np.asarray([zs.real, zs.imag]).T
        # we step through each area in the forward model and return the appropriate values
        tx = self.transform
        xvals = self.data['x']
        yvals = self.data['y']
        res = np.asarray(
            [[self.inverse[area].interpolate(coords, xvals, smoothing=1),
              self.inverse[area].interpolate(coords, yvals, smoothing=1)]
             for area in map(int, sorted(list(set(self.data['id']))))
             if area != 0]
        ).transpose((2,0,1))
        if tx is not None:
            res = np.asarray(
                [[np.dot(tx, [xy[0], xy[1], 1])[0:2] if xy[0] is not None else [None, None]
                  for xy in ptdat]
                 for ptdat in res])
        # there's a chance that the coords are outside the triangle mesh; we want to make sure
        # that these get handled correctly...
        for (i,ptdat) in enumerate(res):
            for row in ptdat:
                if None in set(row.flatten()) and rho[i] > 86 and rho[i] <= 90:
                    # we try to get a fixed version by reducing rho slightly
                    res[i] = self.angle_to_cortex(theta[i], rho[i] - 0.5);
        return res

             
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

    def __init__(self, model, **projection_params):
        '''
        RegisteredRetinotopyModel(model, <projection parameters...>) yields a
        retinotopy mesh model object in which the given RetinotopyModel object model describes
        the 2D retinotopy that is predicted for the vertices that result from a map projection,
        defined using the given projection_params dictionary, of the given registration. In other
        words, the resulting RegisteredRetinotopyModel will, when given a hemisphere object to
        which to apply the model, will look up the appropriate registration (by the name
        registration_name) make a map projection of using the given projection_params dictionary,
        and apply the model to the resulting coordinates.
        See also neuropythy.freesurfer's Hemisphere.projection_data method for details on the
        projection parameters;
        '''
        self.model = model
        if 'registration' not in projection_params:
            projection_params['registration'] = 'fsaverage_sym'
        if 'chirality' in projection_params:
            chirality = projection_params['chirality']
        elif 'hemi' in projection_params:
            chirality = projection_params['hemi']
        elif 'hemisphere' in projection_params:
            chirality = projection_params['hemisphere']
        else:
            chirality = None
        if chirality is None:
            self.projection_data = nfs.Hemisphere._make_projection(**projection_params)
        else:
            sub = nfs.freesurfer_subject(projection_params['registration'])
            hemi = sub.LH if chirality.upper() == 'LH' else sub.RH
            self.projection_data = hemi.projection_data(**projection_params)
        
    def cortex_to_angle(self, *args):
        '''
        The cortex_to_angle method of the RegisteredRetinotopyModel class is identical to that
        of the RetinotopyModel class, but the method may be given a map, mesh, or hemisphere, in
        which case the result is applied to the coordinates after the appropriate transformation (if
        any) is first applied.
        '''
        if len(args) == 1:
            if isinstance(args[0], ncx.CorticalMesh):
                if args[0].coordinates.shape[0] == 2:
                    X = args[0].coordinates
                    return self.model.cortex_to_angle(X[0], X[1])
                else:
                    m = self.projection_data['forward_function'](args[0])
                    res = np.zeros((3, args[0].coordinates.shape[1]))
                    c2a = np.asarray(self.cortex_to_angle(m.coordinates))
                    res[:, m.vertex_labels] = c2a if len(c2a) == len(res) else c2a.T
                    return res
            elif isinstance(args[0], nfs.Hemisphere):
                regname = self.projection_data['registration']
                if regname is None or regname == 'native':
                    regname = args[0].subject.id
                if regname not in args[0].topology.registrations:
                    raise ValueError('given hemisphere is not registered to ' + regname)
                else:
                    return self.cortex_to_angle(args[0].registration_mesh(regname))
            else:
                X = np.asarray(args[0])
                if len(X.shape) != 2:
                    raise ValueError('given coordinate matrix must be rectangular')
                X = X if X.shape[0] == 2 or X.shape[0] == 3 else X.T
                if X.shape[0] == 2:
                    return self.model.cortex_to_angle(X[0], X[1])
                elif X.shape[0] == 3:
                    Xp = self.projection_data['forward_function'](X)
                    return self.model.cortex_to_angle(Xp[0], Xp[1])
                else:
                    raise ValueError('coordinate matrix must be 2 or 3 dimensional')
        else:
            return self.model.cortex_to_angle(*args)
                    
    def angle_to_cortex(self, *args):
        '''
        The angle_to_cortex method of the RegisteredRetinotopyModel class is identical to that
        of the RetinotopyModel class, but the method may be given a map, mesh, or hemisphere, in
        which case the result is applied to the 'polar_angle' and 'eccentricity' properties.
        '''
        if len(args) == 1:
            if isinstance(args[0], CorticalMesh) or isinstance(args[0], nfs.Hemisphere):
                ang = ncx.retinotopy_data(args[0])
                ecc = ncx.retinotopy_data(args[0])
                return self.model.angle_to_cortex(ang, ecc)
            else:
                tr = np.asarray(args)
                if tr.shape[1] == 2: tr = tr.T
                elif tr.shape[0] != 2: raise ValueError('cannot interpret argument')
                return self.model.angle_to_cortex(tr[0], tr[1])
        else:
            return self.model.angle_to_cortex(*args)
                    

def load_fmm_model(filename, radius=math.pi/3.0, sphere_radius=100.0):
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
        models_path = os.path.join(os.path.dirname(__file__), '..', 'lib', 'models')
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
    with (gzip.open(filename, 'rb') if gz else open(filename, 'r')) as f:
        lines = f.read().split('\n')
    if len(lines) < 3 or lines[0] != 'Flat Mesh Model Version: 1.0':
        raise ValueError('Given file does not contain to a valid flat mesh model!')
    n = int(lines[1].split(':')[1].strip())
    m = int(lines[2].split(':')[1].strip())
    reg = lines[3].split(':')[1].strip()
    hemi = lines[4].split(':')[1].strip().upper()
    center = map(float, lines[5].split(':')[1].strip().split(','))
    onxaxis = map(float, lines[6].split(':')[1].strip().split(','))
    method = lines[7].split(':')[1].strip().lower()
    tx = np.asarray(
        [map(float, row.split(','))
         for row in lines[8].split(':')[1].strip(' \t[]').split(';')])
    crds = []
    for row in lines[9:(n+9)]:
        try:
            (left,right) = row.split(' :: ')
            crds.append(map(float, left.split(',')))
        except:
            print row
            raise
    crds = np.asarray([map(float, left.split(','))
                       for row in lines[9:(n+9)]
                       for (left,right) in [row.split(' :: ')]])
    vals = np.asarray([map(float, right.split(','))
                       for row in lines[9:(n+9)]
                       for (left,right) in [row.split(' :: ')]])
    tris = -1 + np.asarray(
        [map(int, row.split(','))
         for row in lines[(n+9):(n+m+9)]])
    mdl = RegisteredRetinotopyModel(
        RetinotopyMeshModel(tris, crds,
                            90 - 180/math.pi*vals[:,0], vals[:,1], vals[:,2],
                            transform=tx),
        registration=reg,
        center=center,
        center_right=onxaxis,
        method=method,
        radius=radius,
        sphere_radius=sphere_radius,
        chirality=hemi)
    return mdl
