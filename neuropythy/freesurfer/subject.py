####################################################################################################
# neuropythy/freesurfer/subject.py
# Simple tools for use with FreeSurfer in Python
# By Noah C. Benson

import numpy as np
import numpy.linalg
import scipy as sp
import scipy.spatial as space
import nibabel.freesurfer.io as fsio
from   nibabel.freesurfer.mghformat import load as mghload, MGHImage
import os, math, copy, re
import itertools
from   warnings import warn
import pysistence
from   numbers import (Number, Integral)
from   types import TupleType
from   pysistence import make_dict

from   neuropythy.cortex    import (CorticalMesh)
from   neuropythy.topology  import (Topology, Registration)
from   neuropythy.immutable import (Immutable)
import neuropythy.geometry  as      geo

# These static functions are just handy
def spherical_distance(pt0, pt1):
    '''spherical_distance(a, b) yields the angular distance between points a and b, both of which
       should be expressed in spherical coordinates as (longitude, latitude).'''
    dtheta = pt1[0] - pt0[0]
    dphi   = pt1[1] - pt0[1]
    a = np.sin(dphi/2)**2 + np.cos(pt0[1]) * np.cos(pt1[1]) * np.sin(dtheta/2)**2
    return 2 * np.arcsin(np.sqrt(a))

class PropertyBox(list):
    '''PropertyBox is a simple class for lazily loading properties into a hemisphere or mesh. It
       should generally not be used directly and instead obtained from the Hemisphere class. Note
       that a PropertyBox object should behave like a list.'''
    def __init__(self, loader):
        self._loading_fn = loader
    def __len__(self):
        if 'data' not in self.__dict__: self.__load_data()
        return len(self.data)
    def __getitem__(self, key):
        if 'data' not in self.__dict__: self.__load_data()
        return self.data[key]
    def __iter__(self):
        if 'data' not in self.__dict__: self.__load_data()
        return iter(self.data)
    def __reversed__(self):
        if 'data' not in self.__dict__: self.__load_data()
        return reversed(self.data)
    def __contains__(self, item):
        if 'data' not in self.__dict__: self.__load_data()
        return item in self.data
    def __load_data(self):
        f = self._loading_fn
        dat = f()
        self.__dict__['data'] = dat

       
class Hemisphere(Immutable):
    '''
    The neuropythy.freesurfer.Hemisphere class inherits from neuropythy.Immutable and encapsulates
    the data contained in a subject's Freesurfer hemisphere. This includes the various surface data
    found in the Freesurfer subject's directory as well as certain volume data.
    '''
    

    ################################################################################################
    # Lazy/Static Interface
    # This code handles the lazy interface; only three data are non-lazy and these are coordinates,
    # faces, and options; they may be set directly.

    # The following methods and static variable allow one to set particular members of the object
    # directly:
    @staticmethod
    def _check_property(self, name, val):
        if not isinstance(val, np.ndarray) and not isinstance(val, list):
            raise ValueError('property values must be lists or numpy arrays')
        return True
    @staticmethod
    def _check_properties(self, val):
        if not isinstance(val, dict):
            raise ValueError('properties must be a dictionary')
        for (k, v) in val.iteritems(): Hemisphere._check_property(self, k, v)
        if isinstance(val, pysistence.persistent_dict.PDict):
            return val
        else:
            return make_dict(val)
    @staticmethod
    def _check_options(self, val):
        # Options just have to be a dictionary and are converted to an immutable one
        if not isinstance(val, dict):
            raise ValueError('options must be a dictionary')
        if isinstance(val, pysistence.persistent_dict.PDict):
            return val
        else:
            return make_dict(val)
    __settable_members = {
        'properties': lambda h,v: Hemisphere._check_properties(h,v),
        'options':    lambda h,v: Hemisphere._check_options(h,v)}
    
    # This static variable and these functions explain the dependency hierarchy in cached data
    def _make_surface(self, coords, faces, name, reg=None):
        mesh = CorticalMesh(
            coords,
            faces,
            subject = self.subject,
            hemisphere = self,
            meta_data = self.meta_data.using(
                **{'subject': self.subject,
                   'hemisphere': self,
                   'name': name,
                   'registration': reg}))
        if self.is_persistent(): mesh = mesh.persist()
        return mesh
    def _load_surface_data(self, name):
        path = name
        if not os.path.exists(path):
            path = self.subject.surface_path(name, self.name)
        if not os.path.exists(path):
            return None
        else:
            data = fsio.read_geometry(path)
            data[0].setflags(write=False)
            data[1].setflags(write=False)
            return data
    def _load_surface_data_safe(self, name):
        try:
            return self._load_surface_data(name)
        except:
            return (None, None)
    def _load_surface(self, name, preloaded=None, reg=None):
        data = self._load_surface_data(name) if preloaded is None else preloaded
        return self._make_surface(data[0], data[1], name, reg=reg)
    def _load_sym_surface(self, name):
        path = self.subject.surface_path(name, self.name)
        if not os.path.exists(path):
            return None
        else:
            data = fsio.read_geometry(path)
            data[0].setflags(write=False)
            data[1].setflags(write=False)
            return self._make_surface(data[0], data[1], name)
    def _load_ribbon(self):
        path = self.subject.volume_path('ribbon', self.name)
        if not os.path.exists(path):
            return None
        else:
            data = mghload(path)
            # for now, no post-processing, just loading of the MGHImage
            return data
    def _make_topology(self, faces, sphere, fsave, fssym):
        if sphere is None:
            return None
        if self.subject.id == 'fsaverage_sym':
            names = ['fsaverage_sym']
            surfs = [sphere]
        elif self.subject.id == 'fsaverage':
            names = ['fsaverage']
            surfs = [sphere]
        else:
            names = [self.subject.id, 'fsaverage', 'fsaverage_sym'];
            surfs = [sphere, fsave, fssym]
        regs = {names[i]: surfs[i] for i in range(len(names)) if surfs[i] is not None}
        # get the base path for the subject's surf directory and the prefix
        base = os.path.join(self.subject.directory, 'surf')
        prefix = self.chirality.lower() + '.'
        # find possible additional matches
        for fl in os.listdir(base):
            if len(fl) > 14 and fl.startswith(prefix) and fl.endswith('.sphere.reg'):
                data = self._load_surface_data_safe(fl[3:])
                if data[0] is not None:
                    regs[fl[3:-11]] = data[0]
        # If this is the fsaverage_sym and we didn't find the retinotopy topology, we need to
        # load it out of the data directory
        if self.subject.id == 'fsaverage_sym' or self.subject.id == 'fsaverage':
            path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'lib', 'data', self.subject.id, 'surf')
            if os.path.exists(path):
                regfls = {fl[3:-11]:os.path.join(path, fl) for fl in os.listdir(path)
                          if re.match('^lh\.[^\.]*\.sphere\.reg$', fl) is not None}
                for (regnm,regfl) in regfls.iteritems():
                    if regnm in regs: continue
                    data = self._load_surface_data_safe(regfl)
                    if data is None or data[0] is None:
                        warn('Could not load %s %s registration' % (self.subject.id, regnm))
                    else:
                        regs[regnm] = data[0]
            if 'retinotopy' not in regs:
                retnames = ['retinotopy_' + r
                            for r in ['benson17', 'benson14', 'benson17-uncorrected']]
                retreg = next((r for r in retnames if r in regs), None)
                if retreg is not None:
                    regs['retinotopy'] = regs[retreg]
                elif self.subject.id == 'fsaverage_sym':
                    warn('Could not load fsaverage_sym retinotopy registration!')
        return Topology(faces, regs)
    @staticmethod
    def calculate_edge_data(faces):
        limit = max(faces.flatten()) + 1
        edge2face = {}
        idx = {}
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
        return (np.array(edge_list[0:k]).transpose(), edge2face)
    @staticmethod
    def _check_meta_data(opts):
        md = opts.get('meta_data', {})
        if not isinstance(md, dict):
            raise ValueError('hemisphere meta-data must be a dictionary')
        if type(md) is pysistence.persistent_dict.PDict:
            return md
        else:
            return make_dict(**md)

    # This function will clear the lazily-evaluated members when a given value is changed
    def __update_values(self, name):
        for (sname, (deps, fn)) in Hemisphere.__lazy_members.items():
            if name in deps and sname in self.__dict__:
                del self.__dict__[sname]


        
    __lazy_members = {
        'meta_data':            (('options',), lambda opts: Hemisphere._check_meta_data(opts)),
        'property_names':       (('properties',), lambda props: set(props.keys())),
        'white_surface':        (('_load_surface',), lambda f: f('white')),
        'pial_surface':         (('_load_surface',), lambda f: f('pial')),
        'inflated_surface':     (('_load_surface',), lambda f: f('inflated')),

        'sphere_surface_data':  (('_load_surface_data',),      lambda f: f('sphere')),
        'fs_surface_data':      (('_load_surface_data_safe','subject'),
                                 lambda f,sub: (
                                     f('sphere') if sub.id == 'fsaverage' else 
                                     f('sphere.reg'))),
        'sym_surface_data':     (('_load_surface_data_safe','chirality','subject'), 
                                 lambda f, ch, sub: (
                                     f('sphere')                   if sub.id == 'fsaverage_sym' else
                                     f('fsaverage_sym.sphere.reg') if ch == 'LH'                else
                                     sub.RHX.sym_surface_data      if sub.RHX is not None       else
                                     (None,None))),
        'faces':                (('sphere_surface_data',), lambda dat: dat[1].T),
        'edge_data':            (('faces',), lambda F: Hemisphere.calculate_edge_data(F)),
        'edges':                (('edge_data',), lambda ED: ED[0]),
        'edge_face_index':      (('edge_data',), lambda ED: ED[1]),

        'sphere_surface':       (('_load_surface','subject','sphere_surface_data','topology'), 
                                 lambda f,sub,dat,topo: f(
                                     'sphere',
                                     preloaded=dat,
                                     reg=(topo.registrations[sub.id]
                                          if sub.id in topo.registrations
                                          else None))),
        'fs_sphere_surface':    (('_load_surface','fs_surface_data','topology'),
                                 lambda f,dat,topo: (
                                     None if dat == (None,None) else
                                     f('sphere.reg',
                                       preloaded=dat,
                                       reg=(topo.registrations['fsaverage']
                                            if 'fsaverage' in topo.registrations
                                            else None)))),
        'sym_sphere_surface':   (('_load_surface','sym_surface_data','topology'),
                                 lambda f,dat,topo: (
                                     None if dat is None else
                                     f('fsaverage_sym.sphere.reg',
                                       preloaded=dat,
                                       reg=(topo.registrations['fsaverage_sym']
                                            if 'fsaverage_sym' in topo.registrations
                                            else None)))),

        'vertex_count':         (('sphere_surface_data',), lambda dat: dat[0].shape[0]),
        'midgray_surface':      (('_make_surface', 'white_surface', 'pial_surface'),
                                 lambda f,W,P: f(
                                     0.5*(W.coordinates + P.coordinates),
                                     W.faces,
                                     'midgray')),
        'occipital_pole_index': (('inflated_surface',),
                                 lambda mesh: np.argmin(mesh.coordinates[1])),
        'ribbon':               (('_load_ribbon',), lambda f: f()),
        'chirality':            (('name',), 
                                 lambda name: 'LH' if name == 'LH' or name == 'RHX' else 'RH'),
        'topology':             (('_make_topology', 'sphere_surface_data',
                                  'fs_surface_data', 'sym_surface_data'),
                                 lambda f,sph,fs,sym: f(
                                     sph[1], sph[0],
                                     None if fs  is None or fs[0]  is None else fs[0],
                                     None if sym is None or sym[0] is None else sym[0]))}

    # Make a surface out of coordinates, if desired
    def surface(self, coords, name=None):
        '''
        hemi.surface(coords) yields a new surface with the given coordinate matrix and the topology
        of this hemisphere, hemi. The coordinate matrix must have hemi.vertex_count points.
        '''
        coords = np.asarray(coords)
        coords = (coords   if coords.shape[1] == self.vertex_count else
                  coords.T if coords.shape[0] == self.vertex_count else
                  None)
        if coords is None: raise ValueError('Coordinate matrix was invalid size!')
        return self._make_surface(coords, self.faces, name)
    def registration_mesh(self, name):
        '''
        hemi.registration_mesh(name) yields a CorticalMesh object for the given hemisphere hemi, as
        determined, by the name, which may be a registration name found in
        hemi.topology.registrations.
        Alternately, name may be a registration object, in which case it is used.
        '''
        if isinstance(name, Registration):
            return self.surface(name.coordinates)
        elif not isinstance(name, basestring) or name not in self.topology.registrations:
            raise ValueError('registration not found in topology')
        else:
            return self.surface(self.topology.registrations[name].coordinates, name=name)

    
    ################################################################################################
    # The Constructor
    def __init__(self, subject, hemi_name, **args):
        if not isinstance(subject, Subject):
            raise ValueError('Argument subject must be a FreeSurfer.Subject object')
        if hemi_name.upper() not in set(['RH', 'LH', 'RHX', 'LHX']):
            raise ValueError('Argument hemi_name must be RH, LH, RHX, or LHX')
        Immutable.__init__(self,
                           Hemisphere.__settable_members,
                           {'name':      hemi_name.upper(),
                            'subject':   subject,
                            'directory': os.path.join(subject.directory, 'surf')},
                           Hemisphere.__lazy_members)
        self.properties = args.pop('properties', make_dict())
        self.options = args
        self.__init_properties()
    
    ################################################################################################
    # The display function
    def __repr__(self):
        return "Hemisphere(" + self.name + ", <subject: " + self.subject.id + ">)"

    
    ################################################################################################
    # Property code
    def add_property(self, name, prop=Ellipsis):
        '''hemi.add_property(name, prop) adds (or overwrites) the given property with the given name
           in the given hemi. The name must be a valid dictionary key and the prop argument must be
           a list of numpy array of values, one per vertex.
           hemi.add_property(d) adds all of the properties in the given dictionary of properties, 
           d.
           Note that in either case, if the value of prop or the value in a dictionary item is
           None, then the item is removed from the property list instead of being added.'''
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
                self.properties = self.properties.using(**{name: prop})

    def has_property(self, name):
        '''hemi.has_property(name) yields True if the given hemisphere contains the property with
           the given name and False otherwise.'''
        return name in self.properties

    def remove_property(self, name):
        '''hemi.remove_property(name) removes the property with the given name from the given hemi.
           The name argument may also be an iterable collection of names, in which case all are
           removed.'''
        if hasattr(name, '__iter__'):
            for n in name: self.remove_property(n)
        elif name in self.properties:
            self.__dict__['properties'] = self.properties.without(name)
    
    def property_value(self, name):
        '''hemi.property_value(name) yields a list of the property values with the given name. If
           name is an iterable, then property_value automatically threads across it. If no such
           property is found in the mesh, then None is returned.'''
        if hasattr(name, '__iter__'):
            return map(lambda n: self.property_value(n), name)
        else:
            return self.properties.get(name, None)
            
    def prop(self, name, arg=Ellipsis):
        '''hemi.prop(...) is a generic function for handling mesh properties. It may be called
           with a variety of arguments:
             * hemi.prop(name) is equivalent to hemi.property_value(name) if name is either not
               an iterable or is an iterable that is not a dictionary
             * hemi.prop(d) is equivalent to hemi.add_property(d) if d is a dictionary
             * hemi.prop(name, val) is equivalent to hemi.add_property(name, val) if val is not
               None
             * hemi.prop(name, None) is equivalent to hemi.remove_property(name)'''
        if arg is Ellipsis:
            if isinstance(name, dict):
                self.add_property(name)
            else:
                return self.property_value(name)
        else:
            self.add_property(name, arg)

    def address(self, coords, registration=None, nearest=True):
        '''
        hemi.address(coords) yields the address dictionary of the given coords to their closest
        points in the 3D sphere registration of the given hemi. The optional argument registration
        may be specified to indicate that a different registration should be used; the default
        (None) indicates that the subject's native registration should be used. The optional
        argument nearest (default True) may also be set to False to indicate that the nearest point
        in the mesh should not be looked up
        '''
        coords = np.asarray(coords)
        if len(coords.shape) == 1:
            if coords.shape[0] == 0: return []
            else: return self.address([coords], registration=registration)[:,0]
        transposed = True if coords.shape[0] != 3 else False
        coords = coords.T if transposed else coords
        reg = self.topology.registrations[self.subject.id if registration is None else registration]
        if nearest: coords = reg.nearest(coords)
        return reg.address(coords)

    def unaddress(self, addrs, registration=None):
        '''
        hemi.unaddress(addrs) yields the coordinates of the given address dictionary addrs, looked
        up in the native registration of the given hemisphere hemi. The optional argument
        registration can be provided to indicate that a registration other than the subject's native
        registration should be used; the default (None) uses the native.
        '''
        reg = self.topology.registrations[self.subject.id if registration is None else registration]
        return reg.unaddress(addrs).T
            
    def interpolate(self, from_hemi, property_name, 
                    apply=True, method='automatic', mask=None, null=None, n_jobs=1):
        '''
        hemi.interpolate(from_hemi, prop) yields a list of property values that have been resampled
        onto the given hemisphere hemi from the property with the given name prop of the given 
        from_mesh. If the optional apply is set to True (the default) or to a new property name
        string, the property is added to hemi as well; otherwise it is only returned. Optionally,
        prop may be a list of numpy array of values the same length as the mesh's vertex list; in
        this case, the apply option must be a new property name string, otherwise it is treated as
        False. Note that in order to work, the hemi and from_hemi objects must share a registration
        such as fsaverage or fsaverage_sym.

        Options:
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
        if from_hemi.chirality != self.chirality:
            raise ValueError('hemispheres have opposite chiralities')
        if isinstance(property_name, basestring):
            if not from_hemi.has_property(property_name):
                raise ValueError('given property ' + property_name + ' is not in from_hemi!')
            data = from_hemi.prop(property_name)
        elif type(property_name).__module__ == np.__name__:
            data = property_name
            property_name = apply if isinstance(apply, basestring) else None
        elif hasattr(property_name, '__iter__'):
            if all(isinstance(s, basestring) for s in property_name):
                return {p: self.interpolate(from_hemi, p, 
                                            method=method, mask=mask,
                                            null=null, n_jobs=n_jobs)
                        for p in property_name}
            else:
                data = np.asarray(property_name)
                property_name = apply if isinstance(apply, basestring) else None
        else:
            raise ValueError('property_name is not a string or valid list')
        # pass data along to the topology object...
        result = self.topology.interpolate_from(from_hemi.topology, data,
                                                method=method, mask=mask,
                                                null=null, n_jobs=n_jobs)
        if result is not None:
            if apply is True and property_name is not None:
                self.prop(property_name, result)
            elif isinstance(apply, basestring):
                self.prop(apply, result)
        return result

    def partial_volume_factor(self, distance_cutoff=None, angle_cutoff=2.7):
        '''
        mesh.partial_volume_factor() yields an array of partial voluming risk metric values, one per
        vertex. Each value is a number between 0 and 1 such that a 1 indicates a very high risk for
        partial voluming and a 0 indicates a low risk.
        Partial voluming factors are calculated as follows:
        For each vertex u on the pial cortical surface, the set V of all vertices within d mm of u
        is found; d is chosen to be the mean edge length on the pial surface (if the option
        distance_cutoff is None) or the provided option.
        The partial volume factor for u, f(u) = length({v in V | (N(v) . N(u)) < k}) / length(V); the
        function N(u) indicates the vector normal to the cortical surface at vertex u and k is the
        cosine of the angle_cutoff option, which, by default, is 2.7 rad, or approximately 155 deg.
        '''
        pial = self.pial_surface
        normals = pial.vertex_normals.T
        d = np.mean(pial.edge_lengths) if distance_cutoff is None else distance_cutoff
        k = np.cos(angle_cutoff)
        # get the neighbors
        neis = pial.vertex_spatial_hash.query_ball_point(pial.coordinates.T, r=d)
        # calculate the fraction with large angles:
        return [float(len([1 for v in V if np.dot(u,normals[v]) < k])) / float(len(V))
                for (u,V) in zip(normals, neis)]
        
    
    # This [private] function and this variable set up automatic properties from the FS directory
    # in order to be auto-loaded, a property must appear in this dictionary:
    _auto_properties = {
        'sulc':             ('convexity',              lambda f: fsio.read_morph_data(f)),
        'thickness':        ('thickness',              lambda f: fsio.read_morph_data(f)),
        'area':             ('white_surface_area',     lambda f: fsio.read_morph_data(f)),
        'area.mid':         ('midgray_surface_area',   lambda f: fsio.read_morph_data(f)),
        'area.pial':        ('pial_surface_area',      lambda f: fsio.read_morph_data(f)),
        'curv':             ('curvature',              lambda f: fsio.read_morph_data(f)),

        'prf_eccen':        ('PRF_eccentricity',       lambda f: fsio.read_morph_data(f)),
        'prf_angle':        ('PRF_polar_angle',        lambda f: fsio.read_morph_data(f)),
        'prf_size':         ('PRF_size',               lambda f: fsio.read_morph_data(f)),
        'prf_varex':        ('PRF_variance_explained', lambda f: fsio.read_morph_data(f)),

        'retinotopy_eccen': ('eccentricity',           lambda f: fsio.read_morph_data(f)),
        'retinotopy_angle': ('polar_angle',            lambda f: fsio.read_morph_data(f)),
        'retinotopy_areas': ('visual_area',            lambda f: fsio.read_morph_data(f)),

        'retino_eccen':     ('eccentricity',           lambda f: fsio.read_morph_data(f)),
        'retino_angle':     ('polar_angle',            lambda f: fsio.read_morph_data(f)),
        'retino_areas':     ('visual_area',            lambda f: fsio.read_morph_data(f))}

    # properties grabbed out of MGH or MGZ files
    _mgh_properties = {
        'prf_eccen':         ('PRF_eccentricity',       lambda f: mghload(f).get_data().flatten()),
        'prf_angle':         ('PRF_polar_angle',        lambda f: mghload(f).get_data().flatten()),
        'prf_size':          ('PRF_size',               lambda f: mghload(f).get_data().flatten()),
        'prf_varex':         ('PRF_variance_explained', lambda f: mghload(f).get_data().flatten()),
        'prf_vexpl':         ('PRF_variance_explained', lambda f: mghload(f).get_data().flatten()),

        'retinotopy_eccen':  ('eccentricity',           lambda f: fsio.read_morph_data(f)),
        'retinotopy_angle':  ('polar_angle',            lambda f: fsio.read_morph_data(f)),
        'retinotopy_areas':  ('visual_area',            lambda f: fsio.read_morph_data(f)),
        'retino_eccen':      ('eccentricity',           lambda f: fsio.read_morph_data(f)),
        'retino_angle':      ('polar_angle',            lambda f: fsio.read_morph_data(f)),
        'retino_areas':      ('visual_area',            lambda f: fsio.read_morph_data(f)),

        'predicted_eccen':   ('predicted_eccentricity', lambda f: mghload(f).get_data().flatten()),
        'predicted_angle':   ('predicted_polar_angle',  lambda f: mghload(f).get_data().flatten()),
        'predicted_areas':   ('predicted_visual_area',  lambda f: mghload(f).get_data().flatten()),
        'predicted_v123roi': ('predicted_visual_area',  lambda f: mghload(f).get_data().flatten()),

        'predict_eccen':     ('predicted_eccentricity', lambda f: mghload(f).get_data().flatten()),
        'predict_angle':     ('predicted_polar_angle',  lambda f: mghload(f).get_data().flatten()),
        'predict_areas':     ('predicted_visual_area',  lambda f: mghload(f).get_data().flatten()),
        'predict_v123roi':   ('predicted_visual_area',  lambda f: mghload(f).get_data().flatten()),

        'benson14_eccen':    ('benson14_eccentricity',  lambda f: mghload(f).get_data().flatten()),
        'benson14_angle':    ('benson14_polar_angle',   lambda f: mghload(f).get_data().flatten()),
        'benson14_areas':    ('benson14_visual_area',   lambda f: mghload(f).get_data().flatten()),
        'benson14_v123roi':  ('benson14_visual_area',   lambda f: mghload(f).get_data().flatten()),

        'eccen_predict':     ('predicted_eccentricity', lambda f: mghload(f).get_data().flatten()),
        'angle_predict':     ('predicted_polar_angle',  lambda f: mghload(f).get_data().flatten()),
        'areas_predict':     ('predicted_visual_area',  lambda f: mghload(f).get_data().flatten()),
        'v123roi_predict':   ('predicted_visual_area',  lambda f: mghload(f).get_data().flatten()),

        'eccen_benson14':    ('benson14_eccentricity',  lambda f: mghload(f).get_data().flatten()),
        'angle_benson14':    ('benson14_polar_angle',   lambda f: mghload(f).get_data().flatten()),
        'areas_benson14':    ('benson14_visual_area',   lambda f: mghload(f).get_data().flatten()),
        'v123roi_benson14':  ('benson14_visual_area',   lambda f: mghload(f).get_data().flatten())}
    
    # funciton for initializing the auto-loading properties
    def __init_properties(self):
        # if this is an xhemi, we want to load the opposite of the chirality
        loadchi = 'RH' if self.name == 'RHX' else \
                  'LH' if self.name == 'LHX' else \
                  self.chirality
        dir = self.directory
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        autoprops = Hemisphere._auto_properties
        for file in files:
            if len(file) > 2 and file[0:2].upper() == loadchi and file[3:] in autoprops:
                (name, fn) = autoprops[file[3:]]
                #self.prop(name, PropertyBox(lambda: fn(os.path.join(dir, file))))
                self.prop(name, fn(os.path.join(dir, file)))
        # We also want to auto-add labels:
        dir = os.path.join(self.subject.directory, 'label')
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        for file in files:
            if len(file) > 9 and file[0:2].upper() == loadchi and file[-6:] == '.label':
                if len(file) < 17 or file[-13:-6] != '.thresh':
                    lbl = set(fsio.read_label(os.path.join(dir, file)))
                    self.prop(
                        file[3:-6],
                        [True if k in lbl else False for k in range(self.vertex_count)])
                #else:
                #    (lbl, sclr) = fsio.read_label(os.path.join(dir, file), read_scalars=True)
                #    lbl = {lbl[i]: i for i in range(len(lbl))}
                #    self.prop(
                #        file[3:-13] + '_threshold',
                #        [sclr[lbl[k]] if k in lbl else None for k in range(self.vertex_count)])
        # And MGH/MGZ files in surf:
        dir = os.path.join(self.subject.directory, 'surf')
        files = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]
        for f in files:
            if (f[-4:].lower() == '.mgh' or f[-4:].lower() == '.mgz') and \
                    f[2] == '.' and f[0:2].upper() == loadchi and \
                    f[3:-4] in Hemisphere._mgh_properties:
                (name, fn) = Hemisphere._mgh_properties[f[3:-4]]
                try:
                    pvals = fn(os.path.join(dir, f))
                except:
                    pvals = None
                if pvals is not None:
                    self.prop(name, pvals)

    # This method is a convenient way to get the occipital pole coordinates for the various
    # surfaces in a hemisphere...
    def occipital_pole(self, surfname):
        surfname = surfname.lower()
        if not surfname.lower().endswith('_surface'):
            surfname = surfname + '_surface'
        surf = self.__getattr__(surfname)
        opIdx = self.occipital_pole_index
        return surf.coordinates[:, opIdx]

    # These methods, values, and the projection fn are all related to map projection
    @staticmethod
    def _orthographic_projection(X, params): 
        X = np.asarray(X)
        X = X if X.shape[0] < 4 else X.T
        return X[1:3]
    @staticmethod
    def _orthographic_projection_inverse(X, params): 
        X = np.asarray(X)
        X = X if X.shape[0] < 4 else X.T
        r = params['sphere_radius']
        Xnorm = X / r
        return np.asarray([r * np.sqrt(1.0 - (Xnorm ** 2).sum(0)), X[0], X[1]])
    @staticmethod
    def _equirectangular_projection(X, params):
        X = np.asarray(X)
        X = X if X.shape[0] < 4 else X.T
        X = X / np.sqrt((X ** 2).sum(0))
        r = params['sphere_radius']
        return r / math.pi * np.asarray([np.arctan2(X[1], X[0]), np.arcsin(X[2])])
    @staticmethod
    def _equirectangular_projection_inverse(X, params):
        X = np.asarray(X)
        X = X if X.shape[0] < 4 else X.T
        r = params['sphere_radius']
        X = math.pi / r * X
        cos1 = np.cos(X[1])
        return np.asarray([cos1 * np.cos(X[0]) * r, 
                           cos1 * np.sin(X[0]) * r,
                           np.sin(X[1]) * r])
    @staticmethod
    def _mercator_projection(X, params):
        X = np.asarray(X)
        X = X if X.shape[0] < 4 else X.T
        X = X / np.sqrt((X ** 2).sum(0))
        r = params['sphere_radius']
        return r * np.asarray([np.arctan2(X[1], X[0]),
                               np.log(np.tan(0.25 * math.pi + 0.5 * np.arcsin(X[2])))])
    @staticmethod
    def _mercator_projection_inverse(X, params):
        X = np.asarray(X)
        X = X if X.shape[0] < 4 else X.T
        r = params['sphere_radius']
        X = X / r
        return r * np.asarray([np.cos(X[0]),
                               np.sin(X[0]),
                               np.sin(2 * (np.arctan(np.exp(X[1])) - 0.25*math.pi))])
    @staticmethod
    def _sinusoidal_projection(X, params):
        X = np.asarray(X)
        X = X if X.shape[0] < 4 else X.T
        X = X / np.sqrt((X ** 2).sum(0))
        r = params['sphere_radius']
        phi = np.arcsin(X[2])
        return r / math.pi * np.asarray([np.arctan2(X[1], X[0]) * np.cos(phi), phi])
    @staticmethod
    def _sinusoidal_projection_inverse(X, params):
        X = np.asarray(X)
        X = X if X.shape[0] < 4 else X.T
        r = params['sphere_radius']
        X = math.pi * X / r
        z = np.sin(X[1])
        cosphi = np.cos(X[1])
        return np.asarray([np.cos(X[0] / cosphi) * r,
                           np.sin(X[0] / cosphi) * r,
                           np.sin(X[1]) * r])

    __projection_methods = {
        'orthographic':    lambda X,p: Hemisphere._orthographic_projection(X,p),
        'equirectangular': lambda X,p: Hemisphere._equirectangular_projection(X,p),
        #'mollweide':       lambda X,p: Hemisphere._mollweide_projection(X,p),
        'mercator':        lambda X,p: Hemisphere._mercator_projection(X,p),
        'sinusoidal':      lambda X,p: Hemisphere._sinusoidal_projection(X,p)}
    __projection_methods_inverse = {
        'orthographic':    lambda X,p: Hemisphere._orthographic_projection_inverse(X,p),
        'equirectangular': lambda X,p: Hemisphere._equirectangular_projection_inverse(X,p),
        #'mollweide':       lambda X,p: None,
        'mercator':        lambda X,p: Hemisphere._mercator_projection_inverse(X,p),
        'sinusoidal':      lambda X,p: Hemisphere._sinusoidal_projection_inverse(X,p)}

    @staticmethod
    def _make_projection(params):
        '''
        Create the projection forward and inverse functions, based on the parameters, which should
        be identical to those given in the Hemisphere.projection() and Hemisphere.projection_data()
        funtcions, except that automatic values such as None that get interpreted by the hemisphere
        object should all already be filled out explicitly. Yields a persistent dictionary that is
        identical to the given params argument map, but includes both an inverse and a forward
        function in the entries 'inverse_function' and 'forward_function'.
        '''
        registration = 'native' if 'registration' not in params else params['registration']
        if not isinstance(registration, basestring):
            raise ValueError('registration parameter must be a string')
        center = None if 'center' not in params else params['center']
        if not hasattr(center, '__iter__') or len(center) < 2 or len(center) > 3:
            raise ValueError('center argument must be explicit and must be a 2 or 3-length vector')
        center = np.asarray(center)
        if len(center) == 2:
            # convert to 3D cartesian coordinates from the, presumed, longitude and latitude in
            # angular coordinates (radians)
            center_sc = center
            cos_phi = np.cos(center[1])
            center = np.asarray([cos_phi * np.cos(center[0]),
                                 cos_phi * np.sin(center[0]),
                                 np.sin(center[1])])
        else:
          center_sc = [np.arctan2(center[0], center[1]),
                       np.arcsin(center[2] / np.sqrt(sum(center ** 2)))]
        center_right = [0,1,0] if 'center_right' not in params else params['center_right']
        if not hasattr(center_right, '__iter__') or len(center_right) < 2 or len(center_right) > 3:
            raise ValueError('center_right argument must be explicit and must be a 2 or 3-length' \
                             + ' vector')
        center_right = np.asarray(center_right)
        if len(center_right) == 2:
            center_right_sc = center_right
            cos_phi = np.cos(center_right[1])
            center_right = np.asarray([cos_phi * np.cos(center_right[0]),
                                       cos_phi * np.sin(center_right[0]),
                                       np.sin(center[1])])
        radius = None if 'radius' not in params else params['radius']
        if not (isinstance(radius, Number) or np.issubdtype(type(radius), np.float)):
            raise ValueError('radius option must be explicitly given and must be a number')
        radius = abs(radius)
        sphere_radius = None if 'sphere_radius' not in params else params['sphere_radius']
        if not (isinstance(sphere_radius, Number) or np.issubdtype(type(sphere_radius), np.float)):
            raise ValueError('sphere_radius option must be explicitly given and must be a number')
        sphere_radius = abs(sphere_radius)
        method = None if 'method' not in params else params['method']
        if not isinstance(method, basestring) or method not in Hemisphere.__projection_methods:
            raise ValueError('method parameter not given or not recognized')
        method = method.lower()
        chirality = None if 'chirality' not in params else params['chirality']
        if chirality is not None:
            if not isinstance(chirality, basestring):
                raise ValueError('chirality must be None or a string representing left or right')
            chirality = chirality.upper()
            if chirality == 'L' or chirality == 'R':
                chirality += 'H'
            elif chirality == 'LEFT' or chirality == 'RIGHT':
                chirality = chirality[0] + 'H'
            elif chirality != 'LH' and chirality != 'RH':
                raise ValueError('Chiraliry must be one of LH, RH, L, R, Left, Right, or None')
        params['center'] = center
        params['center_right'] = center_right
        params['center_spherical'] = center_sc
        params['radius'] = radius
        params['sphere_radius'] = sphere_radius
        params['method'] = method
        params['chirality'] = chirality
        params['registration'] = registration
        # Setup Projection Data:
        FR = geo.alignment_matrix_3D(center, [1.0, 0.0, 0.0])
        cr = np.dot(FR, center_right)
        rot_ang = math.pi/2 -  np.arctan2(cr[1], -cr[2])
        FR = np.dot(geo.rotation_matrix_3D([1.0,0.0,0.0], rot_ang), FR)
        IR = numpy.linalg.inv(FR)
        params['forward_affine_transform'] = FR
        params['inverse_affine_transform'] = IR
        fwdprojfn = Hemisphere.__projection_methods[method]
        invprojfn = None if method not in Hemisphere.__projection_methods_inverse else \
                    Hemisphere.__projection_methods_inverse[method]
        def __fwdfn(obj):
            # in this forward-function, we want to interpret what the argument actually is...
            if isinstance(obj, Hemisphere):
                # check to make sure it's the right chirality
                if chirality is not None and obj.chirality != chirality:
                    raise ValueError('Cannot project hemisphere if opposite chirality')
                # See if it has the appropriate registration
                usereg = obj.subject.id if registration == 'native' else registration
                if usereg not in obj.topology.registrations:
                    raise ValueError('Given hemisphere is not registered to the ' + registration \
                                     + ' registration')
                reg = obj.topology.registrations[usereg]
                mesh = obj._make_surface(reg.coordinates, obj.topology.triangles, usereg)
                proj = __fwdfn(mesh)
                proj_params = proj.options['projection_parameters']
                proj.options = proj.options.using(
                    projection_parameters=proj_params.using(hemisphere=obj))
                return proj
            elif isinstance(obj, CorticalMesh):
                sc = obj.spherical_coordinates
                submesh = obj.select(lambda u: spherical_distance(center_sc, sc[0:2,u]) < radius)
                submesh.coordinates = __fwdfn(submesh.coordinates)
                submesh.options = submesh.options.using(
                    projection_parameters=params.using(mesh=obj))
                return submesh
            else:
                obj = np.asarray(obj)
                X = obj if obj.shape[0] == 3 else obj.T
                return fwdprojfn(np.dot(FR, X), params)
        def __invfn(obj):
            if isinstance(obj, CorticalMesh):
                obj_proj_params = obj.options['projection_parameters']
                mesh = obj_proj_params['mesh']
                X = mesh.coordinates.copy()
                X[:, obj.vertex_labels] = __invfn(obj.coordinates)
                X.flags.writeable = False
                return mesh.using(coordinates=X)
            else:
                X = np.asarray(obj)
                X = X if X.shape[0] == 2 else X.T
                return np.dot(IR, invprojfn(X, params))
        params['forward_function'] = __fwdfn
        params['inverse_function'] = __invfn
        params = make_dict(params)
        return params
            
    def __interpret_projection_params(self, params):
        params = params.copy()
        # Figure out which spherical surface we're using...
        regname = None if 'registration' not in params else params['registration'].lower()
        if regname is None or regname == 'native':
            regname = self.subject.id
        elif regname not in self.topology.registrations:
            raise ValueError('Unrecognized registration name: %s' % regname)
        params['registration'] = regname
        coords = self.topology.registrations[regname].coordinates.T
        params['sphere_radius'] = np.mean(np.sqrt(np.sum(coords**2, 0)))
        # important parameters: center and radius => indices
        if 'center' not in params:
            params['center'] = coords[:, self.occipital_pole_index]
        elif isinstance(params['center'], int):
            params['center'] = coords[:, params['center']]
        if 'radius' not in params:
            params['radius'] = math.pi / 3.0
        # also, we need to worry about the method...
        if 'method' not in params:
            params['method'] = 'equirectangular'
        # And the chirality...
        if 'chirality' not in params:
            params['chirality'] = self.chirality
        return params

    def projection_data(self, **params):
        '''
        hemi.projection_data() yields an immutable dictionary of data describing a map projection
        of the given hemisphere hemi.

        The following options are understood; options that are not understood are stored as 
        meta-data and placed in the projection_parameters entry of resulting projected meshes' 
        options member variable:
          * center (default: use the occipital pole) specifies the 3D cartesian or 2D spherical 
            (longitude, latitude, both in radians) vector that is the center of the map projection.
          * radius (default: pi/3) specifies the distance (in radians along the spherical surface)
            within which vertices should be included in the projection.
          * method (default: 'equirectangular') specifies the map projection type; see also
            Hemisphere._make_projection.
          * registration (default: self.subject.id) specifies the registration to which the map
            should be aligned when reprojecting.
          * chirality (default: self.chirality) specifies which chirality the projection should be
            restricted to when reprojecting; if None, then chirality is not restricted.
        '''
        params = self.__interpret_projection_params(params)
        return Hemisphere._make_projection(params)
    
    def projection(self, **params):
        '''
        hemi.projection() yields a map projection that is centered at the occipital pole of the
        hemisphere's spherical topology registration (also hemi.sphere_surface). A variety of
        options that modify the projection may be given; these options are detailed in the
        Hemisphere.projection_data method.
        '''
        proj = self.projection_data(**params)
        return proj['forward_function'](self)

# Setup the subject directory:
_subjects_dirs = []
def subject_paths():
    '''
    subject_paths() yields a list of paths to Freesurfer subject directories in which subjects are
    automatically searched for when identified by subject-name only. These paths are searched in
    the order returned from this function.
    If you must edit these paths, it is recommended to use add_subject_path, but the _subjects_dirs
    variable may be set as well.
    '''
    return copy.copy(_subjects_dirs)

def add_subject_path(path, index=0):
    '''
    add_subject_path(path) will add the given path to the list of subject directories in which to
    search for Freesurfer subjects. The optional argument index may be given to specify the
    precedence of this path when searching for new subject; the default, 0, always inserts the path
    at the front of the list; a value of k indicates that the new list should have the new path at
    index k.
    The path may contain :'s, in which case the individual directories are separated and added.
    If the given path is not a directory or the path could not be inserted, yields False;
    otherwise, yields True. If the string contains a : and multiple paths, then True is yielded
    only if all paths were successfully inserted.
    See also subject_paths.
    '''
    paths = [p for p in path.split(':') if len(p) > 0]
    if len(paths) > 1:
        tests = [add_subject_path(p, index=index) for p in reversed(paths)]
        return all(t for t in tests)
    else:
        if not os.path.isdir(path): return False
        if path in _subjects_dirs:   return True
        try:
            _subjects_dirs.insert(index, path)
            return True
        except:
            return False
# add the SUBJECTS_DIR environment variable...
if 'FREESURFER_HOME' in os.environ:
    add_subject_path(os.path.join(os.environ['FREESURFER_HOME'], 'subjects'))
if 'SUBJECTS_DIR' in os.environ:
    add_subject_path(os.environ['SUBJECTS_DIR'])
    
def find_subject_path(sub):
    '''
    find_subject_path(sub) yields the full path of a Freesurfer subject with the name given by the
    string sub, if such a subject can be found in the Freesurfer search paths. See also
    add_subject_path.
    If no subject is found, then None is returned.
    '''
    looks_like_sub = lambda d: all(os.path.isdir(os.path.join(d, sfx) if sfx is not None else d)
                                   for sfx in [None, 'mri', 'surf', 'label'])
    # if it's a full/relative path already, use it:
    if looks_like_sub(sub): return sub
    for sdir in _subjects_dirs:
        tmp = os.path.join(sdir, sub)
        if looks_like_sub(tmp): return tmp
    return None
    
class Subject(Immutable):
    '''FreeSurfer.Subject objects encapsulate the data contained in a FreeSurfer
       subject directory.'''

    ################################################################################################
    # Lazy/Static Interface
    # This code handles the lazy interface; only three data are non-lazy and these are coordinates,
    # faces, and options; they may be set directly.

    # The following methods and static variable allow one to set particular members of the object
    # directly:
    @staticmethod
    def _check_options(self, val):
        # Options just have to be a dictionary and are converted to an immutable one
        if not isinstance(val, dict):
            raise ValueError('options must be a dictionary')
        if type(val) is pysistence.persistent_dict.PDict:
            return val
        else:
            return make_dict(**val)
    __settable_members = {
        'options': lambda m,v: Subject._check_options(m,v)}

    # This static variable and these functions explain the dependency hierarchy in cached data
    @staticmethod
    def _check_meta_data(opts):
        md = opts.get('meta_data', {})
        if not isinstance(md, dict):
            raise ValueError('subject meta-data must be a dictionary')
        if type(md) is pysistence.persistent_dict.PDict:
            return md
        else:
            return make_dict(**md)
    def _load_hemisphere(self, name):
        hem = Hemisphere(self, name)
        if self.is_persistent(): hem.persist()
        return hem
    def _load_hemisphere_safe(self, name):
        try:
            hem = Hemisphere(self, name)
            if self.is_persistent(): hem.persist()
            return hem
        except:
            return None
    __lazy_members = {
        'meta_data': (('options',), lambda opts: Subject._check_meta_data(opts)),
        'LH':  (('_load_hemisphere',),      lambda f: f('LH')),
        'RH':  (('_load_hemisphere',),      lambda f: f('RH')),
        'LHX': (('_load_hemisphere_safe',), lambda f: f('LHX')),
        'RHX': (('_load_hemisphere_safe',), lambda f: f('RHX'))}
    
    
    ################################################################################################
    # The Constructor
    def __init__(self, subject, **args):
        subpath = find_subject_path(subject)
        if subpath is None: raise ValueError('No valid subject found: %s' % subject)
        (dr, name) = os.path.split(subpath)
        Immutable.__init__(self,
                           Subject.__settable_members,
                           {'subjects_dir': dr,
                            'id': name,
                            'directory': subpath},
                           Subject.__lazy_members)
        self.options = args
    
    ################################################################################################
    # The display function
    def __repr__(self):
        return "Subject(" + self.id + ")"


    ################################################################################################
    # Standard Methods

    def surface_path(self, type, hemi):
        hemi = hemi.upper()
        if hemi == 'RHX':
            if not os.path.isdir(os.path.join(self.directory, 'xhemi', 'surf')):
                raise ValueError('Subject does not have an xhemi/surf directory')
            path = os.path.join(self.directory, 'xhemi', 'surf', 'lh.' + type)
        elif hemi == 'LHX':
            if not os.path.isdir(os.path.join(self.directory, 'xhemi', 'surf')):
                raise ValueError('Subject does not have an xhemi/surf directory')
            path = os.path.join(self.directory, 'xhemi', 'surf', 'rh.' + type)
        elif hemi == 'LH':
            path = os.path.join(self.directory, 'surf', 'lh.' + type)
        elif hemi == 'RH':
            path = os.path.join(self.directory, 'surf', 'rh.' + type)
        else:
            raise ValueError('Argument hemi must be LH, RH, LHX, or RHX')
        if not os.path.isfile(path):
            raise ValueError('file %s not found' % path)
        return path
    def volume_path(self, type, hemi):
        # hemi may be None, in which case we have only these surfaces:
        if hemi is None:
            nm = type + '.'
        else:
            nm = hemi.lower() + '.' + type + '.'
        flnm = os.path.join(self.directory, 'mri', nm + 'mgz')
        if not os.path.isfile(flnm):
            flnm = os.path.join(self.directory, 'mri', nm + 'mgh')
            if not os.path.isfile(flnm):
                raise ValueError('no such file (or mgz equivalent) found: ' + flnm)
        return flnm
            

####################################################################################################
# Some FreeSurfer specific functions

def cortex_to_ribbon_map_smooth(sub, hemi=None, k=12, distance=4, sigma=0.35355):
    '''cortex_to_ribbon_map_smooth(sub) yields a dictionary whose keys are the indices of the ribbon
         voxels for the given FreeSurfer subject sub and whose values are a tuple of both (0) a list
         of vertex labels associated with the given voxel and (1) a list of the weights associated
         with each vertex (both in identical order).

       These associations are determined by finding the k nearest neighbors (of the midgray surface)
       to each voxel then finding the lines from white-to-pial that pass within a certain distance
       of the voxel center and weighting each appropriately.

       The following options are accepted:
         * k (default: 4) is the number of vertices to find near each voxel center
         * distance (default: 3) is the cutoff for including a vertex in a voxel region
         * hemi (default: None) specifies which hemisphere to operate over; this may be 'lh', 'rh',
           or None (to do both)
         * sigma (default: 1 / (2 sqrt(2))) specifies the standard deviation of the Gaussian used
           to weight the vertices contributing to a voxel'''
    
    # we can speed things up slightly by doing left and right hemispheres separately
    if hemi is None:
        return (cortex_to_ribbon_map(sub, k=k, distance=distance, sigma=sigma, hemi='lh'),
                cortex_to_ribbon_map(sub, k=k, distance=distance, sigma=sigma, hemi='rh'))
    if not isinstance(hemi, basestring):
        raise ValueError('hemi must be a string \'lh\', \'rh\', or None')
    if hemi.lower() == 'lh' or hemi.lower() == 'left':
        hemi = sub.LH
    elif hemi.lower() == 'rh' or hemi.lower() == 'right':
        hemi = sub.RH

    # first we find the indices of all voxels that are in the ribbon
    ribdat = hemi.ribbon.get_data()
    idcs = np.transpose(ribdat.nonzero()) + 1  # voxel transforms assume 1-based indexing
    # we also need the transformation from surface to voxel
    #tmtx = hemi.ribbon.header.get_ras2vox()
    tmtx = np.linalg.inv(hemi.ribbon.header.get_vox2ras_tkr())

    # now we want to make an octree from the midgray voxels
    txcoord = lambda mtx: np.dot(tmtx[0:3], np.vstack((mtx, np.ones(mtx.shape[1]))))
    midgray = txcoord(hemi.midgray_surface.coordinates)
    near = space.KDTree(midgray.T)
    
    # then we find the k nearest vertices for each voxel with a distance cutoff
    (d, nei) = near.query(idcs, k=k, distance_upper_bound=distance, p=2)
    ## we want the i'th row of d and nei to be a list of the i'th-nearests to each voxel center
    d = d.T
    nei = nei.T
    
    # okay, now we want to find the closest point along each line segment... we do this iteratively;
    # in the process we replace the d array with the distance of the nearest point along the segment
    # between pial and white for the given vertex. Any inf value will be left alone and will become
    # a zero when run through exp

    ## grab the pial and white matter coordinates...
    pialX = txcoord(hemi.pial_surface.coordinates)
    whiteX = txcoord(hemi.white_surface.coordinates)
    pial2white = whiteX - pialX
    pial2white_norm = np.sqrt(np.sum(pial2white ** 2, 0))
    same_idcs = np.where(pial2white_norm < 1e-9)
    pial2white_norm[same_idcs] = 1
    pial2white[:, same_idcs] = 0
    pial2white_normed = pial2white / pial2white_norm
    ## At this point we also want a transpose of indices
    idcs = idcs.T
    ## Iterate over the neighbors
    for i in range(d.shape[0]):
        # which elements are finite? (which do we care about)
        (finite_vox_idcs,) = np.where(d[i] != np.inf)
        # we do a closest-point search along the segment from white to pial in these locations
        ## to start, we project out the normal
        finite_vtx_idcs = nei[i, finite_vox_idcs]
        finite_vox_inplane = idcs[:, finite_vox_idcs] - pialX[:, finite_vtx_idcs]
        finite_p2w_normed = pial2white_normed[:, finite_vtx_idcs]
        finite_vox_prods = np.sum(finite_vox_inplane * finite_p2w_normed, 0)
        finite_vox_ipnorm = np.sqrt(np.sum(finite_vox_inplane ** 2, 0))
        ## If the product is less than 0, that means the voxel center is outside of the pial surface
        (inval,) = np.where((finite_vox_prods < -0.1 * finite_vox_ipnorm) \
                                + (finite_vox_prods > 1.1 * finite_vox_ipnorm))
        if len(inval) > 0:
            # annotate that these are invalid globally
            d[i, finite_vox_idcs[inval]] = np.inf
            # and locally
            finite_vox_idcs = np.delete(finite_vox_idcs, inval)
            finite_vox_inplane = np.delete(finite_vox_inplane, inval, 1)
            finite_p2w_normed = np.delete(finite_p2w_normed, inval, 1)
            finite_vox_prods = np.delete(finite_vox_prods, inval)
        ## otherwise, find the distance along the plane to the voxel center...
        finite_vox_inplane -= finite_p2w_normed * finite_vox_prods
        ## the distance to the line is just the norm of this then
        mindist2s = np.sum(finite_vox_inplane ** 2, 0)
        # We now have the in-plane distances to the closest vertices; we can just save this
        d[i, finite_vox_idcs] = mindist2s
    ## At this point we've filled the d-values (those not infinite) to the in-plane squared-
    ## distances from the relevant voxel center to the surface segment; we can turn d into weights;
    ## all infinite distances will automatically get a weight of 0
    d = np.exp(-0.5 * d / (sigma * sigma))
    ## we want to divide weights by the total weight so that all weight columns add up to 1...
    wtot = np.sum(d, 0)
    ### we have to check if any of these values are 0 and give them a nearest-vertex assignment
    zero_idcs = np.where(wtot < 1e-9)
    if len(zero_idcs) > 0:
        (d0s, nei0s) = near.query(idcs.T[zero_idcs], k=1, p=2)
        nei[0, zero_idcs] = nei0s
        d[0, zero_idcs] = 1.0
        d[1:, zero_idcs] = 0.0
        wtot[zero_idcs] = 1.0
    ### okay, now divide
    d /= wtot
    
    # With d = weights and nei = indices, we can now build the dictionary
    vlab = np.array(hemi.white_surface.vertex_labels)
    res = {}
    for i in range(idcs.shape[1]):
        (wh,) = np.where(d[:,i] > 0)
        res[tuple(idcs[:,i].tolist())] = (
            vlab[nei[wh, i]].tolist(),
            d[wh, i].tolist())
    return {[i-1 for i in k]: v for (k,v) in res}

def _line_voxel_overlap(vx, xs, xe):
    '''
    _line_voxel_overlap((i,j,k), xs, xe) yields the fraction of the vector from xs to xe that
      overlaps with the voxel (i,j,k); i.e., the voxel that starts at (i,j,k) and ends at
      (i,j,k) + 1.
    '''
    # We want to consider the line segment from smaller to larger coordinate:
    seg = [(x0,x1) if x0 <= x1 else (x1,x0) for (x0,x1) in zip(xs,xe)]
    # First, figure out intersections of start and end values
    isect = [(min(seg_end, vox_end), max(seg_start, vox_start))
             if seg_end >= vox_start and seg_start <= vox_end else None
             for ((seg_start, seg_end), (vox_start, vox_end)) in zip(seg, [(x,x+1) for x in vx])]
    # If any of these are None, we can't possibly overlap
    if None in isect or any(a == b for (a,b) in isect):
        return 0.0
    return np.linalg.norm([e - s for (s,e) in isect]) \
        / np.linalg.norm([e - s for (s,e) in zip(xs,xe)])

def cortex_to_ribbon_map_lines(sub, hemi=None):
    '''
    cortex_to_ribbon_map_lines(sub) yields a dictionary whose keys are the indices of the ribbon 
      voxels for the given FreeSurfer subject sub and whose values are a tuple of both (0) a list of
      vertex labels associated with the given voxel and (1) a list of the weights associated with
      each vertex (both in identical order).

    These associations are determined by projecting the vectors from the white surface vertices to 
      the pial surface vertices into the the ribbon and weighting them by the fraction of the vector
      that lies in the voxel.

    The following options are accepted:
      * hemi (default: None) specifies which hemisphere to operate over; this may be 'lh', 'rh', or
        None (to do both)
    '''
    
    # we can speed things up slightly by doing left and right hemispheres separately
    if hemi is None:
        return (cortex_to_ribbon_map_lines(sub, hemi='lh'),
                cortex_to_ribbon_map_lines(sub, hemi='rh'))
    if not isinstance(hemi, basestring):
        raise ValueError('hemi must be a string \'lh\', \'rh\', or None')
    if hemi.lower() == 'lh' or hemi.lower() == 'left':
        hemi = sub.LH
    elif hemi.lower() == 'rh' or hemi.lower() == 'right':
        hemi = sub.RH
    else:
        raise RuntimeError('Unrecognized hemisphere: ' + hemi)

    # first we find the indices of all voxels that are in the ribbon
    ribdat = hemi.ribbon.get_data()
    # 1-based indexing is assumed:
    idcs = map(tuple, np.transpose(ribdat.nonzero()) + 1)
    # we also need the transformation from surface to voxel
    tmtx = np.linalg.inv(hemi.ribbon.header.get_vox2ras_tkr())
    # given that the voxels assume 1-based indexing, this means that the center of each voxel is at
    # (i,j,k) for integer (i,j,k) > 0; we want the voxels to start and end at integer values with 
    # respect to the vertex positions, so we subtract 1/2 from the vertex positions, which should
    # make the range 0-1, for example, cover the vertices in the first voxel
    txcoord = lambda mtx: np.dot(tmtx[0:3], np.vstack((mtx, np.ones(mtx.shape[1])))) + 1.5

    # Okay; get the transformed coordinates for white and pial surfaces:
    pialX = txcoord(hemi.pial_surface.coordinates)
    whiteX = txcoord(hemi.white_surface.coordinates)
    
    # make a list of voxels through which each vector passes:
    min_idx = [min(i) for i in idcs]
    max_idx = [max(i) for i in idcs]
    vtx_voxels = [((i,j,k), (id, olap))
                  for (id, (xs, xe)) in enumerate(zip(whiteX.T, pialX.T))
                  for i in range(int(math.floor(min(xs[0], xe[0]))),
                                 int(math.ceil(max(xs[0], xe[0]))))
                  for j in range(int(math.floor(min(xs[1], xe[1]))),
                                 int(math.ceil(max(xs[1], xe[1]))))
                  for k in range(int(math.floor(min(xs[2], xe[2]))),
                                 int(math.ceil(max(xs[2], xe[2]))))
                  for olap in [_line_voxel_overlap((i,j,k), xs, xe)]
                  if olap > 0]
    
    # and accumulate these lists... first group by voxel index then sum across these
    first_fn = lambda x: x[0]
    vox_byidx = {
        vox: ([q[0] for q in dat], [q[1] for q in dat])
        for (vox,xdat) in itertools.groupby(sorted(vtx_voxels, key=first_fn), key=first_fn)
        for dat in [[q[1] for q in xdat]]}
    return {
        tuple([i-1 for i in idx]): (ids, np.array(olaps) / np.sum(olaps))
        for idx in idcs
        if idx in vox_byidx
        for (ids, olaps) in [vox_byidx[idx]]}

def cortex_to_ribbon_map(sub, hemi=None, method='lines', options={}):
    '''
    cortex_to_ribbon_map(sub) yields a dictionary whose keys are the indices of the ribbon voxels
      for the given FreeSurfer subject sub and whose values are a tuple of both (0) a list of vertex
      labels associated with the given voxel and (1) a list of the weights associated with each
      vertex (both in identical order).

    The following options may be given:
      * hemi (default: None) may be 'lh', 'rh', or None; if Nonem then the result is a tuple of the
        (LH_dict, RH_dict) where LH_dict and RH_dict are the individual results from calling the
        cortex_to_ribbon_map function with hemi set to 'lh' or 'rh', respectively
      * method (default: 'lines') may be 'lines' or 'smooth'; the lines method projects the line
        segments from the white to the pial into the volume and weights each voxel by the length of
        the line segments contained in them; the smooth method uses Gaussian weights based on
        distance of the nearby vertices to the voxel center; see cortex_to_ribbon_map_lines and
        cortex_to_ribbon_map_smooth for more details.
      * options (default: {}) may be set to a dictionary of options to pass along to the lines or
        smooth functions
    '''
    if method.lower() == 'lines':
        return cortex_to_ribbon_map_lines(sub, hemi=hemi, **options)
    elif method.lower() == 'smooth':
        return cortex_to_ribbon_map_smooth(sub, hemi=hemi, **options)
    else:
        raise RuntimeError('Unrecognized method: ' + str(method))


def _cortex_to_ribbon_map_into_volume_array(vol, m, dat, method):
    dat = np.array(dat)
    if method == 'weighted':
        for k,v in m.iteritems():
            vol[k[0], k[1], k[2]] = np.dot(dat[v[0]], v[1])
    elif method == 'max':
        for k,v in m.iteritems():
            mx = max(range(len(v[0])), key=lambda i: v[1][i])
            vol[k[0], k[1], k[2]] = dat[v[0][mx]]
    else:
        raise ValueError('Unsupported method %s' % method)
    return vol

def cortex_to_ribbon(sub, data, map='lines', hemi=None, method='weighted',
                     fill=0, dtype=None):
    '''
    cortex_to_ribbon(sub, data) projects the given cortical-surface data to the given subject's
    ribbon and returns the resulting MGHImage volume.
    
    The following options may be given:
      * map (default: 'lines') specifies that a particular cortex_to_ribbon mapping should be used;
        such a mapping can be generated by the cortex_to_ribbon_map function. If None, then the
        mapping is generated. May additionally be a string or a (string, dict) tuple; in this case
        the string is taken to be the method parameter to cortex_to_ribbon_map and the dict is
        taken to be the options.
      * hemi (default: None) specifies that only the given hemisphere should be projected; if this
        is None, then both hemispheres are projected.
      * method (default: 'weighted') specifies the method to use when projecting the map into the
        volume. The value 'weighted' indicates that a weighted mean of vertex data should be used to
        fill each voxel, while the value 'max' indicates that the value of the vertex with the max
        weight should be chosen for each voxel. Generally, 'max' should be used for integer,
        category, or label data.
      * fill (default: 0) specifies the value to be assigned to all voxels outside of the ribbon.
      * dtype (default: None) specifies the data type that should be exported. If None, this will be
        automatically set to np.float32 for floating-point data and np.int32 for integer data.
    '''
    hemi = hemi.lower() if isinstance(hemi, basestring) else hemi
    # First, interpret the arguments:
    mtd = None
    if map is None:
        map = cortex_to_ribbon_map(sub, hemi=hemi, method='lines')
    if isinstance(map, TupleType):
        if len(map) < 1 or len(map) > 2: raise ValueError('Invalid map tuple: %s' % map)
        if isinstance(map[0], basestring):
            (mtd, mtdopts) = map if len(map) == 2 else (map[0], {})
            map = cortex_to_ribbon_map(sub, hemi=hemi, method=mtd, options=mtdopts)
    elif isinstance(map, basestring):
        map = cortex_to_ribbon_map(sub, hemi=hemi, method=map)
    else:
        raise ValueError('Could not interpret map option: %s' % map)
    if hemi is None:
        for (x,xnm) in zip([map, data], ['map', 'data']):
            if not isinstance(x, TupleType) or len(x) != 2:
                raise ValueError('%s must match hemi argument' % xnm)
    if not isinstance(map,  TupleType): map  = (map,  None) if hemi == 'lh' else (None, map)
    if not isinstance(data, TupleType): data = (data, None) if hemi == 'lh' else (None, data)
    # Figure out the dtype
    if dtype is None:
        if method == 'max' and all(isinstance(d, Integral)
                                   for dat in data if dat is not None
                                   for d in dat):
            dtype = np.int32
        else:
            dtype = np.float32
    # start with a duplicate of the left ribbon data
    vol0 = sub.LH.ribbon if hemi is None or hemi == 'lh' else sub.RH.ribbon
    vol0dims = vol0.get_data().shape
    arr = fill * np.ones(vol0dims) if fill != 0 else np.zeros(vol0dims)
    # apply the surface data maps to the volume
    if hemi is None or hemi == 'lh':
        arr = _cortex_to_ribbon_map_into_volume_array(arr, map[0], data[0], method=method)
    if hemi is None or hemi == 'rh':
        arr = _cortex_to_ribbon_map_into_volume_array(arr, map[1], data[1], method=method)
    hdr = vol0.header.copy()
    hdr.set_data_dtype(dtype)
    arr = np.asarray(np.round(arr) if np.issubdtype(dtype, np.int) else arr,
                     dtype=dtype)
    return MGHImage(arr, vol0.affine, hdr, vol0.extra, vol0.file_map)
    
    
    
    
    
