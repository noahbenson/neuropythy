####################################################################################################
# neuropythy/cortex/core.py
# Simple tools for dealing with the cortical surface in Python
# By Noah C. Benson

import numpy                 as np
import numpy.linalg          as npla
import scipy                 as sp
import scipy.sparse          as sps
import scipy.optimize        as spopt
import neuropythy.geometry   as geo
import pyrsistent            as pyr
import nibabel               as nib
import nibabel.freesurfer.io as fsio
import pimms

@pimms.immutable
class Cortex(geo.Topology):
    '''
    Cortex is a class that handles a single cortical hemisphere; cortex tracks both the spherical
    registrations that can be used for cross-subject interpolation as well as the various surfaces
    and cortical layers that can be produced from the combined white/pial surfaces.
    Cortex is the go-to class for performing interpolation between subjects as it is a Topology
    object and thus knows how to search for common registrations for interpolation and comparison.
    Cortex also holds the required methods for creating map projections from spherical
    registrations.

    Cortex(chirality, tess, surfaces, registrations) is typically used to initialize a cortex
    object. The chirality should be either 'lh' or 'rh'; tess must be the tesselation of the cortex
    object. The surfaces and registrations arguments should both be (possibly lazy) maps whose
    values are the appropriate mesh objects. The surfaces must include 'white' and 'pial'. If the
    registrations includes the key 'native' this is taken to be the default registration for the
    particular cortex object.
    '''
    def __init__(self, chirality, tess, surfaces, registrations):
        Topology.__init__(self, tess, registrations)
        self.chirality = chirality
        self.surfaces = surfaces

    @pimms.param
    def chirality(ch):
        '''
        cortex.chirality gives the chirality ('lh' or 'rh') for the given cortex.
        '''
        ch = ch.lower()
        if ch != 'lh' and ch != 'rh':
            raise ValueError('chirality must be \'lh\' or \'rh\'')
        return ch
    @pimms.param
    def surfaces(surfs):
        '''
        cortex.surfaces is a mapping of the surfaces of the given cortex; this must include the
        surfaces 'white' and 'pial'.
        '''
        if pimms.is_lazy_map(surfs) or pimms.is_pmap(surfs):
            return surfs
        elif pimms.is_map(surfs):
            return pyr.pmap(surfs)
        else:
            raise ValueError('surfaces must be a mapping object')
    @pimms.require
    def validate_surfaces(surfaces):
        '''
        validate_surfaces requires that the surfaces map contain the keys 'white' and 'pial'.
        '''
        if 'white' in surfaces and 'pial' in surfaces: return True
        else: raise ValueError('surfaces parameter must contain both \'white\' and \'pial\'')

    @pimms.value
    def white_surface(surfaces):
        '''
        cortex.white_surface is the mesh representing the white-matter surface of the given cortex.
        '''
        return surfaces['white']
    @pimms.value
    def pial_surface(surfaces):
        '''
        cortex.pial_surface is the mesh representing the pial surface of the given cortex.
        '''
        return surfaces['pial']
    @pimms.value
    def white_to_pial_vectors(white_surface, pial_surface):
        '''
        cortex.white_to_pial_vectors is a (3 x n) matrix of the unit direction vectors that point
          from the n vertices in the cortex's white surface to their equivalent positions in the
          pial surface.
        '''
        u = pial_surface.coordinates - white_surface.coordinates
        d = np.sqrt(np.sum(u**2, axis=0))
        z = np.isclose(d, 0)
        return pimms.imm_array((~z) / (d + z))
    @pimms.value
    def repr(chirality, tess, vertex_count):
        '''
        cortex.repr is equivalent to repr(cortex).
        '''
        arg = (chirality.upper(), tess.face_count, vertex_count)
        return 'Cortex(<%s>, <%d faces>, <%d vertices>)' % arg

    def __repr__(self):
        return self.repr
    def surface(self, name='white'):
        '''
        cortex.surface() yields the white surface of the given cortex
        cortex.surface(name) yields the surface with the given name (e.g., 'white', 'pial',
          'inflated', 'midgray').
        cortex.surface(fraction) yields the surface that is <fraction> of the distance from white
          to pial (0 is equivalent to 'white'; 1 is equivalent to 'pial'). Layers outside of the
          range 0-1 may be returned by following the vectors between white and pial surfaces, but
          they may have odd appearances, and this should not be confused with surface inflation.
        cortex.surface([dist]) yields the layer that is the given distance from the white surface.
        '''
        if pimms.is_str(name):
            return self.surfaces[name]
        elif pimms.is_vector(name, 'real') and len(name) == 1:
            x0 = self.white_surface
            dx = self.white_to_pial_vectors
            return self.make_mesh(x0 + name[0]*dx)
        elif pimms.is_real(name):
            x0 = self.white_surface
            x1 = self.pial_surface
            return self.make_mesh((1 - name)*x0 + name*x1)
        else:
            raise ValueError('could not understand surface layer: %s' % name)

####################################################################################################
# Below are various graphics-related functions; some are wrapped in a try block in case matplotlib
# is not installed.
try:
    import matplotlib, matplotlib.pyplot, matplotlib.tri
    
    _curv_cmap_dict = {
        name: ((0.0, 0.0, 0.5),
               (0.5, 0.5, 0.2),
               (1.0, 0.2, 0.0))
        for name in ['red', 'green', 'blue']}
    _curv_cmap = matplotlib.colors.LinearSegmentedColormap('curv', _curv_cmap_dict)
    _angle_cmap_withneg = matplotlib.colors.LinearSegmentedColormap(
        'polar_angle_full',
        {'red':   ((0.000, 1.0,   1.0),
                   (0.250, 0.5,   0.5),
                   (0.500, 0.0,   0.0),
                   (0.625, 0.0,   0.0),
                   (0.750, 0.0,   0.0),
                   (0.875, 0.833, 0.833),
                   (1.000, 1.0,   1.0)),
         'green': ((0.000, 0.0,   0.0),
                   (0.250, 0.0,   0.0),
                   (0.500, 0.0,   0.0),
                   (0.625, 0.833, 0.833),
                   (0.750, 0.667, 0.667),
                   (0.875, 0.833, 0.833),
                   (1.000, 0.0,   0.0)),
         'blue':  ((0.000, 0.0,   0.0),
                   (0.250, 0.5,   0.5),
                   (0.500, 1.0,   1.0),
                   (0.625, 0.833, 0.833),
                   (0.750, 0.0,   0.0),
                   (0.875, 0.0,   0.0),
                   (1.000, 0.0,   0.0))})
    _angle_cmap = matplotlib.colors.LinearSegmentedColormap(
        'polar_angle',
        {'red':   ((0.000, 1.0,   1.0),
                   (0.250, 0.5,   0.5),
                   (0.500, 0.0,   0.0),
                   (0.625, 0.0,   0.0),
                   (0.750, 0.0,   0.0),
                   (0.875, 0.833, 0.833),
                   (1.000, 1.0,   1.0)),
         'green': ((0.000, 0.0,   0.0),
                   (0.250, 0.0,   0.0),
                   (0.500, 0.0,   0.0),
                   (0.625, 0.833, 0.833),
                   (0.750, 0.667, 0.667),
                   (0.875, 0.833, 0.833),
                   (1.000, 0.0,   0.0)),
         'blue':  ((0.000, 0.0,   0.0),
                   (0.250, 0.5,   0.5),
                   (0.500, 1.0,   1.0),
                   (0.625, 0.833, 0.833),
                   (0.750, 0.0,   0.0),
                   (0.875, 0.0,   0.0),
                   (1.000, 0.0,   0.0))})
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
    _vertex_angle_empirical_prefixes = ['prf_', 'measured_', 'empiirical_']
    _vertex_angle_model_prefixes = ['model_', 'predicted_', 'inferred_', 'template_', 'atlas_',
                                    'benson14_', 'benson17_']
    _vertex_angle_prefixes = ([''] + _vertex_angle_model_prefixes + _vertex_angle_model_prefixes)
except: pass

def vertex_curvature_color(m):
    return [0.2,0.2,0.2,1.0] if m['curvature'] > -0.025 else [0.7,0.7,0.7,1.0]
def vertex_weight(m):
    return next((m[k]
                 for name in ['weight', 'variance_explained']
                 for pref in ([''] + _vertex_angle_empirical_prefixes)
                 for k in [pref + name]
                 if k in m),
                1.0)
def vertex_angle(m):
    ang0 = next((m[kk] for k in _vertex_angle_prefixes for kk in [k+'polar_angle'] if kk in m),
                None)
    if ang0 is not None: return ang0
    ang0 = next((m[kk]
                 for name in ['angle', 'theta']
                 for k in _vertex_angle_prefixes for kk in [k + name]
                 if kk in m),
                None)
    if ang0 is not None:
        return np.mod(90.0 - 180.0/np.pi*ang0 + 180, 360) - 180
    return None
def vertex_eccen(m):
    ecc0 = next((m[k]
                 for kk in _vertex_angle_prefixes
                 for k in [kk + 'eccentricity']
                 if k in m),
                None)
    if ecc0 is not None: return ecc0
    ecc0 = next((m[k]
                 for kk in _vertex_angle_prefixes
                 for k in [kk + 'rho']
                 if k in m),
                None)
    if ecc0 is not None:
        return 180.0/np.pi*ecc0
    return None
def vertex_angle_color(m, weight_min=0.2, weighted=True, hemi=None, property_name=Ellipsis,
                       null_color='curvature'):
    global _angle_cmap_withneg
    if m is Ellipsis:
        return lambda x: vertex_angle_color(x, weight_min=0.2, weighted=weighted, hemi=hemi,
                                            property_name=property_name, null_color=null_color)
    if isinstance(null_color, basestring):
        null_color = null_color.lower()
        if null_color == 'curvature' or null_color == 'curv':
            nullColor = np.asarray(vertex_curvature_color(m))
        else:
            raise ValueError('bad null color: %s' % null_color)
    else:
        nullColor = np.asarray(null_color)
    if property_name is Ellipsis or property_name is None:
        ang = vertex_angle(m)
    else:
        ang = m[property_name]
    if ang is None: return nullColor
    if isinstance(hemi, basestring):
        hemi = hemi.lower()
        if hemi == 'lh' or hemi == 'left':
            ang = ang
        elif hemi == 'rh' or hemi == 'right':
            ang = -ang
        elif hemi == 'abs':
            ang = np.abs(ang)
        else: raise ValueError('bad hemi argument: %s' % hemi)
    w = vertex_weight(m)
    if weighted and (not pimms.is_number(w) or w < weight_min):
        return nullColor
    angColor = np.asarray(_angle_cmap_withneg((ang + 180.0) / 360.0))
    if weighted:
        return angColor*w + nullColor*(1-w)
    else:
        return angColor
def vertex_eccen_color(m, weight_min=0.2, weighted=True,
                       property_name=Ellipsis, null_color='curvature'):
    global _eccen_cmap
    if m is Ellipsis:
        return lambda x: vertex_eccen_color(x, weight_min=0.2, weighted=weighted,
                                            property_name=property_name, null_color=null_color)
    if isinstance(null_color, basestring):
        null_color = null_color.lower()
        if null_color == 'curvature' or null_color == 'curv':
            nullColor = np.asarray(vertex_curvature_color(m))
        else:
            raise ValueError('bad null color: %s' % null_color)
    else:
        nullColor = np.asarray(null_color)
    if property_name is Ellipsis or property_name is None:
        ecc = vertex_eccen(m)
    else:
        ecc = m[property_name]
    if ecc is None: return nullColor
    w = vertex_weight(m)
    if weighted and (not pimms.is_number(w) or w < weight_min):
        return nullColor
    eccColor = np.asarray(_eccen_cmap((ecc if 0 < ecc < 90 else 0 if ecc < 0 else 90)/90.0))
    if weighted:
        return eccColor*w + nullColor*(1-w)
    else:
        return eccColor
def curvature_colors(m):
    '''
    curvature_colors(m) yields an array of curvature colors for the vertices of the given
      property-bearing object m.
    '''
    return np.asarray(m.map(vertex_curvature_color))
def angle_colors(*args, **kwargs):
    '''
    angle_colors(obj) yields an array of colors for the polar angle map of the given
      property-bearing object (cortex, tesselation, mesh).
    angle_colors(dict) yields an array of the color for the particular vertex property
      mapping that is given as dict.
    angle_colors() yields a functor version of angle_colors that can be called with one of the
      above arguments; note that this is useful precisely because the returned function
      preserves the arguments passed; e.g. angle_colors(weighted=False)(mesh) is equivalent to
      angle_colors(mesh, weighted=False).

    The following options are accepted:
      * weighted (True) specifies whether to use weight as opacity.
      * weight_min (0.2) specifies that below this weight value, the curvature (or null color)
        should be plotted.
      * property_name (Ellipsis) specifies the specific property that should be used as the
        polar angle value; if Ellipsis, will attempt to auto-detect this value.
      * null_color ('curvature') specifies a color that should be used as the background.
    '''
    if len(args) == 0:
        def _angle_color_pass(*args, **new_kwargs):
            return angle_colors(*args, **{k:(new_kwargs[k] if k in new_kwargs else kwargs[k])
                                          for k in set(kwargs.keys() + new_kwargs.keys())})
        return _angle_color_pass
    elif len(args) > 1:
        raise ValueError('angle_colors accepts at most one argument')
    m = args[0]
    if isinstance(m, geo.VertexSet):
        return np.asarray(m.map(vertex_angle_color(**kwargs)))
    else:
        return vertex_angle_color(m, **kwargs)
def eccen_colors(*args, **kwargs):
    '''
    eccen_colors(obj) yields an array of colors for the eccentricity map of the given
      property-bearing object (cortex, tesselation, mesh).
    eccen_colors(dict) yields an array of the color for the particular vertex property mapping
      that is given as dict.
    eccen_colors() yields a functor version of eccen_colors that can be called with one of the
      above arguments; note that this is useful precisely because the returned function
      preserves the arguments passed; e.g. eccen_colors(weighted=False)(mesh) is equivalent to
      eccen_colors(mesh, weighted=False).

    The following options are accepted:
      * weighted (True) specifies whether to use weight as opacity.
      * weight_min (0.2) specifies that below this weight value, the curvature (or null color)
        should be plotted.
      * property_name (Ellipsis) specifies the specific property that should be used as the
        eccentricity value; if Ellipsis, will attempt to auto-detect this value.
      * null_color ('curvature') specifies a color that should be used as the background.
    '''
    if len(args) == 0:
        def _eccen_color_pass(*args, **new_kwargs):
            return eccen_colors(*args, **{k:(new_kwargs[k] if k in new_kwargs else kwargs[k])
                                          for k in set(kwargs.keys() + new_kwargs.keys())})
        return _eccen_color_pass
    elif len(args) > 1:
        raise ValueError('eccen_colors accepts at most one argument')
    m = args[0]
    if isinstance(m, geo.VertexSet):
        return np.asarray(m.map(vertex_eccen_color(**kwargs)))
    else:
        return vertex_eccen_color(m, **kwargs)
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
        the_map = the_map.with_prop(weight=weights)
    if isinstance(color, np.ndarray):
        colors = color
    else:
        if color is None or color == 'curv' or color == 'curvature':
            color = vertex_curvature_color
        elif color == 'angle' or color == 'polar_angle':
            color = vertex_angle_color
        elif color == 'eccen' or color == 'eccentricity':
            color = vertex_eccen_color
        colors = np.asarray(the_map.map(color))
    cmap = colors_to_cmap(colors)
    zs = np.asarray(range(the_map.vertex_count), dtype=np.float) / (the_map.vertex_count - 1)
    if plotter is None:
        return (tri, zs, cmap)
    else:
        return plotter.tripcolor(tri, zs, cmap=cmap, shading='gouraud')
