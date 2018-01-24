####################################################################################################
# neuropythy/graphics/core.py
# Core implementation of the neuropythy graphics library for plotting cortical surfaces

import numpy               as np
import numpy.linalg        as npla
import scipy               as sp
import scipy.sparse        as sps
import scipy.spatial       as spspace
import neuropythy.geometry as geo
import pyrsistent          as pyr
import os, sys, types, six, itertools, pimms

from neuropythy.util       import (ObjectWithMetaData, to_affine)
from neuropythy.vision     import (visual_area_names, visual_area_numbers)

if sys.version_info[0] == 3: from   collections import abc as colls
else:                        import collections            as colls


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
    _sigma_cmap = matplotlib.colors.LinearSegmentedColormap(
        'radius',
        {'red':   ((0.0,       0.0, 0.0),
                   (1.25/40.0, 0.5, 0.5),
                   ( 2.5/40.0, 0.0, 0.0),
                   ( 5.0/40.0, 0.0, 0.0),
                   (10.0/40.0, 0.0, 0.0),
                   (20.0/40.0, 1.0, 1.0),
                   (40.0/40.0, 1.0, 1.0)),
         'green': ((0.0,       0.0, 0.0),
                   (1.25/40.0, 0.0, 0.0),
                   ( 2.5/40.0, 0.0, 0.0),
                   ( 5.0/40.0, 1.0, 1.0),
                   (10.0/40.0, 1.0, 1.0),
                   (20.0/40.0, 1.0, 1.0),
                   (40.0/40.0, 1.0, 1.0)),
         'blue':  ((0.0,       0.0, 0.0),
                   (1.25/40.0, 0.5, 0.5),
                   ( 2.5/40.0, 1.0, 1.0),
                   ( 5.0/40.0, 1.0, 1.0),
                   (10.0/40.0, 0.0, 0.0),
                   (20.0/40.0, 0.0, 0.0),
                   (40.0/40.0, 1.0, 1.0))})
    _cmag_cmap = matplotlib.colors.LinearSegmentedColormap(
        'radius',
        {'red':   ((0.0,       0.0, 0.0),
                   ( 2.0/32.0, 1.0, 1.0),
                   ( 8.0/32.0, 1.0, 1.0),
                   (32.0/32.0, 1.0, 1.0)),
         'green': ((0.0,       0.0, 0.0),
                   ( 2.0/32.0, 0.0, 0.0),
                   ( 8.0/32.0, 1.0, 1.0),
                   (32.0/32.0, 1.0, 1.0)),
         'blue':  ((0.0,       0.0, 0.0),
                   ( 2.0/32.0, 0.0, 0.0),
                   ( 8.0/32.0, 0.0, 0.0),
                   (32.0/32.0, 1.0, 1.0))})
    _vertex_angle_empirical_prefixes = ['prf_', 'measured_', 'empiirical_']
    _vertex_angle_model_prefixes = ['model_', 'predicted_', 'inferred_', 'template_', 'atlas_',
                                    'benson14_', 'benson17_']
    _vertex_angle_prefixes = ([''] + _vertex_angle_model_prefixes + _vertex_angle_model_prefixes)
except: pass

def vertex_curvature_color(m):
    return [0.2,0.2,0.2,1.0] if m['curvature'] > -0.025 else [0.7,0.7,0.7,1.0]
def vertex_prop(m, property=None):
    if property is None or property is Ellipsis: return None
    elif pimms.is_vector(property): return property
    elif pimms.is_number(property): return property
    elif not pimms.is_str(property): raise ValueError('argument is not property-like: %s'%property)
    elif property in m: return m[property]
    else: return None
def vertex_weight(m, property=None):
    p = vertex_prop(m, property)
    if p is None:
        return next((m[k]
                     for name in ['weight', 'variance_explained']
                     for pref in ([''] + _vertex_angle_empirical_prefixes)
                     for k in [pref + name]
                     if k in m),
                    1.0)
    return p
def vertex_angle(m, property=None):
    p = vertex_prop(m, property)
    if p is None:
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
    return p
def vertex_eccen(m, property=None):
    p = vertex_prop(m, property)
    if p is None:
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
    return p
def vertex_sigma(m, property=None):
    p = vertex_prop(m, property)
    if p is not None: return p
    return next((m[k]
                 for kk in _vertex_angle_prefixes
                 for nm in ['sigma', 'radius', 'size']
                 for k in [kk + nm]
                 if k in m),
                None)
def vertex_varea(m, property=None):
    p = vertex_prop(m, property)
    if p is not None: return p
    return next((m[k]
                 for kk in _vertex_angle_prefixes
                 for nm in ['visual_area', 'varea', 'label']
                 for k in [kk + nm]
                 if k in m),
                None)
def vertex_angle_color(m, weight_min=0.2, weighted=True, hemi=None, property=Ellipsis,
                       weight=Ellipsis, null_color='curvature'):
    global _angle_cmap_withneg
    if m is Ellipsis:
        kw0 = {'weight_min':weight_min, 'weighted':weighted, 'hemi':hemi,
               'property':property, 'weight':weight, 'null_color':null_color}
        def _pass_fn(x, **kwargs):
            return vertex_angle_color(x, **{k:(kwargs[k] if k in kwargs else kw0[k])
                                            for k in list(set(kwargs.keys() + kw0.keys()))})
        return _pass_fn
    if isinstance(null_color, basestring):
        null_color = null_color.lower()
        if null_color == 'curvature' or null_color == 'curv':
            nullColor = np.asarray(vertex_curvature_color(m))
        else:
            raise ValueError('bad null color: %s' % null_color)
    else:
        nullColor = np.asarray(null_color)
    ang = vertex_angle(m, property)
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
    w = vertex_weight(m, property=weight) if weighted else None
    if weighted and (not pimms.is_number(w) or w < weight_min):
        return nullColor
    angColor = np.asarray(_angle_cmap_withneg((ang + 180.0) / 360.0))
    if weighted:
        return angColor*w + nullColor*(1-w)
    else:
        return angColor
def vertex_eccen_color(m, weight_min=0.1, weighted=True, hemi=None,
                       property=Ellipsis, null_color='curvature', weight=Ellipsis):
    global _eccen_cmap
    if m is Ellipsis:
        kw0 = {'weight_min':weight_min, 'weighted':weighted, 'hemi':hemi,
               'property':property, 'weight':weight, 'null_color':null_color}
        def _pass_fn(x, **kwargs):
            return vertex_eccen_color(x, **{k:(kwargs[k] if k in kwargs else kw0[k])
                                            for k in list(set(kwargs.keys() + kw0.keys()))})
        return _pass_fn
    if isinstance(null_color, basestring):
        null_color = null_color.lower()
        if null_color == 'curvature' or null_color == 'curv':
            nullColor = np.asarray(vertex_curvature_color(m))
        else:
            raise ValueError('bad null color: %s' % null_color)
    else:
        nullColor = np.asarray(null_color)
    ecc = vertex_eccen(m, property)
    if ecc is None: return nullColor
    w = vertex_weight(m, property=weight) if weighted else None
    if weighted and (not pimms.is_number(w) or w < weight_min):
        return nullColor
    eccColor = np.asarray(_eccen_cmap((ecc if 0 < ecc < 90 else 0 if ecc < 0 else 90)/90.0))
    if weighted:
        return eccColor*w + nullColor*(1-w)
    else:
        return eccColor
def vertex_sigma_color(m, weight_min=0.1, weighted=True, hemi=None,
                       property=Ellipsis, null_color='curvature', weight=Ellipsis):
    global _sigma_cmap
    if m is Ellipsis:
        kw0 = {'weight_min':weight_min, 'weighted':weighted, 'hemi':hemi,
               'property':property, 'weight':weight, 'null_color':null_color}
        def _pass_fn(x, **kwargs):
            return vertex_sigma_color(x, **{k:(kwargs[k] if k in kwargs else kw0[k])
                                            for k in list(set(kwargs.keys() + kw0.keys()))})
        return _pass_fn
    if isinstance(null_color, basestring):
        null_color = null_color.lower()
        if null_color == 'curvature' or null_color == 'curv':
            nullColor = np.asarray(vertex_curvature_color(m))
        else:
            raise ValueError('bad null color: %s' % null_color)
    else:
        nullColor = np.asarray(null_color)
    sig = vertex_sigma(m, property)
    if sig is None: return nullColor
    w = vertex_weight(m, property=weight) if weighted else None
    if weighted and (not pimms.is_number(w) or w < weight_min):
        return nullColor
    sigColor = np.asarray(_sigma_cmap((sig if 0 < sig < 40 else 0 if sig < 0 else 40)/40.0))
    if weighted:
        return sigColor*w + nullColor*(1-w)
    else:
        return sigColor
_varea_colors = {'V1': (1,0,0), 'V2': (0,1,0),    'V3': (0,0,1),
                 'hV4':(0,1,1), 'VO1':(0,0.5,1), 'VO2':(0,1,0.5),
                 'LO1':(1,0,1), 'LO2':(0.5,0,1), 'TO1':(1,0,0.5), 'TO2':(0.5,0,0.5),
                 'V3a':(1,1,0), 'V3a':(0.5,1,0)}
def vertex_varea_color(m, property=Ellipsis, null_color='curvature', hemi=None,
                       weight=Ellipsis, weight_min=0.1, weighted=False):
    if m is Ellipsis:
        kw0 = {'weight_min':weight_min, 'weighted':weighted, 'hemi':hemi,
               'property':property, 'weight':weight, 'null_color':null_color}
        def _pass_fn(x, **kwargs):
            return vertex_varea_color(x, **{k:(kwargs[k] if k in kwargs else kw0[k])
                                            for k in list(set(kwargs.keys() + kw0.keys()))})
        return _pass_fn
    if isinstance(null_color, basestring):
        null_color = null_color.lower()
        if null_color == 'curvature' or null_color == 'curv':
            nullColor = np.asarray(vertex_curvature_color(m))
        else:
            raise ValueError('bad null color: %s' % null_color)
    else:
        nullColor = np.asarray(null_color)
    lbl = vertex_varea(m, property)
    if lbl is None or lbl == 0: return nullColor
    w = vertex_weight(m, property=weight) if weighted else None
    if weighted and (not pimms.is_number(w) or w < weight_min):
        return nullColor
    if not pimms.is_str(lbl):
        lbl = None if lbl is None or lbl < 0 or lbl > len(visual_area_names) else \
              visual_area_names[lbl]
    lblColor = np.asarray(_varea_colors.get(lbl, None))
    if lblColor is None: return nullColor
    if weighted: return lblColor*w + nullColor*(1-w)
    else:        return lblColor
def curvature_colors(m):
    '''
    curvature_colors(m) yields an array of curvature colors for the vertices of the given
      property-bearing object m.
    '''
    return np.asarray(m.map(vertex_curvature_color))
def retino_colors(vcolorfn, *args, **kwargs):
    'See eccen_colors, angle_colors, sigma_colors, and varea_colors.'
    if len(args) == 0:
        def _retino_color_pass(*args, **new_kwargs):
            return retino_colors(vcolorfn, *args,
                                 **{k:(new_kwargs[k] if k in new_kwargs else kwargs[k])
                                    for k in set(kwargs.keys() + new_kwargs.keys())})
        return _retino_color_pass
    elif len(args) > 1:
        raise ValueError('retinotopy color functions accepts at most one argument')
    m = args[0]
    # we need to handle the arguments
    if isinstance(m, (geo.VertexSet, pimms.ITable)):
        tbl = m.properties if isinstance(m, geo.VertexSet) else m
        n = tbl.row_count
        # if the weight or property arguments are lists, we need to thread these along
        if 'property' in kwargs:
            props = kwargs['property']
            del kwargs['property']
            if not (pimms.is_vector(props) or pimms.is_matrix(props)):
                props = [props for _ in range(n)]
        else: props = None
        if 'weight' in kwargs:
            ws = kwargs['weight']
            del kwargs['weight']
            if not pimms.is_vector(ws) and not pimms.is_matrix(ws): ws = [ws for _ in range(n)]
        else: ws = None
        vcolorfn0 = vcolorfn(Ellipsis, **kwargs) if len(kwargs) > 0 else vcolorfn
        if props is None and ws is None: vcfn = lambda m,k:vcolorfn0(m)
        elif props is None:              vcfn = lambda m,k:vcolorfn0(m, weight=ws[k])
        elif ws is None:                 vcfn = lambda m,k:vcolorfn0(m, property=props[k])
        else: vcfn = lambda m,k:vcolorfn0(m, property=props[k], weight=ws[k])
        return np.asarray([vcfn(r,kk) for (kk,r) in enumerate(tbl.rows)])
    else:
        return vcolorfn(m, **kwargs)
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
      * property (Ellipsis) specifies the specific property that should be used as the
        eccentricity value; if Ellipsis, will attempt to auto-detect this value.
      * weight (Ellipsis) specifies  the specific property that should be used as the weight value.
      * null_color ('curvature') specifies a color that should be used as the background.
    '''
    return retino_colors(vertex_angle_color, *args, **kwargs)
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
      * property (Ellipsis) specifies the specific property that should be used as the
        eccentricity value; if Ellipsis, will attempt to auto-detect this value.
      * weight (Ellipsis) specifies  the specific property that should be used as the weight value.
      * null_color ('curvature') specifies a color that should be used as the background.
    '''
    return retino_colors(vertex_eccen_color, *args, **kwargs)
def sigma_colors(*args, **kwargs):
    '''
    sigma_colors(obj) yields an array of colors for the pRF-radius map of the given
      property-bearing object (cortex, tesselation, mesh).
    sigma_colors(dict) yields an array of the color for the particular vertex property mapping
      that is given as dict.
    sigma_colors() yields a functor version of sigma_colors that can be called with one of the
      above arguments; note that this is useful precisely because the returned function
      preserves the arguments passed; e.g. sigma_colors(weighted=False)(mesh) is equivalent to
      sigma_colors(mesh, weighted=False).

    The following options are accepted:
      * weighted (True) specifies whether to use weight as opacity.
      * weight_min (0.2) specifies that below this weight value, the curvature (or null color)
        should be plotted.
      * property (Ellipsis) specifies the specific property that should be used as the
        eccentricity value; if Ellipsis, will attempt to auto-detect this value.
      * weight (Ellipsis) specifies  the specific property that should be used as the weight value.
      * null_color ('curvature') specifies a color that should be used as the background.
    '''
    return retino_colors(vertex_sigma_color, *args, **kwargs)
def varea_colors(*args, **kwargs):
    '''
    varea_colors(obj) yields an array of colors for the visual area map of the given
      property-bearing object (cortex, tesselation, mesh).
    varea_colors(dict) yields an array of the color for the particular vertex property mapping
      that is given as dict.
    varea_colors() yields a functor version of varea_colors that can be called with one of the
      above arguments; note that this is useful precisely because the returned function
      preserves the arguments passed; e.g. varea_colors(weighted=False)(mesh) is equivalent to
      varea_colors(mesh, weighted=False).

    The following options are accepted:
      * weighted (True) specifies whether to use weight as opacity.
      * weight_min (0.2) specifies that below this weight value, the curvature (or null color)
        should be plotted.
      * property (Ellipsis) specifies the specific property that should be used as the
        eccentricity value; if Ellipsis, will attempt to auto-detect this value.
      * weight (Ellipsis) specifies  the specific property that should be used as the weight value.
      * null_color ('curvature') specifies a color that should be used as the background.
    '''
    return retino_colors(vertex_varea_color, *args, **kwargs)
def colors_to_cmap(colors):
    colors = np.asarray(colors)
    if len(colors.shape) == 1: return colors_to_cmap([colors])[0]
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

_cortex_colormaps = {'angle': angle_colors,     'polar_angle':  angle_colors,
                     'eccen': eccen_colors,     'eccentricity': eccen_colors,
                     'sigma': sigma_colors,     'radius':       sigma_colors,
                     'varea': varea_colors,     'visual_area':  varea_colors,
                     'curv':  curvature_colors, 'curvature':    curvature_colors}
def cortex_plot(the_map, color=None, axes=None):
    '''
    cortex_plot(map) yields a plot of the given 2D cortical mesh, map.

    The following options are accepted:
      * color (default: None) specifies a function that, when passed a single argument, a dict
        of the properties of a single vertex, yields an RGBA list for that vertex. By default,
        uses the curvature colors.
      * axes (default: None) specifies a particular set of matplotlib pyplot axes that should be
        used. If axes is Ellipsis, then instead of attempting to render the plot, a tuple of
        (tri, zs, cmap) is returned; in this case, tri is a matplotlib.tri.Triangulation
        object for the given map and zs and cmap are an array and colormap (respectively) that
        will produce the correct colors. Without axes equal to Ellipsis, these would instead
        be rendered as axes.tripcolor(tri, zs, cmap, shading='gouraud'). If axes is None, then
        uses the current axes.
    '''
    if axes is None: axes = matplotlib.pyplot.gca()
    tri = matplotlib.tri.Triangulation(the_map.coordinates[0],
                                       the_map.coordinates[1],
                                       triangles=the_map.tess.indexed_faces.T)
    if pimms.is_matrix(color):
        colors = color
    else:
        if color is None:         color = vertex_curvature_color
        elif pimms.is_str(color): color = _cortex_colormaps[color]
        colors = np.asarray(the_map.map(color))
    cmap = colors_to_cmap(colors)
    zs = np.asarray(range(the_map.vertex_count), dtype=np.float) / (the_map.vertex_count - 1)
    if axes is Ellipsis:
        return (tri, zs, cmap)
    else:
        return axes.tripcolor(tri, zs, cmap=cmap, shading='gouraud')
