####################################################################################################
# neuropythy/vision/retinotopy.py
# Tools for registering the cortical surface to a particular potential function
# By Noah C. Benson

import numpy                        as np
import numpy.linalg                 as npla
import nibabel.freesurfer.mghformat as fsmgh
import nibabel.freesurfer.io        as fsio
import pyrsistent                   as pyr
import os, sys, gzip, six, types, pimms

from .. import geometry       as geo
from .. import freesurfer     as nyfs
from .. import mri            as mri
from .. import io             as nyio
from ..util               import (zinv, library_path, is_tuple, is_list)
from ..registration       import (mesh_register, java_potential_term)
from ..java               import (to_java_doubles, to_java_ints)
from functools            import reduce

from .models import (RetinotopyModel, SchiraModel, RetinotopyMeshModel, RegisteredRetinotopyModel,
                     load_fmm_model, visual_area_names, visual_area_numbers)

# Tools for extracting retinotopy data from a subject:
_empirical_retinotopy_names = {
    'polar_angle':  ['prf_polar_angle',       'empirical_polar_angle',  'measured_polar_angle',
                     'training_polar_angle',  'polar_angle'],
    'eccentricity': ['prf_eccentricity',      'empirical_eccentricity', 'measured_eccentricity',
                     'training_eccentricity', 'eccentricity'],
    'radius':       ['pRF_size', 'pRF_radius', 'empirical_size', 'empirical_radius',
                     'measured_size', 'measured_radius', 'size', 'radius',
                     'pRF_sigma', 'empirical_sigma', 'measured_sigma', 'sigma'],
    'weight':       ['prf_weight',       'prf_variance_explained',       'prf_vexpl',
                     'empirical_weight', 'empirical_variance_explained', 'empirical_vexpl',
                     'measured_weight',  'measured_variance_explained',  'measured_vexpl',
                     'training_weight',  'training_variance_explained',  'training_vexpl',
                     'weight',           'variance_explained',           'vexpl']}

# handy function for picking out properties automatically...
def empirical_retinotopy_data(hemi, retino_type):
    '''
    empirical_retinotopy_data(hemi, t) yields a numpy array of data for the given cortex object hemi
    and retinotopy type t; it does this by looking at the properties in hemi and picking out any
    combination that is commonly used to denote empirical retinotopy data. These common names are
    stored in _empirical_retintopy_names, in order of preference, which may be modified.
    The argument t should be one of 'polar_angle', 'eccentricity', 'weight'.
    '''
    dat = _empirical_retinotopy_names[retino_type.lower()]
    hdat = {s.lower(): s for s in six.iterkeys(hemi.properties)}
    return next((hemi.prop(hdat[s.lower()]) for s in dat if s.lower() in hdat), None)

_predicted_retinotopy_names = {
    'polar_angle':  ['predicted_polar_angle',   'model_polar_angle',
                     'registered_polar_angle',  'template_polar_angle'],
    'eccentricity': ['predicted_eccentricity',  'model_eccentricity',
                     'registered_eccentricity', 'template_eccentricity'],
    'visual_area':  ['predicted_visual_area',   'model_visual_area',
                     'registered_visual_area',  'template_visual_area']}

def predicted_retinotopy_data(hemi, retino_type):
    '''
    predicted_retinotopy_data(hemi, t) yields a numpy array of data for the given cortex object hemi
    and retinotopy type t; it does this by looking at the properties in hemi and picking out any
    combination that is commonly used to denote empirical retinotopy data. These common names are
    stored in _predicted_retintopy_names, in order of preference, which may be modified.
    The argument t should be one of 'polar_angle', 'eccentricity', 'visual_area'.
    '''
    dat = _predicted_retinotopy_names[retino_type.lower()]
    hdat = {s.lower(): s for s in six.iterkeys(hemi.properties)}
    return next((hemi.prop(hdat[s]) for s in dat if s.lower() in hdat), None)

_retinotopy_names = {
    'polar_angle':  set(['polar_angle']),
    'eccentricity': set(['eccentricity']),
    'visual_area':  set(['visual_area', 'visual_roi', 'visual_region', 'visual_label']),
    'weight':       set(['weight', 'variance_explained'])}

def basic_retinotopy_data(hemi, retino_type):
    '''
    basic_retinotopy_data(hemi, t) yields a numpy array of data for the given cortex object hemi
    and retinotopy type t; it does this by looking at the properties in hemi and picking out any
    combination that is commonly used to denote empirical retinotopy data. These common names are
    stored in _predicted_retintopy_names, in order of preference, which may be modified.
    The argument t should be one of 'polar_angle', 'eccentricity', 'visual_area', or 'weight'.
    Unlike the related functions empirical_retinotopy_data and predicted_retinotopy_data, this
    function calls both of these (predicted first then empirical) in the case that it does not
    find a valid property.
    '''
    dat = _retinotopy_names[retino_type.lower()]
    val = next((hemi.prop(s) for s in six.iterkeys(hemi.properties) if s.lower() in dat), None)
    if val is None and retino_type.lower() != 'weight':
        val = predicted_retinotopy_data(hemi, retino_type)
    if val is None and retino_type.lower() != 'visual_area':
        val = empirical_retinotopy_data(hemi, retino_type)
    return val

def extract_retinotopy_argument(obj, retino_type, arg, default='any'):
    '''
    extract_retinotopy_argument(o, retino_type, argument) yields retinotopy data of the given
    retinotopy type (e.g., 'polar_angle', 'eccentricity', 'variance_explained', 'visual_area',
    'weight') from the given hemisphere or cortical mesh object o, according to the given
    argument. If the argument is a string, then it is considered a property name and that is
    returned regardless of its value. If the argument is an iterable, then it is returned. If
    the argument is None, then retinotopy will automatically be extracted, if found, by calling
    the retinotopy_data function.
    The option default (which, by default, is 'any') specifies which function should be used to
    extract retinotopy in the case that the argument is None. The value 'any' indicates that the
    function retinotopy_data should be used, while the values 'empirical' and 'predicted' specify
    that the empirical_retinotopy_data and predicted_retinotopy_data functions should be used,
    respectively.
    '''
    if   pimms.is_str(arg):        values = obj.prop(arg)
    elif hasattr(arg, '__iter__'): values = arg
    elif arg is not None:          raise ValueError('cannot interpret retinotopy arg: %s' % arg)
    elif default == 'predicted':   values = predicted_retinotopy_data(obj, retino_type)
    elif default == 'empirical':   values = empirical_retinotopy_data(obj, retino_type)
    elif default == 'any':         values = basic_retinotopy_data(obj, retino_type)
    else:                          raise ValueError('bad default retinotopy: %s' % default)
    if values is None:
        raise RuntimeError('No %s retinotopy data found given argument: %s' % (retino_type, arg))
    n = obj.vertex_count
    values = np.asarray(values)
    if len(values) != n:
        found = False
        # could be that we were given a mesh data-field for a map
        try:              values = values[obj.labels]
        except Exception: values = None
        if values is None:
            raise RuntimeError('%s data: length %s should be %s' % (retino_type, len(values), n))
    return values

_default_polar_angle_units = {
    'polar_angle': 'deg',
    'polar angle': 'deg',
    'angle':       'rad',
    'theta':       'rad',
    'polang':      'deg',
    'ang':         'rad'}
_default_polar_angle_axis = {
    'polar_angle': 'UVM',
    'polar angle': 'UVM',
    'angle':       'RHM',
    'theta':       'RHM',
    'polang':      'UVM',
    'ang':         'RHM'}
_default_polar_angle_dir = {
    'polar_angle': 'cw',
    'polar angle': 'cw',
    'angle':       'ccw',
    'theta':       'ccw',
    'polang':      'cw',
    'ang':         'ccw'}
_default_eccentricity_units = {
    'eccentricity': 'deg',
    'eccen':        'deg',
    'rho':          'rad',
    'ecc':          'deg',
    'radius':       'rad'}
_default_x_units = {
    'x':            'deg',
    'longitude':    'deg',
    'lon':          'deg'}
_default_y_units = {
    'y':            'deg',
    'latitude':     'deg',
    'lat':          'deg'}
_default_z_units = {
    'z':            'rad',
    'complex':      'deg',
    'complex-rad':  'rad',
    'coordinate':   'deg'}
def _clean_angle_deg(polang):
    polang = np.asarray(polang)
    clean = np.mod(polang + 180, 360) - 180
    is180 = np.isclose(polang, -180)
    clean[is180] = np.abs(clean[is180]) * np.sign(polang[is180])
    return clean
def _clean_angle_rad(polang):
    polang = np.asarray(polang)
    clean = np.mod(polang + np.pi, np.pi*2) - np.pi
    return clean
_retinotopy_style_fns = {
    'visual':       lambda t,e: (_clean_angle_deg(90.0 - 180.0/np.pi * t), e),
    'visual-rad':   lambda t,e: (_clean_angle_rad(np.pi/2 - t), e * np.pi/180.0),
    'spherical':    lambda t,e: (_clean_angle_rad(t), e*np.pi/180.0),
    'standard':     lambda t,e: (_clean_angle_rad(t), e),
    'cartesian':    lambda t,e: (np.pi/180.0 * e * np.cos(t), np.pi/180.0 * e * np.sin(t)),
    'geographical': lambda t,e: (e * np.cos(t), e * np.sin(t)),
    'complex':      lambda t,e: e * np.exp(t * 1j),
    'complex-rad':  lambda t,e: np.pi/180.0 * e * np.exp(t * 1j),
    'z':            lambda t,e: np.pi/180.0 * e * np.exp(t * 1j)}

def as_retinotopy(data, output_style='visual', units=Ellipsis, prefix=None, suffix=None):
    '''
    as_retinotopy(data) converts the given data, if possible, into a 2-tuple, (polar_angle, eccen),
      both in degrees, with 0 degrees of polar angle corresponding to the upper vertical meridian
      and negative values corresponding to the left visual hemifield.
    as_retinotopy(data, output_style) yields the given retinotopy data in the given output_style;
      as_retinotopy(data) is equivalent to as_retinotopy(data, 'visual').

    This function is intended as a general conversion routine between various sources of retinotopy
    data. All lookups are done in a case insensitive manner. Data may be specified in any of the
    following ways:
      * A cortical mesh containing recognized properties (such as 'polar_angle' and 'eccentricity'
        or 'latitude' and 'longitude'.
      * A dict with recognized fields.
      * A tuple of (polar_angle, eccentricity) (assumed to be in 'visual' style).
      * A numpy vector of complex numbers (assumed in 'complex' style).
      * An n x 2 or 2 x n matrix whose rows/columns are (polar_angle, eccentricity) values (assumed
        in 'visual' style).

    The following output_styles are accepted:
      * 'visual':       polar-axis:         upper vertical meridian
                        positive-direction: clockwise
                        fields:             ['polar_angle' (degrees), 'eccentricity' (degrees)]
      * 'spherical':    polar-axis:         right horizontal meridian
                        positive-direction: counter-clockwise
                        fields:             ['theta' (radians), 'rho' (radians)]
      * 'standard':     polar-axis:         right horizontal meridian
                        positive-direction: counter-clockwise
                        fields:             ['angle' (radians), 'eccentricity' (degrees)]
      * 'cartesian':    axes:               x/y correspond to RHM/UVM
                        positive-direction: left/up
                        fields:             ('x' (degrees), 'y' (degrees))
      * 'geographical': axes:               x/y correspond to RHM/UVM
                        positive-direction: left/up
                        fields:             ('longitude' (degrees), 'latitude' (degrees))
      * 'complex':      axes:               x/y correspond to RHM/UVM
                        positive-direction: left/up
                        fields:             longitude (degrees) + I*latitude (degrees)
      * 'complex-rad':  axes:               x/y correspond to RHM/UVM
                        positive-direction: left/up
                        fields:             longitude (radians) + I*latitude (radians)
      * 'visual-rad':   polar-axis:         upper vertical meridian
                        positive-direction: clockwise
                        fields:             ['angle' (radians), 'eccentricity' (radians)]

    The following options may be given:
      * units (Ellipsis) specifies the unit that should be assumed (degrees or radians);
        if Ellipsis is given, then auto-detect the unit if possible. This may be a map whose keys
        are 'polar_angle' and 'eccentricity' (or the equivalent titles in data) and whose keys are
        the individual units.
      * prefix (None) specifies a prefix that is required for any keys or property names.
      * suffix (None) specifies a suffix that is required for any keys or property names.
    '''
    # simple sanity check:
    output_style = output_style.lower()
    if output_style not in _retinotopy_style_fns:
        raise ValueError('Unrecognized output style: %s' % output_style)
    # First step: get the retinotopy into a format we can deal with easily
    if is_tuple(data) and len(data) == 2:
        data = {'polar_angle': data[0], 'eccentricity': data[1]}
    if isinstance(data, list):
        data = np.asarray(data)
    if pimms.is_nparray(data):
        if pimms.is_vector(data, np.complexfloating):
            data = {'complex': data}
        else:
            if data.shape[0] != 2: data = data.T
            data = {'polar_angle': data[0], 'eccentricity': data[1]}
    # We now assume that data is a dict type; or is a mesh;
    # figure out the data we have and make it into theta/rho
    if isinstance(data, geo.VertexSet):
        pnames = {k.lower():k for k in six.iterkeys(data.properties)}
        mem_dat = lambda k: k in pnames
        get_dat = lambda k: data.prop(pnames[k])
    else:
        def _make_lambda(data,k): return lambda:data[k]
        data = pimms.lazy_map({k.lower():_make_lambda(data,k) for k in six.iterkeys(data)})
        mem_dat = lambda k: k in data
        get_dat = lambda k: data[k]
    # Check in a particular order:
    suffix = '' if suffix is None else suffix.lower()
    prefix = '' if prefix is None else prefix.lower()
    (angle_key, eccen_key, x_key, y_key, z_key) = [
        next((k for k in aliases if mem_dat(prefix + k + suffix)), None)
        for aliases in [['polar_angle', 'polar angle', 'angle', 'ang', 'polang', 'theta'],
                        ['eccentricity', 'eccen', 'ecc', 'rho'],
                        ['x', 'longitude', 'lon'], ['y', 'latitude', 'lat'],
                        ['z', 'complex', 'complex-rad', 'coordinate']]]
    rad2deg = 180.0 / np.pi
    deg2rad = np.pi / 180.0
    (hpi, dpi) = (np.pi / 2.0, np.pi * 2.0)
    if angle_key and eccen_key:
        akey = prefix + angle_key + suffix
        ekey = prefix + eccen_key + suffix
        theta = np.asarray(get_dat(akey))
        rho   = np.asarray(get_dat(ekey))
        theta = theta * (deg2rad if _default_polar_angle_units[angle_key]  == 'deg' else 1)
        rho   = rho   * (rad2deg if _default_eccentricity_units[eccen_key] == 'rad' else 1)
        if _default_polar_angle_axis[angle_key] == 'UVM': theta = theta - hpi
        if _default_polar_angle_dir[angle_key] == 'cw':   theta = -theta
        ok = np.where(np.isfinite(theta))[0]
        theta[ok[theta[ok] < -np.pi]] += dpi
        theta[ok[theta[ok] >  np.pi]] -= dpi
    elif x_key and y_key:
        (x,y) = [np.asarray(get_dat(prefix + k + suffix)) for k in [x_key, y_key]]
        if _default_x_units[x_key] == 'rad': x *= rad2deg
        if _default_y_units[y_key] == 'rad': y *= rad2deg
        theta = np.arctan2(y, x)
        rho   = np.sqrt(x*x + y*y)
    elif z_key:
        z = get_dat(prefix + z_key + suffix)
        theta = np.angle(z)
        rho   = np.abs(z)
        if _default_z_units[z_key] == 'rad': rho *= rad2deg
    else:
        raise ValueError('could not identify a valid retinotopic representation in data')
    # Now, we just have to convert to the requested output style
    f = _retinotopy_style_fns[output_style]
    return f(theta, rho)

retinotopic_property_aliases = {
    'radius': [
        (set(['radius', 'size', 'sigma', 'rad', 'sz', 'sig',
              'prf_radius', 'prf_size', 'prf_sigma', 'prf_rad', 'prf_sz', 'prf_sig',
              'prfradius', 'prfsize', 'prfsigma', 'prfrad', 'prfsz', 'prfsig']),
         lambda r: r),
        (set(['fwhm']), lambda fwhm: fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0))))],
    'variance_explained': [
        (set(['variance_explained', 'varexp', 'varexpl', 'vexpl', 'weight',
              'coefficient_of_determination', 'cod', 
              'r2', 'rsquared', 'r_squared', 'rsqr', 'rsq']),
         lambda x: x)],
    'visual_area': [
        (set(['visual_area', 'visual_roi', 'visual_label',
              'varea', 'vroi', 'vlabel', 'visarea', 'visroi', 'vislabel', 'vislbl']),
         lambda x: x)]}
def retinotopy_data(m, source='any'):
    '''
    retinotopy_data(m) yields a dict containing a retinotopy dataset with the keys 'polar_angle',
      'eccentricity', and any other related fields for the given retinotopy type; for example,
      'pRF_size' and 'variance_explained' may be included for measured retinotopy datasets and
      'visual_area' may be included for atlas or model datasets. The coordinates are always in the
      'visual' retinotopy style, but can be reinterpreted with as_retinotopy.
    retinotopy_data(m, source) may be used to specify a particular source for the data; this may be
      either 'empirical', 'model', or 'any'; or it may be a prefix/suffix beginning/ending with
      an _ character.
    '''
    if pimms.is_map(source):
        if all(k in source for k in ['polar_angle', 'eccentricity']): return source
    if geo.is_vset(m): return retinotopy_data(m.properties, source=source)
    source = source.lower()
    model_rets = ['predicted', 'model', 'template', 'atlas', 'inferred']
    empir_rets = ['empirical', 'measured', 'prf', 'data']
    wild = False
    extra_fields = {'empirical':('radius','variance_explained'), 'model':('radius','visual_area')}
    check_fields = []
    if source in empir_rets:
        fixes = empir_rets
        check_fields = extra_fields['empirical']
    elif source in model_rets:
        fixes = model_rets
        check_fields = extra_fields['model']
    elif source in ['any', '*', 'all']:
        fixes = model_rets + empir_rets
        check_fields = extra_fields['model'] + extra_fields['empirical']
        wild = True
    elif source in ['none', 'basic']:
        fixes = []
        check_fields = extra_fields['model'] + extra_fields['empirical']
        wild = True
    else: fixes = []
    # first, try all the fixes as prefixes then suffixes
    (z, prefix, suffix) = (None, None, None)
    if wild:
        try: z = as_retinotopy(m, 'visual')
        except Exception: pass
    for fix in fixes:
        if z: break
        try:
            z = as_retinotopy(m, 'visual', prefix=(fix + '_'))
            prefix = fix + '_'
        except Exception: pass
    for fix in fixes:
        if z: break
        try:
            z = as_retinotopy(m, 'visual', suffix=('_' + fix))
            suffix = fix + '_'
        except Exception: pass
    # if none of those worked, try with no prefix/suffix
    if not z:
        try:
            pref = source if source.endswith('_') else (source + '_')
            z = as_retinotopy(m, 'visual', prefix=pref)
            prefix = pref
            check_fields = extra_fields['model'] + extra_fields['empirical']
        except Exception:
            raise
            try:
                suff = source if source.startswith('_') else ('_' + source)
                z = as_retinotopy(m, 'visual', suffix=suff)
                suffix = suff
                check_fields = extra_fields['model'] + extra_fields['empirical']
            except Exception: pass
    # if still not z... we couldn't figure it out
    if not z: raise ValueError('Could not find an interpretation for source %s' % source)
    # okay, we found it; make it into a dict
    res = {'polar_angle': z[0], 'eccentricity': z[1]}
    # check for extra fields if relevant
    pnames = {k.lower():k for k in m} if check_fields else {}
    for fname in set(check_fields):
        for (aliases, trfn) in retinotopic_property_aliases.get(fname, []):
            if trfn is None: trfn = lambda x:x
            for f in aliases:
                if prefix: f = prefix + f
                if suffix: f = f + suffix
                f = f.lower()
                if f in pnames:
                    res[fname] = trfn(m[pnames[f]])
                    trfn = None
                    break
            if trfn is None: break
    # That's it
    return res

pRF_data_Wandell2015 = pyr.pmap(
    {k.lower():pyr.pmap(v)
     for (k,v) in six.iteritems(
             {"V1":  {'m':0.16883, 'b':0.02179}, "V2":  {'m':0.16912, 'b':0.14739},
              "V3":  {'m':0.26397, 'b':0.34221}, "hV4": {'m':0.52963, 'b':0.44501},
              "V3a": {'m':0.35722, 'b':1.00189}, "V3b": {'m':0.35722, 'b':1.00189},
              "VO1": {'m':0.68505, 'b':0.47988}, "VO2": {'m':0.93893, 'b':0.26177},
              "LO1": {'m':0.85645, 'b':0.36149}, "LO2": {'m':0.74762, 'b':0.45887},
              "TO1": {'m':1.37441, 'b':0.17240}, "TO2": {'m':1.65694, 'b':0.00000}})})
pRF_data_Kay2013 = pyr.pmap(
    {k.lower():pyr.pmap({'m':v, 'b':0.5})
     for (k,v) in six.iteritems({'V1':0.16, 'V2':0.18, 'V3':0.25, 'hV4':0.36})})
pRF_data = pyr.pmap({'wandell2015':pRF_data_Wandell2015, 'kay2013':pRF_data_Kay2013})
def predict_pRF_radius(eccentricity, visual_area='V1', source='Wandell2015'):
    '''
    predict_pRF_radius(eccentricity) yields an estimate of the pRF size for a patch of cortex at the
      given eccentricity in V1.
    predict_pRF_radius(eccentricity, area) yields an estimate in the given visual area (may be given
      by the keyword visual_area).
    predict_pRF_radius(eccentricity, area, source) uses the given source to estimate the pRF size
      (may be given by the keyword source).

    The following visual areas can be specified:
      * 'V1' (default), 'V2', 'V3'
      * 'hV4'
      * 'V3a', 'V3b'
      * 'VO1', 'VO2'
      * 'LO1', 'LO2'
      * 'TO1', 'TO2'

    The following sources may be given:
      * 'Wandell2015': Wandell BA, Winawer J (2015) Computational neuroimaging and population
                       receptive fields. Trends Cogn Sci. 19(6):349-57.
                       doi:10.1016/j.tics.2015.03.009.
      * 'Kay2013: Kay KN, Winawer J, Mezer A, Wandell BA (2013) Compressive spatial summation in
                  human visual cortex. J Neurophysiol. 110(2):481-94.
    The default source is 'Wandell2015'.
    '''
    visual_area = visual_area.lower()
    if pimms.is_str(source):
        source = source.lower()
        if source not in pRF_data:
            raise ValueError('Given source (%s) not found in pRF-size database' % source)
        dat = pRF_data[source]
        dat = dat[visual_area]
    else:
        dat = {'m':source[0], 'b':source[1]}
    return dat['m']*eccentricity + dat['b']

def fit_pRF_radius(ctx, retinotopy=Ellipsis, mask=None, weight=Ellipsis, slope_only=False):
    '''
    fit_pRF_radius(ctx) fits a line, m*eccen + b, to the pRF radius and yields the tuple (m, b).

    The following options may be given:
      * retinotopy (default: Ellipsis) specifies the prefix for the retinotopy (passed to
        retinotopy_data() to find the retinotopic dataset).
      * mask (default: None) specifies the mask over which to perform the calculation. This is
        passed to the to_mask() function. In the case that mask is a set or frozenset, then it is
        treated as a conjunction (intersection) of masks.
      * weight (default: None) specifies that a weight should be used; if this is True or Ellipsis,
        will use the variance_explained if it is part of the retinotopy dataset; if this is False or
        None, uses no weight; otherwise, this must be a weight property or property name.
      * slope_only (default: False) may be set to True to instead fit radius = m*eccen and return
        only m.
    '''
    rdat = retinotopy_data(ctx, retinotopy)
    if 'radius' not in rdat: raise ValueError('No pRF radius found in dataset %s' % retinotopy)
    rad = rdat['radius']
    (ang,ecc) = as_retinotopy(rdat, 'visual')
    if isinstance(mask, (set, frozenset)):
        mask = reduce(np.intersect1d, [ctx.mask(m, indices=True) for m in mask])
    else: mask = ctx.mask(mask, indices=True)
    # get a weight if provided:
    if weight in [False, None]: wgt = np.ones(rad.shape)
    elif weight in [True, Ellipsis]:
        if 'variance_explained' in rdat: wgt = rdat['variance_explained']
        else: wgt = np.ones(rad.shape)
    else: wgt = ctx.property(weight)
    # get the relevant eccen and radius values
    (ecc,rad,wgt) = [x[mask] for x in (ecc,rad,wgt)]
    # fit a line...
    if slope_only:
        ecc = np.reshape(ecc * wgt, (len(ecc), 1))
        rad = np.reshape(rad * wgt, (len(rad), 1))
        return np.linalg.lstsq(ecc, rad)[0]
    else:
        return tuple(np.polyfit(ecc, rad, 1, w=wgt))

def _retinotopic_field_sign_triangles(m, retinotopy):
    t = m.tess if isinstance(m, geo.Mesh) or isinstance(m, geo.Topology) else m
    # get the polar angle and eccen data as a complex number in degrees
    if pimms.is_str(retinotopy):
        (x,y) = as_retinotopy(retinotopy_data(m, retinotopy), 'geographical')
    elif retinotopy is Ellipsis:
        (x,y) = as_retinotopy(retinotopy_data(m, 'any'),      'geographical')
    else:
        (x,y) = as_retinotopy(retinotopy,                     'geographical')
    # Okay, now we want to make some coordinates...
    coords = np.asarray([x, y])
    us = coords[:, t.indexed_faces[1]] - coords[:, t.indexed_faces[0]]
    vs = coords[:, t.indexed_faces[2]] - coords[:, t.indexed_faces[0]]
    (us,vs) = [np.concatenate((xs, np.full((1, t.face_count), 0.0))) for xs in [us,vs]]
    xs = np.cross(us, vs, axis=0)[2]
    xs[np.isclose(xs, 0)] = 0
    return np.sign(xs)

def retinotopic_field_sign(m, element='vertices', retinotopy=Ellipsis, invert_field=False):
    '''
    retinotopic_field_sign(mesh) yields a property array of the field sign of every vertex in the 
    mesh m; this value may not be exactly 1 (same as VF) or -1 (mirror-image) but some value
    in-between; this is because the field sign is calculated exactly (1, 0, or -1) for each triangle
    in the mesh then is average onto the vertices. To get only the triangle field signs, use
    retinotopic_field_sign(m, 'triangles').

    The following options are accepted:
      * element ('vertices') may be 'vertices' to specify that the vertex signs should be returned
        or 'triangles' (or 'faces') to specify that the triangle field signs should be returned.
      * retinotopy (Ellipsis) specifies the retinotopic dataset to be used. If se to 'empirical' or
        'predicted', the retinotopy data is auto-detected from the given categories; if set to
        Ellipsis, a property pair like 'polar_angle' and 'eccentricity' or 'lat' and 'lon' are
        searched for using the as_retinotopy function; otherwise, this may be a retinotopy dataset
        recognizable by as_retinotopy.
      * invert_field (False) specifies that the inverse of the field sign should be returned.
    '''
    tsign = _retinotopic_field_sign_triangles(m, retinotopy)
    t = m.tess if isinstance(m, geo.Mesh) or isinstance(m, geo.Topology) else m
    if invert_field: tsign = -tsign
    element = element.lower()
    if element == 'triangles' or element == 'faces': return tsign
    vfs = t.vertex_faces
    vfs = np.asarray([np.mean(tsign[list(ii)]) if len(ii) > 0 else 0 for ii in vfs])
    return vfs

visual_area_field_signs = pyr.pmap({'V1' :-1, 'V2' :1, 'V3' :-1, 'hV4':1,
                                    'VO1':-1, 'VO2':1, 'LO1':1,  'LO2':-1,
                                    'V3b':-1, 'V3a':1, 'TO1':-1, 'TO2':1})
'''
visual_area_field_signs is a persistent map of field signs as observed empirically for visual areas
V1, V2, V3, hV4, VO1, LO1, V3a, and V3b.
'''

# Tools for retinotopy model loading:
_default_schira_model = None
def get_default_schira_model():
    global _default_schira_model
    if _default_schira_model is None:
        #try:
            _default_schira_model = RegisteredRetinotopyModel(
                SchiraModel(),
                geo.MapProjection(
                    registration='fsaverage_sym',
                    chirality='lh',
                    center=[-7.03000, -82.59000, -55.94000],
                    center_right=[58.58000, -61.84000, -52.39000],
                    radius=np.pi/2.5,
                    method='orthographic'))
        #except Exception: raise
    return _default_schira_model

_retinotopy_model_paths = [os.path.join(library_path(), 'models')]
def retinotopy_model(name='benson17', hemi=None,
                     radius=np.pi/2.5, sphere_radius=100.0,
                     search_paths=None, update=False):
    '''
    retinotopy_model() yields a standard retinotopy model of V1, V2, and V3 as well as other areas
    (depending on the options). The model itself is represented as a RegisteredRetinotopyModel
    object, which may internally store a set of meshes with values at the vertices that define the
    polar angle and eccentricity, or as another object (such as with the SchiraModel). The mesh
    models are loaded from files in the neuropythy lib directory. Because the model's class is
    RegisteredRetinotopyModel, so the details of the model's 2D projection onto the cortical surface
    are included in the model.
    
    The following options may be given:
      * name (default: 'benson17') indicates the name of the model to load; the Benson17 model is
        included with the neuropythy library along with various others. If name is a filename, this
        file is loaded (must be a valid fmm or fmm.gz file). Currently, models that are included
        with neuropythy are: Benson17, Benson17-uncorrected, Schira10, and Benson14 (which is
        identical to Schira10, as Schira10 was used by Benson14).
      * hemi (default: None) specifies that the model should go with a particular hemisphere, either
        'lh' or 'rh'. Generally, model files are names lh.<model>.fmm.gz or rh.<model>.fmm.gz, but
        models intended for the fsaverage_sym don't necessarily get a prefix. Note that you can
        leave this as none and just specify that the model name is 'lh.model' instead.
      * radius, sphere_radius (defaults: pi/2.5 and 100.0, respectively) specify the radius of the
        projection (on the surface of the sphere) and the radius of the sphere (100 is the radius
        for Freesurfer spheres). See neuropythy.registration.load_fmm_model for mode details.
      * search_paths (default: None) specifies directories in which to look for fmm model files. No
        matter what is included in these files, the neuropythy library's folders are searched last.
    '''
    origname = name
    tup = (name,hemi,radius,sphere_radius)
    if tup in retinotopy_model.cache:
        return retinotopy_model.cache[tup]
    if os.path.isfile(name):
        fname = name
        name = None
    elif name.lower() in ['schira', 'schira10', 'schira2010', 'benson14', 'benson2014']:
        tmp = get_default_schira_model()
        retinotopy_model.cache[tup] = tmp
        return tmp
    else:
        name = name if hemi is None else ('%s.%s' % (hemi.lower(), name))
        if len(name) > 4 and name[-4:] == '.fmm':
            fname = name
            name = name[:-4]
        elif len(name) > 7 and name[-7:] == '.fmm.gz':
            fname = name
            name = name[:-7]
        else:
            fname = name + '.fmm'
            # Find it in the search paths...
            spaths = ([] if search_paths is None else search_paths) + _retinotopy_model_paths
            fname = next(
                (os.path.join(path, nm0)
                 for path in spaths
                 for nm0 in os.listdir(path)
                 for nm in [nm0[:-4] if len(nm0) > 4 and nm0[-4:] == '.fmm'    else \
                            nm0[:-7] if len(nm0) > 7 and nm0[-7:] == '.fmm.gz' else \
                            None]
                 if nm is not None and nm == name),
                None)
    if fname is None: raise ValueError('Cannot find an FFM file with the name %s' % origname)
    # Okay, load the model...
    mdl = load_fmm_model(fname).persist()
    retinotopy_model.cache[tup] = mdl
    return mdl
retinotopy_model.cache = {}

def occipital_flatmap(cortex, radius=None):
    '''
    occipital_flatmap(cortex) yields a flattened mesh of the occipital cortex of the given cortex
      object.
      
    Note that if the cortex is not registrered to fsaverage, this will fail.

    The option radius may be given to specify the fraction of the cortical sphere (in radians) to
    include in the map.
    '''
    mdl = retinotopy_model('benson17', hemi=cortex.chirality)
    mp = mdl.map_projection
    if radius is not None: mp = mp.copy(radius=radius)
    return mp(cortex)

# Tools for retinotopy registration:
def _retinotopy_vectors_to_float(ang, ecc, wgt, weight_min=0):
    ok = np.isfinite(wgt) & np.isfinite(ecc) & np.isfinite(ang)
    ok[ok] &= wgt[ok] > weight_min
    bad = np.logical_not(ok)
    if np.sum(bad) > 0:
        wgt = np.array(wgt)
        wgt[bad] = 0
    return (ang, ecc, wgt)

def retinotopy_mesh_field(mesh, mdl,
                          polar_angle=None, eccentricity=None, weight=None,
                          weight_min=0, scale=1, sigma=None, shape=2, suffix=None,
                          max_eccentricity=Ellipsis,
                          max_polar_angle=180,
                          angle_type='both',
                          exclusion_threshold=None):
    '''
    retinotopy_mesh_field(mesh, model) yields a list that can be used with mesh_register as a
      potential term. This should generally be used in a similar fashion to retinotopy_anchors.

    Options:
      * polar_angle (default None) specifies that the given data should be used in place of the
        'polar_angle' or 'PRF_polar_angle'  property values. The given argument must be numeric and
        the same length as the the number of vertices in the mesh. If None is given, then the
        property value of the mesh is used; if a list is given and any element is None, then the
        weight for that vertex is treated as a zero. If the option is a string, then the property
        value with the same name isused as the polar_angle data.
      * eccentricity (default None) specifies that the given data should be used in places of the
        'eccentricity' or 'PRF_eccentricity' property values. The eccentricity option is handled 
        virtually identically to the polar_angle option.
      * weight (default None) specifies that the weight or scale of the data; this is handled
        generally like the polar_angle and eccentricity options, but may also be 1, indicating that
        all vertices with polar_angle and eccentricity values defined will be given a weight of 1.
        If weight is left as None, then the function will check for 'weight',
        'variance_explained', 'PRF_variance_explained', and 'retinotopy_weight' values and will use
        the first found (in that order). If none of these is found, then a value of 1 is assumed.
      * weight_min (default 0) specifies that the weight must be higher than the given value inn
        order to be included in the fit; vertices with weights below this value have their weights
        truncated to 0.
      * scale (default 1) specifies a constant by which to multiply all weights for all anchors; the
        value None is interpreted as 1.
      * shape (default 2.0) specifies the exponent in the harmonic function.
      * sigma (default None) specifies, if given as a number, the sigma parameter of the Gaussian
        potential function; if sigma is None, however, the potential function is harmonic.
      * suffix (default None) specifies any additional arguments that should be appended to the 
        potential function description list that is produced by this function; i.e., the 
        retinotopy_anchors function produces a list, and the contents of suffix, if given and not
        None, are appended to that list (see mesh_register).
      * max_eccentricity (default Ellipsis) specifies how the eccentricity portion of the potential
        field should be normalized. Specifically, in order to ensure that polar angle and
        eccentricity contribute roughly equally to the potential, this should be approximately the
        max eccentricity appearing in the data on the mesh. If the argument is the default then the
        actual max eccentricity will be used.
      * max_polar_angle (default: 180) is used the same way as the max_eccentricity function, but if
        Ellipsis is given, the value 180 is always assumed regardless of measured data.
      * exclusion_threshold (default None) specifies that if the initial norm of a vertex's gradient
        is greater than exclusion_threshold * std + median (where std and median are calculated over
        the vertices with non-zero gradients) then its weight is set to 0 and it is not kept as part
        of the potential field.
      * angle_type (default: None) specifies that only one type of angle should be included in the
        mesh; this may be one of 'polar', 'eccen', 'eccentricity', 'angle', or 'polar_angle'. If
        None, then both polar angle and eccentricity are included.

    Example:
     # The retinotopy_anchors function is intended for use with mesh_register, as follows:
     # Define our Schira Model:
     model = neuropythy.retinotopy_model()
     # Make sure our mesh has polar angle, eccentricity, and weight data:
     mesh.prop('polar_angle',  polar_angle_vertex_data);
     mesh.prop('eccentricity', eccentricity_vertex_data);
     mesh.prop('weight',       variance_explained_vertex_data);
     # register the mesh using the retinotopy and model:
     registered_mesh = neuropythy.registration.mesh_register(
        mesh,
        ['mesh', retinotopy_mesh_field(mesh, model)],
        max_step_size=0.05,
        max_steps=2000)
    '''
    #TODO: given a 3D mesh and a registered model, we should be able to return a 3D version of the
    # anchors by unprojecting them
    if pimms.is_str(mdl):
        mdl = retinotopy_model(mdl)
    if not isinstance(mdl, RetinotopyMeshModel):
        if isinstance(mdl, RegisteredRetinotopyModel):
            mdl = mdl.model
        if not isinstance(mdl, RetinotopyMeshModel):
            raise RuntimeError('given model is not a RetinotopyMeshModel instance!')
    if not hasattr(mdl, 'data') or 'polar_angle' not in mdl.data or 'eccentricity' not in mdl.data:
        raise ValueError('Retinotopy model does not have polar angle and eccentricity data')
    if not isinstance(mesh, CorticalMesh):
        raise RuntimeError('given mesh is not a CorticalMesh object!')
    n = mesh.vertex_count
    X = mesh.coordinates.T
    if weight_min is None: weight_min = 0
    # make sure we have our polar angle/eccen/weight values:
    # (weight is odd because it might be a single number, so handle that first)
    (polar_angle, eccentricity, weight) = [
        extract_retinotopy_argument(mesh, name, arg, default='empirical')
        for (name, arg) in [
                ('polar_angle', polar_angle),
                ('eccentricity', eccentricity),
                ('weight', [weight for i in range(n)] \
                           if isinstance(weight, Number) or np.issubdtype(type(weight), np.float) \
                           else weight)]]
    # Make sure they contain no None/invalid values
    (polar_angle, eccentricity, weight) = _retinotopy_vectors_to_float(
        polar_angle, eccentricity, weight,
        weight_min=weight_min)
    if np.sum(weight > 0) == 0:
        raise ValueError('No positive weights found')
    idcs = [i for (i,w) in enumerate(weight) if w > 0]
    # Okay, let's get the model data ready
    mdl_1s = np.ones(mdl.forward.coordinates.shape[0])
    mdl_coords = np.dot(mdl.transform, np.vstack((mdl.forward.coordinates.T, mdl_1s)))[:2].T
    mdl_faces  = mdl.forward.triangles
    mdl_data   = np.asarray([mdl.data['polar_angle'], mdl.data['eccentricity']])
    # Get just the relevant weights and the scale
    wgts = weight[idcs] * (1 if scale is None else scale)
    # and the relevant polar angle/eccen data
    msh_data = np.asarray([polar_angle, eccentricity])[:,idcs]
    # format shape correctly
    shape = np.full((len(idcs)), float(shape), dtype=np.float32)
    # Last thing before constructing the field description: normalize both polar angle and eccen to
    # cover a range of 0-1:
    if max_eccentricity is Ellipsis: max_eccentricity = np.max(msh_data[1])
    if max_polar_angle  is Ellipsis: max_polar_angle  = 180
    if max_polar_angle is not None:
        msh_data[0] /= max_polar_angle
        mdl_data[0] /= max_polar_angle
    if max_eccentricity is not None:
        msh_data[1] /= max_eccentricity
        mdl_data[1] /= max_eccentricity
    # Check if we are making an eccentricity-only or a polar-angle-only field:
    if angle_type is not None:
        angle_type = angle_type.lower()
        if angle_type != 'both' and angle_type != 'all':
            convert = {'eccen':'eccen', 'eccentricity':'eccen', 'radius':'eccen',
                       'angle':'angle', 'polar_angle': 'angle', 'polar': 'angle'}
            angle_type = convert[angle_type]
            mdl_data = [mdl_data[0 if angle_type == 'angle' else 1]]
            msh_data = [msh_data[0 if angle_type == 'angle' else 1]]
    # okay, we've partially parsed the data that was given; now we can construct the final list of
    # instructions:
    if sigma is None:
        field_desc = ['mesh-field', 'harmonic', mdl_coords, mdl_faces, mdl_data, idcs, msh_data,
                      'scale', wgts, 'order', shape]
    else:
        if not hasattr(sigma, '__iter__'): sigma = [sigma for _ in wgts]
        field_desc = ['mesh-field', 'gaussian', mdl_coords, mdl_faces, mdl_data, idcs, msh_data,
                      'scale', wgts, 'order', shape, 'sigma', sigma]
    if suffix is not None: field_desc += suffix
    # now, if we want to exclude outliers, we do so here:
    if exclusion_threshold is not None:
        jpe = java_potential_term(mesh, field_desc)
        jcrds = to_java_doubles(mesh.coordinates)
        jgrad = to_java_doubles(np.zeros(mesh.coordinates.shape))
        jpe.calculate(jcrds,jgrad)
        gnorms = np.sum((np.asarray([[x for x in row] for row in jgrad])[:, idcs])**2, axis=0)
        gnorms_pos = gnorms[gnorms > 0]
        mdn = np.median(gnorms_pos)
        std = np.std(gnorms_pos)
        gn_idcs = np.where(gnorms > mdn + std*3.5)[0]
        for i in gn_idcs: wgts[i] = 0;
    return field_desc

        
def retinotopy_anchors(mesh, mdl,
                       polar_angle=None, eccentricity=None,
                       weight=None, weight_min=0.1,
                       field_sign_weight=0, field_sign=None, invert_field_sign=False,
                       radius_weight=0, radius_weight_source='Wandell2015', radius=None,
                       model_field_sign=None,
                       model_hemi=Ellipsis,
                       scale=1,
                       shape='Gaussian', suffix=None,
                       sigma=[0.1, 2.0, 8.0],
                       select='close'):
    '''
    retinotopy_anchors(mesh, model) is intended for use with the mesh_register function and the
    retinotopy_model() function and/or the RetinotopyModel class; it yields a description of the
    anchor points that tie relevant vertices the given mesh to points predicted by the given model
    object. Any instance of the RetinotopyModel class should work as a model argument; this includes
    SchiraModel objects as well as RetinotopyMeshModel objects such as those returned by the
    retinotopy_model() function. If the model given is a string, then it is passed to the
    retinotopy_model() function first.

    Options:
      * polar_angle (default None) specifies that the given data should be used in place of the
        'polar_angle' or 'PRF_polar_angle'  property values. The given argument must be numeric and
        the same length as the the number of vertices in the mesh. If None is given, then the
        property value of the mesh is used; if a list is given and any element is None, then the
        weight for that vertex is treated as a zero. If the option is a string, then the property
        value with the same name isused as the polar_angle data.
      * eccentricity (default None) specifies that the given data should be used in places of the
        'eccentricity' or 'PRF_eccentricity' property values. The eccentricity option is handled 
        virtually identically to the polar_angle option.
      * weight (default None) specifies that the weight or scale of the data; this is handled
        generally like the polar_angle and eccentricity options, but may also be 1, indicating that
        all vertices with polar_angle and eccentricity values defined will be given a weight of 1.
        If weight is left as None, then the function will check for 'weight',
        'variance_explained', 'PRF_variance_explained', and 'retinotopy_weight' values and will use
        the first found (in that order). If none of these is found, then a value of 1 is assumed.
      * weight_min (default 0) specifies that the weight must be higher than the given value inn
        order to be included in the fit; vertices with weights below this value have their weights
        truncated to 0.
      * scale (default 1) specifies a constant by which to multiply all weights for all anchors; the
        value None is interpreted as 1.
      * shape (default 'Gaussian') specifies the shape of the potential function (see mesh_register)
      * model_hemi (default: None) specifies the hemisphere of the model to load; if None, then
        looks for a non-specific model.
      * suffix (default None) specifies any additional arguments that should be appended to the 
        potential function description list that is produced by this function; i.e., the 
        retinotopy_anchors function produces a list, and the contents of suffix, if given and not
        None, are appended to that list (see mesh_register).
      * select (default 'close') specifies a function that will be called with two arguments for
        every vertex given an anchor; the arguments are the vertex label and the matrix of anchors.
        The function should return a list of anchors to use for the label (None is equivalent to
        lambda id,anc: anc). The parameter may alternately be specified using the string 'close':
        select=['close', [k]] indicates that any anchor more than k times the average edge-length in
        the mesh should be excluded; a value of just ['close', k] on the other hand indicates that
        any anchor more than k distance from the vertex should be exlcuded. The default value,
        'close', is equivalent to ['close', [40]].
      * sigma (default [0.1, 2.0, 4.0]) specifies how the sigma parameter should be handled; if
        None, then no sigma value is specified; if a single number, then all sigma values are
        assigned that value; if a list of three numbers, then the first is the minimum sigma value,
        the second is the fraction of the minimum distance between paired anchor points, and the 
        last is the maximum sigma --- the idea with this form of the argument is that the ideal
        sigma value in many cases is approximately 0.25 to 0.5 times the distance between anchors
        to which a single vertex is attracted; for any anchor a to which a vertex u is attracted,
        the sigma of a is the middle sigma-argument value times the minimum distance from a to all
        other anchors to which u is attracted (clipped by the min and max sigma).
      * field_sign_weight (default: 0) specifies the amount of weight that should be put on the
        retinotopic field of the model as a method of attenuating the weights on those anchors whose
        empirical retinotopic values and predicted model locations do not match. The weight that
        results is calculated from the difference in empirical field-sign for each vertex and the
        visual area field sign based on the labels in the model. The higher the field-sign weight,
        (approaching 1) the more the resulting value is a geometric mean of the field-sign-based
        weight and the original weights. As this value approaches 0, the resulting weights are more
        like the original weights.
      * radius_weight (default: 0) specifies the amount of weight that should be put on the
        receptive field radius of the model as a method of attenuating the weights on those anchors
        whose empirical retinotopic values and predicted model locations do not match. The weight
        that results is calculated from the difference in empirical RF radius for each vertex and
        the predicted RF radius based on the labels in the model. The higher the radius weight,
        (approaching 1) the more the resulting value is a geometric mean of the field-sign-based
        weight and the original weights. As this value approaches 0, the resulting weights are more
        like the original weights.
      * radius_weight_source (default: 'Wandell2015') specifies the source for predicting RF radius;
        based on eccentricity and visual area label.

    Example:
     # The retinotopy_anchors function is intended for use with mesh_register, as follows:
     # Define our Schira Model:
     model = neuropythy.registration.SchiraModel()
     # Make sure our mesh has polar angle, eccentricity, and weight data:
     mesh.prop('polar_angle',  polar_angle_vertex_data);
     mesh.prop('eccentricity', eccentricity_vertex_data);
     mesh.prop('weight',       variance_explained_vertex_data);
     # register the mesh using the retinotopy and model:
     registered_mesh = neuropythy.registration.mesh_register(
        mesh,
        ['mesh', retinotopy_anchors(mesh, model)],
        max_step_size=0.05,
        max_steps=2000)
    '''
    if pimms.is_str(mdl):
        hemi = None
        if pimms.is_str(model_hemi):
            model_hemi = model_hemi.upper()
            hemnames = {k:h
                        for (h,als) in [('LH', ['LH','L','LEFT','RHX','RX']),
                                        ('RH', ['RH','R','RIGHT','LHX','LX'])]
                        for k in als}
            if model_hemi in hemnames: hemi = hemnames[model_hemi]
            else: raise ValueError('Unrecognized hemisphere name: %s' % model_hemi)
        elif model_hemi is not None:
            raise ValueError('model_hemi must be a string, Ellipsis, or None')
        mdl = retinotopy_model(mdl, hemi=hemi)
    if not isinstance(mdl, RetinotopyModel):
        raise RuntimeError('given model is not a RetinotopyModel instance!')
    if not isinstance(mesh, geo.Mesh):
        raise RuntimeError('given mesh is not a Mesh object!')
    n = mesh.vertex_count
    X = mesh.coordinates.T
    if weight_min is None: weight_min = 0
    # make sure we have our polar angle/eccen/weight values:
    # (weight is odd because it might be a single number, so handle that first)
    (polar_angle, eccentricity, weight) = [
        extract_retinotopy_argument(mesh, name, arg, default='empirical')
        for (name, arg) in [
                ('polar_angle', polar_angle),
                ('eccentricity', eccentricity),
                ('weight', np.full(n, weight) if pimms.is_number(weight) else weight)]]
    # Make sure they contain no None/invalid values
    (polar_angle, eccentricity, weight) = _retinotopy_vectors_to_float(
        polar_angle, eccentricity, weight,
        weight_min=weight_min)
    if np.sum(weight > 0) == 0:
        raise ValueError('No positive weights found')
    idcs = np.where(weight > 0)[0]
    # Interpret the select arg if necessary (but don't apply it yet)
    select = ['close', [40]] if select == 'close'   else \
             ['close', [40]] if select == ['close'] else \
             select
    if select is None:
        select = lambda a,b: b
    elif ((pimms.is_vector(select) or is_list(select) or is_tuple(select))
          and len(select) == 2 and select[0] == 'close'):
        if pimms.is_vector(select[1]): d = np.mean(mesh.edge_lengths) * select[1][0]
        else:                          d = select[1]
        select = lambda idx,ancs: [a for a in ancs if a[0] is not None if npla.norm(X[idx] - a) < d]
    # Okay, apply the model:
    res = mdl.angle_to_cortex(polar_angle[idcs], eccentricity[idcs])
    oks = np.isfinite(np.sum(np.reshape(res, (res.shape[0], -1)), axis=1))
    # Organize the data; trim out those not selected
    data = [[[i for _ in r], r, [ksidx[tuple(a)] for a in r]]
            for (i,r0,ok) in zip(idcs, res, oks) if ok
            for ksidx in [{tuple(a):(k+1) for (k,a) in enumerate(r0)}]
            for r in [select(i, r0)]
            if len(r) > 0]
    # Flatten out the data into arguments for Java
    idcs = [int(i) for d in data for i in d[0]]
    ancs = np.asarray([pt for d in data for pt in d[1]]).T
    labs = np.asarray([ii for d in data for ii in d[2]]).T
    # Get just the relevant weights and the scale
    wgts = np.asarray(weight[idcs] * (1 if scale is None else scale))
    # add in the field-sign weights and radius weights if requested here;
    if not np.isclose(field_sign_weight, 0) and mdl.area_name_to_id is not None:
        id2n = mdl.area_id_to_name
        if field_sign is True or field_sign is Ellipsis or field_sign is None:
            from .cmag import cmag
            r = {'polar_angle': polar_angle,  'eccentricity': eccentricity}
            #field_sign = retinotopic_field_sign(mesh, retinotopy=r)
            field_sign = cmag(mesh, r)['field_sign']
        elif pimms.is_str(field_sign): field_sign = mesh.prop(field_sign)
        field_sign = np.asarray(field_sign)
        if invert_field_sign: field_sign = -field_sign
        fswgts = 1.0 - 0.25 * np.asarray(
            [(fs - visual_area_field_signs[id2n[l]]) if l in id2n else 0
             for (l,fs) in zip(labs,field_sign[idcs])])**2
        # average the weights at some fraction with the original weights
        fswgts = field_sign_weight*fswgts + (1 - field_sign_weight)*wgts
    else: fswgts = None
    # add in radius weights if requested as well
    if not np.isclose(radius_weight, 0) and mdl.area_name_to_id is not None:
        id2n = mdl.area_id_to_name
        emprad = extract_retinotopy_argument(mesh, 'radius', radius, default='empirical')
        emprad = emprad[idcs]
        emprad = np.argsort(np.argsort(emprad)) * (1.0 / len(emprad)) - 0.5
        eccs = eccentricity[idcs]
        prerad = np.asarray([predict_pRF_radius(ecc, id2n[lbl], source=radius_weight_source)
                             for (ecc,lbl) in zip(eccs,labs)])
        prerad = np.argsort(np.argsort(prerad)) * (1.0 / len(prerad)) - 0.5
        rdwgts = 1.0 - (emprad - prerad)**2
        # average the weights at some fraction with the original weights
        rdwgts = radius_weight*rdwgts + (1-radius_weight)*wgts
    else: rdwgts = None
    # apply the weights
    if fswgts is not None:
        if rdwgts is not None: wgts = np.power(fswgts*rdwgts*wgts, 1.0/3.0)
        else:                  wgts = np.sqrt(fswgts*wgts)
    elif rdwgts is not None:   wgts = np.sqrt(rdwgts*wgts)
    # Figure out the sigma parameter:
    if sigma is None: sigs = None
    elif pimms.is_number(sigma): sigs = sigma
    elif pimms.is_vector(sigma) and len(sigma) == 3:
        [minsig, mult, maxsig] = sigma
        sigs = np.clip(
            [mult*min([npla.norm(a0 - a) for a in anchs if a is not a0]) if len(iii) > 1 else maxsig
             for (iii,anchs,_) in data
             for a0 in anchs],
            minsig, maxsig)
    else:
        raise ValueError('sigma must be a number or a list of 3 numbers')
    # okay, we've partially parsed the data that was given; now we can construct the final list of
    # instructions:
    tmp =  (['anchor', shape,
             np.asarray(idcs, dtype=np.int),
             np.asarray(ancs, dtype=np.float64),
             'scale', np.asarray(wgts, dtype=np.float64)]
            + ([] if sigs is None else ['sigma', sigs])
            + ([] if suffix is None else suffix))
    return tmp

####################################################################################################
# Registration Calculations
# The registration system is a set of pimms calculation functions all wrapped into a pimms plan;
# this way it is modular and easy to modify.

@pimms.calc('empirical_retinotopy')
def calc_empirical_retinotopy(cortex,
                              polar_angle=None, eccentricity=None, pRF_radius=None, weight=None,
                              eccentricity_range=None, weight_min=0,
                              invert_rh_angle=False,
                              partial_voluming_correction=False):
    '''
    calc_empirical_retinotopy computes the value empirical_retinotopy, which is an itable object
      storing the retinotopy data for the registration.

    Required afferent parameters:
      @ cortex Must be the cortex object that is to be registered to the model of retinotopy.
 
    Optional afferent parameters:
      @ polar_angle May be an array of polar angle values or a polar angle property name; if None
        (the default), attempts to auto-detect an empirical polar angle property.
      @ eccentricity May be an array of eccentricity values or an eccentricity property name; if
        None (the default), attempts to auto-detect an empirical eccentricity property.
      @ pRF_radius May be an array of receptive field radius values or the property name for such an
        array; if None (the default), attempts to auto-detect an empirical radius property.
      @ weight May be an array of weight values or a weight property name; if None (the default),
        attempts to auto-detect an empirical weight property, such as variance_explained.
      @ eccentricity_range May be a maximum eccentricity value or a (min, max) eccentricity range
        to be used in the registration; if None, then no clipping is done.
      @ weight_min May be given to indicate that weight values below this value should not be
        included in the registration; the default is 0.
      @ partial_voluming_correction May be set to True (default is False) to indicate that partial
        voluming correction should be used to adjust the weights.
      @ invert_rh_angle May be set to True (default is False) to indicate that the right hemisphere
        has its polar angle stored with opposite sign to the model polar angle.

    Efferent values:
      @ empirical_retinotopy Will be a pimms itable of the empirical retinotopy data to be used in
        the registration; the table's keys will be 'polar_angle', 'eccentricity', and 'weight';
        values that should be excluded for any reason will have 0 weight and undefined angles.
    '''
    data = {}  # the map we build up in this function
    n = cortex.vertex_count
    (emin,emax) = (-np.inf,np.inf)       if eccentricity_range is None          else \
                  (0,eccentricity_range) if pimms.is_number(eccentricity_range) else \
                  eccentricity_range
    # Step 1: get our properties straight ##########################################################
    (ang, ecc, rad, wgt) = [
        np.array(extract_retinotopy_argument(cortex, name, arg, default='empirical'))
        for (name, arg) in [
                ('polar_angle', polar_angle),
                ('eccentricity', eccentricity),
                ('radius', pRF_radius),
                ('weight', np.full(n, weight) if pimms.is_number(weight) else weight)]]
    if wgt is None: wgt = np.ones(len(ecc))
    bad = np.logical_not(np.isfinite(np.prod([ang, ecc, wgt], axis=0)))
    ecc[bad] = 0
    wgt[bad] = 0
    if rad is not None: rad[bad] = 0
    # do partial voluming correction if requested
    if partial_voluming_correction: wgt = wgt * (1 - cortex.partial_voluming_factor)
    # now trim and finalize
    bad = bad | (wgt <= weight_min) | (ecc < emin) | (ecc > emax)
    wgt[bad] = 0
    ang[bad] = 0
    ecc[bad] = 0
    for x in [ang, ecc, wgt, rad]:
        if x is not None:
            x.setflags(write=False)
    # that's it!
    dat = dict(polar_angle=ang, eccentricity=ecc, weight=wgt)
    if rad is not None: dat['radius'] = rad
    return (pimms.itable(dat),)
@pimms.calc('model')
def calc_model(cortex, model_argument, model_hemi=Ellipsis, radius=np.pi/3):
    '''
    calc_model loads the appropriate model object given the model argument, which may given the name
    of the model or a model object itself.

    Required afferent parameters:
      @ model_argument Must be either a RegisteredRetinotopyModel object or the name of a model that
        can be loaded.

    Optional afferent parameters:
      @ model_hemi May be used to specify the hemisphere of the model; this is usually only used
        when the fsaverage_sym hemisphere is desired, in which case this should be set to None; if
        left at the default value (Ellipsis), then it will use the hemisphere of the cortex param.

    Provided efferent values:
      @ model Will be the RegisteredRetinotopyModel object to which the mesh should be registered.
    '''
    if pimms.is_str(model_argument):
        h = cortex.chirality if model_hemi is Ellipsis else \
            None             if model_hemi is None     else \
            model_hemi
        model = retinotopy_model(model_argument, hemi=h, radius=radius)
    else:
        model = model_argument
    if not isinstance(model, RegisteredRetinotopyModel):
        raise ValueError('model must be a RegisteredRetinotopyModel')
    return model
@pimms.calc('native_mesh', 'preregistration_mesh', 'preregistration_map')
def calc_initial_state(cortex, model, empirical_retinotopy, resample=Ellipsis, prior=None):
    '''
    calc_initial_state is a calculator that prepares the initial state of the registration process.
    The initial state consists of a flattened 2D mesh ('native_map') that has been made from the
    initial cortex, and a 'registration_map', on which registration is to be performed. The former
    of these two meshes will always share vertex labels with the cortex argument, and the latter
    mesh this mesh may be identical to the native mesh or may be a resampled version of it.

    Optional afferent parameters:
      @ resample May specify that the registration_map should be resampled to the 'fsaverage' or the
        'fsaverage_sym' map; the advantage of this is that the resampling prevents angles already
        distorted by inflation, registration, and flattening from being sufficiently small to
        dominate the registration initially. The default value is Ellipsis, which specifies that the
        'fsaverage' or 'fsaverage_sym' resampling should be applied if the model is registered to
        either of those, and otherwise no resampling should be applied.
      @ prior May specify an alternate registration to which the native mesh should be projected
        prior to flattening and registration. The default value, None, indicates that the model's
        default registration should just be used. Generally models will be registered to either the 
        fsaverage or fsaverage_sym atlases; if your fsaverage subject has a geometry file matching
        the pattern ?h.*.sphere.reg, such as lh.retinotopy.sphere.reg, you can use that file as a
        registration (with registration/prior name 'retinotopy') in place of the fsaverage's native
        lh.sphere from which to start the overall retinotopy registration.

    Provided efferent values:
      @ native_mesh Will be the 3D mesh registered to the model's required registration space.
      @ preregistration_mesh Will be the 3D mesh that is ready to be used in registration; this may
        be identical to native_mesh if there is no resampling.
      @ preregistration_map Will be the 2D flattened mesh for use in registration; this mesh is
        a flattened version of preregistration_mesh.
    '''
    model_reg = model.map_projection.registration
    model_reg = 'native' if model_reg is None else model_reg
    model_chirality = None if model_reg == 'fsaverage_sym' else model.map_projection.chirality
    ch = 'lh' if model_chirality is None else model_chirality
    if model_chirality is not None and cortex.chirality != model_chirality:
        raise ValueError('Inverse-chirality hemisphere cannot be registered to model')
    # make sure we are registered to the model space
    if model_reg not in cortex.registrations:
        raise ValueError('given Cortex is not registered to the model registration: %s' % model_reg)
    # give this registration the correct data
    native_mesh = cortex.registrations[model_reg].with_prop(empirical_retinotopy)
    preregmesh = native_mesh # will become the preregistration mesh below
    # see about the prior
    if prior is not None:
        try:
            mdl_ctx = getattr(nyfs.subject(model_reg), ch)
            nativ = mdl_ctx.registrations['native']
            prior = mdl_ctx.registrations[prior]
        except Exception: raise ValueError('Could not find given prior %s' % prior)
        addr = nativ.address(native_mesh)
        preregmesh = native_mesh.copy(coordinates=prior.unaddress(addr))
        # and now, resampling...
    if resample is Ellipsis:
        resample = model_reg if model_reg == 'fsaverage' or model_reg == 'fsaverage_sym' else None
    if resample is not None and resample is not False:
        # make a map from the appropriate hemisphere...
        preregmesh = getattr(nyfs.subject(resample), ch).registrations['native']
        # resample properties over...
        preregmesh = preregmesh.with_prop(native_mesh.interpolate(preregmesh.coordinates, 'all'))
    # make the map projection now...
    preregmap = model.map_projection(preregmesh)
    return {'native_mesh':          native_mesh,
            'preregistration_mesh': preregmesh,
            'preregistration_map':  preregmap}
@pimms.calc('anchors')
def calc_anchors(preregistration_map, model, model_hemi,
                 scale=1, sigma=Ellipsis, radius_weight=0, field_sign_weight=0,
                 invert_rh_field_sign=False):
    '''
    calc_anchors is a calculator that creates a set of anchor instructions for a registration.

    Required afferent parameters:
      @ invert_rh_field_sign May be set to True (default is False) to indicate that the right
        hemisphere's field signs will be incorrect relative to the model; this generally should be
        used whenever invert_rh_angle is also set to True.

    '''
    wgts = preregistration_map.prop('weight')
    rads = preregistration_map.prop('radius')
    if np.isclose(radius_weight, 0): radius_weight = 0
    ancs = retinotopy_anchors(preregistration_map, model,
                              polar_angle='polar_angle',
                              eccentricity='eccentricity',
                              radius='radius',
                              weight=wgts, weight_min=0, # taken care of already
                              radius_weight=radius_weight, field_sign_weight=field_sign_weight,
                              scale=scale,
                              invert_field_sign=(model_hemi == 'rh' and invert_rh_field_sign),
                              **({} if sigma is Ellipsis else {'sigma':sigma}))
    return ancs

@pimms.calc('registered_map')
def calc_registration(preregistration_map, anchors,
                      max_steps=2000, max_step_size=0.05, method='random'):
    '''
    calc_registration is a calculator that creates the registration coordinates.
    '''
    # if max steps is a tuple (max, stride) then a trajectory is saved into
    # the registered_map meta-data
    pmap = preregistration_map
    if is_tuple(max_steps) or is_list(max_steps):
        (max_steps, stride) = max_steps
        traj = [preregistration_map.coordinates]
        x = preregistration_map.coordinates
        for s in np.arange(0, max_steps, stride):
            x = mesh_register(
                preregistration_map,
                [['edge',      'harmonic',      'scale', 1.0],
                 ['angle',     'infinite-well', 'scale', 1.0],
                 ['perimeter', 'harmonic'],
                 anchors],
                initial_coordinates=x,
                method=method,
                max_steps=stride,
                max_step_size=max_step_size)
            traj.append(x)
        pmap = pmap.with_meta(trajectory=np.asarray(traj))
    else:
        x = mesh_register(
            preregistration_map,
            [['edge',      'harmonic',      'scale', 1.0],
             ['angle',     'infinite-well', 'scale', 1.0],
             ['perimeter', 'harmonic'],
             anchors],
            method=method,
            max_steps=max_steps,
            max_step_size=max_step_size)
    return pmap.copy(coordinates=x)
@pimms.calc('registered_mesh', 'registration_prediction', 'prediction', 'predicted_mesh')
def calc_prediction(registered_map, preregistration_mesh, native_mesh, model):
    '''
    calc_registration_prediction is a pimms calculator that creates the both the prediction and the
    registration_prediction, both of which are pimms itables including the fields 'polar_angle',
    'eccentricity', and 'visual_area'. The registration_prediction data describe the vertices for
    the registered_map, not necessarily of the native_mesh, while the prediction describes the
    native mesh.

    Provided efferent values:
      @ registered_mesh Will be a mesh object that is equivalent to the preregistration_mesh but
        with the coordinates and predicted fields (from the registration) filled in. Note that this
        mesh is still in the resampled configuration is resampling was performed.
      @ registration_prediction Will be a pimms ITable object with columns 'polar_angle', 
        'eccentricity', and 'visual_area'. For values outside of the model region, visual_area will
        be 0 and other values will be undefined (but are typically 0). The registration_prediction
        describes the values on the registrered_mesh.
      @ prediction will be a pimms ITable object with columns 'polar_angle', 'eccentricity', and
        'visual_area'. For values outside of the model region, visual_area will be 0 and other
        values will be undefined (but are typically 0). The prediction describes the values on the
        native_mesh and the predicted_mesh.
    '''
    # invert the map projection to make the registration map into a mesh
    coords3d = np.array(preregistration_mesh.coordinates)
    idcs = registered_map.labels
    coords3d[:,idcs] = registered_map.meta('projection').inverse(registered_map.coordinates)
    rmesh = preregistration_mesh.copy(coordinates=coords3d)
    # go ahead and get the model predictions...
    d = model.cortex_to_angle(registered_map.coordinates)
    id2n = model.area_id_to_name
    (ang, ecc) = d[0:2]
    lbl = np.asarray(d[2], dtype=np.int)
    rad = np.asarray([predict_pRF_radius(e, id2n[l]) if l > 0 else 0 for (e,l) in zip(ecc,lbl)])
    d = {'polar_angle':ang, 'eccentricity':ecc, 'visual_area':lbl, 'radius':rad}
    # okay, put these on the mesh
    rpred = {}
    for (k,v) in six.iteritems(d):
        v.setflags(write=False)
        tmp = np.zeros(rmesh.vertex_count, dtype=v.dtype)
        tmp[registered_map.labels] = v
        tmp.setflags(write=False)
        rpred[k] = tmp
    rpred = pyr.pmap(rpred)
    rmesh = rmesh.with_prop(rpred)
    # next, do all of this for the native mesh..
    if native_mesh is preregistration_mesh:
        pred = rpred
        pmesh = rmesh
    else:
        # we need to address the native coordinates in the prereg coordinates then unaddress them
        # in the registered coordinates; this will let us make a native-registered-map and repeat
        # the exercise above
        addr = preregistration_mesh.address(native_mesh.coordinates)
        natreg_mesh = native_mesh.copy(coordinates=rmesh.unaddress(addr))
        d = model.cortex_to_angle(natreg_mesh)
        (ang,ecc) = d[0:2]
        lbl = np.asarray(d[2], dtype=np.int)
        rad = np.asarray([predict_pRF_radius(e, id2n[l]) if l > 0 else 0 for (e,l) in zip(ecc,lbl)])
        pred = pyr.m(polar_angle=ang, eccentricity=ecc, radius=rad, visual_area=lbl)
        pmesh = natreg_mesh.with_prop(pred)
    return {'registered_mesh'        : rmesh,
            'registration_prediction': rpred,
            'prediction'             : pred,
            'predicted_mesh'         : pmesh}

#: retinotopy_registration is the pimms calculation plan executed by register_retinotopy()
retinotopy_registration = pimms.plan(
    retinotopy = calc_empirical_retinotopy,
    model      = calc_model,
    initialize = calc_initial_state,
    anchors    = calc_anchors,
    register   = calc_registration,
    predict    = calc_prediction)

def register_retinotopy(hemi,
                        model='benson17', model_hemi=Ellipsis,
                        polar_angle=None, eccentricity=None, weight=None, pRF_radius=None,
                        weight_min=0.1,
                        eccentricity_range=None,
                        partial_voluming_correction=False,
                        radius_weight=1, field_sign_weight=1, invert_rh_field_sign=False,
                        scale=20.0,
                        sigma=Ellipsis,
                        select='close',
                        prior=None,
                        resample=Ellipsis,
                        radius=np.pi/3,
                        max_steps=2000, max_step_size=0.05, method='random',
                        yield_imap=False):
    '''
    register_retinotopy(hemi) registers the given hemisphere object, hemi, to a model of V1, V2,
      and V3 retinotopy, and yields a copy of hemi that is identical but additionally contains
      the registration 'retinotopy', whose coordinates are aligned with the model.

    Registration attempts to align the vertex positions of the hemisphere's spherical surface with a
    model of polar angle and eccentricity. This alignment proceeds through several steps and can
    be modified by several options. A description of these steps and options are provided here. For
    most cases, the default options should work relatively well.

    Method:
      (1) Prepare for registration by several intitialization substeps:
            a. Extract the polar angle, eccentricity and weight data from the hemisphere. These
               data are usually properties on the mesh and can be modifies by the options
               polar_angle, eccentricity, and weight, which can be either property names or list
               of property values. By default (None), a property is chosen using the functions
               neuropythy.vision.extract_retinotopy_argument with the default option set to
               'empirical'.
            b. If partial voluming correction is enabled (via the option
               partial_voluming_correction), multiply the weight by (1 - p) where p is 
               hemi.partial_volume_factor.
            c. If there is a prior that is specified as a belief about the retinotopy, then a
               Registration is created for the hemisphere such that its vertices are arranged
               according to that prior (see also the prior option). Note that because hemi's
               coordinates must always be projected into the registration specified by the model,
               the prior must be the name of a registration to which the model's specified subject
               is also registered. This is clear in the case of an example. The default value for
               this is 'retinotopy'; assuming that our model is specified on the fsaverage_sym, 
               surface, the initial positions of the coordinates for the registration process would
               be the result of starting with hemi's fsaverage_sym-aligned coordinates then warping
               these coordinates in a way that is equivalent to the warping from fsaverage_sym's 
               native spherical coordinates to fsaverage_sym's retinotopy registration coordinates.
               Note that the retinotopy registration would usually be specified in a file in the
               fsaverage_sym subject's surf directory: surf/lh.retinotopy.sphere.reg.
               If no prior is specified (option value None), then the vertices that are used are
               those aligned with the registration of the model, which will usually be 'fsaverage'
               or 'fsaverage_sym'.
            d. If the option resample is not None, then the vertex coordinates are resampled onto
               either the fsaverage or fsaverage_sym's native sphere surface. (The value of resample
               should be either 'fsaverage' or 'fsaverage_sym'.) Resampling can prevent vertices
               that have been rearranged by alignment with the model's specified registration or by
               application of a prior from beginning the alignment with very high initial gradients
               and is recommended for subject alignments.
               If resample is None then no changes are made.
            e. A 2D projection of the (possibly aligned, prior-warped, and resampled) cortical
               surface is made according to the projection parameters of the model. This map is the
               mesh that is warped to eventually fit the model.
      (2) Perform the registration by running neuropythy.registration.mesh_register. This step
          consists of two major components.
            a. Create the potential function, which we will minimize. The potential function is a
               complex function whose inputs are the coordinates of all of the vertices and whose
               output is a potential value that increases both as the mesh is warped and as the
               vertices with retinotopy predictions get farther away from the positions in the model
               that their retinotopy values would predict they should lie. The balance of these
               two forces is best controlled by the option functional_scale. The potential function
               fundamentally consists of four terms; the first three describe mesh deformations and
               the last describes the model fit.
                - The edge deformation term is described for any vertices u and v that are connected
                  by an edge in the mesh; it's value is c/p (r(u,v) - r0(u,v))^2 where c is the
                  edge_scale, p is the number of edges in the mesh, r(a,b) is the distance between
                  vertices a and b, and r0(a,b) is the distance between a and b in the initial mesh.
                - The angle deformation term is described for any three vertices (u,v,w) that form
                  an angle in the mesh; its value is c/m h(t(u,v,w), t0(u,v,w)) where c is the
                  angle_scale argument, m is the number of angles in the mesh, t is the value of the
                  angle (u,v,w), t0 is the value of the angle in the initial mesh, and h(t,t0) is an
                  infinite-well function that asymptotes to positive infinity as t approaches both 0
                  and pi and is minimal when t = t0 (see the nben's 
                  nben.mesh.registration.InfiniteWell documentation for more details).
                - The perimeter term prevents the perimeter vertices form moving significantly;
                  this primarily prevents the mesh from wrapping in on itself during registration.
                  The form of this term is, for any vertex u on the mesh perimeter, 
                  (x(u) - x0(u))^2 where x and x0 are the position and initial position of the
                  vertex.
                - Finally, the functional term is minimized when the vertices best align with the
                  retinotopy model.
            b. Register the mesh vertices to the potential function using the nben Java library. The
               particular parameters of the registration are method, max_steps, and max_step_size.

    Options:
      * model specifies the instance of the retinotopy model to use; this must be an
        instance of the RegisteredRetinotopyModel class or a string that can be passed to the
        retinotopy_model() function (default: 'standard').
      * model_hemi specifies the hemisphere of the model; generally you shouldn't have to set this
        unless you are using an fsaverage_sym model, in which case it should be set to None; in all
        other cases, the default value (Ellipsis) instructs the function to auto-detect the
        hemisphere.
      * polar_angle, eccentricity, pRF_radius, and weight specify the property names for the
        respective quantities; these may alternately be lists or numpy arrays of values. If weight
        is not given or found, then unity weight for all vertices is assumed. By default, each will
        check the  hemisphere's properties for properties with compatible names; it will prefer the
        properties PRF_polar_angle, PRF_ecentricity, and PRF_variance_explained if possible.
      * weight_min (default: 0.1) specifies the minimum value a vertex must have in the weight
        property in order to be considered as retinotopically relevant.
      * eccentricity_range (default: None) specifies that any vertex whose eccentricity is too low
        or too high should be given a weight of 0 in the registration.
      * partial_voluming_correction (default: True), if True, specifies that the value
        (1 - hemi.partial_volume_factor) should be applied to all weight values (i.e., weights
        should be down-weighted when likely to be affected by a partial voluming error).
      * field_sign_weight (default: 1) indicates the relative weight (between 0 and 1) that should
        be given to the field-sign as a method of determining which anchors have the strongest
        springs. A value of 1 indicates that the effective weights of anchors should be the 
        geometric mean of the empirical retinotopic weight and field-sign-based weight; a value of 0
        indicates that no attention should be paid to the field sign weight.
      * radius_weight (default: 1) indicates the relative weight (between 0 and 1) that should
        be given to the pRF radius as a method of determining which anchors have the strongest
        springs. A value of 1 indicates that the effective weights of anchors should be the 
        geometric mean of the empirical retinotopic weight and pRF-radius-based weight; a value of 0
        indicates that no attention should be paid to the radius-based weight.
      * sigma specifies the standard deviation of the Gaussian shape for the Schira model anchors;
        see retinotopy_anchors for more information.
      * scale (default: 1.0) specifies the strength of the functional constraints (i.e. the anchors:
        the part of the minimization responsible for ensuring that retinotopic coordinates are
        aligned); the anatomical constraints (i.e. the edges and angles: the part of the
        minimization responsible for ensuring that the mesh is not overly deformed) are always held
        at a strength of 1.0.
      * select specifies the select option that should be passed to retinotopy_anchors.
      * max_steps (default 2,000) specifies the maximum number of registration steps to run. This
        may be a tuple (max_steps, stride) in which case the registered map that is returned will
        contain a piece of meta-data, 'trajectory' containing the vertex coordinates every stride
        steps of the registration.
      * max_step_size (default 0.05) specifies the maxmim distance a single vertex is allowed to
        move in a single step of the minimization.
      * method (default 'random') is the method argument passed to mesh_register. This should be
        'random', 'pure', or 'nimble'. Generally, 'random' is recommended.
      * yield_imap (default: False) specifies whether the return value should be the new
        Mesh object or a pimms imap (i.e., a persistent mapping of the result of a pimms
        calculation) containing the meta-data that was used during the registration
        calculations. If this is True, then register_retinotopy will return immediately, and
        calculations will only be performed as the relevant data are requested from the returned
        imap. The item 'predicted_mesh' gives the return value when yield_imap is set to False.
      * radius (default: pi/3) specifies the radius, in radians, of the included portion of the map
        projection (projected about the occipital pole).
      * sigma (default Ellipsis) specifies the sigma argument to be passed onto the 
        retinotopy_anchors function (see help(retinotopy_anchors)); the default value, Ellipsis,
        is interpreted as the default value of the retinotopy_anchors function's sigma option.
      * prior (default: None) specifies the prior that should be used, if found, in the 
        topology registrations for the subject associated with the retinotopy_model's registration.
      * resample (default: Ellipsis) specifies that the data should be resampled to one of
        the uniform meshes, 'fsaverage' or 'fsaverage_sym', prior to registration; if None then no
        resampling is performed; if Ellipsis, then auto-detect either fsaverage or fsaverage_sym
        based on the model_hemi option (if it is None, fsaverage_sym, else fsaverage).
    '''
    # create the imap
    m = retinotopy_registration(
        cortex=hemi, model_argument=model, model_hemi=model_hemi,
        polar_angle=polar_angle, eccentricity=eccentricity, weight=weight, pRF_radius=pRF_radius,
        weight_min=weight_min,  eccentricity_range=eccentricity_range,
        partial_voluming_correction=partial_voluming_correction,
        radius_weight=radius_weight, field_sign_weight=field_sign_weight,
        invert_rh_field_sign=invert_rh_field_sign,
        scale=scale, sigma=sigma, select=select, prior=prior, resample=resample, radius=radius,
        max_steps=max_steps, max_step_size=max_step_size, method=method)
    return m if yield_imap else m['predicted_mesh']

# Tools for registration-free retinotopy prediction:
def predict_retinotopy(sub, template='benson14', registration='fsaverage'):
    '''
    predict_retinotopy(subject) yields a pair of dictionaries each with four keys: angle, eccen,
      sigma, and varea. Each of these keys maps to a numpy array with one entry per vertex.  The
      first element of the yielded pair is the left hemisphere map and the second is the right
      hemisphere map. The values are obtained by resampling the Benson et al. 2014 anatomically
      defined template of retinotopy to the given subject.

    The following optional arguments may be given:
      * template (default: 'benson14') specifies the template to use.
      * registration (default: 'fsaverage') specifies the subject registration to use; generally can
        only be 'fsaverage' or 'fsaverage_sym'.
    '''
    template = template.lower()
    retino_tmpls = predict_retinotopy.retinotopy_templates[registration]
    hemis = ['lh','rh'] if registration == 'fsaverage' else ['sym']
    if template not in retino_tmpls:
        libdir = os.path.join(library_path(), 'data')
        search_paths = [libdir]
        # just hard-baked-in for now.
        suff = 'v4_0' if registration == 'fsaverage' else 'v3_0'
        filenames = {(hname, fnm): ('%s.%s_%s.%s.mgz' % (hname,template,fnm,suff))
                     for fnm in ['angle','eccen','varea','sigma']
                     for hname in hemis}
        # find an appropriate directory
        tmpl_path = next((os.path.join(path0, registration)
                          for path0 in search_paths
                          if all(os.path.isfile(os.path.join(path0, registration, 'surf', s))
                                 for s in six.itervalues(filenames))),
                         None)
        if tmpl_path is None:
            raise ValueError('No subject found with appropriate surf/*.%s_* files!' % template)
        tmpl_sub = nyfs.subject(registration)
        spath = os.path.join(tmpl_path, 'surf')
        retino_tmpls[template] = pimms.persist(
            {h:{k: pimms.imm_array(dat)
                for k in ['angle', 'eccen', 'varea', 'sigma']
                for dat in [nyio.load(os.path.join(tmpl_path, 'surf', filenames[(h,k)]))]}
             for h in hemis})

    # Okay, we just need to interpolate over to this subject
    tmpl = retino_tmpls[template]
    if not all(s in tmpl for s in hemis):
        raise ValueError('could not find matching template')
    if registration == 'fsaverage_sym':
        sym = nyfs.subject('fsaverage_sym')
        if isinstance(sub, mri.Subject):
            subj_hems = (sub.lh, sub.hemis['rhx'])
            tmpl_hems = (sym.lh, sym.lh)
            chrs_hems = ('lh','rh')
        else:
            subj_hems = (sub,)
            tmpl_hems = (sym.lh,)
            chrs_hems = (sub.chirality,)
    else:
        fsa = nyfs.subject('fsaverage')
        if isinstance(sub, mri.Subject):
            subj_hems = (sub.lh, sub.rh)
            tmpl_hems = (fsa.lh, fsa.rh)
            chrs_hems = ('lh','rh')
        else:
            subj_hems = (sub,)
            tmpl_hems = ((fsa.lh if sub.chirality == 'lh' else fsa.rh),)
            chrs_hems = (sub.chirality,)
    tpl = tuple([th.interpolate(sh, tmpl[h if registration == 'fsaverage' else 'sym'])
                for (sh,th,h) in zip(subj_hems, tmpl_hems, chrs_hems)])
    return tpl[0] if len(tpl) == 1 else tpl
predict_retinotopy.retinotopy_templates = pyr.m(fsaverage={}, fsaverage_sym={})

def retinotopy_comparison(arg1, arg2, arg3=None,
                          eccentricity_range=None, polar_angle_range=None, visual_area_mask=None,
                          weight=Ellipsis, weight_min=None, visual_area=Ellipsis,
                          method='rmse', distance='scaled', gold=None):
    '''
    retinotopy_comparison(dataset1, dataset2) yields a pimms itable comparing the two retinotopy
      datasets.
    retinotopy_error(obj, dataset1, dataset2) is equivalent to retinotopy_comparison(x, y) where x
      and y are retinotopy(obj, dataset1) and retinotopy_data(obj, dataset2).
    
    The datasets may be specified in a number of ways, some of which may be incompatible with
    certain options. The simplest way to specify a dataset is as a vector of complex numbers, which
    are taken to represent positions in the visual field with (a + bi) corresponding to the
    coordinate (a deg, b deg) in the visual field. Alternately, an n x 2 or 2 x n matrix will be
    interpreted as (polar angle, eccentricity) coordinates, in terms of visual degrees (see the
    as_retinotopy function: as_retinotopy(arg, 'visual') yields this input format). Alternately,
    the datasets may be mappings such as those retuend by the retinotopy_data function; in this case
    as_retinotopy is used to extract the visual coordinates (so they need not be specified in visual
    coordinates specifically in this case). In this last case, additional properties such as the
    variance explained and pRF size can be returned, making it valuable for more sophisticated error
    methods or distance metrics.

    The returned dataset will always have a row for each row in the two datasets (which must have
    the same number of rows). However, many rows may have a weight of 0 even if no weights were 
    specified in the options; this is because other limitations may have been specified (such as
    in the eccentricity_range or visual_areas). The returned dataset will always contain the
    following columns:
      * 'weight' gives the weight assigned to this particular vertex; the weights will always sum to
        1 unless all vertices have 0 weight.
      * 'polar_angle_1' and 'polar_angle_2', 'eccentricity_1', 'eccenticity_2', 'x_1', 'x_2', 'y_1',
        'y_2', 'z_1', and 'z_2' all give the visual field coordinates in degrees; the z values give
        complex numbers equivalent to the x/y values.
      * 'radius_1' and 'radius_2' give the radii (sigma parameters) of the pRF gaussians.
      * 'polar_angle_error', 'eccentricity_error', and 'center_error' all give the difference
        between the visual field points in the two datasets; note that polar_angle_error in
        particular is an error measure of rotations around the visual field and not of visual field
        position. The 'center_error' is the distance between the centers of the visual field, in
        degrees. The 'radius_error' value is also given.
      * 'visual_area_1' and 'visual_area_2' specify the visual areas of the individual datasets; if
        either of the datasets did not have a visual area, it will be omitted. Additionally, the
        property 'visual_area' specifies the visual area suggested for use in analyses; this is
        chosen based on the following: (1) if there is a gold standard dataset specified that has
        a visual area, use it; (2) if only one of the datasets has a visual area, use it; (3) if
        both have a visual area, then use the (varea1 == varea2) * varea1 (the areas that agree are
        kept and all others are set to 0); (4) if neither has a visual area, then this property is
        omitted. In all cases where a 'visual_area' property is included, those vertices that do not
        overlap with the given visual_area_option option will be set to 0 along with the
        corresponding weights.
      * A variety of other lazily-calculated error metrics are included.

    The following options are accepted:
      * eccentricity_range (default: None) specifies the range of eccentricity to include in the
        calculation (in degrees). This may be specified as emax or (emin, emax).
      * polar_angle_range (default: None) specifies the range of polar angles to include in the
        calculation. Like eccentricity range it may be specified as (amin, amax) but amax alone is
        not allowed. Additionally the strings 'lh' and 'rvf' are equivalent to (0,180) and the
        strings 'rh' and 'lvf' are equivalent to (-180,0).
      * weight (default: Ellipsis) specifies the weights to be used in the calculation. This may be
        None to specify that no weights should be used, or a property name or an array of weight
        values. Alternately, it may be a tuple (w1, w2) of the weights for datasets 1 and 2. If the
        argument is Ellipsis, then it will use weights if they are found in the retinotopy dataset;
        both datasets may contain weights in which the product is used.
      * weight_min (default: None) specifies the minimum weight a vertex must have to be included in
        the calculation.
      * visual_area (default: Ellipsis) specifies the visual area labels to be used in the
        calculation. This may be None to specify that no labels should be used, or a property name
        or an array of labels. Alternately, it may be a tuple (l1, l2) of the labels for datasets 1 
        and 2. If the argument is Ellipsis, then it will use labels if they are found in the
        retinotopy dataset; both datasets may contain labels in which the gold standard's labels are
        used if there is a gold standard and the overlapping labels are used otherwise.
      * visual_area_mask (default: None) specifies a list of visual areas included in the
        calculation; this is applied to all datasets with a visual_area key; see the 'visual_area'
        columns above and the visual_area option. If None, then no visual areas are filtered;
        otherwise, arguments should like (1,2,3), which would usually specify that areas V1, V2, and
        V3, be included.
      * gold (default: None) specifies which dataset should be considered the gold standard; this
        should be either 1 or 2. If a gold-standard dataset is specified, then it is used in certain
        calculations; for example, when scaling an error by eccentricity, the gold-standard's
        eccentricity will be used unless there is no gold standard, in which case the mean of the
        two values are used.
    '''
    if arg3 is not None: (obj, dsets) = (arg1, [retinotopy_data(arg1, aa) for aa in (arg2,arg3)])
    else:                (obj, dsets) = (None,    [arg1, arg2])
    (gi,gold) = (None,False) if not gold else (gold-1,True)
    # we'll build up this result as we go...
    result = {}
    # they must have a retinotopy representation:
    vis = [as_retinotopy(ds, 'visual') for ds in dsets]
    ps = (vis[0][0], vis[1][0])
    es = (vis[0][1], vis[1][1])
    rs = [ds['radius'] if 'radius' in ds else None for ds in dsets]
    for ii in (0,1):
        s = '_%d' % (ii + 1)
        (p,e) = (ps[ii],es[ii])
        result['polar_angle'  + s] = p
        result['eccentricity' + s] = e
        if rs[ii] is not None: result['radius' + s] = rs[ii]
        p = np.pi/180.0 * (90.0 - p)
        (x,y) = (e*np.cos(p), e*np.sin(p))
        result['x' + s] = x
        result['y' + s] = y
        result['z' + s] = x + 1j * y
    n = len(ps[0])
    # figure out the weight
    if pimms.is_vector(weight) and len(weight) == 2:
        ws = [(None  if w is None                   else
               ds[w] if pimms.is_str(w) and w in ds else
               geo.to_property(obj, w))
              for (w,ds) in zip(weight, dsets)]
        weight = Ellipsis
    else:
        ws = [next((ds[k] for k in ('weight','variance_explained') if k in ds), None)
              for ds in dsets]
    if pimms.is_vector(weight, 'real'):
        wgt = weight
    elif pimms.is_str(weight):
        if obj is None: raise ValueError('weight property name but no vertex-set given')
        wgt = geo.to_property(obj, weight)
    elif weight is Ellipsis:
        if gold: wgt = ws[gi]
        elif ws[0] is None and ws[1] is None: wgt = None
        elif ws[0] is None: wgt = ws[1]
        elif ws[1] is None: wgt = ws[0]
        else: wgt = ws[0] * ws[1]
    else: raise ValueError('Could not parse weight argument')
    if wgt is None: wgt = np.ones(n)
    if ws[0] is not None: result['weight_1'] = ws[0]
    if ws[1] is not None: result['weight_2'] = ws[1]
    # figure out the visual areas
    if is_tuple(visual_area) and len(visual_area) == 2:
        ls = [(None  if l is None                   else
               ds[l] if pimms.is_str(l) and l in ds else
               geo.to_property(obj, l))
              for (l,ds) in zip(visual_area, dsets)]
        visual_area = Ellipsis
    else:
        ls = [next((ds[k] for k in ('visual_area','label') if k in ds), None)
              for ds in dsets]
    if pimms.is_vector(visual_area):
        lbl = visual_area
    elif pimms.is_str(visual_area):
        if obj is None: raise ValueError('visual_area property name but no vertex-set given')
        lbl = geo.to_property(obj, visual_area)
    elif visual_area is None:
        lbl = None
    elif visual_area is Ellipsis:
        if gold: lbl = ls[gi]
        elif ls[0] is None and ls[1] is None: lbl = None
        elif ls[0] is None: lbl = ls[1]
        elif ls[1] is None: lbl = ls[0]
        else: lbl = l[0] * (l[0] == l[1])
    else: raise ValueError('Could not parse visual_area argument')
    if ls[0] is not None: result['visual_area_1'] = ls[0]
    if ls[1] is not None: result['visual_area_2'] = ls[1]
    # Okay, now let's do some filtering; we clear weights as we go
    wgt = np.array(wgt)
    # Weight must be greater than the min
    if weight_min is not None: wgt[wgt < weight_min] = 0
    # Visual areas must be in the mask
    lbl = None if lbl is None else np.array(lbl)
    if lbl is not None and visual_area_mask is not None:
        if pimms.is_int(visual_area_mask): visual_area_mask = [visual_area_mask]
        oomask = (0 == np.sum([lbl == va for va in visual_area_mask], axis=0))
        wgt[oomask] = 0
        lbl[oomask] = 0
    if lbl is not None: result['visual_area'] = lbl
    # eccen must be in range
    if eccentricity_range is not None:
        er = eccentricity_range
        if pimms.is_real(er): er = (0,er)
        if gold: wgt[(es[gi] < er[0]) | (es[gi] > er[1])] = 0
        else:    wgt[(es[0] < er[0]) | (es[0] > er[1]) | (es[1] < er[0]) | (es[1] > er[1])] = 0
    # angle must be in range
    if polar_angle_range is not None:
        pr = polar_angle_range
        if pimms.is_str(pr):
            pr = pr.lower()
            if   pr in ['lh', 'rvf']: pr = (   0, 180)
            elif pr in ['rh', 'lvf']: pr = (-180,   0)
            else: raise ValueError('unrecognized polar angle range argument: %s' % pr)
        if gold: wgt[(ps[gi] < pr[0]) | (ps[gi] > pr[1])] = 0
        else:    wgt[(ps[0] < pr[0]) | (ps[0] > pr[1]) | (ps[1] < pr[0]) | (ps[1] > pr[1])] = 0
    # okay! Now we can add the weight into the result
    result['weight'] = wgt * zinv(np.sum(wgt))
    # now we add a bunch of calculations we can perform on the data!
    # first: metrics of distance
    gsecc = es[gi] if gold else np.mean(es, axis=0)
    gsang = ps[gi] if gold else np.mean(ps, axis=0)
    gsrad = rs[gi] if gold else rs[0] if rs[1] is None else rs[1] if rs[0] is None else \
            np.mean(rs, axis=0)
    gsecc_inv = zinv(gsecc)
    gsrad_inv = None if gsrad is None else zinv(gsrad)
    for (tag,resprop) in [('z', 'center'), ('polar_angle', 'polar_angle'),
                          ('eccentricity', 'eccentricity'), ('x', 'x'), ('y', 'y')]:
        serr = result[tag + '_1'] - result[tag + '_2']
        aerr = np.abs(serr)
        result[resprop + '_error'] = serr
        result[resprop + '_abs_error'] = aerr
        result[resprop + '_scaled_error'] = aerr * gsecc_inv
        if gsrad_inv is not None:
            result[resprop + '_radii_error'] = aerr * gsrad_inv
    return pimms.itable(result)

def visual_isolines(hemi, retinotopy='any', visual_area=Ellipsis, mask=None, surface='midgray',
                    weights=Ellipsis, min_weight=0.05,
                    eccentricity_range=None, polar_angle_range=None,
                    eccentricity_lines=8, polar_angle_lines=8,
                    max_step_scale=1.0/16.0):
    '''
    visual_isolines(hemi) yields a dictionary of the iso-angle and iso-eccentricity lines for the
      given hemisphere hemi. The returned structure depends on the optional arguments and is
      documented below.
    visual_isolines(hemi, retino) uses the retinotopy found by retinotopy_data(hemi, retino).
    
    The following optional arguments may be given:
      * visual_area (default: Ellipsis) specifies the property that should be used as the visual
        area labels; if Ellipsis, then uses any visual area property found in the retinotopy data
        and otherwise uses none; if set to None, then the visual area is always ignored. Note that
        visual area values of 0 are ignored as well.
      * mask (default: None) specifies an additional mask that should be applied; when visual_area
        is None or not found, then this is the only masking that is performed.
      * weights (default: Ellipsis) specifies the property that should be used as the weights on
        the polar angle and eccentricity data; if Ellipsis, indicates that the variance explained or
        weight property found in the retinotopy data should be used if any.
      * min_weight (default: 0.05) specifies the minimum weight that should be included.
      * polar_angle_range (default: None) may speficy the (min,max) polar angle to include.
      * eccentricity_range (default: None) may specify the max or (min,max) eccentricity to include.
      * eccentricity_lines (default: 8) specifies the eccentricity values at which to draw
        iso-eccentricity lines. If this is an integer, it specifies that these should be chosen
        using even spacing along the inverse CDF of the eccentricity values above the weight
        threshold.
      * polar_angle_lines (default: 8) specifies the polar angle values at which to draw the
        iso-angular lines. If this is an integer, it specifies that these should be chosen using
        even spacing along the inverse CDF of the eccentricity values above the weight threshold.
      * surface (default: 'midgray') specifies the surface to use for calculating surface
        coordinates if hemi is a topology and not a mesh.

    Return value:
      The return value of this function is a nested dictionary structure. If a visual area property
      is used, then the first level of keys are the visual areas (excluding 0) that are found.
      After this level, the next level's keys are 'polar_angle' and 'eccentricity' with the
      following level's keys indicating the polar angle or eccentricity value that was used. These
      internal dicts of iso-angle keys map to values that are lists of lines (there may be many
      contiguous lines at one angle); each line is given by a map of addresses and other meta-data
      about the line.
    '''
    from neuropythy import (to_mask, retinotopy_data, isolines, is_cortex, to_mesh)
    from neuropythy.util import curry
    # first off, apply the mask if specified:
    if mask is not None: mask = to_mask(hemi, mask)
    else: mask = hemi.tess.indices
    # now, get the retinotopy
    retino = retinotopy_data(hemi, retinotopy)
    if   visual_area is Ellipsis: visual_area = retino.get('visual_area', None)
    elif visual_area is not None: visual_area = hemi.property(visual_area, mask=mask, null=0)
    # if there is a visual area, we just recurse setting these values as a mask
    if visual_area is not None:
        vas = np.unique(visual_area)
        kw = dict(weights=weights, min_weight=min_weight, surface=surface, visual_area=None,
                  eccentricity_lines=eccentricity_lines, polar_angle_lines=polar_angle_lines,
                  eccentricity_range=eccentricity_range, polar_angle_range=polar_angle_range,
                  max_step_scale=max_step_scale)
        def recurse(va):
            vamask = to_mask(hemi, (visual_area, va))
            return visual_isolines(hemi, retinotopy, mask=np.intersect1d(mask, vamask), **kw)
        return pimms.lazy_map({va:curry(recurse, va) for va in vas if va != 0})
    # If there are weights, figure them out and apply them to the mask
    ve = 'variance_explained'
    weights = (retino[ve] if weights is Ellipsis and ve in retino   else
               None       if weights is Ellipsis or weights is None else
               hemi.property(weights))
    if weights is not None and min_weight is not None:
        mask = np.intersect1d(mask, np.where(weights >= min_weight)[0])
    # make sure polar angle and eccen are in retino:
    (ang,ecc) = as_retinotopy(retino, 'visual')
    retino['polar_angle'] = ang
    retino['eccentricity'] = ecc
    # if there's a surface to get...
    mesh = to_mesh((hemi, surface))
    # when we calculate the isolines we use this function which also adds in the polar angles and
    # eccentricities of the addressed lines
    def calc_isolines(hemi, dat, ln, mask=None):
        addrs = isolines(hemi, dat, ln, mask=mask, yield_addresses=True)
        (angs,eccs) = [[hemi.interpolate(addr, retino[nm]) for addr in addrs]
                       for nm in ('polar_angle', 'eccentricity')]
        vxys = [np.asarray([e*np.cos(t), e*np.sin(t)])
                for (a,e) in zip(angs,eccs) for t in [np.pi/180.0*(90.0 - a)]]
        sxys = [mesh.unaddress(addr) for addr in addrs]
        # trim/cut where needed
        if max_step_scale is not None: 
            iiss = []
            for (vxy,ecc) in zip(vxys, eccs):
                dists = np.sqrt(np.sum((vxy[:,:-1] - vxy[:,1:])**2, 0))
                es = np.mean([ecc[:-1], ecc[1:]], 0)
                mx = max_step_scale * es
                bd = dists > mx
                ok = np.where(~bd)[0]
                bd = np.where(bd)[0]
                (oi,bi) = (0,0)
                iis = []
                while True:
                    if   oi >= len(ok): break
                    elif bi >= len(bd):
                        iis.append(ok[oi:])
                        break
                    elif ok[oi] < bd[bi]:
                        n = bd[bi] - ok[oi]
                        iis.append(ok[oi:(oi+n)])
                        oi += n
                    else: bi += ok[oi] - bd[bi]
                iiss.append(iis)
            # okay, fix all the lists...
            (angs,eccs) = [[u[ii]   for (u,iis) in zip(q,iiss) for ii in iis] for q in (angs,eccs)]
            (vxys,sxys) = [[u[:,ii] for (u,iis) in zip(q,iiss) for ii in iis] for q in (vxys,sxys)]
            addrs = [{k:addr[k][:,ii] for k in ('faces','coordinates')}
                     for (addr,iis) in zip(addrs,iiss) for ii in iis]
        return pimms.persist({'addresses': addrs, 'polar_angles': angs, 'eccentricities':eccs,
                              'visual_coordinates': vxys, 'surface_coordinates': sxys})
    # okay, we are operating over just the mask we have plus polar angle/eccentricity ranges
    r = {}
    (ang,ecc) = as_retinotopy(retino, 'visual')
    if eccentricity_range is not None:
        rng     = eccentricity_range
        (mn,mx) = rng if pimms.is_vector(rng) else (np.min(ecc), rng)
        mask    = np.setdiff1d(mask, np.where((ecc < mn) | (ecc > mx))[0])
    if polar_angle_range is not None:
        rng     = polar_angle_range
        (mn,mx) = rng if pimms.is_vector(rng) else (np.min(ang), rng)
        # we need to center the angle on this range here
        aa = np.mean(rng)
        aa = np.mod(ang + aa, 360) - aa
        mask = np.setdiff1d(mask, np.where((aa < mn) | (aa > mx))[0])
    for (p,dat,lns,rng) in zip(['polar_angle','eccentricity'], [ang,ecc],
                               [polar_angle_lines, eccentricity_lines],
                               [polar_angle_range, eccentricity_range]):
        # first, figure out the lines themselves
        if pimms.is_int(lns): lns = np.percentile(dat[mask], np.linspace(0, 100, 2*lns + 1)[1::2])
        # now grab them from the hemisphere...
        r[p] = pimms.lazy_map({ln:curry(calc_isolines, hemi, dat, ln, mask=mask) for ln in lns})
    return pyr.pmap(r)

def clean_retinotopy_potential(hemi, retinotopy=Ellipsis, mask=Ellipsis, weight=Ellipsis,
                               surface='midgray', min_weight=Ellipsis, min_eccentricity=0.75,
                               visual_area=None, map_visual_areas=Ellipsis,
                               visual_area_field_signs=Ellipsis,
                               measurement_uncertainty=0.3, measurement_knob=1,
                               magnification_knob=0, fieldsign_knob=6, edge_knob=0):
    '''
    clean_retinotopy_potential(hemi) yields a retinotopic potential function for the given
      hemisphere that, when minimized, should yeild a cleaned/smoothed version of the retinotopic
      maps.

    The potential function f returned by clean_retinotopy_potential() is a PotentialFunction object,
    as defined in neuropythy.optimize. The potential function consists of four terms that are summed
    with weights derived from the four '*_knob' options (see below). The function f as well as the
    three terms that it comprises require as input a matrix X of the pRF centers of mesh or the
    masked part of the mesh (X0 is the initial measurement matrix). These four potential terms are:
      * The measurement potential. The retinotopic map that is being minimized is referred to as the
        measured map, and the measurement potential function, fm(X), increases as X becomes farther
        from X0. Explicitly, fm(X) is the sum over all pRF centers (x,y) in X (with initial position
        (x0,y0) in X0) of exp(-0.5 * ((x - x0)**2 + (y - y0)**2) / s**2). The parameter s is the
        initial eccentricity (sqrt(x0**2 + y0**2)) times the measurement_uncertainty option.
      * The magnification potential. The retinotopy cleaning is performed in part by attempting to
        smooth the visual magnification (the inverse of cortical magnification: deg**2 / mm**2)
        across the cortical surface; the magnification potential fg(X) specifies how the visual
        magnification contributes to the overall potential: it decreases as the magnification
        becomes smoother and increases as it becomes less smooth. Explicitly, fg(X) is equal to the
        sum over all pairs of faces (s,t) sharing one edge e of
        (vmag(s) - sgn(vmag(t))*vmag(e))**2 + (vmag(t) - sgn(vmag(s))*vmag(e))**2. Note that the
        function vmag(s) yields the areal visual magnification (deg**2 / mm**2) of any face s and
        vmag(e) is the square of the linear magnification of any edge e; additionally, the sign of
        vmag(s) for a face s is always equal to the fieldsign of the face (while for edges vmag(e)
        is always positive).
      * The fieldsign potential. The next way in which the potential function attempts to clean the
        retinotopy is via the use of fieldsign: adjacent faces should have the same fieldsign under
        most circumstanced. This is modeled by the function fs(X), which is 0 for any pair of faces
        that have the same fieldsign and non-zero for faces that have different fieldsigns. The
        form of fs(X) is the sum over all pairs of adjacent triangles (s,t) of -vmag(s)*vmag(t) if
        vmag(s) and vmag(t) have different fieldsigns, otherwise 0.
      * The edge potential. Finally, the potential function attempts to force edges to be smooth by
        penalizing edges whose endpoints are far apart in the visual field. The edge potential
        function fe(X) is equal to the sum for all edges (u,v) of
        (x[u] - x[v])**2 + (y[u] - y[v])**2 / mean(eccen(u), eccen(v)).

    Note additionally that all four potential functions are normalized by a factor intended to keep
    them on similar scales (this factor is not mentioned above or below, but it is automatically
    applied to all potential terms). For the magnification, fieldsign, and edge potential terms, the
    normalization factor is 1/m where m is the number of non-perimeter edges (or, alternately, the
    number of adjacent face pairs) in the mesh. For the measurement potential term, the
    normalization factor is 1/W where W is the sum of the weights on the measurement vertices (if
    no weights are given, they are considered to be 1 for each vertex).

    The following options may be given:
      * retinotopy (default: Ellipsis) specifies the retinotopy data to use for the hemisphere;
        the argument may be a map from retinotopy_data or a valid argument to it. The default
        indicates that the result of calling retinotopy_data(hemi) is used.
      * mask (default: Ellipsis) specifies that the specific mask should be used; by default, the
        mask is made using the vertices kept in to_flatmap('occipital_pole', hemi, radius=pi/2.75).
      * weight (default: Ellipsis) specifies the weight to use; the default indicates that the
        weight found in the retinotopy data, if any, should be used. If None, then all values
        in the mask are equally weighted.
      * visual_area (default: Ellipsis) specifies the visual area labels to use; the default
        indicates that the visual area property found in the retinotopy data should be used, if any.
        If None then no visual area splitting is done. This property is only important if 
        map_visual_areas is not False or None; otherwise it is ignored.
      * map_visual_areas (default: Ellipsis) specifies whether the return value should be a lazy map
        whose keys are visual area labels and whose values are recursed calls to this function for
        only the subset of the mesh with the associated label. May be False or None to specify that
        a single potential should be yielded. May be a list of labels to specify that only those
        visual areas should be mapped; the default value (Ellipsis) uses all labels in visual_areas
        except for 0.
      * min_weight (default: Ellipsis) specifies the minimum weight to include, after the
        weights have been normalized such that sum(weights) == 1. If the value is a list or
        tuple containing a single item [p] then p is taken to be a percentile below which
        vertices should be excluded. The default, Ellipsis, is equivalent to [5].
      * min_eccentricity (default: 0.75) specifies the eccentricity below which no measurement-based
        potential is applied; i.e., by default, vertices with eccentricity below 0.75 degrees will
        be considered as having 0 weight.
      * surface (default: 'midgray') specifies which surface should be used to establish cortical
        magnification; may be 'pial', 'midgray', or 'white'.
      * measurement_uncertainty (default: 0.3) is used to determine the standard deviation of the 
        Gaussian potential well used to prevent individual vertices with valid retinotopic
        measurements from straying too far from their initial measured positions. In other words, if
        a vertex has a weight that is above threshold and a pRF center of (x0,y0), then the 
        measurement-potential for that vertex is exp(-0.5 * ((x - x0)**2 + (y - y0)**2)/s**2) where
        (x,y) is the center of the pRF during minimization and s is equal to
        measurement_uncertainty * sqrt(x0**2 + y0**2).
      * measurement_knob, magnification_knob, fieldsign_knob, and edge_knob (defaults: 1, 0, 12, 0,
        respectively) specify the relative weights of the terms of the potential function on a log2
        scale. In other words, if the measurement, magnification, fieldsign, and edge potential
        terms are fm, fg, fs, and fe while the knobs are km, kg, ks, and ke, then the overall
        potential function f is equal to:
        f(X) = (2**km * fm(X) + 2**kg * fg(X) + 2**ks * fs(X) + 2**ke * fe(X)) / q
        where w = (2**km + 2**kg + 2**ks + 2**ke)
        If any knob is set to None, then its value is 0 instead of 2**k.
    '''
    from neuropythy.util import curry
    import neuropythy.optimize as op
    # first, get the mesh and the retinotopy data
    mesh = geo.to_mesh((hemi, surface))
    rdat = (retinotopy_data(mesh) if retinotopy is Ellipsis   else
            retinotopy            if pimms.is_map(retinotopy) else
            retinotopy_data(mesh, retinotopy))
    lbls = (rdat.get('visual_area') if visual_area is Ellipsis else
            None                    if visual_area is None     else
            hemi.property(visual_area))
    wght = (rdat.get('variance_explained') if weight is Ellipsis                      else
            weight                         if pimms.is_vector(weight)                 else
            rdat.get(weight)               if pimms.is_str(weight) and weight in rdat else
            hemi.property(weight)          if weight is not None                      else
            None)
    # figure out the mask
    if mask is Ellipsis:
        if mesh.coordinates.shape[0] == 2: mask = mesh.indices
        else: mask = geo.to_flatmap('occipital_pole', hemi, radius=np.pi/2.75).labels
    else: mask = hemi.mask(mask, indices=True)
    global_field_sign = None
    # if we are splitting on visual area, we should do that here:
    if map_visual_areas and lbls is not None:
        # get the visual areas
        vas = (np.unique(lbls)                    if map_visual_areas == 'all'           else
               np.setdiff1d(np.unique(lbls), [0]) if map_visual_areas in [True,Ellipsis] else
               np.unique(map_visual_areas))
        # we also want to have the field-sign map handy if provided
        if visual_area_field_signs is None: visual_area_field_signs = {}
        elif visual_area_field_signs is Ellipsis: visual_area_field_signs = {1:-1, 2:1, 3:-1, 4:1}
        # special case when map_visual_areas is an integer/string (label)
        if pimms.is_int(map_visual_areas) or pimms.is_str(map_visual_areas):
            mask = np.intersect1d(mask, np.where(lbls == map_visual_areas)[0])
            global_field_sign = visual_area_field_signs.get(map_visual_areas)
        else: # otherwise we return a lazy map
            kw = dict(retinotopy=rdat, mask=mask, weight=wght,
                      surface=surface, min_weight=min_weight, min_eccentricity=min_eccentricity,
                      visual_area=lbls, measurement_uncertainty=measurement_uncertainty,
                      measurement_knob=measurement_knob,
                      magnification_knob=magnification_knob, fieldsign_knob=fieldsign_knob,
                      edge_knob=edge_knob, visual_area_field_signs=visual_area_field_signs)
            return pimms.lazy_map(
                {lbl: curry(clean_retinotopy_potential, hemi, map_visual_areas=lbl, **kw)
                 for lbl in vas})
    # fix rdat, weight, and mesh to match the mask
    (supermesh, orig_mask) = (mesh, mask)
    rdat = {k:(v[mask] if len(v) > len(mask) else v) for (k,v) in six.iteritems(rdat)}
    mesh = supermesh.submesh(mask)
    # possible that the mask got further downsampled:
    mask = supermesh.tess.index(mesh.labels)
    if len(mask) == len(orig_mask): smsk = np.arange(len(mask))
    else:
        tmp = set(mask)
        smsk = np.asarray([k for (k,u) in enumerate(orig_mask) if u in tmp])
    n = mesh.vertex_count # number vertices
    N = 2*n # number parameters
    if wght is None:                  wght = np.ones(n)
    elif len(wght) == len(orig_mask): wght = np.array(wght)[smsk]
    elif len(wght) > n:               wght = np.array(wght)[mask]
    else:                             wght = np.array(wght)
    wght[~np.isfinite(wght)] = 0
    if min_eccentricity < 0 or np.isclose(min_eccentricity, 0):
        raise ValueError('min_eccentricity should be small but must be > 0')
    # we'll need a potential function...
    # The input to the potential function will be a 2 x N matrix of (x,y) visual field coordinates:
    xy      = op.identity
    (x,y)   = (xy[np.arange(0,N,2)],xy[np.arange(1,N,2)])
    # We have a few components to the potential function
    # [1] Deviation from measurements:
    # These are the initial measurements we will use
    xy0     = np.array(as_retinotopy(rdat, 'geographical')).T
    if len(xy0) == len(orig_mask): xy0 = xy0[smsk]
    elif len(xy0) > n:             xy0 = xy0[mask]
    (x0,y0) = xy0.T
    xy0     = xy0.flatten()
    ecc0    = np.sqrt(x0**2 + y0**2)
    ii      = np.where(ecc0 > min_eccentricity)[0]
    minw    = (0                             if min_weight is None          else
               np.percentile(wght[ii], 5)    if min_weight is Ellipsis      else
               min_weight                    if pimms.is_number(min_weight) else
               0                             if np.std(wght[ii]) < 0.00001  else
               np.percentile(wght[ii], min_weight[0]))
    ii      = np.intersect1d(ii, np.where(wght > minw)[0])
    wsum    = np.sum(wght[ii])
    if wsum < 0 or np.isclose(wsum, 0): raise ValueError('all-zero weights given')
    wght    = wght / wsum
    s2_meas = (measurement_uncertainty * ecc0[ii])**2
    d2_meas = (x[ii] - x0[ii])**2 + (y[ii] - y0[ii])**2
    f_meas  = (1 - op.exp(-0.5*d2_meas/s2_meas)) * wght[ii]
    f_meas  = op.sum(f_meas)
    # [2] For adjacent triangles, how different are the cortical magnifications?
    sarea   = mesh.face_areas
    faces   = mesh.tess.indexed_faces.T
    selen   = mesh.edge_lengths
    # we want to ensure that vmag is locally smooth across triangles, but we need
    # to make sure there aren't any edges or faces with 0 surface-areas (so that
    # we don't divide by zero)
    mnden   = 0.0001
    (e,s,t) = np.transpose([(i,e[0],e[1]) for (i,e) in enumerate(mesh.tess.edge_faces)
                            if len(e) == 2 and selen[i] > mnden
                            if sarea[e[0]] > mnden and sarea[e[1]] > mnden])    
    m       = len(e)
    (fis,q) = np.unique(np.concatenate([s,t]), return_inverse=True)
    (s,t)   = np.reshape(q, (2,-1))
    faces   = faces[fis]
    sarea   = sarea[fis]
    selen   = selen[e]
    (u,v)   = mesh.tess.indexed_edges[:,e]
    # we will use visual mag instead of cortical mag: this way we aren't worried about
    # varea going to 0 and creating a singularity, and it should have a linear 
    # relationship with eccentricity
    velen2  = (x[u] - x[v])**2 + (y[u] - y[v])**2
    vme     = velen2 / selen**2 # visual magnification: edges
    varea   = op.signed_face_areas(faces)
    vmf     = varea / sarea # visual magnification: faces
    vms     = vmf[s]
    vmt     = vmf[t]
    vsgns   = op.sign(vmf)
    f_magn  = (1.0 / m) * op.sum((vms - vsgns[t]*vme)**2 + (vmt - vsgns[s]*vme)**2)
    # [3] we want a special function for faces whose vmags are different signs
    if global_field_sign is None:
        f_sign = op.compose(op.piecewise(0, ((-np.inf, 0), -op.identity)), vms*vmt)
        f_sign = (1.0 / m) * op.sum(f_sign)
    else:
        vmfsgn = vmf * global_field_sign
        f_sign = op.compose(op.piecewise(0, ((-np.inf, 0), -op.identity)), vmfsgn)
        f_sign = (1.0 / m) * op.sum(f_sign)
    # and the edge potential...
    ex      = 0.5*(x[u] + x[v])
    ey      = 0.5*(y[u] + y[v])
    eecc2   = (ex**2 + ey**2)
    f_edge  = (1.0 / m) * op.sum(((x[u] - x[v])**2 + (y[u] - y[v])**2) / (eecc2 + 0.05))
    # This is the potential function:
    (k_meas, k_magn, k_sign, k_edge) = [
        0 if knob is None else (2**knob)
        for knob in (measurement_knob, magnification_knob, fieldsign_knob, edge_knob)]
    fs = (k_meas*f_meas, k_magn*f_magn, k_sign*f_sign, k_edge*f_edge)
    f = (fs[0] + fs[1] + fs[2] + fs[3]) / (k_meas + k_magn + k_sign + k_edge)
    xy0 = np.reshape(xy0, (-1,2))
    object.__setattr__(f, 'meta_data',
                       pyr.m(f_meas=f_meas, f_magn=f_magn, f_sign=f_sign, f_edge=f_edge,
                             mesh=mesh, X0=xy0))
    return f

def clean_retinotopy(hemi, retinotopy=Ellipsis, mask=Ellipsis, weight=Ellipsis,
                     surface='midgray', min_weight=Ellipsis, min_eccentricity=0.75,
                     visual_area=Ellipsis, map_visual_areas=Ellipsis,
                     visual_area_field_signs=Ellipsis,
                     measurement_uncertainty=0.3, measurement_knob=1,
                     magnification_knob=0, fieldsign_knob=6, edge_knob=0,
                     yield_report=False, steps=100, rounds=4, output_style='visual',
                     jitter=None, average=None):

    '''
    clean_retinotopy(hemi) attempts to cleanup the retinotopic maps on the given cortical mesh by
      minimizing an objective function that tracks the smoothness of the fields, the orthogonality
      of polar angle to eccentricity, and the deviation of the values from the measured values; the
      yielded result is the smoothed retinotopy, as would be returned by
      as_retinotopy(..., 'visual').

    The argument hemi may be a Cortex object or a Mesh. See also clean_retinotopy_potential for
    information on the various parameters related to the potential function that is minimized by
    clean_retinotopy(). The following additional options are also accepted:
      * output_style (default: 'visual') specifies the style of the output data that should be
        returned; this should be a string understood by as_retinotopy.
      * yield_report (False) may be set to True, in which case a tuple (retino, report) is returned,
        where the report is the return value of the scipy.optimization.minimize function.
    '''
    # parse our args
    if jitter in [True, Ellipsis, 'auto', 'automatic']: jitter = (4, 0.05, 1)
    if is_tuple(jitter) and len(jitter) > 0:
        if len(jitter) > 3:  raise ValueError('jitter tuple must be (mod, scale, phase)')
        if len(jitter) == 1: jitter = jitter + (0.005,)
        if len(jitter) == 2: jitter = jitter + (1,)
        (jitter_mod, jitter_scale, jitter_phase) = jitter
    else: jitter = None
    if average in [True, Ellipsis, 'auto', 'automatic']: average = (4, 3)
    if is_tuple(average) and len(average) > 0:
        if len(average) > 2:  raise ValueError('average tuple must be (mod, phase)')
        if len(average) == 1: average = average + (3,)
        (average_mod, average_phase) = average
    else: average = None
    if visual_area_field_signs is None: visual_area_field_signs = {}
    # First, make the potential function:
    f = clean_retinotopy_potential(hemi, retinotopy=retinotopy, mask=mask, weight=weight,
                                   surface=surface, min_weight=min_weight,
                                   min_eccentricity=min_eccentricity,
                                   measurement_uncertainty=measurement_uncertainty,
                                   measurement_knob=measurement_knob,
                                   magnification_knob=magnification_knob,
                                   fieldsign_knob=fieldsign_knob, edge_knob=edge_knob,
                                   visual_area=visual_area, map_visual_areas=map_visual_areas,
                                   visual_area_field_signs=visual_area_field_signs)
    # at this point, it's possible that we got a lazy map back; if so we're going to want to iterate
    # through it; otherwise, we'll want to just iterate through the single return value...
    m = f if pimms.is_map(f) else {None: f}
    (x,y) = np.full((2, hemi.vertex_count), np.nan) # the output x/y prf centers
    tess = hemi if geo.is_tess(hemi) else hemi.tess
    for (k,f) in six.iteritems(m):
        # The initial parameter vector is stored in the meta-data:
        X0 = f.meta_data['X0']
        submesh = f.meta_data['mesh']
        X = X0
        for ii in range(rounds):
            mtd = 'L-BFGS-B' if (ii % 2) == 0 else 'TNC'
            if jitter is not None and ii % jitter_mod == jitter_phase:
                ec = np.sqrt(np.sum(X**2, axis=1))
                th = (np.random.rand(len(ec)) - 0.5)*2*np.pi
                r  = np.random.exponential(ec*jitter_scale)
                X = X + np.transpose([r*np.cos(th), r*np.sin(th)])
            if average is not None and ii % average_mod == average_phase:
                X = np.array([X[k] if len(nn) == 0 else np.mean(X[list(nn)],0)
                              for (k,nn) in enumerate(submesh.tess.indexed_neighborhoods)])
            rr = f.minimize(X, method=mtd, options=dict(maxiter=steps, disp=False))
            X = rr.x
        X = np.reshape(X, X0.shape)
        if X.shape[0] != 2: X = X.T
        for (u,v) in zip([x,y], X): u[tess.index(submesh.labels)] = v
    return as_retinotopy({'x':x, 'y':y}, output_style)

