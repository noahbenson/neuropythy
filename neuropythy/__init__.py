####################################################################################################
# __init__.py

'''Tools for analyzing and registering cortical meshes.'''

submodules = ('neuropythy.util.conf',
              'neuropythy.util.core',
              'neuropythy.util.filemap',
              'neuropythy.util.labels',
              'neuropythy.util',
              'neuropythy.math.core',
              'neuropythy.math',
              'neuropythy.java',
              'neuropythy.io.core',
              'neuropythy.io',
              'neuropythy.geometry.util',
              'neuropythy.geometry.mesh',
              'neuropythy.geometry',
              'neuropythy.optimize.core',
              'neuropythy.optimize',
              'neuropythy.mri.core',
              'neuropythy.mri.images',
              'neuropythy.mri',
              'neuropythy.freesurfer.core',
              'neuropythy.freesurfer',
              'neuropythy.hcp.files',
              'neuropythy.hcp.core',
              'neuropythy.hcp',
              'neuropythy.registration.core',
              'neuropythy.registration',
              'neuropythy.vision.models',
              'neuropythy.vision.retinotopy',
              'neuropythy.vision.cmag',
              'neuropythy.vision',
              'neuropythy.graphics.core',
              'neuropythy.graphics',
              'neuropythy.datasets.core',
              'neuropythy.datasets.benson_winawer_2018',
              'neuropythy.datasets.hcp',
              'neuropythy.datasets.visual_performance_fields',
              'neuropythy.datasets.hcp_lines',
              'neuropythy.datasets',
              'neuropythy.plans.core',
              'neuropythy.plans.prfclean',
              'neuropythy.plans',
              'neuropythy.commands.surface_to_ribbon',
              'neuropythy.commands.benson14_retinotopy',
              'neuropythy.commands.register_retinotopy',
              'neuropythy.commands.atlas',
              'neuropythy.commands')
'''neuropythy.submodules is a tuple of all the sub-modules of neuropythy in a loadable order.'''

def reload_neuropythy():
    '''
    reload_neuropythy() reloads all of the modules of neuropythy and returns the reloaded
    neuropythy module. This is similar to reload(neuropythy) except that it reloads all the
    neuropythy submodules prior to reloading neuropythy.

    Example:
      import neuropythy as ny
      # ... some nonsense that breaks the library ...
      ny = ny.reload_neuropythy()
    '''
    import sys, six
    if not six.PY2:
        try:              from importlib import reload
        except Exception: from imp import reload
    for mdl in submodules:
        if mdl in sys.modules:
            sys.modules[mdl] = reload(sys.modules[mdl])
    return reload(sys.modules['neuropythy'])

from   .util       import (config, is_image, library_path, to_affine, is_address, address_data,
                           is_curve_spline, to_curve_spline, curve_spline, flattest,
                           is_list, is_tuple, to_hemi_str, is_dataframe, to_dataframe, auto_dict,
                           label_index, is_label_index, to_label_index)
from   .util       import label_indices as labels
from   .io         import (load, save, to_nifti)
from   .mri        import (is_subject, is_cortex, to_cortex, to_image, to_image_spec,
                           is_image_spec, image_interpolate, image_apply, image_copy, image_clear,
                           is_pimage, to_image_type)
from   .vision     import (retinotopy_data, empirical_retinotopy_data, predicted_retinotopy_data,
                           register_retinotopy, retinotopy_anchors, retinotopy_model,
                           neighborhood_cortical_magnification, as_retinotopy,
                           retinotopy_comparison, to_logeccen, from_logeccen)
from   .geometry   import (mesh, tess, topo, map_projection, path_trace, 
                           is_vset, is_mesh, is_tess, is_topo, is_flatmap, paths_to_labels,
                           is_map_projection, is_path, is_path_trace, close_path_traces,
                           to_mesh, to_tess, to_property, to_mask, to_flatmap, to_map_projection,
                           isolines, map_projections)
from   .freesurfer import (subject as freesurfer_subject, to_mgh)
from   .hcp        import (subject as hcp_subject)
from   .datasets   import data
from . import util
from . import math
from . import freesurfer
from . import hcp
from . import plans

# things we might want to load but that might fail
try:
    from .graphics import cortex_plot
    from .         import graphics
except Exception: pass

# Version information...
__version__ = '0.12.0'
