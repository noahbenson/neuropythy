####################################################################################################
# __init__.py

'''Tools for analyzing and registering cortical meshes.'''

submodules = ('neuropythy.util.conf',
              'neuropythy.util.core',
              'neuropythy.util',
              'neuropythy.java',
              'neuropythy.io.core',
              'neuropythy.io',
              'neuropythy.geometry.util',
              'neuropythy.geometry.mesh',
              'neuropythy.geometry',
              'neuropythy.mri.core',
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
              'neuropythy.datasets',
              'neuropythy.commands.surface_to_ribbon',
              'neuropythy.commands.benson14_retinotopy',
              'neuropythy.commands.register_retinotopy',
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
        try:    from importlib import reload
        except: from imp import reload
    for mdl in submodules:
        if mdl in sys.modules:
            sys.modules[mdl] = reload(sys.modules[mdl])
    return reload(sys.modules['neuropythy'])

from   .util       import config
from   .io         import (load, save, to_nifti)
from   .mri        import (Cortex, Subject)
from   .vision     import (retinotopy_data, empirical_retinotopy_data, predicted_retinotopy_data,
                           register_retinotopy, retinotopy_anchors, retinotopy_model,
                           neighborhood_cortical_magnification, as_retinotopy, retinotopy_data,
                           retinotopy_comparison)
from   .geometry   import (VertexSet, Mesh, Tesselation, Topology,
                           to_mesh, to_tess, to_property, to_mask)
from   .freesurfer import (subject as freesurfer_subject, to_mgh)
from   .hcp        import (subject as hcp_subject)
from   .datasets   import data
from . import freesurfer
from . import hcp

# things we might want to load but that might fail
try:
    from .graphics import cortex_plot
    from .         import graphics
except: pass

# Version information...
__version__ = '0.7.0'



