# __init__.py

'''Tools for analyzing and registering cortical meshes.'''

submodules = ('neuropythy.util.core'
              'neuropythy.util',
              'neuropythy.java',
              'neuropythy.geometry.util',
              'neuropythy.geometry.mesh',
              'neuropythy.geometry',
              'neuropythy.mri.core',
              'neuropythy.mri',
              'neuropythy.freesurfer.core',
              'neuropythy.freesurfer',
              'neuropythy.registration.core',
              'neuropythy.registration',
              'neuropythy.vision.models',
              'neuropythy.vision.retinotopy',
              'neuropythy.vision.cmag',
              'neuropythy.vision',
              'neuropythy.commands.surface_to_ribbon',
              'neuropythy.commands.benson14_retinotopy',
              'neuropythy.commands.register_retinotopy',
              'neuropythy.commands')
'''neuropythy.submodules is a tuple of all the sub-modules of neuropythy in a loadable order.'''

def reload_neuropythy():
    '''
    reload_neuropythy() reloads all of the modules of neuropythy and returns the reloaded
    neuropythy module.
    '''
    import sys
    for mdl in submodules:
        if mdl in sys.modules:
            sys.modules[mdl] = reload(sys.modules[mdl])
    return reload(sys.modules['neuropythy'])

from mri        import (Cortex)
from vision     import (retinotopy_data, empirical_retinotopy_data, predicted_retinotopy_data,
                        register_retinotopy, retinotopy_anchors, retinotopy_model,
                        neighborhood_cortical_magnification,
                        as_retinotopy, mesh_retinotopy)


import freesurfer
from   freesurfer import subject as freesurfer_subject


# Version information...
__version__ = '0.4.0'



