# __init__.py

'''Tools for analyzing and registering cortical meshes.'''

from freesurfer import (freesurfer_subject,
                        Hemisphere as FreeSurferHemisphere,
                        Subject    as FreeSurferSubject)
from cortex     import (CorticalMesh, mesh_smooth)
from vision     import (retinotopy_data, empirical_retinotopy_data, predicted_retinotopy_data,
                        register_retinotopy, retinotopy_anchors, retinotopy_model,
                        neighborhood_cortical_magnification)

# Version information...
__version__ = '0.2.29'

description = 'Integrate Python environment with FreeSurfer and perform mesh registration'


def reload_neuropythy():
    '''
    reload_neuropythy() reloads all of the modules of neuropythy and returns the reloaded
    neuropythy module.
    '''
    import sys
    mdls = ('neuropythy.immutable',
            'neuropythy.util.command',
            'neuropythy.util',
            'neuropythy.java',
            'neuropythy.geometry.util',
            'neuropythy.geometry.mesh',
            'neuropythy.geometry',
            'neuropythy.topology',
            'neuropythy.cortex',
            'neuropythy.freesurfer.subject',
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
    for mdl in mdls:
        if mdl in sys.modules:
            reload(sys.modules[mdl])
    return reload(sys.modules['neuropythy'])
