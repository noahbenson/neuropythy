# __init__.py

'''Tools for analyzing and registering cortical meshes.'''

from freesurfer import (freesurfer_subject,
                        Hemisphere as FreeSurferHemisphere,
                        Subject    as FreeSurferSubject)
from cortex     import (CorticalMesh)
from vision     import (retinotopy_data, empirical_retinotopy_data, predicted_retinotopy_data,
                        register_retinotopy, retinotopy_anchors, V123_model)

# Version information...
__version__ = '0.1.6'

description = 'Integrate Python environment with FreeSurfer and perform mesh registration'
    
