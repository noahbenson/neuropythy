# __init__.py

'''Tools for analyzing and registering cortical meshes.'''

from freesurfer import (freesurfer_subject,
                        Hemisphere as FreeSurferHemisphere,
                        Subject    as FreeSurferSubject)
from cortex     import (CorticalMesh)
from vision     import (retinotopy_data, empirical_retinotopy_data, predicted_retinotopy_data,
                        register_retinotopy, retinotopy_anchors, V123_model)

# Version information...
_version_major = 0
_version_minor = 1
_version_micro = 4
__version__ = "%s.%s.%s" % (_version_major, _version_minor, _version_micro)

description = 'Integrate Python environment with FreeSurfer and perform mesh registration'
    
