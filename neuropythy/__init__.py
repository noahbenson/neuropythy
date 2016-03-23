# __init__.py

'''Tools for analyzing and registering cortical meshes.'''

from .cortex import (cortex_to_mrvolume, CorticalMesh)
import freesurfer
from .registration import mesh_register

# Version information...
_version_major = 0
_version_minor = 1
_version_micro = 0
__version__ = "%s.%s.%s" % (_version_major, _version_minor, _version_micro)

description = 'Integrate Python environment with FreeSurfer and perform mesh registration'

# how to construct a freesurfer subject:
def subject(name, subjects_dir=None):
    '''
    subject(name) yields a freesurfer Subject object for the subject with the given name. If the
    given subject cannot be found, then a ValueError is raised. The option subjects_dir may be
    given to specify the location of the subject directory.
    '''
    if subjects_dir is None:
        return freesurfer.Subject(name)
    else:
        return freesurfer.Subject(name, subjects_dir=subjects_dir)
    
