####################################################################################################
# neuropythy/freesurfer/__init__.py
# This file defines the FreeSurfer tools that are available as part of neuropythy.

from .subject import (Subject, Hemisphere, 
                      cortex_to_ribbon_map, cortex_to_ribbon, cortex_to_ribbon_map_lines)

# how to construct a freesurfer subject:
def freesurfer_subject(name, subjects_dir=None):
    '''
    freesurfer_subject(name) yields a freesurfer Subject object for the subject with the given name.
    If the given subject cannot be found, then a ValueError is raised. The option subjects_dir may
    be given to specify the location of the subject directory.
    '''
    if subjects_dir is None:
        return Subject(name)
    else:
        return Subject(name, subjects_dir=subjects_dir)
