####################################################################################################
# neuropythy/freesurfer/__init__.py
# This file defines the FreeSurfer tools that are available as part of neuropythy.

from .subject import (Subject, Hemisphere, 
                      cortex_to_ribbon_map, cortex_to_ribbon, cortex_to_ribbon_map_lines)

# how to construct a freesurfer subject:
__freesurfer_subjects = {}
def _load_freesurfer_subject(name, subjects_dir=None):
    '''
    _load_freesurfer_subject(name) yields a freesurfer Subject object for the subject with the given
    name. If the given subject cannot be found, then a ValueError is raised. The option subjects_dir
    may be given to specify the location of the subject directory.
    '''
    if subjects_dir is None:
        return Subject(name)
    else:
        return Subject(name, subjects_dir=subjects_dir)
    
def freesurfer_subject(name, subjects_dir=None):
    '''
    freesurfer_subject(name) yields a freesurfer Subject object for the subject with the given name.
    Subjects are cached and not reloaded.
    '''
    if subjects_dir not in __freesurfer_subjects: __freesurfer_subjects[subjects_dir] = {}
    sddat = __freesurfer_subjects[subjects_dir]
    if name in sddat: return sddat[name]
    sub = _load_freesurfer_subject(name, subjects_dir=subjects_dir)
    sddat[name] = sub
    return sub
