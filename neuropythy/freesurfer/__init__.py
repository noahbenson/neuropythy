####################################################################################################
# neuropythy/freesurfer/__init__.py
# This file defines the FreeSurfer tools that are available as part of neuropythy.

import os
from .subject import (Subject, Hemisphere, 
                      cortex_to_ribbon_map, cortex_to_ribbon, cortex_to_ribbon_map_lines,
                      find_subject_path, subject_paths, add_subject_path)

# how to construct a freesurfer subject:
__freesurfer_subjects = {}
def freesurfer_subject(name):
    '''
    freesurfer_subject(name) yields a freesurfer Subject object for the subject with the given name.
    Subjects are cached and not reloaded.
    Note that subects returned by freesurfer_subject() are always persistent Immutable objects; this
    means that you must create a transient version of the subject to modify it via the member
    function sub.transient().
    '''
    subpath = find_subject_path(name)
    if subpath is None: return None
    fpath = '/' + os.path.relpath(subpath, '/')
    if fpath in __freesurfer_subjects:
        return __freesurfer_subjects[fpath]
    else:
        sub = Subject(subpath).persist()
        if isinstance(sub, Subject): __freesurfer_subjects[fpath] = sub
        return sub
