####################################################################################################
# neuropythy/freesurfer/__init__.py
# This file defines the FreeSurfer tools that are available as part of neuropythy.

from .core import (FreeSurferSubject, 
                   find_subject_path, subject_paths, add_subject_path, clear_subject_paths)

def freesurfer_subject(name):
    '''
    freesurfer_subject(name) yields a freesurfer Subject object for the subject with the given name.
    Subjects are cached and not reloaded.
    Note that subects returned by freesurfer_subject() are always persistent Immutable objects; this
    means that you must create a transient version of the subject to modify it via the member
    function sub.transient().
    '''
    import os
    subpath = find_subject_path(name)
    if subpath is None: return None
    fpath = '/' + os.path.relpath(subpath, '/')
    if fpath in freesurfer_subject._cache:
        return freesurfer_subject._cache[fpath]
    else:
        sub = FreeSurferSubject(subpath).persist()
        if isinstance(sub, FreeSurferSubject): freesurfer_subject._cache[fpath] = sub
        return sub
freesurfer_subject._cache = {}
