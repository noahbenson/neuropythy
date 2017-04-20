####################################################################################################
# main/benson2014_retinotopy.py
# The code for the function that handles the registration of retinotopy
# By Noah C. Benson

import numpy as np
import scipy as sp
import os, sys
from math import pi
from numbers import Number

import nibabel.freesurfer.io as fsio
import nibabel.freesurfer.mghformat as fsmgh
from pysistence import make_dict

from neuropythy.freesurfer import (freesurfer_subject, add_subject_path,
                                   cortex_to_ribbon, cortex_to_ribbon_map,
                                   Hemisphere)
from neuropythy.util import CommandLineParser
from neuropythy.vision import (predict_retinotopy)

benson14_retinotopy_help = \
   '''
   The benson14_retinotopy command can be used to project the anatomically defined
   template of retinotopy to a subject's left and right hemisphere(s).  At least
   one subject id (either a freesurfer subject name, if SUBJECTS_DIR is set
   appropriately in the environment, or a path to a subject directory) must be
   given. Each subject must have been registered to the fsaverage_sym subject using
   the FreeSurfer surfreg command (after xhemireg for right hemispheres).  In each
   subject's freesurfer directory, a variety of output data is deposited:
    * surf/lh.angle_benson14.mgz   surf/rh.angle_benson14.mgz
      surf/lh.eccen_benson14.mgz   surf/rh.eccen_benson14.mgz
      surf/lh.v123roi_benson14.mgz surf/rh.v123roi_benson14.mgz
      These files contain predictions of polar angle, eccentricity, and visual-area
      label for each hemisphere. The files are mgz format, so contain volumes;
      however, the volumes in each of these files is (1 x 1 x n) where n is the
      number of vertices in the hemisphere's Freesurfer meshes.
    * mri/angle_benson14.mgz
      mri/eccen_benson14.mgz
      mri/v123roi_benson14.mgz
      These contain the data from the above surface data projected into the
      subject's 3D volume. Note that the volumes are oriented like Freesurfer's
      mri/brain.mgz file; if you want to convert this to the orientation of your
      original anatomical scan, use mri_convert:
       > mri_convert -rl mri/rawavg.mgz mri/angle_benson14.mgz \\
                     mri/scanner.angle_benson14.mgz
   The following options are accepted:
    * --eccen-tag=|-y<tag>
      --angle-tag=|-t<tag>
      --label-tag=|-l<tag>
      These options specify the output tag to use for the predicted measurement
      that results from the registration. By default, these are
      'eccen_benson14', 'angle_benson14', and 'v123roi_benson14'.
      The output files have the name <hemi>.<tag>.mgz
    * --no-volume-export|-x
      --no-surface-export|-z
      These flags indicate that the various data produced and written to the
      filesystem under normal execution should be suppressed. The volume export
      refers to the predicted volume files exported to the subject's mri directory
      and the surface export refers to the <hemi>.eccen_benson14.mgz and similar
      files that are written to the subject's surf directory.
    * --subjects-dir=|-d
      Specifies additional subject directory search locations (in addition to the
      SUBJECTS_DIR environment variable and the FREESURFER_HOME/subjects
      directories, which are given here in descending search priority) when looking
      for subjects by name. This option cannot be specified multiple times, but it
      may contain : characters to separate directories, as in PATH.
    * --no-overwrite|-n
      This flag indicates that, when writing output files, no file should ever be
      replaced, should it already exist.
    * --template=|-t
      Specifies the specific template that should be applied. By default this is
      'Benson17', the 2017 version of the template originally described in the paper
      by Benson et al. (2014). The option 'benson14' is also accepted.
    * --
      This token, by itself, indicates that the arguments that remain should not be
      processed as flags or options, even if they begin with a -.
   '''
_benson14_parser_instructions = [
    # Flags
    ('h', 'help',                   'help',              False),
    ('v', 'verbose',                'verbose',           False),
    ('x', 'no-volume-export',       'no_vol_export',     False),
    ('z', 'no-surface-export',      'no_surf_export',    False),
    ('n', 'no-overwrite',           'no_overwrite',      False),
    # Options                       
    ('e', 'eccen-tag',              'eccen_tag',         'benson14_eccen'),
    ('a', 'angle-tag',              'angle_tag',         'benson14_angle'),
    ('l', 'label-tag',              'label_tag',         'benson14_varea'),
    ('d', 'subjects-dir',           'subjects_dir',      None),
    ('t', 'template',               'template',          'benson17')
    ]
_benson14_parser = CommandLineParser(_benson14_parser_instructions)
def benson14_retinotopy_command(*args):
    '''
    benson14_retinotopy_command(args...) runs the benson14_retinotopy command; see 
    benson14_retinotopy_help for mor information.
    '''
    # Parse the arguments...
    (args, opts) = _benson14_parser(args)
    # help?
    if opts['help']:
        print benson14_retinotopy_help
        return 1
    # verbose?
    verbose = opts['verbose']
    def note(s):
        if verbose: print s
        return verbose
    # Add the subjects directory, if there is one
    if 'subjects_dir' in opts and opts['subjects_dir'] is not None:
        add_subject_path(opts['subjects_dir'])
    ow = not opts['no_overwrite']
    nse = opts['no_surf_export']
    nve = opts['no_vol_export']
    tr = {'polar_angle':  opts['angle_tag'],
          'eccentricity': opts['eccen_tag'],
          'visual_area':      opts['label_tag']}
    # okay, now go through the subjects...
    for subnm in args:
        note('Processing subject %s:' % subnm)
        sub = freesurfer_subject(subnm)
        note('   - Interpolating template...')
        (lhdat, rhdat) = predict_retinotopy(sub, template=opts['template'])
        # Export surfaces
        if nse:
            note('   - Skipping surface export.')
        else:
            note('   - Exporting surfaces:')
            for (t,dat) in lhdat.iteritems():
                flnm = os.path.join(sub.directory, 'surf', 'lh.' + tr[t] + '.mgz')
                if ow or not os.path.exist(flnm):
                    note('    - Exporting LH prediction file: %s' % flnm)
                    img = fsmgh.MGHImage(
                        np.asarray([[dat]], dtype=(np.int32 if t == 'visual_area' else np.float32)),
                        np.eye(4))
                    img.to_filename(flnm)
                else:
                    note('    - Not overwriting existing file: %s' % flnm)
            for (t,dat) in rhdat.iteritems():
                flnm = os.path.join(sub.directory, 'surf', 'rh.' + tr[t] + '.mgz')
                if ow or not os.path.exist(flnm):
                    note('    - Exporting RH prediction file: %s' % flnm)
                    img = fsmgh.MGHImage(
                        np.asarray([[dat]], dtype=(np.int32 if t == 'visual_area' else np.float32)),
                        np.eye(4))
                    img.to_filename(flnm)
                else:
                    note('    - Not overwriting existing file: %s' % flnm)
        # Export volumes
        if nve:
            note('   - Skipping volume export.')
        else:
            surf2rib = cortex_to_ribbon_map(sub, hemi=None)
            note('   - Exporting Volumes:')
            for t in lhdat.keys():
                flnm = os.path.join(sub.directory, 'mri', tr[t] + '.mgz')
                if ow or not os.path.exist(flnm):
                    note('    - Preparing volume file: %s' % flnm)
                    vol = cortex_to_ribbon(sub,
                                           (lhdat[t], rhdat[t]),
                                           map=surf2rib,
                                           method=('max' if t == 'visual_area' else 'weighted'),
                                           dtype=(np.int32 if t == 'visual_area' else np.float32))
                    note('    - Exporting volume file: %s' % flnm)
                    vol.to_filename(flnm)
                else:
                    note('    - Not overwriting existing file: %s' % flnm)
        note('   Subject %s finished!' % sub.id)
    return 0
            
