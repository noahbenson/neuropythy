####################################################################################################
# main/benson2014_retinotopy.py
# The code for the function that handles the registration of retinotopy
# By Noah C. Benson

from __future__ import print_function

import numpy                        as     np
import scipy                        as     sp
import nibabel                      as     nib
import nibabel.freesurfer.io        as     fsio
import nibabel.freesurfer.mghformat as     fsmgh
import os, sys, six, pimms

from   ..freesurfer                 import (subject, add_subject_path)
from   ..util                       import CommandLineParser
from   ..vision                     import (predict_retinotopy, retinotopy_model, clean_retinotopy)
from   ..                           import io as nyio

info = \
   '''
   The benson14_retinotopy command can be used to project the anatomically defined
   template of retinotopy to a subject's left and right hemisphere(s).  At least
   one subject id (either a freesurfer subject name, if SUBJECTS_DIR is set
   appropriately in the environment, or a path to a subject directory) must be
   given. In each subject's freesurfer directory, a variety of output data is
   deposited:
    * surf/lh.benson14_angle  surf/rh.benson14_angle
      surf/lh.benson14_eccen  surf/rh.benson14_eccen
      surf/lh.benson14_varea  surf/rh.benson14_varea
      surf/lh.benson14_sigma  surf/rh.benson14_sigma
      These files contain predictions of polar angle, eccentricity, visual-area
      label, and pRF radius for each surface vertex in each hemisphere of the
      subject's hemispheres. The files are, by default, in FreeSurfer's curv
      format, but their format can be modified with the --surf-format flag.
    * mri/benson14_angle.mgz
      mri/benson14_eccen.mgz
      mri/benson14_varea.mgz
      mri/benson14_sigma.mgz
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
      --sigma-tag=|-s<tag>
      These options specify the output tag to use for the predicted measurement
      that results from the registration. By default, these are
      'eccen_benson14', 'angle_benson14', 'varea_benson14', and 'sigma_benson14'.
      The output files have the name <hemi>.<tag>.mgz
    * --surf-format=|-o<nifti|nii.gz|nii|mgh|mgz|curv>
      --vol-format=|-v<nifti|nii.gz|nii|mgh|mgz>
      These flags specify what format the output should be in; note that nii.gz
      and nifti are identical; curv is a FreeSurfer curv (AKA morph data) file.
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
      by Benson et al. (2014). The option 'benson14' is also accepted. If the 
    * --reg=|-R<fsaverage|fsaverage_sym>
      Specifies the registration to look for the template in. This is, by default,
      fsaverage, but for the templates aligned to the fsaverage_sym hemisphere,
      this should specify fsaverage_sym.
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
    ('s', 'sigma-tag',              'sigma_tag',         'benson14_sigma'),
    ('d', 'subjects-dir',           'subjects_dir',      None),
    ('t', 'template',               'template',          'benson17'),
    ('o', 'surf-format',            'surf_format',       'curv'),
    ('v', 'vol-format',             'vol_format',        'mgz'),
    ('R', 'reg',                    'registration',      'fsaverage')]
_benson14_parser = CommandLineParser(_benson14_parser_instructions)
def main(*args):
    '''
    benson14_retinotopy.main(args...) runs the benson14_retinotopy command; see 
    benson14_retinotopy.info for more information.
    '''
    # Parse the arguments...
    (args, opts) = _benson14_parser(args)
    # help?
    if opts['help']:
        print(info, file=sys.stdout)
        return 1
    # verbose?
    if opts['verbose']:
        def note(s):
            print(s, file=sys.stdout)
            return True
    else:
        def note(s): return False
    # based on format, how do we export?
    sfmt = opts['surf_format'].lower()
    if sfmt in ['curv', 'auto', 'automatic', 'morph']:
        sfmt = 'freesurfer_morph'
        sext = ''
    elif sfmt == 'nifti':
        sext = '.nii.gz'
    elif sfmt in ['mgh', 'mgz', 'nii', 'nii.gz']:
        sext = '.' + sfmt
    else:
        raise ValueError('Unknown surface format: %s' % opts['surf_format'])
    vfmt = opts['vol_format'].lower()
    if vfmt == 'nifti':
        vext = '.nii.gz'
    elif vfmt in ['mgh', 'mgz', 'nii', 'nii.gz']:
        vext = '.' + vfmt
    else:
        raise ValueError('Unknown volume format: %s' % opts['vol_format'])
    # Add the subjects directory, if there is one
    if 'subjects_dir' in opts and opts['subjects_dir'] is not None:
        add_subject_path(opts['subjects_dir'])
    ow = not opts['no_overwrite']
    nse = opts['no_surf_export']
    nve = opts['no_vol_export']
    tr = {'angle': opts['angle_tag'],
          'eccen': opts['eccen_tag'],
          'varea': opts['label_tag'],
          'sigma': opts['sigma_tag']}
    # okay, now go through the subjects...
    for subnm in args:
        note('Processing subject %s:' % subnm)
        sub = subject(subnm)
        note('   - Interpolating template...')
        (lhdat, rhdat) = predict_retinotopy(sub,
                                            template=opts['template'],
                                            registration=opts['registration'])
        # Export surfaces
        if nse:
            note('   - Skipping surface export.')
        else:
            note('   - Exporting surfaces:')
            for (t,dat) in six.iteritems(lhdat):
                flnm = os.path.join(sub.path, 'surf', 'lh.' + tr[t] + sext)
                if ow or not os.path.exist(flnm):
                    note('    - Exporting LH prediction file: %s' % flnm)
                    nyio.save(flnm, dat, format=sfmt)
                else:
                    note('    - Not overwriting existing file: %s' % flnm)
            for (t,dat) in six.iteritems(rhdat):
                flnm = os.path.join(sub.path, 'surf', 'rh.' + tr[t] + sext)
                if ow or not os.path.exist(flnm):
                    note('    - Exporting RH prediction file: %s' % flnm)
                    nyio.save(flnm, dat, format=sfmt)
                else:
                    note('    - Not overwriting existing file: %s' % flnm)
        # Export volumes
        if nve:
            note('   - Skipping volume export.')
        else:
            note('   - Exporting Volumes:')
            for t in lhdat.keys():
                flnm = os.path.join(sub.path, 'mri', tr[t] + vext)
                if ow or not os.path.exist(flnm):
                    note('    - Preparing volume file: %s' % flnm)
                    dtyp = (np.int32 if t == 'visual_area' else np.float32)
                    vol = sub.cortex_to_image(
                        (lhdat[t], rhdat[t]),
                        method=('nearest' if t == 'visual_area' else 'linear'),
                        dtype=dtyp)
                    note('    - Exporting volume file: %s' % flnm)
                    nyio.save(flnm, vol, like=sub)
                else:
                    note('    - Not overwriting existing file: %s' % flnm)
        note('   Subject %s finished!' % sub.name)
    return 0
            
