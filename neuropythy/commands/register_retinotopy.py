####################################################################################################
# main/register_retinotopy.py
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
from neuropythy.vision import (register_retinotopy, retinotopy_model)


register_retinotopy_help = \
   '''
   The register_retinotopy command can be used to register a subject's
   hemisphere(s) to a model of V1-V3. At least one  subject id (either a freesurfer
   subject name, if SUBJECTS_DIR is set appropriately in the environment, or a path
   to a subject directory) must be given. Registration to a retinotopic model of
   V1-V3 is performed for both hemispheres of all of these subjects.
   In each subject's freesurfer directory, a variety of output data is deposited:
    * surf/lh.retinotopy_sym.sphere.reg
      xhemi/surf/lh.retinotopy_sym.sphere.reg
      These files contain the registrations of the left and right hemispheres to
      the retinotopy model. They are in the same format as Freesurfer's other 
      surface-data files such as surf/lh.white.
    * surf/lh.angle_predict.mgz   surf/rh.angle_predict.mgz
      surf/lh.eccen_predict.mgz   surf/rh.eccen_predict.mgz
      surf/lh.v123roi_predict.mgz surf/rh.v123roi_predict.mgz
      These files contain predictions of polar angle, eccentricity, and visual-area
      label for each hemisphere. The files are mgz format, so contain volumes;
      however, the volumes in each of these files is (1 x 1 x n) where n is the
      number of vertices in the hemisphere's Freesurfer meshes.
    * mri/angle_predict.mgz
      mri/eccen_predict.mgz
      mri/v123roi_predict.mgz
      These contain the data from the above surface data projected into the
      subject's 3D volume. Note that the volumes are oriented like Freesurfer's
      mri/brain.mgz file; if you want to convert this to the orientation of your
      original anatomical scan, use mri_convert:
       > mri_convert -rl mri/rawavg.mgz mri/angle_predict.mgz \\
                     mri/scanner.angle_predict.mgz
   The following options are accepted:
    * --eccen-lh=|-e<file>
      --angle-lh=|-a<file>
      --weight-lh=|-w<file>
      --eccen-rh=|-A<file>
      --angle-rh=|-E<file>
      --weight-rh=|-W<file>
      Each of these arguments specifies the name of a data file to load in as a
      representation of the subject's eccentricity, polar angle, or weight; these
      should be given the names of either an mgh/mgz files whose size is 1 x 1 x n,
      where n is the number of vertices in the hemisphere for the subject, or a
      FreeSurfer curv-style filename with n vertices. By default, files in the
      subject's surf directory that match a template are automatically loaded and
      used. This template is name <hemi>.<tag><name>, optionally ending with .mgz,
      where tag is one of (and in order of preference) 'prf_', 'empirical_',
      'measured_', 'training_', or '', and name is one of 'eccentricity'/'eccen',
      'polar_angle'/'angle', or 'weight'/'variance_explained'/'vexpl'.
    * --cutoff=|-c<value>
      The cutoff value to use for the weight; 0.1 by default. Weights less than
      this will be truncated to 0.
    * -N|--no-partial-correction
      Indicates that partial voluming correction should not be performed.
    * --angle-radians|-r
      This flag specifies that the angle-file only is in radians instead of
      degrees.
    * --eccen-radians|-R
      This flag specifies that the eccen-file only is in radians instead of
      degrees.
    * --mathematical|-m
      This flag specifies that the angle file addresses the visual space in the way
      standard in geometry; i.e., with the right horizontal meridian represented as
      0 and with the upper vertical meridian represented as 90 degrees or pi/4
      instead of the convention in which the opper vertical meridian represented as
      0 and the right horizontal meridian represented as 90 degrees or pi/4
      radians.
    * --edge-strength=|-D<weight>
      --angle-strength=|-T<weight>
      --functional-strength=|-F<weight>
      Each of these specifies the strength of the appropriate potential-field
      component. By default, these are each 1. Note that each field is already
      normalized by the number of components over which it operates; e.g., the edge
      strength is normalized by the number of edges in the mesh.
    * --max-steps=|-s<steps>
      This option specifies the maximum number of steps to run the registration; by
      default this is 2000.
    * --max-step-size=|-S<value>
      This specifies the max step-size for any single vertex; by default this is
      0.05.
    * --prior=|-p<name>
      This specifies the name of the prior registration to use in the fsaverage_sym
      subject; by default this is retinotopy. The prior may be omitted if the value
      "-" or "none" is given.
    * --eccen-tag=|-y<tag>
      --angle-tag=|-t<tag>
      --label-tag=|-l<tag>
      These options specify the output tag to use for the predicted measurement
      that results from the registration. By default, these are
      'eccen_predict', 'angle_predict', and 'v123roi_predict'.
      The output files have the name <hemi>.<tag>.mgz
    * --registration-name=|-u<string>
      This parameter indicates that the registration file, by default named 
      lh.retinotopy_sym.sphere.reg, should instead be named lh.<string>.sphere.reg.
    * --max-output-eccen=|-M<val>
      This specifies the maximum eccentricity to include in the output; there is no
      particular need to limit one's output, but it can be done with this argument.
      By default this is 90.
    * --no-volume-export|-x
      --no-surface-export|-z
      --no-registration-export|-X
      These flags indicate that the various data produced and written to the
      filesystem under normal execution should be suppressed. The volume export
      refers to the predicted volume files exported to the subject's mri directory;
      the registration export refers to the <hemi>.retinotopy_sym.sphere.reg file,
      written to the subject's surf directory, that contains the registered
      coordinates for the subject; and the surface export refers to the
      <hemi>.eccen_predict.mgz and similar files that are written to the
      subject's surf directory.
    * --subjects-dir=|-d
      Specifies additional subject directory search locations (in addition to the
      SUBJECTS_DIR environment variable and the FREESURFER_HOME/subjects
      directories, which are given here in descending search priority) when looking
      for subjects by name. This option cannot be specified multiple times, but it
      may contain : characters to separate directories, as in PATH.
    * --no-overwrite|-n
      This flag indicates that, when writing output files, no file should ever be
      replaced, should it already exist.
    * --
      This token, by itself, indicates that the arguments that remain should not be
      processed as flags or options, even if they begin with a -.
   '''
_retinotopy_parser_instructions = [
    # Flags
    ('h', 'help',                   'help',              False),
    ('v', 'verbose',                'verbose',           False),
    ('r', 'angle-radians',          'angle_radians',     False),
    ('R', 'eccen-radians',          'eccen_radians',     False),
    ('m', 'mathematical',           'angle_math',        False),
    ('x', 'no-volume-export',       'no_vol_export',     False),
    ('z', 'no-surface-export',      'no_surf_export',    False),
    ('X', 'no-registration-export', 'no_reg_export',     False),
    ('n', 'no-overwrite',           'no_overwrite',      False),
    ('N', 'no-partial-correction',  'part_vol_correct',  True),
    # Options                       
    ['e', 'eccen-lh',               'eccen_lh_file',     None],
    ['a', 'angle-lh',               'angle_lh_file',     None],
    ['w', 'weight-lh',              'weight_lh_file',    None],
    ['E', 'eccen-rh',               'eccen_rh_file',     None],
    ['A', 'angle-rh',               'angle_rh_file',     None],
    ['W', 'weight-rh',              'weight_rh_file',    None],
    ['c', 'cutoff',                 'weight_cutoff',     '0.1'],
    ['D', 'edge-strength',          'edge_strength',     '1'],
    ['T', 'angle-strength',         'angle_strength',    '1'],
    ['F', 'functional-strength',    'func_strength',     '1'],
    ['s', 'max-steps',              'max_steps',         '2000'],
    ['S', 'max-step-size',          'max_step_size',     '0.05'],
    ['p', 'prior',                  'prior',             'retinotopy'],
    ['y', 'eccen-tag',              'eccen_tag',         'eccen_predict'],
    ['t', 'angle-tag',              'angle_tag',         'angle_predict'],
    ['l', 'label-tag',              'label_tag',         'v123roi_predict'],
    ['u', 'registration-name',      'registration_name', 'retinotopy_sym'],
    ['M', 'max-output-eccen',       'max_out_eccen',     '90'],
    ['d', 'subjects-dir',           'subjects_dir',      None]]
_retinotopy_parser = CommandLineParser(_retinotopy_parser_instructions)
def _guess_surf_file(fl):
    if len(fl) > 4 and (fl[-4:] == '.mgz' or fl[-4:] == '.mgh'):
        return np.squeeze(np.array(fsmgh.load(fl).dataobj))
    else:
        return fsio.read_morph_data(fl)
def register_retinotopy_command(args):
    '''
    register_retinotopy_command(args) can be given a list of arguments, such as sys.argv[1:]; these
    arguments may include any options and must include at least one subject id. All subjects whose
    ids are given are registered to a retinotopy model, and the resulting registration, as well as
    the predictions made by the model in the registration, are exported.
    '''
    # Parse the arguments
    (args, opts) = _retinotopy_parser(args)
    # First, help?
    if opts['help']:
        print register_retinotopy_help
        return 1
    # and if we are verbose, lets setup a note function
    verbose = opts['verbose']
    def note(s):
        if verbose: print s
        return verbose
    # Add the subjects directory, if there is one
    if 'subjects_dir' in opts and opts['subjects_dir'] is not None:
        add_subject_path(opts['subjects_dir'])
    # Parse the simple numbers
    for o in ['weight_cutoff', 'edge_strength', 'angle_strength', 'func_strength',
              'max_step_size', 'max_out_eccen']:
        opts[o] = float(opts[o])
    opts['max_steps'] = int(opts['max_steps'])
    # These are for now not supported: #TODO
    if opts['angle_math'] or opts['angle_radians'] or opts['eccen_radians']:
        print 'Mathematical angles and angles not in degrees are not yet supported.'
        return 1
    # The remainder of the args can wait for now; walk through the subjects:
    tag_key = {'eccen': 'eccentricity', 'angle': 'polar_angle', 'label': 'visual_area'}
    for subnm in args:
        sub = freesurfer_subject(subnm)
        note('Processing subject: %s' % sub.id)
        # we need to register this subject...
        res = {}
        ow = not opts['no_overwrite']
        for h in ['LH','RH']:
            note('   Processing hemisphere: %s' % h)
            hemi = sub.__getattr__(h)
            # See if we are loading custom values...
            (ang,ecc,wgt) = (None,None,None)
            suffix = '_' + h.lower() + '_file'
            if opts['angle'  + suffix] is not None: ang = _guess_surf_file(opts['angle'  + suffix])
            if opts['eccen'  + suffix] is not None: ecc = _guess_surf_file(opts['eccen'  + suffix])
            if opts['weight' + suffix] is not None: wgt = _guess_surf_file(opts['weight' + suffix])
            # Do the registration
            note('    - Running Registration...')
            res[h] = register_retinotopy(hemi, retinotopy_model(),
                                         polar_angle=ang, eccentricity=ecc, weight=wgt,
                                         weight_cutoff=opts['weight_cutoff'],
                                         partial_voluming_correction=opts['part_vol_correct'],
                                         edge_scale=opts['edge_strength'],
                                         angle_scale=opts['angle_strength'],
                                         functional_scale=opts['func_strength'],
                                         prior=opts['prior'],
                                         max_predicted_eccen=opts['max_out_eccen'],
                                         max_steps=opts['max_steps'],
                                         max_step_size=opts['max_step_size'])
            # Perform the hemi-specific outputs now:
            if not opts['no_reg_export']:
                regnm = '.'.join([h.lower(), opts['registration_name'], 'sphere', 'reg'])
                flnm = (os.path.join(sub.directory, 'surf', regnm) if h == 'LH' else
                        os.path.join(sub.directory, 'xhemi', 'surf', regnm))
                if ow or not os.path.exist(flnm):
                    note('    - Exporting registration file: %s' % flnm)
                    fsio.write_geometry(flnm, res[h].coordinates.T, res[h].faces.T,
                                        'Created by neuropythy (github.com/noahbenson/neuropythy)')
                else:
                    note('    - Skipping registration file: %s (file exists)' % flnm)
            if not opts['no_surf_export']:
                for dim in ['angle', 'eccen', 'label']:
                    flnm = os.path.join(sub.directory, 'surf',
                                        '.'.join([h.lower(), opts[dim + '_tag'], 'mgz']))
                    if ow or not os.path.exist(flnm):
                        note('    - Exporting prediction file: %s' % flnm)
                        img = fsmgh.MGHImage(
                            np.asarray([[res[h].prop(tag_key[dim])]],
                                       dtype=(np.int32 if dim == 'label' else np.float32)),
                            np.eye(4))
                        img.to_filename(flnm)
                    else:
                        note('    - Skipping prediction file: %s (file exists)' % flnm)
        # Do the volume exports here
        if not opts['no_vol_export']:
            note('   Processing volume data...')
            note('    - Calculating cortex-to-ribbon mapping...')
            surf2rib = cortex_to_ribbon_map(sub, hemi=None)
            for dim in ['angle', 'eccen', 'label']:
                flnm = os.path.join(sub.directory, 'mri', opts[dim + '_tag'] + '.mgz')
                if ow or not os.path.exist(flnm):
                    note('    - Generating volume file: %s' % flnm)
                    vol = cortex_to_ribbon(sub,
                                           (res['LH'].prop(tag_key[dim]),
                                            res['RH'].prop(tag_key[dim])),
                                           map=surf2rib,
                                           method=('max'   if dim == 'label' else 'weighted'),
                                           dtype=(np.int32 if dim == 'label' else np.float32))
                    note('    - Exporting volume file: %s' % flnm)
                    vol.to_filename(flnm)
                else:
                    note('    - Skipping volume file: %s (file exists)' % flnm)
        # That is it for this subject!
        note('   Subject %s finished!' % sub.id)
    # And if we made it here, all was successful.
    return 0    

