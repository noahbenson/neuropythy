####################################################################################################
# neuropythy/commands/retinotopy.py
# The code for the function that handle the processing of retinotopy data from the command-line.
# By Noah C. Benson

from __future__ import print_function

import os, sys, six, re, pimms, textwrap, warnings
import numpy                        as     np
import scipy                        as     sp
import nibabel                      as     nib
import nibabel.freesurfer.io        as     fsio
import nibabel.freesurfer.mghformat as     fsmgh
import pyrsistent                   as     pyr

from   ..freesurfer                 import (subject as freesurfer_subject)
from   ..hcp                        import (subject as hcp_subject)
from   ..util                       import (curry, library_path, AutoDict, auto_dict)
from   ..util.conf                  import config
from   ..                           import io as nyio

# Note that much of this code has been stolen from atlas.py; at some point in the future, this
# should all be separated out into a set of classes/functions that handle the common operations.
@pimms.calc('worklog')
def calc_worklog(stdout=Ellipsis, stderr=Ellipsis, verbose=False):
    '''
    calc_worklog constructs the worklog from the stdout, stderr, stdin, and verbose arguments.
    '''
    try: cols = int(os.environ['COLUMNS'])
    except Exception: cols = 80
    return pimms.worklog(columns=cols, stdout=stdout, stderr=stderr, verbose=verbose)
@pimms.calc('subject')
def calc_subject(argv, worklog):
    '''
    calc_subject converts a subject_id into a subject object.

    Afferent parameters:
      @ argv
        The FreeSurfer subject name(s), HCP subject ID(s), or path(s) of the subject(s) to which the
        atlas should be applied.
    '''
    if len(argv) == 0: raise ValueError('No subject-id given')
    elif len(argv) > 1: worklog.warn('WARNING: Unused subject arguments: %s' % (argv[1:],))
    subject_id = argv[0]
    try:
        sub = freesurfer_subject(subject_id)
        if sub is not None:
            worklog('Using FreeSurfer subject: %s' % sub.path)
            return sub
    except Exception: pass
    try:
        sub = hcp_subject(subject_id)
        if sub is not None:
            worklog('Using HCP subject: %s' % sub.path)
            return sub
    except Exception: pass
    raise ValueError('Could not load subject %s' % subject_id)
@pimms.calc('hemisphere_data', 'hemisphere_tags')
def calc_hemispheres(subject, hemispheres='lh,rh'):
    '''
    calc_hemispheres extracts the relevant hemispheres from the subject object.

    Afferent parameters:
      @ hemispheres 
        The names of the hemispheres to be used in the retinotopy calculations. The default value
        is "lh,rh", but other hemisphere names, such as "lh_LR32k,rh_LR32k" (for an HCP-pipeline
        subject) can be given. All hemispheres should be separated by commas. If you want to give
        a hemisphere a particular tag, you can specify it as tag:hemi, e.g.:
        "lh:lh_LR32k,rh:rh_LR32k"; tags are used with other options for specifying data specific
        to a hemisphere.
    '''
    if not hemispheres: hemispheres = 'lh,rh'
    # first, separate by commas
    hs = hemispheres.split(',')
    hemis = {}
    hlist = []
    for h in hs:
        (tag,name) = h.split(':') if ':' in h else (h,h)
        h = subject.hemis.get(name, None)
        if h is None:
            raise ValueError('Give subject does not have a hemisphere named "%s"' % (name,))
        hemis[tag] = h
        hlist.append(tag)
    return {'hemisphere_data': pyr.pmap(hemis),
            'hemisphere_tags': hlist}
@pimms.calc('label_data')
def calc_labels(subject, hemisphere_tags, hemisphere_data, labels=None):
    '''
    calc_labels finds the available label data for the subject on which the retinotopy operations
    are being performed.

    Afferent parameters:
      @ labels 
        The filenames of the files containing label data for the subject's hemispheres. Label data
        can be provided in mgz, annot, or curv files containing visual area labels, one per vertex.
        The labels argument may be specified as a comma-separated list of filenames (in the same
        order as the hemispheres, which are lh then rh by default) or as a single template filename
        that may contain the character * as a stand-in for the hemisphere tag. For example,
        '/data/*.v123_labels.mgz' would look for the file /data/lh.v123_labels.mgz for the 'lh'
        hemisphere and for /data/rh_LR32k.v123_labels.mgz for the 'rh_LR32k' hemisphere.
        Note that labels are not required--without labels, no field-sign minimization is performed,
        so retinotopic cleaning may be less reliable. Note that additionally, labels may be
        preceded by the relevant tag; so instead of '/data/*.v123_labels.mgz' with, as in the 
        example, hemispheres 'lh,rh_LR32k', one could use the arguments
        'lh:/data/lh.v123_labels.mgz,rh:/data/rh_LR32k.v123_labels.mgz' (for labels) and
        'lh,rh:rh_LR32k' for hemispheres.
    '''
    lbls = {}
    # no argument this is fine--no labels are used
    if lbls is None: return {'label_data': pyr.m()}
    if not pimms.is_str(labels): raise ValueError('could not understand non-string labels')
    # first, it might just be a template pattern
    fls = {}
    if '*' in labels:
        sparts = labels.split('*')
        for h in hemisphere_tags:
            flnm = h.join(sparts)
            fls[h] = os.path.expanduser(os.path.expandvars(flnm))
    else:
        # okay, separate commas...
        lsplit = labels.split(',')
        for (k,l) in enumerate(lsplit):
            if ':' in l: (tag,name) = l.split(':')
            elif k < len(hemisphere_tags): (tag,name) = (hemisphere_taks[k],l)
            else: raise ValueError('could not match labels to hemispheres')
            if tag not in hemisphere_data:
                raise ValueError('Tag %s (in labels arg) does not exist' % (tag,))
            fls[tag] = os.path.expanduser(os.path.expandvars(name))
    for (tag,name) in six.iteritems(fls):
        if not os.path.isfile(name):
            raise ValueError('Labels filename %s not found' % (name,))
        hem = hemisphere_data[tag]
        tmp = nyio.load(name)
        if not pimms.is_vector(tmp) or len(tmp) != hem.vertex_count:
            raise ValueError('Labels file %s does not contain label data' % (name,))
        lbls[tag] = np.asarray(tmp)
    return {'label_data': pimms.persist(lbls)}
@pimms.calc('raw_retinotopy')
def calc_retinotopy(hemisphere_data, hemisphere_tags, label_data,
                    angles='*.prf_angle.mgz',
                    eccens='*.prf_eccen.mgz',
                    weights='*.prf_vexpl.mgz'):
    '''
    calc_retinotopy imports the raw retinotopy data for the given subject.

    Afferent parameters:
      @ angles 
        The filenames of the polar-angle files that are needed for each hemisphere. For more 
        information on how these files are specified, see the help text for the labels parameter.
        If angles is not supplied, then the default value is '*.prf_angle.mgz'. Polar angles
        MUST be encoded in clockwise degrees of rotation starting from the positive y-axis.
      @ eccens 
        The filenames of the eccentricity files that are needed for each hemisphere. For more 
        information on how these files are specified, see the help text for the labels parameter.
        If eccens is not supplied, then the default value is '*.prf_eccen.mgz'. Eccentricity
        MUST be encoded in degrees of visual angle.
      @ weights 
        The filenames of the weights (usually fraction of variance explained) files that are needed
        for each hemisphere. For more innformation on how these files are specified, see the help
        text for the labels parameters. If eccens is not supplied, then the default value is
        '*.prf_vexpl.mgz'. Variance explained should be encoded as a fraction with 1 indicating
        100% variance explained.
    '''
    retino = {}
    for (k,val) in zip(['angle', 'eccen', 'weight'], [angles, eccens, weights]):
        if not pimms.is_str(val): raise ValueError('could not understand non-string %ss' % k)
        # first, it might just be a template pattern
        fls = {}
        if '*' in val:
            sparts = val.split('*')
            for h in hemisphere_tags:
                flnm = h.join(sparts)
                fls[h] = os.path.expanduser(os.path.expandvars(flnm))
        else:
            # okay, separate commas...
            lsplit = val.split(',')
            for (kk,l) in enumerate(lsplit):
                if ':' in l: (tag,name) = l.split(':')
                elif kk < len(hemisphere_tags): (tag,name) = (hemisphere_tags[kk],l)
                else: raise ValueError('could not match %ss to hemispheres' % (k,))
                if tag not in hemisphere_data:
                    raise ValueError('Tag %s (in %ss arg) does not exist' % (tag,k))
                fls[tag] = os.path.expanduser(os.path.expandvars(name))
        retino[k] = fls
    # now go through and load them
    res = {}
    for (k,fls) in six.iteritems(retino):
        rr = {}
        for (tag,name) in six.iteritems(fls):
            if not os.path.isfile(name):
                raise ValueError('%ss filename %s not found' % (k,name,))
            hem = hemisphere_data[tag]
            tmp = nyio.load(name)
            if not pimms.is_vector(tmp) or len(tmp) != hem.vertex_count:
                raise ValueError('%ss file %s does not contain label data' % (k,name,))
            rr[tag] = np.asarray(tmp)
        res[k] = rr
    return {'raw_retinotopy': pimms.persist(res)}
@pimms.calc('clean_retinotopy')
def calc_clean_maps(raw_retinotopy, hemisphere_data, label_data, worklog,
                    no_clean=False):
    '''
    calc_clean_maps calculates cleaned retintopic maps.

    Afferent parameters:
      @ no_clean 
        May be set to True to indicate that cleaning of the retinotopic maps should be skipped;
        downstream calculations such as that of cortical magnification will instead be performed on
        the raw and not the clean retinotopy.
    '''
    from neuropythy.vision import (clean_retinotopy, as_retinotopy)
    if no_clean:
        worklog('Skipping retinotopic map cleaning...')
        cl = {h:{'polar_angle': raw_retinotopy['angle'][h],
                 'eccentricity': raw_retinotopy['eccen'][h],
                 'visual_area': label_data[h]}
              for h in six.iterkeys(hemisphere_data)}
        return {'clean_retinotopy': pimms.persist(cl)}
    worklog('Calculating cleaned retinotopic maps...')
    wl = worklog.indent()
    res = {}
    for (h,hem) in six.iteritems(hemisphere_data):
        wl('%s' % h)
        # collect the retinotopy and run the cleaning
        ret = {'polar_angle':        raw_retinotopy['angle'][h],
               'eccentricity':       raw_retinotopy['eccen'][h],
               'variance_explained': raw_retinotopy['weight'][h]}
        if label_data and h in label_data: ret['visual_area'] = label_data[h]
        cl = clean_retinotopy(hem, retinotopy=ret)
        res[h] = {'polar_angle': cl[0], 'eccentricity': cl[1]}
        if label_data and h in label_data: res[h]['visual_area'] = label_data[h]
    # That's all...
    return {'clean_retinotopy': pimms.persist(res)}
@pimms.calc('spotlight_cmag')
def calc_spotlight_cmag(hemisphere_data, clean_retinotopy, worklog,
                        no_spotlight=False, surface='midgray', nnearest=0.1):
    '''
    calc_spotlight_cmag calculates the spotlight-based cortical magnification data for the given
    retinotopic maps, after cleaning if not skipped.

    Afferent parameters:
      @ no_spotlight 
        May be used to indicate that no spotlight search should be performed.
      @ surface 
        Specifies the surface (white, pial, midgray) on which to calculate the cotical
        magnification; by default this is midgray.
      @ nnearest 
        Specifies the number of vertices to include in the nearest-neighbor calculation used for
        spotlight cortical magnification. If not specified, then 10% of vertices are used. This
        should be a number f where 0 <= f < 1 (a fraction of the total vertices) or a number of
        vertices; the former is encouaged.
    '''
    from neuropythy.vision import (areal_cmag, as_retinotopy)
    if no_spotlight:
        worklog('Skipping spotlight cortical magnification calculation.')
        return {'spotlight_cmag': None}
    worklog('Calculating spotlight cortical magnification...')
    wl = worklog.indent()
    res = {}
    for (h,cl) in six.iteritems(clean_retinotopy):
        hem = hemisphere_data[h]
        lbls = cl.get('visual_area', None)
        if lbls is None:
            wl('%s: skipping (no label data provided)' % (h,))
            continue
        else:
            wl('%s' % (h,))
        hh = np.zeros(hem.vertex_count)
        for l in np.unique(lbls):
            if l == 0: continue
            ii = np.where(lbls == l)[0]
            nn = nnearest if nnearest >= 1 else int(np.ceil(nnearest * len(ii)))
            cm = areal_cmag(hem, cl, surface_area=(surface+'_surface_area'), mask=ii, nnearest=nn)
            xy = np.transpose(as_retinotopy(cl, 'geographical'))
            hh[ii] = cm(xy[ii])
        res[h] = hh
    return {'spotlight_cmag': res}
@pimms.calc('radtan_cmag')
def calc_radtan_cmag(hemisphere_data, clean_retinotopy, worklog,
                     no_radtan=False, surface='midgray'):
    '''
    calc_radtan_cmag calculates the radial/tangential cortical magnification data for the given
    retinotopic maps, after cleaning if not skipped.

    Afferent parameters:
      @ no_radtan 
        May be used to indicate that no radial/tangential cortical magnification should be
        calculated.
      @ surface 
        Specifies the surface (white, pial, midgray) on which to calculate the cotical
        magnification; by default this is midgray.
    '''
    from neuropythy.vision import (disk_vmag, as_retinotopy)
    if no_radtan:
        worklog('Skipping radial/tangential cortical magnification calculation.')
        return {'radtan_cmag': None}
    worklog('Calculating radial and tangential cortical magnification...')
    wl = worklog.indent()
    res = {}
    for (h,hem) in six.iteritems(hemisphere_data):
        hem = hemisphere_data[h]
        cl = clean_retinotopy[h]
        lbls = cl.get('visual_area', None)
        if lbls is None:
            wl('%s: WARNING: Calculating without labels' % (h,))
            res[h] = disk_vmag(hem, cl)
        else:
            wl('%s' % (h,))
            hh = np.zeros((hem.vertex_count, 2))
            for l in np.unique(lbls):
                if l == 0: continue
                ii = np.where(lbls == l)[0]
                cm = disk_vmag(hem, cl, surface=surface, mask=ii)
                hh[ii,:] = cm[ii]
        res[h] = hh
    return {'radtan_cmag': res}
@pimms.calc('filemap', 'export_all_fn')
def calc_filemap(subject, worklog, radtan_cmag, spotlight_cmag, clean_retinotopy, no_clean=False,
                 output_path=None, overwrite=False, output_format='mgz', create_directory=False):
    '''
    calc_filemap is a calculator that prepares the calculated cortical magnification data and the
    cleaned retinotopic maps for exporting.

    Afferent parameters
      @ output_path 
        The directory into which the atlas files should be written. If not provided or None then
        uses the subject's surf directory. If this directory doesn't exist, then it uses the
        subject's directory itself.
      @ overwrite 
        Whether to overwrite existing atlas files. If True, then atlas files that already exist will
        be overwritten. If False, then no files are overwritten.
      @ create_directory 
        Whether to create the output path if it doesn't exist. This is False by default.
      @ output_format 
        The desired output format of the files to be written. May be one of the following: 'mgz',
        'mgh', or either 'curv' or 'morph'.

    Efferent values:
      @ filemap 
        A pimms lazy map whose keys are filenames and whose values are interpolated atlas
        properties.
      @ export_all_fn 
        A function of no arguments that, when called, exports all of the files in the filemap to the
        output_path.
    '''
    if output_path is None:
        output_path = os.path.join(subject.path, 'surf')
        if not os.path.isdir(output_path): output_path = subject.path
    output_format = 'mgz' if output_format is None else output_format.lower()
    if output_format.startswith('.'): output_format = output_format[1:]
    (fmt,ending) = (('mgh','.mgz') if output_format == 'mgz' else
                    ('mgh','.mgh') if output_format == 'mgh' else
                    ('freesurfer_morph',''))
    # make the filemap...
    worklog('Preparing Filemap...')
    fm = {}
    for (h,cl) in six.iteritems(clean_retinotopy if not no_clean else {}):
        for (k,val) in six.iteritems(cl):
            flnm = ('%s.clean_%s.' + output_format) % (h,k)
            fm[flnm] = val
    for (h,cm) in six.iteritems(radtan_cmag if radtan_cmag is not None else {}):
        flnm = ('%s.rad_cmag.' % h) + output_format
        fm[flnm] = cm[0]
        flnm = ('%s.tan_cmag.' % h) + output_format
        fm[flnm] = cm[1]
    for (h,cm) in six.iteritems(spotlight_cmag if spotlight_cmag is not None else {}):
        flnm = ('%s.spot_cmag.' % h) + output_format
        fm[flnm] = cm
    # okay, make that a persistent map:
    filemap = pimms.persist(fm)
    output_path = os.path.expanduser(os.path.expandvars(output_path))
    # the function for exporting all properties:
    def export_all():
        '''
        This function will export all files from its associated filemap and return a list of the
        filenames.
        '''
        if not os.path.isdir(output_path):
            if not create_directory:
                raise ValueError('No such path and create_direcotry is False: %s' % output_path)
            os.makedirs(os.path.abspath(output_path), 0o755)
        filenames = []
        worklog('Writing Files...')
        wl = worklog.indent()
        for (flnm,val) in six.iteritems(filemap):
            flnm = os.path.join(output_path, flnm)
            wl(flnm)
            filenames.append(nyio.save(flnm, val, fmt))
        return filenames
    return {'filemap': filemap, 'export_all_fn': export_all}

retinotopy_plan_data = pyr.pmap(
    {'init_worklog':calc_worklog,
     'init_subject':calc_subject,
     'init_hemispheres':calc_hemispheres,
     'init_labels':calc_labels,
     'init_retinotopy':calc_retinotopy,
     'cleaning':calc_clean_maps,
     'spotlight_cmag':calc_spotlight_cmag,
     'radtan_cmag':calc_radtan_cmag,
     'filemap':calc_filemap})
retinotopy_plan = pimms.plan(retinotopy_plan_data)

retinotopy_cmdline_abbrevs = {'output_format':    'f',
                              'overwrite':        'o',
                              'create_directory': 'c',
                              'hemispheres':      'H',
                              'labels':           'l',
                              'angles':           'a',
                              'eccens':           'e',
                              'weights':          'w',
                              'output_path':      'o',
                              'verbose':          'v'}

def _format_afferent_doc(docstr, abbrevs=None, cols=80):
    try:
        (ln1, docs) = docstr.split('\n\n')
    except Exception: return ''
    anm0 = ln1.split(' (')[0]
    anm = anm0.replace('_', '-')
    header = '  --' + anm
    if abbrevs and anm0 in abbrevs: header = header + ' | -' + abbrevs[anm0]
    docs = [
        '\n     '.join(
            textwrap.wrap(
                ' '.join([s.strip() for s in ss.split('\n')[1:]]),
                cols-6))
        for ss in docs.split('\n)')]
    return header + '\n' + ''.join(['   * ' + d for d in docs])

info = \
    '''SYNTAX:  python -m neuropythy retinotopy <subject-id>

Neuropythy's retinotopy command performs cleaning of a set of retinotopic
maps for a subject then calculates cortical magnification.

The following optional arguments may be provided:
  --help | -h
    * Prints this help message.
''' + '\n'.join(
    [s for s in [_format_afferent_doc(retinotopy_plan.afferent_docs[af],
                                      retinotopy_cmdline_abbrevs)
                 for af in retinotopy_plan.afferents
                 if af in retinotopy_plan.afferent_docs]
     if len(s) > 0])

def main(*argv):
    '''
    neuropythy.commands.retinotopy.main() runs the main function of the retinotopy command in the
       neuropythy library. See the `python -m neuropythy retinotopy --help` or atlas.info for more
       information.
    '''
    argv = [aa for arg in argv for aa in (arg if pimms.is_vector(arg) else [arg])]
    imap = pimms.argv_parse(
        retinotopy_plan, argv,
        arg_abbrevs=retinotopy_cmdline_abbrevs)
    argv = imap['argv']
    if len(argv) == 0 or '--help' in argv or '-h' in argv:
        print(info)
        return 1
    try: imap['export_all_fn']()
    except Exception as e:
        raise
        sys.stderr.write('\nERROR:\n' + str(e) + '\n')
        sys.stderr.flush()
        sys.exit(2)
    return 0
