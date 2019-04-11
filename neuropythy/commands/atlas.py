####################################################################################################
# neuropythy/commands/atlas.py
# The code for the function that handle the application of atlases to the cortical surface of a
# subject.
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
from   ..util                       import (curry, library_path, AutoDict)
from   ..util.conf                  import config
from   ..                           import io as nyio

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
@pimms.calc('atlas_map', 'atlas_subject')
def calc_atlases(worklog, atlas_subject_id='fsaverage'):
    '''
    cacl_atlases finds all available atlases in the possible subject directories of the given atlas
    subject.

    In order to be a template, it must either be a collection of files (either mgh/mgz or FreeSurfer
    curv/morph-data files) named as '<hemi>.<template>_<quantity><ending>' such as the files
    'lh.wang2015_mplbl.mgz' and 'rh.wang2015_mplbl.mgz'. They may additionally have a version
    prior to the ending, as in 'lh.benson14_angle.v2_5.mgz'. Files without versions are considered 
    to be of a higher version than all versioned files. All files must be found in the atlas
    subject's surf/ directory; however, all subjects in all FreeSurfer subjects paths with the same
    subject id are searched if the atlas is not found in the atlas subejct's directory.

    Afferent parameters:
      @ atlas_subject_id 
        The FreeSurfer subject name subject path of the subject that is to be used as the atlas
        subject from which the atlas is interpolated. HCP subjects are not currently supported.

    Efferent values:
      @ atlas_map 
        A persistent map whose keys are atlas names, the values of which are themselves persistent
        maps whose keys are the versions of the given atlas (None potentially being included). The
        values of these maps are again maps of hemisphere names then finally of the of the quantity
        names (such as 'eccen' or 'maxprob') to the property vectors imported from the appropriate
        files.
    '''
    try:              sub = freesurfer_subject(atlas_subject_id)
    except Exception: sub = None
    if sub is None:
        try: sub = hcp_subject(atlas_subject_id)
        except Exception: sub = None
    if sub is None: raise ValueError('Could not load atlas subject %s' % atlas_subject_id)
    worklog('Using Atlas subject: %s' % sub.path)
    # Now find the requested atlases
    atlases = AutoDict()
    atlas_patt = r'^([lr]h)\.([^_]+)_([^.]+)(\.(v(\d+(_\d+)*)))?((\.mg[hz])|\.nii(\.gz)?)?$'
    atlas_hemi_ii = 1
    atlas_atls_ii = 2
    atlas_meas_ii = 3
    atlas_vrsn_ii = 6
    libdir = os.path.join(library_path(), 'data')
    for pth in [libdir] + config['freesurfer_subject_paths'] + [sub.path]:
        # see if appropriate files are in this directory
        pth = os.path.join(pth, sub.name, 'surf')
        if not os.path.isdir(pth): continue
        for fl in os.listdir(pth):
            m = re.match(atlas_patt, fl)
            if m is None: continue
            fl = os.path.join(pth, fl)
            (h, atls, meas, vrsn) = [
                m.group(ii) for ii in (atlas_hemi_ii, atlas_atls_ii, atlas_meas_ii, atlas_vrsn_ii)]
            if vrsn is not None: vrsn = tuple([int(s) for s in vrsn.split('_')])
            atlases[atls][vrsn][h][meas] = curry(nyio.load, fl)
    # convert the possible atlas maps into persistent/lazy maps
    atlas_map = pyr.pmap({a:pyr.pmap({v:pyr.pmap({h:pimms.lazy_map(hv)
                                                  for (h,hv) in six.iteritems(vv)})
                                      for (v,vv) in six.iteritems(av)})
                          for (a,av) in six.iteritems(atlases)})
    return {'atlas_map':atlas_map, 'atlas_subject':sub}
@pimms.calc('subject_cortices', 'atlas_cortices')
def calc_cortices(subject, atlas_subject, worklog, hemis=None):
    '''
    calc_cortices extracts the hemisphere objects (of the subject) to which the atlas is being
    applied. By default these are 'lh' and 'rh', but for HCP subjects other hemispheres may be
    desired.

    Afferent parameters:
      @ hemis 
        The hemispheres onto which to put the atlas; this may take a number of forms: 'lh' or 'rh' 
        (or --hemis=lh / --hemis=rh) applies the atlas to the given hemisphere only; otherwise a
        list of hemispheres may be specified as a python object (e.g.
        --hemis='("lh_LR32k","rh_LR32k")') or as a comma or whitespace separated string (such as
        'lh_LR32k,lh_LR59k' or 'rh rh_LR164k'). 'lr', 'both', and 'all' are equivalent to 'lh rh';
        this is the default behavior if hemis is not provided explicitly.
      @ subject 
        The neuropythy subject object onto which the atlas is being projected.
      @ atlas_subject 
        Theneuropythy subject object from which the atlas is being projected.
    '''
    if hemis is None or hemis is Ellipsis: hemis = 'lr'
    if pimms.is_str(hemis):
        if hemis.lower() in ['lr', 'both', 'all']: hemis = ('lh','rh')
        else: hemis = re.split(r'([,;:]|\s)+', hemis)[::2]
    if atlas_subject.name == 'fsaverage_sym':
        hemis = ['rhx' if h == 'rh' else h for h in hemis]
    sctcs = {}
    actcs = {}
    worklog('Preparing Hemispheres...')
    for h in hemis:
        if h not in subject.hemis:
            raise ValueError('Subject %s does not have requested hemi %s' % (subject.name, h))
        sctcs[h] = curry(lambda sub,h: sub.hemis[h], subject,       h)
        h = 'lh' if h == 'rhx' else h
        if h not in atlas_subject.hemis:
            raise ValueError('Atlas subject %s does not have requested hemi %s' % (
                atlas_subject.name, h))
        actcs[h] = curry(lambda sub,h: sub.hemis[h], atlas_subject, h)
    return {'subject_cortices': pimms.lazy_map(sctcs),
            'atlas_cortices':   pimms.lazy_map(actcs)}
@pimms.calc('atlas_properties', 'atlas_version_tags')
def calc_atlas_projections(subject_cortices, atlas_cortices, atlas_map, worklog, atlases=Ellipsis):
    '''
    calc_atlas_projections calculates the lazy map of atlas projections.

    Afferent parameters:
      @ atlases 
        The atlases that should be applied to the subject. This can be specified as a list/tuple of
        atlas names or as a string where the atlas names are separated by whitespace, commas, or
        semicolons. For example, to specify the 'benson14' atlas as well as the 'wang15' atlas, then
        ('benson14', 'wang15'), 'benson14 wang15' or 'benson14,wang15' would all be acceptable. To
        specify an atlas version, separate the atlas-name and the version with a colon (:), such as
        'benson14:2.5'. If no version is provided, then the highest version found is used. If
        atlases is set to None or Ellipsis (the default), this is equivalent to 'benson14,wang15'.

    Efferent values:
      @ atlas_properties 
        The atlas properties is a nested pimms lazy map whose key-path are like those of the
        atlas_map afferent parameter but which contains only those atlases requested via the atlases
        afferent parameter and whose deepest values are interpolated property vectors for the 
        target subject.
      @ atlas_version_tags 
        Each atlas can be specified as <atlas> or <atlas>:<version>; if the version is specified,
        then the version tag string (e.g., '.v1_5') is included in this dictionary; if only <atlas>
        was specified then this string is ''. If <atlas>: is specified, then the version string for
        whichever atlas was used is included.
    '''
    # Parse the atlases argument first:
    if atlases is Ellipsis:   atlases = ('benson14', 'wang15')
    if pimms.is_str(atlases): atlases = tuple(re.split(r'([,;]|\s)+', atlases)[::2])
    def _atlas_to_atlver(atl):
        atl0 = atl
        if not pimms.is_vector(atl):
            if ':' in atl:
                atl = atl.split(':')
                if len(atl) != 2: raise ValueError('Cannot parse atlas spec: %s' % atl0)
            else: atl = [atl, None]
        if len(atl) != 2: raise ValueError('Improperly specified atlas: %s' % atl0)
        if pimms.is_str(atl[1]):
            if len(atl[1]) == 0: atl = (atl[0], None)
            else:
                if atl[1][0] == 'v': atl[1] = atl[1][1:]
                try: atl = (atl[0], tuple([int(x) for x in re.split(r'[-_.]+', atl[1])]))
                except Exception:
                    raise ValueError('Could not parse atlas version string: %s' % atl[1])
        elif pimms.is_int(atl[1]):  atl = (atl[0], (atl[1],))
        elif pimms.is_real(atl[1]): atl = (atl[0], (int(atl[1]), int(10*(atl[1] - int(atl[1]))),))
        elif pimms.is_vector(atl[1], int): atl = (atl[0], tuple(atl[1]))
        elif atl[1] is not None:
            raise ValueError('atlas version must be a string (like "v1_5_1") or a list of ints')
        else: atl = tuple(atl)
        return atl + (atl0,)
    # Okay, let's find these versions of the atlases in the atlas_map...
    worklog('Preparing Atlases...')
    wl = worklog.indent()
    atl_props = AutoDict()
    avt = AutoDict()
    # keyfn is for sorting versions (newest version last)
    keyfn = lambda k:((np.inf,)     if k is None                 else
                      k + (np.inf,) if len(k) == 0 or k[-1] != 0 else 
                      k)
    for (atl,version,atl0) in [_atlas_to_atlver(atl) for atl in atlases]:
        if atl not in atlas_map: raise ValueError('Could not find an atlas named %s' % atl)
        atldat = atlas_map[atl]
        # if the version is None, we pick the highest of the available versions
        if version is None: v = sorted(atldat.keys(), key=keyfn)[-1]
        elif version in atldat: v = version
        else: raise ValueError('Could not find specific version %s of atlas %s' % (version,atl))
        # update the atlas-version-tag data
        wl('Atlas: %s, Version: %s' % (atl, v))
        avt[atl][v] = '' if v is None or ':' not in atl0 else ('.v' + '_'.join(map(str, v)))
        lmaps = atlas_map[atl][v]
        # convert these maps into interpolated properties...
        for (h,hmap) in six.iteritems(lmaps):
            hmap = pimms.lazy_map(
                {m:curry(
                    lambda hmap,h,m: atlas_cortices[h].interpolate(
                        subject_cortices[h],
                        hmap[m]),
                    hmap, h, m)
                 for m in six.iterkeys(hmap)})
            lmaps = lmaps.set(h, hmap)
        # add the lmaps (the persistent/lazy maps for this atlas version) in the atlprops
        atl_props[atl][v] = lmaps
    # That's all; we can return atl_props once we persist it
    return {'atlas_properties':  pimms.persist(atl_props),
            'atlas_version_tags': pimms.persist(avt)}
@pimms.calc('filemap', 'export_all_fn')
def calc_filemap(atlas_properties, subject, atlas_version_tags, worklog,
                 output_path=None, overwrite=False, output_format='mgz', create_directory=False):
    '''
    calc_filemap is a calculator that converts the atlas properties nested-map into a single-depth
    map whose keys are filenames and whose values are the interpolated property data.

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
    fm = AutoDict()
    for (atl,atldat) in six.iteritems(atlas_properties):
        for (ver,verdat) in six.iteritems(atldat):
            vstr = atlas_version_tags[atl][ver]
            for (h,hdat) in six.iteritems(verdat):
                for m in six.iterkeys(hdat):
                    flnm = '%s.%s_%s%s%s' % (h, atl, m, vstr, ending)
                    flnm = os.path.join(output_path, flnm)
                    fm[flnm] = curry(lambda hdat,m: hdat[m], hdat, m)
    # okay, make that a lazy map:
    filemap = pimms.lazy_map(fm)
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
        worklog('Extracting Files...')
        wl = worklog.indent()
        for flnm in six.iterkeys(filemap):
            wl(flnm)
            filenames.append(nyio.save(flnm, filemap[flnm], fmt))
        return filenames
    return {'filemap': filemap, 'export_all_fn': export_all}


atlas_plan_data = pyr.pmap(
    {'init_worklog':calc_worklog,
     'init_subject':calc_subject,
     'init_atlases':calc_atlases,
     'init_cortices':calc_cortices,
     'atlas_properties':calc_atlas_projections,
     'filemap':calc_filemap})
atlas_plan = pimms.plan(atlas_plan_data)

atlas_cmdline_abbrevs = {'output_format':    'f',
                         'atlases':          'a',
                         'overwrite':        'o',
                         'create_directory': 'c',
                         'subject_id':       's',
                         'hemis':            'H',
                         'output_path':      'o',
                         'atlas_subject_id': 'r',
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
    '''SYNTAX:  python -m neuropythy atlas <subject-id>

Neuropythy's atlas command interpolates atlases from the cortical surface of one
subject (usually an average subject such as fsaverage) onto that of another
subject. Most commonly this is used to see the anatomically-based prediction of
an ROI label or a parameter map (such as a retinotopic map) on an individual
subject.

The following optional arguments may be provided:
  --help | -h
    * Prints this help message.
''' + '\n'.join(
    [s for s in [_format_afferent_doc(atlas_plan.afferent_docs[af], atlas_cmdline_abbrevs)
                 for af in atlas_plan.afferents
                 if af in atlas_plan.afferent_docs]
     if len(s) > 0])

def main(*argv):
    '''
    neuropythy.commands.atlas.main() runs the main function of the atlas command in the neuropythy
      library. See the `python -m neuropythy atlas --help` or atlas.info for more information.
    '''
    argv = [aa for arg in argv for aa in (arg if pimms.is_vector(arg) else [arg])]
    imap = pimms.argv_parse(
        atlas_plan, argv,
        arg_abbrevs=atlas_cmdline_abbrevs)
    argv = imap['argv']
    if len(argv) == 0 or '--help' in argv or '-h' in argv:
        print(info)
        return 1
    try: imap['export_all_fn']()
    except Exception as e:
        sys.stderr.write('\nERROR:\n' + str(e) + '\n')
        sys.stderr.flush()
        sys.exit(2)
    return 0
