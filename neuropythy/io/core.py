####################################################################################################
# neuropythy/io/core.py
# This file implements the load and save functions that can be used to read and write various kinds
# of neuroscience data

import numpy      as np
import pyrsistent as pyr
import nibabel    as nib
import os, six, json, gzip, pimms

# The list of import-types we understand
importers = pyr.m()
'''
neuropythy.io.core.importers is the persistent map of file-types that can be imported by neuropythy.
See also neuropythy.io.load.
'''

def guess_import_format(filename, **kwargs):
    '''
    guess_import_format(filename) attempts to guess the file format for the given filename; it does
      this guessing by looking at the file extension and using registered sniff-tests from
      importers. It will not attempt to load the file, so if the extension of the filename is
      missing, it is unlikely that this function will deduce the file-type (though load will often
      succeeed at extracting the data by trying all types exhaustively). If guess_import_format
      cannot deduce the format, it yields None.

    Note that if the filename has an extension that is recognized by neuropythy but the file itself
    is of another format, this function will never look beyond the extention in the filename;
    neither this function nor load perform that level of deduction.
    
    Keyword arguments that are passed to load should also be passed to guess_import_format.
    '''
    # first try file extension
    (_,filename) = os.path.split(filename)
    if '.' in filename:
        fnm = filename.lower()
        fmt = next((k for (k,(_,es,_)) in six.iteritems(importers)
                    if any(fnm.endswith('.' + e) for e in es)),
                   None)
        if fmt: return fmt
    # that didn't work; let's check the sniffers
    for (k,(_,_,sniff)) in six.iteritems(importers):
        try:
            if sniff(filename, **kwargs): return k
        except Exception: pass
    return None
def load(filename, format=None, **kwargs):
    '''
    load(filename) yields the data contained in the file referenced by the given filename in a
      neuropythy or neuropythy-friendly format.
    load(filename, format) specifies that the given format should be used; this should be the name
      of the importer (though a file extension that is recognized also will work).

    Additionally, functions located in load.<format> may be used; so, for example, the following
    are equivalent calls:
      load(filename, 'nifti')
      load.nifti(filename)
    In fact, the load.nifti function is just the nifti importer, so help(load.nifti) will also
    yield documentation for the nifti importer.

    Keyword options may be passed to load; these must match those accepted by the given import
    function.
    '''
    from neuropythy.util import ObjectWithMetaData
    filename = os.path.expanduser(filename)
    if format is None:
        format = guess_import_format(filename, **kwargs)
        if format is None:
            # try formats and see if one works!
            for (k,(f,_,_)) in six.iteritems(importers):
                try:              return f(filename, **kwargs)
                except Exception: pass
            raise ValueError('Could not deduce format of file %s' % filename)
    format = format.lower()
    if format not in importers:
        raise ValueError('Format \'%s\' not recognized by neuropythy' % format)
    (f,_,_) = importers[format]
    obj = f(filename, **kwargs)
    if isinstance(obj, ObjectWithMetaData): return obj.with_meta(source_filename=filename)
    else: return obj
def importer(name, extensions=None, sniff=None):
    '''
    @importer(name) is a decorator that declares that the following function is an file loading
      function that should be registered with the neuropythy load function. See also the
      forget_importer function.

    Any importer function must take, as its first argument, a filename; after that it may take any
    number of keyword arguments, but no other non-keyword arguments. These keyword arguments can be
    passed to the neuropythy load function.
    
    The following options are accepted:
      * extensions (default: None) may be a string or a collection of strings that indicate possible
        file extensions for files of this type.
      * sniff (default: None) may optionally be a function f(s) that yields True when the given
        string s is a filename for a file of this type. If no sniff is given, this type can still
        be detected by running the importer and catching any raised exception.
    '''
    name = name.lower()
    if name in importers:
        raise ValueError('An importer for type %s already exists; see forget_importer' % name)
    if extensions is None:         extensions = ()
    elif pimms.is_str(extensions): (extensions,)
    else:                          extensions = tuple(extensions)
    def _importer(f):
        global importers
        importers = importers.set(name, (f, extensions, sniff))
        setattr(load, name, f)
        return f
    return _importer
def forget_importer(name):
    '''
    forget_importer(name) yields True if an importer of type name was successfully forgotten from
      the neuropythy importers list and false otherwise. This function must be called before an
      importer can be replaced.
    '''
    global importers
    name = name.lower()
    if name in importers:
        importers = importers.discard(name)
        delattr(load, name)
        return True
    else:
        return False



# The list of exporter types we understand
exporters = pyr.m()
'''
neuropythy.io.core.exporters is the persistent map of file-types that can be exported by neuropythy.
See also neuropythy.io.save.
'''

def guess_export_format(filename, data, **kwargs):
    '''
    guess_export_format(filename, data) attempts to guess the export file format for the given
      filename and data (to be exported); it does this guessing by looking at the file extension and
      using registered sniff-tests from exporters.  It will not attempt to save the file, so if the
      extension of the filename is missing, it is less likely that this function will deduce the
      file-type (though save will often succeeed at extracting the data by trying all types
      exhaustively). If guess_export_format cannot deduce the format, it yields None.

    Note that if the filename has an extension that is recognized by neuropythy but the data itself
    is inappropriate for that format, this function will never look beyond the extention in the
    filename; neither this function nor save perform that level of deduction.
    
    Keyword arguments that are passed to save should also be passed to guess_export_format.
    '''
    
    # First try file endings
    (_,filename) = os.path.split(filename)
    fnm = filename.lower()
    fmt = None
    # to make sure we get the most specific ending, sort the exporters by their length
    es = sorted(((k,e) for (k,es) in six.iteritems(exporters) for e in es[1]),
                key=lambda x:-len(x[1]))
    for (k,e) in es:
        if fnm.endswith(('.' + e) if e[0] != '.' else e):
            return k
    # that didn't work; let's check the sniffers
    for (k,(_,_,sniff)) in six.iteritems(exporters):
        try:
            if sniff(filename, data, **kwargs): return k
        except Exception: pass
    return None
def save(filename, data, format=None, **kwargs):
    '''
    save(filename, data) writes the given data to the given filename then yieds that filename.
    save(filename, data, format) specifies that the given format should be used; this should be the
      name of the exporter (though a file extension that is recognized also will work).

    Additionally, functions located in save.<format> may be used; so, for example, the following
    are equivalent calls:
      save(filename, image, 'nifti')
      save.nifti(filename, image)
    In fact, the save.nifti function is just the nifti exporter, so help(save.nifti) will also
    yield documentation for the nifti exporter.

    Keyword options may be passed to save; these must match those accepted by the given export
    function.
    '''
    filename = os.path.expanduser(os.path.expandvars(filename))
    if format is None:
        format = guess_export_format(filename, data, **kwargs)
        if format is None:
            raise ValueError('Could not deduce export format for file %s' % filename)
    else:
        format = format.lower()
        if format not in exporters:
            # it might be an extension
            fmt = next((k for (k,(_,es,_)) in six.iteritems(exporters) if format in es), None)
            if fmt is None:
                # okay, no idea what it is
                raise ValueError('Format \'%s\' not recognized by neuropythy' % format)
            format = fmt
    (f,_,_) = exporters[format]
    return f(filename, data, **kwargs)
def exporter(name, extensions=None, sniff=None):
    '''
    @exporter(name) is a decorator that declares that the following function is an file saveing
      function that should be registered with the neuropythy save function. See also the
      forget_exporter function.

    Any exporter function must take, as its first argument, a filename and, as its second argument,
    the object to be exported; after that it may take any number of keyword arguments, but no other
    non-keyword arguments. These keyword arguments can be passed to the neuropythy save function.
    
    The following options are accepted:
      * extensions (default: None) may be a string or a collection of strings that indicate possible
        file extensions for files of this type.
      * sniff (default: None) may optionally be a function f(s, d) that yields True when the given
        string s is a filename for a file of this type and/or the given object d is an object that
        can be exported as this type. If no sniff is given, this type can still be detected by
        running all exporters exhaustively and catching any raised exceptions; though this may
        result in partial files written to disk, so is not used by save.
    '''
    name = name.lower()
    if name in exporters:
        raise ValueError('An exporter for type %s already exists; use forget_exporter' % name)
    extensions = (extensions,) if pimms.is_str(extensions) else \
                 ()            if extensions is None       else \
                 tuple(extensions)
    def _exporter(f):
        global exporters
        exporters = exporters.set(name, (f, extensions, sniff))
        setattr(save, name, f)
        return f
    return _exporter
def forget_exporter(name):
    '''
    forget_exporter(name) yields True if an exporter of type name was successfully forgotten from
      the neuropythy exporters list and false otherwise. This function must be called before an
      exporter can be replaced.
    '''
    global exporters
    name = name.lower()
    if name in exporters:
        exporters = exporters.discard(name)
        delattr(save, name)
        return True
    else:
        return False

####################################################################################################
# General/universal importers/exporters

# JSON: used with neuropythy.util's normalize/denormalize system
@importer('json', ('json', 'json.gz', 'json.bz2', 'json.lzma'))
def load_json(filename, to='auto'):
    '''
    load_json(filename) yields the object represented by the json file or stream object filename.
    
    The optional argument to may be set to None to indicate that the JSON data should be returned
    verbatim rather than parsed by neuropythy's denormalize system.
    '''
    from neuropythy.util import denormalize as denorm
    if pimms.is_str(filename):
        try:
            with gzip.open(filename, 'rt') as fl: dat = json.load(fl)
        except Exception:
            with open(filename, 'rt') as fl: dat = json.load(fl)
    else:
        dat = json.load(filename)
        filename = '<stream>'
    if to is None: return dat
    elif to == 'auto': return denorm(dat)
    else: raise ValueError('unrecognized to option: %s' % to)
@exporter('json', ('json', 'json.gz', 'json.bz2', 'json.lzma'))
def save_json(filename, obj, normalize=True):
    '''
    save_json(filename, obj) writes the given object to the given filename (or stream) in a
      normalized JSON format.

    The optional argument normalize (default True) may be set to False to prevent the object from
    being run through neuropythy's normalize system.
    '''
    from neuropythy.util import normalize as norm
    dat = norm(obj) if normalize else obj
    if pimms.is_str(filename):
        if any(filename.endswith(s) for s in ('.gz', '.bz2', '.lzma')):
            with gzip.open(filename, 'wt') as fl: json.dump(dat, fl)
        else:
            with open(filename, 'wt') as fl: json.dump(dat, fl)
    else: json.dump(dat, filename)
    return filename
@importer('csv', ('csv', 'csv.gz', 'csv.bz2', 'csv.lzma'))
def load_csv(filename, **kw):
    '''
    load_csv(filename) yields a pandas dataframe of the contents of the CSV file. If pandas cannot
      be loaded, then an error is raised.

    All optional arguments are passed along to the pandas.read_csv function.
    '''
    import pandas
    if any(filename.endswith(s) for s in ('.gz', '.bz2', '.lzma')):
        with gzip.open(filename, 'rt') as fl: data = pandas.read_csv(fl, **kw)
    else:
        with open(filename, 'rt') as fl: data = pandas.read_csv(fl, **kw)
    return data    
@exporter('csv', ('csv', 'csv.gz', 'csv.bz2', 'csv.lzma'))
def save_csv(filename, dat, **kw):
    '''
    save_csv(filename, d) writes a pandas dataframe d to a CSV file with the given name. If pandas
      cannot be loaded, then an error is raised. If d is not a dataframe, to_dataframe() is called
      on it.

    All optional arguments are passed along to the pandas.DataFrame.to_csv function.
    '''
    import pandas
    from neuropythy.util import to_dataframe
    d = to_dataframe(dat)
    if any(filename.endswith(s) for s in ('.gz', '.bz2', '.lzma')):
        with gzip.open(filename, 'wt', newlines='') as fl: d.to_csv(fl, **kw)
    else:
        with open(filename, 'wt') as fl: d.to_csv(fl, **kw)
    return data
@importer('tsv', ('tsv', 'tsv.gz', 'tsv.bz2', 'tsv.lzma'))
def load_tsv(filename, sep='\t', **kw):
    '''
    load_tsv(filename) yields a pandas dataframe of the contents of the TSV file. If pandas cannot
      be loaded, then an error is raised.

    All optional arguments are passed along to the pandas.read_csv function. Note that this function
    is identical to the load_csv() function except that it has a default sep value of '\t' instead
    of ','.
    '''
    import pandas
    if any(filename.endswith(s) for s in ('.gz', '.bz2', '.lzma')):
        with gzip.open(filename, 'rt') as fl: data = pandas.read_csv(fl, sep=sep, **kw)
    else:
        with open(filename, 'rt') as fl: data = pandas.read_csv(fl, sep=sep, **kw)
    return data
@exporter('tsv', ('tsv', 'tsv.gz', 'tsv.bz2', 'tsv.lzma'))
def save_tsv(filename, dat, sep='\t', **kw):
    '''
    save_tsv(filename, d) writes a pandas dataframe d to a TSV file with the given name. If pandas
      cannot be loaded, then an error is raised. If d is not a dataframe, to_dataframe() is called
      on it.

    All optional arguments are passed along to the pandas.DataFrame.to_csv function. Note that this
    function is identical to save_csv() except that it has a default sep value of '\t' instead of
    ','.
    '''
    import pandas
    from neuropythy.util import to_dataframe
    d = to_dataframe(dat)
    if any(filename.endswith(s) for s in ('.gz', '.bz2', '.lzma')):
        with gzip.open(filename, 'wt', newlines='') as fl: d.to_csv(fl, sep=sep, **kw)
    else:
        with open(filename, 'wt') as fl: d.to_csv(fl, sep=sep, **kw)
    return data

# Nifti!
@importer('nifti', ('nii', 'nii.gz', 'nii.bz2', 'nii.lzma'))
def load_nifti(filename, to='auto'):
    '''
    load_nifti(filename) yields the Nifti1Image or Nifti2Image referened by the given filename by
      using the nibabel load function.
    
    The optional argument to may be used to coerce the resulting data to a particular format; the
    following arguments are understood:
      * 'header' will yield just the image header
      * 'data' will yield the image's data-array
      * 'field' will yield a squeezed version of the image's data-array and will raise an error if
        the data object has more than 2 non-unitary dimensions (appropriate for loading surface
        properties stored in image files)
      * 'affine' will yield the image's affine transformation
      * 'image' will yield the raw image object
      * 'auto' is equivalent to 'image' unless the image has no more than 2 non-unitary dimensions,
        in which case it is assumed to be a surface-field and the return value is equivalent to
        the 'field' value.
    '''
    img = nib.load(filename)
    to = to.lower()
    if to == 'image':    return img
    elif to == 'data':   return img.get_data()
    elif to == 'affine': return img.affine
    elif to == 'header': return img.header
    elif to == 'field':
        dat = np.squeeze(np.asarray(img.get_data()))
        if len(dat.shape) > 2:
            raise ValueError('image requested as field has more than 2 non-unitary dimensions')
        return dat
    elif to in ['auto', 'automatic']:
        dims = set(np.shape(img.get_data()))
        if 1 < len(dims) < 4 and 1 in dims:
            return np.squeeze(np.asarray(img.get_data()))
        else:
            return img
    else:
        raise ValueError('unrecognized \'to\' argument \'%s\'' % to)
def to_nifti(obj, like=None, header=None, affine=None, extensions=Ellipsis, version=1):
    '''
    to_nifti(obj) yields a Nifti2Image object that is as equivalent as possible to the given object
      obj. If obj is a Nifti2Image already, then it is returned unmolested; other deduction rules
      are described below.

    The following options are accepted:
      * like (default: None) may be provided to give a guide for the various header- and meta-data
        that is included in the image. If this is a nifti image object, its meta-data are used; if
        this is a subject, then the meta-data are deduced from the subject's voxel and native
        orientation matrices. All other specific options below override anything deduced from the
        like argument.
      * header (default: None) may be a Nifti1 or Niti2 image header to be used as the nifti header
        or to replace the header in a new image.
      * affine (default: None) may specify the affine transform to be given to the image object.
      * extensions (default: Ellipsis) may specify a nifti extensions object that should be included
        in the header. The default value, Ellipsis, indicates that the extensions should not be
        changed, and that None should be used if extensions are not implied in obj (if, for example,
        obj is a data array rather than an image object with a header already.
      * version (default: 2) may be specified as 1 or 2 for a Nifti1Image or Nifti2Image object,
        respectively.
    '''
    from neuropythy.mri import Subject
    obj0 = obj
    # First go from like to explicit versions of affine and header:
    if like is not None:
        if isinstance(like, nib.analyze.AnalyzeHeader) or \
           isinstance(like, nib.freesurfer.mghformat.MGHHeader):
            if header is None: header = like
        elif isinstance(like, nib.analyze.SpatialImage):
            if header is None: header = like.header
            if affine is None: affine = like.affine
        elif isinstance(like, Subject):
            if affine is None: affine = like.voxel_to_native_matrix
        else:
            raise ValueError('Could not interpret like argument with type %s' % type(like))
    # check to make sure that we have to change something:
    elif ((version == 1 and isinstance(obj, nib.nifti1.Nifti1Image)) or
          (version == 2 and isinstance(obj, nib.nifti2.Nifti2Image))):
        if ((header is None or obj.header is header) and
            (extensions is Ellipsis or extensions is obj.header.extensions or
             (extensions is None and len(obj.header.extensions) == 0))):
            return obj
    # okay, now look at the header and affine etc.
    if header is None:
        if isinstance(obj, nib.analyze.SpatialImage):
            header = obj.header
        else:
            header = nib.nifti1.Nifti1Header() if version == 1 else nib.nifti2.Nifti2Header()
    if affine is None:
        if isinstance(obj, nib.analyze.SpatialImage):
            affine = obj.affine
        else:
            affine = np.eye(4)
    if extensions is None:
        extensions = nib.nifti1.Nifti1Extensions()
    # Figure out what the data is
    if isinstance(obj, nib.analyze.SpatialImage):
        obj = obj.dataobj
    elif not pimms.is_nparray(obj):
        obj = np.asarray(obj)
    if len(obj.shape) < 3: obj = np.asarray([[obj]])
    # Okay, make a new object now...
    if version == 1:
        obj = nib.nifti1.Nifti1Image(obj, affine, header)
    elif version == 2:
        obj = nib.nifti2.Nifti2Image(obj, affine, header)
    else:
        raise ValueError('invalid version given (should be 1 or 2): %s' % version)
    # add the extensions if they're needed
    if extensions is not Ellipsis and (len(extensions) > 0 or len(obj.header.extensions) > 0):
        obj.header.extensions = extensions
    # Okay, that's it!
    return obj
@exporter('nifti', ('nii', 'nii.gz'))
def save_nifti(filename, obj, like=None, header=None, affine=None, extensions=Ellipsis, version=1):
    '''
    save_nifti(filename, obj) saves the given object to the given filename in the nifti format and
      returns the filename.

    All options that can be given to the to_nifti function can also be passed to this function; they
    are used to modify the object prior to exporting it.   
    '''
    obj = to_nifti(obj, like=like, header=header, affine=affine, version=version,
                   extensions=extensions)
    obj.to_filename(filename)
    return filename



