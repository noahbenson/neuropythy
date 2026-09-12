####################################################################################################
# neuropythy/io/__init__.py

'''
neuropythy.io is a namespace that contains tools for loading and saving data in neuroscientific
formats. It is intended as an extension of the nibabel libraries in that it is good at
auto-detecting many common formats and data-types and yields data in the neuropythy object system.
'''

from .core import (load, save, importer, exporter, forget_importer, forget_exporter,
                   to_nifti, load_json, save_json, load_csv, save_csv, load_tsv, save_tsv)

