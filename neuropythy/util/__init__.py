# -*- coding: utf-8 -*-
####################################################################################################
# neuropythy/util/__init__.py
# This file defines the general tools that are available as part of neuropythy.

from .conf     import (config, to_credentials, detect_credentials, load_credentials)
from .core     import (is_tuple, is_list, is_set, is_map, is_str, curry, is_iterable,
                       is_hemi_str, like_hemi_str, to_hemi_str,
                       is_grayheight, like_grayheight, to_grayheight,
                       is_interpolation_method, like_interpolation_method, to_interpolation_method,
                       ObjectWithMetaData, is_metaobj, normalize, denormalize,
                       to_affine, apply_affine,
                       is_dataframe, to_dataframe, dataframe_select,
                       AutoDict, auto_dict,
                       DataStruct, to_data_struct, is_data_struct,
                       library_path, tmpdir, to_pathlist,
                       try_through)
from .labels   import (label_colors, LabelIndex, is_label_index, to_label_index, label_indices)
from .filemap  import (FileMap, file_map, is_file_map, pseudo_path, is_pseudo_path, to_pseudo_path,
                       osf_crawl, url_download)
