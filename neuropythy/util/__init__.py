####################################################################################################
# neuropythy/util/__init__.py
# This file defines the general tools that are available as part of neuropythy.

from .core     import (ObjectWithMetaData, to_affine, simplex_summation_matrix,
                       simplex_averaging_matrix, is_image, curry,
                       numel, rows, part, hstack, vstack, repmat, replace_close, chop,
                       flatter, flattest,
                       plus, cplus, minus, cminus, times, ctimes,
                       inv, zinv, divide, cdivide, zdivide, czdivide, power, cpower,
                       sine, cosine, tangent, cotangent, secant, cosecant,
                       arcsine, arccosine, arctangent,
                       library_path, address_data, is_address, AutoDict,
                       curve_spline, curve_intersection, CurveSpline,
                       DataStruct, data_struct, tmpdir)
from .conf     import (config, to_credentials, detect_credentials, load_credentials)
from .filemap  import (FileMap, file_map, pseudo_dir)



