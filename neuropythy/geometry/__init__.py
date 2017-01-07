####################################################################################################
# neuropythy/geometry/__init__.py
# This file defines common rotation functions that are useful with cortical mesh spheres, such as
# those produced with FreeSurfer.

import numpy as np
import math

from .util import (
    normalize,
    vector_angle_cos,
    vector_angle,
    rotation_matrix_3D,
    rotation_matrix_2D,
    alignment_matrix_3D,
    alignment_matrix_2D,
    line_intersection_2D,
    segment_intersection_2D,
    line_segment_intersection_2D,
    triangle_area,
    cartesian_to_barycentric_2D,
    cartesian_to_barycentric_3D,
    barycentric_to_cartesian,
    triangle_address,
    triangle_unaddress,
    point_in_triangle)
from .mesh import Mesh

