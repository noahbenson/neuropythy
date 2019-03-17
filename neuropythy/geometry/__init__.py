####################################################################################################
# neuropythy/geometry/__init__.py
# This file defines common rotation functions that are useful with cortical mesh spheres, such as
# those produced with FreeSurfer.

'''
The neuropythy.geometry package contains a number of utility functions for calculating 2D and 3D
geometrical values as well as three classes: Mesh, Tesselation, and Topology. The Tesselation
class stores information about triangle meshes---essentially all the information except the 2D or 3D
coordinates of the vertices and that information which requires those coordinates. The Mesh class
simply reifies the Tesselation class with these coordinates and the relevant values that can be
derived from them. Finally, the Topology class tracks a Tesselation object and a set of Meshes that
share that tesselation in common.
'''

from .util import (
    normalize,
    vector_angle_cos,
    vector_angle,
    spherical_distance,
    rotation_matrix_3D,
    rotation_matrix_2D,
    alignment_matrix_3D,
    alignment_matrix_2D,
    point_on_line, point_on_segment, point_in_segment, points_close,
    lines_colinear, segments_colinear, segments_overlapping,
    lines_touch_2D, segments_touch_2D,
    line_intersection_2D,
    segment_intersection_2D,
    line_segment_intersection_2D,
    triangle_area,
    triangle_normal,
    cartesian_to_barycentric_2D,
    cartesian_to_barycentric_3D,
    barycentric_to_cartesian,
    triangle_address,
    triangle_unaddress,
    point_in_triangle,
    point_in_tetrahedron,
    point_in_prism,
    tetrahedral_barycentric_coordinates,
    prism_barycentric_coordinates)
from .mesh import (VertexSet, Tesselation, Mesh, Topology, MapProjection, Path, PathTrace,
                   mesh, is_mesh, is_flatmap,
                   tess, is_tess,
                   topo, is_topo,
                   is_vset, is_path, deduce_chirality,
                   map_projection, is_map_projection,
                   load_map_projection, load_projections_from_path,
                   projections_path, map_projections, 
                   path_trace, is_path_trace, close_path_traces,
                   to_tess, to_mesh, to_property, to_mask, isolines, smooth_lines,
                   to_map_projection, to_flatmap)

