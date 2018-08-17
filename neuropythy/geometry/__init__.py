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
from .mesh import (VertexSet, Tesselation, Mesh, Topology, MapProjection,
                   to_tess, to_mesh, to_property, to_mask, tkr_vox2ras)

