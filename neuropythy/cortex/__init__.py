####################################################################################################
# neuropythy/cortex/__init__.py
# Data structures and simple tools for dealing with the cortex and cortical data.
# By Noah C. Benson

'''
The neuropythy.cortex package contains definitions of the relevant data-structures and various tools
for interacting with cortical data. The primary data include:

  * Cortex, a class based on neuropythy.geometry.Topology, which tracks the various layers of the 
    cortex in the form of CorticalMesh objects.
  * CorticalMesh, a class based on neuropythy.geometry.Mesh, which tracks a single layer of a the
    cortex.
'''

from .core import (Cortex,
                   vertex_curvature_color, vertex_weight,
                   vertex_angle, vertex_eccen, vertex_angle_color, vertex_eccen_color,
                   angle_colors, eccen_colors, curvature_colors, cortex_plot))
