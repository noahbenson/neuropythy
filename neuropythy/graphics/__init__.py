####################################################################################################
# neuropythy/graphics/__init__.py
# Simple tools for making matplotlib/pyplot graphics with neuropythy.
# By Noah C. Benson

'''
The neuropythy.graphics package contains definitions of the various tools for making plots with
cortical data. The primary entry point is the function cortex_plot.
'''

from .core import (vertex_curvature_color, vertex_weight,
                   vertex_angle, vertex_eccen, vertex_sigma, vertex_varea,
                   vertex_angle_color, vertex_eccen_color, vertex_sigma_color, vertex_varea_color,
                   angle_colors, eccen_colors, sigma_colors, varea_colors,
                   curvature_colors, cortex_plot)
