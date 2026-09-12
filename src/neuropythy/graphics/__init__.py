####################################################################################################
# neuropythy/graphics/__init__.py
# Simple tools for making matplotlib/pyplot graphics with neuropythy.
# By Noah C. Benson

'''
The neuropythy.graphics package contains definitions of the various tools for making plots with
cortical data. The primary entry point is the function cortex_plot.
'''

from .core import (
    cmap_curvature,
    cmap_polar_angle_sym, cmap_polar_angle_lh, cmap_polar_angle_rh, cmap_polar_angle,
    cmap_theta_sym, cmap_theta_lh, cmap_theta_rh, cmap_theta,
    cmap_eccentricity, cmap_log_eccentricity, cmap_radius, cmap_log_radius,
    cmap_cmag, cmap_log_cmag, label_cmap,
    vertex_curvature_color, vertex_weight,
    vertex_angle, vertex_eccen, vertex_sigma, vertex_varea,
    vertex_angle_color, vertex_eccen_color, vertex_sigma_color, vertex_varea_color,
    angle_colors, eccen_colors, sigma_colors, radius_colors, varea_colors, to_rgba,
    color_overlap, visual_field_legend, curvature_colors, cortex_plot, cortex_plot_colors,
    ROIDrawer, trace_roi, scale_for_cmap)
