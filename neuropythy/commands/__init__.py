####################################################################################################
# command/__init__.py
# The main function, if neuropythy is invoked directly as command.
# By Noah C. Benson

import os, sys, math
import pysistence

import register_retinotopy
import benson14_retinotopy
import surface_to_image

# The commands that can be run by main:
commands = pysistence.make_dict(
    register_retinotopy = register_retinotopy.main,
    benson14_retinotopy = benson14_retinotopy.main,
    surface_to_image    = surface_to_image.main)
