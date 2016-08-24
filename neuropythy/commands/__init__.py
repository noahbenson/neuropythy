####################################################################################################
# command/__init__.py
# The main function, if neuropythy is invoked directly as command.
# By Noah C. Benson

import os, sys, math
import pysistence

from .register_retinotopy import register_retinotopy_command

# The commands that can be run by main:
commands = pysistence.make_dict(
    register_retinotopy=register_retinotopy_command)
