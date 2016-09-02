####################################################################################################
# command/__init__.py
# The main function, if neuropythy is invoked directly as command.
# By Noah C. Benson

import os, sys, math
import pysistence

from .register_retinotopy import (register_retinotopy_command, register_retinotopy_help)
from .benson14_retinotopy import (benson14_retinotopy_command, benson14_retinotopy_help)
from .surface_to_ribbon   import (surface_to_ribbon_command,   surface_to_ribbon_help)

# The commands that can be run by main:
commands = pysistence.make_dict(
    register_retinotopy=register_retinotopy_command,
    benson14_retinotopy=benson14_retinotopy_command,
    surface_to_ribbon=surface_to_ribbon_command)
