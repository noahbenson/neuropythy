####################################################################################################
# command/__init__.py
# The main function, if neuropythy is invoked directly as command.
# By Noah C. Benson

import pyrsistent as _pyr

import register_retinotopy as _reg
import benson14_retinotopy as _b14
import surface_to_image    as _s2i

# The commands that can be run by main:
commands = _pyr.m(
    register_retinotopy = _reg.main,
    benson14_retinotopy = _b14.main,
    surface_to_image    = _s2i.main)

__all__ = ['commands']

