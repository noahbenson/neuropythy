####################################################################################################
# registration.py
# Tools for registering the cortical surface to a particular potential function
# By Noah C. Benson

'''
The neuropythy.registration package includes tools for performing cortical mesh registration to 2D
models that are projected to the cortical surface. See specifically, the help string for the 
mesh_register function.
'''

from .core       import (mesh_register, java_potential_term)
