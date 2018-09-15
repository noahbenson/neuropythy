####################################################################################################
# neuropythy/datasets/__init__.py
# Datasets for neuropythy.
# by Noah C. Benson

# mainly just to force these to load when datasets is loaded:
from .benson_winawer_2018 import (BensonWinawer2018Dataset)
# TODO: https://openneuro.org/crn/datasets/ds001499/snapshots/1.1.0/download -- add the BOLD5000
#     : dataset to neuropythy (see bold5000.org)
# import this last so that we get the most updated version of data
from .core import (data, Dataset)


