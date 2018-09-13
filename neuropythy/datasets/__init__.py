####################################################################################################
# neuropythy/datasets/__init__.py
# Datasets for neuropythy.
# by Noah C. Benson

# mainly just to force these to load when datasets is loaded:
from .benson_winawer_2018 import (BensonWinawer2018Dataset)
# import this last so that we get the most updated version of data
from .core import (data, Dataset)


