####################################################################################################
# neuropythy/mri/__init__.py
# Data structures and simple tools for dealing with the cortex and cortical data.
# By Noah C. Benson

'''
The neuropythy.mri package contains definitions of the relevant data-structures and various tools
for interacting with cortical data. The primary data include:

  * Cortex, a class based on neuropythy.geometry.Topology, which tracks the various layers of the 
    cortex in the form of CorticalMesh objects.
  * Subject, a class that tracks data connected to an individual subject.
'''

from .core   import (Subject, Cortex, is_subject, is_cortex, to_cortex,
                     cortex_to_image_interpolation)
from .images import (to_image_meta_data, to_image, to_image_header)
