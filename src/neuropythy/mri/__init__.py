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

from .core   import (Subject, Cortex, is_subject, is_cortex, to_cortex)
from .images import (to_image_spec, to_image, to_image_header, image_interpolate, image_apply,
                     image_reslice, is_image_spec, image_array_to_spec, image_header_to_spec,
                     image_to_spec, image_copy, image_clear, is_pimage, is_npimage,
                     to_image_type)
from ..util  import (is_image, is_image_header)
