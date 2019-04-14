####################################################################################################
# Models and routines used in visual neuroscience.
# By Noah C. Benson

from .models     import (load_fmm_model, visual_area_names, visual_area_numbers,
                         RetinotopyModel, RetinotopyMeshModel, RegisteredRetinotopyModel,
                         SchiraModel)
from .retinotopy import (empirical_retinotopy_data, predicted_retinotopy_data, retinotopy_data,
                         extract_retinotopy_argument, retinotopy_comparison,
                         register_retinotopy, retinotopy_registration,
                         retinotopy_anchors, retinotopy_model, predict_retinotopy,
                         retinotopy_data, as_retinotopy, retinotopic_field_sign,
                         predict_pRF_radius, fit_pRF_radius, occipital_flatmap,
                         clean_retinotopy_potential, clean_retinotopy, visual_isolines)
from .cmag       import (mag_data, is_mag_data, neighborhood_cortical_magnification,
                         cmag, areal_cmag, field_of_view, isoline_vmag)

