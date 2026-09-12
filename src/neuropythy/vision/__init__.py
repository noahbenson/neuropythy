####################################################################################################
# Models and routines used in visual neuroscience.
# By Noah C. Benson

from .models     import (load_fmm_model, visual_area_names,
                         visual_area_numbers, visual_area_field_signs,
                         RetinotopyModel, RetinotopyMeshModel, RegisteredRetinotopyModel,
                         SchiraModel)
from .retinotopy import (empirical_retinotopy_data, predicted_retinotopy_data, retinotopy_data,
                         extract_retinotopy_argument, retinotopy_comparison, to_logeccen,
                         register_retinotopy, retinotopy_registration, from_logeccen,
                         retinotopy_anchors, retinotopy_model, predict_retinotopy,
                         retinotopy_data, as_retinotopy, retinotopic_field_sign,
                         predict_pRF_radius, fit_pRF_radius, occipital_flatmap,
                         clean_retinotopy_potential, clean_retinotopy, visual_isolines,
                         visual_field_mesh, retinotopic_property_aliases,
                         sectors_to_labels, labels_to_sectors, sector_bounds, refit_sectors)
from .cmag       import (mag_data, is_mag_data, neighborhood_cortical_magnification, face_vmag,
                         face_rtcmag, cmag, areal_cmag, field_of_view, isoline_vmag, disk_vmag)

