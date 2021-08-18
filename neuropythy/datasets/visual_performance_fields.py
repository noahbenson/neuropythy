####################################################################################################
# Visual Performance Fields Dataset Description
# This file implements a neuropythy dataset object fo the visual performance
# fields project by Benson, Kupers, Barbot, Carrasco, and Winawer.
# By Noah C. Benson

import neuropythy, pimms

from ..util import (config)
from .core  import (Dataset, add_dataset)
from .hcp   import (to_nonempty_path, HCPMetaDataset)

config.declare('visual_performance_fields_path',
               environ_name='VISUAL_PERFORMANCE_FIELDS_PATH',
               default_value=None, filter=to_nonempty_path)
@pimms.immutable
class VisualPerformanceFieldsDataset(HCPMetaDataset):
    '''
    The VisualPerformanceFieldsDataset class declares the basic data structures and the functions
    for generating, loading, or preprocessing the visual performance fields dataset associated
    with the paper by Benson, Kupers, Carrasco, and Winawer (2021) in the journal eLife. The 
    dataset is typically accessed via the neuropythy library:
    ```
    import neuropythy as ny
    ny.data['visual_performance_fields']
    ```
    
    This dataset contains the following members:
      * subject_list is a list of the subject IDs for each HCP subject analyzed in this notebook.
      * roi_angles is a tuple of angle-widths for the distance-based ROIs around the V1/V2
        boundary or around the V1 horizontal meridian.
      * roi_eccens is a tuple of 2-tuples, each of which contains the (min_eccen, max_eccen) of
        one eccentricity ring used in defining the distance-based ROIs.
      * roi_angles_behavior and roi_eccens_behavior store the angles and eccentricities of the
        behavior-matched distance-based ROIs examined in the paper.
      * roi_angles_silva2018 and roi_eccens_silva2018 store the angles and eccentricities of the
        distance-based ROIs that are matched to the Silva et al. (2018) paper.
      * osf_url is the Open Science Framework URL for the project; this isn't an official URL, but
        it has the format "osf://XXXXX/" where the XXXXX is the OSF-id of the project.
      * metadata_table is a pandas dataframe of all the meta-data (i.e., the HCP behavioral data)
        for each subject.
      * gender is a dictionary whose keys are subject IDs and whose values are either 'M' or 'F'.
      * agegroup is a dictionary of age-groups for each subject; the age-group value is the age in
        the middleof the age range for a particular subject. E.g., if  subject u is in the 26-30
        age-group, then agegroup[u] will be 28.
      * inferred_maps is a dictionary mapping subject IDs to retinotopic mapping properties as
        inferred by Bayesian inference (Benson and Winawer, 2018, DOI:10.7554/eLife.40224). This
        dictionary is not used in the paper, but the data were used in analyses from earlier drafts
        of the manuscript, so they are included.
      * subjects is a dictionary of subject objects, each of which contain, as properties or
        meta-data, the sector information for each hemisphere.
      * DROI_table is a dataframe of information about the distance-based ROIs examined in the
        paper.
      * DROI_behavior_table and DROI_silva2018_table are equivalent distance-based ROI dataframes
        that are matched to the Barbot et al. (2020) data or the Silva et al. (2018) data.
      * asymmetry is a nested data-structure that contains informationn about the asymmetry of the
        ROIs examined in the paper.
      * asymmetry_table is a dataframe of information about the asymmetries for each subject.
      * barbot2020_data is a dictionary of data from Barbot et al. (2020) that is compared to the
        surface areas of the distance-based ROIs.
      * abrams2012_data is a dictionary of data from Abrams et al. (2012).
    '''
    
    # Constants / Magic Numbers ####################################################################
    roi_angles = (10, 20, 30, 40, 50)
    roi_angles_behavior = (7.5, 22.5, 37.5, 45)
    roi_angles_silva2018 = (5.625, 16.875,  28.125,  39.375, 45)
    roi_eccens = ((1,2), (2,3), (3,4), (4,5), (5,6), (6,7))
    roi_eccens_behavior = (4, 5)
    roi_eccens_silva2018 = ((1,2), (2,3), (3,4), (4,5), (5,6))

    # The eccentricity ranges we look at:
    # The URL for the OSF page from which we download our data.
    osf_url = 'osf://5gprz/'

    def __init__(self, angles=None, eccens=None, url=None, cache_directory=Ellipsis,
                 metadata_path=None, genetic_path=None, behavioral_path=None,
                 meta_data=None, create_directories=True, create_mode=0o755):
        cdir = cache_directory
        if cdir is Ellipsis:
            cdir = config['visual_performance_fields_path']
        HCPMetaDataset.__init__(self, name='visual_performance_fields',
                                cache_directory=cdir,
                                meta_data=meta_data,
                                create_mode=create_mode,
                                create_directories=create_directories,
                                cache_required=True,
                                metadata_path=metadata_path,
                                genetic_path=genetic_path,
                                behavioral_path=behavioral_path)
        self.url = url
        self.angles = angles
        self.eccens = eccens

    # Administrative Data ##########################################################################
    # Things that are required to load or organize the data but that aren't the data themselves.
    @pimms.param
    def url(url):
        '''
        url is the URL from which the performance-fields data is loaded.
        '''
        if url is None or url is Ellipsis: return VisualPerformanceFieldsDataset.osf_url
        if not pimms.is_str(u): raise ValueError('url must be a string')
        return u
    @pimms.value
    def pseudo_path(url, cache_directory):
        '''
        pseudo_path is the neuropythy psueod-path object responsible for loading the OSF data
        related to the visual performance fields dataset.
        '''
        from neuropythy.util import pseudo_path
        return pseudo_path(url, cache_path=cache_directory).persist()
    inferred_map_files = {'polar_angle':  '%s.%s.inf-MSMAll_angle.native32k.mgz',
                          'eccentricity': '%s.%s.inf-MSMAll_eccen.native32k.mgz',
                          'radius':       '%s.%s.inf-MSMAll_sigma.native32k.mgz',
                          'visual_area':  '%s.%s.inf-MSMAll_varea.native32k.mgz'}
    @pimms.value
    def hcp_data():
        '''
        hcp_data is the HCP Retinotopy dataset that is used in conjunction with the Visual
        Performance Fields dataset.
        '''
        return neuropythy.data['hcp_lines']
    @pimms.value
    def subject_list(hcp_data):
        '''
        subjcet_list is the list of subjects used in the dataset.
        '''
        import numpy as np
        # We want to filter out the subjects without mean sets of hand-drawn lines.
        excl = hcp_data.exclusions
        sids = [sid for sid in hcp_data.subject_list
                if ('mean', sid, 'lh') not in excl
                if ('mean', sid, 'rh') not in excl]
        sids = np.sort(sids)
        return tuple(sids)

    # Supporting Data ##############################################################################
    # Data that is critical for calculating the wedge ROIs but that aren't themselves the data used
    # to make the paper plots.
    @pimms.value
    def inferred_maps(pseudo_path, subject_list):
        '''
        inferred_maps is a nested-dictionary structure containing the retinotopic maps inferred by
        using Bayesian inference on the retinotopic maps of the subjects in the HCP 7T Retinotopy
        Dataset.
        '''
        import os, six
        from neuropythy.util import curry
        from neuropythy import load
        inffiles = VisualPerformanceFieldsDataset.inferred_map_files
        def _load_infmaps(sid,h,patt):
            flnm = pseudo_path.local_path('inferred_maps', patt % (sid,h))
            return load(flnm)
        return pimms.persist(
            {sid: {h: pimms.lmap({('inf_'+k): curry(_load_infmaps, sid, h, v)
                                  for (k,v) in six.iteritems(inffiles)})
                   for h in ['lh','rh']}
             for sid in subject_list})
    @pimms.param
    def angles(angs):
        '''
        angles is the set of polar angle deltas that are calculated. By default this is
          (10, 20, 30, 40, 50).
        '''
        if angs is None or angs is Ellipsis: return VisualPerformanceFieldsDataset.roi_angles
        else: return tuple(angs)
    @pimms.param
    def eccens(eccs):
        '''
        eccens is the set of polar angle deltas that are calculated. By default this is
          ((1,2), (2,3), (3,4), (4,5), (5,6), (6,7))
        '''
        import numpy as np
        if eccs is None or eccs is Ellipsis: eccs = VisualPerformanceFieldsDataset.roi_eccens
        sh = np.shape(eccs)
        if len(sh) == 2 and sh[1] == 2:
            eccs = sorted(eccs, key=lambda x:x[0])
            assert all(u[1] == v[0] for (u,v) in zip(eccs[:-1], eccs[1:])), \
                'eccens must be contiguous tuples'
            return tuple([tuple(u) for u in eccs])
        elif len(sh) == 1:
            eccs = sorted(eccs)
            return tuple(zip(eccs[:-1], eccs[1:]))
        else: raise ValueError("eccens must be a list of 2-tuples or a tuples of values")
    @pimms.value
    def sector_angles(angles):
        '''
        The angles used to calculate the sectors from which cortical magnifications are calculated.
        '''
        import numpy as np
        angles = np.asarray(angles)
        angles = np.concatenate([angles, 90 - angles, 90 + angles, 180 - angles, [0, 90, 180]])
        angles = np.unique(angles)
        angles.setflags(write=False)
        return angles
    @pimms.value
    def sector_eccens(eccens):
        '''
        The eccens used to calculate the sectors from which cortical magnifications are calculated.
        '''
        import numpy as np
        eccens = np.unique(eccens)
        eccens.setflags(write=False)
        return eccens
    @staticmethod
    def _generate_sector(hcp_data, sid, h, sector_angles, sector_eccens):
        import pyrsistent as pyr
        sects = hcp_data.refit_sectors(sid, h, sector_angles, sector_eccens)
        sects = {k+1: pyr.pmap(s) for (k,s) in enumerate(sects)}
        for uu in sects.values():
            for u in uu.values():
                u.setflags(write=False)
        return sects
    @pimms.value
    def sectors(hcp_data, sector_angles, sector_eccens, subject_list):
        '''
        sectors is a nested lazy-map whose first-level keys are subject ids and whose second-level
          keys are either 'lh' or 'rh'. The values are the dictionaries of sectors for each
          hemisphere, divided according to sector_angles and sector_eccens.
        '''
        import pyrsistent as pyr, neuropythy as ny
        f = VisualPerformanceFieldsDataset._generate_sector
        return pyr.pmap(
            {sid: pimms.lmap(
                {h: ny.util.curry(f, hcp_data, sid, h, sector_angles, sector_eccens)
                 for h in ('lh','rh')})
             for sid in subject_list})
    @pimms.value
    def sector_key(sectors, sector_angles, sector_eccens):
        '''
        sector_key is a 2-tuple of persistent maps; the first gives the sector label when queried
          by the sector boundary ids (i.e., 4-tuples of the min and max angles) and the second gives
          the inverse.
        '''
        import pyrsistent as pyr
        key = {}
        ii = 0
        for va in [1,2,3]:
            for (a0,a1) in zip(sector_angles[:-1], sector_angles[1:]):
                for (e0,e1) in zip(sector_eccens[:-1], sector_eccens[1:]):
                    ii += 1
                    key[(va,a0,a1,e0,e1)] = ii
        inv = {ii:k for (k,ii) in key.items()}
        return (pyr.pmap(key), pyr.pmap(inv))
    @staticmethod
    def _generate_sector_labels(hcp_data, sid, h, sectors, key):
        import numpy as np
        hem = hcp_data.subjects[sid].hemis[h]
        sct = sectors[sid][h]
        lbls = np.zeros(hem.vertex_count, dtype=np.int)
        for (tup,k) in key.items():
            if not pimms.is_tuple(tup):
                print(tup, k)
            (va,tup) = (tup[0], tup[1:])
            lbls[sct[va][tup]] = k
        lbls.setflags(write=False)
        return lbls
    def generate_sector_labels(self, sid, h, bins='default'):
        '''
        Generates and yields the sector labels for the given subject id and hemisphere; the optional
          third argument bins can be set to 'behavior' for the behaviorally-matched sector labels or
          to 'silva2018' for the ROIs used in Silva et al. (2018).
        '''
        f = VisualPerformanceFieldsDataset._generate_sector_labels
        (scts,skey) = (
            (self.behavior_sectors, self.behavior_sector_key)   if bins == 'behavior'  else
            (self.silva2018_sectors, self.silva2018_sector_key) if bins == 'silva2018' else
            (self.sectors, self.sector_key)                     if bins == 'default'   else
            (None, None))
        if skey is None: raise ValueError('Unrecognized bins parameter: %s' % (bins,))
        skey = skey[0]
        return f(self.hcp_data, sid, h, scts, skey)
    @staticmethod
    def _sector_labels(hcp_data, pseudo_path, tag, sid, h, sectors, key):
        import neuropythy as ny, numpy as np
        try:                cpath = pseudo_path.local_path('sectors%s' % tag, '%s.%s.mgz' % (h,sid))
        except ValueError:  cpath = None
        if cpath is None:
            return VisualPerformanceFieldsDataset._generate_sector_labels(
                hcp_data, sid, h, sectors, key)
        else:
            lbl = np.array(ny.load(cpath))
            lbl.setflags(write=False)
            return lbl
    @pimms.value
    def sector_labels(hcp_data, pseudo_path, sectors, sector_key):
        '''
        sector_labels is a nested lazy-map whose first-level keys are subject ids and whose
          second-level keys are hemisphere names 'lh' or 'rh'. The values are property maps
          for the associated subject's hemisphere's sector labels. The label key is given
          by sector_key.
        '''
        import pyrsistent as pyr, neuropythy as ny
        f = VisualPerformanceFieldsDataset._sector_labels
        skey = sector_key[0]
        return pyr.pmap(
            {sid: pimms.lmap(
                {h: ny.util.curry(f, hcp_data, pseudo_path, '', sid, h, sectors, skey)
                 for h in ('lh','rh')})
             for sid in sectors.keys()})
    @pimms.value
    def behavior_sectors(hcp_data, subject_list):
        '''
        behavior_sectors is a nested lazy-map like sectors whose first-level keys are subject ids
          and whose second-level keys are either 'lh' or 'rh'. The values are the dictionaries of
          sectors for each hemisphere, divided according to sector_angles and sector_eccens. Unlike
          sectors, behavior_sectors uses the behavior-matched ROIs.
        '''
        import numpy as np, pyrsistent as pyr, neuropythy as ny
        CLS = VisualPerformanceFieldsDataset
        angs = np.array(CLS.roi_angles_behavior)
        sector_angles = np.concatenate([angs, 90 - angs, 90 + angs, 180 - angs, [0, 90, 180]])
        sector_angles = np.unique(sector_angles)
        sector_eccens = np.unique(CLS.roi_eccens_behavior)
        f = lambda sid,h: CLS._generate_sector(hcp_data, sid, h, sector_angles, sector_eccens)
        return pyr.pmap({sid: pimms.lmap({h: ny.util.curry(f, sid, h) for h in ('lh','rh')})
                         for sid in subject_list})
    @pimms.value
    def behavior_sector_key(behavior_sectors):
        '''
        behavior_sector_key is a 2-tuple of persistent maps; the first gives the sector label when
          queried by the sector boundary ids (i.e., 4-tuples of the min and max angles) and the
          second gives the inverse.
        '''
        import pyrsistent as pyr, numpy as np
        CLS = VisualPerformanceFieldsDataset
        angs = np.asarray(CLS.roi_angles_behavior)
        sector_angles = np.concatenate([angs, 90 - angs, 90 + angs, 180 - angs, [0, 90, 180]])
        sector_angles = np.unique(sector_angles)
        sector_eccens = np.unique(CLS.roi_eccens_behavior)
        key = {}
        ii = 0
        for va in [1,2,3]:
            for (a0,a1) in zip(sector_angles[:-1], sector_angles[1:]):
                for (e0,e1) in zip(sector_eccens[:-1], sector_eccens[1:]):
                    ii += 1
                    key[(va,a0,a1,e0,e1)] = ii
        inv = {ii:k for (k,ii) in key.items()}
        return (pyr.pmap(key), pyr.pmap(inv))
    @pimms.value
    def behavior_sector_labels(hcp_data, pseudo_path, behavior_sectors, behavior_sector_key):
        '''
        behavior_sector_labels is a nested lazy-map like sector_labels whose first-level keys
          are subject ids and whose second-level keys are hemisphere names 'lh' or 'rh'. The
          values are property maps for the associated subject's hemisphere's sector labels. The
          label key is given by sector_key.
        '''
        import pyrsistent as pyr, neuropythy as ny
        f = VisualPerformanceFieldsDataset._sector_labels
        skey = behavior_sector_key[0]
        return pyr.pmap(
            {sid: pimms.lmap(
                {h: ny.util.curry(f, hcp_data, pseudo_path, '_behavior',
                                  sid, h, behavior_sectors, skey)
                 for h in ('lh','rh')})
             for sid in behavior_sectors.keys()})
    @staticmethod
    def _generate_hemi(sub, sid, h, infmaps, lbls, bhv_lbls):
        import neuropythy as ny, six
        hem = sub.hemis[h]
        bi = infmaps[sid][h]
        lbls = lbls[sid][h]
        bhv_lbls = bhv_lbls[sid][h]
        ps = {'vpf_sector':lbls, 'vpf_behavior_sector': bhv_lbls}
        for (k,v) in six.iteritems(bi): ps[k] = v
        return hem.with_prop(ps)
    @pimms.value
    def silva2018_sectors(hcp_data, subject_list):
        '''
        silva2018_sectors is a nested lazy-map like sectors whose first-level keys are subject ids
          and whose second-level keys are either 'lh' or 'rh'. The values are the dictionaries of
          sectors for each hemisphere, divided according to sector_angles and sector_eccens. Unlike
          sectors, silva2018_sectors uses the ROIs matched to Silva et al. (2018), Figure 4.
        '''
        import numpy as np, pyrsistent as pyr, neuropythy as ny
        CLS = VisualPerformanceFieldsDataset
        angs = np.array(CLS.roi_angles_silva2018)
        sector_angles = np.concatenate([angs, 90 - angs, 90 + angs, 180 - angs, [0, 90, 180]])
        sector_angles = np.unique(sector_angles)
        sector_eccens = np.unique(CLS.roi_eccens_silva2018)
        f = lambda sid,h: CLS._generate_sector(hcp_data, sid, h, sector_angles, sector_eccens)
        return pyr.pmap({sid: pimms.lmap({h: ny.util.curry(f, sid, h) for h in ('lh','rh')})
                         for sid in subject_list})
    @pimms.value
    def silva2018_sector_key(silva2018_sectors):
        '''
        silva2018_sector_key is a 2-tuple of persistent maps; the first gives the sector label when
          queried by the sector boundary ids (i.e., 4-tuples of the min and max angles) and the
          second gives the inverse.
        '''
        import pyrsistent as pyr, numpy as np
        CLS = VisualPerformanceFieldsDataset
        angs = np.asarray(CLS.roi_angles_silva2018)
        sector_angles = np.concatenate([angs, 90 - angs, 90 + angs, 180 - angs, [0, 90, 180]])
        sector_angles = np.unique(sector_angles)
        sector_eccens = np.unique(CLS.roi_eccens_silva2018)
        key = {}
        ii = 0
        for va in [1,2,3]:
            for (a0,a1) in zip(sector_angles[:-1], sector_angles[1:]):
                for (e0,e1) in zip(sector_eccens[:-1], sector_eccens[1:]):
                    ii += 1
                    key[(va,a0,a1,e0,e1)] = ii
        inv = {ii:k for (k,ii) in key.items()}
        return (pyr.pmap(key), pyr.pmap(inv))
    @pimms.value
    def silva2018_sector_labels(hcp_data, pseudo_path, silva2018_sectors, silva2018_sector_key):
        '''
        silva2018_sector_labels is a nested lazy-map like sector_labels whose first-level keys
          are subject ids and whose second-level keys are hemisphere names 'lh' or 'rh'. The
          values are property maps for the associated subject's hemisphere's sector labels. The
          label key is given by sector_key.
        '''
        import pyrsistent as pyr, neuropythy as ny
        f = VisualPerformanceFieldsDataset._sector_labels
        skey = silva2018_sector_key[0]
        return pyr.pmap(
            {sid: pimms.lmap(
                {h: ny.util.curry(f, hcp_data, pseudo_path, '_silva2018',
                                  sid, h, silva2018_sectors, skey)
                 for h in ('lh','rh')})
             for sid in silva2018_sectors.keys()})
    @staticmethod
    def _generate_subject(hcpdata, sid, infmaps, lbls, bhv_lbls):
        import neuropythy as ny, six
        sub = hcpdata.subjects[sid]
        f = VisualPerformanceFieldsDataset._generate_hemi
        lh = f(sub, sid, 'lh', infmaps, lbls, bhv_lbls)
        rh = f(sub, sid, 'rh', infmaps, lbls, bhv_lbls)
        return sub.with_hemi(lh=lh, rh=rh)
    @pimms.value
    def subjects(inferred_maps, sector_labels, behavior_sector_labels, subject_list, hcp_data):
        '''
        subjects is a dictionary of subject objects for all subjects used in the visual performance
        fields dataset. All subject objects in the subejcts dict include property data on the native
        hemispheres for inferred retinotopic maps and for V1 boundary distances.
        '''
        from neuropythy.util import curry
        f = VisualPerformanceFieldsDataset._generate_subject
        return pimms.lmap(
            {sid: curry(f, hcp_data, sid, inferred_maps, sector_labels, behavior_sector_labels)
             for sid in subject_list})

    # ROI-calculation Functions ####################################################################
    # These are methods that calculate the distance-based ROIs ("DROIs").
    @staticmethod
    def _generate_DROI(sector_labels, sector_key, ref_angle, angle_delta,
                       greater=True, lesser=True, visual_areas=(1,),
                       min_variance_explained=0, surface_area='midgray', eccentricity_range=(1,7)):
        import numpy as np
        ii = []
        skey = sector_key[0]
        if pimms.is_number(visual_areas): visual_areas = (visual_areas,)
        min_DROI_angle = ref_angle - angle_delta if lesser else ref_angle
        max_DROI_angle = ref_angle + angle_delta if greater else ref_angle
        min_DROI_eccen = eccentricity_range[0]
        max_DROI_eccen = eccentricity_range[1]
        for (tup,lbl) in skey.items():
            (va, mnang, mxang, mnecc, mxecc) = tup
            if va not in visual_areas: continue
            if mnang < min_DROI_angle: continue
            if mxang > max_DROI_angle: continue
            if mnecc < min_DROI_eccen: continue
            if mxecc > max_DROI_eccen: continue
            ii.append(np.where(sector_labels == lbl)[0])
        if len(ii) == 0:
            return []
        else:
            ii = np.concatenate(ii)
            return np.unique(ii)
    @staticmethod
    def generate_DROI(self, sid, h, ref_angle, angle_delta, bins='default',
                      greater=True, lesser=True, visual_areas=(1,),
                      min_variance_explained=0, surface_area='midgray', eccentricity_range=(1,7)):
        '''
        Generate a set of distance-based ROIs for the given subject and hemisphere from the given
          set of sector labels and sector key
        '''
        (sct,skey) = (
            (self.behavior_sector_labels,  self.behavior_sector_key)  if bins == 'behavior'  else
            (self.silva2018_sector_labels, self.silva2018_sector_key) if bins == 'silva2018' else
            (self.sector_labels,           self.sector_key)           if bins == 'default'   else
            (None,                         None))
        if skey is None: raise ValueError('Unrecognized bins parameter: %s' % (bins,))
        sct = sct[sid][h]
        return VisualPerformanceFieldsDataset._generate_DROI(
            sub, h, sct, skey,
            greater=greater, lesser=lesser, visual_areas=visual_areas,
            min_variance_explained=min_variance_explained,
            surface_area=surface_area,
            eccentricity_range=eccentricity_range)
    @staticmethod
    def _vertical_DROI_from_ventral_dorsal(vnt, drs):
        '''
        Given vnt and drs return values from the generate_DROI_data() function,
        yields the 'vertical' ROI (the V1 parts of vnt and drs) of the combined ROI.
        '''
        import neuropythy as ny, numpy as np, six
        (iiv, iid) = [np.where(q['visual_area'] == 1)[0] for q in (vnt,drs)]
        if len(iiv) == 0 and len(iid) == 0: return {k:[] for k in six.iterkeys(drs)}
        if len(iiv) == 0: return {k:v[iid] for (k,v) in six.iteritems(drs)}
        if len(iid) == 0: return {k:v[iiv] for (k,v) in six.iteritems(vnt)}
        res = {}
        for (k,v) in six.iteritems(vnt):
            res[k] = np.concatenate([v[iiv], drs[k][iid]])
        return res
    @staticmethod
    def _generate_subject_DROI_boundary_data(sub, h, paradigm, angle_delta,
                                             sector_label, sector_key,
                                             min_variance_explained=0,
                                             surface_area='midgray',
                                             eccentricity_range=(1,7)):
        import neuropythy as ny, numpy as np, copy, six
        CLS = VisualPerformanceFieldsDataset
        paradigm = paradigm.lower()
        if paradigm == 'vertical':
            # This is the weird case: we handle it separately: just run the function
            # for both ventral and dorsal and concatenate the V1 parts
            (vnt,drs) = [CLS._generate_subject_DROI_boundary_data(sub, h, DROIs, para, angle_delta)
                         for para in ['ventral', 'dorsal']]
            return CLS._vertical_DROI_from_ventral_dorsal(vnt, drs)
        # Get the hemisphere.
        hem = sub.hemis[h]
        # Some things depend on the paradigm.
        if paradigm == 'ventral':
            center = 0
            gt = True
            lt = True
            vas = (1,2)
        elif paradigm == 'dorsal':
            center = 180
            gt = True
            lt = True
            vas = (1,2)
        elif paradigm == 'horizontal':
            center = 90
            gt = True
            lt = True
            vas = (1,)
        elif paradigm == 'hventral':
            center = 90
            gt = False
            lt = True
            vas = (1,)
        elif paradigm == 'hdorsal':
            center = 90
            gt = True
            lt = False
            vas = (1,)
        elif paradigm == 'ventral_v1':
            center = 0
            gt = True
            lt = False
            vas = (1,)
        elif paradigm == 'dorsal_v1':
            center = 180
            gt = False
            lt = True
            vas = (1,)
        elif paradigm == 'ventral_v2':
            center = 0
            gt = True
            lt = False
            vas = (2,)
        elif paradigm == 'dorsal_v2':
            center = 180
            gt = False
            lt = True
            vas = (2,)
        else: raise ValueError('unrecognized paradigm: %s' % (paradigm,))
        # Get the indices
        f = VisualPerformanceFieldsDataset._generate_DROI
        ii = f(sector_label, sector_key,
               center, angle_delta,
               greater=gt, lesser=lt,
               visual_areas=vas,
               min_variance_explained=min_variance_explained,
               surface_area=surface_area,
               eccentricity_range=eccentricity_range)
        # Grab the other data:
        surface_area = hem.property('midgray_surface_area')
        white_surface_area = hem.property('white_surface_area')
        pial_surface_area = hem.property('pial_surface_area')
        sa = surface_area[ii]
        wsa = white_surface_area[ii]
        psa = pial_surface_area[ii]
        th = hem.prop('thickness')[ii]
        vl = sa * th
        return {'surface_area_mm2': sa,
                'white_surface_area_mm2': wsa,
                'pial_surface_area_mm2': psa,
                'mean_thickness_mm':th,
                'volume_mm3': vl,
                'indices': ii,
                'visual_area': hem.prop('visual_area')[ii]}
    def generate_subject_DROI_boundary_data(self, sid, h, paradigm, angle_delta,
                                            bins='default',
                                            min_variance_explained=0,
                                            surface_area='midgray',
                                            eccentricity_range=(1,7)):
        '''
        generate_subject_DROI_boundary_data(sid, h, DROIs, paradigm, delta) yields a dict of data
          about the given distance ROI; the ROI is defined by the paradigm, which must be one of
          'vertical', 'horizontal', 'ventral', or 'dorsal', and delta, which is the distance in
          polar angle degrees from the given boundary.
        '''
        f = VisualPerformanceFieldsDataset._generate_subject_DROI_boundary_data
        (sct,skey) = (
            (self.behavior_sector_labels,  self.behavior_sector_key)  if bins == 'behavior'  else
            (self.silva2018_sector_labels, self.silva2018_sector_key) if bins == 'silva2018' else
            (self.sector_labels,           self.sector_key)           if bins == 'default'   else
            (None, None))
        if skey is None: raise ValueError('Unrecognized bins parameter: %s' % (bins,))
        sct = sct[sid][h]
        return f(self.subjects[sid], h, sct, skey, paradigm, angle_delta,
                 min_variance_explained=min_variance_explained,
                 surface_area=surface_area,
                 eccentricity_range=eccentricity_range)
    @staticmethod
    def _generate_subject_DROI_data(sub, h, sector_labels, sector_key, angle_delta,
                                    results='summary',
                                    min_variance_explained=0,
                                    surface_area='midgray',
                                    eccentricity_range=(1,7)):
        import neuropythy as ny, numpy as np, six
        paradigms = ['ventral','dorsal','horizontal','hdorsal','hventral',
                     'ventral_v1','ventral_v2','dorsal_v1','dorsal_v2']
        results = results.lower()
        kw = dict(min_variance_explained=min_variance_explained,
                  surface_area=surface_area,
                  eccentricity_range=eccentricity_range)
        # Reorganize these into the paradigm-based ROIs.
        f = VisualPerformanceFieldsDataset._generate_subject_DROI_boundary_data
        (vnt,drs,hrz,hdrs,hvnt,vnt1,vnt2,drs1,drs2) = [
            f(sub, h, para, angle_delta, sector_labels, sector_key, **kw)
            for para in paradigms]
        # we don't need to run vertical because we can derive it from the other measures:
        ver = VisualPerformanceFieldsDataset._vertical_DROI_from_ventral_dorsal(vnt, drs)
        # depending on the results arg, we return these or their summaries
        res = {'vertical': ver, 'horizontal': hrz, 'ventral': vnt, 'dorsal': drs,
               'hdorsal': hdrs, 'hventral': hvnt, 'ventral_v1': vnt1, 'dorsal_v1': drs1,
               'ventral_v2': vnt2, 'dorsal_v2': drs2}
        if results == 'summary':
            fix = {'surface_area_mm2': np.nansum,
                   'white_surface_area_mm2': np.nansum,
                   'pial_surface_area_mm2': np.nansum,
                   'volume_mm3': np.nansum,
                   'mean_thickness_mm': lambda x: np.nan if len(x) == 0 else np.nanmean(x)}
            return {k: {kk: fix[kk](vv) for (kk,vv) in six.iteritems(v) if kk in fix}
                    for (k,v) in six.iteritems(res)}
        else:
            return res
    def generate_subject_DROI_data(self, sid, h, angle_delta, results='summary',
                                   bins='default',
                                   min_variance_explained=0,
                                   surface_area='midgray',
                                   eccentricity_range=(1,7)):
        '''
        generate_subject_DROI_data(sid, h, angle_delta) yields distance-based ROI data for
          the set of ROIs (ventral, dorsal, horizontal, vertical) for the given
          subject and hemisphere.
        '''
        (sct,skey) = (
            (self.behavior_sector_labels,  self.behavior_sector_key)  if bins == 'behavior'  else
            (self.silva2018_sector_labels, self.silva2018_sector_key) if bins == 'silva2018' else
            (self.sector_labels,           self.sector_key)           if bins == 'default'   else
            (None, None))
        if skey is None: raise ValueError('Unrecognized bins parameter: %s' % (bins,))
        sct = sct[sid][h]
        return VisualPerformanceFieldsDataset._generate_subject_DROI_data(
            self.subjects[sid], h, sct, skey, angle_delta, results=results,
            min_variance_explained=min_variance_explained,
            surface_area=surface_area,
            eccentricity_range=eccentricity_range)
    @staticmethod
    def _generate_subject_DROI_table(subjects, sid, sector_labels, sector_key,
                                     min_variance_explained=0,
                                     surface_area='midgray'):
        import neuropythy as ny, numpy as np, six
        angles = np.unique([k for kk in sector_key[0].keys() for k in kk[1:3]])
        angles = angles[:next(ii+1 for (ii,ang) in enumerate(angles) if ang >= 45)]
        if angles[0] == 0: angles = angles[1:]
        eccens = np.unique([k for kk in sector_key[0].keys() for k in kk[3:]])
        eccens = list(zip(eccens[:-1], eccens[1:]))
        # For each eccentricity range, we'll start by making the DROIs, which will
        # use some sharerd parameters.
        kw = dict(min_variance_explained=min_variance_explained,
                  surface_area=surface_area)
        # We'll buiild up a table.
        tbl = ny.auto_dict(None, [])
        sub = subjects[sid]
        sct = sector_labels[sid]
        CLS = VisualPerformanceFieldsDataset
        for h in ['lh','rh']:
            # go through the eccen ranges and angles:
            for erng in eccens:
                kw['eccentricity_range'] = tuple(erng)
                for ang in angles:
                    # get all the summary data:
                    alldat = CLS._generate_subject_DROI_data(sub, h, sct[h], sector_key, ang, **kw)
                    for (para,dat) in six.iteritems(alldat):
                        # append to the appropriate columns:
                        tbl['sid'].append(sid)
                        tbl['hemisphere'].append(h)
                        tbl['boundary'].append(para)
                        tbl['min_eccentricity_deg'].append(erng[0])
                        tbl['max_eccentricity_deg'].append(erng[1])
                        tbl['angle_delta_deg'].append(ang)
                        for (k,v) in six.iteritems(dat):
                            tbl[k].append(v)
        return ny.to_dataframe(tbl)
    def generate_subject_DROI_table(self, sid, bins='default', min_variance_explained=0,
                                    surface_area='midgray'):
        """
        Calculate the distance-based ROIs for a single subject; indended for use with
        multiprocessing. This function will load the subject's data instead of running
        the calculation if the relevant data-file exists.
        """
        (sctlbl,skey) = (
            (self.behavior_sector_labels,  self.behavior_sector_key)  if bins == 'behavior'  else
            (self.silva2018_sector_labels, self.silva2018_sector_key) if bins == 'silva2018' else
            (self.sector_labels,           self.sector_key)           if bins == 'default'   else
            (None, None))
        if skey is None: raise ValueError('Unrecognized bins parameter: %s' % (bins,))
        f = VisualPerformanceFieldsDataset._generate_subject_DROI_table
        return f(self.subjects, sid, sctlbl, skey,
                 min_variance_explained=min_variance_explained,
                 surface_area=surface_area)
    @staticmethod
    def _generate_subject_DROI_details(subjects, sid, h, sector_labels, sector_key,
                                       eccentricity_range=None,
                                       min_variance_explained=0):
        from neuropythy.util import curry
        import six, pyrsistent as pyr, numpy as np
        paradigm_order = ['dorsal', 'ventral', 'vertical', 'horizontal']
        angles = np.unique([k for kk in sector_key[0].keys() for k in kk[1:3]])
        angles = angles[:next(ii+1 for (ii,ang) in enumerate(angles) if ang >= 45)]
        if eccentricity_range is None:
            e = np.unique([k for kk in sector_key[0].keys() for k in kk[3:]])
            e = list(zip(e[:-1], e[1:]))
        else:
            e = eccentricity_range
        if e is None or e is Ellipsis: e = eccens
        kw = dict(min_variance_explained=min_variance_explained)
        if (pimms.is_list(e) or pimms.is_tuple(e)) and all(pimms.is_tuple(q) for q in e):
            f = VisualPerformanceFieldsDataset._generate_subject_DROI_details
            res = [f(subjects, sid, h, sector_labels, sector_key, eccentricity_range=q, **kw)
                   for q in e]
            q = res[0]
            def _merge(p, a):
                r = {}
                for k in six.iterkeys(q[p][a]):
                    u = [u[p][a][k] for u in res if len(u[p][a][k]) > 0]
                    if len(u) == 0: u = np.asarray([], dtype=np.float)
                    else: u = np.concatenate(u)
                    r[k] = u
                return pyr.pmap(r)
            return pyr.pmap({k: pimms.lmap({a: curry(_merge, k, a) for a in angles})
                             for k in paradigm_order})
        sct = sector_labels[sid][h]
        sub = subjects[sid]
        kw['eccentricity_range'] = e
        kw['results'] = 'all'
        f = VisualPerformanceFieldsDataset._generate_subject_DROI_data
        gendeets = lambda sid, h, k: f(sub, h, sct, sector_key, k, **kw)
        lm0 = pimms.lmap({k: curry(gendeets, sid, h, k) for k in angles})
        pfn = lambda p: pimms.lmap({k:curry(lambda k:lm0[k][p], k) for k in angles})
        return pimms.lmap({p: curry(pfn, p) for p in paradigm_order})
    def generate_subject_DROI_details(self, sid, h, eccentricity_range=None, bins='default',
                                      min_variance_explained=0):
        """
        Calculate the distance-based ROI details for a single subject. Details contain
        similar data as the subject's DROI table, but is in a nested dictionary format.
        """
        f = VisualPerformanceFieldsDataset._generate_subject_DROI_details
        (sctlbl,skey) = (
            (self.behavior_sector_labels,  self.behavior_sector_key)  if bins == 'behavior'  else
            (self.silva2018_sector_labels, self.silva2018_sector_key) if bins == 'silva2018' else
            (self.sector_labels,           self.sector_key)           if bins == 'default'   else
            (None, None))
        if skey is None: raise ValueError('Unrecognized bins parameter: %s' % (bins,))
        return f(self.subjects, sid, h, sctlbl, skey,
                 eccentricity_range=eccentricity_range,
                 min_variance_explained=min_variance_explained)
    def generate_DROI_details(self, eccentricity_range=None, bins='default',
                              min_variance_explained=0):
        '''
        generate_DROI_details() yields a set of lazily computed DROI detailed analyses; these
        analyses are used to generate the DROI table(s).
        '''
        import six
        from neuropythy.util import curry
        f = VisualPerformanceFieldsDataset._generate_subject_DROI_details
        (sctlbl,skey) = (
            (self.behavior_sector_labels,  self.behavior_sector_key)  if bins == 'behavior'  else
            (self.silva2018_sector_labels, self.silva2018_sector_key) if bins == 'silva2018' else
            (self.sector_labels,           self.sector_key)           if bins == 'default'   else
            (None, None))
        if skey is None: raise ValueError('Unrecognized bins parameter: %s' % (bins,))
        m = {
            sid: pimms.lmap(
                {h: curry(f, self.subjects, sid, h, sctlbl, skey,
                          eccentricity_range=eccentricity_range,
                          min_variance_explained=min_variance_explained)
                 for h in ['lh','rh']})
            for sid in six.iterkeys(self.subjects)}
        return pimms.persist(m)
    @staticmethod
    def _generate_DROI_tables_call(tup):
        import os, neuropythy as ny, numpy as np
        # set our niceness!
        os.nice(10)
        # turn off numpy warnings.
        np.seterr(all='ignore')
        flnm = os.path.join(tup[3], '%s.csv' % (tup[0],))
        if os.path.isfile(flnm): return flnm
        f = VisualPerformanceFieldsDataset._generate_subject_DROI_table
        dd = ny.data['visual_performance_fields']
        subjects = dd.subjects
        bins = tup[1]
        (sctlbl,skey) = (
            (dd.behavior_sector_labels,  dd.behavior_sector_key)  if bins == 'behavior'  else
            (dd.silva2018_sector_labels, dd.silva2018_sector_key) if bins == 'silva2018' else
            (dd.sector_labels,           dd.sector_key)           if bins == 'default'   else
            (None, None))
        if skey is None: raise ValueError('Unrecognized bins parameter: %s' % (bins,))
        tbl = f(subjects, tup[0], sctlbl, skey, min_variance_explained=tup[2])
        ny.save(flnm, tbl)
        return flnm
    def generate_DROI_tables(self, bins='default', nprocs=None, printstatus=False,
                             min_variance_explained=0, method=None, tempdir=None):
        '''
        generate_DROI_tables() recalculates the set of distance-based ROIs for each subject
        based on the data in the inferred maps and pRFs.
        '''
        import neuropythy as ny, os, six, pyrsistent as pyr, multiprocessing as mp, numpy as np
        drois = {}
        subject_list = self.subject_list
        nsubs = len(subject_list)
        f = VisualPerformanceFieldsDataset._generate_DROI_tables_call
        if tempdir is None: tdir = ny.util.tmpdir()
        else: tdir = tempdir
        if nprocs is None or nprocs is Ellipsis: nprocs = mp.cpu_count()
        if nprocs < 2: nprocs = 1
        if nprocs > 1:
            for ii in np.arange(0, nsubs, nprocs):
                mx = np.min([len(subject_list), ii + nprocs])
                if printstatus:
                    print("%2d - %3d%%" % ((ii + 1)/nsubs * 100, mx/nsubs * 100))
                with mp.Pool(nprocs) as pool:
                    sids = subject_list[ii:ii+nprocs]
                    tbls = pool.map(f, [(sid, bins, min_variance_explained, tdir) for sid in sids])
                # add these data into the overall roi table
                for (sid,tbl) in zip(sids,tbls):
                    drois[sid] = ny.load(tbl)
        else:
            for (ii,sid) in enumerate(subject_list):
                if printstatus and ii % 10 == 9:
                    print('%3d (%d), %4.1f%%' % (ii, sid, ii/len(subject_list)*100.0))
                drois[sid] = f((sid, bins, min_variance_explained, tdir))
        return pyr.pmap(drois)
    def generate_DROI_table(self):
        '''
        generate_DROI_table() is a method that recalculates the DROI summary table from the
        individual subject DROIs.
        '''
        import pandas
        df = pandas.concat(list(self.DROI_tables.values()))
        df.set_index(['sid','hemisphere'])
        return df
    
    # Distance-based ROIs ##########################################################################
    # The distance-based/wedge ROIs themselves.
    @pimms.value
    def DROI_details(subjects, sector_labels, sector_key):
        '''
        DROI_details is a nested-dictionary structure of the various DROI details of each subject
        and hemisphere.
        '''
        import neuropythy as ny, os, six
        from neuropythy.util import curry
        f = curry(VisualPerformanceFieldsDataset._generate_subject_DROI_details, subjects)
        m = {sid: pimms.lmap({h: curry(f, sid, h, sector_labels, sector_key) for h in ['lh','rh']})
             for sid in six.iterkeys(subjects)}
        return pimms.persist(m)
    @pimms.value
    def DROI_tables(subject_list, pseudo_path):
        '''
        DROI_tables (distance-based regions of interest) is a dictionary of ROIS used in the visual
        performance field project. DROI_tables[sid] is a dataframe of the ROI-data for the subject
        with ID sid.
        '''
        import neuropythy as ny, os, six
        # Load one subject.
        def _load_DROI(sid):
            # get a subject-specific cache_path
            cpath = pseudo_path.local_path('DROIs', '%s.csv' % (sid,))
            return ny.load(cpath)
        return pimms.lmap({sid: ny.util.curry(_load_DROI, sid) for sid in subject_list})
    @pimms.value
    def DROI_table(pseudo_path):
        '''
        DROI_table (distance-based ROI table) is a dataframe summarizing all the data from all the
        hemispheres and all the distance-based wedge ROIs used in the visual performance fields
        project.
        '''
        import neuropythy as ny
        df = ny.load(pseudo_path.local_path('DROI_table.csv'))
        df.set_index(['sid','hemisphere'])
        return df
    @staticmethod
    def generate_DROI_summary(DROI_table, bins='default', exclude_periphery=True):
        '''
        generate_DROI_summary(table) converts the DROI table into a summary.
        '''
        import neuropythy as ny, numpy as np
        if bins == 'behavior':
            angles = VisualPerformanceFieldsDataset.roi_angles_behavior
            eccens = VisualPerformanceFieldsDataset.roi_eccens_behavior
        elif bins == 'silva2018':
            angles = VisualPerformanceFieldsDataset.roi_angles_silva2018
            eccens = VisualPerformanceFieldsDataset.roi_eccens_silva2018
        elif bins == 'default':
            angles = VisualPerformanceFieldsDataset.roi_angles
            eccens = VisualPerformanceFieldsDataset.roi_eccens
        else:
            raise ValueError('Unrecognized bins argument: %s' % (bins,))
        eccens = list(zip(eccens[:-1], eccens[1:]))
        # in eccens, by default, we exclude the peripheral (6-7 degree) eccentricity band.
        if exclude_periphery:
            eccens = [uv for uv in eccens if uv != (6,7)]
        emns = [ee[0] for ee in eccens]
        emxs = [ee[1] for ee in eccens]
        def _dfsel(df, k, ang, emns=[1,2,3,4,5], emxs=[2,3,4,5,6]):
            tbls = [ny.util.dataframe_select(df, angle_delta_deg=ang,
                                             min_eccentricity_deg=mn,
                                             max_eccentricity_deg=mx)
                    for (mn,mx) in zip(emns,emxs)]
            tbls = [tbl[['sid','hemisphere',k]] for tbl in tbls]
            tbl = tbls[0]
            for t in tbls[1:]:
                tt = tbl.merge(t, on=['sid','hemisphere'])
                tt[k] = tt[k+'_x'] + tt[k+'_y']
                tbl = tt[['sid','hemisphere',k]]
            tl = tbl.loc[tbl['hemisphere'] == 'lh']
            tr = tbl.loc[tbl['hemisphere'] == 'rh']
            tt = tl.merge(tr, on='sid')
            tt = tt.sort_values('sid')
            return tt[k+'_x'].values + tt[k+'_y'].values
        dat = {
            para: {
                k: pimms.imm_array([_dfsel(df, k, ang, emns=emns, emxs=emxs) for ang in angles])
                for k in ['surface_area_mm2', 'pial_surface_area_mm2', 'white_surface_area_mm2',
                          'mean_thickness_mm', 'volume_mm3']}
            for para in ['horizontal','vertical','dorsal','ventral','hdorsal','hventral',
                         'dorsal_v1','ventral_v1','dorsal_v2','ventral_v2']
            for df in [ny.util.dataframe_select(DROI_table, boundary=para)]}
        return pimms.persist(dat)
    @pimms.value
    def DROI_summary(DROI_table):
        '''
        DROI_summary (distance-based ROI summary) is a nested-dictionary data structure that
        provides easily-plottable summaries of the DROI_table. A value DROI_summary[b][k][u][s]
        corresponds to the boundary b ('ventral', 'dorsal', 'vertical', or 'horizontal'), the
        key k ('surface_area_mm2', 'mean_thickness_mm', or 'volume_mm3'), angle bin u (where 
        u = 0, 1, 2, 3, 4 indicates 0-10, 10-20, 20-30, 30-40, 40-50 degrees away from the relevant
        boundary), and subject s. Note that subject number is not necessarily consistent across
        boundaries and keys as some subjects have no vertices in small ROIs and thus get excluded
        from the relevant collections. All data are collapsed across eccentricities from 1-6
        degrees.
        '''
        return VisualPerformanceFieldsDataset.generate_DROI_summary(DROI_table)
    @pimms.value
    def DROI_behavior_tables(subject_list, pseudo_path):
        '''
        DROI_behavior_tables (distance-based regions of interest) is a dictionary of ROIS used in
        the visual performance field project. It is similar to DROI_tables, but it uses ROIs that
        are matched to the Barbot et al. (2020) measurements.
        '''
        import neuropythy as ny, os, six
        # Load one subject.
        def _load_DROI(sid):
            # get a subject-specific cache_path
            cpath = pseudo_path.local_path('DROIs_behavior', '%s.csv' % (sid,))
            return ny.load(cpath)
        return pimms.lmap({sid: ny.util.curry(_load_DROI, sid) for sid in subject_list})
    @pimms.value
    def DROI_behavior_table(pseudo_path):
        '''
        DROI_behavior_table (distance-based ROI table) is a dataframe summarizing all the data from
        all the hemispheres and all the distance-based wedge ROIs used in the visual performance fields
        project. Like DROI_table, but using the bejavior-matched DROIs.
        '''
        import neuropythy as ny
        df = ny.load(pseudo_path.local_path('DROI_table_behavior.csv'))
        df.set_index(['sid','hemisphere'])
        return df
    @pimms.value
    def DROI_behavior_summary(DROI_behavior_table):
        '''
        DROI_behavior_summary (distance-based ROI summary) is a nested-dictionary data structure
        like DROI_summary, but that uses the DROIs matched to the behavioral data from Barbot et al.
        (2020).
        '''
        return VisualPerformanceFieldsDataset.generate_DROI_summary(DROI_behavior_table,
                                                                    bins='behavior')
    @pimms.value
    def DROI_silva2018_tables(subject_list, pseudo_path):
        '''
        DROI_behavior_tables (distance-based regions of interest) is a dictionary of ROIS used in
        the visual performance field project. It is similar to DROI_tables, but it uses ROIs that
        are matched to Figure 4 from Silva et al. (2018).
        '''
        import neuropythy as ny, os, six
        # Load one subject.
        def _load_DROI(sid):
            # get a subject-specific cache_path
            cpath = pseudo_path.local_path('DROIs_silva2018', '%s.csv' % (sid,))
            return ny.load(cpath)
        return pimms.lmap({sid: ny.util.curry(_load_DROI, sid) for sid in subject_list})
    @pimms.value
    def DROI_silva2018_table(pseudo_path):
        '''
        DROI_silva2018_table (distance-based ROI table) is a dataframe summarizing all the data from
        all the hemispheres and all the distance-based wedge ROIs used in the visual performance
        fields project. Like DROI_table, but using the DROIs matched to Figure 4 of Silva et al.
        (2018).
        '''
        import neuropythy as ny
        df = ny.load(pseudo_path.local_path('DROI_table_silva2018.csv'))
        df.set_index(['sid','hemisphere'])
        return df
    @pimms.value
    def DROI_silva2018_summary(DROI_silva2018_table):
        '''
        DROI_silva2018_summary (distance-based ROI summary) is a nested-dictionary data structure
        like DROI_summary, but that uses the DROIs matched to the Figure 4 from Silva et al.
        (2018).
        '''
        return VisualPerformanceFieldsDataset.generate_DROI_summary(DROI_silva2018_table,
                                                                    bins='silva2018')
    @pimms.value
    def asymmetry(DROI_summary):
        '''
        asymmetry is a nested dictionary structure containing the surface-area asymmetry estimates
        for each subject. The value asymmetry[k][a][sno] is the percent asymmetry between the axes
        defined by comparison name k ('HMA' for HM:VM asymmetry, 'VMA' for LVM:UVM asymmetry,
        'HVA_cumulative' for cumulative HM:VM asymmetry, or 'VMA_cumulative' for cumulative LVM:UVM
        asymmetry), subject number sno (0-180 for the HCP subject whose ID is subject_list[sno]),
        and angle-distance a (10, 20, 30, 40, or 50 indicating the angle-wedge size in degrees of
        polar angle).

        Asymmetry is defined as (value1 - value2) / mean(value1, value2) where value1 and value2 are
        either the horizontal and vertical ROI surface areas respectively or the lower-vetical
        (dorsal) and upper-vertical (ventral) ROI surface areas respectively. The values reported
        in this data structure are percent asymmetry: difference / mean * 100.
        '''
        import neuropythy as ny, six, numpy as np
        quant = 'surface_area_mm2'
        dat = {}
        for (k,(k1,k2)) in zip(['HVA','VMA'], [('horizontal','vertical'), ('dorsal','ventral')]):
            for iscum in [True,False]:
                # Grab and prep the data.
                ys1 = np.asarray(DROI_summary[k1][quant])
                ys2 = np.asarray(DROI_summary[k2][quant])
                if not iscum:
                    res = []
                    for ys in (ys1,ys2):
                        (cum,yr) = (0, [])
                        for yy in ys:
                            yr.append(yy - cum)
                            cum = yy
                        res.append(yr)
                    (ys1,ys2) = [np.asarray(u) for u in res]
                # Calculate the asymmetries.
                asym = []
                for (y1,y2) in zip(ys1, ys2):
                    mu = np.nanmean([y1, y2], axis=0)
                    dy = y1 - y2
                    asym.append(dy * ny.math.zinv(mu) * 100)
                # Append the data
                dat[k + '_cumulative' if iscum else k] = pimms.imm_array(asym)
        return pimms.persist(dat)
    @pimms.value
    def asymmetry_table(asymmetry, agegroup, gender, subject_list):
        '''
        asymmetry_table is a pandas dataframe of asymmetry data for each subject
        that is organized by age-group and gender. In order for this table to
        be available, you must have access to the HCP restricted data, and 
        neuropythy must be configured to use it; otherwise an error will be
        raised.
        '''
        import neuropythy as ny
        import numpy as np
        # We can start by grabbing the subject ages and genders in the same order as
        # is used in data.asymmetry, which tracks the HVA and VMA of each subject.
        # This order is the same as that given in data.subject_list.
        agegroups = [agegroup[sid] for sid in subject_list]
        genders   = [gender[sid]   for sid in subject_list]

        # Now we just make a dataframe; all the columns are already ordered by subject
        # ID so we can just put them together. We do, however, need to expand each
        # column by the wedge-size because we are flattening that dimension out into
        # a single column.
        angle_col = 'ROI Width [deg polar angle]'
        age_col = 'Age-Group [years]'
        gender_col = 'Gender'
        data_cols = ['Local HVA', 'Cumulative HVA', 'Local VMA', 'Cumulative VMA']
        data_cols = [d + ' [%]' for d in data_cols]
        data_keys = ['HVA', 'HVA_cumulative', 'VMA', 'VMA_cumulative']
        nsubs = len(subject_list)
        
        df = {}
        roi_angles = VisualPerformanceFieldsDataset.roi_angles
        df[angle_col]  = np.reshape([np.full(nsubs, k) for k in roi_angles], -1)
        df['sid']      = np.reshape([subject_list for k in roi_angles], -1)
        df[gender_col] = np.reshape([genders           for k in roi_angles], -1)
        df[age_col]    = np.reshape([agegroups         for k in roi_angles], -1)
        for (col,k) in zip(data_cols, data_keys):
            df[col] = asymmetry[k].flatten()
        df = ny.to_dataframe(df)
        df = df[['sid',age_col,gender_col,angle_col] + data_cols]
        df.set_index('sid')
        return df
    @pimms.value
    def barbot2020_data(pseudo_path):
        '''
        Load and return the data from Barbot et al. (2020).
        '''
        import os, numpy as np, pyrsistent as pyr
        from scipy.io import loadmat
        filename = pseudo_path.local_path('supp', 'barbot_xue_carrasco_2020.mat')
        pp_raw = loadmat(filename)
        pp_dat = pp_raw['DATA']
        pp_dat = {nm: pp_dat[nm][0,0].T for nm in pp_dat.dtype.names}
        behavior_mtx = 10**pp_dat['SF_75THRESH_ALL']
        behavior_mean = np.mean(behavior_mtx, axis=1)
        behavior_sem = np.sqrt(np.var(behavior_mtx, axis=1) / behavior_mtx.shape[1])
        behavior_angs = pp_dat['ANGLES_RHM0_UVM90_LHM180_LVM270'].flatten().astype('float')
        behavior_angs = np.mod(90 - behavior_angs + 180, 360) - 180
        result = {'raw': pp_raw, 'matrix': behavior_mtx,
                  'mean': behavior_mean, 'sem': behavior_sem,
                  'angles': behavior_angs}
        return pyr.pmap(result)
    @pimms.value
    def abrams2012_data(pseudo_path):
        '''
        Load and return the data from Abrams et al. (2012).
        '''
        import os, numpy as np, pyrsistent as pyr
        from scipy.io import loadmat
        filename = pseudo_path.local_path('supp', 'abrams_nizam_carrasco_2012.mat')
        pp_raw = loadmat(filename)
        pp_dat = pp_raw['DATA']
        pp_dat = {nm: pp_dat[nm][0,0].T for nm in pp_dat.dtype.names}
        behavior_mtx = pp_dat['CS_75THRESH']
        behavior_mean = np.mean(behavior_mtx, axis=1)
        behavior_sem = np.sqrt(np.var(behavior_mtx, axis=1) / behavior_mtx.shape[1])
        behavior_angs = pp_dat['ANGLES'].flatten().astype('float')
        behavior_angs = np.mod(90 - behavior_angs + 180, 360) - 180
        result = {'raw': pp_raw, 'matrix': behavior_mtx,
                  'mean': behavior_mean, 'sem': behavior_sem,
                  'angles': behavior_angs}
        return pyr.pmap(result)

    # Tables for Plotting or General Summary #######################################################
    @staticmethod
    def _cortex_V1_mask(hem, skey):
        import numpy as np
        ii = hem.indices
        v1_sectors = tuple([lbl for (k,lbl) in skey.items()
                            if k[0] == 1 if k[3] >= 1 if k[4] <= 6])
        v1_sectors = np.array(v1_sectors)
        return np.isin(hem.prop('vpf_sector'), v1_sectors)
    @staticmethod
    def _subject_V1_surface_area(subjects, skey, sid, h, area_property='midgray_surface_area'):
        import numpy as np
        hem = subjects[sid].hemis[h]
        ii = VisualPerformanceFieldsDataset._cortex_V1_mask(hem, skey)
        return np.nansum(hem.property(area_property, mask=ii))
    @staticmethod
    def _subject_V1_thickness(subjects, skey, sid, h):
        import numpy as np
        hem = subjects[sid].hemis[h]
        ii = VisualPerformanceFieldsDataset._cortex_V1_mask(hem, skey)
        return np.nanmean(hem.property('thickness', mask=ii))
    @staticmethod
    def _subject_V1_volume(subjects, skey, sid, h, area_property='midgray_surface_area'):
        import numpy as np
        hem = subjects[sid].hemis[h]
        ii = VisualPerformanceFieldsDataset._cortex_V1_mask(hem, skey)
        area = hem.property(area_property, mask=ii)
        thic = hem.property('thickness', mask=ii)
        return np.nansum(area * thic)
    @staticmethod
    def _subject_cortex_surface_area(subjects, sid, h, area_property='midgray_surface_area'):
        import numpy as np
        hem = subjects[sid].hemis[h]
        ii = hem.mask('cortex_label')
        return np.nansum(hem.property(area_property, mask=ii))
    @staticmethod
    def _subject_cortex_volume(subjects, sid, h, area_property='midgray_surface_area'):
        import numpy as np
        hem = subjects[sid].hemis[h]
        ii = hem.mask('cortex_label')
        area = hem.property(area_property, mask=ii)
        thic = hem.property('thickness', mask=ii)
        return np.nansum(area * thic)
    @staticmethod 
    def _generate_restricted_v1_summary_call(tup):
        try:
            import neuropythy as ny, numpy as np
            data = ny.data['visual_performance_fields']
            subjects = data.subjects
            skey = data.sector_key[0]
            CLS = VisualPerformanceFieldsDataset
            (sid, saprop) = tup
            res = []
            for h in ['lh','rh']:
                v1sa = CLS._subject_V1_surface_area(subjects, skey, sid, h, area_property=saprop)
                v1vo = CLS._subject_V1_volume(subjects, skey, sid, h, area_property=saprop)
                cxsa = CLS._subject_cortex_surface_area(subjects, sid, h, area_property=saprop)
                cxvo = CLS._subject_cortex_volume(subjects, sid, h, area_property=saprop)
                v1th = CLS._subject_V1_thickness(subjects, skey, sid, h)
                res.append((v1sa, v1vo, cxsa, cxvo, v1th))
            return tuple(res)
        except Exception as e:
            import numpy as np, sys, os
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            return ((np.nan, fname, exc_tb.tb_lineno, str(type(e)), str(e.args[0])),
                    (np.nan, fname, exc_tb.tb_lineno, str(type(e)), str(e.args[0])))
    @staticmethod
    def _generate_restricted_v1_summary_table(subjects, gender, agegroup,
                                              surface_area='midgray_surface_area',
                                              printstatus=False, nprocs=None):
        '''
        Generates and returns the summary table of V1 measurements regarding the
        1-6 degree eccentricity ROI. The resulting dataframe is formatted in a
        way that is deliberately friendly to seaborn. The resulting summary
        includes HCP restricted data.
        '''
        import neuropythy as ny, os, six, pyrsistent as pyr, multiprocessing as mp, numpy as np
        subject_list = subjects.keys()
        nsubs = len(subject_list)
        f = VisualPerformanceFieldsDataset._generate_restricted_v1_summary_call
        if nprocs is None or nprocs is Ellipsis: nprocs = mp.cpu_count()
        if nprocs < 2: nprocs = 1
        dat = ny.auto_dict(None, [])
        if printstatus:
            print('V1 Summary Processing blocks:')
        for ii in np.arange(0, nsubs, nprocs):
            mx = np.min([nsubs, ii + nprocs])
            sids = subject_list[ii:mx]
            if printstatus:
                print("  * %2d - %3d%% progress" % ((ii + 1)/nsubs * 100, mx/nsubs * 100))
            with mp.Pool(nprocs) as pool:
                sids = subject_list[ii:ii+nprocs]
                tups = pool.map(f, [(sid,surface_area) for sid in sids])
            # add these data into the overall roi table
            for (sid,hemtups) in zip(sids,tups):
                for (h, tup) in zip(['lh','rh'], hemtups):
                    dat['sid'].append(sid)
                    dat['Gender'].append(gender[sid])
                    dat['Age'].append(agegroup[sid])
                    dat['Hemisphere'].append(h)
                    (v1sa, v1vo, cxsa, cxvo, v1th) = tup
                    if np.isnan(v1sa):
                        raise ValueError('Subject %s, hemi %s, raised nan' % (sid, h),
                                         cxvo, v1th, v1vo, cxsa)
                    dat[r'V1 Surface Area [cm$^2$]'].append(v1sa / 100.0)
                    dat[r'V1 Gray Volume [cm$^3$]'].append(v1vo / 1000.0)
                    dat[r'Mean V1 Gray Thickness [mm]'].append(v1th)
                    dat[r'Cortex Surface Area [cm$^2$]'].append(cxsa / 100.0)
                    dat[r'Cortex Gray Volume [cm$^3$]'].append(cxvo / 1000.0)
                    dat[r'Normalized V1 Surface Area [%]'].append(v1sa / cxsa * 100)
                    dat[r'Normalized V1 Gray Volume [%]'].append(v1vo / cxvo * 100)
        return ny.to_dataframe(dat)
    def generate_restricted_v1_summary_table(self, surface_area='midgray_surface_area',
                                  printstatus=False, nprocs=None):
        '''
        Generates and returns the summary table of V1 measurements, including
        restricted HCP data, regarding the 1-6 degree eccentricity ROI. The
        resulting dataframe is formatted in a way that is deliberately friendly
        to seaborn.
        '''
        return type(self)._generate_v1_summary_table(self.subjects, self.gender, self.agegroup,
                                                     surface_area=surface_area,
                                                     printstatus=printstatus, nprocs=nprocs)
    @pimms.value
    def restricted_v1_summary_table(subjects, gender, agegroup):
        '''
        restricted_v1_pair_summary_table is a dataframe containing summary data,
        including restricted HCP data, about the V1 surface area, volume, and
        thickness for the ROIs examined in the associated paper (i.e., V1
        limited to 1-6 degrees of eccentricity). In order to access these data,
        you must have configured neuropythy to have access to the HCP restricted
        data.

        Note that because this table requires the restricted HCP data, it must be
        generated when requested (it cannot be stored on a public repository).
        Accordingly, it may take a substantial amount of time to load this
        value.
        '''
        return VisualPerformanceFieldsDataset._generate_restricted_v1_summary_table(subjects,
                                                                                    gender,
                                                                                    agegroup)
    @staticmethod 
    def _generate_v1_summary_call(tup):
        import neuropythy as ny, numpy as np
        data = ny.data['visual_performace_fields']
        subjects = data.subjects
        skey = data.sector_key[0]
        CLS = VisualPerformanceFieldsDataset
        (sid, saprop) = tup
        sub = data.subjects[sid]
        sas = []
        ths = []
        for h in ['lh','rh']:
            hem = sub.hemis[h]
            lbl = hem.prop('visual_area') == 1
            sas.append(hem.prop(saprop)[lbl])
            ths.append(hem.prop('thickness')[lbl])
        area = np.nansum(np.concatenate(sas))
        thick = np.nanmean(np.concatenate(ths))
        return (area, thick)
    @staticmethod
    def _generate_v1_summary_table(subjects, surface_area='midgray_surface_area',
                                   printstatus=False, nprocs=None):
        '''
        Generates and returns the summary table of V1 measurements for the
        entire V1 area as drawn by the raters for the HCP_lines dataset used in
        this project.
        '''
        import neuropythy as ny, os, six, pyrsistent as pyr, multiprocessing as mp, numpy as np
        subject_list = subjects.keys()
        nsubs = len(subject_list)
        f = VisualPerformanceFieldsDataset._generate_v1_summary_call
        if nprocs is None or nprocs is Ellipsis: nprocs = mp.cpu_count()
        if nprocs < 2: nprocs = 1
        dat = ny.auto_dict(None, [])
        if printstatus:
            print('V1 Summary Processing blocks:')
        for ii in np.arange(0, nsubs, nprocs):
            mx = np.min([nsubs, ii + nprocs])
            sids = subject_list[ii:mx]
            if printstatus:
                print("  * %2d - %3d%% progress" % ((ii + 1)/nsubs * 100, mx/nsubs * 100))
            with mp.Pool(nprocs) as pool:
                tups = pool.map(f, [(sid,surface_area) for sid in sids])
            # add these data into the overall roi table
            for (sid,tup) in zip(sids,tups):
                dat['sid'].append(sid)
                (sa, th) = tup
                dat['surface_area_cm2'].append(sa / 100.0)
                dat['mean_thickness_mm'].append(th)
        return ny.to_dataframe(dat)
    def generate_v1_summary_table(self, surface_area='midgray_surface_area',
                                  printstatus=False, nprocs=None):
        '''
        Generates and returns the summary table of V1 measurements regarding the
        full 0-6 degree eccentricity ROI.
        '''
        return type(self)._generate_v1_summary_table(self.subjects, surface_area=surface_area,
                                                     printstatus=printstatus, nprocs=nprocs)
    @pimms.value
    def v1_summary_table(subjects, pseudo_path):
        '''
        v1_summary_table is a dataframe containing summary data about the V1
        surface area, volume, and thickness for the ROIs examined in the
        associated paper (i.e., V1 limited to 0-6 degrees of eccentricity).
        '''
        import neuropythy as ny
        df = ny.load(pseudo_path.local_path('v1_summary_table.csv'))
        df.set_index(['sid'])
        return df
    @staticmethod
    def _generate_asymmetry_pair_table(subject_list, asymmetry, retinotopy_siblings,
                                       agegroup, gender,
                                       suffix='_cumulative'):
        import neuropythy as ny, pimms, numpy as np
        pair_data = ny.auto_dict(None, [])
        sid_to_index = {sid: k for (k,sid) in enumerate(np.sort(subject_list))}
        # Start by finding the subset of pairs we want to use:
        mzpairs = retinotopy_siblings['MZ']
        dzpairs = retinotopy_siblings['DZ']
        sbpairs = retinotopy_siblings['']
        # Find all the sibling pairs:
        sibs = set([tuple(sorted([k,v]))
                    for pairs in [mzpairs, dzpairs, sbpairs]
                    for (k,vs) in pairs.items()
                    for v in (vs if pimms.is_tuple(vs) else [vs])])
        # Now find unrelated pairs:
        urpairs = set([(a,b)
                       for a in subject_list
                       for b in subject_list
                       if a < b and (a,b) not in sibs])
        # The unrelated pairs get limited to age- and gender-matched pairs
        mzpairs = set([tuple(sorted([k,v]))
                       for (k,vs) in mzpairs.items()
                       for v in (vs if pimms.is_tuple(vs) else [vs])
                       if k in sid_to_index and v in sid_to_index])
        dzpairs = set([tuple(sorted([k,v]))
                       for (k,vs) in dzpairs.items()
                       for v in (vs if pimms.is_tuple(vs) else [vs])
                       if k in sid_to_index and v in sid_to_index])
        for (k,pairs) in [('MZ',mzpairs), ('DZ',dzpairs), ('UR',urpairs)]:
            for (t1,t2) in pairs:
                (k1,k2) = [sid_to_index.get(t, None) for t in (t1,t2)]
                if k1 is None or k2 is None: continue
                pair_data['sid_1'].append(t1)
                pair_data['sid_2'].append(t2)
                pair_data['relationship'].append(k)
                pair_data['same_age'].append(agegroup[t1] == agegroup[t2])
                pair_data['same_sex'].append(gender[t1] == gender[t2])
                for col in ['HVA','VMA']:
                    for (ii,w) in enumerate([10,20,30,40,50]):
                        pair_data['%s%d_1' % (col,w)].append(asymmetry[col+suffix][ii][k1])
                        pair_data['%s%d_2' % (col,w)].append(asymmetry[col+suffix][ii][k2])
        return ny.to_dataframe(pair_data)
    def generate_asymmetry_pair_table(self, suffix='_cumulative'):
        '''
        Yields a dataframe of asymmetry data sorted by pairs. Pair types include MZ (monozygotic
        twins), DZ (dizygotic twins), and 'UR' (unrelaited pair). This data will fail to load if you
        have not configured neuropythy to have access to the HCP restricted data.

        The optional argument suffix may be set to '' to calculate instantaneous instead of
        cumulative ROI asymmetry.
        '''
        return type(self)._generate_asymmetry_pair_table(self.subject_list, 
                                                         self.asymmetry, 
                                                         self.retinotopy_siblings, 
                                                         self.agegroup, 
                                                         self.gender)
    @pimms.value
    def asymmetry_pair_table(subject_list, asymmetry, retinotopy_siblings, agegroup, gender):
        '''
        asymmetry_pair_table is a dataframe of asymmetry data sorted by pairs. Pair types include
        MZ (monozygotic twins), DZ (dizygotic twins), and 'UR' (unrelaited pair). This data will
        fail to load if you have not configured neuropythy to have access to the HCP restricted
        data.
        '''
        return VisualPerformanceFieldsDataset._generate_asymmetry_pair_table(
            subject_list, asymmetry, retinotopy_siblings, agegroup, gender)
    
add_dataset('visual_performance_fields',
            lambda:VisualPerformanceFieldsDataset())
