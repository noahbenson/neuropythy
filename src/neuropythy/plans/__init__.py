####################################################################################################
# neuropythy/plans/__init__.py
# A neuropythy module that stores PIMMS calculation plans.
# By Noah C. Benson

from .core import (limit_param, unlimit_param, imap_forget, imap_efferents)
from .prfclean import (prfclean_plan)

def run_prfclean(hemi, **kw):
    '''
    run_prfclean(hemi) runs the calculations associated with neuropythy.plans.prfclean_plan and
      yields a duplicate hemisphere object with the cleaned pRF maps included as properties. These
      properties

    In addition to including the cleaned pRF properties as properties of the returned hemisphere,
    this function appends meta-data describing the model parameters that were fit by the pRF
    cleaning.

    The following options may be given:
     * labels (one per vertex of coftex) or the name of a property of cortex.
       Note that the labels must not use 0 as an ROI: 0 must always indicate
       vertices that should not be optimized. Default: 'visual_area'
     * retinotopy: The argument for neuropythy's retinotopy_data function; this
       is typically a string prefix for retinotopy data on the cortex's
       properties ('prf_' for HCP subjects), or a map of the data itself.
       Default: 'prf_'.
     * tag: The prefix to give the output properties and the meta-data on the
       returned cortex object. The meta-data is named <teg>params.
       Default: 'prfclean_'
     * step_fn: A function that gets run every step of the optimization. The
       function must accept 2 arguments: the step number and the imap from the
       prfclean_plan. Default: None.
     * steps: The number of steps to run for minimization. Default: 200.
     * lr: The learning rate of the minimization. Default: 0.1.
     * max_eccen: The maximum eccentricity in the optimized maps. This should be
       the ecceentricity of the peripheral boundary of the labeled maps.
       Default: 90.
     * fieldsigns: The fieldsign values for each visual area. This may either be
       a dictionary whose keys are visual area labels and whose values are all
       either 1 or -1, or it can be Ellipsis or None. If the value is Ellipsis
       (the default), then the dictionary visual_area_field_signs from the
       neuropythy.vision package is used. If the value is None, then even
       labels are given fieldsigns of 1 and odd labels are given fieldsigns of
       -1.
     * model_knob: The base-2 log of the constant weight that the vmag_likelihood
       is multiplied by prior to summation with meas_likelihood. Default: 12.
     * boundary_knob: The base-2 log of the constant weight that the
       boundary_likelihood is multiplied by prior to summation with
       meas_likelihood. Default: 2.
     * start_coords: May specify the (2 x N) coordinate matrix of measurements
       to use as the starting point in the optimization. The N must be the
       number of vertices in the cortex; though values are only needed wherever
       there are non-zero labels. If None, then the measurements are used as
       the starting coordinates. The first row of the start coordinates must be
       the polar angle theta given in counter-clockwise radians starting from
       the right horizontal meridian, and the second row must be the
       eccentricity of the vertices in visual degrees.
     * prior_cmag_tan_params: The initial values of the tangential cortical
       magnificatiioin model parametters, whose interpretation depends on the
       value given for cmag_tan_method (see it for more details). Note that
       cmag_tan_method describes parameters such as loga, which are used in the
       model via the transformation a = exp(loga). The value that is given for
       the prior should *not* be the transformed param--i.e., provide a not loga
       as the prioir value.
     * prior_uvm_ipsi: The prior value of the upper-vertical-meridian's
       ipsilateral representation. This value is the number of radians into the
       ipsilateral visual field that the PRF representations of all visual areas
       extend for the upper vertical meridian.
     * prior_lvm_ipsi: The prior value of the lower-vertical-meridian's
       ipsilateral representation. This value is the number of radians into the
       ipsilateral visual field that the PRF representations of all visual areas
       extend for the lower vertical meridian.
     * grad_ipsi: Whether the uvm_ipsi and lvm_ipsi parameters, representing the
       ipsilateral representation in each visual area, are part of the optimized
       parameters whose gradients are required.
     * grad_coords: Specifies whether the tensors representing the coordinates of
       the PRF centers should track their gradients (default: True).
     * grad_cmag_tan: Specifies whether the tensors representing the tangential
       cortical magnification parameters of the retinotopy model should track
       their gradients (default: True).
     * grad_cmag_rad: Specifies whether the tensors representing the radial 
       cortical magnification parameters of the retinotopy model should track
       their gradients (default: True).
     * device: The pytorch device name to be used for all tensors (default: 'cpu').
     * meas_weight: A vector of weights, one per vertex in mesh, that specifies
       how strongly the model should believe that particular vertex's weight.
       Weights are the divisor for the meas_stddev. See calc_meas_stddev for
       more information about the standard deviation formula.
     * prior_cmag_eccen_offsets: The initial value of the offset parameter c2 
       from Horton and Hoyt's (1991) equation cmag(ecc) = c1 / (c2 + ecc).
       This should be either a single number, which is used for all visual
       areas, or a dictionary of visual area labels mapped to values.
     * dtype: May optionally specify the dtype for PyTorch to use in the
       optimization. This may be a PyTorch dtype object or it may be a string,
       in which case it is looked up as a member of the torch module (i.e.,
       'float' 
     * cmag_tan_method: The method to use for calculating tangential cortical
       magnification. This may be either 'sinusoid' (the default), 'beta', or
       None, in which case tangential cortical magnification is modeled to be
       uniform across polar angle. For thee sinusoid, the parameters are tan_hva
       and tan_vma where the actual hva and vma are arctan(tan_hva)*2/pi and
       arctan(tan_vma)*2/pi, limiting them both to the +/- 1 range. For the
       'beta' method, the params are loga and logb, where the beta-distribution
       parameters that are used are simply a = exp(loga) and b = exp(logb).
     * yield_imap: If True, yields the imap object instead of the hemi object
       after the minimization. Default: False.
    '''
    import numpy as np
    stepfn = kw.pop('step_fn', None)
    lr = kw.pop('lr', 0.1)
    steps = kw.pop('steps', 200)
    tag = kw.pop('tag', 'prfclean_')
    yield_imap = kw.pop('yield_imap', False)
    kw = dict(dict(retinotopy='prf_', labels='visual_area', max_eccen=90), **kw)
    # Go ahead and make the imap now:
    imap = prfclean_plan(cortex=hemi, **kw)
    llh0 = imap['likelihood']
    gradients = imap['gradients']
    grad_effs = imap_efferents(imap, gradients.keys())
    # Okay, we're going to run it!
    torch = imap['torch']
    opt = torch.optim.LBFGS(list(gradients.values()), lr=lr)
    for step in range(steps):
        if stepfn is not None:
            r = stepfn(step, imap)
            if r is not None: imap = r
        def closure():
            opt.zero_grad()
            imap_forget(imap, grad_effs)
            nll = -imap['likelihood']
            nll.backward()
            return nll
        opt.step(closure)
    # Finished minimizing--if we need to yield the imap we are done.
    if yield_imap: return imap
    # We need to put the data on the hemisphere.
    mesh = imap['mesh']
    ang = np.full(hemi.vertex_count, np.nan)
    ecc = np.full(hemi.vertex_count, np.nan)
    ang[mesh.labels] = np.mod(90 - 180/np.pi*imap['theta'].detach().numpy() + 180, 360) - 180
    ecc[mesh.labels] = imap['eccen'].detach().numpy()
    hemi = hemi.with_prop({(tag + 'polar_angle'): ang, (tag + 'eccentricity'): ecc})
    params = {k: np.array(imap[k].detach().numpy())
              for k in ['cmag_eccen_offsets', 'cmag_tan_params', 'uvm_ipsi', 'lvm_ipsi']}
    hemi = hemi.with_meta({(tag + 'params'): params})
    return hemi


