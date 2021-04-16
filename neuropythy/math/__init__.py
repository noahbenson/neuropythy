####################################################################################################
# neuropythy/math/__init__.py
# The Neuropythy math core module.
# by Noah C. Benson

from .core import (pi, half_pi, quarter_pi, tau, inf, nan, radperdeg, degperrad,
                   pytorch, to_torchdtype, torchdtype_to_numpydtype,
                   isarray, istensor, issparse, isdense, eq, ne, le, lt, ge, gt,
                   clone, totensor, astensor, toarray, asarray, asdense, promote, reshape,
                   add, sub, mul, div, mod, safesqrt, sqrt, exp, log, log10, log2, abs,
                   arcsin, sin, arccos, cos, arctan, tan, lgamma, hypot, hypot2,
                   triarea, eudist2, eudist, trisides2, trisides, trialtitudes,
                   rangemod, radmod, degmod, branch, zinv, numel, rows, check_sparsity, unbroadcast,
                   sum, prod, mean, var, std, median, min, max, argmin, argmax, all,
                   beta_log_prob, beta_prob, normal_log_prob, normal_prob,
                   cauchy_log_prob, cauchy_prob, halfcauchy_log_prob, halfcauchy_prob, 
                   laplace_log_prob, laplace_prob, exp_log_prob, exp_prob,
                   gennorm_log_prob, gennorm_prob, gumbel_log_prob, gumbel_prob)

pytorch_ok = False
try:
    if pytorch() is not None:
        pytorch_ok = True
except ImportError: pass

