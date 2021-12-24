# -*- coding: utf-8 -*-
####################################################################################################
# neuropythy/math/__init__.py
# The Neuropythy math core module.
# by Noah C. Benson

"""The neuropythy math sub-package.

The neuropythy math sub-package is intended as a universal set of functions that
unite `numpy` and `torch` libraries into a coherent interface. The need for this
arises from the fact that, while `numpy` can work fine with either `numpy`
arrays or typical python lists/tuples, `torch` only works with its own arrays.
This is frustrating when writing API functions that would ideally accept either
kind of object. The functions in this module do this and allow one to write
simple APIs that deal well with any of the native, `numpy`, or `torch` tensor
types.
"""

from .core import (
    torch, pytorch_found,
    
    # Constants
    pi, half_pi, quarter_pi, tau, inf, nan, radperdeg, degperrad,

    # Utility / Setup Functions
    to_torchdtype, torchdtype_to_numpydtype, to_nunmpydtype, to_dtype,
    is_array, is_tensor, is_ndcoll, is_sparse, is_dense,
    like_ndcoll, like_tensor, like_ndcoll, like_spares, like_dense,
    to_array, to_tensor, to_ndcoll, to_sparse, to_dense,
    as_array, as_tensor, as_ndcoll, as_sparse, as_dense,
    to_readonly, as_readonly, is_readonly,
    reshape_indices, flatten_indices, unflatten_indices,
    is_numeric, clone, shape,

    # Creation functions.
    zeros, ones, full, empty, rand, randn, randint, permutation,

    # Comparisons
    eq, ne, le, lt, ge, gt,

    # Edit, Change, Query
    promote, reshape, squeeze, numel, rows, check_sparsity, unbroadcast,

    # Arithmetic
    add, sub, mul, div, mod, safesqrt, sqrt, exp, log, log10, log2, abs,
    arcsin, sin, arccos, cos, arctan, tan, lgamma, hypot, hypot2,
    triarea, eudist2, eudist, trisides2, trisides, trialtitudes,
    rangemod, radmod, degmod, branch, zinv,

    # Accumulators
    sum, prod, mean, var, std, median, min, max, argmin, argmax, all,

    # Probability density functions
    beta_log_prob, beta_prob, normal_log_prob, normal_prob,
    cauchy_log_prob, cauchy_prob, halfcauchy_log_prob, halfcauchy_prob, 
    laplace_log_prob, laplace_prob, exp_log_prob, exp_prob,
    gennorm_log_prob, gennorm_prob, gumbel_log_prob, gumbel_prob)

