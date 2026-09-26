"""Compatibility import path; use :mod:`expert_method.ridge_sinkhorn.three_seed_study`."""

import sys as _sys

from expert_method.ridge_sinkhorn import three_seed_study as _implementation

_sys.modules[__name__] = _implementation
