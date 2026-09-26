"""Compatibility import path; use :mod:`expert_method.ridge_sinkhorn.matrix`."""

import sys as _sys

from expert_method.ridge_sinkhorn import matrix as _implementation

_sys.modules[__name__] = _implementation
