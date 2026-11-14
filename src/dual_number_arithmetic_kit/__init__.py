"""Dual number arithmetic for exact forward-mode automatic differentiation.

This package exposes a single value type, :class:`Dual`, which carries a
function value and its first derivative with respect to one chosen input.
Only the operations needed for scalar forward-mode AD are provided; anything
else (higher-order derivatives, vector arguments, symbolic expressions) is
out of scope by design.
"""

from .core import Dual

__all__ = ["Dual"]
