"""Core dual-number implementation.

A dual number is represented as a pair ``(value, derivative)`` where the
second component is the derivative of ``value`` with respect to a single
independent variable.  Arithmetic on these pairs follows the usual rules of
dual numbers, which makes forward-mode automatic differentiation exact for
polynomial and elementary-function compositions (up to floating-point
rounding).

Why tuples rather than a mutable class?

* Dual numbers are values.  Making them immutable means they can be shared
  freely between callers without defensive copying, and equality is
  structural in the obvious way.
* ``__slots__`` on a tuple subclass keeps memory small when differentiating
  long chains of operations.
* The representation is simple enough that the derivative rules are visible
  in one place instead of being spread across mutating helpers.

Only ``+``, ``-``, ``*``, ``/``, integer powers, ``sin``, ``cos``, ``exp``,
and ``log`` are provided.  These are the primitives needed to differentiate
most scalar expressions; anything more can be composed from them.
"""

from __future__ import annotations

import math
from typing import Tuple

__all__ = ["Dual"]


class Dual(Tuple[float, float]):
    """An immutable dual number ``(value, derivative)``.

    Parameters
    ----------
    value:
        The function value at the evaluation point.
    derivative:
        The derivative with respect to the independent variable.  Defaults to
        ``0.0``, which is the correct derivative of a constant.

    Examples
    --------
    >>> x = Dual(2.0, 1.0)
    >>> y = x * x + Dual(3.0)
    >>> y
    (7.0, 4.0)
    """

    __slots__ = ()

    def __new__(cls, value: float = 0.0, derivative: float = 0.0) -> "Dual":
        return tuple.__new__(cls, (float(value), float(derivative)))

    @property
    def value(self) -> float:
        """The function value (the primal component)."""
        return self[0]

    @property
    def derivative(self) -> float:
        """The derivative (the tangent component)."""
        return self[1]

    def __repr__(self) -> str:
        return f"Dual(value={self.value!r}, derivative={self.derivative!r})"

    def __add__(self, other: object) -> "Dual":
        if isinstance(other, Dual):
            return Dual(self.value + other.value, self.derivative + other.derivative)
        if isinstance(other, (int, float)):
            # Adding a constant does not change the derivative.
            return Dual(self.value + other, self.derivative)
        return NotImplemented

    def __radd__(self, other: object) -> "Dual":
        return self.__add__(other)

    def __sub__(self, other: object) -> "Dual":
        if isinstance(other, Dual):
            return Dual(self.value - other.value, self.derivative - other.derivative)
        if isinstance(other, (int, float)):
            return Dual(self.value - other, self.derivative)
        return NotImplemented

    def __rsub__(self, other: object) -> "Dual":
        if isinstance(other, (int, float)):
            return Dual(other - self.value, -self.derivative)
        return NotImplemented

    def __mul__(self, other: object) -> "Dual":
        if isinstance(other, Dual):
            return Dual(
                self.value * other.value,
                self.value * other.derivative + self.derivative * other.value,
            )
        if isinstance(other, (int, float)):
            return Dual(self.value * other, self.derivative * other)
        return NotImplemented

    def __rmul__(self, other: object) -> "Dual":
        return self.__mul__(other)

    def __truediv__(self, other: object) -> "Dual":
        if isinstance(other, Dual):
            denom = other.value * other.value
            return Dual(
                self.value / other.value,
                (self.derivative * other.value - self.value * other.derivative) / denom,
            )
        if isinstance(other, (int, float)):
            return Dual(self.value / other, self.derivative / other)
        return NotImplemented

    def __rtruediv__(self, other: object) -> "Dual":
        if isinstance(other, (int, float)):
            return Dual(other / self.value, -other * self.derivative / (self.value * self.value))
        return NotImplemented

    def __pow__(self, exponent: int) -> "Dual":
        # Only integer powers are supported so that the derivative rule is
        # exact and does not require ``log`` of a possibly non-positive value.
        if not isinstance(exponent, int):
            return NotImplemented
        if exponent == 0:
            return Dual(1.0, 0.0)
        new_value = self.value ** exponent
        new_derivative = exponent * (self.value ** (exponent - 1)) * self.derivative
        return Dual(new_value, new_derivative)

    def __neg__(self) -> "Dual":
        return Dual(-self.value, -self.derivative)

    def __pos__(self) -> "Dual":
        return self

    def __abs__(self) -> "Dual":
        if self.value > 0:
            return self
        if self.value < 0:
            return -self
        # At zero the derivative is undefined; we choose the subgradient 0.
        return Dual(0.0, 0.0)

    def sin(self) -> "Dual":
        """Sine of a dual number."""
        return Dual(math.sin(self.value), math.cos(self.value) * self.derivative)

    def cos(self) -> "Dual":
        """Cosine of a dual number."""
        return Dual(math.cos(self.value), -math.sin(self.value) * self.derivative)

    def exp(self) -> "Dual":
        """Exponential of a dual number."""
        value = math.exp(self.value)
        return Dual(value, value * self.derivative)

    def log(self) -> "Dual":
        """Natural logarithm of a dual number.

        Raises ``ValueError`` when the value is non-positive, matching the
        behaviour of ``math.log`` on a plain float.
        """
        if self.value <= 0:
            raise ValueError("math domain error")
        return Dual(math.log(self.value), self.derivative / self.value)
