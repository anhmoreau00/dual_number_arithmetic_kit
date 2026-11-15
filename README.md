# dual_number_arithmetic_kit

Exact forward-mode automatic differentiation of scalar functions using dual numbers built on Python tuples.

## Usage

```python
from dual_number_arithmetic_kit import Dual

# Evaluate f(x) = sin(x^2) and its derivative at x = 1.
x = Dual(1.0, 1.0)
f = (x * x).sin()

print(f.value)       # 0.8414709848078965
print(f.derivative)  # 1.0806046117362795  (2 * cos(1))
```

## Why this exists

Forward-mode AD needs a value that carries both a function result and its
derivative through a chain of arithmetic.  The usual approach is to build a
full expression graph or to use a library with a large surface area.  For
scalar work, a dual number is enough: it is just a pair, and every operation
has a closed-form rule.  The trade-off is that only first derivatives with
respect to one variable are supported, and only the elementary functions
listed below are built in.  Everything else must be composed from those
primitives.

## API

`Dual(value, derivative=0.0)` is a tuple subclass with two read-only
properties, `.value` and `.derivative`.  It supports:

* `+`, `-`, `*`, `/` with another `Dual` or with an `int`/`float`
* integer powers via `**`
* unary `-` and `+`
* `abs()`
* methods `sin()`, `cos()`, `exp()`, `log()`

`log()` raises `ValueError` for non-positive values, matching `math.log`.
`abs()` at zero returns a derivative of zero; the true derivative is
undefined there, and zero is the subgradient this library chooses.

## The awkward edge

`**` accepts only integer exponents.  A non-integer exponent would require
`log(value)`, which is undefined for non-positive values, so the operator
returns `NotImplemented` and Python raises `TypeError`.  Use `exp(n * log(x))`
if you need a real power and know the domain is positive.
