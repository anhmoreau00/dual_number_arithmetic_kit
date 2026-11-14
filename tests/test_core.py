import math
import unittest

from dual_number_arithmetic_kit import Dual


class DualConstructionTests(unittest.TestCase):
    def test_default_derivative_is_zero(self):
        d = Dual(3.5)
        self.assertEqual(d.value, 3.5)
        self.assertEqual(d.derivative, 0.0)

    def test_tuple_equality(self):
        self.assertEqual(Dual(1.0, 2.0), (1.0, 2.0))

    def test_repr_round_trip(self):
        d = Dual(2.0, -1.0)
        self.assertIn("value=2.0", repr(d))
        self.assertIn("derivative=-1.0", repr(d))


class DualArithmeticTests(unittest.TestCase):
    def test_add_dual(self):
        self.assertEqual(Dual(1.0, 1.0) + Dual(2.0, 3.0), (3.0, 4.0))

    def test_add_scalar(self):
        self.assertEqual(Dual(1.0, 1.0) + 2.0, (3.0, 1.0))

    def test_radd_scalar(self):
        self.assertEqual(2.0 + Dual(1.0, 1.0), (3.0, 1.0))

    def test_sub_dual(self):
        self.assertEqual(Dual(5.0, 2.0) - Dual(1.0, 1.0), (4.0, 1.0))

    def test_rsub_scalar(self):
        self.assertEqual(5.0 - Dual(1.0, 2.0), (4.0, -2.0))

    def test_mul_dual(self):
        # d/dx (x * x) at x=3 is 6.
        self.assertEqual(Dual(3.0, 1.0) * Dual(3.0, 1.0), (9.0, 6.0))

    def test_mul_scalar(self):
        self.assertEqual(Dual(3.0, 1.0) * 2.0, (6.0, 2.0))

    def test_rmul_scalar(self):
        self.assertEqual(2.0 * Dual(3.0, 1.0), (6.0, 2.0))

    def test_truediv_dual(self):
        # d/dx (1 / x) at x=2 is -1/4.
        result = Dual(1.0, 0.0) / Dual(2.0, 1.0)
        self.assertAlmostEqual(result.value, 0.5)
        self.assertAlmostEqual(result.derivative, -0.25)

    def test_truediv_scalar(self):
        self.assertEqual(Dual(4.0, 2.0) / 2.0, (2.0, 1.0))

    def test_rtruediv_scalar(self):
        # d/dx (1 / x) at x=2 is -1/4.
        result = 1.0 / Dual(2.0, 1.0)
        self.assertAlmostEqual(result.value, 0.5)
        self.assertAlmostEqual(result.derivative, -0.25)

    def test_pow_zero(self):
        self.assertEqual(Dual(3.0, 1.0) ** 0, (1.0, 0.0))

    def test_pow_positive(self):
        # d/dx (x^3) at x=2 is 12.
        self.assertEqual(Dual(2.0, 1.0) ** 3, (8.0, 12.0))

    def test_pow_negative(self):
        # d/dx (x^-1) at x=2 is -1/4.
        result = Dual(2.0, 1.0) ** -1
        self.assertAlmostEqual(result.value, 0.5)
        self.assertAlmostEqual(result.derivative, -0.25)

    def test_pow_non_integer_returns_notimplemented(self):
        with self.assertRaises(TypeError):
            _ = Dual(2.0, 1.0) ** 0.5

    def test_neg(self):
        self.assertEqual(-Dual(2.0, 3.0), (-2.0, -3.0))

    def test_pos(self):
        self.assertEqual(+Dual(2.0, 3.0), (2.0, 3.0))

    def test_abs_positive(self):
        self.assertEqual(abs(Dual(2.0, 3.0)), (2.0, 3.0))

    def test_abs_negative(self):
        self.assertEqual(abs(Dual(-2.0, 3.0)), (2.0, -3.0))

    def test_abs_zero(self):
        self.assertEqual(abs(Dual(0.0, 3.0)), (0.0, 0.0))


class DualFunctionTests(unittest.TestCase):
    def test_sin(self):
        result = Dual(0.0, 1.0).sin()
        self.assertAlmostEqual(result.value, 0.0)
        self.assertAlmostEqual(result.derivative, 1.0)

    def test_cos(self):
        result = Dual(0.0, 1.0).cos()
        self.assertAlmostEqual(result.value, 1.0)
        self.assertAlmostEqual(result.derivative, 0.0)

    def test_exp(self):
        result = Dual(0.0, 1.0).exp()
        self.assertAlmostEqual(result.value, 1.0)
        self.assertAlmostEqual(result.derivative, 1.0)

    def test_log(self):
        result = Dual(1.0, 1.0).log()
        self.assertAlmostEqual(result.value, 0.0)
        self.assertAlmostEqual(result.derivative, 1.0)

    def test_log_non_positive_raises(self):
        with self.assertRaises(ValueError):
            Dual(0.0, 1.0).log()
        with self.assertRaises(ValueError):
            Dual(-1.0, 1.0).log()


class DualCompositionTests(unittest.TestCase):
    def test_chain_rule(self):
        # f(x) = sin(x^2) at x=1: f(1)=sin(1), f'(1)=2*cos(1).
        x = Dual(1.0, 1.0)
        f = (x * x).sin()
        self.assertAlmostEqual(f.value, math.sin(1.0))
        self.assertAlmostEqual(f.derivative, 2.0 * math.cos(1.0))

    def test_product_and_quotient(self):
        # f(x) = (x^2 + 1) / x at x=2: f(2)=2.5, f'(2)= (2x*x - (x^2+1))/x^2 = 3/4.
        x = Dual(2.0, 1.0)
        f = (x * x + 1.0) / x
        self.assertAlmostEqual(f.value, 2.5)
        self.assertAlmostEqual(f.derivative, 0.75)


if __name__ == "__main__":
    unittest.main()
