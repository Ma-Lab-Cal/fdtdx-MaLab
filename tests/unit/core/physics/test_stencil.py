"""Unit tests for fdtdx.core.physics.stencil.

Checks the staggered-grid tap weights against the classical closed forms, the moment
(Taylor) conditions that define them, and the Courant reduction factor they imply.
"""

from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest

from fdtdx.core.physics.stencil import (
    SUPPORTED_CURL_ORDERS,
    staggered_stencil_coefficients,
    stencil_cfl_factor,
    stencil_radius,
)

#: Classical staggered-grid (Fornberg) weights, as exact rationals.
KNOWN_COEFFICIENTS = {
    2: (Fraction(1),),
    4: (Fraction(9, 8), Fraction(-1, 24)),
    6: (Fraction(75, 64), Fraction(-25, 384), Fraction(3, 640)),
    8: (Fraction(1225, 1024), Fraction(-245, 3072), Fraction(49, 5120), Fraction(-5, 7168)),
}


class TestCoefficientValues:
    """The tap weights match the tabulated rationals."""

    @pytest.mark.parametrize("order", SUPPORTED_CURL_ORDERS)
    def test_matches_known_rationals(self, order):
        coeffs = staggered_stencil_coefficients(order)
        expected = KNOWN_COEFFICIENTS[order]
        assert len(coeffs) == len(expected) == order // 2
        for got, want in zip(coeffs, expected):
            assert got == pytest.approx(float(want), rel=1e-15, abs=1e-300)

    @pytest.mark.parametrize("order", SUPPORTED_CURL_ORDERS)
    def test_radius_is_half_the_order(self, order):
        assert stencil_radius(order) == order // 2

    def test_rejects_unsupported_order(self):
        with pytest.raises(ValueError):
            stencil_radius(3)
        with pytest.raises(ValueError):
            staggered_stencil_coefficients(3)
        with pytest.raises(ValueError):
            staggered_stencil_coefficients(0)


class TestTaylorConsistency:
    """The weights annihilate every odd moment below the order and normalize the first."""

    @pytest.mark.parametrize("order", SUPPORTED_CURL_ORDERS)
    def test_odd_moment_conditions(self, order):
        # The taps sit at +-(m - 1/2) cells from the staggered derivative point, so the even part
        # cancels identically and the odd moments must satisfy sum_m a_m (2m-1)^(2k-1) = delta_k1.
        coeffs = staggered_stencil_coefficients(order)
        r = order // 2
        for k in range(1, r + 1):
            moment = sum(a * (2 * m - 1) ** (2 * k - 1) for m, a in enumerate(coeffs, start=1))
            assert moment == pytest.approx(1.0 if k == 1 else 0.0, abs=1e-9)

    @pytest.mark.parametrize("order", SUPPORTED_CURL_ORDERS)
    def test_exact_for_polynomials_up_to_the_order(self, order):
        # The leading truncation term is h^order * f^(order+1), so every polynomial of degree
        # <= order is differentiated exactly.
        coeffs = staggered_stencil_coefficients(order)
        h = 0.5
        x0 = 1.25  # staggered derivative point, away from the origin so odd/even cannot alias
        for degree in range(order + 1):

            def f(x, d=degree):
                return x**d

            approx = sum(a * (f(x0 + (m - 0.5) * h) - f(x0 - (m - 0.5) * h)) for m, a in enumerate(coeffs, start=1)) / h
            exact = degree * x0 ** (degree - 1) if degree >= 1 else 0.0
            assert approx == pytest.approx(exact, rel=1e-10, abs=1e-10)

        # ... and the next degree up is not exact, i.e. the order is not accidentally higher.
        degree = order + 1
        approx = (
            sum(a * ((x0 + (m - 0.5) * h) ** degree - (x0 - (m - 0.5) * h) ** degree) for m, a in enumerate(coeffs, 1))
            / h
        )
        assert abs(approx - degree * x0 ** (degree - 1)) > 1e-12


class TestCflFactor:
    """The Courant reduction factor is the l1 norm of the taps."""

    def test_order_two_is_unity(self):
        assert stencil_cfl_factor(2) == pytest.approx(1.0)

    def test_order_four_is_seven_sixths(self):
        assert stencil_cfl_factor(4) == pytest.approx(7.0 / 6.0)

    @pytest.mark.parametrize("order", SUPPORTED_CURL_ORDERS)
    def test_equals_l1_norm_and_grows_with_order(self, order):
        coeffs = staggered_stencil_coefficients(order)
        assert stencil_cfl_factor(order) == pytest.approx(float(np.sum(np.abs(coeffs))))
        assert stencil_cfl_factor(order) >= 1.0

    def test_monotone_in_order(self):
        factors = [stencil_cfl_factor(o) for o in SUPPORTED_CURL_ORDERS]
        assert all(b > a for a, b in pairwise(factors))
