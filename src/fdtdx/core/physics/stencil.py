"""Finite-difference stencil coefficients for the staggered (Yee) grid.

The curl operators in :mod:`fdtdx.core.physics.curl` approximate a first derivative at the
half-integer point between two samples of the same field component. On a uniform grid the
optimal central stencil of order ``2r`` uses ``r`` antisymmetric taps on each side::

    d f/dx |_{i+1/2} ≈ (1/dx) * sum_{m=1}^{r} a_m * (f[i+m] - f[i-m+1])

with the classical staggered-grid coefficients (Fornberg). Order 2 reduces to the plain
Yee difference ``f[i+1] - f[i]``.

Only even orders are meaningful for a centered staggered stencil. This module carries no
JAX dependency so it can be imported by :mod:`fdtdx.config` without circular imports.
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

#: Spatial derivative orders supported by the curl operators.
SUPPORTED_CURL_ORDERS: tuple[int, ...] = (2, 4, 6, 8)


@lru_cache(maxsize=None)
def staggered_stencil_coefficients(order: int) -> tuple[float, ...]:
    """Return the staggered central-difference coefficients ``(a_1, ..., a_r)`` for ``order = 2r``.

    The coefficients are computed exactly with rational arithmetic and returned as floats::

        order 2: (1,)
        order 4: (9/8, -1/24)
        order 6: (75/64, -25/384, 3/640)
        order 8: (1225/1024, -245/3072, 49/5120, -5/7168)

    Args:
        order (int): Even derivative order ``2r`` with ``r >= 1``.

    Returns:
        tuple[float, ...]: The ``r`` tap weights ``a_m`` multiplying ``f[i+m] - f[i-m+1]``.

    Raises:
        ValueError: If ``order`` is not a positive even integer.
    """
    if not isinstance(order, int) or order < 2 or order % 2 != 0:
        raise ValueError(f"Stencil order must be a positive even integer, got {order!r}")
    r = order // 2
    coeffs: list[float] = []
    for m in range(1, r + 1):
        # Closed form of the staggered-grid weights (Fornberg 1988):
        #   a_m = (-1)^(m+1) / (2m - 1) * prod_{n != m} (2n-1)^2 / |(2m-1)^2 - (2n-1)^2|
        num = Fraction((-1) ** (m + 1), 2 * m - 1)
        for n in range(1, r + 1):
            if n == m:
                continue
            num *= Fraction((2 * n - 1) ** 2, abs((2 * m - 1) ** 2 - (2 * n - 1) ** 2))
        coeffs.append(float(num))
    return tuple(coeffs)


def stencil_radius(order: int) -> int:
    """Number of taps on each side of the staggered derivative point for ``order``.

    Args:
        order (int): Even derivative order.

    Returns:
        int: ``order // 2``, the halo width the curl operators need on every face.
    """
    if order not in SUPPORTED_CURL_ORDERS:
        raise ValueError(f"curl_order must be one of {SUPPORTED_CURL_ORDERS}, got {order!r}")
    return order // 2


def stencil_cfl_factor(order: int) -> float:
    """Reduction of the Courant stability limit for a stencil of the given order.

    The von Neumann limit of the leapfrog scheme with a spatial stencil of taps ``a_m`` is
    ``c dt / dx <= 1 / (sqrt(D) * sum_m |a_m|)`` in ``D`` dimensions. This returns
    ``sum_m |a_m|`` (``1`` for order 2, ``7/6`` for order 4, ...), so the stable 3D Courant
    number is ``1 / (sqrt(3) * stencil_cfl_factor(order))``.

    Args:
        order (int): Even derivative order.

    Returns:
        float: ``sum_m |a_m| >= 1``.
    """
    return float(sum(abs(a) for a in staggered_stencil_coefficients(order)))
