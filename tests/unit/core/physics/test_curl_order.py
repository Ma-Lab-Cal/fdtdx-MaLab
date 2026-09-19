"""Unit tests for the order-``config.curl_order`` staggered curl stencils.

Three things are checked here:

* the order-2 path is **bit-identical** to the classic Yee implementation (a verbatim copy of it
  lives in this file), with and without a PML correction;
* the discrete derivative converges at the requested order on a smooth periodic field;
* the halo width of the padding helpers follows ``config.curl_stencil_radius``.
"""

from contextlib import contextmanager
from itertools import pairwise
from unittest.mock import MagicMock, Mock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.misc import pad_fields
from fdtdx.core.physics.curl import _metric_scale, curl_E, curl_H
from fdtdx.fdtd.update import pad_fields_for_boundaries


@contextmanager
def _x64_enabled():
    """Scoped float64 enable: the order-6/8 truncation error is below the float32 noise floor."""
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _make_config(curl_order=2, spacing=1.0):
    return SimulationConfig(
        time=400e-15,
        grid=UniformGrid(spacing=spacing),
        courant_factor=0.99,
        curl_order=curl_order,
    )


def _mock_objects(pml_objects=None):
    objects = MagicMock()
    objects.pml_objects = pml_objects or []
    return objects


# ──────────────────────────────────────────────────────────────
# Verbatim copy of the pre-higher-order (classic Yee) implementation
# ──────────────────────────────────────────────────────────────


def _legacy_curl_E(config, E_pad, psi_H, objects, simulate_boundaries):
    shape = E_pad.shape[1] - 2, E_pad.shape[2] - 2, E_pad.shape[3] - 2
    dx_scale = _metric_scale(config, axis=0, shape=shape, stencil="forward")
    dy_scale = _metric_scale(config, axis=1, shape=shape, stencil="forward")
    dz_scale = _metric_scale(config, axis=2, shape=shape, stencil="forward")

    Ex = E_pad[0]
    Ey = E_pad[1]
    Ez = E_pad[2]
    center = (slice(1, -1), slice(1, -1), slice(1, -1))

    dyEz = (Ez[1:-1, 2:, 1:-1] - Ez[center]) * dy_scale
    dzEy = (Ey[1:-1, 1:-1, 2:] - Ey[center]) * dz_scale
    dzEx = (Ex[1:-1, 1:-1, 2:] - Ex[center]) * dz_scale
    dxEz = (Ez[2:, 1:-1, 1:-1] - Ez[center]) * dx_scale
    dxEy = (Ey[2:, 1:-1, 1:-1] - Ey[center]) * dx_scale
    dyEx = (Ex[1:-1, 2:, 1:-1] - Ex[center]) * dy_scale

    curl_components = [dyEz - dzEy, dzEx - dxEz, dxEy - dyEx]
    psi_H_updated = {}
    for pml in objects.pml_objects:
        a = pml.axis
        i, j = (a + 1) % 3, (a + 2) % 3
        if a == 0:
            d_a_F_j, d_a_F_i = dxEz, dxEy
        elif a == 1:
            d_a_F_j, d_a_F_i = dyEx, dyEz
        else:
            d_a_F_j, d_a_F_i = dzEy, dzEx
        corr_1, corr_2, psi_1_new, psi_2_new = pml.step_cpml(
            d_a_F_j[pml.grid_slice],
            d_a_F_i[pml.grid_slice],
            *psi_H[pml.name],
            is_curl_E=True,
            simulate_boundaries=simulate_boundaries,
        )
        curl_components[i] = curl_components[i].at[pml.grid_slice].add(-corr_1)
        curl_components[j] = curl_components[j].at[pml.grid_slice].add(corr_2)
        psi_H_updated[pml.name] = (psi_1_new, psi_2_new)
    return jnp.stack(curl_components, axis=0), psi_H_updated


def _legacy_curl_H(config, H_pad, psi_E, objects, simulate_boundaries):
    shape = H_pad.shape[1] - 2, H_pad.shape[2] - 2, H_pad.shape[3] - 2
    dx_scale = _metric_scale(config, axis=0, shape=shape, stencil="backward")
    dy_scale = _metric_scale(config, axis=1, shape=shape, stencil="backward")
    dz_scale = _metric_scale(config, axis=2, shape=shape, stencil="backward")

    Hx = H_pad[0]
    Hy = H_pad[1]
    Hz = H_pad[2]
    center = (slice(1, -1), slice(1, -1), slice(1, -1))

    dyHz = (Hz[center] - Hz[1:-1, :-2, 1:-1]) * dy_scale
    dzHy = (Hy[center] - Hy[1:-1, 1:-1, :-2]) * dz_scale
    dzHx = (Hx[center] - Hx[1:-1, 1:-1, :-2]) * dz_scale
    dxHz = (Hz[center] - Hz[:-2, 1:-1, 1:-1]) * dx_scale
    dxHy = (Hy[center] - Hy[:-2, 1:-1, 1:-1]) * dx_scale
    dyHx = (Hx[center] - Hx[1:-1, :-2, 1:-1]) * dy_scale

    curl_components = [dyHz - dzHy, dzHx - dxHz, dxHy - dyHx]
    psi_E_updated = {}
    for pml in objects.pml_objects:
        a = pml.axis
        i, j = (a + 1) % 3, (a + 2) % 3
        if a == 0:
            d_a_F_j, d_a_F_i = dxHz, dxHy
        elif a == 1:
            d_a_F_j, d_a_F_i = dyHx, dyHz
        else:
            d_a_F_j, d_a_F_i = dzHy, dzHx
        corr_1, corr_2, psi_1_new, psi_2_new = pml.step_cpml(
            d_a_F_j[pml.grid_slice],
            d_a_F_i[pml.grid_slice],
            *psi_E[pml.name],
            is_curl_E=False,
            simulate_boundaries=simulate_boundaries,
        )
        curl_components[i] = curl_components[i].at[pml.grid_slice].add(-corr_1)
        curl_components[j] = curl_components[j].at[pml.grid_slice].add(corr_2)
        psi_E_updated[pml.name] = (psi_1_new, psi_2_new)
    return jnp.stack(curl_components, axis=0), psi_E_updated


def _random_fields(shape=(3, 7, 8, 9)):
    keys = jax.random.split(jax.random.PRNGKey(11), 2)
    return jax.random.normal(keys[0], shape), jax.random.normal(keys[1], shape)


def _fake_pml(axis, n, grid_slice):
    pml = MagicMock()
    pml.name = f"pml_{axis}"
    pml.axis = axis
    pml.grid_slice = grid_slice
    region = jnp.ones(tuple(len(range(*s.indices(dim))) for s, dim in zip(grid_slice, n)))
    pml.step_cpml.return_value = (0.1 * region, 0.2 * region, 0.3 * region, 0.4 * region)
    return pml


# ──────────────────────────────────────────────────────────────
# (a) bit-identity of the order-2 path
# ──────────────────────────────────────────────────────────────


class TestOrderTwoIsBitIdentical:
    """``curl_order=2`` must reproduce the classic Yee scheme exactly, not just closely."""

    @pytest.mark.parametrize("periodic", [(False, False, False), (True, False, True)])
    def test_no_pml(self, periodic):
        config = _make_config(curl_order=2)
        assert config.curl_stencil_radius == 1
        E, H = _random_fields()
        E_pad = pad_fields(E, periodic, width=config.curl_stencil_radius)
        H_pad = pad_fields(H, periodic, width=config.curl_stencil_radius)
        objects = _mock_objects()

        got_E, _ = curl_E(config, E_pad, {}, objects, True)
        want_E, _ = _legacy_curl_E(config, E_pad, {}, objects, True)
        got_H, _ = curl_H(config, H_pad, {}, objects, True)
        want_H, _ = _legacy_curl_H(config, H_pad, {}, objects, True)

        assert jnp.array_equal(got_E, want_E)
        assert jnp.array_equal(got_H, want_H)

    def test_with_pml_object(self):
        config = _make_config(curl_order=2)
        n = (7, 8, 9)
        E, H = _random_fields((3, *n))
        E_pad = pad_fields(E, (False, False, False))
        H_pad = pad_fields(H, (False, False, False))
        grid_slice = (slice(0, 3), slice(None), slice(None))
        psi = {"pml_0": (jnp.zeros((3, n[1], n[2])), jnp.zeros((3, n[1], n[2])))}

        got_E, got_psi_H = curl_E(config, E_pad, psi, _mock_objects([_fake_pml(0, n, grid_slice)]), False)
        want_E, want_psi_H = _legacy_curl_E(config, E_pad, psi, _mock_objects([_fake_pml(0, n, grid_slice)]), False)
        got_H, got_psi_E = curl_H(config, H_pad, psi, _mock_objects([_fake_pml(0, n, grid_slice)]), False)
        want_H, want_psi_E = _legacy_curl_H(config, H_pad, psi, _mock_objects([_fake_pml(0, n, grid_slice)]), False)

        assert jnp.array_equal(got_E, want_E)
        assert jnp.array_equal(got_H, want_H)
        assert jnp.array_equal(got_psi_H["pml_0"][0], want_psi_H["pml_0"][0])
        assert jnp.array_equal(got_psi_E["pml_0"][1], want_psi_E["pml_0"][1])

    def test_jaxpr_is_literally_unchanged(self):
        """Not just equal outputs: the traced program is the same, op for op."""
        config = _make_config(curl_order=2)
        objects = _mock_objects()
        pad = pad_fields(jnp.zeros((3, 7, 8, 9)), (False, False, False))
        got_E = jax.make_jaxpr(lambda a: curl_E(config, a, {}, objects, True)[0])(pad)
        want_E = jax.make_jaxpr(lambda a: _legacy_curl_E(config, a, {}, objects, True)[0])(pad)
        got_H = jax.make_jaxpr(lambda a: curl_H(config, a, {}, objects, True)[0])(pad)
        want_H = jax.make_jaxpr(lambda a: _legacy_curl_H(config, a, {}, objects, True)[0])(pad)
        assert str(got_E) == str(want_E)
        assert str(got_H) == str(want_H)

    def test_higher_order_actually_differs(self):
        """Guards against the order-2 identity being trivially true for every order."""
        E, H = _random_fields()
        order4 = _make_config(curl_order=4)
        E_pad = pad_fields(E, (True, True, True), width=2)
        H_pad = pad_fields(H, (True, True, True), width=2)
        got_E, _ = curl_E(order4, E_pad, {}, _mock_objects(), True)
        got_H, _ = curl_H(order4, H_pad, {}, _mock_objects(), True)
        legacy_E, _ = _legacy_curl_E(_make_config(), pad_fields(E, (True, True, True)), {}, _mock_objects(), True)
        legacy_H, _ = _legacy_curl_H(_make_config(), pad_fields(H, (True, True, True)), {}, _mock_objects(), True)
        assert not jnp.allclose(got_E, legacy_E)
        assert not jnp.allclose(got_H, legacy_H)


# ──────────────────────────────────────────────────────────────
# (b) convergence order on a smooth periodic field
# ──────────────────────────────────────────────────────────────


def _derivative_error(order, n, kind):
    """Max error of the discrete d/dx of a sine, sampled on a periodic grid of ``n`` cells."""
    config = _make_config(curl_order=order)
    r = config.curl_stencil_radius
    h = 2.0 * np.pi / n
    x = jnp.arange(n, dtype=jnp.float64) * h
    transverse = jnp.ones((n, 3, 3), dtype=jnp.float64)
    zeros = jnp.zeros((n, 3, 3), dtype=jnp.float64)
    if kind == "E":
        # E_z sampled at integer x; the staggered derivative lands at x + h/2.
        field = jnp.stack([zeros, zeros, jnp.sin(x)[:, None, None] * transverse], axis=0)
        curl, _ = curl_E(config, pad_fields(field, (True, True, True), width=r), {}, _mock_objects(), True)
        expected = jnp.cos(x + h / 2)
    else:
        # H_z sampled at x + h/2; the staggered derivative lands at integer x.
        field = jnp.stack([zeros, zeros, jnp.sin(x + h / 2)[:, None, None] * transverse], axis=0)
        curl, _ = curl_H(config, pad_fields(field, (True, True, True), width=r), {}, _mock_objects(), True)
        expected = jnp.cos(x)
    # curl_y = d_z F_x - d_x F_z = -d_x F_z here.
    got = -curl[1][:, 1, 1] / h
    return float(jnp.max(jnp.abs(got - expected)))


class TestConvergenceOrder:
    """Halving the cell size must reduce the truncation error by ``2**order``."""

    @pytest.mark.parametrize("order", [2, 4, 6])
    @pytest.mark.parametrize("kind", ["E", "H"])
    def test_observed_order(self, order, kind):
        with _x64_enabled():
            coarse = _derivative_error(order, 16, kind)
            fine = _derivative_error(order, 32, kind)
        observed = np.log2(coarse / fine)
        assert observed >= order - 0.3, f"order {order} ({kind}): observed {observed:.2f} ({coarse}, {fine})"

    def test_higher_order_is_more_accurate_at_equal_resolution(self):
        with _x64_enabled():
            errors = [_derivative_error(order, 16, "E") for order in (2, 4, 6, 8)]
        assert all(b < a for a, b in pairwise(errors))


# ──────────────────────────────────────────────────────────────
# (c) halo widths
# ──────────────────────────────────────────────────────────────


class TestPaddingWidth:
    """``pad_fields`` / ``pad_fields_for_boundaries`` grow the halo with the stencil radius."""

    def test_pad_fields_width_two_shape_and_wrap(self):
        field = jnp.arange(3 * 4 * 5 * 6, dtype=jnp.float32).reshape(3, 4, 5, 6)
        padded = pad_fields(field, (True, False, False), width=2)
        assert padded.shape == (3, 8, 9, 10)
        # wrap on axis 0: ghosts 0,1 come from the last two cells
        assert jnp.array_equal(padded[:, 0, 2:-2, 2:-2], field[:, -2])
        assert jnp.array_equal(padded[:, 1, 2:-2, 2:-2], field[:, -1])
        # constant zero on axes 1 and 2
        assert jnp.all(padded[:, :, :2, :] == 0)
        assert jnp.all(padded[:, :, :, -2:] == 0)

    def test_pad_fields_rejects_zero_width(self):
        with pytest.raises(ValueError):
            pad_fields(jnp.zeros((3, 2, 2, 2)), (False, False, False), width=0)

    @pytest.mark.parametrize("width", [1, 2, 3])
    def test_pad_fields_for_boundaries_shape(self, width):
        objects = Mock()
        objects.boundary_objects = []
        objects.volume.grid_shape = (5, 6, 7)
        config = Mock()
        config.symmetry = (0, 0, 0)
        config.has_nonuniform_grid = False
        config.uniform_spacing.return_value = 1.0
        field = jnp.ones((3, 5, 6, 7))
        padded = pad_fields_for_boundaries(field, objects, config, width=width, field_type="E")
        assert padded.shape == (3, 5 + 2 * width, 6 + 2 * width, 7 + 2 * width)
        assert jnp.array_equal(padded[:, width:-width, width:-width, width:-width], field)

    def test_curl_rejects_wrong_halo(self):
        config = _make_config(curl_order=4)
        # only a one-cell halo: the order-4 curl would silently shrink the output
        E_pad = pad_fields(jnp.zeros((3, 2, 2, 2)), (False, False, False), width=1)
        with pytest.raises(ValueError):
            curl_E(config, E_pad, {}, _mock_objects(), True)
