"""Unit tests for the image (mirror-with-parity) halos of PEC/PMC walls and the Bloch phase halo.

At order 2 the curl never reads a wall's halo, so a zero halo is exact and PEC/PMC leave the
padding alone. From order 4 on, the stencil reaches ``r`` cells past the wall plane and the halo
must carry the parity-weighted mirror image of the interior. These tests check that image against
an independently written index formula, for both wall types, both faces and both fields.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.misc import pad_fields
from fdtdx.core.physics.symmetry import field_component_parity
from fdtdx.objects.boundaries.bloch import BlochBoundary
from fdtdx.objects.boundaries.pec import PerfectElectricConductor
from fdtdx.objects.boundaries.pmc import PerfectMagneticConductor

VOLUME = (8, 9, 10)
SPACING = 50e-9


@pytest.fixture
def config():
    return SimulationConfig(
        time=100e-15,
        grid=UniformGrid(spacing=SPACING),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        gradient_config=None,
    )


def _place(boundary, config, volume_shape=VOLUME):
    axis, direction = boundary.axis, boundary.direction
    slices = [[0, volume_shape[i]] for i in range(3)]
    slices[axis] = [0, 1] if direction == "-" else [volume_shape[axis] - 1, volume_shape[axis]]
    return boundary.place_on_grid(
        grid_slice_tuple=tuple(tuple(s) for s in slices), config=config, key=jax.random.PRNGKey(0)
    )


def _make_wall(cls, axis, direction, **kwargs):
    shape_list: list[int | None] = [None, None, None]
    shape_list[axis] = 1
    return cls(axis=axis, partial_grid_shape=tuple(shape_list), direction=direction, **kwargs)


def _random_padded(width, shape=VOLUME):
    field = jax.random.normal(jax.random.PRNGKey(7), (3, *shape))
    return field, pad_fields(field, (False, False, False), width=width)


def _half_offset(field_type, component, axis):
    """Yee offset of a component along ``axis``: 1/2 for E_c iff c == axis, for H_c iff c != axis."""
    return 0.5 if ((component == axis) == (field_type == "E")) else 0.0


def _expected_image(padded, axis, width, field_type, wall, plane, side):
    """Reference halo: parity * value at the mirrored sample, by explicit index arithmetic."""
    arr = np.array(padded)
    out = arr.copy()
    n_total = arr.shape[axis + 1]
    for component in range(3):
        parity = field_component_parity(field_type, component, axis, wall)
        off = _half_offset(field_type, component, axis)
        for p in range(n_total):
            x = p - width + off
            if not ((x < plane) if side == "-" else (x > plane)):
                continue
            source = round(2 * plane - x - off + width)
            dst = [slice(None)] * 4
            src = [slice(None)] * 4
            dst[0] = src[0] = component
            dst[axis + 1] = p
            src[axis + 1] = source
            for other in range(3):
                if other != axis:
                    interior = slice(width, arr.shape[other + 1] - width)
                    dst[other + 1] = interior
                    src[other + 1] = interior
            out[tuple(dst)] = parity * arr[tuple(src)]
    return out


def _interior_cross_section(padded, axis, width):
    index = [slice(None)] * 4
    for other in range(3):
        if other != axis:
            index[other + 1] = slice(width, padded.shape[other + 1] - width)
    return tuple(index)


WALLS = [
    (PerfectElectricConductor, -1, "-"),
    (PerfectElectricConductor, -1, "+"),
    (PerfectMagneticConductor, 1, "-"),
    (PerfectMagneticConductor, 1, "+"),
]


class TestPecPmcImageHalo:
    """The halo beyond a wall equals parity x mirror, for both fields and both faces."""

    @pytest.mark.parametrize("cls,wall,direction", WALLS)
    @pytest.mark.parametrize("axis", [0, 1, 2])
    @pytest.mark.parametrize("field_type", ["E", "H"])
    def test_matches_reference_image(self, config, cls, wall, direction, axis, field_type):
        width = 2
        boundary = _place(_make_wall(cls, axis, direction), config)
        _, padded = _random_padded(width)
        got = boundary.apply_pad_correction(padded, VOLUME, SPACING, width=width, field_type=field_type)

        # The wall cell is cell 0 on a "-" face and cell N-1 on a "+" face; an electric wall zeroes
        # tangential E at that cell's lower edge, a magnetic one zeroes tangential H half a cell up.
        wall_cell = 0 if direction == "-" else VOLUME[axis] - 1
        plane = wall_cell + (0.5 if wall == 1 else 0.0)
        want = _expected_image(padded, axis, width, field_type, wall, plane, direction)

        sub = _interior_cross_section(padded, axis, width)
        assert np.allclose(np.array(got)[sub], want[sub], atol=0, rtol=0)
        assert not np.allclose(np.array(got)[sub], np.array(padded)[sub])  # something was written

    @pytest.mark.parametrize("cls,wall,direction", WALLS)
    @pytest.mark.parametrize("field_type", ["E", "H"])
    def test_width_one_is_a_noop(self, config, cls, wall, direction, field_type):
        boundary = _place(_make_wall(cls, 1, direction), config)
        _, padded = _random_padded(1)
        got = boundary.apply_pad_correction(padded, VOLUME, SPACING, width=1, field_type=field_type)
        assert jnp.array_equal(got, padded)

    @pytest.mark.parametrize("cls,wall,direction", WALLS)
    def test_without_field_type_is_a_noop(self, config, cls, wall, direction):
        boundary = _place(_make_wall(cls, 2, direction), config)
        _, padded = _random_padded(2)
        assert jnp.array_equal(boundary.apply_pad_correction(padded, VOLUME, SPACING, width=2), padded)

    def test_interior_is_untouched_except_the_wall_cell_half_samples(self, config):
        """A PEC on the min face only rewrites the ghost cells, never the domain."""
        width = 2
        boundary = _place(_make_wall(PerfectElectricConductor, 0, "-"), config)
        field, padded = _random_padded(width)
        got = boundary.apply_pad_correction(padded, VOLUME, SPACING, width=width, field_type="E")
        assert jnp.array_equal(got[:, width:-width, width:-width, width:-width], field)

    def test_pec_plus_face_overwrites_the_exterior_half_samples(self, config):
        """The PEC '+' plane sits at cell N-1, so the half-offset samples there are exterior."""
        width = 2
        boundary = _place(_make_wall(PerfectElectricConductor, 0, "+"), config)
        _, padded = _random_padded(width)
        got = boundary.apply_pad_correction(padded, VOLUME, SPACING, width=width, field_type="E")
        last = width + VOLUME[0] - 1  # padded index of cell N-1
        sub = _interior_cross_section(padded, 0, width)[2:]
        # E_x (normal, even parity, sampled at N-1/2) mirrors to cell N-2 ...
        assert jnp.allclose(got[0, last][sub], padded[0, last - 1][sub])
        # ... while tangential E_y/E_z sit on the plane and are left alone.
        assert jnp.array_equal(got[1, last][sub], padded[1, last][sub])
        assert jnp.array_equal(got[2, last][sub], padded[2, last][sub])

    def test_pmc_minus_face_overwrites_the_exterior_integer_samples(self, config):
        """The PMC '-' plane sits half a cell out, so cell 0's integer samples are exterior."""
        width = 2
        axis = 1
        boundary = _place(_make_wall(PerfectMagneticConductor, axis, "-"), config)
        _, padded = _random_padded(width)
        got = boundary.apply_pad_correction(padded, VOLUME, SPACING, width=width, field_type="E")
        cross = _interior_cross_section(padded, axis, width)[2:4]
        first = width  # padded index of cell 0
        # tangential E_x/E_z (even under a magnetic wall) mirror from cell 1
        for component in (0, 2):
            assert jnp.allclose(got[component, :, first][cross], padded[component, :, first + 1][cross])
        # normal E_y sits half a cell off the integer grid and stays where it is
        assert jnp.array_equal(got[1, :, first][cross], padded[1, :, first][cross])

    @pytest.mark.parametrize("cls,wall,direction", WALLS)
    def test_uniform_field_halo_carries_the_component_parity(self, config, cls, wall, direction):
        """A constant field images to +-1 in the halo, one sign per component parity."""
        width = 3
        boundary = _place(_make_wall(cls, 0, direction), config)
        padded = pad_fields(jnp.ones((3, *VOLUME)), (False, False, False), width=width)
        cross = _interior_cross_section(padded, 0, width)[2:4]
        for field_type in ("E", "H"):
            got = boundary.apply_pad_correction(padded, VOLUME, SPACING, width=width, field_type=field_type)
            ghosts = slice(0, width) if direction == "-" else slice(padded.shape[1] - width, None)
            for component in range(3):
                parity = field_component_parity(field_type, component, 0, wall)
                assert jnp.allclose(got[component, ghosts][(slice(None), *cross)], float(parity))


class TestBlochPhaseHalo:
    """The Bloch phase is applied to every ghost layer, not just the outermost one."""

    def _bloch(self, config, axis=0, direction="-", k=2.0e6):
        vector = [0.0, 0.0, 0.0]
        vector[axis] = k
        return _place(_make_wall(BlochBoundary, axis, direction, bloch_vector=tuple(vector)), config)

    @pytest.mark.parametrize("direction", ["-", "+"])
    @pytest.mark.parametrize("width", [1, 2, 3])
    def test_all_ghost_layers_get_the_phase(self, config, direction, width):
        boundary = self._bloch(config, axis=0, direction=direction)
        field = jax.random.normal(jax.random.PRNGKey(5), (3, *VOLUME)) + 0j
        padded = pad_fields(field, (True, False, False), width=width)
        got = boundary.apply_pad_correction(padded, VOLUME, SPACING, width=width, field_type="E")

        phase = boundary.get_bloch_phase(VOLUME, SPACING)
        factor = jnp.conj(phase) if direction == "-" else phase
        ghosts = slice(0, width) if direction == "-" else slice(padded.shape[1] - width, None)
        assert jnp.allclose(got[:, ghosts], padded[:, ghosts] * factor)
        # the rest of the array is untouched
        interior = slice(width, padded.shape[1] - width)
        assert jnp.array_equal(got[:, interior], padded[:, interior])

    def test_width_one_matches_the_legacy_single_ghost_behaviour(self, config):
        boundary = self._bloch(config, axis=1, direction="+")
        field = jax.random.normal(jax.random.PRNGKey(6), (3, *VOLUME)) + 0j
        padded = pad_fields(field, (False, True, False), width=1)
        got = boundary.apply_pad_correction(padded, VOLUME, SPACING)
        phase = boundary.get_bloch_phase(VOLUME, SPACING)
        expected = padded.at[:, :, -1].set(padded[:, :, -1] * phase)
        assert jnp.array_equal(got, expected)

    def test_zero_bloch_vector_is_a_noop(self, config):
        boundary = self._bloch(config, axis=0, direction="-", k=0.0)
        _, padded = _random_padded(2)
        assert jnp.array_equal(boundary.apply_pad_correction(padded, VOLUME, SPACING, width=2, field_type="E"), padded)
