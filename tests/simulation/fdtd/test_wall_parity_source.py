"""Simulation tests: sources and fields next to a PEC/PMC mirror wall.

A wall plane is a *node* row inside its own cell, not a cell edge, so on one of the two faces the
plane cuts the wall cell in half:

* an electric plane sits at the lower edge of its cell, so a ``"+"`` (max) face leaves the upper
  half of its wall cell outside the domain;
* a magnetic plane sits at the centre of its cell, so a ``"-"`` (min) face leaves the lower half of
  its wall cell outside the domain.

Two things follow, and both are checked here. First, nothing placed in that half-cell can drive the
simulation: a source there is either overwritten by the wall's image halo or held at zero on the
plane, so the run completes, stays finite and radiates nothing — :class:`TestDeadWallCell` checks
that this is now a placement error rather than a silent zero. Second, the samples in that half-cell
are images of the first interior cell and must be filled as such at *every* curl order; with a zero
halo they instead decouple into a closed subsystem (at a min-face PMC, ``Ex``, ``Ey`` and ``Hz`` of
cell 0 drive only each other) that no interior field reaches yet that any full-volume detector still
reports. :class:`TestWallCellMirror` checks the image, and :class:`TestQuarterMatchesFullDomain`
checks that the interior is the same physics as an explicit full-domain run.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.core.switch import OnOffSwitch
from fdtdx.objects.detectors.field import FieldDetector

_WAVELENGTH = 0.78e-6
_RESOLUTION = 32e-9
_PML = 6
_NX = 24
_NY = 16
_NZ = 13  # 1 PMC wall cell + 12 interior cells
_NT = 320


def _config(curl_order, nt=_NT, shape=(_NX, _NY, _NZ)):
    grid = fdtdx.UniformGrid(spacing=_RESOLUTION).resolve(shape)
    config = fdtdx.SimulationConfig(
        time=1.0,
        grid=grid,
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        curl_order=curl_order,
        gradient_config=None,
    )
    return config.aset("time", nt * float(config.time_step_duration))


def _dipole(name="src"):
    return fdtdx.PointDipoleSource(
        temporal_profile=fdtdx.SingleFrequencyProfile(),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        polarization=1,  # E_y: tangential to the z wall (even), normal to the y wall (even)
        partial_grid_shape=(1, 1, 1),
        name=name,
    )


def _build(curl_order, boundary_types, src_cells, shape=(_NX, _NY, _NZ), nt=_NT, pml_faces=None, omit=()):
    """Place a box with the given per-face boundaries and one E_y dipole per entry of ``src_cells``.

    ``omit`` drops boundary objects entirely (``{"min_z"}``), leaving that face open — the only way
    to express it, since a zero-thickness PML fails constraint resolution.
    """
    config = _config(curl_order, nt=nt, shape=shape)
    volume = fdtdx.SimulationVolume(partial_grid_shape=shape)
    faces = ("minx", "maxx", "miny", "maxy", "minz", "maxz")
    pml_faces = faces if pml_faces is None else pml_faces
    bounds, constraints = fdtdx.boundary_objects_from_config(
        fdtdx.BoundaryConfig(
            **{f"boundary_type_{f}": boundary_types.get(f, "pml") for f in faces},
            **{f"thickness_grid_{f}": (_PML if f in pml_faces else 1) for f in faces},
        ),
        volume,
    )
    dropped = {bounds[name].name for name in omit}
    bounds = {k: v for k, v in bounds.items() if k not in omit}
    constraints = [c for c in constraints if c.object not in dropped]
    detector = FieldDetector(
        partial_grid_shape=shape,
        components=("Ey",),
        reduce_volume=False,
        switch=OnOffSwitch(fixed_on_time_steps=[int(nt) - 1]),
        name="F",
    )
    objects, constraints = [volume, detector] + list(bounds.values()), list(constraints)
    constraints.append(detector.place_at_center(volume))
    for i, cell in enumerate(src_cells):
        source = _dipole(f"src{i}")
        objects.append(source)
        constraints.append(
            source.place_relative_to(
                volume,
                axes=(0, 1, 2),
                own_positions=(-1, -1, -1),
                other_positions=(-1, -1, -1),
                grid_margins=tuple(int(c) for c in cell),
            )
        )
    key = jax.random.PRNGKey(0)
    return fdtdx.place_objects(object_list=objects, config=config, constraints=constraints, key=key), key


def _run(curl_order, boundary_types, src_cells, **kwargs):
    E, H, _ = _run_all(curl_order, boundary_types, src_cells, **kwargs)
    return E, H


def _run_all(curl_order, boundary_types, src_cells, **kwargs):
    """Raw Yee E and H plus the co-located ``E_y`` the FieldDetector recorded."""
    (objects, arrays, _, config, _), key = _build(curl_order, boundary_types, src_cells, **kwargs)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key, show_progress=False)
    detected = np.asarray(arrays.detector_states["F"]["fields"])
    detected = detected.reshape((-1,) + detected.shape[-4:])[-1][0]
    return np.asarray(arrays.fields.E), np.asarray(arrays.fields.H), detected


#: The mirror corner of the atom-trap cavity: E_y needs an electric wall on y and a magnetic one on z.
_MIRROR_CORNER = {"boundary_type_miny": "pec", "boundary_type_minz": "pmc"}
_CORNER = {"miny": "pec", "minz": "pmc"}
_CORNER_PML = ("minx", "maxx", "maxy", "maxz")


class TestDeadWallCell:
    """A source in the half-cell outside a wall plane is rejected instead of radiating nothing."""

    @pytest.mark.parametrize("curl_order", [2, 4])
    def test_dipole_in_the_min_pmc_wall_cell_is_rejected(self, curl_order):
        with pytest.raises(ValueError, match="lies entirely inside the wall cell"):
            _build(curl_order, _CORNER, [(_NX // 2, 0, 0)], pml_faces=_CORNER_PML)

    @pytest.mark.parametrize("curl_order", [2, 4])
    def test_dipole_in_the_min_pec_wall_cell_is_allowed(self, curl_order):
        """The electric plane is at the *lower* edge of its cell, so a min-face PEC has no dead cell.

        ``E_y`` there sits half a cell above the plane — inside the domain — which is exactly how the
        mirror corner of a reduced cavity is meant to be driven.
        """
        E, _ = _run(curl_order, _CORNER, [(_NX // 2, 0, 1)], pml_faces=_CORNER_PML)
        assert np.isfinite(E).all()
        assert np.abs(E[1, :, :, 2:]).max() > 1e-4  # radiates past the walls' own cells

    @pytest.mark.parametrize("curl_order", [2, 4])
    def test_open_face_is_rejected_above_order_two(self, curl_order):
        """An open face is an implicit order-2 mirror; a wider stencil reads past it."""
        types = {"miny": "pec"}
        ctx = (
            pytest.raises(ValueError, match="requires every outer face to be terminated")
            if curl_order > 2
            else _does_not_raise()
        )
        with ctx:
            _build(curl_order, types, [(_NX // 2, 0, 1)], pml_faces=_CORNER_PML, omit={"min_z"})


class _does_not_raise:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class TestWallCellMirror:
    """The half-cell outside a wall carries the image of the first interior cell, at every order."""

    @pytest.mark.parametrize("curl_order", [2, 4])
    def test_min_pmc_wall_cell_mirrors_the_first_interior_cell(self, curl_order):
        """Across a magnetic plane at z = 1/2, ``E_x``, ``E_y`` and ``H_z`` are even: cell 0 = cell 1.

        Without the image halo those three samples form a closed 2D system decoupled from the
        domain, and this equality fails while the run reports no error.
        """
        E, H = _run(curl_order, _CORNER, [(_NX // 2, 0, 3)], pml_faces=_CORNER_PML)
        scale = max(np.abs(E[1, :, :, 1]).max(), 1e-30)
        assert scale > 1e-4  # the comparison is meaningless if nothing propagated
        for field, component, name in ((E, 0, "Ex"), (E, 1, "Ey"), (H, 2, "Hz")):
            got, want = field[component, :, :, 0], field[component, :, :, 1]
            assert np.abs(got - want).max() / scale < 1e-5, f"{name} is not the mirror of cell 1"

    @pytest.mark.parametrize("curl_order", [2, 4])
    def test_min_pmc_wall_cell_odd_components_vanish_on_the_plane(self, curl_order):
        """``E_z``, ``H_x`` and ``H_y`` sample the plane itself, where a magnetic wall zeroes them."""
        E, H = _run(curl_order, _CORNER, [(_NX // 2, 0, 3)], pml_faces=_CORNER_PML)
        scale = max(np.abs(E[1, :, :, 1]).max(), 1e-30)
        assert np.abs(H[0, :, :, 0]).max() / scale < 1e-6
        assert np.abs(H[1, :, :, 0]).max() / scale < 1e-6
        assert np.abs(E[2, :, :, 0]).max() / scale < 1e-5


class TestQuarterMatchesFullDomain:
    """A min-face PMC reproduces the matching half of an explicit full-domain run."""

    @pytest.mark.parametrize("curl_order", [2, 4])
    def test_pmc_half_matches_the_mirrored_full_domain(self, curl_order):
        """Reduced cells ``1..NZ-1`` are the physical domain; the full run mirrors them about z=1/2.

        Reduced cell ``c`` maps to full cell ``c + NZ - 2``, so the dipole at reduced ``k=3`` becomes
        a pair at full ``k in {NZ-2-2, NZ-2+3}`` — its own image across the centre of the full box.
        """
        nz_full = 2 * (_NZ - 1)
        shift, k = _NZ - 2, 3
        reduced_E, _ = _run(
            curl_order, _CORNER, [(_NX // 2, 4, k)], pml_faces=_CORNER_PML
        )
        full_E, _ = _run(
            curl_order,
            {"miny": "pec"},
            [(_NX // 2, 4, shift + k), (_NX // 2, 4, shift + 1 - k)],
            shape=(_NX, _NY, nz_full),
            pml_faces=("minx", "maxx", "maxy", "minz", "maxz"),
        )
        got = reduced_E[..., 1:]
        want = full_E[..., shift + 1 : shift + _NZ]
        assert got.shape == want.shape
        rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-30)
        assert rel < 2e-3, f"quarter/full mismatch {rel:.3e}"


class TestDetectorAtAWall:
    """An ``exact_interpolation`` detector on a wall row records the field, not half of it.

    The co-location stencil takes a backward half-step average along each axis, so for a detector
    touching a wall it averages the first cell against the halo. With a zero halo that is exactly
    half the field in the wall row; the halo's true value is the wall's parity image, which is set by
    the wall condition and not by whatever lies behind the wall.
    """

    @pytest.mark.parametrize("curl_order", [2, 4])
    def test_pec_wall_row_matches_the_mirrored_full_domain(self, curl_order):
        """Quarter (PEC on min_y) vs a full box with the dipole and its image about y = 0.

        ``E_y`` is normal to the electric plane, hence even, and sampled half a cell off it, so
        reduced row ``j`` mirrors to reduced ``-1-j`` and the full-domain pair sits at
        ``{NY+j0, NY-1-j0}``.
        """
        j0 = 2
        _, _, quarter = _run_all(
            curl_order, {"miny": "pec"}, [(_NX // 2, j0, _NZ // 2)],
            pml_faces=("minx", "maxx", "maxy", "minz", "maxz"),
        )
        _, _, full = _run_all(
            curl_order, {},
            [(_NX // 2, _NY + j0, _NZ // 2), (_NX // 2, _NY - 1 - j0, _NZ // 2)],
            shape=(_NX, 2 * _NY, _NZ),
        )
        full = full[:, _NY:, :]
        assert quarter.shape == full.shape
        scale = max(np.abs(full).max(), 1e-30)
        assert scale > 1e-4
        rel_wall_row = np.abs(quarter[:, 0, :] - full[:, 0, :]).max() / scale
        rel_all = np.abs(quarter - full).max() / scale
        # The wall row is the one a zero halo would halve, so pin it separately.
        assert rel_wall_row < 5e-3, f"wall row mismatch {rel_wall_row:.3e}"
        assert rel_all < 5e-3, f"detector mismatch {rel_all:.3e}"
