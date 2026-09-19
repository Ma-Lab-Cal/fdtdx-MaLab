"""Simulation tests for the higher-order (``config.curl_order``) spatial curl stencils.

Three properties are checked at order 4, each of which the order-2 scheme gets for free:

* **Reversibility** — the wider stencil reads further into the PML, so the reversible adjoint has
  to record a thicker interface slab. One forward run followed by ``full_backward`` must return to
  the (zero) initial state.
* **Mirror-plane consistency** — at order 2 the interior never reads the halo of a PEC/PMC wall, so
  a zero halo is exact. At order 4 it reads ``r = 2`` cells across the plane, and only the image
  (mirror-with-parity) halo makes a half-domain run reproduce the symmetric full-domain run. Both
  wall types are checked against a full domain twice the size.
* **Stability and dispersion** — the order-4 time step is ``6/7`` of the order-2 one; a periodic
  box must stay bounded over hundreds of steps and reproduce the plane-wave frequency accurately.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.constants import c as c0
from fdtdx.core.physics.symmetry import field_component_parity
from fdtdx.fdtd.backward import full_backward
from fdtdx.fdtd.forward import forward
from fdtdx.interfaces.recorder import Recorder

# ── Shared constants (everything is deliberately tiny: these run on CPU) ──────────────────────
_SPACING = 50e-9
_WAVELENGTH = 1.0e-6
_PML_CELLS = 4


def _step_forward(state, config, objects, key, record_boundaries=False):
    return forward(
        state=state,
        config=config,
        objects=objects,
        key=key,
        record_detectors=False,
        record_boundaries=record_boundaries,
        simulate_boundaries=True,
    )


def _place(objects, constraints, config, key=None):
    key = jax.random.PRNGKey(0) if key is None else key
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    return obj, arrays, config


# ══════════════════════════════════════════════════════════════════════════════════════════════
# (a) reversibility of the order-4 scheme
# ══════════════════════════════════════════════════════════════════════════════════════════════


def _build_reversible_box(curl_order, cells=12):
    config = fdtdx.SimulationConfig(
        time=1e-15,
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        curl_order=curl_order,
        gradient_config=fdtdx.GradientConfig(method="reversible", recorder=Recorder(modules=[])),
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(cells, cells, cells))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={"min_x": "pec", "max_x": "pec"},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    source = fdtdx.PointDipoleSource(
        name="dipole",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        polarization=2,
        static_amplitude_factor=1.0,
    )
    constraints.append(
        source.set_grid_coordinates(
            axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(cells // 2, cells // 2, cells // 2)
        )
    )
    objects.append(source)
    return _place(objects, constraints, config)


class TestOrderFourReversibility:
    """``full_backward`` must undo the forward run with the wider PML interface record."""

    @pytest.mark.parametrize("curl_order", [2, 4])
    def test_initial_state_is_reconstructed(self, curl_order):
        objects, arrays, config = _build_reversible_box(curl_order)
        key = jax.random.PRNGKey(1)
        steps = config.time_steps_total

        state = (jnp.asarray(0, dtype=jnp.int32), arrays)
        for _ in range(steps):
            state = _step_forward(state, config, objects, key, record_boundaries=True)

        excited = float(jnp.max(jnp.abs(state[1].fields.E)))
        assert excited > 1e-6, "the dipole did not excite the box"

        state = full_backward(
            state=state, objects=objects, config=config, key=key, record_detectors=False, reset_fields=True
        )
        _, reconstructed = state
        assert float(jnp.max(jnp.abs(reconstructed.fields.E))) < 1e-4 * excited
        assert float(jnp.max(jnp.abs(reconstructed.fields.H))) < 1e-4 * excited

    def test_interface_record_is_wider_at_order_four(self):
        from fdtdx.fdtd.update import pml_interface_width

        _, arrays_2, config_2 = _build_reversible_box(2)
        _, arrays_4, config_4 = _build_reversible_box(4)
        assert pml_interface_width(config_2) == 1
        assert pml_interface_width(config_4) == 3
        del arrays_2, arrays_4


# ══════════════════════════════════════════════════════════════════════════════════════════════
# (b) mirror-plane consistency: half domain with a wall == upper half of the full domain
# ══════════════════════════════════════════════════════════════════════════════════════════════

_MIRROR_NX = _MIRROR_NZ = 10
_MIRROR_T = 3
_MIRROR_M = 20  # cells of the reduced (half) domain along y
_MIRROR_STEPS = 12  # < _MIRROR_M, so the full domain's unpaired outermost cell cannot reach the kept half


def _build_mirror_domain(ny, min_y_type, curl_order, dipole_cells, polarization):
    """A vacuum box with a y-invariant dielectric rod and one dipole per entry in ``dipole_cells``.

    The rod spans the whole y extent on purpose: fdtdx rasterizes materials per cell, so a structure
    that is symmetric about a cell boundary is *not* symmetric about the integer node half a cell
    away, and only a y-invariant structure is an exact mirror for every Yee component at once.
    """
    config = fdtdx.SimulationConfig(
        time=200e-15,
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        curl_order=curl_order,
        gradient_config=None,
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_MIRROR_NX, ny, _MIRROR_NZ))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_MIRROR_T, override_types={"min_y": min_y_type})
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    rod = fdtdx.UniformMaterialObject(
        name="rod", material=fdtdx.Material(permittivity=6.0), partial_grid_shape=(4, ny, 4)
    )
    constraints.append(
        rod.set_grid_coordinates(axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(_MIRROR_T, 0, _MIRROR_T))
    )
    objects.append(rod)

    for index, jy in enumerate(dipole_cells):
        source = fdtdx.PointDipoleSource(
            name=f"dipole_{index}",
            partial_grid_shape=(1, 1, 1),
            wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
            polarization=polarization,
            static_amplitude_factor=1.0,
        )
        constraints.append(
            source.set_grid_coordinates(
                axes=(0, 1, 2), sides=("-", "-", "-"), coordinates=(_MIRROR_NX // 2, jy, _MIRROR_NZ // 2)
            )
        )
        objects.append(source)
    return _place(objects, constraints, config)


def _symmetrize_along_y(field, field_type, wall, plane_twice, ny):
    """Project a ``(3, Nx, ny, Nz)`` field onto the mirror-symmetric subspace of a y plane.

    ``plane_twice`` is twice the plane position in cell units. Component ``c`` sits at
    ``y = j + off`` with ``off = 1/2`` for ``E_c`` iff ``c == 1`` and for ``H_c`` iff ``c != 1``, so
    its mirror index is ``(plane_twice - 2 off) - j``. Samples whose mirror falls outside the array
    (the single outermost one of each integer-sampled component) are zeroed.
    """
    source = np.array(field)
    out = np.zeros_like(source)
    for component in range(3):
        parity = field_component_parity(field_type, component, 1, wall)
        two_off = 1 if ((component == 1) == (field_type == "E")) else 0
        mirror = (plane_twice - two_off) - np.arange(ny)
        valid = (mirror >= 0) & (mirror < ny)
        out[component][:, valid, :] = 0.5 * (
            source[component][:, valid, :] + parity * source[component][:, mirror[valid], :]
        )
    return jnp.asarray(out)


def _mirror_case(wall_type, curl_order):
    """Run the full domain and its upper half with a wall, return (full_kept, half) field dicts."""
    wall = -1 if wall_type == "pec" else 1
    if wall == -1:
        # An electric plane sits on the integer node at the reduced domain's min edge.
        offset, full_ny, polarization = _MIRROR_M, 2 * _MIRROR_M, 1
        plane_twice = 2 * offset
        dipole_y = offset + 4
        full_dipoles = (dipole_y, 2 * offset - 1 - dipole_y)  # E_y is half-offset: cell flip
    else:
        # A magnetic plane sits half a cell above the reduced domain's min edge, so the full domain
        # has an odd cell count and its middle cell is bisected by the plane.
        offset, full_ny, polarization = _MIRROR_M - 1, 2 * _MIRROR_M - 1, 0
        plane_twice = 2 * offset + 1
        dipole_y = offset + 4
        full_dipoles = (dipole_y, 2 * offset + 1 - dipole_y)  # E_x is integer-sampled
    half_dipoles = (dipole_y - offset,)

    objects_f, arrays_f, config_f = _build_mirror_domain(full_ny, "pml", curl_order, full_dipoles, polarization)
    objects_h, arrays_h, config_h = _build_mirror_domain(
        full_ny - offset, wall_type, curl_order, half_dipoles, polarization
    )

    key = jax.random.PRNGKey(3)
    k_e, k_h = jax.random.split(key)
    dtype = arrays_f.fields.E.dtype
    E = _symmetrize_along_y(
        1e-3 * jax.random.normal(k_e, arrays_f.fields.E.shape, dtype=dtype), "E", wall, plane_twice, full_ny
    )
    H = _symmetrize_along_y(
        1e-3 * jax.random.normal(k_h, arrays_f.fields.H.shape, dtype=dtype), "H", wall, plane_twice, full_ny
    )
    arrays_f = arrays_f.aset("fields->E", E).aset("fields->H", H)
    arrays_h = arrays_h.aset("fields->E", E[:, :, offset:, :]).aset("fields->H", H[:, :, offset:, :])

    state_f = (jnp.asarray(0, dtype=jnp.int32), arrays_f)
    state_h = (jnp.asarray(0, dtype=jnp.int32), arrays_h)
    for _ in range(_MIRROR_STEPS):
        state_f = _step_forward(state_f, config_f, objects_f, key)
        state_h = _step_forward(state_h, config_h, objects_h, key)

    # A magnetic wall's plane is half a cell inside the domain, so the reduced domain's first cell
    # carries the *exterior* integer-sampled components; they are ghosts and only agree from order 4
    # on (where the image halo defines them). Compare the physical cells.
    skip = 0 if wall == -1 else 1
    full = {n: getattr(state_f[1].fields, n)[:, :, offset + skip :, :] for n in ("E", "H")}
    half = {n: getattr(state_h[1].fields, n)[:, :, skip:, :] for n in ("E", "H")}
    return full, half


def _relative_error(a, b):
    return float(jnp.max(jnp.abs(a - b)) / (jnp.max(jnp.abs(a)) + 1e-30))


class TestMirrorPlaneConsistency:
    """A half domain terminated by a wall reproduces the kept half of the symmetric full domain."""

    @pytest.mark.parametrize("wall_type", ["pec", "pmc"])
    @pytest.mark.parametrize("curl_order", [2, 4])
    def test_half_domain_matches_full_domain(self, wall_type, curl_order):
        full, half = _mirror_case(wall_type, curl_order)
        for name in ("E", "H"):
            assert float(jnp.max(jnp.abs(full[name]))) > 1e-6, f"{name} never got excited"
            error = _relative_error(full[name], half[name])
            assert error < 1e-4, f"{wall_type} order {curl_order}: {name} relative error {error:.2e}"

    @pytest.mark.parametrize("wall_type", ["pec", "pmc"])
    def test_order_four_needs_the_image_halo(self, wall_type, monkeypatch):
        """Without the image halo the same comparison is off by percents, so the test has teeth."""
        from fdtdx.objects.boundaries.boundary import BaseBoundary

        monkeypatch.setattr(BaseBoundary, "_apply_image_halo", lambda self, padded, width, field_type, wall: padded)
        full, half = _mirror_case(wall_type, 4)
        assert max(_relative_error(full[name], half[name]) for name in ("E", "H")) > 1e-2


# ══════════════════════════════════════════════════════════════════════════════════════════════
# (b2) the standing wave a max-face wall reflects, which exercises the "+"-face image halo
# ══════════════════════════════════════════════════════════════════════════════════════════════

_WALL_RES = 25e-9  # 40 cells per wavelength
_WALL_PML = 8
_QUARTER_WAVE = round(_WAVELENGTH / (4 * _WALL_RES))
_WALL_NZ = _WALL_PML + 34 + _QUARTER_WAVE + 2
_WALL_STEPS = 600


def _wall_standing_wave_ratio(wall_type, curl_order):
    """|E_x| one cell from a z-max wall divided by |E_x| a quarter wavelength away.

    A PEC forces a node at the wall (ratio << 1), a PMC an antinode (ratio >> 1). The max face is
    the interesting one for the image halo: its exterior region reaches half a cell into the domain.
    """
    config = fdtdx.SimulationConfig(
        time=500e-15,
        grid=fdtdx.UniformGrid(spacing=_WALL_RES),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        curl_order=curl_order,
        gradient_config=None,
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(3, 3, _WALL_NZ))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_WALL_PML,
        override_types={
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
            "max_z": wall_type,
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)
    source = fdtdx.UniformPlaneSource(
        partial_grid_shape=(None, None, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_WALL_PML + 2,)),
        ]
    )
    objects.append(source)

    objects_c, arrays, config = _place(objects, constraints, config)
    key = jax.random.PRNGKey(0)
    state = (jnp.asarray(0, dtype=jnp.int32), arrays)
    near = far = 0.0
    for step in range(_WALL_STEPS):
        state = _step_forward(state, config, objects_c, key)
        if step >= _WALL_STEPS - 120:  # a few optical periods of steady state
            Ex = state[1].fields.E[0]
            near = max(near, float(jnp.max(jnp.abs(Ex[:, :, _WALL_NZ - 2]))))
            far = max(far, float(jnp.max(jnp.abs(Ex[:, :, _WALL_NZ - 1 - _QUARTER_WAVE]))))
    return near / far


class TestMaxFaceWallReflection:
    """A max-face PEC/PMC must reflect the same way at order 4 as at order 2.

    The ``"+"`` face is where the image halo also overwrites in-domain samples (the half-offset
    samples of the wall cell lie outside a PEC plane), so a geometry slip there would move the
    standing-wave node by half a cell and show up as a changed ratio.
    """

    @pytest.mark.parametrize("wall_type,expected", [("pec", "node"), ("pmc", "antinode")])
    def test_node_or_antinode_at_the_wall(self, wall_type, expected):
        ratios = {order: _wall_standing_wave_ratio(wall_type, order) for order in (2, 4)}
        if expected == "node":
            assert all(r < 0.25 for r in ratios.values()), ratios
        else:
            assert all(r > 5.0 for r in ratios.values()), ratios
        # The standing-wave pattern is a boundary-condition property, not a stencil property.
        assert ratios[4] == pytest.approx(ratios[2], rel=0.05), ratios


# ══════════════════════════════════════════════════════════════════════════════════════════════
# (c) stability and dispersion of a plane wave in a periodic box
# ══════════════════════════════════════════════════════════════════════════════════════════════

_PW_NZ = 16  # cells per wavelength of the seeded mode
_PW_STEPS = 200


def _build_periodic_box(curl_order):
    config = fdtdx.SimulationConfig(
        time=200e-15,
        grid=fdtdx.UniformGrid(spacing=_SPACING),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        curl_order=curl_order,
        gradient_config=None,
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(4, 4, _PW_NZ))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=1,
        override_types={face: "periodic" for face in ("min_x", "max_x", "min_y", "max_y", "min_z", "max_z")},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)
    return _place(objects, constraints, config)


def _run_plane_wave(curl_order):
    """Seed a single-mode standing wave and return (modal amplitude trace, energy trace, dt)."""
    objects, arrays, config = _build_periodic_box(curl_order)
    k = 2.0 * np.pi / (_PW_NZ * _SPACING)
    z = jnp.arange(_PW_NZ, dtype=arrays.fields.E.dtype) * _SPACING
    profile = jnp.sin(k * z)
    E = jnp.zeros_like(arrays.fields.E).at[0].set(profile[None, None, :])
    arrays = arrays.aset("fields->E", E).aset("fields->H", jnp.zeros_like(arrays.fields.H))

    key = jax.random.PRNGKey(0)
    state = (jnp.asarray(0, dtype=jnp.int32), arrays)
    amplitudes, energies = [], []
    for _ in range(_PW_STEPS):
        fields = state[1].fields
        amplitudes.append(float(jnp.mean(fields.E[0] * profile[None, None, :])))
        energies.append(float(jnp.sum(fields.E**2) + jnp.sum(fields.H**2)))
        state = _step_forward(state, config, objects, key)
    return np.asarray(amplitudes), np.asarray(energies), config.time_step_duration, k


def _measured_omega(amplitudes, dt):
    """Angular frequency of a sampled cosine via ``a[n+1] + a[n-1] = 2 cos(w dt) a[n]``."""
    a = amplitudes
    window = slice(1, len(a) - 1)
    numerator = a[2:] + a[:-2]
    denominator = 2.0 * a[window]
    mask = np.abs(denominator) > 0.2 * np.max(np.abs(a))
    return float(np.arccos(np.median(numerator[mask] / denominator[mask])) / dt)


class TestPeriodicPlaneWaveStability:
    """The order-4 scheme must stay bounded and reproduce the plane-wave frequency."""

    def test_energy_bounded_and_finite(self):
        amplitudes, energies, _, _ = _run_plane_wave(4)
        assert np.all(np.isfinite(energies))
        assert np.all(np.isfinite(amplitudes))
        # The conserved leapfrog quantity is time-centered, so the plain sum E^2 + H^2 wobbles
        # within the discretization level over a period; what must not happen is growth.
        assert energies.max() / energies[0] < 1.3
        assert energies.min() / energies[0] > 0.7
        window = 50  # longer than the ~33-step oscillation period
        assert energies[-window:].max() / energies[:window].max() == pytest.approx(1.0, abs=0.01)
        assert np.abs(amplitudes[-window:]).max() / np.abs(amplitudes[:window]).max() == pytest.approx(1.0, abs=0.01)

    def test_wave_propagates_at_the_right_frequency(self):
        amplitudes, _, dt, k = _run_plane_wave(4)
        assert np.abs(amplitudes).max() > 0.4  # the mode is still there after 200 steps
        assert amplitudes.min() < -0.2  # ... and it oscillated through zero
        omega = _measured_omega(amplitudes, dt)
        exact = c0 * k
        assert abs(omega - exact) / exact < 2e-2, f"omega {omega:.4e} vs {exact:.4e}"

    def test_dispersion_error_beats_order_two(self):
        errors = {}
        for order in (2, 4):
            amplitudes, _, dt, k = _run_plane_wave(order)
            errors[order] = abs(_measured_omega(amplitudes, dt) - c0 * k) / (c0 * k)
        assert errors[4] < errors[2], errors
