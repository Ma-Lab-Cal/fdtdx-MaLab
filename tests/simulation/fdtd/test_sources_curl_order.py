"""Simulation tests: plane-wave / dipole sources under the higher-order curl stencil.

The plane-wave sources (``UniformPlaneSource``, ``GaussianPlaneSource``, ``ModePlaneSource``)
inject a pair of E/H current sheets on adjacent planes (``_tfsf_inject_E_face`` /
``_tfsf_inject_H_face`` in ``src/fdtdx/objects/sources/tfsf.py``). That injection uses the same
local Courant number as the surrounding curl update, so it is stencil-independent: at
``curl_order=4`` the two sheets should still cancel the backward-launched wave, leaving only a
small residual leak bounded by the numerical dispersion of the wider stencil.

Three properties are checked:

* **Scattered-field leak** (:func:`test_uniform_plane_source_scattered_field_leak`) — a CW
  ``UniformPlaneSource`` in a periodic/PML vacuum box. The field behind the source (scattered-field
  region) must stay far smaller than the field ahead of it (total-field region) at both
  ``curl_order`` 2 and 4, order 4 must not leak much more than order 2, and the steady-state
  forward amplitude must be unchanged between orders (the two stencils propagate the same physical
  wave).
* **Dipole stability** (:func:`test_point_dipole_source_stable_at_curl_order_four`) — a
  ``PointDipoleSource`` radiating into a fully-PML box at ``curl_order=4`` stays finite and does
  not blow up (cheap smoke test; not a quantitative check).
* **Pulsed transmitted energy** (:func:`test_gaussian_pulse_transmitted_energy_matches_order_two`)
  — the same domain as the leak test, driven by a ``GaussianPulseProfile`` instead of a CW source.
  The time-integrated Poynting flux through the forward detector (the transmitted pulse energy)
  must match between ``curl_order`` 2 and 4.
"""

import jax
import jax.numpy as jnp
import numpy as np

import fdtdx

# ── Shared domain constants (everything is deliberately tiny: these run on CPU) ───────────────
_WAVELENGTH = 1.0e-6
_RESOLUTION = 30e-9  # ~33 cells / wavelength
_PML_CELLS = 14
_NXY = 4
_NZ = 120

_SOURCE_Z = 40
_BACKWARD_DET_Z = 24  # scattered-field region: behind (smaller z than) the "+"-direction source
_FORWARD_DET_Z = 90  # total-field region: ahead of (larger z than) the source

_CW_SIM_TIME = 60e-15  # ~5 transits of the domain length in vacuum
_N_AVG_PERIODS = 8  # optical periods averaged for the CW steady-state amplitude

_PULSE_SIM_TIME = 80e-15  # long enough for the whole Gaussian pulse to cross the domain
_PULSE_SPECTRAL_WIDTH_WAVELENGTH = 8e-6  # ~12% fractional bandwidth around _WAVELENGTH

_DIPOLE_CELLS = 20
_DIPOLE_PML_CELLS = 6
_DIPOLE_SIM_TIME = 30e-15


# ── Helpers ─────────────────────────────────────────────────────────────────────────────────


def _build_domain(curl_order: int, sim_time: float):
    """Vacuum box: periodic in x/y, PML in z. Returns (objects, constraints, config, volume)."""
    config = fdtdx.SimulationConfig(
        time=sim_time,
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        curl_order=curl_order,
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_NXY, _NXY, _NZ))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=_PML_CELLS,
        override_types={
            "min_x": "periodic",
            "max_x": "periodic",
            "min_y": "periodic",
            "max_y": "periodic",
        },
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)
    return objects, constraints, config, volume


def _add_plane_source(objects, constraints, volume, wave_character, temporal_profile=None):
    kwargs = dict(
        partial_grid_shape=(None, None, 1),
        wave_character=wave_character,
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    if temporal_profile is not None:
        kwargs["temporal_profile"] = temporal_profile
    source = fdtdx.UniformPlaneSource(**kwargs)
    constraints.extend(
        [
            source.same_size(volume, axes=(0, 1)),
            source.place_at_center(volume, axes=(0, 1)),
            source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_SOURCE_Z,)),
        ]
    )
    objects.append(source)
    return source


def _add_field_detector(name, z_idx, volume, objects, constraints):
    det = fdtdx.FieldDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        components=("Ex",),
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z_idx,)),
        ]
    )
    objects.append(det)
    return det


def _add_flux_detector(name, z_idx, volume, objects, constraints):
    det = fdtdx.PoyntingFluxDetector(
        name=name,
        partial_grid_shape=(None, None, 1),
        direction="+",
        fixed_propagation_axis=2,
        reduce_volume=True,
        plot=False,
    )
    constraints.extend(
        [
            det.same_size(volume, axes=(0, 1)),
            det.place_at_center(volume, axes=(0, 1)),
            det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(z_idx,)),
        ]
    )
    objects.append(det)
    return det


def _run(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key, show_progress=False)
    return arrays


def _steady_state_amplitude(arrays, name, config, wave_character, n_periods=_N_AVG_PERIODS):
    """RMS-based |Ex| amplitude of a reduce_volume FieldDetector over the last ``n_periods``.

    Uses ``sqrt(2 * mean(window**2))`` rather than a raw ``max(abs(window))`` peak: with only a
    few dozen samples per optical period, the two curl orders have different step counts per
    period, so a bare peak-of-samples estimate carries an order-dependent sampling-phase bias of
    the same magnitude as the physical effect under test. Averaging the squared signal over many
    full periods removes that bias and is exact for a pure sinusoid.
    """
    trace = np.array(arrays.detector_states[name]["fields"][:, 0])
    steps_per_period = wave_character.get_period() / config.time_step_duration
    n_avg = max(round(n_periods * steps_per_period), 1)
    window = trace[-n_avg:]
    return float(np.sqrt(2.0 * np.mean(np.square(window))))


def _integrated_flux(arrays, name, config):
    """Time-integrated Poynting flux (transmitted pulse energy, per unit area).

    Trapezoidal rather than rectangular quadrature: the two curl orders take a different number
    of (differently sized) time steps for the same physical run, so a plain ``sum() * dt``
    Riemann sum carries an order-dependent O(dt) quadrature bias on top of the physical
    curl-order effect under test. Trapezoidal quadrature is O(dt^2) and mostly cancels that bias.
    """
    flux = np.array(arrays.detector_states[name]["poynting_flux"][:, 0])
    return float(np.trapezoid(flux, dx=config.time_step_duration))


# ── (1) CW plane wave: scattered-field leak vs total-field amplitude ──────────────────────────


def test_uniform_plane_source_scattered_field_leak():
    """A CW UniformPlaneSource must leave the scattered-field region almost field-free.

    Checks, at curl_order 2 and 4:
      (a) backward(scattered)/forward(total) amplitude ratio < 2e-3.
      (b) the order-4 leak ratio is not more than 1.5x the order-2 leak ratio.
      (c) the forward steady-state amplitude is unchanged between orders (within 1%).
    """
    wave_character = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)

    forward_amp = {}
    leak_ratio = {}
    for curl_order in (2, 4):
        objects, constraints, config, volume = _build_domain(curl_order, _CW_SIM_TIME)
        _add_plane_source(objects, constraints, volume, wave_character)
        _add_field_detector("forward", _FORWARD_DET_Z, volume, objects, constraints)
        _add_field_detector("backward", _BACKWARD_DET_Z, volume, objects, constraints)

        arrays = _run(objects, constraints, config)

        fwd = _steady_state_amplitude(arrays, "forward", config, wave_character)
        bwd = _steady_state_amplitude(arrays, "backward", config, wave_character)
        assert fwd > 1e-3, f"curl_order={curl_order}: forward amplitude too small ({fwd:.3e}) to be a reference"

        forward_amp[curl_order] = fwd
        leak_ratio[curl_order] = bwd / fwd

    for curl_order, leak in leak_ratio.items():
        assert leak < 2e-3, f"curl_order={curl_order}: scattered-field leak ratio {leak:.3e} >= 2e-3"

    assert leak_ratio[4] <= 1.5 * leak_ratio[2], (
        f"order-4 leak ratio {leak_ratio[4]:.3e} is more than 1.5x the order-2 leak ratio {leak_ratio[2]:.3e}"
    )

    rel_amp_diff = abs(forward_amp[4] - forward_amp[2]) / forward_amp[2]
    assert rel_amp_diff < 0.01, (
        f"forward steady-state amplitude changed by {rel_amp_diff:.3%} between curl_order=2 "
        f"({forward_amp[2]:.6e}) and curl_order=4 ({forward_amp[4]:.6e})"
    )


# ── (2) PointDipoleSource stability smoke test at curl_order=4 ────────────────────────────────


def test_point_dipole_source_stable_at_curl_order_four():
    """A PointDipoleSource radiating into a fully-PML box stays finite and energy-bounded.

    Cheap smoke test (not a quantitative accuracy check): confirms the order-4 stencil does not
    destabilize a source that radiates directly into the PML on every side.
    """
    config = fdtdx.SimulationConfig(
        time=_DIPOLE_SIM_TIME,
        grid=fdtdx.UniformGrid(spacing=_RESOLUTION),
        backend="cpu",
        dtype=jnp.float32,
        courant_factor=0.99,
        curl_order=4,
    )
    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_DIPOLE_CELLS, _DIPOLE_CELLS, _DIPOLE_CELLS))
    objects.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_DIPOLE_PML_CELLS)
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    source = fdtdx.PointDipoleSource(
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(wavelength=_WAVELENGTH),
        polarization=2,
        static_amplitude_factor=1.0,
    )
    constraints.append(source.place_at_center(volume, axes=(0, 1, 2)))
    objects.append(source)

    energy_det = fdtdx.EnergyDetector(name="energy", reduce_volume=True, plot=False)
    constraints.extend(energy_det.same_position_and_size(volume))
    objects.append(energy_det)

    arrays = _run(objects, constraints, config)

    E = np.array(arrays.fields.E)
    H = np.array(arrays.fields.H)
    assert np.all(np.isfinite(E)), "E field contains non-finite values"
    assert np.all(np.isfinite(H)), "H field contains non-finite values"

    energy = np.array(arrays.detector_states["energy"]["energy"][:, 0])
    assert np.all(np.isfinite(energy)), "energy trace contains non-finite values"

    half = len(energy) // 2
    first_half_max = float(energy[:half].max())
    second_half_max = float(energy[half:].max())
    assert second_half_max < 5.0 * first_half_max + 1e-30, (
        f"energy grew unboundedly: first-half max {first_half_max:.3e}, second-half max {second_half_max:.3e}"
    )


# ── (3) Gaussian pulse: transmitted energy matches order 2 ────────────────────────────────────


def test_gaussian_pulse_transmitted_energy_matches_order_two():
    """A pulsed UniformPlaneSource's transmitted energy must match between curl_order 2 and 4.

    Same domain as the CW leak test, but driven by a GaussianPulseProfile. The time-integrated
    Poynting flux through the forward detector (transmitted pulse energy) at curl_order=4 must
    match the curl_order=2 reference within 2%.
    """
    wave_character = fdtdx.WaveCharacter(wavelength=_WAVELENGTH)
    profile = fdtdx.GaussianPulseProfile(
        center_wave=wave_character,
        spectral_width=fdtdx.WaveCharacter(wavelength=_PULSE_SPECTRAL_WIDTH_WAVELENGTH),
    )

    transmitted_energy = {}
    for curl_order in (2, 4):
        objects, constraints, config, volume = _build_domain(curl_order, _PULSE_SIM_TIME)
        _add_plane_source(objects, constraints, volume, wave_character, temporal_profile=profile)
        _add_flux_detector("forward", _FORWARD_DET_Z, volume, objects, constraints)

        arrays = _run(objects, constraints, config)
        transmitted_energy[curl_order] = _integrated_flux(arrays, "forward", config)

    assert transmitted_energy[2] > 0, "reference (curl_order=2) transmitted energy is not positive"
    rel_err = abs(transmitted_energy[4] - transmitted_energy[2]) / transmitted_energy[2]
    assert rel_err < 0.02, (
        f"transmitted pulse energy differs by {rel_err:.3%} between curl_order=2 "
        f"({transmitted_energy[2]:.6e}) and curl_order=4 ({transmitted_energy[4]:.6e})"
    )
