"""Simulation tests for the segmented (checkpoint + re-forward) reversible gradient path.

``GradientConfig(recording_mode="segmented")`` stores full-field checkpoints in the forward pass and
regenerates the PML interface record segment by segment during the backward pass, so the recorder
buffer only spans one segment. The gradient must be the same as the classic full-record reversible
pass (up to float rounding), for any number of checkpoints, with and without lossy media, and with
recorder compression modules active within the segments.
"""

from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.config import GradientConfig, SimulationConfig
from fdtdx.constants import c as c0
from fdtdx.core.grid import UniformGrid
from fdtdx.fdtd.fdtd import checkpointed_fdtd, reversible_fdtd
from fdtdx.interfaces.recorder import Recorder
from fdtdx.interfaces.time_filter import LinearReconstructEveryK


@contextmanager
def _x64_enabled():
    prev = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


_RESOLUTION = 50e-9
_PML_CELLS = 3
_VOLUME_CELLS = 12
_TIME_STEPS = 23  # deliberately prime so that most checkpoint counts leave a shorter tail segment


def _build_scene(grad_cfg: GradientConfig | None, conductivity: float, dtype=jnp.float64, curl_order: int = 2):
    """PML box with a lossy slab and a CW dipole; the gradient config sizes the recorder."""
    probe = SimulationConfig(
        time=1e-15,
        grid=UniformGrid(spacing=_RESOLUTION),
        backend="cpu",
        dtype=dtype,
        courant_factor=0.99,
        curl_order=curl_order,
    )
    config = probe.aset("time", _TIME_STEPS * probe.time_step_duration)
    config = config.aset("gradient_config", grad_cfg)
    assert config.time_steps_total == _TIME_STEPS

    objects, constraints = [], []
    volume = fdtdx.SimulationVolume(partial_grid_shape=(_VOLUME_CELLS, _VOLUME_CELLS, _VOLUME_CELLS))
    objects.append(volume)
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(thickness=_PML_CELLS, boundary_type="pml")
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    objects.extend(bound_dict.values())
    constraints.extend(c_list)

    material = fdtdx.Material(permittivity=2.0, electric_conductivity=conductivity)
    slab = fdtdx.UniformMaterialObject(
        name="slab",
        partial_grid_shape=(None, None, 3),
        material=material,
    )
    constraints.extend(
        [
            slab.same_size(volume, axes=(0, 1)),
            slab.place_at_center(volume, axes=(0, 1)),
            slab.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(_VOLUME_CELLS // 2,)),
        ]
    )
    objects.append(slab)

    source = fdtdx.PointDipoleSource(
        name="dip",
        partial_grid_shape=(1, 1, 1),
        wave_character=fdtdx.WaveCharacter(frequency=c0 / 600e-9),
        polarization=1,
        amplitude=1.0,
    )
    constraints.append(
        source.set_grid_coordinates(
            axes=(0, 1, 2),
            sides=("-", "-", "-"),
            coordinates=(_VOLUME_CELLS // 2, _VOLUME_CELLS // 2, _VOLUME_CELLS // 2 - 2),
        )
    )
    objects.append(source)

    key = jax.random.PRNGKey(0)
    obj, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects,
        config=config,
        constraints=constraints,
        key=key,
    )
    arrays, obj, _ = fdtdx.apply_params(arrays, obj, params, key)
    return obj, arrays, config


def _grad_config(mode: str, num_ckpt: int, modules=()) -> GradientConfig:
    return GradientConfig(
        method="reversible",
        recorder=Recorder(modules=list(modules)),
        num_checkpoints_reversible=num_ckpt,
        recording_mode=mode,
    )


def _interior(a: jax.Array, margin: int = _PML_CELLS + 2) -> jax.Array:
    """Drop the PML and the cells next to it, where the reverse reconstruction is not defined."""
    return a[..., margin:-margin, margin:-margin, margin:-margin]


def _loss_and_grad(obj, arrays, config, impl=reversible_fdtd):
    key = jax.random.PRNGKey(1)

    def loss(inv_eps):
        ar = arrays.aset("inv_permittivities", inv_eps)
        _, ar = impl(arrays=ar, objects=obj, config=config, key=key, show_progress=False)
        return jnp.sum(_interior(ar.fields.E) ** 2)

    value, grad = jax.value_and_grad(loss)(arrays.inv_permittivities)
    return float(value), np.asarray(_interior(grad))


def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-300))


class TestSegmentedRecorderSizing:
    def test_recorder_buffer_spans_one_segment(self):
        with _x64_enabled():
            for num_ckpt in (0, 3, _TIME_STEPS - 1):
                _, arrays, config = _build_scene(_grad_config("segmented", num_ckpt), conductivity=0.0)
                expected = -(-_TIME_STEPS // (num_ckpt + 1))
                assert config.gradient_config.reversible_segment_length(_TIME_STEPS) == expected
                assert arrays.recording_state is not None
                for buf in arrays.recording_state.data.values():
                    assert buf.shape[0] == expected

    def test_full_mode_buffer_spans_run(self):
        with _x64_enabled():
            _, arrays, _ = _build_scene(_grad_config("full", 3), conductivity=0.0)
            for buf in arrays.recording_state.data.values():
                assert buf.shape[0] == _TIME_STEPS

    def test_mismatched_recorder_length_raises(self):
        with _x64_enabled():
            obj, arrays, config = _build_scene(_grad_config("full", 0), conductivity=0.0)
            bad = config.aset("gradient_config->recording_mode", "segmented")
            bad = bad.aset("gradient_config->num_checkpoints_reversible", 2)
            with pytest.raises(Exception, match="recorder was initialized"):
                reversible_fdtd(arrays=arrays, objects=obj, config=bad, key=jax.random.PRNGKey(0), show_progress=False)


class TestSegmentedMatchesFull:
    """The segmented pass is a pure memory optimisation: same gradient as the full record."""

    @pytest.mark.parametrize("num_ckpt", [0, 1, 4, _TIME_STEPS - 1])
    def test_lossless_float64(self, num_ckpt):
        with _x64_enabled():
            obj, arrays, config = _build_scene(_grad_config("full", 0), conductivity=0.0)
            loss_ref, grad_ref = _loss_and_grad(obj, arrays, config)

            obj_s, arrays_s, config_s = _build_scene(_grad_config("segmented", num_ckpt), conductivity=0.0)
            loss_s, grad_s = _loss_and_grad(obj_s, arrays_s, config_s)

        assert np.isfinite(grad_ref).all() and np.isfinite(grad_s).all()
        assert np.linalg.norm(grad_ref) > 0
        assert abs(loss_s - loss_ref) <= 1e-12 * abs(loss_ref)
        assert _rel_err(grad_s, grad_ref) < 1e-9

    @pytest.mark.parametrize("num_ckpt", [2, 7])
    def test_lossy_same_checkpoints_float64(self, num_ckpt):
        """With identical checkpoints the two modes perform identical reverse arithmetic."""
        with _x64_enabled():
            obj_f, arrays_f, config_f = _build_scene(_grad_config("full", num_ckpt), conductivity=5e3)
            _, grad_f = _loss_and_grad(obj_f, arrays_f, config_f)
            obj_s, arrays_s, config_s = _build_scene(_grad_config("segmented", num_ckpt), conductivity=5e3)
            _, grad_s = _loss_and_grad(obj_s, arrays_s, config_s)
        assert np.isfinite(grad_f).all() and np.isfinite(grad_s).all()
        assert _rel_err(grad_s, grad_f) < 1e-10

    def test_lossy_segmented_tracks_checkpointed_reference_float64(self):
        """Many short segments bound the reverse drift of a lossy medium (exact reference: autodiff)."""
        with _x64_enabled():
            obj_c, arrays_c, config_c = _build_scene(
                GradientConfig(method="checkpointed", num_checkpoints=8), conductivity=5e3
            )
            _, grad_c = _loss_and_grad(obj_c, arrays_c, config_c, impl=checkpointed_fdtd)
            errs = []
            for num_ckpt in (0, _TIME_STEPS - 1):
                obj_s, arrays_s, config_s = _build_scene(_grad_config("segmented", num_ckpt), conductivity=5e3)
                _, grad_s = _loss_and_grad(obj_s, arrays_s, config_s)
                errs.append(_rel_err(grad_s, grad_c))
        assert errs[-1] <= errs[0] * 1.5 + 1e-12
        assert errs[-1] < 1e-6

    def test_compression_module_within_segments_float64(self):
        """LinearReconstructEveryK works per segment (each segment's last step is a saved node)."""
        with _x64_enabled():
            obj, arrays, config = _build_scene(_grad_config("full", 0), conductivity=0.0)
            _, grad_ref = _loss_and_grad(obj, arrays, config)
            obj_s, arrays_s, config_s = _build_scene(
                _grad_config("segmented", 3, modules=[LinearReconstructEveryK(k=2)]), conductivity=0.0
            )
            for buf in arrays_s.recording_state.data.values():
                assert buf.shape[0] < _TIME_STEPS  # compressed and segmented
            _, grad_s = _loss_and_grad(obj_s, arrays_s, config_s)
        assert np.isfinite(grad_s).all()
        # Linear interpolation of the interface record is an approximation, so only loose agreement.
        assert _rel_err(grad_s, grad_ref) < 5e-2

    def test_run_fdtd_dispatch_and_float32(self):
        obj, arrays, config = _build_scene(_grad_config("full", 0), conductivity=0.0, dtype=jnp.float32)
        _, grad_ref = _loss_and_grad(obj, arrays, config, impl=fdtdx.run_fdtd)
        obj_s, arrays_s, config_s = _build_scene(_grad_config("segmented", 5), conductivity=0.0, dtype=jnp.float32)
        _, grad_s = _loss_and_grad(obj_s, arrays_s, config_s, impl=fdtdx.run_fdtd)
        assert np.isfinite(grad_s).all()
        assert _rel_err(grad_s, grad_ref) < 1e-4
