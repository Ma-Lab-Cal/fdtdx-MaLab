from __future__ import annotations

from collections.abc import Callable
from functools import partial

import equinox.internal as eqxi
import jax
import jax.numpy as jnp

from fdtdx.config import SimulationConfig
from fdtdx.core.progress import _make_pbar, _wrap_body_with_progress
from fdtdx.fdtd.backward import backward
from fdtdx.fdtd.container import ArrayContainer, FieldState, ObjectContainer, PmlAuxField, SimulationState
from fdtdx.fdtd.forward import forward, forward_step_primals
from fdtdx.fdtd.stop_conditions import StoppingCondition, TimeStepCondition
from fdtdx.interfaces.state import RecordingState
from fdtdx.objects.detectors.detector import DetectorState


def _reversible_slice_boundaries(time_steps_total: int, num_slices: int) -> list[int]:
    """Compute the time-step boundaries partitioning a run into at most ``num_slices`` slices.

    The run is cut into segments of equal length ``L = ceil(time_steps_total / num_slices)`` plus a
    shorter tail, so that a fixed-trip-count loop can process the segments without unrolling one
    while loop per segment. Returns ``[0, L, 2L, ..., time_steps_total]``; every slice has length
    ``>= 1`` and at most ``L``, and there are at most ``num_slices`` slices (fewer when ``L`` does
    not divide evenly, e.g. ``T=10, num_slices=7`` gives ``L=2`` and 5 slices). The interior
    boundaries are the times at which a full-field checkpoint is taken.

    Args:
        time_steps_total (int): Total number of forward time steps ``T``.
        num_slices (int): Requested number of slices (``= num_checkpoints_reversible + 1``).

    Returns:
        list[int]: The boundary time steps, starting at 0 and ending at ``time_steps_total``.
    """
    if num_slices < 1:
        raise ValueError(f"num_slices must be >= 1, got {num_slices}")
    if time_steps_total <= 0:
        return [0]
    segment_length = max(1, -(-time_steps_total // num_slices))
    boundaries = list(range(0, time_steps_total, segment_length))
    boundaries.append(time_steps_total)
    return boundaries


def reversible_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    """Run a memory-efficient differentiable FDTD simulation leveraging time-reversal symmetry.

    This implementation exploits the time-reversal symmetry of Maxwell's equations to perform
    backpropagation without storing the electromagnetic fields at each time step. During the
    backward pass, the fields are reconstructed by running the simulation in reverse, only
    requiring O(1) field memory instead of O(T) where T is the number of time steps.

    The only exception is boundary conditions which break time-reversal symmetry - the fields at
    the PML interfaces are recorded during the forward pass and replayed during backpropagation.
    Two strategies are available (``config.gradient_config.recording_mode``):

    - ``"full"``: the interface record covers every time step (O(T) interface memory). With
      ``num_checkpoints_reversible > 0`` the run is additionally partitioned into segments with a
      full-field checkpoint at every segment start; the reverse reconstruction is reset to the exact
      checkpoint at each segment boundary, which bounds the reconstruction drift of lossy media.
    - ``"segmented"``: the forward pass stores only the segment checkpoints and records nothing.
      The backward pass re-simulates each segment forward from its checkpoint to regenerate the
      interface record for that segment alone (a buffer of segment length instead of ``T``), then
      reverses it. This bounds the memory of the exact reversible adjoint for arbitrarily long runs
      at the cost of one extra forward sweep.

    Args:
        arrays (ArrayContainer): Initial state of the simulation containing:
            - E, H: Electric and magnetic field arrays
            - inv_permittivities, inv_permeabilities: Material properties
            - detector_states: Dictionary of field detectors
            - recording_state: Optional state for recording field evolution
        objects (ObjectContainer): Collection of physical objects in the simulation
            (sources, detectors, boundaries, etc.)
        config (SimulationConfig): Simulation parameters including:
            - time_steps_total: Total number of steps to simulate
            - invertible_optimization: Whether to record boundaries for backprop
        key (jax.Array): JAX PRNGKey for any stochastic operations
        show_progress (bool): Display a tqdm progress bar while the simulation runs.
            Set to False for a minor speed improvement; see the module-level
            benchmark note for overhead estimates. Defaults to True.
            The bar is driven entirely by ``io_callback`` at XLA execution
            time, so it works correctly whether the simulation is
            wrapped in ``jax.jit``.
        progress_callback (Callable[[int, int], None] | None): Optional callback receiving
            ``(current_step, total_steps)`` for custom progress reporting.

    Returns:
        SimulationState: Tuple containing:
            - Final time step (int)
            - ArrayContainer with the final state of all fields and components

    Notes:
        The implementation uses custom vector-Jacobian products (VJPs) to enable
        efficient backpropagation through the entire simulation while maintaining
        numerical stability. This makes it suitable for gradient-based optimization
        of electromagnetic designs.

    Raises:
        NotImplementedError: If the simulation contains dispersive materials. Reversing
            the ADE polarization recurrence is not supported; use the ``"checkpointed"``
            gradient method for dispersive simulations.
    """
    # Checked here in addition to initialization time, since the gradient config can be
    # swapped after ``place_objects`` and this function can be called directly, bypassing
    # ``run_fdtd``.
    if arrays.dispersive_c1 is not None or arrays.fields.dispersive_P_curr is not None:
        raise NotImplementedError(
            "Dispersive time-reversible gradient computation under active development. "
            "Use GradientConfig(method='checkpointed') instead."
        )

    arrays = arrays.reset()

    # Segmentation of the run. ``num_checkpoints_reversible == 0`` (the default) gives a single
    # segment and reproduces the classic full forward + full reverse pass exactly.
    grad_cfg = config.gradient_config
    num_ckpt = 0 if grad_cfg is None else grad_cfg.num_checkpoints_reversible
    segmented = grad_cfg is not None and grad_cfg.is_segmented
    time_steps_total = config.time_steps_total
    if num_ckpt > 0 and num_ckpt + 1 > time_steps_total:
        raise Exception(
            "num_checkpoints_reversible must be <= time_steps_total - 1 "
            f"(got num_checkpoints_reversible={num_ckpt}, time_steps_total={time_steps_total})"
        )
    if segmented and not config.invertible_optimization:
        raise Exception("recording_mode='segmented' requires a Recorder in the gradient config")
    if grad_cfg is None:
        segment_length = max(1, time_steps_total)
    else:
        segment_length = grad_cfg.reversible_segment_length(time_steps_total)
    if grad_cfg is not None and grad_cfg.recorder is not None:
        recorder_steps = getattr(grad_cfg.recorder, "_max_time_steps", None)
        expected_steps = segment_length if segmented else time_steps_total
        if recorder_steps is not None and recorder_steps != expected_steps:
            raise Exception(
                f"The recorder was initialized for {recorder_steps} time steps but recording_mode="
                f"'{grad_cfg.recording_mode}' needs {expected_steps} (the "
                f"{'segment length' if segmented else 'total number of time steps'}). Initialize the arrays "
                "with place_objects using the same gradient config."
            )
    num_full_segments = time_steps_total // segment_length
    tail_length = time_steps_total - num_full_segments * segment_length
    num_segments = num_full_segments + (1 if tail_length > 0 else 0)

    pbar = _make_pbar(
        show_progress=show_progress,
        total_steps=time_steps_total,
        desc="FDTD (reversible)",
        progress_callback=progress_callback,
    )

    # Forward step of the primal pass. In segmented mode nothing is recorded here: the interface
    # record is regenerated segment by segment during the backward pass.
    _forward_body = partial(
        forward,
        config=config,
        objects=objects,
        key=key,
        record_detectors=True,
        record_boundaries=config.invertible_optimization and not segmented,
        simulate_boundaries=True,
    )
    _forward_body_with_progress, _close_pbar = _wrap_body_with_progress(_forward_body, pbar)

    def run_steps(state: SimulationState, n_steps: int, body: Callable) -> SimulationState:
        """Advance ``state`` by the static number ``n_steps`` of steps with ``body``."""
        end_step = state[0] + n_steps
        return eqxi.while_loop(
            max_steps=n_steps,
            cond_fun=lambda s: end_step > s[0],
            body_fun=body,
            init_val=state,
            kind="lax",
        )

    def make_container(fields: FieldState, recording_state: RecordingState | None) -> ArrayContainer:
        return ArrayContainer(
            fields=fields,
            inv_permittivities=arrays.inv_permittivities,
            inv_permeabilities=arrays.inv_permeabilities,
            detector_states=arrays.detector_states,
            recording_state=recording_state,
            electric_conductivity=arrays.electric_conductivity,
            magnetic_conductivity=arrays.magnetic_conductivity,
            initial_inv_permittivities=arrays.initial_inv_permittivities,
        )

    def segmented_forward(arr: ArrayContainer) -> tuple[SimulationState, FieldState]:
        """Run the forward pass, capturing the full field state at the start of every segment.

        This is the single source of truth for the reversible forward stepping. The checkpoints are
        stacked along a leading axis of size ``num_segments`` (``checkpoints[i]`` is the field state
        at time ``i * segment_length``; entry 0 is the initial state), so the number of segments does
        not change the size of the compiled program: the full-length segments run in a
        ``fori_loop`` and only the shorter tail segment (if any) is a separate loop.
        """
        state: SimulationState = (jnp.asarray(0, dtype=jnp.int32), arr)
        # At least one slot so that the (never executed) loop body traces for time_steps_total == 0.
        num_slots = max(num_segments, 1)
        checkpoints = jax.tree.map(lambda x: jnp.zeros((num_slots, *x.shape), x.dtype), arr.fields)

        def segment_body(i, carry):
            cur_state, ckpts = carry
            ckpts = jax.tree.map(lambda buf, x: buf.at[i].set(x), ckpts, cur_state[1].fields)
            cur_state = run_steps(cur_state, segment_length, _forward_body_with_progress)
            return cur_state, ckpts

        state, checkpoints = jax.lax.fori_loop(0, num_full_segments, segment_body, (state, checkpoints))
        if tail_length > 0:
            checkpoints = jax.tree.map(lambda buf, x: buf.at[num_full_segments].set(x), checkpoints, state[1].fields)
            state = run_steps(state, tail_length, _forward_body_with_progress)
        return state, checkpoints

    @jax.custom_vjp
    def reversible_fdtd_primal(
        E: jax.Array,
        H: jax.Array,
        psi_E: PmlAuxField,
        psi_H: PmlAuxField,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
        detector_states: dict[str, DetectorState],
        recording_state: RecordingState | None,
    ):
        arr = ArrayContainer(
            fields=FieldState(
                E=E,
                H=H,
                psi_E=psi_E,
                psi_H=psi_H,
            ),
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            detector_states=detector_states,
            recording_state=recording_state,
            electric_conductivity=arrays.electric_conductivity,
            magnetic_conductivity=arrays.magnetic_conductivity,
            initial_inv_permittivities=arrays.initial_inv_permittivities,
        )
        # The non-gradient primal path needs only the final state; the checkpoints are discarded.
        state, _ = segmented_forward(arr)
        return (
            state[0],
            state[1].fields.E,
            state[1].fields.H,
            state[1].fields.psi_E,
            state[1].fields.psi_H,
            state[1].inv_permittivities,
            state[1].inv_permeabilities,
            state[1].detector_states,
            state[1].recording_state,
        )

    def fdtd_fwd(
        E: jax.Array,
        H: jax.Array,
        psi_E: PmlAuxField,
        psi_H: PmlAuxField,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array,
        detector_states: dict[str, DetectorState],
        recording_state: RecordingState | None,
    ):
        arr = ArrayContainer(
            fields=FieldState(
                E=E,
                H=H,
                psi_E=psi_E,
                psi_H=psi_H,
            ),
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            detector_states=detector_states,
            recording_state=recording_state,
            electric_conductivity=arrays.electric_conductivity,
            magnetic_conductivity=arrays.magnetic_conductivity,
            initial_inv_permittivities=arrays.initial_inv_permittivities,
        )
        s_k, checkpoints = segmented_forward(arr)

        primal_out = (
            s_k[0],
            s_k[1].fields.E,
            s_k[1].fields.H,
            s_k[1].fields.psi_E,
            s_k[1].fields.psi_H,
            s_k[1].inv_permittivities,
            s_k[1].inv_permeabilities,
            s_k[1].detector_states,
            s_k[1].recording_state,
        )
        # The stacked segment checkpoints are threaded to ``fdtd_bwd`` via the residual so the reverse
        # reconstruction can be reset to the exact field at each segment boundary (and, in segmented
        # recording mode, so each segment can be re-simulated to regenerate its interface record).
        residual = (primal_out, checkpoints)
        return primal_out, residual

    def fdtd_bwd(
        residual,
        cot,
    ):
        primal_out, checkpoints = residual
        (
            res_time_step,
            res_E,
            res_H,
            res_psi_E,
            res_psi_H,
            res_inv_permittivities,
            res_inv_permeabilities,
            res_detector_states,
            res_recording_state,
        ) = primal_out
        del res_time_step
        # The recording state is not a primal of the per-step VJP (it is only replayed), so its
        # cotangent is dropped here instead of being carried as a full-size zero array.
        cot_carry = tuple(cot[:8])

        final_fields = FieldState(E=res_E, H=res_H, psi_E=res_psi_E, psi_H=res_psi_H)

        def step_primal_container(fields: FieldState, recording_state: RecordingState | None) -> ArrayContainer:
            return ArrayContainer(
                fields=fields,
                inv_permittivities=res_inv_permittivities,
                inv_permeabilities=res_inv_permeabilities,
                detector_states=res_detector_states,
                recording_state=recording_state,
                electric_conductivity=arrays.electric_conductivity,
                magnetic_conductivity=arrays.magnetic_conductivity,
                initial_inv_permittivities=arrays.initial_inv_permittivities,
            )

        def reverse_body(sr_tuple, record_time_offset):
            """One reverse step: reconstruct the previous state, then pull the cotangent through the step."""
            state, running_cot = sr_tuple
            state = backward(
                state=state,
                config=config,
                objects=objects,
                key=key,
                record_detectors=False,
                reset_fields=False,
                record_time_offset=record_time_offset,
            )
            _, update_vjp = jax.vjp(
                partial(
                    forward_step_primals,
                    config=config,
                    objects=objects,
                    key=key,
                    record_detectors=True,
                    simulate_boundaries=True,
                    electric_conductivity=arrays.electric_conductivity,
                    magnetic_conductivity=arrays.magnetic_conductivity,
                ),
                state[0],
                state[1].fields.E,
                state[1].fields.H,
                state[1].fields.psi_E,
                state[1].fields.psi_H,
                state[1].inv_permittivities,
                state[1].inv_permeabilities,
                state[1].detector_states,
            )
            running_cot = update_vjp(running_cot)
            return state, running_cot

        def reverse_segment(
            segment_start: jax.Array,
            n_steps: int,
            fields_end: FieldState,
            fields_start: FieldState,
            recording_state: RecordingState | None,
            running_cot,
        ) -> tuple[RecordingState | None, tuple]:
            """Reverse one segment ``[segment_start, segment_start + n_steps]``.

            ``fields_end`` is the exact field state at the segment end (a checkpoint or the primal
            output). In segmented recording mode the segment is first re-simulated forward from
            ``fields_start`` to regenerate its interface record (indexed from 0 within the segment).
            """
            if segmented:
                record_time_offset: int | jax.Array = segment_start
                re_forward_body = partial(
                    forward,
                    config=config,
                    objects=objects,
                    key=key,
                    record_detectors=False,
                    record_boundaries=True,
                    simulate_boundaries=True,
                    record_time_offset=record_time_offset,
                )
                re_state = run_steps(
                    (segment_start, step_primal_container(fields_start, recording_state)),
                    n_steps,
                    re_forward_body,
                )
                recording_state = re_state[1].recording_state
                # The reverse sweep starts from the stored boundary state ``fields_end`` (equal to the
                # re-simulated end state up to reduction nondeterminism), not from ``re_state``.
            else:
                record_time_offset = 0

            end_step = segment_start + n_steps
            init = ((end_step, step_primal_container(fields_end, recording_state)), running_cot)
            (_, arr_out), running_cot = eqxi.while_loop(
                max_steps=n_steps,
                cond_fun=lambda sr: sr[0][0] > segment_start,
                body_fun=partial(reverse_body, record_time_offset=record_time_offset),
                init_val=init,
                kind="lax",
            )
            return arr_out.recording_state, running_cot

        # Segment boundary states: ``boundary_fields[i]`` is the exact field state at time
        # ``i * segment_length`` for ``i < num_segments`` and the primal output at the end of the run.
        boundary_fields = jax.tree.map(
            lambda buf, x: jnp.concatenate([buf, x[None]], axis=0), checkpoints, final_fields
        )

        def boundary_at(i) -> FieldState:
            return jax.tree.map(lambda buf: buf[i], boundary_fields)

        recording_state = res_recording_state
        # Tail segment (static length), reversed first: it ends at the primal output.
        if tail_length > 0:
            tail_start = jnp.asarray(num_full_segments * segment_length, dtype=jnp.int32)
            recording_state, cot_carry = reverse_segment(
                tail_start,
                tail_length,
                final_fields,
                boundary_at(num_full_segments),
                recording_state,
                cot_carry,
            )

        def full_segment_body(j, carry):
            recording_state, running_cot = carry
            i = num_full_segments - 1 - j
            segment_start = (i * segment_length).astype(jnp.int32)
            return reverse_segment(
                segment_start,
                segment_length,
                boundary_at(i + 1),
                boundary_at(i),
                recording_state,
                running_cot,
            )

        recording_state, cot_carry = jax.lax.fori_loop(
            0, num_full_segments, full_segment_body, (recording_state, cot_carry)
        )
        del recording_state

        return (
            None,  # cot[1],   E
            None,  # cot[2],   H
            None,  # cot[3],   psi_E
            None,  # cot[4],   psi_H
            cot_carry[5],  #    inv_permittivities
            cot_carry[6],  #    inv_permeabilities
            None,  # cot[7],   detector_states
            None,  # cot[8],   recording_state
        )

    reversible_fdtd_primal.defvjp(fdtd_fwd, fdtd_bwd)

    (
        time_step,
        E,
        H,
        psi_E,
        psi_H,
        inv_permittivities,
        inv_permeabilities,
        detector_states,
        recording_state,
    ) = reversible_fdtd_primal(
        E=arrays.fields.E,
        H=arrays.fields.H,
        psi_E=arrays.fields.psi_E,
        psi_H=arrays.fields.psi_H,
        inv_permittivities=arrays.inv_permittivities,
        inv_permeabilities=arrays.inv_permeabilities,
        detector_states=arrays.detector_states,
        recording_state=arrays.recording_state,
    )
    _close_pbar()

    out_arrs = ArrayContainer(
        fields=FieldState(
            E=E,
            H=H,
            psi_E=psi_E,
            psi_H=psi_H,
        ),
        inv_permittivities=inv_permittivities,
        inv_permeabilities=inv_permeabilities,
        detector_states=detector_states,
        recording_state=recording_state,
        electric_conductivity=arrays.electric_conductivity,
        magnetic_conductivity=arrays.magnetic_conductivity,
        initial_inv_permittivities=arrays.initial_inv_permittivities,
    )
    return time_step, out_arrs


def checkpointed_fdtd(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    stopping_condition: StoppingCondition | None = None,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    """Run an FDTD simulation with gradient checkpointing for memory efficiency.

    This implementation uses checkpointing to reduce memory usage during backpropagation
    by only storing the field state at certain intervals and recomputing intermediate
    states as needed.

    Args:
        arrays (ArrayContainer): Initial state of the simulation containing fields and materials
        objects (ObjectContainer): Collection of physical objects in the simulation
        config (SimulationConfig): Simulation parameters including checkpointing settings
        key (jax.Array): JAX PRNGKey for any stochastic operations
        stopping_condition (StoppingCondition, optional): Custom stopping condition on which simulation is halted.
            If none is provided, we default to TimeStepCondition (simulation progresses until max time is reached)
        show_progress (bool): Display a tqdm progress bar while the simulation runs.
            Set to False for a minor speed improvement; see the module-level
            benchmark note for overhead estimates. Defaults to True.
            The bar is driven entirely by ``io_callback`` at XLA execution
            time, so it works correctly whether the simulation is
            wrapped in ``jax.jit``.

    Returns:
        SimulationState: Tuple containing final time step and ArrayContainer with final state

    Notes:
        The number of checkpoints can be configured through config.gradient_config.num_checkpoints.
        More checkpoints reduce recomputation but increase memory usage.
    """
    arrays = arrays.reset()
    state = (jnp.asarray(0, dtype=jnp.int32), arrays)
    if stopping_condition is not None:
        stopping_condition = stopping_condition.setup(state, config, objects)
    else:
        stopping_condition = TimeStepCondition().setup(state, config, objects)

    pbar = _make_pbar(
        show_progress=show_progress,
        total_steps=config.time_steps_total,
        desc="FDTD (checkpointed)",
        progress_callback=progress_callback,
    )

    _forward_body = partial(
        forward,
        config=config,
        objects=objects,
        key=key,
        record_detectors=True,
        record_boundaries=config.invertible_optimization,
        simulate_boundaries=True,
    )
    _forward_body_with_progress, _close_pbar = _wrap_body_with_progress(_forward_body, pbar)

    state = eqxi.while_loop(
        max_steps=config.time_steps_total,
        cond_fun=partial(
            stopping_condition,
            config=config,
            objects=objects,
        ),
        body_fun=_forward_body_with_progress,
        init_val=state,
        kind="lax" if config.only_forward is None else "checkpointed",
        checkpoints=(None if config.gradient_config is None else config.gradient_config.num_checkpoints),
    )
    _close_pbar()

    return state


def custom_fdtd_forward(
    arrays: ArrayContainer,
    objects: ObjectContainer,
    config: SimulationConfig,
    key: jax.Array,
    reset_container: bool,
    record_detectors: bool,
    start_time: int | jax.Array,
    end_time: int | jax.Array,
    show_progress: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
) -> SimulationState:
    """Run a customizable forward FDTD simulation between specified time steps.

    This function provides fine-grained control over the simulation execution,
    allowing partial time evolution and customization of recording behavior.

    Args:
        arrays (ArrayContainer): Initial state of the simulation
        objects (ObjectContainer): Collection of physical objects
        config (SimulationConfig): Simulation parameters
        key (jax.Array): JAX PRNGKey for stochastic operations
        reset_container (bool): Whether to reset the array container before starting
        record_detectors (bool): Whether to record detector readings
        start_time (int | jax.Array): Time step to start from
        end_time (int | jax.Array): Time step to end at
        show_progress (bool): Display a tqdm progress bar while the simulation runs.
            Set to False for a minor speed improvement; see the module-level
            benchmark note for overhead estimates. Defaults to True.
            The bar is driven entirely by ``io_callback`` at XLA execution
            time, so it works correctly whether the simulation is
            wrapped in ``jax.jit``.

    Returns:
        SimulationState: Tuple containing final time step and ArrayContainer with final state

    Notes:
        This function is useful for implementing custom simulation strategies or
        running partial simulations for analysis purposes.
    """
    if reset_container:
        arrays = arrays.reset()
    state = (jnp.asarray(start_time, dtype=jnp.int32), arrays)

    # start_time and end_time must be statically known Python ints here so that
    # we can compute n_steps for the progress bar without triggering JAX
    # concretization.  They are always statically known at call sites of this
    # function (they control the loop bound, not an array value).
    if isinstance(start_time, jax.Array) or isinstance(end_time, jax.Array):
        # Traced arrays: skip the progress bar entirely to avoid concretization.
        show_progress = False
        progress_callback = None
        n_steps = 0
    else:
        n_steps = int(end_time) - int(start_time)

    pbar = _make_pbar(
        show_progress=show_progress,
        total_steps=n_steps,
        desc="FDTD (forward)",
        step_offset=0 if not show_progress and progress_callback is None else int(start_time),
        progress_callback=progress_callback,
    )

    _forward_body = partial(
        forward,
        config=config,
        objects=objects,
        key=key,
        record_detectors=record_detectors,
        record_boundaries=False,
        simulate_boundaries=True,
    )
    _forward_body_with_progress, _close_pbar = _wrap_body_with_progress(_forward_body, pbar)

    state = eqxi.while_loop(
        max_steps=config.time_steps_total,
        cond_fun=lambda s: end_time > s[0],
        body_fun=_forward_body_with_progress,
        init_val=state,
        kind="lax",
        checkpoints=None,
    )
    _close_pbar()

    return state
