from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal, Self

import jax
import jax.numpy as jnp

from fdtdx.colors import XKCD_DARK_ORANGE, Color
from fdtdx.config import SimulationConfig
from fdtdx.core.axis import get_oriented_transverse_axes
from fdtdx.core.jax.pytrees import autoinit, frozen_field, private_field
from fdtdx.core.misc import linear_interpolated_indexing, normalize_polarization_for_source
from fdtdx.core.null import NULL
from fdtdx.core.switch import OnOffSwitch
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.object import SimulationObject
from fdtdx.objects.sources.profile import SingleFrequencyProfile, TemporalProfile
from fdtdx.typing import SliceTuple3D


@autoinit
class Source(SimulationObject, ABC):
    #: the wave-character
    wave_character: WaveCharacter = frozen_field()

    #: the temporal profile, uses single frequency
    temporal_profile: TemporalProfile = SingleFrequencyProfile()

    #: the static amplitude factor
    static_amplitude_factor: float = frozen_field(default=1.0)

    #: the on-off switch
    switch: OnOffSwitch = frozen_field(default=OnOffSwitch())

    #: color of the object
    color: Color | None = frozen_field(default=XKCD_DARK_ORANGE)

    _is_on_at_time_step_arr: jax.Array = private_field()
    _time_step_to_on_idx: jax.Array = private_field()

    def validate_placement(self, objects) -> list[str]:
        """Reject a source that sits entirely in the dead half-cell of a PEC/PMC wall.

        A wall plane is a node row inside its cell, so on one of its two faces (a ``"-"`` magnetic
        wall, a ``"+"`` electric one) the outer half of the wall cell lies outside the modelled
        domain: every Yee sample there is either a mirror of an interior sample or is driven to zero
        on the plane (see
        :meth:`~fdtdx.objects.boundaries.boundary.BaseBoundary.exterior_cell_range`). A source whose
        whole footprint falls in those cells therefore radiates nothing — it is overwritten by the
        image halo before the curl ever sees it — while the simulation runs to completion, stays
        finite and reports no error. This turns that into an error at placement time.

        Args:
            objects (ObjectContainer): The fully-resolved container of all placed objects.

        Returns:
            list[str]: Error messages describing invalid placement, or ``[]``.
        """
        errors = list(super().validate_placement(objects))
        own = self.grid_slice_tuple
        for boundary in objects.boundary_objects:
            dead = boundary.exterior_cell_range
            if dead is None:
                continue
            axis = boundary.axis
            if not (dead[0] <= own[axis][0] and own[axis][1] <= dead[1]):
                continue  # the source reaches into the domain proper
            other = [a for a in range(3) if a != axis]
            if any(
                own[a][1] <= boundary.grid_slice_tuple[a][0] or boundary.grid_slice_tuple[a][1] <= own[a][0]
                for a in other
            ):
                continue  # no overlap in the plane of the wall
            # Only a "-" magnetic and a "+" electric wall have a dead cell, so the face fixes the
            # wall type and the direction the source has to move to get back into the domain.
            wall_kind = "magnetic" if boundary.direction == "-" else "electric"
            inward = "+1" if boundary.direction == "-" else "-1"
            errors.append(
                f"Source '{self.name}' lies entirely inside the wall cell of the {wall_kind} boundary "
                f"'{boundary.name}' ({'xyz'[axis]} axis, '{boundary.direction}' face), cells "
                f"{dead[0]}:{dead[1]}. The wall plane cuts that cell in half and the source's side of "
                f"it is outside the simulated domain, so the source cannot drive any field: its "
                f"samples are either overwritten by the wall's mirror image or held at zero on the "
                f"plane. Move the source {inward} cell along {'xyz'[axis]}, or drop the wall."
            )
        return errors

    def is_on_at_time_step(self, time_step: jax.Array) -> jax.Array:
        return self._is_on_at_time_step_arr[time_step]

    def adjust_time_step_by_on_off(self, time_step: jax.Array) -> jax.Array:
        time_step = linear_interpolated_indexing(
            point=time_step.reshape(1),
            arr=self._time_step_to_on_idx,
        )
        return time_step

    def _update_on_arrays(self) -> Self:
        # determine number of time steps on
        on_list = self.switch.calculate_on_list(
            time_step_duration=self._config.time_step_duration,
            num_total_time_steps=self._config.time_steps_total,
        )
        on_arr = jnp.asarray(on_list, dtype=jnp.bool)
        self = self.aset("_is_on_at_time_step_arr", on_arr, create_new_ok=True)
        # calculate mapping time step -> on index
        time_to_arr_idx_list = self.switch.calculate_time_step_to_on_arr_idx(
            time_step_duration=self._config.time_step_duration,
            num_total_time_steps=self._config.time_steps_total,
        )
        time_to_arr_idx_arr = jnp.asarray(time_to_arr_idx_list, dtype=jnp.int32)
        self = self.aset("_time_step_to_on_idx", time_to_arr_idx_arr, create_new_ok=True)
        return self

    def apply(
        self,
        key: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        dispersive_c1: jax.Array | None = None,
        dispersive_c2: jax.Array | None = None,
        dispersive_c3: jax.Array | None = None,
        electric_conductivity: jax.Array | None = None,
    ) -> Self:
        self = super().apply(
            key=key,
            inv_permittivities=inv_permittivities,
            inv_permeabilities=inv_permeabilities,
            dispersive_c1=dispersive_c1,
            dispersive_c2=dispersive_c2,
            dispersive_c3=dispersive_c3,
            electric_conductivity=electric_conductivity,
        )
        self = self._update_on_arrays()
        return self

    def place_on_grid(
        self: Self,
        grid_slice_tuple: SliceTuple3D,
        config: SimulationConfig,
        key: jax.Array,
    ) -> Self:
        self = super().place_on_grid(
            grid_slice_tuple=grid_slice_tuple,
            config=config,
            key=key,
        )
        self = self._update_on_arrays()
        return self

    def _resolve_time_signal_config(self, config: SimulationConfig | None) -> SimulationConfig:
        """Resolve the simulation config used for source time-signal sampling."""
        if config is not None:
            return config
        if self._config is NULL:
            raise ValueError(
                "A SimulationConfig is required to sample or plot a source time signal. "
                "Call place_objects(...) before calling this method, or pass config=... explicitly."
            )
        return self._config

    def sample_time_signal(
        self,
        config: SimulationConfig | None = None,
    ):
        """Sample this source's time signal for plotting or analysis.

        The returned signal uses the FDTD time grid from the supplied config, or
        from self._config if the source has already been placed.
        """
        config = self._resolve_time_signal_config(config)
        return self.temporal_profile.sample_time_signal(
            period=self.wave_character.get_period(),
            time_step_duration=config.time_step_duration,
            num_time_steps=config.time_steps_total,
            phase_shift=self.wave_character.phase_shift,
        )

    def frequency_spectrum(
        self,
        config: SimulationConfig | None = None,
        normalize: bool = True,
    ):
        """Return the one-sided FFT magnitude of this source's sampled time signal.

        This is intended for analyzing or visualizing its frequency spectrum.
        """
        config = self._resolve_time_signal_config(config)
        return self.temporal_profile.frequency_spectrum(
            period=self.wave_character.get_period(),
            time_step_duration=config.time_step_duration,
            num_time_steps=config.time_steps_total,
            phase_shift=self.wave_character.phase_shift,
            normalize=normalize,
        )

    def plot_time_signal_and_spectrum(
        self,
        config: SimulationConfig | None = None,
        filename: str | Path | None = None,
        **kwargs,
    ):
        """Plot this source's sampled time signal and one-sided frequency spectrum."""
        config = self._resolve_time_signal_config(config)
        return self.temporal_profile.plot_time_signal_and_spectrum(
            period=self.wave_character.get_period(),
            time_step_duration=config.time_step_duration,
            num_time_steps=config.time_steps_total,
            phase_shift=self.wave_character.phase_shift,
            filename=filename,
            **kwargs,
        )

    @abstractmethod
    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        """Update the electric field component.

        Args:
            E (jax.Array): Current electric field array.
            inv_permittivities (jax.Array): Inverse permittivity values.
            inv_permeabilities (jax.Array | float): Inverse permeability values.
            time_step (jax.Array): Current simulation time step.
            inverse (bool): Whether to perform inverse update for backpropagation.

        Returns:
            jax.Array: Updated electric field array.
        """
        raise NotImplementedError()

    @abstractmethod
    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        """Update the magnetic field component.

        Args:
            H (jax.Array): Current magnetic field array.
            inv_permittivities (jax.Array): Inverse permittivity values.
            inv_permeabilities (jax.Array | float): Inverse permeability values.
            time_step (jax.Array): Current simulation time step.
            inverse (bool): Whether to perform inverse update for backpropagation.

        Returns:
            jax.Array: Updated magnetic field array.
        """
        raise NotImplementedError()


@autoinit
class DirectionalPlaneSourceBase(Source, ABC):
    """Base class for directional plane wave sources.

    Implements common functionality for plane wave sources that propagate in a specific
    direction. Provides methods for calculating wave vectors and orthogonal field components.

    """

    #: Direction of propagation ('+' or '-' along propagation axis).
    direction: Literal["+", "-"] = frozen_field()

    @property
    def propagation_axis(self) -> int:
        return self.grid_shape.index(1)

    @property
    def horizontal_axis(self) -> int:
        return get_oriented_transverse_axes(self.propagation_axis)[0]

    @property
    def vertical_axis(self) -> int:
        return get_oriented_transverse_axes(self.propagation_axis)[1]


@autoinit
class HardConstantAmplitudePlanceSource(DirectionalPlaneSourceBase):
    amplitude: float = frozen_field(default=1.0)
    fixed_E_polarization_vector: tuple[float, float, float] | None = frozen_field(default=None)
    fixed_H_polarization_vector: tuple[float, float, float] | None = frozen_field(default=None)

    def update_E(
        self,
        E: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ) -> jax.Array:
        del inv_permittivities, inv_permeabilities
        if inverse:
            return E
        delta_t = self._config.time_step_duration
        time_phase = (
            2 * jnp.pi * time_step * delta_t / self.wave_character.get_period() + self.wave_character.phase_shift
        )
        magnitude = jnp.real(self.amplitude * jnp.exp(-1j * time_phase))
        magnitude = magnitude * self.static_amplitude_factor
        e_pol, _ = normalize_polarization_for_source(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
            fixed_E_polarization_vector=self.fixed_E_polarization_vector,
            fixed_H_polarization_vector=self.fixed_E_polarization_vector,
            dtype=self._config.dtype,
        )
        E_update = e_pol[:, None, None, None] * magnitude

        E = E.at[:, *self.grid_slice].set(E_update.astype(E.dtype))
        return E

    def update_H(
        self,
        H: jax.Array,
        inv_permittivities: jax.Array,
        inv_permeabilities: jax.Array | float,
        time_step: jax.Array,
        inverse: bool,
    ):
        del inv_permeabilities, inv_permittivities
        if inverse:
            return H
        delta_t = self._config.time_step_duration
        time_phase = (
            2 * jnp.pi * time_step * delta_t / self.wave_character.get_period() + self.wave_character.phase_shift
        )
        magnitude = jnp.real(self.amplitude * jnp.exp(-1j * time_phase))
        magnitude = magnitude * self.static_amplitude_factor
        _, h_pol = normalize_polarization_for_source(
            direction=self.direction,
            propagation_axis=self.propagation_axis,
            fixed_E_polarization_vector=self.fixed_E_polarization_vector,
            fixed_H_polarization_vector=self.fixed_E_polarization_vector,
            dtype=self._config.dtype,
        )
        H_update = h_pol[:, None, None, None] * magnitude

        H = H.at[:, *self.grid_slice].set(H_update.astype(H.dtype))
        return H
