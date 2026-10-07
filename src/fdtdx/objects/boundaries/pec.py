from typing import Literal

import jax
from typing_extensions import override

from fdtdx.colors import XKCD_RED, Color
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.boundaries.boundary import BaseBoundary


@autoinit
class PerfectElectricConductor(BaseBoundary):
    """Implements perfect electric conductor (PEC) boundary conditions.

    PEC enforces E_tangential = 0 at the boundary wall. Zero-padding provides
    the correct ghost cell values for H (and for E_normal), but the curl_H
    computation at the boundary produces nonzero updates for tangential E
    components. This class explicitly zeros them after each E update.

    Component zeroing per axis:
    - PEC on x-face: zero Ey, Ez (tangential to x)
    - PEC on y-face: zero Ex, Ez (tangential to y)
    - PEC on z-face: zero Ex, Ey (tangential to z)

    Note on dispersive media: the tangential polarization ``P`` at a
    PEC-adjacent cell can be left one step out of sync with the freshly
    zeroed tangential ``E`` because ``P`` is updated from ``E_prev``. This
    converges on the next E-update — since ``E_tangential`` is clamped to
    zero the next ``P`` relaxes back toward zero, which is the physically
    correct behavior at a perfect conductor.
    """

    #: RGB color tuple for visualization. Defaults to red.
    color: Color | None = frozen_field(default=XKCD_RED)

    @property
    @override
    def descriptive_name(self) -> str:
        """Gets a human-readable name describing this PEC boundary's location.

        Returns:
            str: Description like "min_x" or "max_z" indicating position
        """
        axis_str = "x" if self.axis == 0 else "y" if self.axis == 1 else "z"
        direction_str = "min" if self.direction == "-" else "max"
        return f"{direction_str}_{axis_str}"

    @property
    @override
    def thickness(self) -> int:
        """Gets the thickness of the PEC boundary layer in grid points.

        Returns:
            int: Number of grid points in the boundary layer (always 1 for PEC)
        """
        return 1

    @property
    def tangential_components(self) -> tuple[int, int]:
        """Gets the indices of E field components tangential to this boundary.

        Returns:
            tuple[int, int]: Indices of the two tangential components (0=Ex, 1=Ey, 2=Ez)
        """
        if self.axis == 0:
            return (1, 2)  # Ey, Ez tangential to x-face
        elif self.axis == 1:
            return (0, 2)  # Ex, Ez tangential to y-face
        else:
            return (0, 1)  # Ex, Ey tangential to z-face

    @override
    def apply_pad_correction(
        self,
        padded_fields: jax.Array,
        volume_shape: tuple[int, int, int],
        resolution: float,
        width: int = 1,
        field_type: Literal["E", "H"] | None = None,
    ) -> jax.Array:
        """Fill the halo beyond the wall with the electric-wall image of the interior.

        The plane sits on the tangential-``E`` node row at the lower edge of the wall cell, so for a
        ``"+"`` face the exterior reaches half a cell into the domain and the normal ``E`` and
        tangential ``H`` samples of the wall cell are overwritten too (see
        :meth:`~fdtdx.objects.boundaries.boundary.BaseBoundary._apply_image_halo`).

        Args:
            padded_fields: Padded field array of shape (3, Nx+2w, Ny+2w, Nz+2w)
            volume_shape: Full simulation volume shape (Nx, Ny, Nz)
            resolution: Grid resolution in meters
            width: Number of ghost cells per face in ``padded_fields``
            field_type: Whether ``padded_fields`` holds the electric or magnetic field

        Returns:
            Padded fields with the image halo written
        """
        del volume_shape, resolution
        return self._apply_image_halo(padded_fields, width, field_type, wall=-1)

    @property
    @override
    def exterior_cell_range(self) -> tuple[int, int] | None:
        """The wall cell of a ``"+"`` face, whose upper half lies outside the electric plane.

        The plane is the tangential-``E`` node row at the cell's lower edge, so on a ``"+"`` face
        ``Ez``, ``Hx`` and ``Hy`` of that cell sit half a cell past it and ``Ex``, ``Ey``, ``Hz``
        sit exactly on it, where the wall zeroes them. A ``"-"`` face keeps its whole wall cell.
        """
        if self.direction != "+":
            return None
        return self._grid_slice_tuple[self.axis]

    @override
    def apply_post_E_update(self, E: jax.Array) -> jax.Array:
        """Zeros tangential E components at this PEC boundary face."""
        comp1, comp2 = self.tangential_components
        sx, sy, sz = self.grid_slice
        E = E.at[comp1, sx, sy, sz].set(0)
        E = E.at[comp2, sx, sy, sz].set(0)
        return E
