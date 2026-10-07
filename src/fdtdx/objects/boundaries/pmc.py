from typing import Literal

import jax
from typing_extensions import override

from fdtdx.colors import XKCD_ORANGE, Color
from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.boundaries.boundary import BaseBoundary


@autoinit
class PerfectMagneticConductor(BaseBoundary):
    """Implements perfect magnetic conductor (PMC) boundary conditions.

    PMC enforces H_tangential = 0 at the boundary wall. Unlike PEC, this requires
    explicit zeroing of tangential H components after each H field update, because
    the Yee grid naturally terminates as PEC (zero E_tangential), not PMC.

    Component zeroing per axis:
    - PMC on x-face: zero Hy, Hz (tangential to x)
    - PMC on y-face: zero Hx, Hz (tangential to y)
    - PMC on z-face: zero Hx, Hy (tangential to z)
    """

    #: RGB color tuple for visualization. Defaults to orange.
    color: Color | None = frozen_field(default=XKCD_ORANGE)

    @property
    @override
    def descriptive_name(self) -> str:
        """Gets a human-readable name describing this PMC boundary's location.

        Returns:
            str: Description like "min_x" or "max_z" indicating position
        """
        axis_str = "x" if self.axis == 0 else "y" if self.axis == 1 else "z"
        direction_str = "min" if self.direction == "-" else "max"
        return f"{direction_str}_{axis_str}"

    @property
    @override
    def thickness(self) -> int:
        """Gets the thickness of the PMC boundary layer in grid points.

        Returns:
            int: Number of grid points in the boundary layer (always 1 for PMC)
        """
        return 1

    @property
    def tangential_components(self) -> tuple[int, int]:
        """Gets the indices of H field components tangential to this boundary.

        Returns:
            tuple[int, int]: Indices of the two tangential components (0=Hx, 1=Hy, 2=Hz)
        """
        if self.axis == 0:
            return (1, 2)  # Hy, Hz tangential to x-face
        elif self.axis == 1:
            return (0, 2)  # Hx, Hz tangential to y-face
        else:
            return (0, 1)  # Hx, Hy tangential to z-face

    @override
    def apply_pad_correction(
        self,
        padded_fields: jax.Array,
        volume_shape: tuple[int, int, int],
        resolution: float,
        width: int = 1,
        field_type: Literal["E", "H"] | None = None,
    ) -> jax.Array:
        """Fill the halo beyond the wall with the magnetic-wall image of the interior.

        The plane sits on the tangential-``H`` node at the centre of the wall cell, so for a ``"-"``
        face the exterior reaches half a cell into the domain and the tangential ``E`` and normal
        ``H`` samples of the wall cell are overwritten too (see
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
        return self._apply_image_halo(padded_fields, width, field_type, wall=1)

    @property
    @override
    def exterior_cell_range(self) -> tuple[int, int] | None:
        """The wall cell of a ``"-"`` face, whose lower half lies outside the magnetic plane.

        The plane is the tangential-``H`` node at the cell's centre, so on a ``"-"`` face ``Ex``,
        ``Ey`` and ``Hz`` of that cell sit half a cell below it and ``Ez``, ``Hx``, ``Hy`` sit
        exactly on it, where a magnetic wall drives them to zero. A ``"+"`` face keeps its whole
        wall cell.
        """
        if self.direction != "-":
            return None
        return self._grid_slice_tuple[self.axis]

    @override
    def apply_post_H_update(self, H: jax.Array) -> jax.Array:
        """Zeros tangential H components at this PMC boundary face.

        ``config.symmetry`` never creates this boundary: a magnetic symmetry plane sits half a cell
        below the reduced domain, where the zero field halo already is the mirror, so zeroing
        tangential H at the first cell would impose the condition half a cell off the plane (see
        :func:`~fdtdx.fdtd.symmetry.make_symmetry_walls`). This method therefore only ever runs for a
        user-placed PMC boundary, whose behaviour is unchanged.
        """
        comp1, comp2 = self.tangential_components
        sx, sy, sz = self.grid_slice
        H = H.at[comp1, sx, sy, sz].set(0)
        H = H.at[comp2, sx, sy, sz].set(0)
        return H
