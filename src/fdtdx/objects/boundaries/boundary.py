from abc import ABC, abstractmethod
from typing import Literal

import jax

from fdtdx.core.jax.pytrees import autoinit, frozen_field, frozen_private_field
from fdtdx.core.physics.symmetry import fill_image_halo
from fdtdx.objects.object import SimulationObject
from fdtdx.typing import GridShape3D, Slice3D, SliceTuple3D


@autoinit
class BaseBoundary(SimulationObject, ABC):
    """Base class for all boundary conditions in FDTD simulations.

    This class defines the interface for boundary conditions, including methods
    for initializing, resetting, and updating boundary states, as well as updating
    the electric and magnetic fields at the boundaries.
    """

    #: Principal axis for boundary (0=x, 1=y, 2=z)
    axis: int = frozen_field()

    #: Direction along axis ("+" or "-")
    direction: Literal["+", "-"] = frozen_field()

    #: Whether this boundary was inserted by ``config.symmetry`` as a mirror plane rather than
    #: requested by the user as an ordinary boundary. Only electric symmetry planes get a wall
    #: object (see ``make_symmetry_walls``, which sets this flag), and such a wall asserts that the
    #: model is mirror-symmetric about it, which lets the detector co-location stencil fill its halo
    #: with the mirrored interior values instead of zeros. A user-placed PEC/PMC boundary makes no
    #: such claim, so its halo stays as it was.
    _is_symmetry_wall: bool = frozen_private_field(default=False)

    @property
    @abstractmethod
    def descriptive_name(self) -> str:
        """Gets a human-readable name describing this boundary's location."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def thickness(self) -> int:
        """Gets the thickness of the boundary in grid points."""
        raise NotImplementedError()

    @property
    def uses_wrap_padding(self) -> bool:
        """Whether this boundary's axis should use wrap (periodic) padding.

        Returns True for boundaries that connect opposite sides of the domain
        (periodic, Bloch). Returns False for terminating boundaries (PEC, PMC, PML).
        """
        return False

    def apply_pad_correction(
        self,
        padded_fields: jax.Array,
        volume_shape: tuple[int, int, int],
        resolution: float,
        width: int = 1,
        field_type: Literal["E", "H"] | None = None,
    ) -> jax.Array:
        """Apply boundary-specific correction to padded fields.

        Called after basic wrap/constant padding. Default is a no-op.
        Subclasses like BlochBoundary override this to apply phase shifts
        to ghost cells, and PEC/PMC walls to write image (mirror) halos.

        Args:
            padded_fields: Padded field array of shape (3, Nx+2w, Ny+2w, Nz+2w)
            volume_shape: Full simulation volume shape (Nx, Ny, Nz)
            resolution: Grid resolution in meters
            width: Number of ghost cells per face in ``padded_fields``. The curl stencil of order
                ``2r`` uses ``width = r``. Defaults to 1 (the classic Yee scheme).
            field_type: Whether ``padded_fields`` holds the electric or magnetic field, or ``None``
                when the caller does not need a field-type-dependent correction (the anisotropic
                material averaging and the detector co-location stencil, both of which only ever
                use ``width = 1``). Defaults to ``None``.

        Returns:
            Padded fields with boundary-specific corrections applied
        """
        del volume_shape, resolution, width, field_type
        return padded_fields

    @property
    def exterior_cell_range(self) -> tuple[int, int] | None:
        """Cells along :attr:`axis` in which every Yee sample lies outside the modelled domain.

        A wall plane is a *node* row, not a cell edge, so on one of its two faces the plane cuts the
        wall cell in half and the outer half is not part of the simulated region. Every sample of
        such a cell is then either strictly outside the plane (and therefore just a mirror of an
        interior sample, written by :meth:`_apply_image_halo`) or sits exactly on the plane, where
        the wall drives it to zero. Nothing placed there — a source, a design voxel — can influence
        the field, and nothing read there is an independent result.

        Returns:
            tuple[int, int] | None: ``(start, stop)`` grid bounds of the dead cells, or ``None`` when
            this boundary has none (the default; PML, Bloch and the two well-behaved wall faces).
        """
        return None

    def _padded_cross_section(self, width: int) -> tuple[slice, slice, slice]:
        """Padded-index slices of this boundary's footprint on the axes it does not terminate.

        The entry for :attr:`axis` is ``slice(None)``; the other two cover the boundary's own grid
        extent shifted by the halo ``width``. Restricting the image halo to that cross-section keeps
        the fill of two perpendicular walls independent of the order they are applied in — the
        corner halo they would fight over is never read by the curl of an in-domain sample.

        Args:
            width (int): Number of ghost cells per face in the padded array.

        Returns:
            tuple[slice, slice, slice]: The cross-section, indexed in padded coordinates.
        """
        region: list[slice] = []
        for a in range(3):
            if a == self.axis:
                region.append(slice(None))
            else:
                lo, hi = self._grid_slice_tuple[a]
                region.append(slice(lo + width, hi + width))
        return region[0], region[1], region[2]

    def _apply_image_halo(
        self,
        padded_fields: jax.Array,
        width: int,
        field_type: Literal["E", "H"] | None,
        wall: int,
    ) -> jax.Array:
        """Write the mirror image of the interior into the samples outside this wall.

        The plane is the node the wall drives to zero: the tangential-E node at the lower edge of
        the wall cell for an electric wall, the tangential-H node at its centre for a magnetic one.

        This runs at every ``width``, including the classic Yee scheme's ``width = 1``. At width 1
        the halo the curl actually reads (the min-face ``H`` ghosts and the max-face ``E`` ghosts;
        the staggered differences never touch the other two) only ever feeds components the wall
        zeroes immediately afterwards, so for a ``"-"`` electric and a ``"+"`` magnetic wall the
        image and the zero halo are indistinguishable. For the other two faces it is *not* a no-op,
        and the difference is a bug fix rather than a refinement: a ``"-"`` magnetic and a ``"+"``
        electric wall leave half of their wall cell outside the plane
        (see :meth:`exterior_cell_range`), and with a zero halo those exterior samples decouple from
        the domain entirely — at a min-face PMC, ``Ex``, ``Ey`` and ``Hz`` of cell 0 form a closed
        2D system that no interior field drives and that drives no interior field, yet still reports
        plausible values into any full-volume detector or energy integral. Filling the image instead
        makes them carry the mirror of the first interior cell, which is what they represent, and
        leaves every interior sample bit-identical (the overwritten samples only ever feed each
        other).

        Args:
            padded_fields (jax.Array): Padded field array of shape ``(3, Nx+2w, Ny+2w, Nz+2w)``.
            width (int): Number of ghost cells per face.
            field_type (Literal["E", "H"] | None): Which field is padded, or ``None`` to skip.
            wall (int): ``-1`` for an electric wall (PEC), ``+1`` for a magnetic wall (PMC).

        Returns:
            jax.Array: The padded array with the exterior samples replaced by their images.
        """
        if field_type is None:
            return padded_fields
        lo, hi = self._grid_slice_tuple[self.axis]
        # "-" terminates the domain from below, so the wall cell is the slab's last one.
        wall_cell = hi - 1 if self.direction == "-" else lo
        # The plane is the node the wall drives to zero inside its own cell, on either face: the
        # tangential-E node at the cell's lower edge for an electric wall, the tangential-H node at
        # the cell centre for a magnetic one. An electric "+" face therefore leaves the upper half of
        # its cell outside the domain, and a magnetic "-" face the lower half of its.
        plane_twice = 2 * wall_cell + (0 if wall == -1 else 1)
        return fill_image_halo(
            padded_fields,
            axis=self.axis,
            width=width,
            field_type=field_type,
            wall=wall,
            plane_twice=plane_twice,
            side=self.direction,
            region=self._padded_cross_section(width),
        )

    def apply_post_E_update(self, E: jax.Array) -> jax.Array:
        """Apply boundary-specific enforcement after E field update.

        Called after each E field update (forward and reverse). Default is a no-op.
        Subclasses like PEC override this to zero tangential E components.

        Args:
            E: Electric field array of shape (3, Nx, Ny, Nz)

        Returns:
            E field with boundary conditions enforced
        """
        return E

    def apply_post_H_update(self, H: jax.Array) -> jax.Array:
        """Apply boundary-specific enforcement after H field update.

        Called after each H field update (forward and reverse). Default is a no-op.
        Subclasses like PMC override this to zero tangential H components.

        Args:
            H: Magnetic field array of shape (3, Nx, Ny, Nz)

        Returns:
            H field with boundary conditions enforced
        """
        return H

    def apply_field_reset(self, fields: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Apply boundary-specific field reset during backward propagation.

        Called during the backward pass to restore each boundary region to its
        correct state. Default is a no-op. Subclasses like PML override this to
        zero their region; BlochBoundary overrides to copy from the opposite face.

        Args:
            fields: Dict mapping field names (e.g. 'E', 'H') to their arrays

        Returns:
            Updated fields dict with this boundary's reset applied
        """
        return fields

    def interface_grid_shape(self, width: int = 1) -> GridShape3D:
        """Shape of the interface slab of this boundary (see :meth:`interface_slice`).

        Args:
            width (int): Thickness of the slab in cells along the boundary axis. Defaults to 1.

        Returns:
            GridShape3D: The boundary's grid shape with the axis extent replaced by ``width``.
        """
        shape = list(self.grid_shape)
        shape[self.axis] = width
        return shape[0], shape[1], shape[2]

    def interface_slice_tuple(self, width: int = 1) -> SliceTuple3D:
        """Grid bounds of the ``width`` innermost cells of this boundary along its axis.

        The interface slab is the part of the boundary region adjacent to the interior. The
        reversible adjoint records the fields there every step, because the boundary update is not
        time reversible and the interior reconstruction reads these cells through the curl stencil.
        A stencil with ``r`` taps per side needs ``width = 2r - 1``.

        Args:
            width (int): Thickness of the slab in cells. Defaults to 1 (classic Yee scheme).

        Returns:
            SliceTuple3D: ``((x0, x1), (y0, y1), (z0, z1))`` of the slab.
        """
        if width < 1 or width > self.grid_shape[self.axis]:
            raise ValueError(
                f"Interface width {width} must be in [1, {self.grid_shape[self.axis]}] for boundary {self.name}"
            )
        slice_list = [*self._grid_slice_tuple]
        lo, hi = self._grid_slice_tuple[self.axis]
        if self.direction == "+":
            slice_list[self.axis] = (lo, lo + width)
        elif self.direction == "-":
            slice_list[self.axis] = (hi - width, hi)
        return slice_list[0], slice_list[1], slice_list[2]

    def interface_slice(self, width: int = 1) -> Slice3D:
        """Slice form of :meth:`interface_slice_tuple`.

        Args:
            width (int): Thickness of the slab in cells. Defaults to 1.

        Returns:
            Slice3D: The slab as a tuple of three slices.
        """
        bounds = self.interface_slice_tuple(width)
        return slice(*bounds[0]), slice(*bounds[1]), slice(*bounds[2])
