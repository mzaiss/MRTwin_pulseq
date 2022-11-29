from __future__ import annotations
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.io import loadmat
from .raw_sim_data import RawSimData
from .. import util


def sigmoid(trajectory: torch.Tensor, nyquist: torch.Tensor) -> torch.Tensor:
    """Differentiable approximation of the sinc voxel dephasing function.

    The true dephasing function of a sinc-shaped voxel (in real space) is a
    box - function, with the FFT conform size [-nyquist, nyquist[. This is not
    differentiable, so we approximate the edges with a narrow sigmod at
    ±(nyquist + 0.5). The difference is neglegible at usual nyquist freqs.
    """
    return torch.prod(torch.sigmoid(
        (nyquist - trajectory[:, :3].abs() + 0.5) * 100
    ), dim=1)


class VoxelGridPhantom:
    """Class for using typical phantoms like those provided by BrainWeb.

    The data is assumed to be defined by a uniform cartesian grid of samples.
    As it is bandwidth limited, we assume that there is no signal above the
    Nyquist frequency. This leads to the usage of sinc-shaped voxels.

    This phantom has two FOVs: ``base_fov`` encodes the physical size in meters
    and is set on load. ``rel_fov`` is initially 1 and is changed by some
    operations like reducing the phantom to a few slices. This allows to either
    use only ``rel_fov`` in the simulation so that the sequence can still
    assume an FOV of 1, or to use ``base_fov * rel_fov`` and use SI units in
    the sequence definition.

    Attributes
    ----------
    PD : torch.Tensor
        (sx, sy, sz) tensor containing the Proton Density
    T1 : torch.Tensor
        (sx, sy, sz) tensor containing the T1 relaxation
    T2 : torch.Tensor
        (sx, sy, sz) tensor containing the T2 relaxation
    T2dash : torch.Tensor
        (sx, sy, sz) tensor containing the T2' dephasing
    D : torch.Tensor
        (sx, sy, sz) tensor containing the Diffusion coefficient
    B0 : torch.Tensor
        (sx, sy, sz) tensor containing the B0 inhomogeneities
    B1 : torch.Tensor
        (coil_count, sx, sy, sz) tensor of RF coil profiles
    coil_sens : torch.Tensor
        (coil_count, sx, sy, sz) tensor of coil sensitivities
    base_fov : torch.Tensor
        Base size of the original loaded data, in meters.
    rel_fov : torch.Tensor
        Actual phantom size relative to ``base_fov``.
    """

    def __init__(
        self,
        PD: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        T2dash: torch.Tensor,
        D: torch.Tensor,
        B0: torch.Tensor,
        B1: torch.Tensor,
        coil_sens: torch.Tensor,
        base_fov: torch.Tensor,
        rel_fov: torch.Tensor,
    ) -> None:
        """Set the phantom attributes to the provided parameters.

        This function does no cloning nor contain any other funcionality. You
        probably want to use :meth:`brainweb` to load a phantom instead.
        """
        self.PD = PD
        self.T1 = T1
        self.T2 = T2
        self.T2dash = T2dash
        self.D = D
        self.B0 = B0
        self.B1 = B1
        self.coil_sens = coil_sens
        self.base_fov = base_fov
        self.rel_fov = rel_fov

    def build(self, PD_threshold: float = 1e-6,
              use_SI_FoV: bool = False) -> RawSimData:
        """Build a :class:`RawSimData` instance for simulation.

        Arguments
        ---------
        PD_threshold: float
            All voxels with a proton density below this value are ignored.
        use_SI_FoV: bool
            If set to ``True``, the built :class:`RawSimData` will have its actual
            physical size in meters. If set to ``False``, the ``rel_fov`` is
            used, which means a sequence FOV of 1 is assumed.
        """
        mask = self.PD > PD_threshold

        fov = (self.base_fov * self.rel_fov) if use_SI_FoV else (self.rel_fov)
        shape = self.PD.shape
        pos_x, pos_y, pos_z = torch.meshgrid(
            torch.linspace(-fov[0]/2, fov[0]/2, int(shape[0]) + 1)[:-1],
            torch.linspace(-fov[1]/2, fov[1]/2, int(shape[1]) + 1)[:-1],
            torch.linspace(-fov[2]/2, fov[2]/2, int(shape[2]) + 1)[:-1],
        )
        voxel_pos = torch.stack([
            pos_x[mask].flatten(),
            pos_y[mask].flatten(),
            pos_z[mask].flatten()
        ], dim=1)

        # RawSimData can't set the device: this is captured by the dephasing fn
        nyquist = torch.tensor(shape, device=util.get_device()) / 2

        return RawSimData(
            self.PD[mask],
            self.T1[mask],
            self.T2[mask],
            self.T2dash[mask],
            self.D[mask],
            self.B0[mask],
            self.B1[:, mask],
            self.coil_sens[:, mask],
            self.base_fov * self.rel_fov,  # Always SI, only used for diffusion
            voxel_pos,
            (torch.tensor(mask.shape) / 2).tolist(),
            dephasing_func=lambda t: sigmoid(t, nyquist),
            recover_func=lambda d: recover(mask, self.base_fov, self.rel_fov, d)
        )

    @classmethod
    def brainweb(cls, file_name: str) -> VoxelGridPhantom:
        """Load a phantom from data produced by `generate_maps.py`."""
        with np.load(file_name) as data:
            T1 = torch.tensor(data['T1_map'])
            T2 = torch.tensor(data['T2_map'])
            T2dash = torch.tensor(data['T2dash_map'])
            PD = torch.tensor(data['PD_map'])
            D = torch.tensor(data['D_map'])

        # Generate a somewhat plausible B0 and B1 map.
        # Visually fitted to look similar to the numerical_brain_cropped
        x_pos, y_pos, z_pos = torch.meshgrid(
            torch.linspace(-1, 1, PD.shape[0]),
            torch.linspace(-1, 1, PD.shape[1]),
            torch.linspace(-1, 1, PD.shape[2]),
            indexing="ij"
        )
        B1 = torch.exp(-(0.4*x_pos**2 + 0.2*y_pos**2 + 0.3*z_pos**2))
        dist2 = (0.4*x_pos**2 + 0.2*(y_pos - 0.7)**2 + 0.3*z_pos**2)
        B0 = 7 / (0.05 + dist2) - 45 / (0.3 + dist2)
        # Normalize such that the weighted average is 0 or 1
        weight = PD / PD.sum()
        B0 -= (B0 * weight).sum()
        B1 /= (B1 * weight).sum()

        return cls(
            PD, T1, T2, T2dash, D, B0, B1[None, :],
            coil_sens=torch.ones(1, *PD.shape),
            base_fov=torch.tensor([0.192, 0.192, 0.192]),
            rel_fov=torch.ones(3)
        )

    @classmethod
    def load(cls, file_name: str) -> VoxelGridPhantom:
        # TODO: docstring (see old_sim_data) and T2dash / D parameters
        mat = loadmat(file_name)
        keys = [
            key for key in mat
            if not (key.startswith('__') and key.endswith('__'))
        ]
        arrays = [mat[key] for key in keys if isinstance(mat[key], np.ndarray)]
        assert len(keys) == 1, "The loaded mat file must contain exatly one array"

        data = torch.from_numpy(arrays[0]).float()
        assert data.ndim == 3 and data.shape[-1] == 5
        # Expand data to 3D
        data = data.unsqueeze(2)
        PD = data[..., 0]
        T1 = data[..., 1]
        T2 = data[..., 2]
        B0 = data[..., 3]
        B1 = data[..., 4]
        # TODO: Use function parameters here instead of constatns
        T2dash = torch.full_like(PD, 30e-3)
        D = torch.full_like(PD, 0.0)

        # TODO: do we know FOV? currently assuming 1.5 mm voxels

        return cls(
            PD, T1, T2, T2dash, D, B0, B1[None, :],
            coil_sens=torch.ones(1, *PD.shape),
            base_fov=torch.tensor(PD.shape) * 1.5e-3,
            rel_fov=torch.ones(3)
        )

    def slices(self, slices: list[int]) -> VoxelGridPhantom:
        """Generate a copy that only contains the selected slice(s).

        Parameters
        ----------
        slice: int or tuple
            The selected slice(s)

        Returns
        -------
        SimData
            A new instance containing the selected slice(s).
        """
        assert 0 <= any([slices]) < self.PD.shape[2]

        fov = self.rel_fov.clone()
        fov[2] *= len(slices) / self.PD.shape[2]

        def select(tensor: torch.Tensor):
            return tensor[..., slices].view(
                *list(self.PD.shape[:2]), len(slices)
            )

        return VoxelGridPhantom(
            select(self.PD),
            select(self.T1),
            select(self.T2),
            select(self.T2dash),
            select(self.D),
            select(self.B0),
            select(self.B1).unsqueeze(0),
            select(self.coil_sens).unsqueeze(0),
            self.base_fov.clone(),
            fov,
        )

    def scale_fft(self, x: int, y: int, z: int) -> VoxelGridPhantom:
        """This is experimental, shows strong ringing and is not recommended"""
        # This function currently only supports downscaling
        assert x <= self.PD.shape[0]
        assert y <= self.PD.shape[1]
        assert z <= self.PD.shape[2]

        # Normalize signal, otherwise magnitude changes with scaling
        norm = (
            (x / self.PD.shape[0]) *
            (y / self.PD.shape[1]) *
            (z / self.PD.shape[2])
        )
        # Center for FT
        cx = self.PD.shape[0] // 2
        cy = self.PD.shape[1] // 2
        cz = self.PD.shape[2] // 2

        def scale(map: torch.Tensor) -> torch.Tensor:
            FT = torch.fft.fftshift(torch.fft.fftn(map))
            FT = FT[
                cx - x // 2:cx + (x+1) // 2,
                cy - y // 2:cy + (y+1) // 2,
                cz - z // 2:cz + (z+1) // 2
            ] * norm
            return torch.fft.ifftn(torch.fft.ifftshift(FT)).abs()

        return VoxelGridPhantom(
            scale(self.PD),
            scale(self.T1),
            scale(self.T2),
            scale(self.T2dash),
            scale(self.D),
            scale(self.B0),
            scale(self.B1.squeeze()).unsqueeze(0),
            scale(self.coil_sens.squeeze()).unsqueeze(0),
            self.base_fov.clone(),
            self.rel_fov.clone(),
        )

    def interpolate(self, x: int, y: int, z: int) -> VoxelGridPhantom:
        """Return a resized copy of this :class:`SimData` instance.

        This uses torch.nn.functional.interpolate in 'area' mode, which is not
        very good: Assumes pixels are squares -> has strong aliasing.

        Use :meth:`resample_fft` instead.

        Parameters
        ----------
        x : int
            The new resolution along the 1st dimension
        y : int
            The new resolution along the 2nd dimension
        z : int
            The new resolution along the 3rd dimension
        mode : str
            Algorithm used for upsampling (via torch.nn.functional.interpolate)

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing resized tensors.
        """
        def resample(tensor: torch.Tensor):
            # Introduce additional dimensions: mini-batch and channels
            return torch.nn.functional.interpolate(
                tensor[None, None, ...], size=(x, y, z), mode='area'
            )[0, 0, ...]
        
        # Code assumes single coil, adopt (iterate) for multi-coil
        assert self.B1.shape[0] == 1 and self.coil_sens.shape[0] == 1

        return VoxelGridPhantom(
            resample(self.PD),
            resample(self.T1),
            resample(self.T2),
            resample(self.T2dash),
            resample(self.D),
            resample(self.B0),
            resample(self.B1[0, ...]).unsqueeze(0),
            resample(self.coil_sens[0, ...]).unsqueeze(0),
            self.base_fov.clone(),
            self.rel_fov.clone(),
        )

    def plot(self) -> None:
        """Print and plot all data stored in this phantom."""
        print("VoxelGridPhantom")
        print(
            f"FOV: base * rel = {self.base_fov} * {self.rel_fov} "
            f"= {self.base_fov * self.rel_fov}"
        )
        # Center slice
        s = self.PD.shape[2] // 2
        # Warn if we only print a part of all data
        if self.coil_sens.shape[0] > 1:
            print(f"Plotting 1st of {self.coil_sens.shape[0]} coil sens maps")
        if self.B1.shape[0] > 1:
            print(f"Plotting 1st of {self.B1.shape[0]} B1 maps")
        if self.PD.shape[2] > 1:
            print(f"Plotting slice {s} / {self.PD.shape[2]}")

        plt.figure(figsize=(12, 10))
        plt.subplot(331)
        plt.title("PD")
        plt.imshow(self.PD[:, :, s].T.cpu(), vmin=0, origin="lower")
        plt.colorbar()
        plt.subplot(332)
        plt.title("T1")
        plt.imshow(self.T1[:, :, s].T.cpu(), vmin=0, origin="lower")
        plt.colorbar()
        plt.subplot(333)
        plt.title("T2")
        plt.imshow(self.T2[:, :, s].T.cpu(), vmin=0, origin="lower")
        plt.colorbar()
        plt.subplot(334)
        plt.title("T2'")
        plt.imshow(self.T2dash[:, :, s].T.cpu(), vmin=0, origin="lower")
        plt.colorbar()
        plt.subplot(335)
        plt.title("D")
        plt.imshow(self.D[:, :, s].T.cpu(), vmin=0, origin="lower")
        plt.colorbar()
        plt.subplot(337)
        plt.title("B0")
        plt.imshow(self.B0[:, :, s].T.cpu(), origin="lower")
        plt.colorbar()
        plt.subplot(338)
        plt.title("B1")
        plt.imshow(self.B1[0, :, :, s].T.cpu(), vmin=0, origin="lower")
        plt.colorbar()
        plt.subplot(339)
        plt.title("coil sens")
        plt.imshow(self.coil_sens[0, :, :, s].T.cpu(), vmin=0, origin="lower")
        plt.colorbar()
        plt.show()


def recover(mask, base_fov, rel_fov, sim_data: RawSimData) -> VoxelGridPhantom:
    """Provided to :class:`RawSimData` to reverse the ``build()``"""
    def to_full(sparse):
        assert sparse.ndim < 3
        if sparse.ndim == 2:
            full = torch.zeros([sparse.shape[0], *mask.shape])
            full[:, mask] = sparse.cpu()
        else:
            full = torch.zeros(mask.shape)
            full[mask] = sparse.cpu()
        return full

    return VoxelGridPhantom(
        to_full(sim_data.PD),
        to_full(sim_data.T1),
        to_full(sim_data.T2),
        to_full(sim_data.T2dash),
        to_full(sim_data.D),
        to_full(sim_data.B0),
        to_full(sim_data.B1),
        to_full(sim_data.coil_sens),
        base_fov,
        rel_fov
    )
