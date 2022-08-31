"""Data needed for simulation."""

from __future__ import annotations
import torch
from torch.nn.functional import interpolate
import numpy as np
from scipy import io
import matplotlib.pyplot as plt
from typing import Union

from . import util


class SimData:
    """This class contains physical data needed for simulating a MRT sequence.

    The simulation data is always 3D, 2D data only contains a single slice.

    The tensors in this class are sparse, which means they only contain voxels
    with a high enough proton density, given as argument in :meth:`__init__`.

    Attributes
    ----------
    PD_threshold : float
        The threshold below which voxels are omitted from the sparse tensors
    PD : torch.Tensor
        Per voxel proton density
    T1 : torch.Tensor
        Per voxel T1 relaxation time (seconds)
    T2 : torch.Tensor
        Per voxel T2 relaxation time (seconds)
    T2dash : torch.Tensor
        Per voxel T2' dephasing time (seconds)
    B0 : torch.Tensor
        Per voxel B0 inhomogentity (Hertz)
    B1 : torch.Tensor
        Per coil and per voxel B1 inhomogenity (normalized amplitude)
    coil_sens : torch.Tensor
        Per coil sensitivity (arbitrary units)
    coil_count : int
        Number of coils
    mask : torch.Tensor
        Bool tensor specifiying which voxels are stored in the other tensors
    voxel_count : int
        Number of voxels set to true in :attr:`mask`
    shape : torch.Tensor
        shape of :attr:`mask` and all full tensors
        (only :attr:`B1` has an additional first dimension for coils)
    voxel_pos : torch.Tensor
        3-dimensional positions of the voxels in the sparse tensors
    """

    def __init__(
        self,
        PD: torch.Tensor,
        T1: torch.Tensor,
        T2: torch.Tensor,
        T2dash: torch.Tensor,
        B0: torch.Tensor,
        B1: torch.Tensor,
        coil_sens: torch.Tensor,
        PD_threshold: float = 1e-6,
        normalize_B0_B1: bool = True
    ) -> None:
        """Create a SimData instance based on the given tensors.

        The attributes of the created instance are sparse tensors, containing
        only voxels for which ``PD`` is larger than ``PD_threshold``.

        If normalize_B0_B1 = True, applies B0 -= B0.mean() and B1 /= B1.mean().
        """
        if not (PD.shape == T1.shape == T2.shape == T2dash.shape == B0.shape):
            raise Exception("Mismatch of voxel-data shapes")

        if B1.ndim < 2 or B1.shape[1:] != PD.shape:
            raise Exception("B1 must have shape [coils, *data.shape]")
        if coil_sens.ndim < 2 or coil_sens.shape[1:] != PD.shape:
            raise Exception("coil_sens must have shape [coils, *data.shape]")

        self.PD_threshold = PD_threshold
        self.mask = PD > PD_threshold
        self.voxel_count = int(torch.count_nonzero(self.mask))
        # TODO: maybe change shape to tuple of ints
        self.shape = util.set_device(torch.Tensor(list(PD.shape)))

        self.PD = util.set_device(PD[self.mask].clamp(min=0))
        self.T1 = util.set_device(T1[self.mask].clamp(min=1e-12))
        self.T2 = util.set_device(T2[self.mask].clamp(min=1e-12))
        self.T2dash = util.set_device(T2dash[self.mask])
        self.B0 = util.set_device(B0[self.mask])
        # TODO: *must* remove outliers - find a better way to detect them
        self.B1 = util.set_device(B1[:, self.mask].clamp(0.8, 1.2))
        self.coil_sens = util.set_device(coil_sens[:, self.mask])
        self.coil_count = int(coil_sens.shape[0])

        if normalize_B0_B1:
            self.B1 /= self.B1.mean()
            self.B0 -= self.B0.mean()

        # Normalized voxel positions, together with the gradient definition
        # a linear cartesian readout will match the definition of a DFT
        pos_x = torch.linspace(-0.5, 0.5, int(self.shape[0]) + 1)[:-1]
        pos_y = torch.linspace(-0.5, 0.5, int(self.shape[1]) + 1)[:-1]
        pos_z = torch.linspace(-0.5, 0.5, int(self.shape[2]) + 1)[:-1]
        pos_x, pos_y, pos_z = torch.meshgrid(pos_x, pos_y, pos_z)

        self.voxel_pos = util.set_device(torch.stack([
            pos_x[self.mask].flatten(),
            pos_y[self.mask].flatten(),
            pos_z[self.mask].flatten()
        ], dim=1))

    def select_slice(self, slice: int) -> SimData:
        """Generate a copy that only contains the selected slice(s).

        Parameters
        ----------
        slice: int or tuple
            The selected slice(s)

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing the selected slice(s).
        """
        assert 0 <= any([slice]) < self.shape[2]
        
        def select(tensor: torch.Tensor):
            full = util.to_full(tensor, self.mask)
            return full[..., slice].view(*self.shape[:2].int().tolist(), torch.numel(torch.tensor([slice])))

        return SimData(
            select(self.PD),
            select(self.T1),
            select(self.T2),
            select(self.T2dash),
            select(self.B0),
            select(self.B1).unsqueeze(0),
            select(self.coil_sens).unsqueeze(0),
            self.PD_threshold,
            False
        )

    def resize(self, *new_size: float, mode='area') -> SimData:
        """Return a resized copy of this :class:`SimData` instance.

        This function uses "linear" interpolation which might wrongly blend
        tissues together (ignores partial volume effect).

        Parameters
        ----------
        new_size : torch.Size
            The size of the resized copy. Has to have the same number of
            dimensions as :attr:`shape`.

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing resized tensors.
        """

        def resample(tensor: torch.Tensor):
            full = util.to_full(tensor, self.mask)
            # Functions expects batch x channels x (depth) x height x width
            if full.shape[-1] == 1:  # 2D, possible modes: 'area', 'bicubic'
                full = full.squeeze().unsqueeze(0).unsqueeze(0)
                full = interpolate(full, size=new_size[:2], mode=mode)
            else:  # 3D, possible modes: 'area', 'trilinear'
                full = full.unsqueeze(0).unsqueeze(0)
                full = interpolate(full, size=new_size, mode=mode)
            return full.view(new_size)

        return SimData(
            resample(self.PD),
            resample(self.T1),
            resample(self.T2),
            resample(self.T2dash),
            resample(self.B0),
            resample(self.B1).unsqueeze(0),
            resample(self.coil_sens).unsqueeze(0),
            self.PD_threshold,
            False
        )

    def zero_pad(self, pad) -> SimData:

        def padding(tensor: torch.Tensor,padding):
            full = util.to_full(tensor, self.mask)
            m = torch.nn.ZeroPad2d(pad)
            # Functions expects batch x channels x (depth) x height x width
            if full.shape[-1] == 1:  # 2D, possible modes: 'area', 'bicubic'
                full = m(full.squeeze().unsqueeze(0).unsqueeze(0))
            else:  # 3D, possible modes: 'area', 'trilinear'
                pass
            return full.squeeze().unsqueeze(2)

        return SimData(
            padding(self.PD,pad),
            padding(self.T1,pad),
            padding(self.T2,pad),
            padding(self.T2dash,pad),
            padding(self.B0,pad),
            padding(self.B1,pad).unsqueeze(0),
            padding(self.coil_sens,pad).unsqueeze(0),
            self.PD_threshold,
            False
        )

    def set_coil_sens(self, coil_sens: torch.Tensor):
        """Set the coil sensitivity sensor of self."""
        self.coil_sens = util.set_device(coil_sens[:, self.mask])
        self.coil_count = coil_sens.shape[0]

    def get_coil_sens(self) -> torch.Tensor:
        """Return the coil sensitivity sensor of self."""
        return util.to_full(self.coil_sens, self.mask)

    def get_phantom_tensor(self) -> tuple[torch.Tensor, ...]:
        """Return a tuple of all tensors in self."""
        return (
            util.to_full(self.PD, self.mask),
            util.to_full(self.T1, self.mask),
            util.to_full(self.T2, self.mask),
            util.to_full(self.T2dash, self.mask),
            util.to_full(self.B0, self.mask),
            util.to_full(self.B1, self.mask).unsqueeze(0),
            util.to_full(self.coil_sens, self.mask).unsqueeze(0)
        )

    @classmethod
    def load_npz(cls, file_name: str, normalize_B0_B1: bool = True) -> SimData:
        """Generate a :class:`SimData` from data loaded from a .npz file.

        The file must contain arrays named 'T1_map', 'T2_map', 'T2dash_map' and
        'PD_map'. This matches the files produced by brainweb/generate_maps.py.

        Parameters
        ----------
        file_name : str
            Name of the .npz numpy archive file

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing the loaded data.

        Raises
        ------
        KeyError
            The specified file is missing one of the required maps
        """
        with np.load(file_name) as data:
            # TODO: This is a 6mm thick slice, change to full volume for 3D
            # volumes or if slice selective pulses are implemented
            T1_map = data['T1_map']
            T2_map = data['T2_map']
            T2dash_map = data['T2dash_map']
            PD_map = data['PD_map']

        return cls(
            torch.as_tensor(PD_map),
            torch.as_tensor(T1_map),
            torch.as_tensor(T2_map),
            torch.as_tensor(T2dash_map),
            torch.zeros(*PD_map.shape),  # B0
            torch.ones(1, *PD_map.shape),  # B1
            torch.ones(1, *PD_map.shape),  # coil sensitivity
            1e-6,
            normalize_B0_B1
        )

    @classmethod
    def load(
        cls,
        file_name: str,
        T2dash: Union[float, torch.Tensor] = 0.03,
        PD_threshold: float = 1e-6,
        normalize_B0_B1: bool = True
    ) -> SimData:
        """Create a :class:`SimData` from data loaded from a .mat file.

        The file must contain (at least) one array of which the last dimension
        must have size 5. This dimension is assumed to specify (in that order):

        * Proton density
        * T1
        * T2
        * B0
        * B1

        All data is per-voxel, multiple coils are not yet supported.
        Data will be normalized (see constructor).

        Parameters
        ----------
        file_name : str
            Name of the matlab .mat file to be loaded
        T2dash : float, optional
            T2dash value set uniformly for all voxels, by default 0.03

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing the loaded data.

        Raises
        ------
        Exception
            The loaded file does not contain the expected data.
        """
        data = _load_tensor_from_mat(file_name)

        if data.ndim < 2 or data.shape[-1] != 5:
            raise Exception(
                f"Expected a tensor with shape [..., 5], "
                f"but got {list(data.shape)}"
            )

        # TODO: Assumes 2D data, expands it to 3D
        data = data.unsqueeze(2)

        # TODO: Better handling of data not included in .mat

        if isinstance(T2dash, float):
            T2dash = torch.full_like(data[..., 0], T2dash)

        return cls(
            data[..., 0],  # PD
            data[..., 1],  # T1
            data[..., 2],  # T2
            T2dash,
            data[..., 3],  # B0
            data[..., 4].unsqueeze(0),  # B1
            torch.ones(1, *data.shape[:-1]),  # coil sensitivity
            PD_threshold,
            normalize_B0_B1
        )

    @classmethod
    def generate(
        cls,
        size: tuple[float, float],
        PD_threshold: float = 1e-6,
        normalize_B0_B1: bool = True
    ) -> SimData:
        """Create a :class:`SimData` from generated data with random size at
        random position.

        The file must contain (at least) one array of which the last dimension
        must have size 5. This dimension is assumed to specify (in that order):

        * Proton density
        * T1
        * T2
        * B0
        * B1

        All data is per-voxel, multiple coils are not yet supported.
        Data will be normalized (see constructor).

        Parameters
        ----------
        size : int
            Size of matrix dimension

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing the loaded data.

        Raises
        ------
        Exception
            The loaded file does not contain the expected data.
        """
        
        # Create Trainings data
        csz = torch.randint(8,max(size),(1,))
        nmb_samples = 1     # Number of samples
        param = torch.zeros((size[0], size[1], 1, 6), dtype=torch.float32)
        data = []
        
        for i in range(nmb_samples):
            rvx = (torch.floor(torch.rand(1) * (size[0] - csz))).int()
            rvy = (torch.floor(torch.rand(1) * (size[0] - csz))).int()
            
            pdbase = torch.rand(1)
            pd = pdbase + torch.rand((csz,csz))*0.5
            t2 = 0.04 + 3*torch.rand((csz,csz))**1.5
            t1 = 0.3+torch.abs(1.3 + 1.5*torch.randn((csz,csz)))
            b0 = (torch.rand((csz,csz)) - 0.5) * 80
            b1 = 0.75+0.5*torch.rand((csz,csz))
            t2dash = 0.005 + torch.rand((csz,csz))*0.04
            
            param[rvx:rvx+csz,rvy:rvy+csz,0,0] = pd
            param[rvx:rvx+csz,rvy:rvy+csz,0,1] = t1
            param[rvx:rvx+csz,rvy:rvy+csz,0,2] = t2
            param[rvx:rvx+csz,rvy:rvy+csz,0,3] = t2dash
            param[rvx:rvx+csz,rvy:rvy+csz,0,4] = b0
            param[rvx:rvx+csz,rvy:rvy+csz,0,5] = b1
            
            data.append(cls(
            param[..., 0],
            param[..., 1],
            param[..., 2],
            param[..., 3],
            param[..., 4],
            param[..., 5].unsqueeze(0),
            torch.ones(1, *param.shape[:-1]),  # coil sensitivity
            PD_threshold,
            normalize_B0_B1
            ))
        
            
        return data
    
    @classmethod
    def generate_pixel_phantom(
        cls,
        size: tuple[float, float],
        PD_threshold: float = 1e-6,
        normalize_B0_B1: bool = True
    ) -> SimData:
        """Create a :class:`SimData` from generated data with random size at
        random position.

        The file must contain (at least) one array of which the last dimension
        must have size 5. This dimension is assumed to specify (in that order):

        * Proton density
        * T1
        * T2
        * B0
        * B1

        All data is per-voxel, multiple coils are not yet supported.
        Data will be normalized (see constructor).

        Parameters
        ----------
        size : int
            Size of matrix dimension

        Returns
        -------
        SimData
            A new :class:`SimData` instance containing the loaded data.

        Raises
        ------
        Exception
            The loaded file does not contain the expected data.
        """
        
        # Create Trainings data
        
        
        param = torch.zeros((size[0], size[1], 1, 6), dtype=torch.float32)
        
        # POS = [[30,30],[32,30],[34,30],[36,30],[38,30],
        #                 [30,50],[30,70],[30,90],[30,110]]
        POS = [[30,30],[30,50],[30,60],[30,65],[30,70],[30,74],[30,78],[30,80],
               [30,82],[30,84],[30,86],[30,88],[30,90]]        
        
        values = [[0.7,0.05],[0.7,0.1],[0.7,0.5],[0.7,1],[0.7,2]]
        # POS = [[30,30],[50,30],[70,30],[90,30],[110,30],
        #        [30,50],[30,70],[30,90],[30,110]]
        
        YY = [10,30,50,60,70,
              78,86,91,96,100,
              104,107,110,112,114]
        
        values = [[1.0,0.05],[1.0,0.07],[0.9,0.09],[0.9,0.11],[0.8,0.2],
                  [0.8,0.3],[0.7,0.5],[0.7,0.7],[0.7,1],[0.6,1.2],
                  [0.6,1.4],[0.5,1.6],[0.5,1.8],[0.5,2.0],[0.5,2.5]]
        
        
            
        
        for i in range(len(values)):
            for XX in range(28,100,4):
                param[XX,YY[i],0,0] = values[i][0] + np.random.rand()/5
                param[XX,YY[i],0,1] = 1
                param[XX,YY[i],0,2] = values[i][1] + np.random.rand()/20
                param[XX,YY[i],0,3] = 0.0362
                param[XX,YY[i],0,4] = 0
                param[XX,YY[i],0,5] = 1
            
        # POS = [[30,30],[32,30],[34,30],[36,30],[38,30],
        #                 [30,50],[30,70],[30,90],[30,110]]
        
        # # POS = [[30,30],[50,30],[70,30],[90,30],[110,30],
        # #        [30,50],[30,70],[30,90],[30,110]]
        
        
        # param[POS[0][0],POS[0][1],0,0] = 2
        # param[POS[0][0],POS[0][1],0,1] = 1
        # param[POS[0][0],POS[0][1],0,2] = 0.05
        # param[POS[0][0],POS[0][1],0,3] = 0.0362
        # param[POS[0][0],POS[0][1],0,4] = 0
        # param[POS[0][0],POS[0][1],0,5] = 1
        
        # param[POS[1][0],POS[1][1],0,0] = 1
        # param[POS[1][0],POS[1][1],0,1] = 1
        # param[POS[1][0],POS[1][1],0,2] = 0.1
        # param[POS[1][0],POS[1][1],0,3] = 0.0362
        # param[POS[1][0],POS[1][1],0,4] = 0
        # param[POS[1][0],POS[1][1],0,5] = 1
        
        # param[POS[2][0],POS[2][1],0,0] = 0.35
        # param[POS[2][0],POS[2][1],0,1] = 1
        # param[POS[2][0],POS[2][1],0,2] = 0.5
        # param[POS[2][0],POS[2][1],0,3] = 0.0362
        # param[POS[2][0],POS[2][1],0,4] = 0
        # param[POS[2][0],POS[2][1],0,5] = 1        

        # param[POS[3][0],POS[3][1],0,0] = 0.27
        # param[POS[3][0],POS[3][1],0,1] = 1
        # param[POS[3][0],POS[3][1],0,2] = 1
        # param[POS[3][0],POS[3][1],0,3] = 0.0362
        # param[POS[3][0],POS[3][1],0,4] = 0
        # param[POS[3][0],POS[3][1],0,5] = 1
        
        # param[POS[4][0],POS[4][1],0,0] = 0.2
        # param[POS[4][0],POS[4][1],0,1] = 1
        # param[POS[4][0],POS[4][1],0,2] = 100
        # param[POS[4][0],POS[4][1],0,3] = 0.0362
        # param[POS[4][0],POS[4][1],0,4] = 0
        # param[POS[4][0],POS[4][1],0,5] = 1
        
        # param[POS[5][0],POS[5][1],0,0] = 1
        # param[POS[5][0],POS[5][1],0,1] = 1
        # param[POS[5][0],POS[5][1],0,2] = 0.1
        # param[POS[5][0],POS[5][1],0,3] = 0.0362
        # param[POS[5][0],POS[5][1],0,4] = 0
        # param[POS[5][0],POS[5][1],0,5] = 1      

        # param[POS[6][0],POS[6][1],0,0] = 0.35
        # param[POS[6][0],POS[6][1],0,1] = 1
        # param[POS[6][0],POS[6][1],0,2] = 0.5
        # param[POS[6][0],POS[6][1],0,3] = 0.0362
        # param[POS[6][0],POS[6][1],0,4] = 0
        # param[POS[6][0],POS[6][1],0,5] = 1
        
        # param[POS[7][0],POS[7][1],0,0] = 0.27
        # param[POS[7][0],POS[7][1],0,1] = 1
        # param[POS[7][0],POS[7][1],0,2] = 1
        # param[POS[7][0],POS[7][1],0,3] = 0.0362
        # param[POS[7][0],POS[7][1],0,4] = 0
        # param[POS[7][0],POS[7][1],0,5] = 1
        
        # param[POS[8][0],POS[8][1],0,0] = 0.2
        # param[POS[8][0],POS[8][1],0,1] = 1
        # param[POS[8][0],POS[8][1],0,2] = 100
        # param[POS[8][0],POS[8][1],0,3] = 0.0362
        # param[POS[8][0],POS[8][1],0,4] = 0
        # param[POS[8][0],POS[8][1],0,5] = 1           
        
        data = cls(
        param[..., 0],
        param[..., 1],
        param[..., 2],
        param[..., 3],
        param[..., 4],
        param[..., 5].unsqueeze(0),
        torch.ones(1, *param.shape[:-1]),  # coil sensitivity
        PD_threshold,
        normalize_B0_B1
        )
        
            
        return data
    
  
    def plot_sim_data(sim_data: SimData,
                      figsize: tuple[float, float] = (6, 4)) -> None:
        """Plot all tensors of ``sim_data`` by using Matplotlib.
    
        Parameters
        ----------
        sim_data : SimData
            The :class:`SimData` instance to be plotted.
        """
        titles = ["PD", "T1 [s]", "T2 [s]", "T2' [s]", "B0 [Hz]", "B1 [rel]"]
        tensors = [
            sim_data.PD, sim_data.T1, sim_data.T2,
            sim_data.T2dash, sim_data.B0, sim_data.B1.squeeze(0)
        ]
    
        plt.figure(3,figsize=figsize)
        for i in range(6):
            # TODO: This plots only the first slice of 3D data
            data = util.to_numpy(util.to_full(
                tensors[i], sim_data.mask).squeeze(0))
            plt.subplot(231+i)
            plt.title(titles[i])
            ax = plt.imshow(data[:, :, data.shape[2]//2], interpolation='none')
            plt.gcf().colorbar(ax)
        plt.show()
        
    def plot_sim_data_3D(sim_data: SimData, prop: int = 0,
                      figsize: tuple[float, float] = (6, 4)) -> None:
        """Plot all slices of ``sim_data`` by using Matplotlib.
    
        Parameters
        ----------
        sim_data : SimData
            The :class:`SimData` instance to be plotted.
        prop : int
            The value gives the property which should be plotted.
        """
        titles = ["PD", "T1 [s]", "T2 [s]", "T2' [s]", "B0 [Hz]", "B1 [rel]"]
        tensors = [
            sim_data.PD, sim_data.T1, sim_data.T2,
            sim_data.T2dash, sim_data.B0, sim_data.B1.squeeze(0)
        ]
        
        tensor = tensors[prop]
        tensor = util.to_full(tensor, sim_data.mask).squeeze(0)
        
        util.plot3D(tensor)
        plt.title(titles[prop])
        plt.show()
        
def _load_tensor_from_mat(file_name: str) -> torch.Tensor:
    mat = io.loadmat(file_name)

    keys = [
        key for key in mat
        if not (key.startswith('__') and key.endswith('__'))
    ]

    arrays = [mat[key] for key in keys if isinstance(mat[key], np.ndarray)]

    if len(keys) == 0:
        raise Exception("The loaded mat file does not contain any variables")

    if len(arrays) != 1:
        raise Exception("The loaded mat file must contain exactly one array")

    return torch.from_numpy(arrays[0]).float()
