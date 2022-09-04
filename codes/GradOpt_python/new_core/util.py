"""This module contains helper functions only."""

from __future__ import annotations
import os
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import io
from . import sequence


use_gpu = False
gpu_dev = 0


def get_device() -> torch.device:
    """Return the device as given by ``util.use_gpu`` and ``util.gpu_dev``."""
    if use_gpu:
        return torch.device(f"cuda:{gpu_dev}")
    else:
        return torch.device("cpu")


def set_device(x: torch.Tensor) -> torch.Tensor:
    """Set the device of the passed tensor as given by :func:`get_deivce`."""
    if use_gpu:
        return x.cuda(gpu_dev)
    else:
        return x.cpu()


def phase_cycler(pulse: int, dphi: float = 137.50776405) -> float:
    """Generate a phase for cycling through phases in a sequence.

    The default value of 360° / Golden Ratio seems to work well, better than
    angles like 117° which produces very similar phases for every 3rd value.

    Parameters
    ----------
        pulse : int
            pulse number for which the phase is calculated
        dphi : float
            phase step size in degrees

    Returns
    -------
        Phase of the given pulse
    """
    return float(np.fmod(pulse * dphi, 360) * np.pi / 180)


def current_fig_as_img(dpi: float = 180) -> np.ndarray:
    """Return the current matplotlib figure as image.

    Parameters
    ----------
    dpi : float
        The resolution of the returned image

    Returns
    -------
    np.ndarray
        The current matplotlib figure converted to a 8 bit rgb image.
    """
    buf = io.BytesIO()
    plt.gcf().savefig(buf, format="png", dpi=dpi)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.uint8)


def to_full(sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Convert a sparse to a full tensor by filling indices given by mask.

    Parameters
    ----------
    sparse : torch.Tensor)
        Sparse tensor containing the data.
    mask : torch.Tensor)
        Mask indicating the indices of the elements in ``sparse``

    Raises
    ------
    ValueError
        If ``mask`` requires more or less elements than ``sparse`` contains.

    Returns
    -------
    torch.Tensor
        The full tensor that has the same shape as ``mask`` and contains the
        data of ``sparse``.
    """
    if mask.count_nonzero() != sparse.shape[-1]:
        raise ValueError(
            f"mask requires {mask.count_nonzero()} elements, "
            f"but sparse contains {sparse.shape[-1]}."
        )
    # TODO: Explanation why this is needed, shouldn't sparse always be 1D?
    if sparse.squeeze().dim() > 1:
        full = torch.zeros(sparse.shape[:-1] + mask.shape,
                           dtype=sparse.dtype, device=sparse.device)
        full[..., mask] = sparse
    else:
        full = torch.zeros(mask.shape,
                           dtype=sparse.dtype, device=sparse.device)
        full[mask] = sparse
    return full


def to_numpy(x: torch.Tensor) -> np.ndarray:
    """Convert a torch tensor to a numpy ndarray."""
    return x.detach().cpu().numpy()


def to_torch(x: np.ndarray) -> torch.Tensor:
    """Convert a numpy ndarray to a torch tensor."""
    return torch.tensor(x, dtype=torch.float)

def plot3D(x: torch.Tensor,figsize=(12,2)) -> None:
    """Plot absolute image of a 3D tensor (x,y,z)
    or 4D tensor (coil,x,y,z)."""
    if x.ndim == 4:
        x = torch.sqrt(torch.sum(x**2,0))
    plt.figure(figsize=figsize)
    plt.imshow(to_numpy(x).transpose(0,2,1).reshape(x.shape[0],x.shape[1]*x.shape[2]))
    plt.colorbar()

def SSIM(a: torch.Tensor, b: torch.Tensor,
         window_size: float = 4.0) -> torch.Tensor:
    """Calculate the structural similarity of two 2D tensors.

    Structural similarity is a metric that tries to estimate how similar two
    images look for humans. The calculated value is per-pixel and describes how
    different or similar that particular pixel looks. While doing so it takes
    the neighbourhood into account, as given by the ``window_size``.

    Parameters
    ----------
    a : torch.Tensor
        A 2D, real valued tensor
    b : torch.Tensor
        A tensor with identical properties as ``a``
    window_size : float
        The window size used when comparing ``a`` and ``b``

    Returns
    -------
    torch.Tensor
        A tensor with the same shape as ``a`` and ``b``, containing for every
        pixel a value between 0 (no similarity) to 1 (identical).
    """
    assert a.shape == b.shape and a.dim() == 2

    x, y = torch.meshgrid([torch.arange(a.shape[0]), torch.arange(a.shape[1])])
    norm = 1 / (2*np.pi*np.sqrt(window_size))

    def gauss(x0: float, y0: float):
        return norm * torch.exp(-((x-x0)**2 + (y-y0)**2) / (2*window_size))

    ssim = torch.zeros_like(a)
    c1 = 1e-4
    c2 = 9e-4

    for x0 in range(a.shape[0]):
        for y0 in range(a.shape[1]):
            window = gauss(x0, y0)
            a_w = a * window
            b_w = b * window

            a_mean = a_w.mean()
            b_mean = b_w.mean()
            a_diff = a_w - a_mean
            b_diff = b_w - b_mean

            ssim[x0, y0] = (
                (
                    (2*a_mean*b_mean + c1)
                    * (2*(a_diff*b_diff).mean() + c2)
                ) / (
                    (a_mean**2 + b_mean**2 + c1)
                    * ((a_diff**2).mean() + (b_diff**2).mean() + c2)
                )
            )

    return ssim


def load_optimizer(optimizer: torch.optim.Optimizer,
                   path: torch.Tensor,
                   NN: torch.nn.Module | None = None
                   ) -> tuple[torch.optim.Optimizer, torch.Tensor,
                              torch.Tensor, torch.Tensor,
                              torch.nn.Module | None]:
    """Load state of optimizer for retraining/restarts

    Parameters
    ----------
    optimizer : torch.optim
        A optimizer
    path : torch.Tensor
        A tensor with the path to the file which sould be loaded

    Returns
    -------
    optimizer : torch.optim
        Optimizer with loaded parameters.
    loss_history : torch.Tensor
        Old loss_history.
    params_target : torch.Tensor
        Sequence parameters for target.
    target_reco : torch.Tensor
        Target reconstruction
    """
    checkin = torch.load(path)
    optimizer.load_state_dict(checkin['optimizer'])
    optimizer.param_groups = checkin['optimizer_params']
    if NN:
        NN.load_state_dict(checkin['NN'])

    return (
        optimizer,
        checkin['loss_history'],
        checkin['params_target'],
        checkin['target_reco'],
        NN
    )


def L1(a: torch.Tensor, b: torch.Tensor,
       absolut: bool = False) -> torch.Tensor:
    """Calculate the L1 norm of two 2D tensors.

    Parameters
    ----------
    a : torch.Tensor
        A 2D, real or imaginar valued tensor
    b : torch.Tensor
        A tensor with identical properties as ``a``
    absolut : bool
        The flag ``absolut`` indicates if the abs() of ``a`` and ``b`` size is
        taken before calculating the L1 norm.

    Returns
    -------
    torch.Tensor
        A tensor with the L1 norm.
    """
    assert a.shape == b.shape

    if absolut:
        norm = torch.sum(torch.abs(torch.abs(a)-torch.abs(b)))
    else:
        norm = torch.sum(torch.abs(a-b))
    return norm


def MSR(a: torch.Tensor, b: torch.Tensor,
        root: bool = False, weighting: torch.Tensor | float = 1,
        norm: bool = False) -> torch.Tensor:
    """Calculate the (R)MSR norm of two 2D tensors.

    Parameters
    ----------
    a : torch.Tensor
        A 2D, real or imaginar valued tensor
    b : torch.Tensor
        A tensor with identical properties as ``a``
    root : torch.bool
        The flag ``root indicates if the square root of the RMS is used.
    weighting : torch.Tensor
        Give a weighting on a and b
    norm : torch.bool
        Gives the normalized MSR on b

    Returns
    -------
    torch.Tensor
        A tensor with the (R)MSE norm.
    """
    assert a.shape == b.shape

    tmp = torch.abs(a*weighting - b*weighting)
    tmp = tmp**2

    tmp = torch.sum(tmp)
    if root:
        tmp = torch.sqrt(tmp)
        
    if norm:
        tmp /= torch.sum(torch.abs(b*weighting))

    return tmp


def plot_kspace_trajectory(seq: sequence.Sequence,
                           figsize: tuple[float, float] = (5, 5),
                           plotting_dims: str = 'xy',
                           plot_timeline: bool = True) -> None:
    """Plot the kspace trajectory produced by self.

    Parameters
    ----------
    kspace : list[Tensor]
        The kspace as produced by ``Sequence.get_full_kspace()``
    figsize : (float, float), optional
        The size of the plotted matplotlib figure.
    plotting_dims : string, optional
        String defining what is plotted on the x and y axis ('xy' 'zy' ...)
    plot_timeline : bool, optional
        Plot a second subfigure with the gradient components per-event.
    """
    assert len(plotting_dims) == 2
    assert plotting_dims[0] in ['x', 'y', 'z']
    assert plotting_dims[1] in ['x', 'y', 'z']
    dim_map = {'x': 0, 'y': 1, 'z': 2}

    # TODO: We could (optionally) plot which contrast a sample belongs to,
    # currently we only plot if it is measured or not

    kspace = seq.get_full_kspace()
    adc_mask = [rep.adc_usage > 0 for rep in seq]

    cmap = plt.cm.get_cmap('rainbow')
    plt.figure(10,figsize=figsize)
    if plot_timeline:
        plt.subplot(211)
    for i, (rep_traj, mask) in enumerate(zip(kspace, adc_mask)):
        kx = to_numpy(rep_traj[:, dim_map[plotting_dims[0]]])
        ky = to_numpy(rep_traj[:, dim_map[plotting_dims[1]]])
        measured = to_numpy(mask)

        plt.plot(kx, ky, c=cmap(i / len(kspace)))
        plt.plot(kx[measured], ky[measured], 'r.')
        plt.plot(kx[~measured], ky[~measured], 'k.')
    plt.xlabel(f"$k_{plotting_dims[0]}$")
    plt.ylabel(f"$k_{plotting_dims[1]}$")
    plt.title('k-space trajectory')
    plt.grid()

    if plot_timeline:
        plt.subplot(212)
        event = 0
        for i, rep_traj in enumerate(kspace):
            x = np.arange(event, event + rep_traj.shape[0], 1)
            event += rep_traj.shape[0]
            rep_traj = to_numpy(rep_traj)

            if i == 0:
                plt.plot(x, rep_traj[:, 0], c='r', label="$k_x$")
                plt.plot(x, rep_traj[:, 1], c='g', label="$k_y$")
                plt.plot(x, rep_traj[:, 2], c='b', label="$k_z$")
            else:
                plt.plot(x, rep_traj[:, 0], c='r', label="_")
                plt.plot(x, rep_traj[:, 1], c='g', label="_")
                plt.plot(x, rep_traj[:, 2], c='b', label="_")
        plt.xlabel("Event")
        plt.ylabel("Gradient Moment")
        plt.legend()
        plt.grid()

    plt.show()


# TODO: This is specific to GRE-like sequences, make it more general!
def get_signal_from_real_system(path, NRep, NCol):
    print('waiting for TWIX file from the scanner... ' + path)
    done_flag = False
    while not done_flag:    
        if os.path.isfile(path):
            # read twix file
            print("TWIX file arrived. Reading....")

            ncoils = 20
            time.sleep(0.2)
            raw = np.loadtxt(path)

            heuristic_shift = 4
            print("raw size: {} ".format(raw.size) + "expected size: {} ".format("raw size: {} ".format(NRep*ncoils*(NCol+heuristic_shift)*2)) )

            if raw.size != NRep*ncoils*(NCol+heuristic_shift)*2:
                  print("get_signal_from_real_system: SERIOUS ERROR, TWIX dimensions corrupt, returning zero array..")
                  raw = np.zeros((NRep,ncoils,NCol+heuristic_shift,2))
                  raw = raw[:,:,:NCol,0] + 1j*raw[:,:,:NCol,1]
            else:
                  raw = raw.reshape([NRep,ncoils,NCol+heuristic_shift,2])
                  raw = raw[:,:,:NCol,0] + 1j*raw[:,:,:NCol,1]

            # raw = raw.transpose([1,2,0]) #ncoils,NRep,NCol
            raw = raw.transpose([0,2,1]) #NRep,NCol,NCoils
            raw = raw.reshape([NRep*NCol,ncoils])
            raw = np.copy(raw)
            done_flag = True

    return torch.tensor(raw,dtype=torch.complex64)
