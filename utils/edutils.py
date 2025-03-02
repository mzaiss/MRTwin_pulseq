#@title re-animate with nufft
import numpy as np
from IPython.display import HTML
from matplotlib.gridspec import GridSpec
from tqdm.auto import tqdm
import matplotlib.animation as animation
from scipy.interpolate import griddata
import torchkbnufft as tkbn
import torch

import MRzeroCore as mr0
import pypulseq as pp
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def get_kspace_trajectory(seq0, time_range: tuple[float, float] | None = None):
    """Get the kspace trajectory produced by seq0 within an optional time range.

    Parameters
    ----------
    seq0 : Sequence
        Sequence object that provides get_full_kspace() method and iterable repetitions
    time_range : tuple[float, float] | None, optional
        Time range (start, end) in seconds to filter. If None, returns full trajectory. Default: None

    Returns
    -------
    tuple
        (k, k_adc, mask_adc): Full k-space trajectory, ADC-sampled k-space, and ADC mask
    """
    if time_range is not None:
        assert len(time_range) == 2 and time_range[0] <= time_range[1], \
            "time_range must be a tuple (start, end) with start <= end"

    kspace = seq0.get_full_kspace()
    adc_mask = [rep.adc_usage > 0 for rep in seq0]
    k_list = []
    k_adc_list = []
    mask_adc_list = []
    time_offset = 0.0

    for i, (rep_traj, mask) in enumerate(zip(kspace, adc_mask)):
        times = torch.cat([torch.tensor([0.0], device=seq0[i].device), 
                          torch.cumsum(seq0[i].event_time, dim=0)]) + time_offset

        if time_range is not None:
            mask_time = (times >= time_range[0]) & (times <= time_range[1])
            if not mask_time.any():
                time_offset += seq0[i].event_time.sum()
                continue
            rep_traj = rep_traj[mask_time[:-1]]
            mask = mask[mask_time[:-1]]
            if len(rep_traj) == 0:
                time_offset += seq0[i].event_time.sum()
                continue

        k_list.append(rep_traj)
        k_adc_list.append(rep_traj[mask])
        mask_adc_list.append(mask)
        time_offset += seq0[i].event_time.sum()

    k = torch.cat(k_list, dim=0) if k_list else torch.tensor([], device=seq0[0].device)
    k_adc = torch.cat(k_adc_list, dim=0) if k_adc_list else torch.tensor([], device=seq0[0].device)
    mask_adc = torch.cat(mask_adc_list) if mask_adc_list else torch.tensor([], dtype=torch.bool, device=seq0[0].device)

    return k, k_adc, mask_adc

def recon_nufft(signal, kspace_loc, verbose=0, Nread=0, fov=0):
    img_shape = [Nread] * 2
    traj = kspace_loc[:, :2].T
    traj = traj / (Nread/fov) * np.pi * 2

    dcf = tkbn.calc_density_compensation_function(ktraj=traj, im_size=img_shape)
    kdat = signal.squeeze().reshape(1, 1, -1)
    dcf = dcf.reshape(1, 1, -1)

    nufft_adj = tkbn.KbNufftAdjoint(im_size=img_shape)
    recon_nufft = nufft_adj(kdat * dcf, traj)
    return torch.flip(recon_nufft, dims=(-2, -1))

def animate_nufft(seq, seq0, k_space_data, k_marker_s=20, dt=1e-3, plot_window=1e-2, Nread=None, Nphase=None, 
                  time_range=None, fps=30, max_frames=None, show=True, save_filename=None, show_progress=False):
# example: ######################################################
# import edutils
# import importlib
# importlib.reload(edutils)
# ani = edutils.animate_nufft(seq, seq0, k_space_data=signal,k_marker_s=4, show=False, dt=seq.duration()[0] / 2000,Nread=256,Nphase=256,
#                     plot_window=seq.duration()[0]/30, fps=20, show_progress=True,save_filename='external.gif', time_range=[0.0,0.17])
#
# ani.save('external.gif', fps=20)
# ani.save('external.mp4', fps=20)
#
#
################################################################     
                      
    if time_range is None:
        time_range = [0, seq.duration()[0]]
    else:
        total_duration = seq0.get_duration()
        time_range = [max(0.0, time_range[0]), min(total_duration, time_range[1])]

    delta_kx = 1/seq.get_definition('FOV')[0]
    delta_ky = 1/seq.get_definition('FOV')[1]
    delta_kz = 1/seq.get_definition('FOV')[2]
    fov = seq.get_definition('FOV')[0]

    ts = np.linspace(time_range[0], time_range[1], int(np.ceil((time_range[1] - time_range[0])/dt))+1)
    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(9, 2, figure=fig)

    ax_rf = fig.add_subplot(gs[0, :])
    ax_x = fig.add_subplot(gs[1, :])
    ax_y = fig.add_subplot(gs[2, :])
    ax_z = fig.add_subplot(gs[3, :])
    ax_kspace = fig.add_subplot(gs[4:6, 0])
    ax_kspace2 = fig.add_subplot(gs[4:6, 1])
    ax_filled_kspace = fig.add_subplot(gs[6:, 0])
    ax_recon_image = fig.add_subplot(gs[6:, 1])

    for a in [ax_rf, ax_x, ax_y, ax_z, ax_kspace, ax_kspace2]:
        a.xaxis.set_ticklabels([])
        a.yaxis.set_ticklabels([])
        a.spines['left'].set_position('zero')
        a.spines['bottom'].set_position('zero')
        a.spines['right'].set_color('none')
        a.spines['top'].set_color('none')

    for a in [ax_rf, ax_x, ax_y, ax_z]:
        a.spines['left'].set_position('center')
        a.spines['left'].set_color('none')
        a.yaxis.set_ticks([])

    p_vrf = ax_rf.axvline(0, color='r')
    p_vx = ax_x.axvline(0, color='r')
    p_vy = ax_y.axvline(0, color='r')
    p_vz = ax_z.axvline(0, color='r')

    gw_pp = seq.get_gradients()
    wv = seq.waveforms(append_RF=True)
    ax_rf.plot(wv[3][0].real, wv[3][1].real)
    ax_rf.plot(wv[3][0].real, wv[3][1].imag)
    if gw_pp[0] is not None:
        ax_x.plot(wv[0][0], wv[0][1])
    if gw_pp[1] is not None:
        ax_y.plot(wv[1][0], wv[1][1])
    if gw_pp[2] is not None:
        ax_z.plot(wv[2][0], wv[2][1])

    k_full, k_adc_full, adc_mask_full = get_kspace_trajectory(seq0)

    p_kspace = ax_kspace.plot([],[])[0]
    p_adc = ax_kspace.plot([],[], 'r.', markersize=1)[0]
    p_cursor = ax_kspace.plot([],[], 'kx')[0]
    p_kspace2 = ax_kspace2.plot([],[])[0]
    p_adc2 = ax_kspace2.plot([],[], 'r.', markersize=1)[0]
    p_cursor2 = ax_kspace2.plot([],[], 'kx')[0]

    filled_kspace_plot = ax_filled_kspace.scatter([0],[0], c=[0], cmap='viridis', alpha=0.9, s=k_marker_s)
    ax_filled_kspace.set_xlim(-0.5*Nread/fov, 0.5*Nread/fov)
    ax_filled_kspace.set_ylim(-0.5*Nread/fov, 0.5*Nread/fov)
    ax_filled_kspace.set_xlabel('kx')
    ax_filled_kspace.set_ylabel('ky')
    ax_filled_kspace.set_facecolor(cm.viridis(0))
    recon_image_plot = ax_recon_image.imshow(np.zeros((Nread, Nphase)).T, cmap='gray', origin='lower')

    ax_kspace.set_xlim(np.nanmin(k_full[:,0]) - delta_kx*10, np.nanmax(k_full[:,0]) + delta_kx*10)
    ax_kspace.set_ylim(np.nanmin(k_full[:,1]) - delta_ky*10, np.nanmax(k_full[:,1]) + delta_ky*10)
    ax_kspace2.set_xlim(np.nanmin(k_full[:,0]) - delta_kx*10, np.nanmax(k_full[:,0]) + delta_kx*10)
    ax_kspace2.set_ylim(np.nanmin(k_full[:,2]) - delta_kz*10, np.nanmax(k_full[:,2]) + delta_kz*10)

    frames = len(ts) - 1
    if max_frames is not None:
        frames = min(max_frames, frames)

    if show_progress:
        progress_bar = tqdm(total=frames)

    def update(frame):
        if show_progress:
            progress_bar.update(frame + 1 - progress_bar.n)
        t_start, t_end = list(zip(ts[:-1], ts[1:]))[frame]

        ax_rf.set_xlim(t_start - plot_window/2, t_end + plot_window/2)
        ax_x.set_xlim(t_start - plot_window/2, t_end + plot_window/2)
        ax_y.set_xlim(t_start - plot_window/2, t_end + plot_window/2)
        ax_z.set_xlim(t_start - plot_window/2, t_end + plot_window/2)

        t = (t_start + t_end)/2
        p_vrf.set_data(([t,t], [0,1]))
        p_vx.set_data(([t,t], [0,1]))
        p_vy.set_data(([t,t], [0,1]))
        p_vz.set_data(([t,t], [0,1]))

        k, k_adc, mask_adc = get_kspace_trajectory(seq0, time_range=[0, t_end])

        if len(k) == 0:
            p_kspace.set_data([], [])
            p_kspace2.set_data([], [])
            p_adc.set_data([], [])
            p_adc2.set_data([], [])
            p_cursor.set_data([], [])
            p_cursor2.set_data([], [])
        else:
            p_kspace.set_xdata(k[:,0].cpu().numpy())
            p_kspace.set_ydata(k[:,1].cpu().numpy())
            p_kspace2.set_xdata(k[:,0].cpu().numpy())
            p_kspace2.set_ydata(k[:,2].cpu().numpy())
            p_adc.set_xdata(k_adc[:,0].cpu().numpy() if len(k_adc) > 0 else [])
            p_adc.set_ydata(k_adc[:,1].cpu().numpy() if len(k_adc) > 0 else [])
            p_adc2.set_xdata(k_adc[:,0].cpu().numpy() if len(k_adc) > 0 else [])
            p_adc2.set_ydata(k_adc[:,2].cpu().numpy() if len(k_adc) > 0 else [])
            try:
                p_cursor.set_xdata([k[-1,0].cpu().numpy()])
                p_cursor.set_ydata([k[-1,1].cpu().numpy()])
                p_cursor2.set_xdata([k[-1,0].cpu().numpy()])
                p_cursor2.set_ydata([k[-1,2].cpu().numpy()])
            except IndexError:
                print('no cursor')
                p_cursor.set_data([], [])
                p_cursor2.set_data([], [])

        if len(k_adc) > 0:
            adc_indices = torch.arange(len(k_adc))  # Sequential indices into k_space_data
            masked_traj_adc = k_adc
            filled_kspace_plot.set_offsets(np.column_stack((masked_traj_adc[:,0].cpu().numpy(), masked_traj_adc[:,1].cpu().numpy())))
            filled_kspace_mag = np.log(np.abs(k_space_data[adc_indices].cpu().numpy()) + 1).ravel()
            filled_kspace_plot.set_array(filled_kspace_mag)
            filled_kspace_plot.set_sizes([k_marker_s])
            filled_kspace_plot.set_clim(0, np.max(filled_kspace_mag) if np.max(filled_kspace_mag) > 0 else 1)

            recon_image = recon_nufft(k_space_data[adc_indices], masked_traj_adc, Nread=Nread, fov=fov)
            recon_image = np.abs(recon_image.cpu().numpy().squeeze())
            recon_image = recon_image / np.max(recon_image) if np.max(recon_image) > 0 else recon_image
            recon_image_plot.set_data(recon_image.T)
            recon_image_plot.set_clim(0, 1)

    ax_rf.text(-0.07, 0, 'RF', transform=ax_rf.transAxes)
    ax_x.text(-0.07, 0, 'GX', transform=ax_x.transAxes)
    ax_y.text(-0.07, 0, 'GY', transform=ax_y.transAxes)
    ax_z.text(-0.07, 0, 'GZ', transform=ax_z.transAxes)
    ax_kspace.text(0.5, 0, 'Ky ↑', horizontalalignment='center', verticalalignment='top', transform=ax_kspace.transAxes)
    ax_kspace.text(0.0, 0.5, 'Kx → ', verticalalignment='center', horizontalalignment='right', transform=ax_kspace.transAxes)
    ax_kspace2.text(0.5, 0, 'Kz ↑', horizontalalignment='center', verticalalignment='top', transform=ax_kspace2.transAxes)
    ax_kspace2.text(0.0, 0.5, 'Kx → ', verticalalignment='center', horizontalalignment='right', transform=ax_kspace2.transAxes)
    ax_filled_kspace.set_title('Filled k-space')
    ax_recon_image.set_title('Reconstruction')

    plt.tight_layout()
    ani = animation.FuncAnimation(fig=fig, func=update, frames=frames, interval=1000/fps)

    if save_filename is not None:
        ani.save(save_filename, fps=fps)
    if show:
        plt.show()
    if not show:
        plt.close()

    return ani

if __name__ == "__main__":
    pass
