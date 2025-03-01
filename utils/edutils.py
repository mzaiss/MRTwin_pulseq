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


def animate_nufft(seq, k_space_data,k_traj0=None, k_marker_s=20, dt=1e-3, plot_window=1e-2, Nread=None,Nphase=None, time_range=None, fps=30, max_frames=None,
            show=True, save_filename=None, show_progress=False):
  if time_range is None:
      time_range = [0, seq.duration()[0]]

  delta_kx = 1/seq.get_definition('FOV')[0]
  delta_ky = 1/seq.get_definition('FOV')[1]
  delta_kz = 1/seq.get_definition('FOV')[2]

  fov=seq.get_definition('FOV')[0]

  def recon_nufft(signal, kspace_loc,verbose=0):
    img_shape = [Nread] * 2
  
    # prepare k-space trajectory traj
    traj = kspace_loc[:, :2].T  # tkbn assumes xy in the first dim, thus .T
    traj = traj / (Nread/fov) * np.pi * 2 # normalize k-space trajectory from -kmax to kmax to -pi to pi for tkbn
  
    if 0:# compute density compensation function manually
        dcf = (traj[0,:]**2 + traj[1,:]**2)**0.5 # density compensation factor
        if verbose: print(dcf.shape,'should be: [num_samples]')
    else:# calculate density compensation function using  tkbn.calc_density_compensation_function
        dcf = tkbn.calc_density_compensation_function(ktraj=traj, im_size=img_shape)
        if verbose: print(dcf.shape,'should be: [batch_size, num_coils,num_samples]')
  
    # prepare kdat
    kdat = signal.squeeze()
    if verbose: print(kdat.shape,'should be: [num_samples]') # should be num_samples
  
    # Reshape kdat and dcf: (1, 1, num_samples) -> (batch_size, num_coils, num_samples) for adjoint
    kdat = kdat.reshape(1, 1, -1)
    dcf = dcf.reshape(1, 1, -1)
  
    # define nufft adjoint operator
    nufft_adj = tkbn.KbNufftAdjoint(im_size=img_shape)
    recon_nufft = nufft_adj(kdat*dcf , traj)
    recon_nufft = torch.flip(recon_nufft, dims=(-2, -1))
    return recon_nufft



  ts = np.linspace(time_range[0], time_range[1], int(np.ceil((time_range[1] - time_range[0])/dt))+1)

  fig = plt.figure(figsize=(12, 12))

  # Updated GridSpec with additional row for filled k-space and reconstruction
  gs = GridSpec(9, 2, figure=fig)

  # Original plots
  ax_rf = fig.add_subplot(gs[0, :])
  ax_x = fig.add_subplot(gs[1, :])
  ax_y = fig.add_subplot(gs[2, :])
  ax_z = fig.add_subplot(gs[3, :])

  ax_kspace = fig.add_subplot(gs[4:6, 0])
  ax_kspace2 = fig.add_subplot(gs[4:6, 1])

  # New plots for filled k-space and image reconstruction in a new row
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

  # Original animation code remains the same
  p_vrf = ax_rf.axvline(0, color='r')
  p_vx = ax_x.axvline(0, color='r')
  p_vy = ax_y.axvline(0, color='r')
  p_vz = ax_z.axvline(0, color='r')

  gw_pp = seq.get_gradients()
  wv = seq.waveforms(append_RF=True)

  ax_rf.plot(wv[3][0].real, wv[3][1].real)
  ax_rf.plot(wv[3][0].real, wv[3][1].imag)

  if gw_pp[0] != None:
      ax_x.plot(wv[0][0], wv[0][1])

  if gw_pp[1] != None:
      ax_y.plot(wv[1][0], wv[1][1])

  if gw_pp[2] != None:
      ax_z.plot(wv[2][0], wv[2][1])

  total_duration = sum(seq.block_durations.values())

  # Original k-space trajectory calculation
  # ... [keep all the calculation code unchanged] ...
  t_excitation, fp_excitation, t_refocusing, _ = seq.rf_times()
  t_adc, _ = seq.adc_times()
  ng = 3
  eps = 1e-8
  gm_pp = []
  tc = []
  for i in range(ng):
      if gw_pp[i] is None:
          gm_pp.append(None)
          continue

      gm_pp.append(gw_pp[i].antiderivative())
      tc.append(gm_pp[i].x)
      # "Sample" ramps for display purposes otherwise piecewise-linear display (plot) fails
      ii = np.flatnonzero((abs(gm_pp[i].c)>0).any(0))

      # Do nothing if there are no ramps
      if ii.shape[0] == 0:
          continue

      starts = np.int64(np.floor((gm_pp[i].x[ii] + eps) / seq.grad_raster_time))
      ends = np.int64(np.ceil((gm_pp[i].x[ii+1] - eps) / seq.grad_raster_time))

      # Create all ranges starts[0]:ends[0], starts[1]:ends[1], etc.
      lengths = ends-starts+1
      inds = np.ones((lengths).sum())
      # Calculate output index where each range will start
      start_inds = np.cumsum(np.concatenate(([0],lengths[:-1])))
      # Create element-wise differences that will cumsum into
      # the final indices: [starts[0], 1, 1, starts[1]-starts[0]-lengths[0]+1, 1, etc.]
      inds[start_inds] = np.concatenate(([starts[0]], np.diff(starts) - lengths[:-1] + 1))

      tc.append(np.cumsum(inds) * seq.grad_raster_time)
  if tc != []:
      tc = np.concatenate(tc)

  t_acc = 1e-10  # Temporal accuracy
  t_acc_inv = 1 / t_acc
  t_ktraj = t_acc * np.unique(
      np.round(
          t_acc_inv
          * np.array(
              [
                  *tc,
                  0,
                  *np.asarray(t_excitation) - 2 * seq.rf_raster_time,
                  *np.asarray(t_excitation) - seq.rf_raster_time,
                  *t_excitation,
                  *np.asarray(t_refocusing) - seq.rf_raster_time,
                  *t_refocusing,
                  *t_adc,
                  total_duration,
              ]
          )
      )
  )

  t_acc = 1e-10  # Temporal accuracy
  t_acc_inv = 1 / t_acc
  i_excitation = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_excitation)))
  i_refocusing = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_refocusing)))
  i_adc = np.searchsorted(t_ktraj, t_acc * np.round(t_acc_inv * np.asarray(t_adc)))

  i_periods = np.unique([0, *i_excitation, *i_refocusing, len(t_ktraj) - 1])
  if len(i_excitation) > 0:
      ii_next_excitation = 0
  else:
      ii_next_excitation = -1
  if len(i_refocusing) > 0:
      ii_next_refocusing = 0
  else:
      ii_next_refocusing = -1

  k_traj = np.zeros((ng, len(t_ktraj)))
  for i in range(ng):
      if gw_pp[i] is None:
          continue

      it = np.where(np.logical_and(
          t_ktraj >= t_acc * round(t_acc_inv * gm_pp[i].x[0]),
          t_ktraj <= t_acc * round(t_acc_inv * gm_pp[i].x[-1]),
      ))[0]
      k_traj[i, it] = gm_pp[i](t_ktraj[it])
      if t_ktraj[it[-1]] < t_ktraj[-1]:
          k_traj[i, it[-1] + 1 :] = k_traj[i, it[-1]]

  # Convert gradient moments to kspace
  dk = -k_traj[:, 0]
  for i in range(len(i_periods) - 1):
      i_period = i_periods[i]
      i_period_end = i_periods[i + 1]
      if ii_next_excitation >= 0 and i_excitation[ii_next_excitation] == i_period:
          if abs(t_ktraj[i_period] - t_excitation[ii_next_excitation]) > t_acc:
              raise Warning(
                  f"abs(t_ktraj[i_period]-t_excitation[ii_next_excitation]) < {t_acc} failed for ii_next_excitation={ii_next_excitation} error={t_ktraj(i_period) - t_excitation(ii_next_excitation)}"
              )
          dk = -k_traj[:, i_period]
          if i_period > 0:
              # Use nans to mark the excitation points since they interrupt the plots
              k_traj[:, i_period - 1] = np.NaN
          # -1 on len(i_excitation) for 0-based indexing
          ii_next_excitation = min(len(i_excitation) - 1, ii_next_excitation + 1)
      elif (
          ii_next_refocusing >= 0 and i_refocusing[ii_next_refocusing] == i_period
      ):
          dk = -2 * k_traj[:, i_period] - dk
          # -1 on len(i_excitation) for 0-based indexing
          ii_next_refocusing = min(len(i_refocusing) - 1, ii_next_refocusing + 1)

      k_traj[:, i_period:i_period_end] = (
          k_traj[:, i_period:i_period_end] + dk[:, None]
      )

  k_traj[:, i_period_end] = k_traj[:, i_period_end] + dk
  k_traj_adc = k_traj[:, i_adc]

  # Original plot elements
  p_kspace = ax_kspace.plot([],[])[0]
  p_adc = ax_kspace.plot([],[], 'r.', markersize=1)[0]
  p_cursor = ax_kspace.plot([],[], 'kx')[0]

  p_kspace2 = ax_kspace2.plot([],[])[0]
  p_adc2 = ax_kspace2.plot([],[], 'r.', markersize=1)[0]
  p_cursor2 = ax_kspace2.plot([],[], 'kx')[0]

  # New plot elements for filled k-space and reconstruction
  # Note: transposed display with origin='lower'
  filled_kspace_plot = ax_filled_kspace.scatter([0],[0], c=[0],cmap='viridis',alpha=0.9, s=k_marker_s)
  ax_filled_kspace.set_xlim((-0.5*Nread/fov,+0.5*Nread/fov))
  ax_filled_kspace.set_ylim((-0.5*Nread/fov,+0.5*Nread/fov))
  ax_filled_kspace.set_xlabel('kx')
  ax_filled_kspace.set_ylabel('ky')
  ax_filled_kspace.set_facecolor(cm.viridis[0])
  recon_image_plot = ax_recon_image.imshow(np.zeros((Nread, Nphase)).T,cmap='gray',origin='lower')

  ax_kspace.set_xlim(np.nanmin(k_traj_adc[0]) - delta_kx*10, np.nanmax(k_traj_adc[0]) + delta_kx*10)
  ax_kspace.set_ylim(np.nanmin(k_traj_adc[1]) - delta_ky*10, np.nanmax(k_traj_adc[1]) + delta_ky*10)

  ax_kspace2.set_xlim(np.nanmin(k_traj_adc[0]) - delta_kx*10, np.nanmax(k_traj_adc[0]) + delta_kx*10)
  ax_kspace2.set_ylim(np.nanmin(k_traj_adc[2]) - delta_kz*10, np.nanmax(k_traj_adc[2]) + delta_kz*10)

  # Create empty k-space for filling
  filled_kspace = np.zeros_like(k_space_data)

  frames = len(ts)-1
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

      mask = np.flatnonzero((t_ktraj <= t_end) & (t_ktraj >= time_range[0]))
      mask_adc = np.flatnonzero((t_adc <= t_end) & (t_adc >= time_range[0]))

      p_kspace.set_xdata(k_traj[0,mask])
      p_kspace.set_ydata(k_traj[1,mask])
      p_kspace2.set_xdata(k_traj[0,mask])
      p_kspace2.set_ydata(k_traj[2,mask])

      p_adc.set_xdata(k_traj_adc[0,mask_adc])
      p_adc.set_ydata(k_traj_adc[1,mask_adc])
      p_adc2.set_xdata(k_traj_adc[0,mask_adc])
      p_adc2.set_ydata(k_traj_adc[2,mask_adc])

      c_ind = abs(t_ktraj-t_end).argmin()
      p_cursor.set_xdata([k_traj[0, c_ind]])
      p_cursor.set_ydata([k_traj[1, c_ind]])
      p_cursor2.set_xdata([k_traj[0, c_ind]])
      p_cursor2.set_ydata([k_traj[2, c_ind]])
      if len(mask_adc)> 0:


        if k_traj0 is None:
          # Update the filled k-space and reconstruction
          filled_kspace_plot.set_offsets(np.column_stack((k_traj_adc[0, mask_adc],k_traj_adc[1, mask_adc])))
          #filled_kspace_plot.set_offsets((k_traj_adc[0,mask_adc],k_traj_adc[1,mask_adc]))
          filled_kspace_mag=np.log(np.abs(k_space_data[mask_adc,:].cpu().numpy())+1).ravel()
          filled_kspace_plot.set_array(filled_kspace_mag)
          filled_kspace_plot.set_clim(0, np.max(filled_kspace_mag) if np.max(filled_kspace_mag) > 0 else 1)

          masked_traj_adc= torch.from_numpy(k_traj_adc[:,mask_adc].T)
        else:
          masked_traj_adc= k_traj0[mask_adc,:]
          # Update the filled k-space and reconstruction
          filled_kspace_plot.set_offsets(np.column_stack((k_traj0[mask_adc,0].numpy(),k_traj0[mask_adc,1].numpy())))
          #filled_kspace_plot.set_offsets((k_traj_adc[0,mask_adc],k_traj_adc[1,mask_adc]))
          filled_kspace_mag=np.log(np.abs(k_space_data[mask_adc,:].cpu().numpy())+1).ravel()
          filled_kspace_plot.set_array(filled_kspace_mag)
          filled_kspace_plot.set_sizes([k_marker_s])
          filled_kspace_plot.set_clim(0, np.max(filled_kspace_mag) if np.max(filled_kspace_mag) > 0 else 1)


        # Perform forward NUFFT
        recon_image = recon_nufft(k_space_data[mask_adc,:],masked_traj_adc)
        #print(mask_adc)

        # Normalize for display
        recon_image=np.abs(recon_image.cpu().numpy().squeeze())
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
    # Add example usage or testing code here if desired
    pass
