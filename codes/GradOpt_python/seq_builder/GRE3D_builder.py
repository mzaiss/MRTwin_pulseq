from __future__ import annotations
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import new_core.sequence as Seq
from new_core import util


class GRE3D:
    """Stores all parameters needed to create a 3D GRE sequence."""

    def __init__(self, adc_count: int, rep_count: int, part_count: int, R_accel: (int,int) = (1,1)):
        """Initialize parameters with default values."""
        self.adc_count = adc_count
        self.event_count = adc_count + 4
        self.rep_count = rep_count // R_accel[0]
        self.part_count = part_count // R_accel[1]
        self.shots = 1 # Number of shots
        self.R_accel = R_accel

        self.pulse_angles = torch.full((self.rep_count*self.part_count, ), 5 * np.pi / 180)
        self.pulse_phases = torch.tensor(
            [util.phase_cycler(r, 117) for r in range(self.rep_count*self.part_count)])
        self.gradm_rewinder = torch.full((rep_count*part_count, ), -adc_count/2-1)
        self.gradm_phase = torch.arange(-rep_count//2+np.mod(rep_count//2,R_accel[0]), rep_count//2, R_accel[0]).repeat(self.part_count)
        self.gradm_part = torch.arange(-(part_count//2+np.mod(part_count//2,R_accel[1])), (part_count+1)//2, R_accel[1]).repeat_interleave(self.rep_count)
        self.gradm_adc = torch.full((rep_count*part_count, ), 1.0)
        self.gradm_spoiler = torch.full((rep_count*part_count, ), 1.5 * adc_count)
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
        self.TE = 0
        self.TR = 0
        
        self.relaxation_time = torch.tensor(1e-5)
        
    def linearEncoding(self, adc_count: int, rep_count: int, part_count: int) -> GRE3D:
        self.gradm_phase = torch.arange(-rep_count//2+np.mod(rep_count//2,self.R_accel[0]), rep_count//2, self.R_accel[0]).repeat(self.part_count)
        self.gradm_part = torch.arange(-(part_count//2+np.mod(part_count//2,self.R_accel[1])), (part_count+1)//2, self.R_accel[1]).repeat_interleave(self.rep_count)
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
        
    def centricEncoding(self, adc_count: int, rep_count: int, part_count: int) -> GRE3D:
        
        # permutation vector 
        def permvec(x) -> np.ndarray:
            permvec = np.zeros((x,),dtype=int) 
            permvec[0] = 0
            for i in range(1,int(x/2)+1):
                permvec[i*2-1] = (-i)
                if i < x/2:
                    permvec[i*2] = i
            return permvec+x//2  
        
        tmp = torch.arange(-rep_count//2+np.mod(rep_count//2,self.R_accel[0]), rep_count//2, self.R_accel[0])
        self.gradm_phase = tmp[permvec(self.rep_count)].repeat(self.part_count)
        tmp = torch.arange(-(part_count//2+np.mod(part_count//2,self.R_accel[1])), (part_count+1)//2, self.R_accel[1])
        self.gradm_part = tmp[permvec(self.part_count)].repeat_interleave(self.rep_count)
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
    
    
    def spiralEncoding(self, spiral_elongation = 0, alternating = False) -> GRE3D:
        """Create spiral encoding in y and z direction.""" 
        # permutation vector 
        def permvec(x) -> np.ndarray:
            permvec = np.zeros((x,),dtype=int) 
            permvec[0] = 0
            for i in range(1,int(x/2)+1):
                permvec[i*2-1] = (-i)
                if i < x/2:
                    permvec[i*2] = i
            return permvec+x//2   
        
        a, b = torch.meshgrid(self.gradm_phase[:self.rep_count],self.gradm_part[::self.rep_count])
        reordering_y = []
        reordering_z = []
        size_y = a.shape[0]
        size_z = a.shape[1]
        
        corr = 0
        if spiral_elongation == 0:
            Iy = 1 # Number of first encoding line in y direction
            Iz = 1 # Number of first encoding line in z directio
            pos_lin = a.shape[0]//2 # Position in y direction
            pos_par = a.shape[1]//2 # Position in z direction
        elif spiral_elongation > 0:
            Iy = int(np.ceil(np.abs(spiral_elongation)*size_y)) + 1
            Iz = 1
            pos_lin = a.shape[0]//2+int(np.ceil(Iy/2))-1 # Position in y direction
            pos_par = a.shape[1]//2 # Position in z direction
        elif spiral_elongation < 0:
            Iy = 1
            Iz = int(np.ceil(np.abs(spiral_elongation)*size_z))
            pos_lin = a.shape[0]//2 # Position in y direction
            pos_par = a.shape[1]//2-int(np.ceil(Iz/2)) # Position in z direction
            for jj in range(0,Iz):
                #print(jj)
                reordering_y.append(a[pos_lin,pos_par+jj])
                reordering_z.append(b[pos_lin,pos_par+jj])
            pos_par += Iz
            corr = 1
        
        sign = 1
        Iy = Iy
        Iz = Iz+corr
                
        while (Iy < size_y) or (Iz < size_z) or len(reordering_y) < size_y*size_z:
            pos_lin = min(pos_lin,size_y-1)
            pos_par = min(pos_par,size_z-1)
            if Iz <= a.shape[1]:
                for ii in range(0,min(Iy,size_y)):
                    #print(ii)
                    reordering_y.append(a[pos_lin-sign*ii,pos_par])
                    reordering_z.append(b[pos_lin-sign*ii,pos_par])
            else:
                Iz = min(Iz,size_z)
            pos_lin -= sign*min(Iy,size_y-1)
            
            if Iy <= size_y:
                for jj in range(0,Iz):
                    #print(jj)
                    reordering_y.append(a[pos_lin,pos_par-sign*jj])
                    reordering_z.append(b[pos_lin,pos_par-sign*jj])
            else:
               Iy = min(Iy,size_y) 
            Iy += 1
            pos_par -= sign*min(Iz,size_z-1)
            Iz += 1
            # print(j)
            # print(i)
            sign *= -1

        num_perm = max(int(np.ceil(spiral_elongation*size_y))-1,int(np.ceil(-spiral_elongation*size_z)))+1
        perm = permvec(num_perm) 
        
        self.gradm_phase = torch.tensor(reordering_y)
        self.gradm_part = torch.tensor(reordering_z)
        
        if alternating:
            self.gradm_phase[:num_perm] = self.gradm_phase[perm]
            self.gradm_part[:num_perm] = self.gradm_part[perm]
        
        self.gradm_spoiler_phase = -self.gradm_phase
        self.gradm_spoiler_part = -self.gradm_part
        cmap = plt.cm.get_cmap('rainbow')
        
        plt.plot(self.gradm_part,self.gradm_phase); plt.xlabel('z'); plt.ylabel('y');plt.title('Spiral elongation = ' + str(spiral_elongation))
        for i in range(num_perm):
            plt.plot(self.gradm_part[i], self.gradm_phase[i],'.', c=cmap(i / num_perm))
        plt.show()
        
    def clone(self) -> GRE3D:
        """Create a copy with cloned tensors."""
        clone = GRE3D(self.adc_count, self.rep_count, self.part_count)

        clone.pulse_angles = self.pulse_angles.clone()
        clone.pulse_phases = self.pulse_phases.clone()
        clone.gradm_rewinder = self.gradm_rewinder.clone()
        clone.gradm_phase = self.gradm_phase.clone()
        clone.gradm_part = self.gradm_part.clone()
        clone.gradm_adc = self.gradm_adc.clone()
        clone.gradm_spoiler = self.gradm_spoiler.clone()
        clone.gradm_spoiler_phase = self.gradm_spoiler_phase.clone()
        clone.gradm_spoiler_part = self.gradm_spoiler_part.clone()
        clone.relaxation_time = self.relaxation_time.clone()

        return clone

    def generate_sequence(self, oversampling = 1) -> Seq.Sequence:
        """Generate a GRE sequence based on the given parameters."""
        
        seq_all = []

        for shot in range(self.shots): 
            seq = Seq.Sequence()
            
            for ii in torch.arange(shot,self.part_count*self.rep_count,self.shots):
                # extra events: pulse + winder + rewinder
                rep = Seq.Repetition.zero(self.event_count+(oversampling-1)*self.adc_count)
                seq.append(rep)
    
                rep.pulse.angle = self.pulse_angles[ii]
                rep.pulse.phase = self.pulse_phases[ii]
                rep.pulse.usage = Seq.PulseUsage.EXCIT
    
                rep.event_time[0] = 1.7e-3 # (1 slice) 3e-3 (4 slices)  # Pulse
                rep.event_time[1] = 0.68e-3 + self.TE # Winder 
                rep.event_time[2:-2] = 0.02e-3 # (64x64)  0.010*1e-3 (128x128)  # Readout
    
                rep.gradm[1, 0] = self.gradm_rewinder[ii]
                rep.gradm[1, 1] = self.gradm_phase[ii]
                rep.gradm[1, 2] = self.gradm_part[ii]
                rep.gradm[2:-2, 0] = self.gradm_adc[ii]/oversampling
    
                # Rewinder / Spoiler, centers readout in rep
                rep.event_time[-1] = 0.060e-3 + self.TR #64x64
                rep.event_time[-2] = 0.68e-3   # Spoiler
    
                rep.gradm[-2, 0] = self.gradm_spoiler[ii]
                rep.gradm[-2, 1] = self.gradm_spoiler_phase[ii]
                rep.gradm[-2, 2] = self.gradm_spoiler_part[ii]
    
                rep.adc_usage[2:-2] = 1
                rep.adc_phase[:] = np.pi/2 - rep.pulse.phase
                
                    
            seq[-1].event_time[-1] = self.relaxation_time
            
            seq_all.append(seq)

        return seq_all        
    
    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_name) -> GRE3D:
        with open(file_name, 'rb') as file:
            return pickle.load(file)


def plot_optimization_progress(
    reco: torch.Tensor, reco_target: torch.Tensor,
    params: GRE2D, params_target: GRE2D,
    kspace_trajectory: list[torch.Tensor], loss_history: list[float],
    figsize: tuple[float, float] = (10, 10), dpi: float = 180
) -> np.ndarray:
    """
    Plot a picture containing the most important sequence properties.

    This function also returns the plotted image as array for gif creation.
    """
    plt.figure(figsize=figsize)
    reco_max = max(np.abs(util.to_numpy(reco[:, :, 0])).max(),
                   np.abs(util.to_numpy(reco_target[:, :, 0])).max())
    plt.subplot(3, 2, 1)
    plt.imshow(np.abs(util.to_numpy(reco[:, :, 0])), vmin=0, vmax=reco_max)
    plt.colorbar()
    plt.title("Reco")
    plt.subplot(3, 2, 3)
    plt.imshow(np.abs(util.to_numpy(reco_target[:, :, 0])), vmin=0, vmax=reco_max)
    plt.colorbar()
    plt.title("Target")

    plt.subplot(3, 2, 2)
    plt.plot(np.abs(util.to_numpy(params.pulse_angles)) * 180 / np.pi, '.')
    plt.plot(util.to_numpy(params_target.pulse_angles) * 180 / np.pi, '.', color='r')
    plt.title("Flip Angles")
    plt.ylim(bottom=0)
    plt.subplot(3, 2, 4)
    plt.plot(np.mod(np.abs(util.to_numpy(params.pulse_phases)) * 180 / np.pi, 360), '.')
    plt.plot(np.mod(np.abs(util.to_numpy(params_target.pulse_phases)) * 180 / np.pi, 360), '.', color='r')
    plt.title("Phase")

    plt.subplot(3, 2, 5)
    plt.plot(loss_history)
    plt.yscale('log')
    plt.grid()
    plt.title("Loss Curve")

    plt.subplot(3, 2, 6)
    for i, rep_traj in enumerate(kspace_trajectory):
        kx = util.to_numpy(rep_traj[:, 0]) / (2*np.pi)
        ky = util.to_numpy(rep_traj[:, 1]) / (2*np.pi)
        plt.plot(kx, ky, c=cm.rainbow(i / len(kspace_trajectory)))
        plt.plot(kx, ky, 'k.')
    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")
    plt.grid()

    img = util.current_fig_as_img(dpi)
    plt.show()
    return img
