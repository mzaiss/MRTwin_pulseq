# %% S0. SETUP env
import MRzeroCore as mr0
import pypulseq as pp
import util
import numpy as np
import matplotlib.pyplot as plt

# makes the ex folder your working directory
import os 
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.chdir(os.path.abspath(os.path.dirname(__file__)))

experiment_id = 'exA00_phantom'

# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
sz = [64, 64]

if 0:
    # (i) load a phantom object from file
    # obj_p = mr0.VoxelGridPhantom.load_mat('../data/phantom2D.mat')
    obj_p = mr0.VoxelGridPhantom.load_mat('../data/numerical_brain_cropped.mat')
    obj_p = obj_p.interpolate(sz[0], sz[1], 1)

# Manipulate loaded data
    obj_p.B0 *= 1    # alter the B0 inhomogeneity
    obj_p.D *= 0 
    obj_p.B0 *= 1    # alter the B0 inhomogeneity
else:
    # or (ii) set phantom  manually to a pixel phantom. Coordinate system is [-0.5, 0.5]^3
    obj_p = mr0.CustomVoxelPhantom(
        pos=[[-0.25, -0.25, 0]],
        PD=[1.0],T1=[3.0],T2=[0.5],T2dash=[30e-3], D=[0.0], B0=0,
        voxel_size=0.1,
        voxel_shape="box"
    )

obj_p.plot()
obj_p.size=torch.tensor([fov, fov, slice_thickness]) 
# Convert Phantom into simulation data
obj_p = obj_p.build()


# %% S4: SETUP SPIN SYSTEM/object on which we can run the MR sequence external.seq from above
sz = [64, 64]

# or (ii) set phantom  manually to a pixel phantom. Coordinate system is [-0.5, 0.5]^3
obj_p = mr0.CustomVoxelPhantom(
    pos=[[-0.4, -0.4, 0], [-0.4, -0.2, 0],
         [-0.3, -0.2, 0], [-0.2, -0.2, 0], [-0.1, -0.2, 0]],
    PD=[1.0, 1.0, 0.5, 0.5, 0.5],
    T1=[1.0, 0.5, 0.5, 0.5, 2],
    T2=0.1,
    T2dash=0.1,
    D=0.0,
    B0=0,
    voxel_size=0.1,
    voxel_shape="box"
)


obj_p.plot()
obj_p.size=torch.tensor([fov, fov, slice_thickness]) 
# Convert Phantom into simulation data
obj_p = obj_p.build()
