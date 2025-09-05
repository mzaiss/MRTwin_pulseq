'''
Exercises - Homework:

1.1. Spin echo EPI  with zig-zag or non blipped trajectory and silent EPI   (evtl propeller EPI)
hint: exersice or any textbook for spin echo, http://mri.beckman.illinois.edu/interactive/topics/contents/fast_imaging/figures/zig.shtml
hint: Fig 2 in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5836719/, reuse radial reco
 reuse radial reco,   pp.make_extended_trapezoid, and check 1.10 spiral trajectory below

1.2. DREAM MRI
two readouts from one measurement: eg. FID, STE as two separate contrasts
hint: https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.24158

1.3. Diffusion weighted MRI
GRE DWI and spin echo DWI
hint: any textbook, https://blog.ismrm.org/2017/06/06/dwe-part-2/ 
# chek your sim by inserting a rectangular "Tumor" only visible in DWI
# typical brain tumor ADC values are around ~1.5 * 10^-3 mm^2/s,
# which lies between GM/WM and CSF (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3000221)
# mr0 uses D in units of 10^-3 * mm ^2/s  this is the same as µm^2/ms
obj_p.D*=1
if 1:
    # construct tumor border region
    for ii in range(15, 25):
        for jj in range(15, 25):
            obj_p.D[ii, jj] = torch.tensor(0.75)
    # construct tumor filling
    for ii in range(16, 24):
        for jj in range(16, 24):
            obj_p.D[ii, jj] = torch.tensor(1.5)

1.4. B0 / T2* mapping  
hint: FLASH phase maps, any text book, both TE after one excitation

1.5. B1 mapping  
hint: Double angle https://onlinelibrary.wiley.com/doi/10.1002/mrm.20896 or (cosine fit)
reuse MP-FLASH

1.6a. T1 mapping  (invrec)
hint: any textbook, reuse MP-FLASH

1.6b. T1 mapping  (two flipangles)
hint: equation 3a and 3b of https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.21969

1.7a. T2 mapping (Spin echo)
hint: any textbook, reuse SE or RARE

1.7b. T2 mapping (t2prep)
hint: https://www.kjronline.org/ViewImage.php?Type=F&aid=549810&id=F6&afn=68_KJR_18_1_113&fn=kjr-18-113-g006_0068KJR
reuse FLASH

1.8. FISP / PSIF 
hint: https://mriquestions.com/psif-vs-fisp.html
oder (fat suppression / water fat imaging)

1.10. Spiral Trajectory -  EPI
any textbook and https://pulseq.github.io/, reuse radial reco
pp.make_arbitrary_grad.make_arbitrary_grad
pp.points_to_waveform
pp.traj_to_grad
grad_raster_time = 10e-6   % better!
seq = pp.Sequence(system=system)  % more acuarate
seq.calculate_kspace()
# Plot k-spaces
ktraj_adc, ktraj, t_excitation, t_refocusing, _ = seq.calculate_kspace()
plt.plot(ktraj.T)  # Plot the entire k-space trajectory
plt.figure()
plt.plot(ktraj[0], ktraj[1], 'b')  # 2D plot
plt.axis('equal')  # Enforce aspect ratio for the correct trajectory display
plt.plot(ktraj_adc[0], ktraj_adc[1], 'r.')
plt.show()

1.11  eight ball echo

1.17 Fastest Flash with trapezoidal gradients and ramp sampling

1.18 Slice selection - slice selection profile simulation and meas
https://github.com/pulseq/MR-Physics-with-Pulseq/tree/main/tutorials/02_rf_pulses
https://github.com/pulseq/MR-Physics-with-Pulseq/blob/main/slides/D1-1320-RF_Pulses_Tony_Stoecker.pdf
 - full free FOV rotation translation

2.1. Code a small Bloch simulator an visualize spins



"""

Tuesday to Thursday 

17:00 Zoom meeting for questions

Excursion Henkestrasse 91 ALDI other side of the street


 
