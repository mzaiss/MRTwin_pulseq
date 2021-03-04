import numpy as np
import scipy.io as sio
from math import pi,ceil, sqrt, pow
import sys

sys.path.append("../scannerloop_libs")
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.opts import Opts


def rectify_flips(flips):
    rflips = np.copy(flips)
    
    for i in range(rflips.shape[0]):
        for j in range(rflips.shape[1]):
            if rflips[i,j,0] < 0:
                rflips[i,j,0] = -rflips[i,j,0]
                rflips[i,j,1] += np.pi
                rflips[i,j,1] = np.mod(rflips[i,j,1], 2*np.pi)
    return rflips

def FOV():
    #FOV = 0.110
    return 0.200

nonsel = 0
if nonsel==1:
    slice_thickness = 200*1e-3
else:
    slice_thickness = 8e-3

def pulseq_write_super(seq_params, seq_fn, plot_seq=False):
    flips_numpy, event_time_numpy, grad_moms_numpy_input, adc_mask_numpy = seq_params
    
    flips_numpy = rectify_flips(flips_numpy)
    event_time_numpy = np.abs(event_time_numpy)
    
    NRep = flips_numpy.shape[1]
    T = flips_numpy.shape[0]
    
    # save pulseq definition
    MAXSLEW = 140
    
    
    deltak = 1.0 / FOV()
    grad_moms_numpy = deltak*grad_moms_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)   
    
    seq.add_block(make_delay(5.0))
    
    
    for rep in range(NRep):
        adc_start = 0
        for event in range(T):
            ###############################
            ## global pulse
            if adc_mask_numpy[event] == 0:  
                RFdur=0
                if flips_numpy[event,rep,3] == 0:
                        RFdur=0
                        if np.abs(flips_numpy[event,rep,0]) > 1e-8:
                
                            RFdur = 1*1e-3
                            kwargs_for_block = {"flip_angle": flips_numpy[event,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[event,rep,1]}
                            rf,_ = make_block_pulse(**kwargs_for_block)
                            
                            seq.add_block(rf)  
                            seq.add_block(make_delay(1e-4))
                                                
                elif flips_numpy[event,rep,3] == 1:
                    RFdur = 0
                    if np.abs(flips_numpy[event,rep,0]) > 1e-8:
                        use = "excitation"
                        
                        if nonsel:
                            RFdur = 1*1e-3
                            kwargs_for_block = {"flip_angle": flips_numpy[event,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[event,rep,1]}
                            rf_ex,_ = make_block_pulse(**kwargs_for_block)
                            seq.add_block(rf_ex)     
                        else:
                            # alternatively slice selective:
                            use = "excitation"
                            RFdur = 1*1e-3
                            kwargs_for_sinc = {"flip_angle": flips_numpy[event,rep,0], "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flips_numpy[event,rep,1]}
                            rf_ex, gz, gzr= make_sinc_pulse(**kwargs_for_sinc)
                            seq.add_block(rf_ex, gz)
                            gzr.amplitude=gzr.amplitude
                            seq.add_block(gzr)
                            RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
                            

                
                elif flips_numpy[event,rep,3] == 2:
                    ###############################
                    ### refocusing pulse
                    use = "refocusing"
                    
                    RFdur = 0
                    
                    if np.abs(flips_numpy[event,rep,0]) > 1e-8:
                      RFdur = 1*1e-3
                      
                      if nonsel:
                          kwargs_for_block = {"flip_angle": flips_numpy[event,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[event,rep,1]}
                          rf_ref,_ = make_block_pulse(**kwargs_for_block)
                          seq.add_block(rf_ref)         
                      else:

                          kwargs_for_sinc = {"flip_angle": flips_numpy[event,rep,0], "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flips_numpy[event,rep,1]}
                          rf_ref, gz_ref,gzr = make_sinc_pulse(**kwargs_for_sinc)
                          seq.add_block(rf_ref, gz_ref)
                          RFdur = gz_ref.rise_time + gz_ref.flat_time + gz_ref.fall_time 
        
                dur = event_time_numpy[event,rep] - RFdur
                if dur < 0:
                    raise Exception('Event Time too short! Event Time: Rep: '+ str(rep) + ', Event: ' +str(event),', increase event_time by at least: ' + str(-dur))
                
                gx_gradmom = grad_moms_numpy[event,rep,0]
                gy_gradmom = grad_moms_numpy[event,rep,1]


                if np.abs(gx_gradmom)>0:
                    gx_adc_ramp = 0
                    if adc_mask_numpy[event+1] and np.abs(grad_moms_numpy[event+1,rep,0]) > 0:   
                        kwargs_for_gx = {"channel": 'x', "system": system, "area": grad_moms_numpy[event+1,rep,0], "duration": event_time_numpy[event+1,rep]}
                        gx_adc = make_trapezoid(**kwargs_for_gx)
                        gx_adc_ramp = gx_adc.amplitude*gx_adc.rise_time/2
                    kwargs_for_gx = {"channel": 'x', "system": system, "area": gx_gradmom-gx_adc_ramp, "duration": dur}
                    gx = make_trapezoid(**kwargs_for_gx) 
                if np.abs(gy_gradmom)>0:
                    gy_adc_ramp = 0
                    if adc_mask_numpy[event+1] and np.abs(grad_moms_numpy[event+1,rep,1]) > 0:   
                        kwargs_for_gy = {"channel": 'y', "system": system, "area": grad_moms_numpy[event+1,rep,1], "duration": event_time_numpy[event+1,rep]}
                        gy_adc = make_trapezoid(**kwargs_for_gy)
                        gy_adc_ramp = gy_adc.amplitude*gy_adc.rise_time/2                    
                    kwargs_for_gy = {"channel": 'y', "system": system, "area": gy_gradmom-gy_adc_ramp, "duration": dur}
                    gy = make_trapezoid(**kwargs_for_gy)                              
                            
                if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                    seq.add_block(gx,gy)
                    seq.add_block(make_delay(1e-3))
                elif np.abs(gx_gradmom) > 0:
                    seq.add_block(gx)
                    seq.add_block(make_delay(1e-3))
                elif np.abs(gy_gradmom) > 0:
                    seq.add_block(gy)
                    seq.add_block(make_delay(1e-3))
                else:
                    seq.add_block(make_delay(dur))
                    seq.add_block(make_delay(1e-3))
            else: #adc mask == 1
                if adc_start == 1:
                    pass
                else:
                    adc_start = 1
                    idx_T = np.nonzero(adc_mask_numpy)[0]                    
                    dur = np.sum(event_time_numpy[idx_T,rep],0)
    
                    gx_gradmom = np.sum(grad_moms_numpy[idx_T,rep,0],0)                                           
                    kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": np.sum(grad_moms_numpy[idx_T,rep,0],0), "flat_time": dur}
                    gx = make_trapezoid(**kwargs_for_gx) 
                    
                    gy_gradmom = np.sum(grad_moms_numpy[idx_T,rep,1],0)             
                    kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": np.sum(grad_moms_numpy[idx_T,rep,1],0), "flat_time": dur}
                    gy = make_trapezoid(**kwargs_for_gy)   

                    # calculate correct delay to have same starting point of flat top
                    x_delay = np.max([0,gy.rise_time-gx.rise_time])+event_time_numpy[idx_T[0],rep]/2   # heuristic delay, to be checked at scanner
                    y_delay = np.max([0,gx.rise_time-gy.rise_time])+event_time_numpy[idx_T[0],rep]/2
                    
                    # adc gradient events are overwritten with correct delays
                    kwargs_for_gx = {"channel": 'x', "system": system,"delay":x_delay, "flat_area": np.sum(grad_moms_numpy[idx_T,rep,0],0), "flat_time": dur}
                    gx = make_trapezoid(**kwargs_for_gx)                      
                    kwargs_for_gy = {"channel": 'y', "system": system,"delay":y_delay, "flat_area": np.sum(grad_moms_numpy[idx_T,rep,1],0), "flat_time": dur}
                    gy = make_trapezoid(**kwargs_for_gy)                       
                    
                    
                    adc_delay = np.max([gx.rise_time,gy.rise_time])
                    kwargs_for_adc = {"num_samples": idx_T.size, "duration": dur, "delay":(adc_delay), "phase_offset": rf_ex.phase_offset - np.pi/4}
                    adc = make_adc(**kwargs_for_adc)    
                    
                    # dont play zero grads (cant even do FID otherwise)
                    if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                        seq.add_block(gx,gy,adc)
                    elif np.abs(gx_gradmom) > 0:
                        seq.add_block(gx,adc)
                    elif np.abs(gy_gradmom) > 0:
                        seq.add_block(gy,adc)
                    else:
                        seq.add_block(adc)

                
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)

def pulseq_write_super_extended_trap(seq_params, seq_fn, plot_seq=False):
    flips_numpy, event_time_numpy, grad_moms_numpy_input, adc_mask_numpy = seq_params
    
    flips_numpy = rectify_flips(flips_numpy)
    event_time_numpy = np.abs(event_time_numpy)
    
    NRep = flips_numpy.shape[1]
    T = flips_numpy.shape[0]
    
    # save pulseq definition
    MAXSLEW = 140
    
    
    deltak = 1.0 / FOV()
    grad_moms_numpy = deltak*grad_moms_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
#    system = Opts(max_grad=30, grad_unit='mT/m', max_slew=170, slew_unit='T/m/s', rf_ringdown_time=100e-6,
#              rf_dead_time=100e-6, adc_dead_time=10e-6)
    seq = Sequence(system)   
    
    seq.add_block(make_delay(2.0))
    
    
    for rep in range(NRep):
        adc_start = 0
        for event in range(T):
            ###############################
            ## global pulse
            if adc_mask_numpy[event] == 0:  
                RFdur=0
                if flips_numpy[event,rep,3] == 0:
                        RFdur=0
                        if np.abs(flips_numpy[event,rep,0]) > 1e-8:
                
                            # alternatively slice selective:
                            RFdur = 1*1e-3
                            #RFdur = event_time_numpy[idx_T,rep]
                            kwargs_for_block = {"flip_angle": flips_numpy[event,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[event,rep,1]}
                            rf,_ = make_block_pulse(**kwargs_for_block)
                            
                            seq.add_block(rf)  
                            seq.add_block(make_delay(1e-4))
                                                
                elif flips_numpy[event,rep,3] == 1:
                    RFdur = 0
                    if np.abs(flips_numpy[event,rep,0]) > 1e-8:
                        use = "excitation"
                        
                        if nonsel:
                            RFdur = 1*1e-3
                            kwargs_for_block = {"flip_angle": flips_numpy[event,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[event,rep,1]}
                            rf_ex,_ = make_block_pulse(**kwargs_for_block)
                            seq.add_block(rf_ex)     
                        else:
                            # alternatively slice selective:
                            use = "excitation"
                            
                            kwargs_for_sinc = {"flip_angle": flips_numpy[event,rep,0], "system": system, "duration": 1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flips_numpy[event,rep,1]}
                            rf_ex, gz, gzr= make_sinc_pulse(**kwargs_for_sinc)
                            seq.add_block(rf_ex, gz)
                            gzr.amplitude=gzr.amplitude
                            seq.add_block(gzr)
                            RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
                            

                
                elif flips_numpy[event,rep,3] == 2:
                    ###############################
                    ### refocusing pulse
                    use = "refocusing"
                    
                    RFdur = 0
                    
                    if np.abs(flips_numpy[event,rep,0]) > 1e-8:
                      RFdur = 1*1e-3
                      
                      if nonsel:
                          kwargs_for_block = {"flip_angle": flips_numpy[event,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[event,rep,1]}
                          rf_ref,_ = make_block_pulse(**kwargs_for_block)
                          seq.add_block(rf_ref)         
                      else:
                        
                          kwargs_for_sinc = {"flip_angle": flips_numpy[event,rep,0], "system": system, "duration": 1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flips_numpy[event,rep,1]}
                          rf_ref, gz_ref,gzr = make_sinc_pulse(**kwargs_for_sinc)
                          seq.add_block(rf_ref, gz_ref)
                          RFdur = gz_ref.rise_time + gz_ref.flat_time + gz_ref.fall_time 
        
                dur = event_time_numpy[event,rep] - RFdur
                if dur < 0:
                    raise Exception('Event Time too short! Event Time: Rep: '+ str(rep) + ', Event: ' +str(event),', increase event_time by at least: ' + str(-dur))
                
                gx_gradmom = grad_moms_numpy[event,rep,0]
                gy_gradmom = grad_moms_numpy[event,rep,1]


                if np.abs(gx_gradmom)>0:
                    kwargs_for_gx = {"channel": 'x', "system": system, "area": gx_gradmom, "duration": dur}
                    gx = make_trapezoid(**kwargs_for_gx)
                    if adc_mask_numpy[event+1] and np.abs(grad_moms_numpy[event+1,rep,0]) > 0:   

                        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": grad_moms_numpy[event+1,rep,0], "flat_time": event_time_numpy[event+1,rep]}
                        gx_adc = make_trapezoid(**kwargs_for_gx)
                        gpre_adc_fall_time = gx.fall_time * gx.amplitude / (gx.amplitude+gx_adc.amplitude)
                        gpre_adc_fall_time = gx.fall_time
                        gpre_adc_times = [0, gx.rise_time, gx.rise_time + gx.flat_time,
                        gx.rise_time + gx.flat_time + gpre_adc_fall_time]
                        gpre_adc_amp = [0, gx.amplitude, gx.amplitude, gx_adc.amplitude]
                        gx = make_extended_trapezoid(channel='x', times=gpre_adc_times, amplitudes=gpre_adc_amp)
                if np.abs(gy_gradmom)>0:
                    kwargs_for_gy = {"channel": 'y', "system": system, "area": gy_gradmom, "duration": dur}
                    gy = make_trapezoid(**kwargs_for_gy)
                    if adc_mask_numpy[event+1] and np.abs(grad_moms_numpy[event+1,rep,1]) > 0:
                        kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": grad_moms_numpy[event+1,rep,1], "flat_time": event_time_numpy[event+1,rep]}
                        gy_adc = make_trapezoid(**kwargs_for_gy)
                        gpre_adc_fall_time = gy.fall_time * gy.amplitude / (gy.amplitude+gy_adc.amplitude)
                        gpre_adc_fall_time = gy.fall_time
                        gpre_adc_times = [0, gy.rise_time, gy.rise_time + gy.flat_time,
                        gy.rise_time + gy.flat_time + gpre_adc_fall_time]
                        gpre_adc_amp = [0, gy.amplitude, gy.amplitude, gy_adc.amplitude]
                        gy = make_extended_trapezoid(channel='y', times=gpre_adc_times, amplitudes=gpre_adc_amp)                                        
                            
                if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                    seq.add_block(gx,gy)
                elif np.abs(gx_gradmom) > 0:
                    seq.add_block(gx)
                elif np.abs(gy_gradmom) > 0:
                    seq.add_block(gy)
                else:
                    seq.add_block(make_delay(dur))             
            else: #adc mask == 1
                if adc_start == 1:
                    pass
                else:
                    adc_start = 1
                    idx_T = np.nonzero(adc_mask_numpy)[0]                    
                    dur = np.sum(event_time_numpy[idx_T,rep],0)

                    gx_gradmom = np.sum(grad_moms_numpy[idx_T,rep,0],0)
                    kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": np.sum(grad_moms_numpy[idx_T,rep,0],0), "flat_time": dur}
                    gx = make_trapezoid(**kwargs_for_gx) 
                    temp_rise_time = gx.rise_time
                    if np.abs(gx_gradmom)>0 and np.abs(grad_moms_numpy[event-1,rep,0]) > 0:
                        gr_times = [0, dur, dur+gx.fall_time]
                        gr_amp = [gx_adc.amplitude, gx_adc.amplitude,0]
                        gx = make_extended_trapezoid(channel='x', times=gr_times, amplitudes=gr_amp)
                    
                    gy_gradmom = np.sum(grad_moms_numpy[idx_T,rep,1],0)
                    kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": np.sum(grad_moms_numpy[idx_T,rep,1],0), "flat_time": dur}
                    gy = make_trapezoid(**kwargs_for_gy)   
                    if np.abs(gy_gradmom)>0 and np.abs(grad_moms_numpy[event-1,rep,1]) > 0:
                        gr_times = [0, dur, dur+gy.fall_time]
                        gr_amp = [gy_adc.amplitude, gy_adc.amplitude,0]
                        gy = make_extended_trapezoid(channel='y', times=gr_times, amplitudes=gr_amp)
                    
                    #TODO: gy.rise_time to be added to adc delay
                    kwargs_for_adc = {"num_samples": idx_T.size, "duration": dur, "delay":(-event_time_numpy[idx_T[0],rep]/2), "phase_offset": rf_ex.phase_offset - np.pi/4}
                    adc = make_adc(**kwargs_for_adc)    
                    
                    # dont play zero grads (cant even do FID otherwise)
                    if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                        seq.add_block(gx,gy,adc)
                    elif np.abs(gx_gradmom) > 0:
                        seq.add_block(gx,adc)
                    elif np.abs(gy_gradmom) > 0:
                        seq.add_block(gy,adc)
                    else:
                        seq.add_block(adc)                    
                
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)

def pulseq_write_GRE(seq_params, seq_fn, plot_seq=False):
    flips_numpy, event_time_numpy, grad_moms_numpy_input = seq_params
    
    event_time_numpy = np.abs(event_time_numpy)
    flips_numpy = rectify_flips(flips_numpy)
    
    NRep = flips_numpy.shape[1]
    
    # save pulseq definition
    MAXSLEW = 140


    
    deltak = 1.0 / FOV()
    grad_moms_numpy = deltak*grad_moms_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)    
    
    seq.add_block(make_delay(4.0))
    

      

    for rep in range(NRep):
        
        ###############################
        ###              first action
        idx_T = 0
        RFdur = 0
        if np.abs(flips_numpy[idx_T,rep,0]) > 1e-16:
            use = "excitation"
            
            if nonsel:
                RFdur = 1*1e-3
                kwargs_for_block = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[idx_T,rep,1]}
                rf,_ = make_block_pulse(**kwargs_for_block)
                seq.add_block(rf)
#                kwargs_for_sinc = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness":  50e-3, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flips_numpy[idx_T,rep,1]}
#                rf, gz, gzr = make_sinc_pulse(kwargs_for_sinc, 3)
#                seq.add_block(rf,gz)
            else:
                # alternatively slice selective:
                kwargs_for_sinc = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flips_numpy[idx_T,rep,1]}
                rf, gz, gzr = make_sinc_pulse(**kwargs_for_sinc)
                seq.add_block(rf, gz)
                seq.add_block(gzr)            
                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
            
#        seq.add_block(make_delay(0.002-RFdur))
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))
         
        
        ###############################
        ###              secoond action
        idx_T = 1
        
        # calculated here, update in next event
        gradmom_rewinder = np.squeeze(grad_moms_numpy[idx_T,rep,:])
        eventtime_rewinder = np.squeeze(event_time_numpy[idx_T,rep])
        
        ###############################
        ###              line acquisition T(3:end-1)
        idx_T = np.arange(2, grad_moms_numpy.shape[0] - 2) # T(2)
        dur = np.sum(event_time_numpy[idx_T,rep])
        
        gx_gradmom = np.sum(grad_moms_numpy[idx_T,rep,0],0)
        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
        gx = make_trapezoid(**kwargs_for_gx)    
        
        gy_gradmom = np.sum(grad_moms_numpy[idx_T,rep,1],0)
        kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": dur}
        gy = make_trapezoid(**kwargs_for_gy)
        
        print(gx.rise_time - event_time_numpy[idx_T[0],rep]/2)
        kwargs_for_adc = {"num_samples": idx_T.size, "duration": gx.flat_time, "delay": (gx.rise_time - event_time_numpy[idx_T[0],rep]/2), "phase_offset": rf.phase_offset - np.pi/4}
        adc = make_adc(**kwargs_for_adc)    
        
        #update rewinder for gxgy ramp times, from second event
        kwargs_for_gxpre = {"channel": 'x', "system": system, "area": gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
        gx_pre = make_trapezoid(**kwargs_for_gxpre)
        
        kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1]-gy.amplitude*gy.rise_time/2, "duration": eventtime_rewinder}
        gy_pre = make_trapezoid(**kwargs_for_gypre)
        
        # dont play zero grads (cant even do FID otherwise)
        if np.abs(gradmom_rewinder[0]) > 0 or np.abs(gradmom_rewinder[1]) > 0:
            seq.add_block(gx_pre, gy_pre)
        
        seq.add_block(make_delay(1e-3))
        
        # dont play zero grads (cant even do FID otherwise)
        if np.abs(gx_gradmom) > 0 or np.abs(gy_gradmom) > 0:
            seq.add_block(gx,gy,adc)
        else:
            seq.add_block(adc)
        
        
        ###############################
        ###     second last extra event  T(end)  # adjusted also for fallramps of ADC
        idx_T = grad_moms_numpy.shape[0] - 2     # T(2)
        
        kwargs_for_gxpost = {"channel": 'x', "system": system, "area": grad_moms_numpy[idx_T,rep,0]-gx.amplitude*gx.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gx_post = make_trapezoid(**kwargs_for_gxpost)  
        
        kwargs_for_gypost = {"channel": 'y', "system": system, "area": grad_moms_numpy[idx_T,rep,1]-gy.amplitude*gy.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gy_post = make_trapezoid(**kwargs_for_gypost)  
        
        # dont play zero grads (cant even do FID otherwise)
        if np.abs(grad_moms_numpy[idx_T,rep,0]) > 0 or np.abs(grad_moms_numpy[idx_T,rep,1]) > 0:
            seq.add_block(gx_post, gy_post)
        
        ###############################
        ###     last extra event  T(end)
        idx_T = grad_moms_numpy.shape[0] - 1 # T(2)
        
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
    
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)

def pulseq_write_GRE_DREAM(seq_params, seq_fn, plot_seq=False):
    flips_numpy, event_time_numpy, grad_moms_numpy_input = seq_params
    
    event_time_numpy = np.abs(event_time_numpy)
    flips_numpy = rectify_flips(flips_numpy)
    
    NRep = flips_numpy.shape[1]
    
    # save pulseq definition
    MAXSLEW = 140

    
    deltak = 1.0 / FOV()
    grad_moms_numpy = deltak*grad_moms_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)    
    
    seq.add_block(make_delay(4.0))
    

      

    for rep in range(NRep):
        
        ###############################
        ###              first action
        idx_T = 0
        RFdur=0
        if np.abs(flips_numpy[idx_T,rep,0]) > 1e-8:

            RFdur = 1*1e-3
            #RFdur = event_time_numpy[idx_T,rep]
            kwargs_for_block = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[idx_T,rep,1]}
            rf,_ = make_block_pulse(**kwargs_for_block)
            
            seq.add_block(rf)  
            seq.add_block(make_delay(1e-4))
            
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))    
            
        ###              second action
        idx_T = 1
        RFdur=0
        dur = event_time_numpy[idx_T,rep]
        if np.abs(flips_numpy[idx_T,rep,0]) > 1e-8:
            RFdur = 1*1e-3
            kwargs_for_block = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[idx_T,rep,1]}
            rf,_ = make_block_pulse(**kwargs_for_block)
            
            seq.add_block(rf) 
            seq.add_block(make_delay(1e-4))
            dur = event_time_numpy[idx_T,rep]-RFdur
            
        if np.abs(grad_moms_numpy[idx_T,rep,0])>0:
            gx_gradmom = grad_moms_numpy[idx_T,rep,0]
            kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
            gx = make_trapezoid(**kwargs_for_gx)    
            gy_gradmom = grad_moms_numpy[idx_T,rep,1]
            kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": dur}
            gy = make_trapezoid(**kwargs_for_gy)
            seq.add_block(gx,gy)
            seq.add_block(make_delay(1e-4))
        else:
            seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))

        ###              third action
        idx_T = 2
        RFdur=0
        dur = event_time_numpy[idx_T,rep]
        if np.abs(flips_numpy[idx_T,rep,0]) > 1e-8:

            # alternatively slice selective:
            RFdur = 1*1e-3
            kwargs_for_block = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[idx_T,rep,1]}
            rf,_ = make_block_pulse(**kwargs_for_block)
            
            seq.add_block(rf) 
            # seq.add_block(make_delay(1e-4))
            
            dur = event_time_numpy[idx_T,rep]-RFdur
            
        if np.abs(grad_moms_numpy[idx_T,rep,0])>0:
            gx_gradmom = grad_moms_numpy[idx_T,rep,0]
            kwargs_for_gx = {"channel": 'x', "system": system, "area": gx_gradmom, "duration": dur}
            gx = make_trapezoid(**kwargs_for_gx)    
            gy_gradmom = grad_moms_numpy[idx_T,rep,1]
            kwargs_for_gy = {"channel": 'y', "system": system, "area": gy_gradmom, "duration": dur}
            gy = make_trapezoid(**kwargs_for_gy)   
            seq.add_block(gx,gy)
            # seq.add_block(make_delay(1e-4))
            print(dur)
        else:
            seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))
  
        ###              first readout (action 3 now)        
        idx_T = 3
        RFdur = 0
        if np.abs(flips_numpy[idx_T,rep,0]) > 1e-16:
            use = "excitation"
            
            if nonsel:
                RFdur = 1*1e-3
                kwargs_for_block = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[idx_T,rep,1]}
                rf,_ = make_block_pulse(**kwargs_for_block)
                seq.add_block(rf)
#                kwargs_for_sinc = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness":  50e-3, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flips_numpy[idx_T,rep,1]}
#                rf, gz, gzr = make_sinc_pulse(kwargs_for_sinc, 3)
#                seq.add_block(rf,gz)
            else:
                # alternatively slice selective:
                kwargs_for_sinc = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flips_numpy[idx_T,rep,1]}
                rf, gz, gzr = make_sinc_pulse(**kwargs_for_sinc)
                seq.add_block(rf, gz)
                seq.add_block(gzr)            
                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
            
#        seq.add_block(make_delay(0.002-RFdur))
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))
         
        
        ###############################
        ###              secoond readout action
        idx_T = 4
        
        # calculated here, update in next event
        gradmom_rewinder = np.squeeze(grad_moms_numpy[idx_T,rep,:])
        eventtime_rewinder = np.squeeze(event_time_numpy[idx_T,rep])
        
        ###############################
        ###              line acquisition T(5:end-2)
        idx_T = np.arange(5, grad_moms_numpy.shape[0] - 2) # T(2)
        dur = np.sum(event_time_numpy[idx_T,rep])
        
        gx_gradmom = np.sum(grad_moms_numpy[idx_T,rep,0],0)
        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
        gx = make_trapezoid(**kwargs_for_gx)    
        
        gy_gradmom = np.sum(grad_moms_numpy[idx_T,rep,1],0)
        kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": dur}
        gy = make_trapezoid(**kwargs_for_gy)
        
        kwargs_for_adc = {"num_samples": idx_T.size, "duration": gx.flat_time, "delay": (gx.rise_time), "phase_offset": rf.phase_offset - np.pi/4}
        adc = make_adc(**kwargs_for_adc)    
        
        #update rewinder for gxgy ramp times, from second event
        kwargs_for_gxpre = {"channel": 'x', "system": system, "area": -grad_moms_numpy[idx_T[0],rep,0]/2 + gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
        gx_pre = make_trapezoid(**kwargs_for_gxpre)
        
        kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1]-gy.amplitude*gy.rise_time/2, "duration": eventtime_rewinder}
        gy_pre = make_trapezoid(**kwargs_for_gypre)
        
        # dont play zero grads (cant even do FID otherwise)
        if np.abs(gradmom_rewinder[0]) > 0 or np.abs(gradmom_rewinder[1]) > 0:
            seq.add_block(gx_pre, gy_pre)
        
        seq.add_block(make_delay(1e-3))
        
        # dont play zero grads (cant even do FID otherwise)
        if np.abs(gx_gradmom) > 0 or np.abs(gy_gradmom) > 0:
            seq.add_block(gx,gy,adc)
        else:
            seq.add_block(adc)
        
        
        ###############################
        ###     second last extra event  T(end)  # adjusted also for fallramps of ADC
        idx_T = grad_moms_numpy.shape[0] - 2     # T(2)
        
        kwargs_for_gxpost = {"channel": 'x', "system": system, "area": grad_moms_numpy[idx_T,rep,0]-gx.amplitude*gx.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gx_post = make_trapezoid(**kwargs_for_gxpost)  
        
        kwargs_for_gypost = {"channel": 'y', "system": system, "area": grad_moms_numpy[idx_T,rep,1]-gy.amplitude*gy.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gy_post = make_trapezoid(**kwargs_for_gypost)  
        
        # dont play zero grads (cant even do FID otherwise)
        if np.abs(grad_moms_numpy[idx_T,rep,0]) > 0 or np.abs(grad_moms_numpy[idx_T,rep,1]) > 0:
            seq.add_block(gx_post, gy_post)
        
        ###############################
        ###     last extra event  T(end)
        idx_T = grad_moms_numpy.shape[0] - 1 # T(2)
        
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
    
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)
        
def pulseq_write_RARE(seq_params, seq_fn, plot_seq=False):
    flips_numpy, event_time_numpy, grad_moms_numpy_input = seq_params
    
    flips_numpy = rectify_flips(flips_numpy)
    
    NRep = flips_numpy.shape[1]
    
    # save pulseq definition
    MAXSLEW = 140
    
    
    deltak = 1.0 / FOV()
    grad_moms_numpy = deltak*grad_moms_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)   
    
    seq.add_block(make_delay(1.48))
    

    for rep in range(NRep):
        
        ###############################
        ###              first action
        idx_T = 0
        RFdur = 0
        if np.abs(flips_numpy[idx_T,rep,0]) > 1e-8:
            use = "excitation"
            
            if nonsel:
                RFdur = 1*1e-3
                kwargs_for_block = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[idx_T,rep,1]}
                rf_ex,_ = make_block_pulse(**kwargs_for_block)
                
                seq.add_block(rf_ex)     
            else:
                # alternatively slice selective:
                use = "excitation"
                
                
                kwargs_for_sinc = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flips_numpy[idx_T,rep,1]}
                rf_ex, gz, gzr= make_sinc_pulse(**kwargs_for_sinc)
                seq.add_block(rf_ex, gz)
                gzr.amplitude=gzr.amplitude
                seq.add_block(gzr)
                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
                
            kwargs_for_gxPre90 = {"channel": 'x', "system": system, "area": grad_moms_numpy[idx_T,rep,0], "duration": event_time_numpy[idx_T,rep]-RFdur}
            gxPre90 = make_trapezoid(**kwargs_for_gxPre90) 
            
            seq.add_block(gxPre90)
        else:
            seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
            
        ###############################
        ###              secoond action
        idx_T = 1
        use = "refocusing"
        
        RFdur = 0
        
        if np.abs(flips_numpy[idx_T,rep,0]) > 1e-8:
          RFdur = 1*1e-3
          
          if nonsel:
              kwargs_for_block = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[idx_T,rep,1]}
              rf_ref,_ = make_block_pulse(**kwargs_for_block)
              seq.add_block(rf_ref)         
          else:
            
              kwargs_for_sinc = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flips_numpy[idx_T,rep,1]}
              rf_ref, gz_ref,gzr = make_sinc_pulse(**kwargs_for_sinc)
              seq.add_block(rf_ref, gz_ref)
              RFdur = gz_ref.rise_time + gz_ref.flat_time + gz_ref.fall_time 
              

        # calculated here, update in next event
        gradmom_rewinder = np.squeeze(grad_moms_numpy[idx_T,rep,:])
        eventtime_rewinder = 1.5*1e-3 
        
        delay_after_rev=np.squeeze(event_time_numpy[idx_T,rep]-RFdur-eventtime_rewinder)
        #print([event_time_numpy[idx_T,rep]], RFdur, eventtime_rewinder)
        
        ###############################
        ###              line acquisition T(3:end-1)
        idx_T = np.arange(2, grad_moms_numpy.shape[0] - 2) # T(2)
        
        dur = np.sum(event_time_numpy[idx_T,rep])
        
        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": np.sum(grad_moms_numpy[idx_T,rep,0],0), "flat_time": dur}
        gx = make_trapezoid(**kwargs_for_gx)   
        
        kwargs_for_adc = {"num_samples": idx_T.size, "duration": gx.flat_time, "delay": gx.rise_time - event_time_numpy[idx_T[0],rep]/2, "phase_offset": rf_ex.phase_offset - np.pi/4}
        adc = make_adc(**kwargs_for_adc)    
        
        #update rewinder for gxgy ramp times, from second event
        kwargs_for_gxpre = {"channel": 'x', "system": system, "area": gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
        gx_pre = make_trapezoid(**kwargs_for_gxpre)
        
        kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1], "duration": eventtime_rewinder}
        gy_pre = make_trapezoid(**kwargs_for_gypre)
    
        seq.add_block(gx_pre, gy_pre)
                    
#        if delay_after_rev < 0:
#            import pdb; pdb.set_trace()
        seq.add_block(make_delay(delay_after_rev))   
        seq.add_block(gx,adc)    
        
        ###############################
        ###     second last extra event  T(end)  # adjusted also for fallramps of ADC
        idx_T = grad_moms_numpy.shape[0] - 2     # T(2)
        
        kwargs_for_gxpost = {"channel": 'x', "system": system, "area": grad_moms_numpy[idx_T,rep,0]-gx.amplitude*gx.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gx_post = make_trapezoid(**kwargs_for_gxpost)  
        
        kwargs_for_gypost = {"channel": 'y', "system": system, "area": grad_moms_numpy[idx_T,rep,1], "duration": event_time_numpy[idx_T,rep]}
        gy_post = make_trapezoid(**kwargs_for_gypost)  
        
        seq.add_block(gx_post, gy_post)
          
        ###############################
        ###     last extra event  T(end)
        idx_T = grad_moms_numpy.shape[0] - 1 # T(2)
        
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
        
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)
    
def pulseq_write_BSSFP(seq_params, seq_fn, plot_seq=False):
    flips_numpy, event_time_numpy, grad_moms_numpy_input = seq_params
    
    flips_numpy = rectify_flips(flips_numpy)
    
    NRep = flips_numpy.shape[1]
    
    # save pulseq definition
    MAXSLEW = 140
    
    deltak = 1.0 / FOV()
    grad_moms_numpy = deltak*grad_moms_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)    
    
    seq.add_block(make_delay(2.0))
    
   
    for rep in range(NRep):
        
        ###############################
        ###              first action
        idx_T = 0
        if np.abs(flips_numpy[idx_T,rep,0]) > 1e-8:
            
            if nonsel:

                use = "excitation"
                
                # alternatively slice selective:
                RFdur = 0.8*1e-3
                kwargs_for_block = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[idx_T,rep,1]}
                rf,_ = make_block_pulse(**kwargs_for_block)
                
                seq.add_block(rf)     
            else:
                # alternatively slice selective:
                use = "excitation"
                
                
                # alternatively slice selective:
                kwargs_for_sinc = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "phase_offset": flips_numpy[idx_T,rep,1], "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4}
                rf, gz, gzr = make_sinc_pulse(**kwargs_for_sinc)
                
                seq.add_block(rf, gz)
                seq.add_block(gzr)            
                
                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
                
        
#        seq.add_block(make_delay(0.002-RFdur))
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))
        
        ###############################
        ###              secoond action
        idx_T = 1
        
        # calculated here, update in next event
        gradmom_rewinder = np.squeeze(grad_moms_numpy[idx_T,rep,:])
        eventtime_rewinder = np.squeeze(event_time_numpy[idx_T,rep])
        
        ###############################
        ###              line acquisition T(3:end-1)
        idx_T = np.arange(2, grad_moms_numpy.shape[0] - 2) # T(2)
        dur = np.sum(event_time_numpy[idx_T,rep])
        
        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": np.sum(grad_moms_numpy[idx_T,rep,0],0), "flat_time": dur}
        gx = make_trapezoid(**kwargs_for_gx)    
        
        kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": np.sum(grad_moms_numpy[idx_T,rep,1],0), "flat_time": dur}
        gy = make_trapezoid(**kwargs_for_gy)
        
        kwargs_for_adc = {"num_samples": idx_T.size, "duration": gx.flat_time, "delay": gx.rise_time - event_time_numpy[idx_T[0],rep]/2, "phase_offset": rf.phase_offset - np.pi/4}
        adc = make_adc(**kwargs_for_adc)    
        
        #update rewinder for gxgy ramp times, from second event
        kwargs_for_gxpre = {"channel": 'x', "system": system, "area": gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
        gx_pre = make_trapezoid(**kwargs_for_gxpre)
        
        kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1]-gy.amplitude*gy.rise_time/2, "duration": eventtime_rewinder}
        gy_pre = make_trapezoid(**kwargs_for_gypre)
    
        seq.add_block(gx_pre, gy_pre)
        seq.add_block(gx,gy,adc)
        
        ###############################
        ###     second last extra event  T(end)  # adjusted also for fallramps of ADC
        idx_T = grad_moms_numpy.shape[0] - 2     # T(2)
        
        kwargs_for_gxpost = {"channel": 'x', "system": system, "area": grad_moms_numpy[idx_T,rep,0]-gx.amplitude*gx.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gx_post = make_trapezoid(**kwargs_for_gxpost)  
        
        kwargs_for_gypost = {"channel": 'y', "system": system, "area": grad_moms_numpy[idx_T,rep,1]-gy.amplitude*gy.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gy_post = make_trapezoid(**kwargs_for_gypost)  
        
        if nonsel:
            seq.add_block(gx_post, gy_post)
        else:
            seq.add_block(gx_post, gy_post, gzr)
            
        
        ###############################
        ###     last extra event  T(end)
        idx_T = grad_moms_numpy.shape[0] - 1 # T(2)
        
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
    
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)    
    
def pulseq_write_slBSSFP(seq_params, seq_fn, plot_seq=False):
    flips_numpy, event_time_numpy, grad_moms_numpy_input = seq_params
    
    flips_numpy = rectify_flips(flips_numpy)
    
    NRep = flips_numpy.shape[1]
    
    # save pulseq definition
    MAXSLEW = 140
      
    
    deltak = 1.0 / FOV()
    grad_moms_numpy = deltak*grad_moms_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)    
    
    seq.add_block(make_delay(2.0))
    
    
    for rep in range(NRep):
        
        ###############################
        ###              first action
        idx_T = 0
        if np.abs(flips_numpy[idx_T,rep,0]) > 1e-8:
            slice_thickness = 200e-3     # slice

            # alternatively slice selective:
            RFdur = event_time_numpy[idx_T,rep]
            kwargs_for_block = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[idx_T,rep,1],"freq_offset": int(flips_numpy[idx_T,rep,2])}
            rf,_ = make_block_pulse(**kwargs_for_block)
            
            seq.add_block(rf)  
            seq.add_block(make_delay(1e-4))
        ###              second action
        idx_T = 1
        if np.abs(flips_numpy[idx_T,rep,0]) > 1e-8:
            slice_thickness = 200e-3     # slice
            
            # alternatively slice selective:
            RFdur = event_time_numpy[idx_T,rep]
            kwargs_for_block = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[idx_T,rep,1],"freq_offset": flips_numpy[idx_T,rep,2]}
            rf,_ = make_block_pulse(**kwargs_for_block)
            
            seq.add_block(rf)  
            seq.add_block(make_delay(1e-4))
                
  
        ###              first readout (action 2 now)            
        idx_T = 2
        RFdur=0
        if np.abs(flips_numpy[idx_T,rep,0]) > 1e-8:
            
            if nonsel:
                slice_thickness = 200e-3     # slice
                use = "excitation"
                
                # alternatively slice selective:
                RFdur = 0.8*1e-3
                kwargs_for_block = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": flips_numpy[idx_T,rep,1]}
                rf,_ = make_block_pulse(**kwargs_for_block)
                
                seq.add_block(rf)     
            else:
                # alternatively slice selective:
                use = "excitation"
                
                slice_thickness = 5e-3
                
                # alternatively slice selective:
                kwargs_for_sinc = {"flip_angle": flips_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "phase_offset": flips_numpy[idx_T,rep,1], "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4}
                rf, gz, gzr = make_sinc_pulse(**kwargs_for_sinc)
                
                seq.add_block(rf, gz)
                seq.add_block(gzr)            
                
                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
                
        
#        seq.add_block(make_delay(0.002-RFdur))
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))
        
        ###############################
        ###              secoond action
        idx_T = 3
        
        # calculated here, update in next event
        gradmom_rewinder = np.squeeze(grad_moms_numpy[idx_T,rep,:])
        eventtime_rewinder = np.squeeze(event_time_numpy[idx_T,rep])
        
        ###############################
        ###              line acquisition T(3:end-1)
        idx_T = np.arange(4, grad_moms_numpy.shape[0] - 2) # T(2)
        dur = np.sum(event_time_numpy[idx_T,rep])
        
        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": np.sum(grad_moms_numpy[idx_T,rep,0],0), "flat_time": dur}
        gx = make_trapezoid(**kwargs_for_gx)    
        
        kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": np.sum(grad_moms_numpy[idx_T,rep,1],0), "flat_time": dur}
        gy = make_trapezoid(**kwargs_for_gy)
        
        kwargs_for_adc = {"num_samples": idx_T.size, "duration": gx.flat_time, "delay": gx.rise_time - event_time_numpy[idx_T[0],rep]/2, "phase_offset": rf.phase_offset - np.pi/4}
        adc = make_adc(**kwargs_for_adc)    
        
        #update rewinder for gxgy ramp times, from second event
        kwargs_for_gxpre = {"channel": 'x', "system": system, "area": gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
        gx_pre = make_trapezoid(**kwargs_for_gxpre)
        
        kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1]-gy.amplitude*gy.rise_time/2, "duration": eventtime_rewinder}
        gy_pre = make_trapezoid(**kwargs_for_gypre)
    
        seq.add_block(gx_pre, gy_pre)
        seq.add_block(gx,gy,adc)
        
        ###############################
        ###     second last extra event  T(end)  # adjusted also for fallramps of ADC
        idx_T = grad_moms_numpy.shape[0] - 2     # T(2)
        
        kwargs_for_gxpost = {"channel": 'x', "system": system, "area": grad_moms_numpy[idx_T,rep,0]-gx.amplitude*gx.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gx_post = make_trapezoid(**kwargs_for_gxpost)  
        
        kwargs_for_gypost = {"channel": 'y', "system": system, "area": grad_moms_numpy[idx_T,rep,1]-gy.amplitude*gy.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gy_post = make_trapezoid(**kwargs_for_gypost)  
        
        if nonsel:
            seq.add_block(gx_post, gy_post)
        else:
            seq.add_block(gx_post, gy_post, gzr)
            
        
        ###############################
        ###     last extra event  T(end)
        idx_T = grad_moms_numpy.shape[0] - 1 # T(2)
        
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
    
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)    
    
def pulseq_write_EPI(seq_params, seq_fn, plot_seq=False):
    raise
    pass
    

        
def append_header(seq_fn, FOV,slice_thickness):
    # append version and definitions
    # with open(r"\\141.67.249.47\MRTransfer\mrzero_src\.git\ORIG_HEAD") as file:
    #     git_version = file.read()    
    with open(seq_fn, 'r') as fin:
        lines = fin.read().splitlines(True)

    updated_lines = []
    updated_lines.append("# Pulseq sequence file\n")
    updated_lines.append("# Created by MRIzero/IMR/GPI pulseq converter\n")
    # updated_lines.append('# MRZero Version: 0.5, git hash: ' + git_version)
    updated_lines.append("# experiment_id: "+seq_fn.split('\\')[-2]+"\n")
    updated_lines.append('#' + seq_fn + "\n")
    updated_lines.append("\n")
    # updated_lines.append("[VERSION]\n")
    # updated_lines.append("major 1\n")
    # updated_lines.append("minor 2\n")   
    # updated_lines.append("revision 1\n")  
    # updated_lines.append("\n")    
    updated_lines.append("[DEFINITIONS]\n")
    updated_lines.append("FOV "+str(round(FOV*1e3))+" "+str(round(FOV*1e3))+" "+str(round(slice_thickness*1e3))+" \n")   
    updated_lines.append("\n")    
    
    updated_lines.extend(lines[3:])

    with open(seq_fn, 'w') as fout:
        fout.writelines(updated_lines)    
        
    
    
#def make_sinc_pulse(kwargs, nargout=1):
#    """
#    Makes a Holder object for an RF pulse Event.
#
#    Parameters
#    ----------
#    kwargs : dict
#        Key value mappings of RF Event parameters_params and values.
#    nargout: int
#        Number of output arguments to be returned. Default is 1, only RF Event is returned. Passing any number greater
#        than 1 will return the Gz Event along with the RF Event.
#
#    Returns
#    -------
#    rf : Holder
#        RF Event configured based on supplied kwargs.
#    gz : Holder
#        Slice select trapezoidal gradient Event.
#    """
#
#    flip_angle = kwargs.get("flip_angle")
#    system = kwargs.get("system", Opts())
#    duration = kwargs.get("duration", 0)
#    freq_offset = kwargs.get("freq_offset", 0)
#    phase_offset = kwargs.get("phase_offset", 0)
#    time_bw_product = kwargs.get("time_bw_product", 4)
#    apodization = kwargs.get("apodization", 0)
#    max_grad = kwargs.get("max_grad", 0)
#    max_slew = kwargs.get("max_slew", 0)
#    slice_thickness = kwargs.get("slice_thickness", 0)
#
#    BW = time_bw_product / duration
#    alpha = apodization
#    N = int(round(duration / 1e-6))
#    t = np.zeros((1, N))
#    for x in range(1, N + 1):
#        t[0][x - 1] = x * system.rf_raster_time
#    tt = t - (duration / 2)
#    window = np.zeros((1, tt.shape[1]))
#    for x in range(0, tt.shape[1]):
#        window[0][x] = 1.0 - alpha + alpha * np.cos(2 * np.pi * tt[0][x] / duration)
#    signal = np.multiply(window, np.sinc(BW * tt))
#    flip = np.sum(signal) * system.rf_raster_time * 2 * np.pi
#    signal = signal * flip_angle / flip
#
#    rf = Holder()
#    rf.type = 'rf'
#    rf.signal = signal
#    rf.t = t
#    rf.freq_offset = freq_offset
#    rf.phase_offset = phase_offset
#    rf.dead_time = system.rf_dead_time
#    rf.ring_down_time = system.rf_ringdown_time
#
#    fill_time = 0
#    if nargout > 1:
#        if slice_thickness == 0:
#            raise ValueError('Slice thickness must be provided')
#
#        system.max_grad = max_grad if max_grad > 0 else system.max_grad
#        system.max_slew = max_slew if max_slew > 0 else system.max_slew
#
#        amplitude = BW / slice_thickness
#        area = amplitude * duration
#        kwargs_for_trap = {"channel": 'z', "system": system, "flat_time": duration, "flat_area": area}
#        gz = make_trapezoid(kwargs_for_trap)
#        
#
#        
#        fill_time = gz.rise_time
#        nfill_time = int(round(fill_time / 1e-6))
#        t_fill = np.zeros((1, nfill_time))
#        for x in range(1, nfill_time + 1):
#            t_fill[0][x - 1] = x * 1e-6
#        temp = np.concatenate((t_fill[0], rf.t[0] + t_fill[0][-1]))
#        temp = temp.reshape((1, len(temp)))
#        rf.t = np.resize(rf.t, temp.shape)
#        rf.t[0] = temp
#        z = np.zeros((1, t_fill.shape[1]))
#        temp2 = np.concatenate((z[0], rf.signal[0]))
#        temp2 = temp2.reshape((1, len(temp2)))
#        rf.signal = np.resize(rf.signal, temp2.shape)
#        rf.signal[0] = temp2
#        
#    if nargout > 2:        
#        centerpos = 0.5
#        gzr_area = -area*(1-centerpos)-0.5*(gz.area-area)
#        kwargs_for_trap_gzr = {"channel": 'z', "system": system, "area": gzr_area}
#        gzr = make_trapezoid(kwargs_for_trap_gzr)        
#
#    # Add dead time to start of pulse, if required
#    if fill_time < rf.dead_time:
#        fill_time = rf.dead_time - fill_time
#        t_fill = (np.arange(int(round(fill_time / 1e-6))) * 1e-6)[np.newaxis, :]
#        rf.t = np.concatenate((t_fill, (rf.t + t_fill[0, -1])), axis=1)
#        rf.signal = np.concatenate((np.zeros(t_fill.shape), rf.signal), axis=1)
#
#    if rf.ring_down_time > 0:
#        t_fill = (np.arange(1, round(rf.ring_down_time / 1e-6) + 1) * 1e-6)[np.newaxis, :]
#        rf.t = np.concatenate((rf.t, rf.t[0, -1] + t_fill), axis=1)
#        rf.signal = np.concatenate((rf.signal, np.zeros(t_fill.shape)), axis=1)
#
#    # Following 2 lines of code are workarounds for numpy returning 3.14... for np.angle(-0.00...)
#    negative_zero_indices = np.where(rf.signal == -0.0)
#    rf.signal[negative_zero_indices] = 0
#
#    if nargout > 2:
#        return rf, gz, gzr
#    elif nargout > 1:
#        return rf, gz
#    else:
#        return rf
#    
#def make_trapezoid(kwargs):
#    """
#    Makes a Holder object for an trapezoidal gradient Event.
#
#    Parameters
#    ----------
#    kwargs : dict
#        Key value mappings of trapezoidal gradient Event parameters_params and values.
#
#    Returns
#    -------
#    grad : Holder
#        Trapezoidal gradient Event configured based on supplied kwargs.
#    """
#
#    channel = kwargs.get("channel", "z")
#    system = kwargs.get("system", Opts())
#    duration = kwargs.get("duration", 0)
#    area_result = kwargs.get("area", -1)
#    flat_time_result = kwargs.get("flat_time", 0)
#    flat_area_result = kwargs.get("flat_area", -1)
#    amplitude_result = kwargs.get("amplitude", -1)
#    max_grad = kwargs.get("max_grad", 0)
#    max_slew = kwargs.get("max_slew", 0)
#    rise_time = kwargs.get("rise_time", 0)
#
#    max_grad = max_grad if max_grad > 0 else system.max_grad
#    max_slew = max_slew if max_slew > 0 else system.max_slew
#    rise_time = rise_time if rise_time > 0 else system.rise_time
#
#    if area_result == -1 and flat_area_result == -1 and amplitude_result == -1:
#        raise ValueError('Must supply either ''area'', ''flat_area'' or ''amplitude''')
#
#    if flat_time_result > 0:
#        amplitude = amplitude_result if (amplitude_result != -1) else (flat_area_result / flat_time_result)
#        if rise_time == 0:
#            rise_time = abs(amplitude) / max_slew
#            rise_time = ceil(rise_time / system.grad_raster_time) * system.grad_raster_time
#        fall_time, flat_time = rise_time, flat_time_result
#    elif duration > 0:
#        if amplitude_result != -1:
#            amplitude = amplitude_result
#        else:
#            if rise_time == 0:
#                dC = 1 / abs(2 * max_slew) + 1 / abs(2 * max_slew)
#                amplitude = (duration - sqrt(pow(duration, 2) - 4 * abs(area_result) * dC)) / (2 * dC)
#            else:
#                amplitude = area_result / (duration - rise_time)
#
#        if rise_time == 0:
#            rise_time = ceil(
#                amplitude / max_slew / system.grad_raster_time) * system.grad_raster_time
#
#        fall_time = rise_time
#        flat_time = (duration - rise_time - fall_time)
#
#        amplitude = area_result / (rise_time / 2 + fall_time / 2 + flat_time) if amplitude_result == -1 else amplitude
#    else:
#        if area_result == -1:
#            raise ValueError("makeTrapezoid:invalidArguments','Must supply area at least")
#        else:
#            #
#            # find the shortest possible duration
#            # first check if the area can be realized as a triangle
#            # if not we calculate a trapezoid
#            rise_time=ceil(sqrt(abs(area_result)/max_slew)/system.grad_raster_time)*system.grad_raster_time
#            amplitude=area_result/rise_time
#            tEff=rise_time
#            if abs(amplitude)>max_grad:
#                tEff=ceil((abs(area_result)/max_grad)/system.grad_raster_time)*system.grad_raster_time
#                amplitude=area_result/tEff
#                rise_time=ceil((abs(amplitude)/max_slew)/system.grad_raster_time)*system.grad_raster_time
#                
#            flat_time=tEff-rise_time
#            fall_time=rise_time        
#
#    if abs(amplitude) > max_grad:
#        raise ValueError("Amplitude violation")
#
#    grad = Holder()
#    grad.type = "trap"
#    grad.channel = channel
#    grad.amplitude = amplitude
#    grad.rise_time = rise_time
#    grad.flat_time = flat_time
#    grad.fall_time = fall_time
#    grad.area = amplitude * (flat_time + rise_time / 2 + fall_time / 2)
#    grad.flat_area = amplitude * flat_time
#
#    return grad    

