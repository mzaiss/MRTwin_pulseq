import numpy as np
import scipy.io as sio
from math import pi,ceil, sqrt, pow
import sys

sys.path.append("../")
from pypulseq.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import makeadc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc import make_sinc_pulse
from pypulseq.make_trap import make_trapezoid
from pypulseq.make_block import make_block_pulse
from pypulseq.opts import Opts

# for trap and sinc
from pypulseq.holder import Holder

def rectify_rf_event(rf_event):
    rrf_event = np.copy(rf_event)
    
    for i in range(rrf_event.shape[0]):
        for j in range(rrf_event.shape[1]):
            if rrf_event[i,j,0] < 0:
                rrf_event[i,j,0] = -rrf_event[i,j,0]
                rrf_event[i,j,1] += np.pi
                rrf_event[i,j,1] = np.mod(rrf_event[i,j,1], 2*np.pi)
    return rrf_event

def FOV():
    #FOV = 0.110
    return 0.200

nonsel = 0
if nonsel==1:
    slice_thickness = 200*1e-3
else:
    slice_thickness = 8e-3


def pulseq_write_GRE(seq_params, seq_fn, plot_seq=False):
    rf_event_numpy, event_time_numpy, gradm_event_numpy_input = seq_params
    
    event_time_numpy = np.abs(event_time_numpy)
    rf_event_numpy = rectify_rf_event(rf_event_numpy)
    
    NRep = rf_event_numpy.shape[1]
    
    # save pulseq definition
    MAXSLEW = 140


    
    deltak = 1.0 / FOV()
    gradm_event_numpy = deltak*gradm_event_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ring_down_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(kwargs_for_opts)
    seq = Sequence(system)    
    
    seq.add_block(make_delay(2.0))
    

      

    for rep in range(NRep):
        
        ###############################
        ###              first action
        idx_T = 0
        RFdur = 0
        if np.abs(rf_event_numpy[idx_T,rep,0]) > 1e-16:
            use = "excitation"
            
            if nonsel:
                RFdur = 1*1e-3
                kwargs_for_block = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": rf_event_numpy[idx_T,rep,1]}
                rf = make_block_pulse(kwargs_for_block, 1)
                seq.add_block(rf)
#                kwargs_for_sinc = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness":  50e-3, "apodization": 0.5, "time_bw_product": 4, "phase_offset": rf_event_numpy[idx_T,rep,1]}
#                rf, gz, gzr = make_sinc_pulse(kwargs_for_sinc, 3)
#                seq.add_block(rf,gz)
            else:
                # alternatively slice selective:
                kwargs_for_sinc = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": rf_event_numpy[idx_T,rep,1]}
                rf, gz, gzr = make_sinc_pulse(kwargs_for_sinc, 3)
                seq.add_block(rf, gz)
                seq.add_block(gzr)            
                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
            
#        seq.add_block(make_delay(0.002-RFdur))
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))
         
        
        ###############################
        ###              secoond action
        idx_T = 1
        
        # calculated here, update in next event
        gradmom_rewinder = np.squeeze(gradm_event_numpy[idx_T,rep,:])
        eventtime_rewinder = np.squeeze(event_time_numpy[idx_T,rep])
        
        ###############################
        ###              line acquisition T(3:end-1)
        idx_T = np.arange(2, gradm_event_numpy.shape[0] - 2) # T(2)
        dur = np.sum(event_time_numpy[idx_T,rep])
        
        gx_gradmom = np.sum(gradm_event_numpy[idx_T,rep,0],0)
        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
        gx = make_trapezoid(kwargs_for_gx)    
        
        gy_gradmom = np.sum(gradm_event_numpy[idx_T,rep,1],0)
        kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": dur}
        gy = make_trapezoid(kwargs_for_gy)
        
        kwargs_for_adc = {"num_samples": idx_T.size, "duration": gx.flat_time, "delay": (gx.rise_time - event_time_numpy[idx_T[0],rep]/2), "phase_offset": rf.phase_offset - np.pi/4}
        adc = makeadc(kwargs_for_adc)    
        
        #update rewinder for gxgy ramp times, from second event
        kwargs_for_gxpre = {"channel": 'x', "system": system, "area": gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
        gx_pre = make_trapezoid(kwargs_for_gxpre)
        
        kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1]-gy.amplitude*gy.rise_time/2, "duration": eventtime_rewinder}
        gy_pre = make_trapezoid(kwargs_for_gypre)
        
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
        idx_T = gradm_event_numpy.shape[0] - 2     # T(2)
        
        kwargs_for_gxpost = {"channel": 'x', "system": system, "area": gradm_event_numpy[idx_T,rep,0]-gx.amplitude*gx.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gx_post = make_trapezoid(kwargs_for_gxpost)  
        
        kwargs_for_gypost = {"channel": 'y', "system": system, "area": gradm_event_numpy[idx_T,rep,1]-gy.amplitude*gy.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gy_post = make_trapezoid(kwargs_for_gypost)  
        
        # dont play zero grads (cant even do FID otherwise)
        if np.abs(gradm_event_numpy[idx_T,rep,0]) > 0 or np.abs(gradm_event_numpy[idx_T,rep,1]) > 0:
            seq.add_block(gx_post, gy_post)
        
        ###############################
        ###     last extra event  T(end)
        idx_T = gradm_event_numpy.shape[0] - 1 # T(2)
        
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
    
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)

def pulseq_write_GRE_DREAM(seq_params, seq_fn, plot_seq=False):
    rf_event_numpy, event_time_numpy, gradm_event_numpy_input = seq_params
    
    rf_event_numpy = rectify_rf_event(rf_event_numpy)
    
    NRep = rf_event_numpy.shape[1]
    
    # save pulseq definition
    MAXSLEW = 140

    
    deltak = 1.0 / FOV()
    gradm_event_numpy = deltak*gradm_event_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ring_down_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(kwargs_for_opts)
    seq = Sequence(system)    
    
    seq.add_block(make_delay(5.0))
    

      

    for rep in range(NRep):
        
        ###############################
        ###              first action
        idx_T = 0
        RFdur=0
        if np.abs(rf_event_numpy[idx_T,rep,0]) > 1e-8:

            # alternatively slice selective:
            RFdur = event_time_numpy[idx_T,rep]
            kwargs_for_block = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": rf_event_numpy[idx_T,rep,1]}
            rf = make_block_pulse(kwargs_for_block, 1)
            
            seq.add_block(rf)  
            seq.add_block(make_delay(1e-4))
            
        ###              second action
        idx_T = 1
        if np.abs(gradm_event_numpy[idx_T,rep,0])>0:
            dur = event_time_numpy[idx_T,rep]
            gx_gradmom = gradm_event_numpy[idx_T,rep,0]
            kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
            gx = make_trapezoid(kwargs_for_gx)    
            seq.add_block(gx) 
            
        ###              third action
        idx_T = 2
        RFdur=0
        if np.abs(rf_event_numpy[idx_T,rep,0]) > 1e-8:

            # alternatively slice selective:
            RFdur = 0.4*1e-3
            kwargs_for_block = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": rf_event_numpy[idx_T,rep,1]}
            rf = make_block_pulse(kwargs_for_block, 1)
            
            seq.add_block(rf) 
            seq.add_block(make_delay(1e-4))
            
            dur = event_time_numpy[idx_T,rep]-RFdur
            gx_gradmom = gradm_event_numpy[idx_T,rep,0]
            kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
            gx = make_trapezoid(kwargs_for_gx)    
            gy_gradmom = gradm_event_numpy[idx_T,rep,1]
            kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": dur}
            gy = make_trapezoid(kwargs_for_gy)   
            seq.add_block(gx,gy) 
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))
  
        ###              first readout (action 3 now)        
        idx_T = 3
        RFdur = 0
        if np.abs(rf_event_numpy[idx_T,rep,0]) > 1e-16:
            use = "excitation"
            
            if nonsel:
                RFdur = 1*1e-3
                kwargs_for_block = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": rf_event_numpy[idx_T,rep,1]}
                rf = make_block_pulse(kwargs_for_block, 1)
                seq.add_block(rf)
#                kwargs_for_sinc = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness":  50e-3, "apodization": 0.5, "time_bw_product": 4, "phase_offset": rf_event_numpy[idx_T,rep,1]}
#                rf, gz, gzr = make_sinc_pulse(kwargs_for_sinc, 3)
#                seq.add_block(rf,gz)
            else:
                # alternatively slice selective:
                kwargs_for_sinc = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": rf_event_numpy[idx_T,rep,1]}
                rf, gz, gzr = make_sinc_pulse(kwargs_for_sinc, 3)
                seq.add_block(rf, gz)
                seq.add_block(gzr)            
                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
            
#        seq.add_block(make_delay(0.002-RFdur))
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))
         
        
        ###############################
        ###              secoond readout action
        idx_T = 4
        
        # calculated here, update in next event
        gradmom_rewinder = np.squeeze(gradm_event_numpy[idx_T,rep,:])
        eventtime_rewinder = np.squeeze(event_time_numpy[idx_T,rep])
        
        ###############################
        ###              line acquisition T(5:end-2)
        idx_T = np.arange(5, gradm_event_numpy.shape[0] - 2) # T(2)
        dur = np.sum(event_time_numpy[idx_T,rep])
        
        gx_gradmom = np.sum(gradm_event_numpy[idx_T,rep,0],0)
        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
        gx = make_trapezoid(kwargs_for_gx)    
        
        gy_gradmom = np.sum(gradm_event_numpy[idx_T,rep,1],0)
        kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": dur}
        gy = make_trapezoid(kwargs_for_gy)
        
        kwargs_for_adc = {"num_samples": idx_T.size, "duration": gx.flat_time, "delay": (gx.rise_time), "phase_offset": rf.phase_offset - np.pi/4}
        adc = makeadc(kwargs_for_adc)    
        
        #update rewinder for gxgy ramp times, from second event
        kwargs_for_gxpre = {"channel": 'x', "system": system, "area": -gradm_event_numpy[idx_T[0],rep,0]/2 + gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
        gx_pre = make_trapezoid(kwargs_for_gxpre)
        
        kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1]-gy.amplitude*gy.rise_time/2, "duration": eventtime_rewinder}
        gy_pre = make_trapezoid(kwargs_for_gypre)
        
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
        idx_T = gradm_event_numpy.shape[0] - 2     # T(2)
        
        kwargs_for_gxpost = {"channel": 'x', "system": system, "area": gradm_event_numpy[idx_T,rep,0]-gx.amplitude*gx.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gx_post = make_trapezoid(kwargs_for_gxpost)  
        
        kwargs_for_gypost = {"channel": 'y', "system": system, "area": gradm_event_numpy[idx_T,rep,1]-gy.amplitude*gy.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gy_post = make_trapezoid(kwargs_for_gypost)  
        
        # dont play zero grads (cant even do FID otherwise)
        if np.abs(gradm_event_numpy[idx_T,rep,0]) > 0 or np.abs(gradm_event_numpy[idx_T,rep,1]) > 0:
            seq.add_block(gx_post, gy_post)
        
        ###############################
        ###     last extra event  T(end)
        idx_T = gradm_event_numpy.shape[0] - 1 # T(2)
        
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
    
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)
        
def pulseq_write_RARE(seq_params, seq_fn, plot_seq=False):
    rf_event_numpy, event_time_numpy, gradm_event_numpy_input = seq_params
    
    rf_event_numpy = rectify_rf_event(rf_event_numpy)
    
    NRep = rf_event_numpy.shape[1]
    
    # save pulseq definition
    MAXSLEW = 140
    
    
    deltak = 1.0 / FOV()
    gradm_event_numpy = deltak*gradm_event_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ring_down_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(kwargs_for_opts)
    seq = Sequence(system)   
    
    seq.add_block(make_delay(1.48))
    

    for rep in range(NRep):
        
        ###############################
        ###              first action
        idx_T = 0
        RFdur = 0
        if np.abs(rf_event_numpy[idx_T,rep,0]) > 1e-8:
            use = "excitation"
            
            if nonsel:
                RFdur = 1*1e-3
                kwargs_for_block = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": rf_event_numpy[idx_T,rep,1]}
                rf_ex = make_block_pulse(kwargs_for_block, 1)
                
                seq.add_block(rf_ex)     
            else:
                # alternatively slice selective:
                use = "excitation"
                
                
                kwargs_for_sinc = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": rf_event_numpy[idx_T,rep,1]}
                rf_ex, gz, gzr= make_sinc_pulse(kwargs_for_sinc, 3)
                seq.add_block(rf_ex, gz)
                gzr.amplitude=gzr.amplitude
                seq.add_block(gzr)
                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
                
            kwargs_for_gxPre90 = {"channel": 'x', "system": system, "area": gradm_event_numpy[idx_T,rep,0], "duration": event_time_numpy[idx_T,rep]-RFdur}
            gxPre90 = make_trapezoid(kwargs_for_gxPre90) 
            
            seq.add_block(gxPre90)
        else:
            seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
            
        ###############################
        ###              secoond action
        idx_T = 1
        use = "refocusing"
        
        RFdur = 0
        
        if np.abs(rf_event_numpy[idx_T,rep,0]) > 1e-8:
          RFdur = 1*1e-3
          
          if nonsel:
              kwargs_for_block = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": rf_event_numpy[idx_T,rep,1]}
              rf_ref = make_block_pulse(kwargs_for_block, 1)
              seq.add_block(rf_ref)         
          else:
            
              kwargs_for_sinc = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": rf_event_numpy[idx_T,rep,1]}
              rf_ref, gz_ref,gzr = make_sinc_pulse(kwargs_for_sinc, 3)
              seq.add_block(rf_ref, gz_ref)
              RFdur = gz_ref.rise_time + gz_ref.flat_time + gz_ref.fall_time 
              

        # calculated here, update in next event
        gradmom_rewinder = np.squeeze(gradm_event_numpy[idx_T,rep,:])
        eventtime_rewinder = 1.5*1e-3 
        
        delay_after_rev=np.squeeze(event_time_numpy[idx_T,rep]-RFdur-eventtime_rewinder)
        #print([event_time_numpy[idx_T,rep]], RFdur, eventtime_rewinder)
        
        ###############################
        ###              line acquisition T(3:end-1)
        idx_T = np.arange(2, gradm_event_numpy.shape[0] - 2) # T(2)
        
        dur = np.sum(event_time_numpy[idx_T,rep])
        
        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": np.sum(gradm_event_numpy[idx_T,rep,0],0), "flat_time": dur}
        gx = make_trapezoid(kwargs_for_gx)   
        
        kwargs_for_adc = {"num_samples": idx_T.size, "duration": gx.flat_time, "delay": gx.rise_time - event_time_numpy[idx_T[0],rep]/2, "phase_offset": rf_ex.phase_offset - np.pi/4}
        adc = makeadc(kwargs_for_adc)    
        
        #update rewinder for gxgy ramp times, from second event
        kwargs_for_gxpre = {"channel": 'x', "system": system, "area": gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
        gx_pre = make_trapezoid(kwargs_for_gxpre)
        
        kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1], "duration": eventtime_rewinder}
        gy_pre = make_trapezoid(kwargs_for_gypre)
    
        seq.add_block(gx_pre, gy_pre)
                    
#        if delay_after_rev < 0:
#            import pdb; pdb.set_trace()
        seq.add_block(make_delay(delay_after_rev))   
        seq.add_block(gx,adc)    
        
        ###############################
        ###     second last extra event  T(end)  # adjusted also for fallramps of ADC
        idx_T = gradm_event_numpy.shape[0] - 2     # T(2)
        
        kwargs_for_gxpost = {"channel": 'x', "system": system, "area": gradm_event_numpy[idx_T,rep,0]-gx.amplitude*gx.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gx_post = make_trapezoid(kwargs_for_gxpost)  
        
        kwargs_for_gypost = {"channel": 'y', "system": system, "area": gradm_event_numpy[idx_T,rep,1], "duration": event_time_numpy[idx_T,rep]}
        gy_post = make_trapezoid(kwargs_for_gypost)  
        
        seq.add_block(gx_post, gy_post)
          
        ###############################
        ###     last extra event  T(end)
        idx_T = gradm_event_numpy.shape[0] - 1 # T(2)
        
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
        
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)
    
def pulseq_write_BSSFP(seq_params, seq_fn, plot_seq=False):
    rf_event_numpy, event_time_numpy, gradm_event_numpy_input = seq_params
    
    rf_event_numpy = rectify_rf_event(rf_event_numpy)
    
    NRep = rf_event_numpy.shape[1]
    
    # save pulseq definition
    MAXSLEW = 140
    
    deltak = 1.0 / FOV()
    gradm_event_numpy = deltak*gradm_event_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ring_down_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(kwargs_for_opts)
    seq = Sequence(system)    
    
    seq.add_block(make_delay(2.0))
    
   
    for rep in range(NRep):
        
        ###############################
        ###              first action
        idx_T = 0
        if np.abs(rf_event_numpy[idx_T,rep,0]) > 1e-8:
            
            if nonsel:

                use = "excitation"
                
                # alternatively slice selective:
                RFdur = 0.8*1e-3
                kwargs_for_block = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": rf_event_numpy[idx_T,rep,1]}
                rf = make_block_pulse(kwargs_for_block, 1)
                
                seq.add_block(rf)     
            else:
                # alternatively slice selective:
                use = "excitation"
                
                
                # alternatively slice selective:
                kwargs_for_sinc = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "phase_offset": rf_event_numpy[idx_T,rep,1], "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4}
                rf, gz, gzr = make_sinc_pulse(kwargs_for_sinc, 3)
                
                seq.add_block(rf, gz)
                seq.add_block(gzr)            
                
                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
                
        
#        seq.add_block(make_delay(0.002-RFdur))
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))
        
        ###############################
        ###              secoond action
        idx_T = 1
        
        # calculated here, update in next event
        gradmom_rewinder = np.squeeze(gradm_event_numpy[idx_T,rep,:])
        eventtime_rewinder = np.squeeze(event_time_numpy[idx_T,rep])
        
        ###############################
        ###              line acquisition T(3:end-1)
        idx_T = np.arange(2, gradm_event_numpy.shape[0] - 2) # T(2)
        dur = np.sum(event_time_numpy[idx_T,rep])
        
        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": np.sum(gradm_event_numpy[idx_T,rep,0],0), "flat_time": dur}
        gx = make_trapezoid(kwargs_for_gx)    
        
        kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": np.sum(gradm_event_numpy[idx_T,rep,1],0), "flat_time": dur}
        gy = make_trapezoid(kwargs_for_gy)
        
        kwargs_for_adc = {"num_samples": idx_T.size, "duration": gx.flat_time, "delay": gx.rise_time - event_time_numpy[idx_T[0],rep]/2, "phase_offset": rf.phase_offset - np.pi/4}
        adc = makeadc(kwargs_for_adc)    
        
        #update rewinder for gxgy ramp times, from second event
        kwargs_for_gxpre = {"channel": 'x', "system": system, "area": gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
        gx_pre = make_trapezoid(kwargs_for_gxpre)
        
        kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1]-gy.amplitude*gy.rise_time/2, "duration": eventtime_rewinder}
        gy_pre = make_trapezoid(kwargs_for_gypre)
    
        seq.add_block(gx_pre, gy_pre)
        seq.add_block(gx,gy,adc)
        
        ###############################
        ###     second last extra event  T(end)  # adjusted also for fallramps of ADC
        idx_T = gradm_event_numpy.shape[0] - 2     # T(2)
        
        kwargs_for_gxpost = {"channel": 'x', "system": system, "area": gradm_event_numpy[idx_T,rep,0]-gx.amplitude*gx.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gx_post = make_trapezoid(kwargs_for_gxpost)  
        
        kwargs_for_gypost = {"channel": 'y', "system": system, "area": gradm_event_numpy[idx_T,rep,1]-gy.amplitude*gy.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gy_post = make_trapezoid(kwargs_for_gypost)  
        
        if nonsel:
            seq.add_block(gx_post, gy_post)
        else:
            seq.add_block(gx_post, gy_post, gzr)
            
        
        ###############################
        ###     last extra event  T(end)
        idx_T = gradm_event_numpy.shape[0] - 1 # T(2)
        
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
    
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)    
    
def pulseq_write_slBSSFP(seq_params, seq_fn, plot_seq=False):
    rf_event_numpy, event_time_numpy, gradm_event_numpy_input = seq_params
    
    rf_event_numpy = rectify_rf_event(rf_event_numpy)
    
    NRep = rf_event_numpy.shape[1]
    
    # save pulseq definition
    MAXSLEW = 140
      
    
    deltak = 1.0 / FOV()
    gradm_event_numpy = deltak*gradm_event_numpy_input  # adjust for FOV
    
    kwargs_for_opts = {"rf_ring_down_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 36, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(kwargs_for_opts)
    seq = Sequence(system)    
    
    seq.add_block(make_delay(2.0))
    
    
    for rep in range(NRep):
        
        ###############################
        ###              first action
        idx_T = 0
        if np.abs(rf_event_numpy[idx_T,rep,0]) > 1e-8:
            slice_thickness = 200e-3     # slice

            # alternatively slice selective:
            RFdur = event_time_numpy[idx_T,rep]
            kwargs_for_block = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": rf_event_numpy[idx_T,rep,1],"freq_offset": int(rf_event_numpy[idx_T,rep,2])}
            rf = make_block_pulse(kwargs_for_block, 1)
            
            seq.add_block(rf)  
            seq.add_block(make_delay(1e-4))
        ###              second action
        idx_T = 1
        if np.abs(rf_event_numpy[idx_T,rep,0]) > 1e-8:
            slice_thickness = 200e-3     # slice
            
            # alternatively slice selective:
            RFdur = event_time_numpy[idx_T,rep]
            kwargs_for_block = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": rf_event_numpy[idx_T,rep,1],"freq_offset": rf_event_numpy[idx_T,rep,2]}
            rf = make_block_pulse(kwargs_for_block, 1)
            
            seq.add_block(rf)  
            seq.add_block(make_delay(1e-4))
                
  
        ###              first readout (action 2 now)            
        idx_T = 2
        RFdur=0
        if np.abs(rf_event_numpy[idx_T,rep,0]) > 1e-8:
            
            if nonsel:
                slice_thickness = 200e-3     # slice
                use = "excitation"
                
                # alternatively slice selective:
                RFdur = 0.8*1e-3
                kwargs_for_block = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": RFdur, "phase_offset": rf_event_numpy[idx_T,rep,1]}
                rf = make_block_pulse(kwargs_for_block, 1)
                
                seq.add_block(rf)     
            else:
                # alternatively slice selective:
                use = "excitation"
                
                slice_thickness = 5e-3
                
                # alternatively slice selective:
                kwargs_for_sinc = {"flip_angle": rf_event_numpy[idx_T,rep,0], "system": system, "duration": 1e-3, "phase_offset": rf_event_numpy[idx_T,rep,1], "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4}
                rf, gz, gzr = make_sinc_pulse(kwargs_for_sinc, 3)
                
                seq.add_block(rf, gz)
                seq.add_block(gzr)            
                
                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time
                
        
#        seq.add_block(make_delay(0.002-RFdur))
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]-RFdur))
        
        ###############################
        ###              secoond action
        idx_T = 3
        
        # calculated here, update in next event
        gradmom_rewinder = np.squeeze(gradm_event_numpy[idx_T,rep,:])
        eventtime_rewinder = np.squeeze(event_time_numpy[idx_T,rep])
        
        ###############################
        ###              line acquisition T(3:end-1)
        idx_T = np.arange(4, gradm_event_numpy.shape[0] - 2) # T(2)
        dur = np.sum(event_time_numpy[idx_T,rep])
        
        kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": np.sum(gradm_event_numpy[idx_T,rep,0],0), "flat_time": dur}
        gx = make_trapezoid(kwargs_for_gx)    
        
        kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": np.sum(gradm_event_numpy[idx_T,rep,1],0), "flat_time": dur}
        gy = make_trapezoid(kwargs_for_gy)
        
        kwargs_for_adc = {"num_samples": idx_T.size, "duration": gx.flat_time, "delay": gx.rise_time - event_time_numpy[idx_T[0],rep]/2, "phase_offset": rf.phase_offset - np.pi/4}
        adc = makeadc(kwargs_for_adc)    
        
        #update rewinder for gxgy ramp times, from second event
        kwargs_for_gxpre = {"channel": 'x', "system": system, "area": gradmom_rewinder[0]-gx.amplitude*gx.rise_time/2, "duration": eventtime_rewinder}
        gx_pre = make_trapezoid(kwargs_for_gxpre)
        
        kwargs_for_gypre = {"channel": 'y', "system": system, "area": gradmom_rewinder[1]-gy.amplitude*gy.rise_time/2, "duration": eventtime_rewinder}
        gy_pre = make_trapezoid(kwargs_for_gypre)
    
        seq.add_block(gx_pre, gy_pre)
        seq.add_block(gx,gy,adc)
        
        ###############################
        ###     second last extra event  T(end)  # adjusted also for fallramps of ADC
        idx_T = gradm_event_numpy.shape[0] - 2     # T(2)
        
        kwargs_for_gxpost = {"channel": 'x', "system": system, "area": gradm_event_numpy[idx_T,rep,0]-gx.amplitude*gx.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gx_post = make_trapezoid(kwargs_for_gxpost)  
        
        kwargs_for_gypost = {"channel": 'y', "system": system, "area": gradm_event_numpy[idx_T,rep,1]-gy.amplitude*gy.fall_time/2, "duration": event_time_numpy[idx_T,rep]}
        gy_post = make_trapezoid(kwargs_for_gypost)  
        
        if nonsel:
            seq.add_block(gx_post, gy_post)
        else:
            seq.add_block(gx_post, gy_post, gzr)
            
        
        ###############################
        ###     last extra event  T(end)
        idx_T = gradm_event_numpy.shape[0] - 1 # T(2)
        
        seq.add_block(make_delay(event_time_numpy[idx_T,rep]))
    
    if plot_seq:
        seq.plot()
    seq.write(seq_fn)
    
    append_header(seq_fn, FOV(),slice_thickness)    
    
def pulseq_write_EPI(seq_params, seq_fn, plot_seq=False):
    pulseq_write_GRE_DREAM(seq_params, seq_fn, plot_seq=plot_seq)
    

        
def append_header(seq_fn, FOV,slice_thickness):
    # append version and definitions
    with open(seq_fn, 'r') as fin:
        lines = fin.read().splitlines(True)
        
    updated_lines = []
    updated_lines.append("# Pulseq sequence file\n")
    updated_lines.append("# Created by MRIzero/IMR/GPI pulseq converter\n")
    updated_lines.append('#' + seq_fn + "\n")
    updated_lines.append("\n")
    updated_lines.append("[VERSION]\n")
    updated_lines.append("major 1\n")
    updated_lines.append("minor 2\n")   
    updated_lines.append("revision 1\n")  
    updated_lines.append("\n")    
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
#    rf.ring_down_time = system.rf_ring_down_time
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

