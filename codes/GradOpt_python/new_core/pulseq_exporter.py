# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio
from math import pi,ceil, sqrt, pow
import sys
import torch
from . import sequence as Seq
from . import util

sys.path.append("../scannerloop_libs")
from pypulseq.Sequence.sequence import Sequence
from pypulseq.calc_duration import calc_duration
from pypulseq.make_adc import make_adc
from pypulseq.make_delay import make_delay
from pypulseq.make_sinc_pulse import make_sinc_pulse
from pypulseq.make_trap_pulse import make_trapezoid
from pypulseq.make_block_pulse import make_block_pulse
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.opts import Opts


def rectify_flips(flips):
    flip_angle = flips.angle
    flip_phase = flips.phase
    
    if flips.angle < 0:
        flip_angle = -flips.angle
        flip_phase = flips.phase + np.pi
        flip_phase = torch.fmod(flip_phase, 2*np.pi)
    return flip_angle.item(),flip_phase


nonsel = 0
if nonsel==1:
    slice_thickness = 200*1e-3
else:
    slice_thickness = 8e-3
    
def pulseq_write_EPG(seq_param, path, FOV, plot_seq=False):
    # save pulseq definition
    MAXSLEW = 200
    FOV = FOV / 1000
    deltak = 1.0 / FOV # /(2*np.pi) before v.2.1.0
    
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 80, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)   
    
    seq.add_block(make_delay(5.0))
    
    # import pdb; pdb.set_trace()
    for i,rep in enumerate(seq_param):
        adc_start = 0
        flip_angle,flip_phase = rectify_flips(rep.pulse)
        for event in range(rep.event_count):
            ###############################
            ## global pulse
            if torch.abs(rep.adc_usage[event]) == 0:  
                RFdur=0
                if event == 0:
                    if rep.pulse.usage == Seq.PulseUsage.UNDEF:
                        RFdur=0
                        if np.abs(rep.pulse.angle) > 1e-8:
                
                            RFdur = 1*1e-3
                            kwargs_for_block = {"flip_angle": flip_angle.item(), "system": system, "duration": RFdur, "phase_offset": flip_phase}
                            rf,_ = make_block_pulse(**kwargs_for_block)
                            
                            seq.add_block(rf)  
                            seq.add_block(make_delay(1e-4))
                                                    
                    elif (rep.pulse.usage == Seq.PulseUsage.EXCIT or
                          rep.pulse.usage == Seq.PulseUsage.STORE):
                        RFdur = 0
                        if np.abs(rep.pulse.angle) > 1e-8:
                            use = "excitation"
                            
                            if nonsel:
                                RFdur = 1*1e-3
                                kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                                rf_ex,_ = make_block_pulse(**kwargs_for_block)
                                seq.add_block(rf_ex)     
                            else:
                                # alternatively slice selective:
                                use = "excitation"
                                RFdur = 1*1e-3
                                kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flip_phase}
                                rf_ex, gz, gzr= make_sinc_pulse(**kwargs_for_sinc)
                                seq.add_block(gzr)
                                seq.add_block(rf_ex, gz)
                                seq.add_block(gzr)
                                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time                            
                    
                    elif rep.pulse.usage == Seq.PulseUsage.REFOC:
                        ###############################
                        ### refocusing pulse
                        use = "refocusing"
                        
                        RFdur = 0
                        
                        if np.abs(rep.pulse.angle) > 1e-8:
                          RFdur = 1*1e-3
                          
                          if nonsel:
                              kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                              rf_ref,_ = make_block_pulse(**kwargs_for_block)
                              seq.add_block(rf_ref)         
                          else:
    
                              kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flip_phase}
                              rf_ref, gz_ref,gzr = make_sinc_pulse(**kwargs_for_sinc)
                              seq.add_block(gzr)
                              seq.add_block(rf_ref, gz_ref)
                              seq.add_block(gzr)
                              RFdur = gz_ref.rise_time + gz_ref.flat_time + gz_ref.fall_time 
        
                dur = rep.event_time[event].item() - RFdur
                if dur < 0:
                    raise Exception('Event Time too short! Event Time: Rep: '+ str(i) + ', Event: ' +str(event),', increase event_time by at least: ' + str(-dur))
                
                gx_gradmom = rep.gradm[event,0].item()*deltak
                gy_gradmom = rep.gradm[event,1].item()*deltak


                if np.abs(gx_gradmom)>0:
                    gx_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1,0]) > 0:
                            kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": rep.gradm[event+1,0].item()*deltak, "flat_time": rep.event_time[event+1].item()}
                            gx_adc = make_trapezoid(**kwargs_for_gx)
                            gx_adc_ramp = gx_adc.amplitude*gx_adc.rise_time/2
                    kwargs_for_gx = {"channel": 'x', "system": system, "area": gx_gradmom-gx_adc_ramp, "duration": dur}
                    gx = make_trapezoid(**kwargs_for_gx) 
                if np.abs(gy_gradmom)>0:
                    gy_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1,0]) > 0:   
                            kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": rep.gradm[event+1,0].item()*deltak, "flat_time": rep.event_time[event+1].item()}
                            gy_adc = make_trapezoid(**kwargs_for_gy)
                            gy_adc_ramp = gy_adc.amplitude*gy_adc.rise_time/2                    
                    kwargs_for_gy = {"channel": 'y', "system": system, "area": gy_gradmom-gy_adc_ramp, "duration": dur}
                    try:
                        gy = make_trapezoid(**kwargs_for_gy)
                    except Exception as e:
                        print(e)
                        print('Event Time too short! Event Time: Rep: '+ str(i) + ', Event: ' +str(event))                              
                            
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
                    idx_T = np.nonzero(torch.abs(rep.adc_usage))                
                    dur = torch.sum(rep.event_time[idx_T],0).item()
    
                    gx_gradmom = torch.sum(rep.gradm[idx_T,0]).item()*deltak                                        
                    kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
                    gx = make_trapezoid(**kwargs_for_gx) 
                    
                    gy_gradmom = torch.sum(rep.gradm[idx_T,1]).item()*deltak
                    kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": dur}
                    gy = make_trapezoid(**kwargs_for_gy)   

                    # calculate correct delay to have same starting point of flat top
                    x_delay = np.max([0,gy.rise_time-gx.rise_time])+rep.event_time[idx_T[0]].item()/2  # heuristic delay, to be checked at scanner
                    y_delay = np.max([0,gx.rise_time-gy.rise_time])+rep.event_time[idx_T[0]].item()/2
                    
                    # adc gradient events are overwritten with correct delays
                    kwargs_for_gx = {"channel": 'x', "system": system,"delay":x_delay, "flat_area": gx_gradmom, "flat_time": dur}
                    gx = make_trapezoid(**kwargs_for_gx)                      
                    kwargs_for_gy = {"channel": 'y', "system": system,"delay":y_delay, "flat_area": gy_gradmom, "flat_time": dur}
                    gy = make_trapezoid(**kwargs_for_gy)                       
                    
                    
                    adc_delay = np.max([gx.rise_time,gy.rise_time])
                    kwargs_for_adc = {"num_samples": idx_T.size()[0], "duration": dur, "delay":(adc_delay), "phase_offset": rf_ex.phase_offset - np.pi/4}
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
    seq.write(path)
    
    append_header(path, FOV,slice_thickness)

def pulseq_write_EPG_3D(seq_param, path, FOV, plot_seq=False, num_slices=1):
    # save pulseq definition
    slice_thickness = 5e-3*num_slices 
    MAXSLEW = 200
    FOV = FOV / 1000
    deltak = 1.0 / FOV # /(2*np.pi) before v.2.1.0
    deltakz = 1.0 / slice_thickness
    
    kwargs_for_opts = {"rf_ringdown_time": 20e-6, "rf_dead_time": 100e-6, "adc_dead_time": 20e-6, "max_grad": 80, "grad_unit": "mT/m", "max_slew": MAXSLEW, "slew_unit": "T/m/s"}
    system = Opts(**kwargs_for_opts)
    seq = Sequence(system)   
    
    seq.add_block(make_delay(5.0))
    
    # import pdb; pdb.set_trace()
    for i,rep in enumerate(seq_param):
        adc_start = 0
        flip_angle,flip_phase = rectify_flips(rep.pulse)
        for event in range(rep.event_count):
            ###############################
            ## global pulse
            if torch.abs(rep.adc_usage[event]) == 0:  
                RFdur=0
                if event == 0:
                    if rep.pulse.usage == Seq.PulseUsage.UNDEF:
                        RFdur=0
                        if torch.abs(rep.pulse.angle) > 1e-8:
                
                            RFdur = 1*1e-3
                            kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                            rf,_ = make_block_pulse(**kwargs_for_block)
                            
                            seq.add_block(rf)  
                            seq.add_block(make_delay(1e-4))
                                                    
                    elif (rep.pulse.usage == Seq.PulseUsage.EXCIT or
                          rep.pulse.usage == Seq.PulseUsage.STORE):
                        RFdur = 0
                        if torch.abs(rep.pulse.angle) > 1e-8:
                            use = "excitation"
                            
                            if nonsel:
                                RFdur = 1*1e-3
                                kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                                rf_ex,_ = make_block_pulse(**kwargs_for_block)
                                seq.add_block(rf_ex)     
                            else:
                                # alternatively slice selective:
                                use = "excitation"
                                RFdur = 1*1e-3
                                kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flip_phase}
                                rf_ex, gz, gzr= make_sinc_pulse(**kwargs_for_sinc)
                                seq.add_block(gzr)
                                seq.add_block(rf_ex, gz)
                                seq.add_block(gzr)
                                RFdur = gz.rise_time + gz.flat_time + gz.fall_time + gzr.rise_time + gzr.flat_time + gzr.fall_time                            
                    
                    elif rep.pulse.usage == Seq.PulseUsage.REFOC:
                        ###############################
                        ### refocusing pulse
                        use = "refocusing"
                        
                        RFdur = 0
                        
                        if torch.abs(rep.pulse.angle) > 1e-8:
                          RFdur = 1*1e-3
                          
                          if nonsel:
                              kwargs_for_block = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "phase_offset": flip_phase}
                              rf_ref,_ = make_block_pulse(**kwargs_for_block)
                              seq.add_block(rf_ref)         
                          else:
    
                              kwargs_for_sinc = {"flip_angle": flip_angle, "system": system, "duration": RFdur, "slice_thickness": slice_thickness, "apodization": 0.5, "time_bw_product": 4, "phase_offset": flip_phase}
                              rf_ref, gz_ref,gzr = make_sinc_pulse(**kwargs_for_sinc)
                              seq.add_block(gzr)
                              seq.add_block(rf_ref, gz_ref)
                              seq.add_block(gzr)
                              RFdur = gz_ref.rise_time + gz_ref.flat_time + gz_ref.fall_time 
        
                dur = rep.event_time[event].item() - RFdur
                if dur < 0:
                    raise Exception('Event Time too short! Event Time: Rep: '+ str(i) + ', Event: ' +str(event),', increase event_time by at least: ' + str(-dur))
                
                gx_gradmom = rep.gradm[event,0].item()*deltak
                gy_gradmom = rep.gradm[event,1].item()*deltak
                gz_gradmom = rep.gradm[event,2].item()*deltakz

                if np.abs(gx_gradmom)>0:
                    gx_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1,0]) > 0:
                            kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": rep.gradm[event+1,0].item()*deltak, "flat_time": rep.event_time[event+1].item()}
                            gx_adc = make_trapezoid(**kwargs_for_gx)
                            gx_adc_ramp = gx_adc.amplitude*gx_adc.rise_time/2
                    kwargs_for_gx = {"channel": 'x', "system": system, "area": gx_gradmom-gx_adc_ramp, "duration": dur}
                    gx = make_trapezoid(**kwargs_for_gx) 
                if np.abs(gy_gradmom)>0:
                    gy_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1,1]) > 0:   
                            kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": rep.gradm[event+1,1].item()*deltak, "flat_time": rep.event_time[event+1].item()}
                            gy_adc = make_trapezoid(**kwargs_for_gy)
                            gy_adc_ramp = gy_adc.amplitude*gy_adc.rise_time/2                    
                    kwargs_for_gy = {"channel": 'y', "system": system, "area": gy_gradmom-gy_adc_ramp, "duration": dur}
                    try:
                        gy = make_trapezoid(**kwargs_for_gy)
                    except Exception as e:
                        print(e)
                        print('Event Time too short! Event Time: Rep: '+ str(i) + ', Event: ' +str(event))                              
                if np.abs(gz_gradmom)>0:
                    gz_adc_ramp = 0
                    if any(rep.adc_usage):
                        if torch.abs(rep.adc_usage[event+1]) and torch.abs(rep.gradm[event+1,0]) > 0:   
                            kwargs_for_gz = {"channel": 'y', "system": system, "flat_area": rep.gradm[event+1,2].item()*deltakz, "flat_time": rep.event_time[event+1].item()}
                            gz_adc = make_trapezoid(**kwargs_for_gz)
                            gz_adc_ramp = gz_adc.amplitude*gz_adc.rise_time/2                    
                    kwargs_for_gz = {"channel": 'z', "system": system, "area": gz_gradmom-gz_adc_ramp, "duration": dur}
                    try:
                        gz = make_trapezoid(**kwargs_for_gz)
                    except Exception as e:
                        print(e)
                        print('Event Time too short! Event Time: Rep: '+ str(i) + ', Event: ' +str(event))                              
                            
                if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom):
                    seq.add_block(gx,gy,gz)
                elif np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                    seq.add_block(gx,gy)
                elif np.abs(gx_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                    seq.add_block(gx,gz)
                elif np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                    seq.add_block(gy,gz)
                elif np.abs(gx_gradmom) > 0:
                    seq.add_block(gx)
                elif np.abs(gy_gradmom) > 0:
                    seq.add_block(gy)
                elif np.abs(gz_gradmom) > 0:
                    seq.add_block(gz)
                else:
                    seq.add_block(make_delay(dur))
            else: #adc mask == 1
                if adc_start == 1:
                    pass
                else:
                    adc_start = 1
                    idx_T = np.nonzero(torch.abs(rep.adc_usage))                
                    dur = torch.sum(rep.event_time[idx_T],0).item()
    
                    gx_gradmom = torch.sum(rep.gradm[idx_T,0]).item()*deltak                                        
                    kwargs_for_gx = {"channel": 'x', "system": system, "flat_area": gx_gradmom, "flat_time": dur}
                    gx = make_trapezoid(**kwargs_for_gx) 
                    
                    gy_gradmom = torch.sum(rep.gradm[idx_T,1]).item()*deltak
                    kwargs_for_gy = {"channel": 'y', "system": system, "flat_area": gy_gradmom, "flat_time": dur}
                    gy = make_trapezoid(**kwargs_for_gy)   

                    gz_gradmom = torch.sum(rep.gradm[idx_T,2]).item()*deltakz
                    kwargs_for_gz = {"channel": 'z', "system": system, "flat_area": gz_gradmom, "flat_time": dur}
                    gz = make_trapezoid(**kwargs_for_gz)  

                    # calculate correct delay to have same starting point of flat top
                    shift = 0.0# rep.event_time[idx_T[0]].item()/2  # heuristic delay, to be checked at scanner
                    x_delay = np.max([0,gy.rise_time-gx.rise_time,gz.rise_time-gx.rise_time])+shift
                    y_delay = np.max([0,gx.rise_time-gy.rise_time,gz.rise_time-gy.rise_time])+shift
                    z_delay = np.max([0,gx.rise_time-gz.rise_time,gy.rise_time-gz.rise_time])+shift
                                     
                    # adc gradient events are overwritten with correct delays
                    kwargs_for_gx = {"channel": 'x', "system": system,"delay":x_delay, "flat_area": gx_gradmom, "flat_time": dur}
                    gx = make_trapezoid(**kwargs_for_gx)                      
                    kwargs_for_gy = {"channel": 'y', "system": system,"delay":y_delay, "flat_area": gy_gradmom, "flat_time": dur}
                    gy = make_trapezoid(**kwargs_for_gy) 
                    kwargs_for_gz = {"channel": 'z', "system": system,"delay":z_delay, "flat_area": gz_gradmom, "flat_time": dur}
                    gz = make_trapezoid(**kwargs_for_gz)                       
                                        
                    adc_delay = np.max([gx.rise_time,gy.rise_time,gz.rise_time])+shift
                    kwargs_for_adc = {"num_samples": idx_T.size()[0], "duration": dur, "delay":(adc_delay), "phase_offset": rf_ex.phase_offset - np.pi/4}
                    adc = make_adc(**kwargs_for_adc)    
                    
                    # dont play zero grads (cant even do FID otherwise)
                    if np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom):
                        seq.add_block(gx,gy,gz,adc)
                    elif np.abs(gx_gradmom) > 0 and np.abs(gy_gradmom) > 0:
                        seq.add_block(gx,gy,adc)
                    elif np.abs(gx_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                        seq.add_block(gx,gz,adc)
                    elif np.abs(gy_gradmom) > 0 and np.abs(gz_gradmom) > 0:
                        seq.add_block(gy,gz,adc)
                    elif np.abs(gx_gradmom) > 0:
                        seq.add_block(gx,adc)
                    elif np.abs(gy_gradmom) > 0:
                        seq.add_block(gy,adc)
                    elif np.abs(gz_gradmom) > 0:
                        seq.add_block(gz,adc)
                    else:
                        seq.add_block(adc)

                
    if plot_seq:
        seq.plot()
    seq.write(path)
    
    append_header(path, FOV,slice_thickness)

def append_header(path, FOV,slice_thickness):
    # append version and definitions
    if sys.platform != 'linux':
        try:
            with open(r"\\141.67.249.47\MRTransfer\mrzero_src\.git\ORIG_HEAD") as file:
                git_version = file.read()
        except:
            git_version = ''
    with open(path, 'r') as fin:
        lines = fin.read().splitlines(True)

    updated_lines = []
    updated_lines.append("# Pulseq sequence file\n")
    updated_lines.append("# Created by MRIzero/IMR/GPI pulseq converter\n")
    if sys.platform != 'linux':
        updated_lines.append('# MRZero Version: 0.5, git hash: ' + git_version)
    if sys.platform == 'linux':
        updated_lines.append("# experiment_id: "+path.split('/')[-2]+"\n")
    else:
        updated_lines.append("# experiment_id: "+path.split('\\')[-2]+"\n")
    updated_lines.append('#' + path + "\n")
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

    with open(path, 'w') as fout:
        fout.writelines(updated_lines)    