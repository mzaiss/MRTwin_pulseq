from __future__ import annotations
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import new_core.sequence as Seq
from new_core import util


class PREPT1:
    """Stores all parameters needed to create a preparation for a sequence."""

    def __init__(self, adc_count: int, prep_count: int, 
                 inversion_time: torch.Tensor | None = None,
                 inversion_pulse_angles: torch.Tensor | None = None):
        """Initialize parameters with default values."""
        self.event_count = 3
        self.prep_count = prep_count
        self.adc_count = adc_count
        
        if inversion_pulse_angles is None:
            inversion_pulse_angles = torch.full((self.prep_count, ), 180 * np.pi / 180)
        if inversion_time is None:
            inversion_time = torch.logspace(-2,0.5,self.prep_count)
                
        self.inversion_pulse_angles = inversion_pulse_angles
        self.inversion_pulse_phase = torch.full((self.prep_count, ), 0 * np.pi / 180)
        self.inversion_time = inversion_time
                        
    def clone(self) -> PREPT1:
        """Create a copy with cloned tensors."""
        clone = PREPT1()

        clone.inversion_pulse_angles = self.inversion_pulse_angles.clone()
        clone.inversion_pulse_phase = self.inversion_pulse_phase.clone()
        clone.inversion_time = self.inversion_time.clone()

        return clone

    def generate_sequence(self) -> list[Seq.Sequence]:
        """Generate a prep sequence based on the given parameters."""
        seq_all = []
        for ii in np.arange(self.prep_count):
            seq = Seq.Sequence()
    
            rep = Seq.Repetition.zero(self.event_count)
            seq.append(rep)
            rep.pulse.angle = self.inversion_pulse_angles[ii]
            rep.pulse.phase = self.inversion_pulse_phase[ii]
            rep.pulse.usage = Seq.PulseUsage.UNDEF
            
            rep.gradm[2,:2] = torch.tensor(2.0 * self.adc_count)
            
            rep.event_time[0] = 1.3e-3  # Pulse
            rep.event_time[1] = self.inversion_time[ii]  # TI
            rep.event_time[2] = 2e-3        # Spoiler
            
            seq_all.append(seq)
           
        return seq_all
    
    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_name) -> PREPT1:
        with open(file_name, 'rb') as file:
            return pickle.load(file)
        
class PREPT2:
    """Stores all parameters needed to create a preparation for a sequence."""

    def __init__(self, adc_count: int, prep_count: int, 
                 TEd: torch.Tensor | None = None, type = 'simple'):
        """Initialize parameters with default values."""
        self.event_count = 2
        self.pulse_dur = 1.3e-3
        self.prep_count = prep_count
        self.adc_count = adc_count
        self.type = type
        
        if TEd is None:
            TEd = torch.logspace(0.01,1.2,self.prep_count)
                
        self.TEd = TEd
                        
    def clone(self) -> PREPT2:
        """Create a copy with cloned tensors."""
        clone = PREPT2()
        clone.event_count = self.event_count
        clone.pulse_dur = self.pulse_dur
        clone.prep_count = self.prep_count
        clone.adc_count = self.adc_count
        clone.TEd = self.TEd.clone()

        return clone

    def generate_sequence(self) -> list[Seq.Sequence]:
        """Generate a prep sequence based on the given parameters."""
        seq_all = []
        for ii in np.arange(self.prep_count):
            seq = Seq.Sequence()
            if self.type == 'simple':
                rep = Seq.Repetition.zero(self.event_count)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(90 * np.pi / 180)
                rep.pulse.phase =torch.tensor( 0*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur
                rep.event_time[1] = self.TEd[ii]/2
                
                rep = Seq.Repetition.zero(self.event_count)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(180 * np.pi / 180)
                rep.pulse.phase = torch.tensor(90*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur
                rep.event_time[1] = self.TEd[ii]/2           
    
                rep = Seq.Repetition.zero(self.event_count)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(90 * np.pi / 180)
                rep.pulse.phase = torch.tensor(180*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur
                rep.event_time[1] = 2e-3                 
                rep.gradm[1,:1] = torch.tensor(2.0 * self.adc_count)
            elif self.type == 'MLEV':
                rep = Seq.Repetition.zero(2)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(90 * np.pi / 180)
                rep.pulse.phase = torch.tensor( 0*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur
                rep.event_time[1] = self.TEd[ii]/8
                #MLEV 4: 1. 180:
                rep = Seq.Repetition.zero(1)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(90 * np.pi / 180)
                rep.pulse.phase = torch.tensor(0*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur      
                rep = Seq.Repetition.zero(1)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(180 * np.pi / 180)
                rep.pulse.phase = torch.tensor(0*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur              
                rep = Seq.Repetition.zero(2)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(90 * np.pi / 180)
                rep.pulse.phase = torch.tensor(0*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur
                rep.event_time[1] = self.TEd[ii]/4
                #MLEV 4: 2. 180:
                rep = Seq.Repetition.zero(1)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(90 * np.pi / 180)
                rep.pulse.phase = torch.tensor(0*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur      
                rep = Seq.Repetition.zero(1)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(180 * np.pi / 180)
                rep.pulse.phase = torch.tensor(0*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur              
                rep = Seq.Repetition.zero(2)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(90 * np.pi / 180)
                rep.pulse.phase = torch.tensor(0*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur
                rep.event_time[1] = self.TEd[ii]/4
                #MLEV 4: 3. 180:        
                rep = Seq.Repetition.zero(1)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(90 * np.pi / 180)
                rep.pulse.phase = torch.tensor(180*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur      
                rep = Seq.Repetition.zero(1)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(180 * np.pi / 180)
                rep.pulse.phase = torch.tensor(270*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur              
                rep = Seq.Repetition.zero(2)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(90 * np.pi / 180)
                rep.pulse.phase = torch.tensor(180*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur
                rep.event_time[1] = self.TEd[ii]/4     
                #MLEV 4: 4. 180:   
                rep = Seq.Repetition.zero(1)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(90 * np.pi / 180)
                rep.pulse.phase = torch.tensor(180*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur      
                rep = Seq.Repetition.zero(1)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(180 * np.pi / 180)
                rep.pulse.phase = torch.tensor(270*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur              
                rep = Seq.Repetition.zero(2)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(90 * np.pi / 180)
                rep.pulse.phase = torch.tensor(180*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur
                rep.event_time[1] = self.TEd[ii]/8    
                #Tip up pulse
                rep = Seq.Repetition.zero(1)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(270 * np.pi / 180)
                rep.pulse.phase = torch.tensor(0*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur
                rep = Seq.Repetition.zero(2)
                seq.append(rep)
                rep.pulse.angle = torch.tensor(360 * np.pi / 180)
                rep.pulse.phase = torch.tensor(180*np.pi/180)
                rep.pulse.usage = Seq.PulseUsage.UNDEF
                rep.event_time[0] = self.pulse_dur                      
                rep.event_time[1] = 2e-3                 
                rep.gradm[1,:1] = torch.tensor(2.0 * self.adc_count)                  
            seq_all.append(seq)
           
        return seq_all
    
    def save(self, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, file_name) -> PREPT1:
        with open(file_name, 'rb') as file:
            return pickle.load(file)