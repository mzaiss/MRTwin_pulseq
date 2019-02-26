import numpy as np
import torch

# target images / sequence parameters holder

class TargetSequenceHolder():
    def __init__(self):
        
        self.target_image = None
        self.flips = None
        self.grad_moms = None
        self.event_time = None
        self.adc_mask = None
        
