# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 12:38:25 2020

@author: zaissmz
"""
#%% ############################################################################

import os, fnmatch
def findReplace(directory, find, replace, filePattern):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath) as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w") as f:
                f.write(s)

#This allows you to do something like:

findReplace("D:/root/Lehre/pulseqMRI/MRtwin_FAU/code/help","(scanner.NEvnt,scanner.NRep)", "(NEvnt,NRep)", "*.py")

findReplace("D:/root/Lehre/pulseqMRI/MRtwin_FAU/code/help","scanner.T ", "NEvnt", "*.py")
     
findReplace("D:/root/Lehre/pulseqMRI/MRtwin_FAU/code/help","T,", "NEvnt,", "*.py")
findReplace("D:/root/Lehre/pulseqMRI/MRtwin_FAU/code/help","T ", "NEvnt ", "*.py")
findReplace("D:/root/Lehre/pulseqMRI/MRtwin_FAU/code/help","T*NRep", "NEvnt*NRep", "*.py")

                
findReplace("C:/root/Lehre/pulseqMRI/MRtwin_FAU/code/MRtwin","import core.opt_helper", " ", "*.py")
findReplace("C:/root/Lehre/pulseqMRI/MRtwin_FAU/code/MRtwin","import core.FID_normscan", " ", "*.py")

#findReplace("C:/root/Lehre/pulseqMRI/MRtwin_FAU/code/MRtwin","grad_moms", "gradm_event", "*.py")
#findReplace("C:/root/Lehre/pulseqMRI/MRtwin_FAU/code/MRtwin","flips", "rf_event", "*.py")
#           
            
#%% ############################################################################
#also dirs
import os
replacement = """flips"""
for dname, dirs, files in os.walk("some_dir"):
    for fname in files:
        fpath = os.path.join(dname, fname)
        with open(fpath) as f:
            s = f.read()
        s = s.replace("{$replace}", replacement)
        with open(fpath, "w") as f:
            f.write(s)