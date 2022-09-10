# MRTwin_pulseq in the FAU CIP pool

## Installation and starting Spyder in the FAU-CIP pool ##

you can clone the current version into your home folder by opening a terminal in the home folder and run:

- git clone https://github.com/mzaiss/MRTwin_pulseq.git

### Installation of MRTwin_pulseq
run the following commands in a terminal:
 1. module load python3/anaconda-2022.05  (or module load python) 
 2. module load torch
 3. navigate to 'MRTwin_pulseq\codes\GradOpt_python\new_core\pre_pass\wheels' and run

 - pip install pre_pass-0.2.0-cp37-abi3-manylinux_2_27_x86_64.whl

### Start spyder
After running the installation you can start the environment by:
 1. module load python3/anaconda-2022.05  (or module load python) 
 2. spyder

### Check the installation
See if you can run the python files ex/ex_help_01_python and ex/exB05.
If errors occur check the versions of the required packages. Or get help from the admin.

## General recommended settings in Spyder

To have plots as separate window, go to Tools->Preferences. Then on the rider IPythonKonsole go to Graphics and choose for the graphics backend: Automatic. 
Then you have to close and restart spyder.

I recommend to switch the layout to matlab layout. G to View->Layouts -> Matlab Layout or Rstudio Layout.

To be able to go to definitions quickly (CTRL-click), you have to add the included paths ('./codes, ./codes/GradOpt_python, ./codes/scannerloop_libs)  again in your spyder path manually in the menu (Tools/PYTHONPATH):


## Diff tool ##
If you want compare changed files or sample solutions you can use the tools diff, or more visually advanced Kompare or Meld.
