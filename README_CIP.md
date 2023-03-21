# MRTwin_pulseq in the FAU CIP pool

## Installation and starting Spyder in the FAU-CIP pool ##

you can clone the current version into your home folder by opening a terminal in the home folder and run:
```
git clone -b mr0-core https://github.com/mzaiss/MRTwin_pulseq.git --depth 1
```

### Installation of MRTwin_pulseq
run the following commands in a terminal in your home folder (the same folder where "MRTwin_pulseq" is located):
```
module load python3/anaconda-2022.05
pip install MRTwin_pulseq/data/MRzeroCore-0.1.0-cp39-cp39-manylinux_2_28_x86_64.whl --force-reinstall 
pip install pypulseq==1.3.1.post1
pip install torchkbnufft==1.3.0 --no-deps
pip install torchvision --no-deps
```

### Start spyder
After running the installation you can start the environment by running in any terminal:
```
module load python3/anaconda-2022.05
spyder
```

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
