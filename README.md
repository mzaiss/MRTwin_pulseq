# MRTwin_pulseq

# MRTwin sequence programming and simulation framework #
MRTwin is part of a larger project for automatic sequence programming. Still all sequences can also be coded manually and simulated using the included Bloch simulation. 
The next sections give a short overview over the coding and simulation environment MRTwin.

## Downloading Python, Pytorch and Spyder ##
(This part is not necessary in the CIP pool at FAU, as it is already installed).
Python is free to download and is available on all types of operating systems. We recommend to install Anaconda. For Linux see https://docs.anaconda.com/anaconda/install/linux/. For windows see https://www.anaconda.com/distribution/windows . 
In addition to python some extension packages are required like Pytorch. install them by using the following commands  
						

 - pip install opencv-python
 -pip install termcolor
 -pip install nevergrad
 -for pytorch find your correct command here: https://pytorch.org/   
					e.g. without gpu:
          conda install pytorch torchvision cpuonly -c pytorch
					with gpu
					conda install pytorch torchvision cudatoolkit=10.1 -c pytorch


**Versions that were tested**

 -  torch.__version__  : '1.3.0'   and  '1.7.0'
 -  np.__version__ 	 : '1.18.1'    and  '1.19.2'
 -  scipy.__version__: '1.4.1'     and  '1.5.2'
 -   matplotlib.__version__: '3.1.1'

You can also run file code/MRTwin/exP01.py to test versions.

## Starting Spyder at your own PC ##

Copy the folder **MRTwin_pulseq** to a folder on your local PC.
Run spyder from the anaconda environment.

## Starting Spyder in the FAU-CIP pool ##
All project file will be in the folder 

**/Proj/ciptmp/zaissmz**

Copy the folder **MRTwin_pulseq** to your home folder.

To start spyder run the following commands in a terminal:
 1. module load python
 2. spyder

## General Settings ##

To have plots as separate window, go to Tools->Preferences. Then on the rider IPythonKonsole go to Graphics and choose for the graphics backend: Automatic. 

Then you have to close and restart spyder.

I recommend to switch the layout to matlab layout. G to View->Layouts -> Matlab Layout or Rstudio Layout.

Once this is set up, make the project code folder **MRTwin_pulseq/code/MRtwin** your current folder. 

If you want compare changed files or sample solutions you can use the tools diff, or more visually advanced Kompare or Meld.

## A brief introduction in python and torch. ##
We use python in here as it is open source and easy to debug and extend. Also with pytorch python provides a rather simple possibility of parallelization of code, auto-differentiation. \\
If you are not familiar with python, please make sure to understand the file exP01.py from the MRTwin code, as it covers most of the used functions in the whole code and course.

## MRTwin exercises ##
After obtaining the MRTwin package, find the main exercise files (exA0x ...) are in ./code/MRTwin. For each file you should create a copy for your solution and name it solA0x, respectively.
The exercises are structured by sequence types:

 -  P : python tutorial
 -  A : basics and GRE
 -  B : spin echo and RARE
 -  C : stimulated echo
 -  D : balanced SSFP
 -  E : export to real system
 -  F : undersampling and reconstruction
