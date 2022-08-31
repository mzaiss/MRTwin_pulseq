# MRTwin_pulseq

# MRTwin sequence programming and simulation framework #
MRTwin is part of a larger project for automatic sequence programming. Still all sequences can also be coded manually and simulated using the included Bloch simulation. 
The next sections give a short overview over the coding and simulation environment MRTwin.
Documentation of the tutorial: https://www.studon.fau.de/studon/goto.php?target=crs_2819947 

## Downloading Python, Pytorch and Spyder ##
(This part is not necessary in the CIP pool at FAU, as it is already installed).
Python is free to download and is available on all types of operating systems. We recommend to install Anaconda. For Linux see https://docs.anaconda.com/anaconda/install/linux/. For windows see https://www.anaconda.com/distribution/windows . 
In addition to python some extension packages are required like Pytorch. install them by using the following commands  
						

 -for pytorch find your correct command here: https://pytorch.org/   

```
 e.g. without gpu:
 conda install pytorch torchvision torchaudio cpuonly -c pytorch
 
 with gpu:
 conda install pytorch torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
```


- The simulation itself requires a compiled file, the pre_pass, to install it open a console, navigate to 'codes\GradOpt_python\new_core\pre_pass\wheels' and run

```
 On Windows:
 pip install wheels/pre_pass-0.2.0-cp37-abi3-win_amd64.whl

 On Linux:
 pre_pass-0.2.0-cp37-abi3-manylinux_2_27_x86_64
```

If not already installed you also need numpy, scipy and matplotlib

**Versions that were tested**

 -  torch.__version__  : '1.3.0'   and  '1.7.0'
 -  np.__version__ 	 : '1.18.1'    and  '1.19.2'
 -  scipy.__version__: '1.4.1'     and  '1.5.2'
 -   matplotlib.__version__: '3.1.1'

You can also run file ./ex/ex_help01_python.py to test versions.

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

Once this is set up, make the project code folder **MRTwin_pulseq/ex** your current folder. 


NOTE: To be able to go to definitions quickly (CTRL-click), you have to add the included paths ('./codes, ./codes/GradOpt\_python, ./codes/scannerloop\_libs)  again in your spyder path manually in the menu (Tools/PYTHONPATH):



If you want compare changed files or sample solutions you can use the tools diff, or more visually advanced Kompare or Meld.



## A brief introduction in python and torch. ##
We use python in here as it is open source and easy to debug and extend. Also with pytorch python provides a rather simple possibility of parallelization of code, auto-differentiation. \\
If you are not familiar with python, please make sure to understand the file ex_help01_python.py from the MRTwin code, as it covers most of the used functions in the whole code and course.

## MRTwin exercises ##
After obtaining the MRTwin package, find the main exercise files (exA0x ...) are in ./ex. For each file you should create a copy for your solution and name it solA0x, respectively.
The exercises are structured by sequence types:

 -  P : python tutorial
 -  A : basics and GRE
 -  B : spin echo and RARE
 -  C : balanced SSFP
 -  D : undersampling and reconstruction -  E : export to real system

