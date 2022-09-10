# MRTwin_pulseq for your home system

##  Get the MRTwin_pulseq package
Go to 

 - https://github.com/mzaiss/MRTwin_pulseq

and click on the CODE button, then download zip, and unzip this file.

If you are familiar to git you can also run 

- git clone https://github.com/mzaiss/MRTwin_pulseq.git

## Downloading Python, Pytorch and Spyder ##
Python is free to download and is available on all types of operating systems. We recommend to install Anaconda. For Linux see https://docs.anaconda.com/anaconda/install/linux/. For windows see https://www.anaconda.com/distribution/windows . 
In addition to python some extension packages are required like Pytorch. Install them by using the following commands in the "Anaconda Prompt".
						
 -for pytorch find your correct command here: https://pytorch.org/   

```
 e.g. without gpu:
 conda install pytorch torchvision cpuonly -c pytorch
 
 with gpu:
 conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge
```


- The MRTwin_pulseq simulation itself requires a compiled file, the pre_pass, to install it open a console, navigate to 'codes\GradOpt_python\new_core\pre_pass\wheels' and run

```
 On Windows:
 pip install pre_pass-0.2.0-cp37-abi3-win_amd64.whl

 On Linux:
 pip install pre_pass-0.2.0-cp37-abi3-manylinux_2_27_x86_64.whl
```

If not already installed you also need numpy, scipy and matplotlib.

The can be installed with 
- pip install numpy
- pip install scipy
- pip install matplotlib

**Versions that were tested**

 -  torch.__version__  : '1.3.0'   and  '1.7.0'
 -  np.__version__ 	 : '1.18.1'    and  '1.19.2'
 -  scipy.__version__: '1.4.1'     and  '1.5.2'
 -   matplotlib.__version__: '3.1.1'

You can also run file ./ex/ex_help01_python.py to test versions.

## Starting Spyder at your own PC ##

Run Spyder from the anaconda environment.

### Check the installation
See if you can run the python files ex/ex_help_01_python and ex/exB05.
If errors occur check the versions of the required packages. Or get help from the admin.


## General Settings of Spyder ##

To have plots as separate window, go to Tools->Preferences. Then on the rider IPythonKonsole go to Graphics and choose for the graphics backend: Automatic. 

Then you have to close and restart spyder.

I recommend to switch the layout to matlab layout. Go to View->Layouts -> Matlab Layout or Rstudio Layout.

Once this is set up, make the project code folder **MRTwin_pulseq/ex** your current folder. 


To be able to go to definitions quickly (CTRL-click), you have to add the included paths ('./codes, ./codes/GradOpt\_python, ./codes/scannerloop\_libs)  again in your spyder path manually in the menu (Tools/PYTHONPATH):

### Diff tool

If you want compare changed files or sample solutions you can use the tools diff, or more visually advanced Kompare or Meld.

