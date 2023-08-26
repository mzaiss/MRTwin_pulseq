# MRTwin_pulseq for your home system

##  Get the MRTwin_pulseq package
 - If you are familiar to git you can run: ``git clone -b mr0-core https://github.com/mzaiss/MRTwin_pulseq.git --depth 1``
 - If not, you can download the code as zip file and extract it somewhere
    - Download the zip file here https://github.com/mzaiss/MRTwin_pulseq/archive/refs/heads/mr0-core.zip
    - Alternatively, go to https://github.com/mzaiss/MRTwin_pulseq switch to the ``mr0-core`` branch on the top left and click on the ``Code`` button, then download zip.


## Downloading Python, Pytorch and Spyder ##
Python is free to download and is available on all types of operating systems. We recommend to install Anaconda. For Linux see https://docs.anaconda.com/anaconda/install/linux/. For windows see https://www.anaconda.com/distribution/windows . 
In addition to python some extension packages are required like Pytorch. Install them by using the following commands in the "Anaconda Prompt".
						
 - for pytorch find your correct command here: https://pytorch.org/   

```
 e.g. without gpu:
 conda install pytorch torchvision cpuonly -c pytorch
 
 with gpu:
 conda install pytorch torchvision cudatoolkit=11.6 -c pytorch -c conda-forge
```


- The MRTwin_pulseq simulation itself requires the MRzero Core for simulation. Install it with (Windows or Linux, Mac not supported):

```
pip install mrzerocore
```

For this course, there are two additional dependencies:

```
pip install torchkbnufft==1.3.0 --no-deps
pip install torchvision --no-deps
pip install numpy==1.23.5
pip install scipy
pip install matplotlib
```

**Dependency Versions**

If there are compatibility problems with the installed versions, you can run file ./ex/ex_help01_python.py to test versions.

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

