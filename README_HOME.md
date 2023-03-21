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


- The MRTwin_pulseq simulation itself requires the MRzero Core for simulation. Install it with:

```

 On Windows (requires python > 3.8):
 pip install MRzeroCore-0.1.0-cp38-none-win_amd64.whl
 
 On Windows (requires python > 3.7):
 pip install MRzeroCore-0.1.0-cp37-abi3-win_amd64.whl

 On Linux (requires python > 3.9):
 pip install MRzeroCore-0.1.0-cp39-cp39-manylinux_2_28_x86_64.whl
 
 On Mac:
 not yet supported
```

If not already installed you also need pypulseq, torchkbnufft, numpy, scipy and matplotlib.

The can be installed with
```
pip install pypulseq==1.3.1.post1
pip install torchkbnufft==1.3.0 --no-deps
pip install torchvision --no-deps
pip install numpy
pip install scipy
pip install matplotlib
```

**Versions that were tested**

 -  pypulseq.__version__ : '1.3.1post1'
 -  torch.__version__  : '1.3.0'   and  '1.7.0'
 -  np.__version__ 	 : '1.18.1'    and  '1.19.2'
 -  scipy.__version__: '1.4.1'     and  '1.5.2'
 -  matplotlib.__version__: '3.1.1'
 -  torchkbnufft.__version__: '1.3.0'

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

