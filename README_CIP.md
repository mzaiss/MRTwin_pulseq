# MRTwin_pulseq in the FAU CIP pool

## Installation and starting Spyder in the FAU-CIP pool ##

you can clone the current version into your home folder by opening a terminal in the home folder and run:
```
git clone -b mr0-core https://github.com/mzaiss/MRTwin_pulseq.git --depth 1
```

### Installation of MRTwin_pulseq
The cip-quota for disk space is limited, thus we need to make a symlink to the more relaxed /proj directory.
We do thsi for the python package installation folder site-packages.
This is done in the first 4 commands, the other 7 install packages an a specific environment python3/anaconda-2022.05

Run all of the following commands in a terminal:
```
mkdir /proj/ciptmp/$USER/
mv /home/cip/guests/$USER/.local/lib/python3.9/site-packages/ /home/cip/guests/$USER/.local/lib/python3.9/site-packages-backup
ln -s /proj/ciptmp/$USER/site-packages /home/cip/guests/$USER/.local/lib/python3.9/site-packages
mv /home/cip/guests/$USER/.local/lib/python3.9/site-packages-backup /proj/ciptmp/$USER/site-packages

module load python3/anaconda-2022.05
pip install mrzerocore
pip install pypulseq==1.3.1.post1
pip install torchkbnufft==1.3.0 --no-deps
pip install scikit-image
pip install PyWavelets
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

To have plots as separate window, which allows resizing and zooming, go to Tools->Preferences. Then on the rider IPythonKonsole go to Graphics and choose for the graphics backend: Automatic. 
Then you have to close and restart spyder.

I recommend to switch the layout to matlab layout. G to View->Layouts -> Matlab Layout or Rstudio Layout.


## Diff tool ##
If you want compare changed files or sample solutions you can use the tools diff, or more visually advanced Kompare or Meld.



# useful terminal commands:

arandr : screen position/projector

## quota and large files

cip-quota

ncdu

du -a |sort -n -r | head -n 20

pip cache purge

## git

git clone https://github.com/mzaiss/MRTwin_pulseq.git 

git pull

git difftool --tool kompare

git diff 
(press q to leave)
git diff --name-only
