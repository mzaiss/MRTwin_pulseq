# MRTwin_pulseq

# MRTwin sequence programming and simulation framework #
MRTwin is part of a larger project for automatic sequence programming. We use it here to code manually and simulate MR images using the included Bloch simulation. 
Documentation/Script of the course: https://www.studon.fau.de/studon/goto.php?target=crs_2819947 
It covers A : basic echoes in 1D, B : gradient echo encoding and fast GRE MRI, C : spin echo and RARE, D : balanced SSFP, E : exporting and scanning at a real MRI system, F : non-cartesian trajectories, undersampling and reconstruction.

## Installation
Check the readmes for your system:
- [README_HOME.md](README_HOME.md) for installation on your own Windows or Linux system, Mac is currently not supported.
- [README_CIP.md](README_CIP.md) for the installation in the FAU CIP pool.

## Check the installation
See if you can run the python files ex/ex_help_01 and ex/exB05.
If errors occur check the versions of the required packages. Or get help from the admin.

## A brief introduction in python and torch. ##
We use python in here as it is open source and easy to debug and extend. Also with pytorch python provides a rather simple possibility of parallelization of code, auto-differentiation. \\
If you are not familiar with python, please make sure to understand the file ex_help01_python.py from the MRTwin code, as it covers most of the used functions in the whole code and course.
