# MRTwin_pulseq

# MRTwin sequence programming and simulation framework
MRTwin is a tool to code sequences manually and simulate MR images using the included Bloch simulation. 
The Documentation/Script of the course is in /script.
It covers A : basic echoes in 1D, B : gradient echo encoding and fast GRE MRI, C : spin echo and RARE, D : balanced SSFP, E : exporting and scanning at a real MRI system, F : non-cartesian trajectories, undersampling and reconstruction.

## Course setup
The course format is full time two weeks. 
We meet  daily from 9:30 am to 5:30 pm (CET).
The first week is very interactive, with small inputs, and interleaved exercises and discussion of them.
The second week is more independent.

For virtual attendence read the [README_VIRTUAL.md](README_VIRTUAL.md).



## Installation
Check the readmes for your system:
- [README_HOME.md](README_HOME.md) for installation on your own Windows or Linux system, Mac is currently not supported.
- [README_CIP.md](README_CIP.md) for the installation in the FAU CIP pool.
- run script data/brainweb/generate_maps.py to generate brainweb phantoms

## Check the installation
See if you can run the python files ex/ex_help_01 and ex/exB05.
If errors occur check the versions of the required packages. Or get help from the admin.

## A brief introduction in python and torch.
We use python in here as it is open source and easy to debug and extend. Also with pytorch python provides a rather simple possibility of parallelization of code, auto-differentiation.

If you are not familiar with python, please make sure to understand the file ex_help01_python.py from the MRTwin code, as it covers most of the used functions in the whole code and course.

## Resources
- [Python](https://www.python.org/): Programming language
- [Pulseq](https://pulseq.github.io/): Open-source, vendor agnostic MRI sequence definition
- [pyPulseq](https://github.com/imr-framework/pypulseq): Pulseq port to python
- [pyTorch](https://pytorch.org/): Tensor computations with automatic differentiation and GPU acceleration
- [Bloch Simulator](https://www.drcmr.dk/BlochSimulator/): Visualization Tool (this repository contains an extension for Pulseq files)
- [LaTeX](https://www.latex-project.org/): Used for the script of the course
- [mrzero-core.readthedocs](https://mrzero-core.readthedocs.io/en/latest/intro.html)

