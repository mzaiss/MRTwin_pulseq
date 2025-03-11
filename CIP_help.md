TUTOR:

/proj/ciptmp/zaissmz/


 ### useful terminal commands:###

arandr : screen position/projector


module load faubox
faubox

firefox

## git ###


git clone https://github.com/mzaiss/MRTwin_pulseq.git 

git pull

git difftool --tool kompare

git diff 
(press q to leave)
git diff --name-only

git add

git add -u   ( only modified files)

git commit

git push


STUDENT: 

Copy 



module load python
spyder


## VS CODE
Open VS Code's command palette (Ctrl+Shift+P)
Type "Python: Select Interpreter"
Look for the Anaconda Python path which should be something like:

/local/python3.9-Anaconda3-2022.05/bin/python

To get plots right and ineractive you might want to add:

import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
plt.ion()  # Turn on interactive mode

and at the end of te script, to avid closing bz garbage collection: 

plt.ioff()  # Turn on interactive mode
plt.show()

