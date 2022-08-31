## pulled form the master branch in 08/2022


### version 2.1.x 
neue  version devJo branch - please tag
- dateien aufräumen, alter core raus
- segemented seqences anpassen ( ADC index)
- pulseq export mit balanced pre gzr
- standard examples re-implementieren.


### version 2.1.0 auf devJo
Erste version mit größeren (und vorerst letzten) API Änderungen bei Simulation & reco


### version 2.0.0_beta auf devJo
Snapshot von Arbeitszustand, aufgrund verschiedener Änderungen an diversen Funktionen
funktionieren nur manche Sequenz Skripte ohne Anpassungen.

Enthält noch alten `core`

### version 1.0.2
aktuelle version master branch - please tag - Felix

### version 1.0.1 
was tagged one year ago for new pulseq version

### version 0.7 - beta  

- scanner/scanner_fast unified (documentation pending)
- less remaining forward methods:
	- forward_mem
	- forward_fast 
	- forward_fast_sparse
	- forward_fast_sparse_supermem 

### TODO and wishlist for this betas alpha
- test case automap - **mo**

- forward_fast_flip(flip tensor sparse, gradient tensor sparse) - maybe rename to **forward_fast_spt**  al na - check opt of this method

- b10 - switch to sequence class super

- opt function: add parameter for adjusting momentum scheduling of ADAM- maybe less restarts possible - **al na**

- seq_holder for target_seq_holder and opt_helper (simplified pulseq  export for target, opt)

- semantic versioning https://semver.org/lang/de/   
as most recent changes are not downwards compatible this version must actually be 1.0.X

- renaming T -> NEvnt,  flips-> rf_event, gradmoms-> gradm_event, adc_mask-> adc_event (m0: added a function batch_rename, but we do this after ISMRM deadline)


### version 0.6 - beta  -pushed as reference for R2 paper evrsion 

- most old scripts and code removed / moved to oldcode folder
- in main folder now only running scripts as examples 
    - aux - for auxiliary files, mostly for seq-file generation
    - b0X - standard target sequence
    - m0x - mrzero optimization standard sequence, e.g. from paper etc.
- individual files now in folder "batch"
    - prefixes: Alex - al, Felix - fe, Moritz - Mo, Nam - Na, Simon - Si
- fov added to scanner
- new torch version 1.7: backwards compatible
- event_time_min required for both target seq and opt, targetSeq.set_event_time_min(event_time_min)
- phantom can be loaded by phantom = spins.get_phantom(sz[0],sz[1],type='object1')  # type='object1' or 'brain1'
- b1 plus can be set #scanner.set_B1plus(phantom[:,:,4])  # use as defined in phantom scanner.set_B1plus(1)  
- heuristic shift and number of coils adjustable: 
 scanner.get_signal_from_real_system(experiment_id,today_datestr,jobtype="iter", iterfile=iteration,h_shift=4,ncoils=20)
 - b0X scripts updated and pngs and seq files stored in test folder
 - pulseq interpreter updated - Version  XY?


### version 0.5

### Local folder structure for better documentation of experiments

Scripts for MRZero experiemts should be saved locally, instead inside the GitLab repository. We added new code to save a copy of the script file on a pre-defined basepath in a new experiment folder.
Inside the experiment folder a text file with the recent git commit hash is generated.

Experiment ID is now the same as the script file name.
All seq files include experiment id, MrZero Version number and git hash inside the header.


**TODO**

*  Adjust gradient moment for rewinder gradients, either by adjusting gradient amplitude or top flat time.


### v.0.4
