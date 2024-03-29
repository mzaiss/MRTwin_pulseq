# BlochSimWeb

This is a version of  Lars Hanson's Bloch simulator
https://www.drcmr.dk/BlochSimulator/

with Pulseq extension

https://asd2511.xyz/BlochSimWeb/

.seq-files can be loaded only in Chrome browser until now.



## Update Log

### 2022.12.07
1. Add Event queue
2. Add speed control feature

#### Problem
If the shape of the RF puls is sinc func. Max speed up factor is 2.
If Block func, max speed up factor is 2.5.

### 2022.12.04

1. Fixed exotic color names being undefined in some environments.
2. Delayed B1 vector updating to avoid flickering (increase delayB1vecUpdate if not gone)

### 2022.11.30
1. Fix RF time
2. Fix Phase problem of RF puls

#### Known problems
1. Mix matter not working
2. In Some sequence gradPuls will over write each other
3. B1 arrow flickering - b1 view
#### Todo List
- [x] Use Gamma change the speed
- [x] Use Single Event fix the flecking problem (watch out the correction in the end of the puls)
- [x] Make a Event queue to make the clear event more elegant
- [ ] Compute the plan size?
- [x] Fix the color BLUE in mix matter case. 

### 2022.11.24
1. Fix flickering problem
2. Reduce GM time
3. Add adc display

### 2022.11.09
1. The speed control of decoupled RF PULS and gradient magnetic field.
2. Fix the color layer of Sequence Figure.
3. Remove *load_seq_filex01*,* load_seq_filex03*,* load_seq_filex10* button

### 2022.10.09

1. Analysis of Seq files.
2. RF puls translation.
3. Gradient magnetic field translation.
4. Re -configure the color of the sequence Figure.
5. Reset Function - non. rep. 

Known problems
1. FILE I/O can only be available in Chorme and Edge browser.
2. Sequence Figure in some cases, the visual effect is not good. Eg. EPI seq Gy
3. ~~Flickering B1 arrow~~
4. Sometimes B1 arrow error occurs.
