
close all
clear all
addpath(genpath('D:\root\ZAISS_LABLOG\LOG_MPI\27_MRI_zero\mrizero_tueb\codes\SequenceSIM\3rdParty\pulseq-master'))

seq=mr.Sequence();              % Create a new sequence object

[FileName,PathName]=uigetfile('*.*','pick seq file','\\mrz3t\Upload\CEST_seq\pulseq_zero\sequences');
seq.read([PathName FileName],'detectRFuse');
% seq.read([PathName FileName]);
seq.plot();
deltak=1/220e-3

[ktraj_adc, ktraj, t_excitation, t_refocusing] = seq.calculateKspace('trajectory_delay',0);

figure; plot(ktraj'); % plot the entire k-space trajectory

% plot k-spaces
ktraj_adc=ktraj_adc(:,:)/deltak;
ktraj=ktraj(:,:)/deltak;
% 
% figure; plot(ktraj'); % plot the entire k-space trajectory
% figure(88); plot(ktraj(1,:),ktraj(2,:),'c',...
%     ktraj_adc(1,:),ktraj_adc(2,:),'go'); hold on;  % a 2D plot
% axis('equal'); % enforce aspect ratio for the correct trajectory display
% legend({'k-pulseq','ADC-pulseq'});

figure(333); plot3(ktraj(1,:),ktraj(2,:),ktraj(3,:),'c', ...
    ktraj_adc(1,:),ktraj_adc(2,:),ktraj_adc(3,:),'go'); hold on;  % a 3D plot

figure(334); plot3(ktraj_adc(1,:),ktraj_adc(2,:),ktraj_adc(3,:),'go'); hold on;  % a 3D plot
