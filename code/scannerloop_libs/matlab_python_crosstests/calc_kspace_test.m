
clear all;
close all;

if isunix
    mrizero_git_dir = '/is/ei/aloktyus/git/mrizero_tueb';
    seq_dir = '/is/ei/aloktyus/git/mrizero_tueb/codes/scannerloop_libs/matlab_python_crosstests/seq_and_data';
else
    mrizero_git_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb';
    seq_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb/codes/scannerloop_libs/matlab_python_crosstests/seq_and_data';
end

addpath([ mrizero_git_dir,'/codes/SequenceSIM']);
addpath([ mrizero_git_dir,'/codes/SequenceSIM/3rdParty/pulseq-master/matlab/']);


scanner_dict = load([seq_dir,'/scanner_dict_tgt.mat']);

sz = double(scanner_dict.sz);

% gradient tranform
grad_moms = scanner_dict.grad_moms;

figure,
subplot(2,3,1), imagesc(scanner_dict.flips(:,:,1)'*180/pi); title('Flips'); colormap(gca,jet(fix(max(max(scanner_dict.flips(:,:,1)'*180/pi))))); colorbar; 
subplot(2,3,4), imagesc(scanner_dict.flips(:,:,2)'*180/pi); title('Phases'); colormap(gca,jet(fix(max(max(scanner_dict.flips(:,:,2)'*180/pi))/20))); colorbar;
subplot(2,3,2), imagesc(scanner_dict.event_times'); title('delays');colorbar
subplot(2,3,3), imagesc(grad_moms(:,:,1)');         title('gradmomx');colorbar
subplot(2,3,6), imagesc(grad_moms(:,:,2)');          title('gradmomy');colorbar
set(gcf,'OuterPosition',[431         379        1040         513])
% plug learned gradients into the sequence constructor
% close all
seq_fn = [seq_dir,'/test.seq'];

FOV = 220e-3;

% Define other gradients and ADC events
deltak=1/FOV;
% read gradient

T = size(scanner_dict.grad_moms,1);
NRep = size(scanner_dict.grad_moms,2);
flips = double(squeeze(scanner_dict.flips(:,:,:)));
event_times = double(squeeze(scanner_dict.event_times(:,:)));
gradmoms = double(squeeze(scanner_dict.grad_moms(:,:,:)))*deltak;  % that brings the gradmoms to the k-space unit of deltak =1/FoV

seq = mr.Sequence();
seq.read(seq_fn);

%% new single-function call for trajectory calculation
[ktraj_adc, ktraj, t_excitation, t_refocusing] = seq.calculateKspace();

% plot k-spaces
ktraj_adc=ktraj_adc/deltak;
ktraj=ktraj/deltak;
figure; plot(ktraj'); % plot the entire k-space trajectory
figure(88); plot(ktraj(1,:),ktraj(2,:),'c',...
    ktraj_adc(1,:),ktraj_adc(2,:),'go'); hold on;  % a 2D plot
axis('equal'); % enforce aspect ratio for the correct trajectory display
legend({'k-pulseq','ADC-pulseq'});

% from SIM
grad_moms = squeeze(scanner_dict.grad_moms);
grad_moms =cat(1,zeros(1,NRep,2),grad_moms);
temp = squeeze(cumsum(grad_moms(:,:,1:2),1));
ktraj_adc_sim_x =temp(:,:,1);  ktraj_adc_sim_x =ktraj_adc_sim_x(:);
ktraj_adc_sim_y =temp(:,:,2);  ktraj_adc_sim_y =ktraj_adc_sim_y(:);

figure(88); plot(ktraj_adc_sim_x,ktraj_adc_sim_y,'bx-','DisplayName','k-sim'); hold on;% a 2D plot

%   ktraj_adc_sim = ktraj_adc_sim(3:end-2,:,:);
%   ktraj_adc_temp = reshape(permute(ktraj_adc_sim,[3,2,1]),2,[]);





