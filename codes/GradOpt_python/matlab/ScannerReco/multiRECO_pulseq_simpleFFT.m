function multiRECO_pulseq_simpleFFT
%% very simple FFT reconstruction from raw data
addpath('D:\root\ZAISS_LABLOG\LOG_MPI\0_CESTtool\CEST_EVAL_GLINT\reco_raw\recoVBVD')
addpath('D:\root\ZAISS_LABLOG\LOG_MPI\0_CESTtool\CEST_EVAL_GLINT\reco_raw\mapVBVD')
%% Load raw data
[filename,rawpname] = uigetfile({'*.dat'},['select raw fils for reconstruction'],'MultiSelect','on','\\mrz3t\Upload\CEST_seq\pulseq_zero\sequences');
if iscell(filename)==0
    temp=filename; clear filename;
    filename{1}=temp; clear temp;
end

origpath=pwd;

set(0, 'DefaultLineLineWidth', 2);
figure(1)
array=1:numel(filename);

twix_obj = mapVBVD([rawpname '/' filename{1}]);

seq=mr.Sequence();
try
seq.read([rawpname '/' sprintf('pulseq.seq')],'detectRFuse');
catch
seq.read([rawpname '/' sprintf('seqiter0.seq')],'detectRFuse');
end
seq.plot();

[ktraj_adc, ktraj] = seq.calculateKspace();
figure(88); plot(ktraj(1,:),ktraj(2,:),'c',...
    ktraj_adc(1,:),ktraj_adc(2,:),'go'); hold on;  % a 2D plot

scanner_dict_tgt = load([rawpname,'/','scanner_dict_tgt.mat']);

   [sos_base, phase_base] = TWIXtoIMG_ADJOINT_gradmoms_from_pulseq(twix_obj,ktraj_adc);
%   [sos_base, phase_base] = TWIXtoIMG_NUFFT_gradmoms_from_pulseq(twix_obj,ktraj_adc);
%MEAS


for ii=array
    
%twix_obj = mapVBVD([rawpname '/' filename{ii}]);
%[sos, phase] = TWIXtoIMG(twix_obj);

twix_obj=mapVBVD([rawpname '/' filename{ii}]);
[sos, phase] = TWIXtoIMG_ADJOINT_gradmoms_from_pulseq(twix_obj,ktraj_adc);

figure(1),
subplot(2,2,1), imagesc((sos),[0 1]), title(sprintf('reco sos, iter %d',ii)), axis('image'); colorbar;
subplot(2,2,3), imagesc((phase)), title('reco phase coil(1) '), axis('image'); colorbar;
subplot(2,2,2), imagesc((sos_base),[0 1]), title(sprintf('MEAS reco sos, iter %d',1)), axis('image'); colorbar;
subplot(2,2,4), imagesc((phase_base)), title('reco phase coil(1) '), axis('image'); colorbar;
set(gcf, 'Outerposition',[404   356   850   592])

% create gif (out.gif)
% drawnow
%       frame = getframe(1);
%       im = frame2im(frame);
%       im_MEAS(:,:,:,ii)=im;
%       [imind,cm] = rgb2ind(im,32);
%       if ii == 1
%           imwrite(imind,cm,'out.gif','gif', 'Loopcount',inf);
%       elseif ii==numel(array)
%           imwrite(imind,cm,'out.gif','gif','WriteMode','append','DelayTime',3);
%       else
%           imwrite(imind,cm,'out.gif','gif','WriteMode','append','DelayTime',0.0005);
%       end
end
set(0, 'DefaultLineLineWidth', 0.5);
last_fn_meas=filename{ii};
% save('togif.mat','im_MEAS','last_fn_meas');
end

