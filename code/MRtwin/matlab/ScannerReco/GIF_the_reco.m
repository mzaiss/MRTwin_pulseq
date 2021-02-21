
clear all; close all;
single = 0;
%%
if isunix
    mrizero_git_dir = '/is/ei/aloktyus/git/mrizero_tueb';
    experiment_id = 'p10_tgtGRESPcartSteadyState_tskFLASHradial_FA_ET_48';
    sdate = '190802';
    seq_dir = ['/media/upload3t/CEST_seq/pulseq_zero/sequences/results/seq',sdate];
else
    mrizero_git_dir = 'D:/root/ZAISS_LABLOG/LOG_MPI/27_MRI_zero/mrizero_tueb';
    %   d = uigetdir('\\mrz3t\Upload\CEST_seq\pulseq_zero\sequences\results', 'Select a sequence folder');
%     d = uigetdir('\\141.67.249.47\MRtransfer\pulseq_zero\sequences\results', 'Select a sequence folder');
    d = uigetdir('D:\root\LABLOG\FAU\1_MRzero\mrzero_src\codes\GradOpt_python\out\sequences\results', 'Select a sequence folder');
    seq_dir=[d '\..'];
    out=regexp(d,'\','split');
    experiment_id=out{end};
end

% addpath([ mrizero_git_dir,'/codes/SequenceSIM']);
% addpath([ mrizero_git_dir,'/codes/SequenceSIM/3rdParty/pulseq-master/matlab/']);


% methodstr='generalized_adjoint';
%methodstr='nufft';
methodstr='adjoint';
%methodstr='generalized_adjoint';

ni = 30;

try
    scanner_dict = load([seq_dir,'/',experiment_id,'/','all_meas_reco_dict.mat']);
    real_exists=1;
catch
    real_exists=0;
    scanner_dict = load([seq_dir,'/',experiment_id,'/','all_sim_reco_dict.mat']);
end

sz = double(scanner_dict.sz);
T = scanner_dict.T;
NRep = scanner_dict.NRep;

niter = numel(scanner_dict.iter_idx);

%niter = 22;

k = 1;
array = 1:1:niter;

% load tgt images
sos_tgt_sim= abs(scanner_dict.(['target_sim_reco_' methodstr])(:,:,1)+1j*scanner_dict.(['target_sim_reco_' methodstr])(:,:,2));
phase_tgt_sim = angle(squeeze(scanner_dict.(['target_sim_reco_' methodstr])(:,:,1)+1j*scanner_dict.(['target_sim_reco_' methodstr])(:,:,2)));

SAR_tgt_sim=sum(reshape((scanner_dict.target_flips(:,:,1).^2),1,[]));

if real_exists
    sos_tgt_real= abs(scanner_dict.(['target_real_reco_' methodstr])(:,:,1)+1j*scanner_dict.(['target_real_reco_' methodstr])(:,:,2));
    phase_tgt_real = angle(squeeze(scanner_dict.(['target_real_reco_' methodstr])(:,:,1)+1j*scanner_dict.(['target_real_reco_' methodstr])(:,:,2)));
end

% load or calculate losses
loss=scanner_dict.all_errors;
SARloss=array*0;
for ii=1:size(scanner_dict.all_flips,1)
    SARloss(ii) = sum(reshape((squeeze(scanner_dict.all_flips(ii,:,:,1).^2)),1,[]))./SAR_tgt_sim*100;
end
TA_loss=array*0;
for ii=1:size(scanner_dict.all_flips,1)
    TA_loss(ii) = sum(reshape((abs(squeeze(scanner_dict.all_event_times(ii,:,:)))),1,[]));
end

% plot iter 1 and tgt, define CLIMS
subaxis(3,4,1), imagesc(sos_tgt_sim'); title('magnitude, target (sim)'), axis('image'); %colorbar;
ax=gca; CLIM_sim_tgt=ax.CLim;

subaxis(3,4,5), imagesc(phase_tgt_sim',[-pi pi]), title(' phase target (sim)'), axis('image'); %colorbar;
ax=gca; PCLIM=ax.CLim;

if real_exists
    subaxis(3,4,4), imagesc(sos_tgt_real'); title('magnitude, target (meas)'), axis('image'); %colorbar;
    ax=gca; CLIM_real_tgt=ax.CLim/40;
    subaxis(3,4,4), imagesc(sos_tgt_real',CLIM_real_tgt); title('magnitude, target (meas)'), axis('image'); %colorbar;
    subaxis(3,4,8), imagesc(phase_tgt_real'), title(' phase target (meas)'), axis('image'); %colorbar;
    ax=gca;
end

array = 1:1:niter; % accelerate plot

if single>0
    array = single; % only a single frame to display
end

frames=cell(numel(array));
kplot=0;
for ii=array  % ii ist die laufvariable der bilder, für alle seq params gilt der index jj der sich aus ii berechnet.
    
    jj=scanner_dict.iter_idx(ii)+1; % index for parameters
    
    sos_sim = abs(squeeze(scanner_dict.(['all_sim_reco_' methodstr])(ii,:,:,1)+1j*scanner_dict.(['all_sim_reco_' methodstr])(ii,:,:,2)));
    phase_sim = angle(squeeze(scanner_dict.(['all_sim_reco_' methodstr])(ii,:,:,1)+1j*scanner_dict.(['all_sim_reco_' methodstr])(ii,:,:,2)));
    
    if real_exists
        sos_real = abs(squeeze(scanner_dict.(['all_real_reco_' methodstr])(ii,:,:,1)+1j*scanner_dict.(['all_real_reco_' methodstr])(ii,:,:,2)));
        phase_real = angle(squeeze(scanner_dict.(['all_real_reco_' methodstr])(ii,:,:,1)+1j*scanner_dict.(['all_real_reco_' methodstr])(ii,:,:,2)));
    end
    
    if ii==1
        subaxis(3,4,2), h2=imagesc(sos_sim',CLIM_sim_tgt); title(sprintf(' sos sim, iter %d, SAR %.1f',jj,SARloss(jj))), axis('image'); %colorbar;
        %subaxis(3,4,2), h2=imagesc(sos_sim'); title(sprintf(' sos sim, iter %d, SAR %.1f',ii,SAR_sim)), axis('image'); %colorbar;
        subaxis(3,4,6), h6=imagesc(phase_sim',PCLIM); title(' phase (sim) '), axis('image'); %colorbar;
        
        if real_exists
            subaxis(3,4,3), h3=imagesc(sos_real',CLIM_real_tgt); title(sprintf(' sos real, iter %d, SAR %.1f',jj,SARloss(jj))), axis('image'); %colorbar;
            subaxis(3,4,7), h7=imagesc(phase_real',PCLIM); title(' phase (meas) '), axis('image'); %colorbar;
            if kplot
                subaxis(3,4,1), h1=imagesc(squeeze(abs(scanner_dict.all_sim_kspace(ii,:,:,1)))); title(sprintf(' magnitude (sim), iter %d',ii)), axis('image'); %colorbar;
                subaxis(3,4,5), h5=imagesc(squeeze(abs(scanner_dict.all_real_kspace(ii,:,:,1)))); title(sprintf(' magnitude (real), iter %d',ii)), axis('image'); %colorbar;
            end
        end
        
        
        fct = 1.3;
        %set(gcf, 'Outerposition',[137         296        1281         767]) % extralarge
%         set(gcf, 'Outerposition',[137*fct         296*fct        1281*fct         767*fct]) % extralarge
        set(gcf, 'Outerposition',[357         135        1302         750]) % extralarge
        
        set(gcf,'color','w');
        %   set(gcf, 'Outerposition',[404   356   850   592]) %large
        % set(gcf, 'Outerposition',[451   346   598   398]) % small
    end
    
    set(h2,'CDATA',sos_sim'); subaxis(3,4,2), title(sprintf(' sos sim, iter %d, SAR %.2f,\n TA %.2f, meanTR %.2f',jj,SARloss(jj),TA_loss(jj),TA_loss(jj)/sz(1))),
    %   set(h2,'CDATA',sos_sim'); subaxis(3,4,2), title(sprintf(' magnitude (sim), iter %d',jj)),
    set(h6,'CDATA',phase_sim');
    if real_exists
        %set(h3,'CDATA',sos_real); subaxis(3,4,3), title(sprintf(' sos real, iter %d, SAR %.2f',jj,SAR_sim)),
        set(h3,'CDATA',sos_real'); subaxis(3,4,3), title(sprintf(' magnitude (meas), iter %d',jj)),
        set(h7,'CDATA',phase_real');
        if kplot
            set(h1,'CDATA',squeeze(abs(scanner_dict.all_sim_kspace(ii,:,:,1)))); subaxis(3,4,1), title('ksim');
            set(h5,'CDATA',squeeze(abs(scanner_dict.all_real_kspace(ii,:,:,1))));subaxis(3,4,5), title('kmeas');
        end
    end
    
    % from SIM
    deltak=1/220e-3;
    % grad_moms = squeeze(scanner_dict.all_grad_moms(ii,:,:,:)*deltak);
    % temp = squeeze(cumsum(grad_moms(:,:,1:2),1))/deltak;
    temp = squeeze(scanner_dict.all_kloc(jj,:,:,:));
    
    % remove ADC
    temp = temp(scanner_dict.target_adc_mask==1,:,:);
    
    subaxis(3,4,9),
    ccc=colormap(gca,parula(size(temp,2)));
    set(gca, 'ColorOrder', ccc, 'NextPlot', 'replacechildren');
     plot(temp(:,:,1),temp(:,:,2),'.-','DisplayName','k-sim','MarkerSize',5); hold on;% a 2D plot
    % set(gca, 'ColorOrder',mr circshift(get(gca, 'ColorOrder'),-1));
%     plot(temp(3:end-2,:,1),temp(3:end-2,:,2),'.','MarkerSize',3,'DisplayName','k-sim'); % a 2D plot
    ylabel('k-space sample locations');
    hold off;
    set(gca,'XTick',[-sz(1) sz(1)]/2); set(gca,'YTick',[-sz(2) sz(2)]/2); set(gca,'XTickLabel',{'-k','k'}); set(gca,'YTickLabel',[]);
    grid on;
    %set(gca, 'Position',get(gca,'Position').*[0.3 0.5 1.6 1.6]) % extralarge
    axis([-1 1 -1 1]*sz(1)/2*1.5);
    
    
    subaxis(3,4,11), plot(180/pi*squeeze(scanner_dict.all_flips(jj,1,:,2)),'r','DisplayName','phase1'); hold on;% a 2D plot
    subaxis(3,4,11), plot(180/pi*squeeze(scanner_dict.all_flips(jj,2,:,2)),'b','DisplayName','phase2'); hold off; xlabel('repetition'); ylabel('RF (phase) [ °]'); axis tight;
    
    subaxis(3,4,10), plot(180/pi*squeeze(scanner_dict.target_flips(1,:,1)),'r.','DisplayName','flips tgt'); hold on;% a 2D plot
    subaxis(3,4,10), plot(180/pi*(squeeze(abs(scanner_dict.all_flips(jj,1,:,1)))),'r','DisplayName','flips');
    subaxis(3,4,10), plot(180/pi*squeeze(scanner_dict.target_flips(2,:,1)),'b.','DisplayName','flips tgt'); % a 2D plot
    subaxis(3,4,10), plot(180/pi*(squeeze(abs(scanner_dict.all_flips(jj,2,:,1)))),'b','DisplayName','flips');hold off;  xlabel('repetition'); ylabel('flip angle [ °]'); axis tight;
    % subaxis(3,3,7), plot(180/pi*squeeze(scanner_dict.flips(ii,1,:,1)),'r--','DisplayName','flips'); hold off; axis([-Inf Inf 0 Inf]);
    ylim= max([5, round( loss(jj)/(10^max([floor(log10(loss(jj))),0])))*(10^max([floor(log10(loss(jj))),0])*2)]);
    subaxis(3,4,12),
    yyaxis left; plot(loss); hold on;  plot(jj,loss(jj),'b.'); plot(loss*0+min(loss(3:end)),'b:');xlabel('optimization iteration');hold off; axis([jj-50 jj+50 -10e-12 ylim]);ylabel('[%] NRMSE error');
    yyaxis right; plot(SARloss); hold on; plot(jj,SARloss(jj),'r.'); hold off; ylabel('[%] SAR'); grid on;
    
    
    % create gif (out.gif)
    % drawnow
    if (single==0)
        frames{ii} = getframe(1);
    end
    
    if jj>=2
        %        keyboard
    end
    saveas(gcf,sprintf('%s/%s/iter%d.fig',seq_dir,experiment_id,jj));
    
end
set(0, 'DefaultLineLineWidth', 0.5);

if (single==0)
    for ii=array
        im = frame2im(frames{ii});
        gifname=sprintf('%s/%s/a_sim_%s_%s.gif',seq_dir,experiment_id,experiment_id,methodstr);
        [imind,cm] = rgb2ind(im,32);
        if ii == 1
            imwrite(imind,cm,gifname,'gif', 'Loopcount',inf);
        elseif ii==array(end)
            imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',3);
        else
            imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',0.6);
        end
    end
end
saveas(gcf,sprintf('%s/%s/lastSIM.fig',seq_dir,experiment_id));
