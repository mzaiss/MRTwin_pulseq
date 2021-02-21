cd('\\141.67.249.47\mrtransfer\mrizero_tueb\codes\GradOpt_python\matlab\ScannerReco')
% clear all; close all;
single = 0;
%%


niter = 17;

k = 1;
array = 1:1:niter;

% load tgt images
% sos_tgt_sim= abs(scanner_dict.(['target_sim_reco_' methodstr])(:,:,1)+1j*scanner_dict.(['target_sim_reco_' methodstr])(:,:,2));
% phase_tgt_sim = angle(squeeze(scanner_dict.(['target_sim_reco_' methodstr])(:,:,1)+1j*scanner_dict.(['target_sim_reco_' methodstr])(:,:,2)));
% 
% SAR_tgt_sim=sum(reshape((scanner_dict.target_flips(:,:,1).^2),1,[]));
% 
% if real_exists
%     sos_tgt_real= abs(scanner_dict.(['target_real_reco_' methodstr])(:,:,1)+1j*scanner_dict.(['target_real_reco_' methodstr])(:,:,2));
%     phase_tgt_real = angle(squeeze(scanner_dict.(['target_real_reco_' methodstr])(:,:,1)+1j*scanner_dict.(['target_real_reco_' methodstr])(:,:,2)));
% end

% load or calculate losses
% loss=scanner_dict.all_errors;
% SARloss=array*0;
% for ii=1:size(scanner_dict.all_flips,1)
%     SARloss(ii) = sum(reshape((squeeze(scanner_dict.all_flips(ii,:,:,1).^2)),1,[]))./SAR_tgt_sim*100;
% end
% TA_loss=array*0;
% for ii=1:size(scanner_dict.all_flips,1)
%     TA_loss(ii) = sum(reshape((abs(squeeze(scanner_dict.all_event_times(ii,:,:)))),1,[]));
% end

% plot iter 1 and tgt, define CLIMS
subaxis(3,4,3), imagesc(phantom_GT); title('T1 Map, Testset (GT)'), axis('image'); %colorbar;
ax=gca; CLIM_sim_tgt=ax.CLim;
set(gca,'xtick',[])
set(gca,'ytick',[])
colormap gray



subaxis(3,4,4), imagesc(flipud(FIT_full_rel)); title('T1 Map, target (inv. recovery, Fit)'), axis('image'); %colorbar;
ax=gca; CLIM_real_tgt=[0.5,4];
subaxis(3,4,4), imagesc(flipud(vivo_fit_T1),CLIM_real_tgt); title('T1 Map, target (inv. recovery, Fit)'), axis('image'); %colorbar;
ax=gca;
set(gca,'xtick',[])
set(gca,'ytick',[])
colormap gray
colorbar()

phantom_mask = phantom_GT > 1e-9;
array = 1:1:niter; % accelerate plot

if single>0
    array = single; % only a single frame to display
end

frames=cell(numel(array));
kplot=0;
for ii=array  % ii ist die laufvariable der bilder, f�r alle seq params gilt der index jj der sich aus ii berechnet.
    
%     iter_idx=[4999,5000,5005,5011,5016,5022,5028,5033,5039,5045,5050,5056,5062,5067,5073,5079,6079,6080,6085,6091,6096,6102,6108,6113,6119,6125,6130,6136,6142,6147,6153,6159,7159,7160,7165,7171,7176,7182,7188,7193,7199,7205,7210,7216,7222,7227,7233,7239,8209,8210,8212,8213,8215,8216,8218,8219,8221,8222,8224,8225,8227,8228,8230,8239]+1; % index for parameters
    iter_idx=49:50:850;
    jj = iter_idx(ii) + 1;
    if ii==1
        subaxis(3,4,7), h2=imagesc(squeeze(T1_maps(ii,:,:)),CLIM_sim_tgt); title(sprintf('Prediction Testset, iter %d,',round(jj/1.7))), axis('image'); %colorbar;
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        colormap gray
%         colorbar()
        %subaxis(3,4,2), h2=imagesc(sos_sim'); title(sprintf(' sos sim, iter %d, SAR %.1f',ii,SAR_sim)), axis('image'); %colorbar;
%         subaxis(3,4,6), h6=imagesc(patch_reco_cnn(ii,:,:),PCLIM); title(' phase (sim) '), axis('image'); %colorbar;
        subaxis(3,4,8), h6=imagesc(flipud(squeeze(T1_maps_NN(ii,:,:))),CLIM_real_tgt); title(sprintf('Prediction in vivo meas, iter %d,',round(jj/1.7))), axis('image'); %colorbar;
        set(gca,'xtick',[])
        set(gca,'ytick',[])
        colormap gray
        
        if real_exists
            subaxis(3,4,2), h3=imagesc(squeeze(patch_target(ii,:,:)),CLIM_real_tgt); title(sprintf('Target Training Sample, iter %d',round(jj/1.7))), axis('image'); %colorbar;
            set(gca,'xtick',[])
            set(gca,'ytick',[])
            colormap gray
            subaxis(3,4,6), h7=imagesc(squeeze(patch_reco_cnn(ii,:,:)),CLIM_real_tgt); title(sprintf('Prediction Training Sample, iter %d',round(jj/1.7))), axis('image'); %colorbar;
            set(gca,'xtick',[])
            set(gca,'ytick',[])
            colormap gray            
%             if kplot
%                 subaxis(3,4,1), h1=imagesc(squeeze(abs(scanner_dict.all_sim_kspace(ii,:,:,1)))); title(sprintf(' magnitude (sim), iter %d',ii)), axis('image'); %colorbar;
%                 subaxis(3,4,5), h5=imagesc(squeeze(abs(scanner_dict.all_real_kspace(ii,:,:,1)))); title(sprintf(' magnitude (real), iter %d',ii)), axis('image'); %colorbar;
%             end
        end
        
        
        fct = 1.3;
        %set(gcf, 'Outerposition',[137         296        1281         767]) % extralarge
%         set(gcf, 'Outerposition',[137*fct         296*fct        1281*fct         767*fct]) % extralarge
        set(gcf, 'Outerposition',[357         135        1302         750]) % extralarge
        
        set(gcf,'color','w');
        %   set(gcf, 'Outerposition',[404   356   850   592]) %large
        % set(gcf, 'Outerposition',[451   346   598   398]) % small
    end
    
    set(h2,'CDATA',squeeze(T1_maps(ii,:,:))); subaxis(3,4,7), title(sprintf('Prediction Testset, iter %d,',round(jj/1.7))),
    set(h6,'CDATA',squeeze(flipud(squeeze(T1_maps_NN(ii,:,:))))); subaxis(3,4,8), title(sprintf('Prediction in vivo meas, iter %d,',round(jj/1.7))),
    %   set(h2,'CDATA',sos_sim'); subaxis(3,4,2), title(sprintf(' magnitude (sim), iter %d',jj)),
%     set(h6,'CDATA',phase_sim');
    if real_exists
        %set(h3,'CDATA',sos_real); subaxis(3,4,3), title(sprintf(' sos real, iter %d, SAR %.2f',jj,SAR_sim)),
        set(h3,'CDATA',squeeze(patch_target(ii,:,:))); subaxis(3,4,2), title(sprintf('Target Training Sample, iter %d',round(jj/1.7))),
        set(h7,'CDATA',squeeze(patch_reco_cnn(ii,:,:))); subaxis(3,4,6), title(sprintf('Prediction Training Sample, iter %d',round(jj/1.7))),
%         if kplot
%             set(h1,'CDATA',squeeze(abs(scanner_dict.all_sim_kspace(ii,:,:,1)))); subaxis(3,4,1), title('ksim');
%             set(h5,'CDATA',squeeze(abs(scanner_dict.all_real_kspace(ii,:,:,1))));subaxis(3,4,5), title('kmeas');
%         end
    end
    
    % from SIM
%     deltak=1/220e-3;
%     % grad_moms = squeeze(scanner_dict.all_grad_moms(ii,:,:,:)*deltak);
%     % temp = squeeze(cumsum(grad_moms(:,:,1:2),1))/deltak;
%     temp = squeeze(scanner_dict.all_kloc(jj,:,:,:));
%     
%     % remove ADC
%     temp = temp(scanner_dict.target_adc_mask==1,:,:);
%     
%     subaxis(3,4,9),
%     ccc=colormap(gca,parula(size(temp,2)));
%     set(gca, 'ColorOrder', ccc, 'NextPlot', 'replacechildren');
%     plot(temp(:,:,1),temp(:,:,2),'-','DisplayName','k-sim'); hold on;% a 2D plot
%     % set(gca, 'ColorOrder',mr circshift(get(gca, 'ColorOrder'),-1));
%     plot(temp(3:end-2,:,1),temp(3:end-2,:,2),'.','MarkerSize',3,'DisplayName','k-sim'); % a 2D plot
%     ylabel('k-space sample locations');
%     hold off;
%     set(gca,'XTick',[-sz(1) sz(1)]/2); set(gca,'YTick',[-sz(2) sz(2)]/2); set(gca,'XTickLabel',{'-k','k'}); set(gca,'YTickLabel',[]);
%     grid on;
%     %set(gca, 'Position',get(gca,'Position').*[0.3 0.5 1.6 1.6]) % extralarge
%     axis([-1 1 -1 1]*sz(1)/2*1.5);
%     
%     
    subaxis(3,4,1), plot(abs(all_TIs(ii,:)),'bx','DisplayName','phase1'); hold on; plot(abs(all_waitings(ii,:))+abs(all_TIs(ii,:)),'rx','DisplayName','phase1');hold off; ylim([0,4]);ylabel('[s] TI/Trec');legend('TI','Trec')% a 2D plot
%     subaxis(3,4,11), plot(180/pi*squeeze(scanner_dict.all_flips(jj,2,:,2)),'b','DisplayName','phase2'); hold off; xlabel('repetition'); ylabel('RF (phase) [°]'); axis tight;
%     
%     subaxis(3,4,10), plot(180/pi*squeeze(scanner_dict.target_flips(1,:,1)),'r.','DisplayName','flips tgt'); hold on;% a 2D plot
%     subaxis(3,4,10), plot(180/pi*(squeeze(scanner_dict.all_flips(jj,1,:,1))),'r','DisplayName','flips');
%     subaxis(3,4,10), plot(180/pi*squeeze(scanner_dict.target_flips(2,:,1)),'b.','DisplayName','flips tgt'); % a 2D plot
%     subaxis(3,4,10), plot(180/pi*(squeeze(scanner_dict.all_flips(jj,2,:,1))),'b','DisplayName','flips');hold off;  xlabel('repetition'); ylabel(' angle [°]'); axis tight;
%     % subaxis(3,3,7), plot(180/pi*squeeze(scanner_dict.flips(ii,1,:,1)),'r--','DisplayName','flips'); hold off; axis([-Inf Inf 0 Inf]);
%     ylim= max([5, round( loss(jj)/(10^max([floor(log10(loss(jj))),0])))*(10^max([floor(log10(loss(jj))),0])*2)]);
    subaxis(3,4,5), plot((signal_csf(ii,:)),'o-','color',[0.49,0.18,0.56]); hold on; plot((signal_white(ii,:)),'o-','color',[0.47,0.67,0.19]);plot((signal_grey(ii,:)),'o-','color',[0.30,0.75,0.93]);hold off;ylabel('signal [a.u.]');legend('CSF','WM','GM')% a 2D plot
    subaxis(3,4,9),
    yyaxis left; plot(all_errors); hold on;  plot(jj,all_errors(jj),'b.'); plot(all_errors*0+min(all_errors(3:end)),'b:');xlabel('optimization iteration');hold off; axis([jj-50 jj+50 -10e-12 50]);ylabel('NRMSE error [%]');
    yyaxis right; plot(T_acq); hold on; plot(jj,T_acq(jj),'r.'); hold off; ylabel('Tacq [s]'); axis([jj-50 jj+50 0 30]);grid on;
    patch_mask = patch_target(ii,:) > 1e-9;
%     subaxis(3,4,10),
%     plot(phantom_GT(:).*phantom_mask(:), T1_maps(ii,:).*phantom_mask(:)','r.'); hold on; plot(patch_target(ii,:).*patch_mask, patch_reco_cnn(ii,:).*patch_mask,'b.');plot(0:6,0:6,'k');hold off;
%     h = legend('Testset','Training Sample','Location','southeast');
%     set(h,'Position', [0.269701198433903,0.279579057625898,0.10108864513656,0.049467274224559])
%     axis([-1,8,-1,8]) 
%     xlabel('target T1 [s]')
%     ylabel('pred. T1 [s]')
    h = subaxis(3,4,12);
%     set(h,'Position', [0.4058,0.1081,0.1625,0.2233])
    errorbar(T1_literature(1),mean_csf_fit(ii),std_csf_fit(ii),std_csf_fit(ii),T1_literature_std(1),T1_literature_std(1),'*','color',[0.49,0.18,0.56])
    hold on
    errorbar(T1_literature(2),mean_white_fit(ii),std_white_fit(ii),std_white_fit(ii),T1_literature_std(2),T1_literature_std(2),'*','color',[0.47,0.67,0.19])
    errorbar(T1_literature(3),mean_grey_fit(ii),std_grey_fit(ii),std_grey_fit(ii),T1_literature_std(3),T1_literature_std(3),'*','color',[0.30,0.75,0.93])
    plot(0:5,0:5)
    h = legend('WM','GM','CSF','y=x','Location','northwest');
%     set(h,'Position', [0.577110293095616,0.231286852420403,0.056765162787207,0.093607303395845])
    hold off
    axis([-1,8,-1,8]) 
    xlabel('T1 literature [s]')
    ylabel('pred. T1 [s]')
    
    % create gif (out.gif)
    % drawnow
    if (single==0)
        frames{ii} = getframe(1);
    end
    
%     if jj>=2
%         %        keyboard
%     end
    saveas(gcf,sprintf('C:/Users/danghi/Documents/MRzero/export/figures/gifs64/iter%d.fig',ii));
    
end
set(0, 'DefaultLineLineWidth', 0.5);

if (single==0)
    for ii=array
        im = frame2im(frames{ii});
        gifname=sprintf('C:/Users/danghi/Documents/MRzero/export/figures/gifs64/a_sim.gif');
        [imind,cm] = rgb2ind(im,32);
        if ii == 1
            imwrite(imind,cm,gifname,'gif', 'Loopcount',inf);
        elseif ii==array(end)
            imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',0.6);
        else
            imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',0.3);
        end
    end
    for ii=array
        im = frame2im(frames{ii});
        [imind,cm] = rgb2ind(im,32);
        if ii==array(end)
            imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',4);
        else
            imwrite(imind,cm,gifname,'gif','WriteMode','append','DelayTime',0.8);
        end
    end
end
saveas(gcf,sprintf('C:/Users/danghi/Documents/MRzero/export/figures/gifs64/lastSIM.fig'));
