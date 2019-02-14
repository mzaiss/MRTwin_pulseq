function [FLAIR FLAIRpvcorr newMaps sig_CSFrem] = fn_syntheticFLAIR_PVcorr(FLAIRparams,recon_images,t1im,t2im,dfim,m0im,t1t2CSF,frac,seqType,dict,r,dfrange,index,plotflag,rotfactor)
% [FLAIR FLAIRpvcorr newMaps sig_CSFrem] = fn_syntheticFLAIR_PVcorr(FLAIRparams,recon_images,t1im,t2im,dfim,m0im,t1t2CSF,frac,seqType,dict,r,dfrange,index,plotflag,rotfactor)
% Function computes synthetic FLAIR image from MRF data with and without
% CSF partial volume correction.
% Reference: Deshmane et al. Proc ISMRM 2016 p 1909.
% 
% INPUTS
%1     FLAIRparams     matrix      [1 x 3] TI, TR, TE of synthetic FLAIR image
%2     recon_images    matrix      [Nimage x siz x siz] MRF image series
%3     t1im            matrix      [siz x siz] T1 map from MRF
%4     t2im            matrix      [siz x siz] T2 map from MRF
%5     dfim            matrix      [siz x siz] df map for IRT, 0 for FISP
%6     m0im            matrix      [siz x siz] M0 map from MRF
%7     t1t2CSF         matrix      [1 x 2] T1 & T2 values of CSF
%8     fracCSF         matrix      [siz x siz] complex weights of CSF
%                                       fraction corresponding to t1t2
%                                       values above
%9     seqType         char        MRF sequence 'IRT' or 'FISP' (default FISP)
%10    dict            matrix      [Nimage x Nelts x Dfrange] MRF dictionary
%11    r               matrix      [Nelts x 2] T1 T2 pairs in dictionary
%12    dfrange         matrix      [1 x Ndf] IRT off-resonance values, 0 for FISP
%13    index           matrix      index in mask
%14    plotflag        binary      1 - show fraction maps; 0 - don't show
%15    rotfactor       double      rot90 rotation factor for visualization
%
% OUTPUTS
%1     FLAIR            matrix      [siz x siz] Unnormalized complex weights
%2     FLAIRpvcorr      matrix      [siz x siz] Normalized positive real fractions
%3     newMaps          struct      [siz x siz] T1, T2, M0, and df maps
%                                       after CSF partial volume removed
%4      sig_CSFrem       matrix      [Nimage x siz x siz] MRF image series
%                                       after CSF partial volume removed
%   
% This code written and maintained by: 
% Anagha Deshmane, Case Western Reserve University
% avd9@case.edu
% 
% First written October 2015
% Update history:
% August 30, 2016    Commented for Archiving
% January 5, 2017    Speedup CSF removal by converting pixelwise calculation to matrix
%                       operation
%
%% Check inputs
if nargin<8
    error('synthetic FLAIR - too few inputs')
end
% recon images
% [m mloc]=max(size(recon_images));
% if mloc ~=1
%     ndims = length(size(recon_images));
%     recon_images = permute(recon_images,[mloc setdiff(1:ndims, mloc)]);
%     clear ndims m mloc
% end
[Nimages siz ~] = size(recon_images);
% T1, T2 of CSF
if t1t2CSF(2)>t1t2CSF(1),
    t1t2CSF = [t1t2CSF(2) t1t2CSF(1)]; % should be [T1, T2]
end
% seqtype and dfim
switch seqType
    case 'FISP'
        dfim=0; dfrange=0;
    case 'IRT'
        % df range dimension
    otherwise
        error('PVMRF - Sequence type unknown')
end
% dict
ndims = length(size(dict));
if strcmp(seqType,'FISP') && ndims>2
    error('PVMRF - incorrect sequence type')
end
if size(dict,1)~=(Nimages+[-1:0]),  
    [m mloc] = find(size(dict)==intersect(size(dict),(Nimages+[-1:0])));
    dict = permute(dict,[mloc setdiff(1:ndims, mloc)]);
    clear ndims m mloc
end
if strcmp(seqType,'IRT')
    [Nimages Nelts Ndf] = size(dict);
    if size(recon_images,1)>Nimages,
        recon_images = recon_images(2:end,:,:);
    end
end
% r
if size(r,2)~=2,
    r=r.';
end
if size(r,1)>size(dict,2)
    r=r(1:size(dict,2),:);
end
% index
if nargin<13 || isempty(index),
    tmpmask = maskim(abs(squeeze(sum(recon_images,1))),7);
    index=find(tmpmask); clear tmpmask; 
end
% plotflag
if nargin<14
    plotflag = 0;
end
% rotfactor
if nargin<15
    rotfactor = 0;
end

%% Remove Fractional CSF signal

[CSFdiff, CSFind] = min(sqrt((r(:,1)-t1t2CSF(1)).^2 + (r(:,2)-t1t2CSF(2)).^2));  %static CSF
sig_CSFrem = zeros(size(recon_images));

if strcmp(seqType,'IRT')
    for n=1:length(index),
        dfind(n) = find(dfrange==dfim(index(n)));
    end
    dCSF = squeeze(dict(:,CSFind,dfind)).';
    sig_CSFrem(:,index) = recon_images(:,index) - dCSF.'.*repmat(frac(1,index),Nimages,1);
    Nperpart = 1000;
    [t1im_CSFrem, t2im_CSFrem, dfim_CSFrem, m0im_CSFrem] = ...
        MRF_TemplateMatching_subMatrix(sig_CSFrem, dict, r, dfrange, index, Nperpart);
else
    dCSF = squeeze(dict(:,CSFind)).';
    sig_CSFrem(:,index) = recon_images(:,index) - dCSF.'*frac(1,index);
    Nperpart = 1000;
    [t1im_CSFrem, t2im_CSFrem, dfim_CSFrem, m0im_CSFrem] = ...
        MRF_TemplateMatching_subMatrix(sig_CSFrem, dict, r, dfrange, index, Nperpart);
end

newMaps.t1im = t1im_CSFrem;
newMaps.t2im = t2im_CSFrem;
newMaps.dfim = dfim_CSFrem;
newMaps.m0im = m0im_CSFrem;

%% Synthetic FLAIR

FLAIR = zeros(siz);
FLAIRpvcorr = zeros(siz);

if isempty(FLAIRparams)
    %T1CSF = 2950; TI =  -T1CSF *log(0.5);
    TR = 15000; TE = 30; TI = 1800;
else
    TI=FLAIRparams(1);
    TR=FLAIRparams(2);
    TE=FLAIRparams(3);
end

for n=1:length(index),
    FLAIR(index(n))       = abs(m0im(index(n)))*(1-2*exp(-TI/t1im(index(n))) + exp(-TR/t1im(index(n))) )*exp(-TE/t2im(index(n)));
    FLAIRpvcorr(index(n)) = abs(m0im_CSFrem(index(n)))*(1-2*exp(-TI/t1im_CSFrem(index(n))) + exp(-TR/t1im_CSFrem(index(n))) )*exp(-TE/t2im_CSFrem(index(n)));
end
if plotflag
    figure; 
    subplot(1,2,1), imagesc(rot90(FLAIR,rotfactor)); axis image off; colormap gray; title('FLAIR MRF')
    subplot(1,2,2), imagesc(rot90(FLAIRpvcorr,rotfactor)); axis image off; colormap gray; title('FLAIR MRF, PV corrected')
    colormap gray;
    
    mask=zeros(siz);mask(index)=1;
    
    figure; 
    subplot(3,4,1), imagesc(rot90(mask.*t1im,rotfactor)); axis image off; title('T1 original'), caxis([0 max(r(:,1))]); colorbar; 
    subplot(3,4,2), imagesc(rot90(mask.*t2im,rotfactor)); axis image off; title('T2 original'), caxis([0 max(r(:,2))]); colorbar;
    subplot(3,4,3), imagesc(rot90(mask.*dfim,rotfactor)); axis image off; title('df original'), colorbar;
    subplot(3,4,4), imagesc(rot90(mask.*abs(m0im),rotfactor)); axis image off; title('M0 original'), colorbar;
    
    subplot(3,4,5), imagesc(rot90(t1im_CSFrem,rotfactor)); axis image off; title('T1 pv corr'), caxis([0 max(r(:,1))]); colorbar; 
    subplot(3,4,6), imagesc(rot90(t2im_CSFrem,rotfactor)); axis image off; title('T2 pv corr'), caxis([0 max(r(:,2))]); colorbar;
    subplot(3,4,7), imagesc(rot90(dfim_CSFrem,rotfactor)); axis image off; title('df pv corr'), colorbar;
    subplot(3,4,8), imagesc(rot90(abs(m0im_CSFrem),rotfactor)); axis image off; title('M0 pv corr'), colorbar;
        
    subplot(3,4,9), imagesc(rot90(mask.*t1im-t1im_CSFrem,rotfactor)); axis image off; title('T1 diff'), caxis([0 max(r(:,1))]); colorbar; 
    subplot(3,4,10), imagesc(rot90(mask.*t2im-t2im_CSFrem,rotfactor)); axis image off; title('T2 diff'), caxis([0 max(r(:,2))]); colorbar;
    subplot(3,4,11), imagesc(rot90(mask.*dfim-dfim_CSFrem,rotfactor)); axis image off; title('df diff'), colorbar;
    subplot(3,4,12), imagesc(rot90(mask.*abs(m0im)-abs(m0im_CSFrem),rotfactor)); axis image off; title('M0 diff'), colorbar;
end

