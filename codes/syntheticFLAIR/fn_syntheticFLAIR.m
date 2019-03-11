function [FLAIR] = fn_syntheticFLAIR(FLAIRparams,t1im,t2im,m0im,tissueParams,index,plotflag,rotfactor)
% [FLAIR] = fn_syntheticFLAIR(FLAIRparams,t1im,t2im,m0im,tissueParams,index,plotflag,rotfactor)
% Function computes synthetic FLAIR image from T1, T2, PD maps
% 
% INPUTS
%1     FLAIRparams     matrix      [1 x 3] TI, TR, TE of synthetic FLAIR image
%2     t1im            matrix      [siz x siz] T1 map 
%3     t2im            matrix      [siz x siz] T2 map 
%4     m0im            matrix      [siz x siz] PD map 
%5     tissueParams    matrix      [3 x 3] T1, T2, PD (columns) of 
%                                      CSF, GM, WM (rows)
%6     index           matrix      index in mask
%7     plotflag        binary      1 - show fraction maps; 0 - don't show
%8     rotfactor       double      rot90 rotation factor for visualization
%
% OUTPUTS
%1     FLAIR            matrix      [siz x siz] Unnormalized complex weights
%   
% This code written and maintained by: 
% Anagha Deshmane, Case Western Reserve University
% avd9@case.edu
% 
% First written October 2015
% Update history:
% February 14, 2019  Updated for simple FLAIR calculation (no Partial
% volume correction)
% August 30, 2016    Commented for Archiving
% January 5, 2017    Speedup CSF removal by converting pixelwise calculation to matrix
%                       operation
%
%% Check inputs
if nargin<5
    error('synthetic FLAIR - too few inputs')
end
siz = size(t1im,1);
% T1, T2 of CSF, WM
t1t2CSF = [tissueParams(1,1) tissueParams(1,2)]; 
t1t2WM = [tissueParams(3,1) tissueParams(3,2)]; 

% index
if nargin<6 || isempty(index),
    index=find(m0im>0); 
end
% plotflag
if nargin<7
    plotflag = 1;
end
% rotfactor
if nargin<8
    rotfactor = 1;
end

%% Synthetic FLAIR

FLAIR = zeros(siz);

if isempty(FLAIRparams)
    TR = 15000; TE = t1t2WM(2); TI = -t1t2CSF(1) *log(0.5);
else
    TI=FLAIRparams(1);
    TR=FLAIRparams(2);
    TE=FLAIRparams(3);
end

for n=1:length(index),
    FLAIR(index(n))       = abs(m0im(index(n)))*(1-2*exp(-TI/t1im(index(n))) + exp(-TR/t1im(index(n))) )*exp(-TE/t2im(index(n)));
    %FLAIRpvcorr(index(n)) = abs(m0im_CSFrem(index(n)))*(1-2*exp(-TI/t1im_CSFrem(index(n))) + exp(-TR/t1im_CSFrem(index(n))) )*exp(-TE/t2im_CSFrem(index(n)));
end
if plotflag
    figure; imagesc(rot90(FLAIR,rotfactor)); axis image off; colormap gray; title(['TI ' num2str(TI) '/TR ' num2str(TR) '/TE ' num2str(TE)]); 
end

