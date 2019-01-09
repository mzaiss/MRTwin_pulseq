function [kspace, roundError] = kReorder(kList, gradients, yshiftflag)
% sorts the acquired values in kList according to the gradient moments (gradMoms) and return a square kspace matrix
% yshiftflag = 1 : increase all ky indices by 1

    if nargin < 3 % needed for some sequences (should be unnecessary if gradients are applied properly?)
        yshiftflag=0;
    end

    dim = sqrt(length(kList)); % assuming square shaped k space (dim x dim)
    kspace = zeros(dim, dim);
    
    % shift and rescale gradient moments to get array indices 
    %normedGradients = max(gradients(1,:))-min(gradients(1,:));
    indexList = round((gradients+0.5)*dim);
    
    if yshiftflag == 1 % increase all ky indices by 1
        indexList(2,:) = indexList(2,:)+1; 
    end
    
    roundError = indexList - rescale(gradients, 1, dim);
    
    for m=1:length(kList) % sort acquired sample to k matrix
        kspace(indexList(1,m), indexList(2,m)) = kList(m);
    end
     
end