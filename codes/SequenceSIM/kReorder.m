function [kspace, roundError] = kReorder(kList, gradients, yshiftflag)

    if nargin < 3
        yshiftflag=0;
    end

    dim = sqrt(length(kList)); % assuming square shaped k space (dim x dim)
    kspace = zeros(dim, dim);
    
    % shift and rescale gradient moments to get array indices 
    indexList = round((gradients+0.5)*dim);
    
    if yshiftflag == 1
        indexList(2,:) = indexList(2,:)+1; 
    end
%     indexList = round(rescale(gradients, 1, dim));
    
    roundError = indexList - rescale(gradients, 1, dim);
    
    for m=1:length(kList)
        kspace(indexList(1,m), indexList(2,m)) = kList(m);
    end
     
end