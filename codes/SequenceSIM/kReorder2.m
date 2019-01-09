function [kspace_out, roundError] = kReorder2(kList, gradMoms, resolution)
% sorts the acquired values in kList according to the gradient moments (gradMoms) and return a square kspace matrix
% is able to deal with multiple samples at the same kspace point

    % square matrix, third axis: multiple values at same kspace position (average later?) 
    kspace = zeros(resolution, resolution, length(kList));
	
	zIndices = ones(resolution, resolution); %counts number of acquired values for each position in kspace matrix
    
    % shift and rescale gradient moments to get array indices 
	% simulation currently returns gradMoms in the interval (-0.5, 0.5)
    indexList = round((gradMoms+0.5)*resolution);
    
    % write values to the calculated position in matrix
    for m=1:length(kList)
		ind1 = indexList(1,m);
		ind2 = indexList(2,m);
		
		if kspace(ind1, ind2, zIndices(ind1, ind2)) ~= 0
			zIndices(ind1, ind2) = zIndices(ind1, ind2) + 1;
		end
		
        kspace(ind1, ind2, zIndices(ind1, ind2)) = kList(m);
    end
     
	 kspace_out = kspace(:,:,1); % at the moment: return first acquired value at each matrix position
end