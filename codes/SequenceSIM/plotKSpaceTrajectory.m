function [] = plotKSpaceTrajectory(gradients, resolution, plotIndices)
% show figure of trajectory in k-space given by gradient moments
% ("gradients") -> 2xN matrix: N points with (kx,ky)
% plotindices = 0: kspace in units with kmax = 0.5
% plotindices = 1: kspace in integer matrix indices

    if ~exist('plotIndices','var')
          plotIndices = 0;
    end

    if plotIndices == 1
        plotcoords = (gradients+0.5)*resolution;
    else 
        plotcoords = gradients;
    end
    
    figure
    p = plot(plotcoords(1,:),plotcoords(2,:), '-o'), xlabel('k_x'), ylabel('k_y');
   
    % color data for trajectory
    cd = [uint8(jet(length(plotcoords))*255) uint8(ones(length(plotcoords),1))].';
    drawnow
    set(p.Edge, 'ColorBinding','interpolated', 'ColorData',cd);
end