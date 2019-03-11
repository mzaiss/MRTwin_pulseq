function [] = plotSimulationResult(PD, kspace)
% show results of simulated sequence, compare input and resulting image
% PD: reference image, kspace: k-matrix with acquired/simulated values

    figure;
    
    % original image
        subplot(2,4,1);
        imshow(PD); % reference image
        subplot(2,4,2);
        Y = fftshift(fft2((PD))); % kspace of reference image

        % absolute, Re and Im values of kspace of reference image
        imagesc(abs(Y)); 
        subplot(2,4,3);
        imagesc(real(Y));
        subplot(2,4,4);
        imagesc(imag(Y));

    % aquired signal
        subplot(2,4,5);
        imshow(abs(ifft2(fftshift(kspace)))); % reconstructed image
        subplot(2,4,6);
        imagesc(abs(kspace));
        subplot(2,4,7);
        imagesc(real(kspace));
        subplot(2,4,8);
        imagesc(imag(kspace));
end