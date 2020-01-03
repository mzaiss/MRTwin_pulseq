function rflips = rectify_flips(flips)
    rflips = flips;
    for ii=1:size(rflips,1)
        for jj=1:size(rflips,2)
            if (rflips(ii,jj,1) < 0)
                rflips(ii,jj,1) = -rflips(ii,jj,1);
                rflips(ii,jj,2) = rflips(ii,jj,2)+ pi;
                rflips(ii,jj,2) = mod(rflips(ii,jj,2), 2*pi);
            end
        end
    end
