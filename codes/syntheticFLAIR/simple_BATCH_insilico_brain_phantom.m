% simple BATCH

figure(1), subplot(2,4,1), imagesc(rot90(t1im)); title('T1');
subplot(2,4,2), imagesc(rot90(t2im)); title('T2');
subplot(2,4,3), imagesc(rot90(abs(m0im))); title('PD');
subplot(2,4,4), imagesc(rot90(angle(m0im))); title('phase');

TI=500, TR=10000, TE=70; % typical SE-FLAIR parameters
[FLAIR05s] = fn_syntheticFLAIR([TI TR TE],t1im,t2im,m0im,T1T2species_true,index,0,1);

TI=2000, TR=10000, TE=70; % typical SE-FLAIR parameters
[FLAIR2s] = fn_syntheticFLAIR([TI TR TE],t1im,t2im,m0im,T1T2species_true,index,0,1);

TI=3000, TR=10000, TE=70; % typical SE-FLAIR parameters
[FLAIR3s] = fn_syntheticFLAIR([TI TR TE],t1im,t2im,m0im,T1T2species_true,index,0,1);

TI=5000, TR=10000, TE=70; % typical SE-FLAIR parameters
[FLAIR5s] = fn_syntheticFLAIR([TI TR TE],t1im,t2im,m0im,T1T2species_true,index,0,1);

figure(1),
subplot(2,4,5), imagesc(rot90(FLAIR05s));  title('FLAIR TI=0.5s');
subplot(2,4,6), imagesc(rot90(FLAIR2s)); title('FLAIR TI=2.0s');
subplot(2,4,7), imagesc(rot90(FLAIR3s));  title('FLAIR TI=3.0s');
subplot(2,4,8), imagesc(rot90(FLAIR5s));  title('FLAIR TI=5.0s');