clear all; close all;

% forward Fourier transform
fftfull =  @(x) ifftshift(fftn(fftshift(x)))/sqrt(numel(x));

% NRMSE error function
e = @(utrue,u) 100*norm(u(:)-utrue(:))/norm(utrue(:));

%% Check numerical vs. analytical derivatives

T = 4;       % number of time points in readout                                                                                                  
sz = [4, 6]; % image size (Nx Ny)                                                                                                       

% set gradient spatial forms
rampX = pi*linspace(-1,1,sz(1) + 1);
rampX = fftshift(rampX(1:end-1),2);
rampX = repmat(rampX.', [1, sz(2)]);

% set gradient spatial forms
rampY = pi*linspace(-1,1,sz(2) + 1);
rampY = fftshift(rampY(1:end-1),2);
rampY = repmat(rampY, [sz(1), 1]);

% initialize gradients (X/Y directions)
g = rand(T,2); g = g(:);

% initialize complex-valued magnetization image
m = rand(sz(1),sz(2)) + 1i*rand(sz(1),sz(2));

% pack the parameters for the gradient function
args = {m, rampX, rampY};
[phi,dg_ana] = phi_grad_readout2d(g(:),args{:}); % compute loss and analytical derivatives

% compute numerical derivatives
h = 1e-4;
dg = zeros(size(g)); dphi_h = zeros(size(g));
for i=1:numel(g)
  dg(i) = 1; dphi_h(i) = phi_grad_readout2d(g+h*dg,args{:})-phi;
  dg(i) = 0;
  
  if mod(i,100) == 0
    dx_num = dphi_h/h;
    fprintf('deriv-err=%1.3f%%\n',e(dx_num(1:i),dg_ana(1:i)))
  end
end
dx_num = dphi_h/h; clear dphi_h i dx

fprintf('deriv-err=%1.3f%%\n',e(dx_num,dg_ana(:)))

% compare analytical and numerical gradients
[dg_ana(:), dx_num]

%% do full optimization
close all;

NRep = 16; % number of repetitions
sz = [16,16];

gtruth_m = load('data/phantom.mat'); gtruth_m = gtruth_m.phantom;
gtruth_m = imresize(gtruth_m,sz);  % resize to something managable

% set the optimizer
p = struct();
nMVM = 200;  % number of optimization iterations
p.length = -nMVM;
p.method = 'LBFGS';

T = 24; % set the number of time points in readout

% set gradient spatial forms
rampX = pi*linspace(-1,1,sz(1) + 1);
rampX = fftshift(rampX(1:end-1),2);
rampX = repmat(rampX.', [1, sz(2)]);

rampY = pi*linspace(-1,1,sz(2) + 1);
rampY = fftshift(rampY(1:end-1),2);
rampY = repmat(rampY, [sz(1), 1]);

% initialize reconstructed image
reco_m = zeros(sz);

for rep = 1:NRep
  
  % initialize gradients
  g = zeros(T,2); g(:,1) = rand(T,1) - 0.5; g = g(:);

  % compute the current error to ground-truth
  error_m = gtruth_m - reshape(reco_m,sz);

  % do optimization for g of E(g), loss --> (||error_m - E.T*E*error_m||^2) 
  args = {error_m, rampX, rampY};
  [g,~] = minimize(g(:),'phi_grad_readout2d',p,args{:});
  g = reshape(g,[],2);
  
  %x = cumsum(x, 1);
  figure(1), plot(g); title(['learned gradients at repetition ', num2str(rep), ' blue - grad X, orange - grad Y']); xlabel('time'); ylabel('gradient strength');
 
  % forward pass to compute the prediction and the operator
  [~,~,reco_current,E] = phi_grad_readout2d(g(:),args{:});
  
  % update the current reconstruction
  reco_m = reco_m + reshape(reco_current,sz);

  figure(2),
    subplot(2,2,1), imagesc(abs(reshape(error_m,sz))); title('current iteration target to predict');
    subplot(2,2,2), imagesc(abs(reshape(reco_current,sz))); title('current iteration prediction');
    subplot(2,2,3), imagesc(abs(reco_m)); title(['curent reconstruction, error=',num2str(e(gtruth_m(:),reco_m(:)))]);
    subplot(2,2,4), imagesc(abs(gtruth_m)); title('all iterations target to predict (final reco)');
 
  % plot samples k-space locations
  field = cumsum(g,1);
  figure(3),
    E = reshape(E,[],sz(1),sz(2)); 
    
    a = 0; b = 0;
    kspace_loc = zeros(T,2);
    for t = 1:T
      basis_func = squeeze(E(t,:,:));
      
      % compute location of the peak in k-space (gives sampled k-space point)
      spectrum = fftfull(basis_func);
      spectrum = abs(spectrum);
      
      maxval = max(spectrum(:));
      [a,b] = find(spectrum == maxval);
      
      %a = field(t,1); b = field(t,2);
      
      kspace_loc(t,1) = a; kspace_loc(t,2) = b;
    end
 
    % color code repetitions
    c = ones(T,1)*rep;
    hold on; scatter(kspace_loc(:,1), kspace_loc(:,2),[],c); hold off; title('kspace sampled locations'); xlabel('readout direction'); ylabel('phase encode direction');
      %hold on; scatter(kspace_loc(:,1), kspace_loc(:,2),[],c); axis([0,sz(1),0,sz(1)]); hold off; title('kspace sampled locations'); xlabel('readout direction'); ylabel('phase encode direction');

  pause
  
end

