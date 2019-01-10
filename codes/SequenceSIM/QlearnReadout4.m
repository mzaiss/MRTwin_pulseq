function QlearnMRIzero
%% Example reinforcement learning - Q-learning code
% for single k-space readout
clear all
close all;

%% create Input image (N,N,3)
% (:,:,1) -> PD
% (:,:,2) -> T2
% (:,:,3) -> T2

resolution = 48; % 100x100 take runs ~12s on a single core
deltak=1/resolution;
PD = phantom(resolution);
PD(PD<0) = 0;
T1 = PD*2;
T2 = PD*2;
InVol = cat(3,PD,T1,T2);
numSpins = 1;

%% Sequence Parameters
SeqOpts.resolution = resolution;
SeqOpts.FOV = 220e-3;
SeqOpts.TE = 10e-3;
SeqOpts.TR = 3000e-3;
SeqOpts.FlipAngle = pi/2;
seqFilename = fullfile(pwd, 'gre.seq');

sequence = WriteGRESequenceWithPulseq(SeqOpts, seqFilename);
sequence.plot();
%FG: reference sequence: gre.seq
% we use the outcome of the standard linear reorderd GRE as a reference k-space and image

tic
for ii=1:1
[kList, gradients] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename, numSpins);
end
toc
kRef = reshape(kList,[resolution resolution]);



%% SETTINGS
%%% Confidence in new trials?
learnRate = 0.99; % How is new value estimate weighted against the old (0-1). 1 means all new and is ok for no noise situations.

%%% Exploration vs. exploitation
% Probability of picking random action vs estimated best action
epsilon = 0.001; % Initial value
epsilonDecay = 1; % Decay factor per iteration.

%%% Future vs present value
discount = 0.9; % When assessing the value of a state & action, how important is the value of the future states?

%%% Inject some noise?
successRate = 1; % How often do we do what we intend to do?
% E.g. What if I'm trying to turn left, but noise causes me to turn right instead. 
% This probability (0-1) lets us try to learn policies robust to "accidentally" doing the wrong action sometimes.

winBonus = 100;  % Option to give a very large bonus when the system reaches the desired state (pendulum upright).
maxEpi = 2000; % Each episode is starting with the pendulum down and doing continuous actions for awhile.
maxit = 2500; % Iterations are the number of actions taken in an episode.

% Gradient limits -- bang-bang control
gLim = 1;
actions_x = [0, -gLim, gLim, 0.5*gLim, -0.5*gLim]; % gradient steps in x
actions_y = [0, -gLim, gLim, 0.5*gLim, -0.5*gLim]; % gradient steps in y

% Make the un-updated values on the value map transparent. If not, then we see the reward function underneath.
transpMap = true;

% Write to video?
doVid = false;

if doVid
    writerObj = VideoWriter('qlearnVid.mp4','MPEG-4');
    writerObj.FrameRate = 60;
    open(writerObj);
end


%% Discretize the state so we can start to learn a value map
% States are until now the time steps - this might not be the best and
% might not be a Markov process
states = 1:5000;

dt=1e-3;

% Local value R and global value Q -- A very important distinction!
%
% R, the reward, is like a cost function. It is good to be near our goal. It doesn't
% account for actions we can/can't take. We use quadratic difference from the top.
%
% Q is the value of a state + action. We have to learn this at a global
% level. Once Q converges, our policy is to always select the action with the
% best value at every state.
%
% Note that R and Q can be very, very different.
% For example, if the pendulum is near the top, but the motor isn't
% powerful enough to get it to the goal, then we have to "pump" the pendulum to
% get enough energy to swing up. In this case, that point near the top
% might have a very good reward value, but a very bad set of Q values since
% many, many more actions are required to get to the goal.

% R = rewardFunc(states(:,1),states(:,2)); % Initialize the "cost" of a given state to be quadratic error from the goal state. Note the signs mean that -angle with +velocity is better than -angle with -velocity
% FG: initialization?! small random values?
Q = zeros(length(states), length(actions_x)*length(actions_y)); % Q is length(x1) x length(x2) x length(actions) - IE a bin for every action-state combination.

% A = zeros(length(actions_x), length(actions_y)); %FG: dirty trick not
% needed any more...


% V will be the best of the Q actions at every state. This is only
% important for my plotting. Not algorithmically significant. I leave
% values we haven't updated at 0, so they appear as blue (unexplored) areas
% in the plot.
V = zeros(size(states,1),1);
Vorig = max(Q,[],2);

%% Q for a EPI
n=1;
Q = zeros(length(states), length(actions_x)*length(actions_y)); % Q is length(x1) x length(x2) x length(actions) - IE a bin for every action-state combination.

Q(:,1)=0.1;
for ii=1:48
    for jj=1:48
            gx(n)=sign(mod(ii,2))+2;
            gy(n)=(mod(jj,48)==0)*5;
            Q(n,gx(n)+gy(n))=1;
            
            n=n+1;
    end
end

figure,imagesc(Q);


%% Start learning!

% Number of episodes or "resets"
% in our case this is how often we run a full sequence of gradient events
for episodes = 1:maxEpi
    
    % FG: initialize sequence
    seq = mr.Sequence();
    SeqOpts.FlipAngle = pi/2;
    rf = mr.makeBlockPulse(SeqOpts.FlipAngle,'Duration',1e-3);
    
    seq.addBlock(rf); % start sequence with 90deg excitation
    Nx = resolution;
    deltak = 1./resolution;
    riseTime = 5e-5; % use the same rise times for all gradients, so we can neglect them
        
    gradXevent=mr.makeTrapezoid('x','FlatArea',-24*deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
    gradYevent=mr.makeTrapezoid('y','FlatArea',24*deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
    % add PULSEQ block of gradient rewinder
    seq.addBlock(gradXevent,gradYevent);  
    
    % prepare updateable events
    gradXevent=mr.makeTrapezoid('x','FlatArea',0*deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
    gradYevent=mr.makeTrapezoid('y','FlatArea',0*deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
    % always have the ADC open during any gradient.
    adc = mr.makeAdc(1,'Duration',dt,'Delay',riseTime); %FG: TODO: number?
        
    % Number of actions we're willing to try before a reset
    % in our case this is the loop over the actual MR sequence time steps (5000 for now)
    tic
    altered_Idx=[];
    z1=1;
    sIdx=z1; % FG
    for g = 1:5000
          
        %% PICK AN ACTION        
        
        % Interpolate the state within our discretization (ONLY for choosing
        % the action. We do not actually change the state by doing this!)
        % [~,sIdx] = min(sum((states - repmat(z1,[size(states,1),1])).^2,2));
               
        % Choose an action:
        % EITHER 1) pick the best action according the Q matrix (EXPLOITATION). OR
        % 2) Pick a random action (EXPLORATION)
        if (rand()>epsilon || episodes == maxEpi) % Pick according to the Q-matrix it's the last episode or we succeed with the rand()>epsilon check. Fail the check if our action doesn't succeed (i.e. simulating noise)
            [~,aIdx] = max(Q(sIdx,:)); % Pick the action the Q matrix thinks is best!
        else
            aIdx = randi(length(actions_x)*length(actions_y),1); % Random action!
%              warning('random action for g =%d',g);
            altered_Idx=[altered_Idx ; aIdx sIdx];
        end
%            aIdx
           
%         A(aIdx)=1;
%         [ix,iy] = find(A); % FG: OPTIMIZE!
%         A(aIdx)=0;
        
        % FG: better:
        ix = mod(aIdx-1, length(actions_x)) + 1; % wrap 1D action index aIdx to 2D indices (gx & gy)
        iy = fix((aIdx-1)/length(actions_y))+1; % fix(a/b) is integer division of a and b
        
        gx = actions_x(ix);  % calculate the gradient in x
        gy = actions_y(iy);  % calculate the gradient in y
        
        % generate PULSEQ events
         gradXevent.amplitude=gx*23.1481;
         gradYevent.amplitude=gy*23.1481;
         
%          if (mod(g,1000)==1)
%          seq.addBlock(rf); % start sequence with 5deg excitation
%          end
         
        % add PULSEQ block of gradient and ADC
        seq.addBlock(gradXevent,gradYevent,adc);       
              
        z1 = z1 + 1; %FG: z1 not needed below?
        
%       [~,snewIdx] = min(sum((states - repmat(z1,[size(states,1),1])).^2,2)); % Interpolate again to find the new state the system is closest to.
        snewIdx = z1; %FG (no interpolation necessary)        
        sIdx=z1; % FG
        
    end %end  g loop, the 5000 time steps
    toc
    
       if episodes ~= maxEpi % On the last iteration, stop learning and just execute. Otherwise...
        %% Update Q
            
        % we save the pulse sequence to disk   
        tic
        seqFilename = fullfile(pwd, 'QSeq.seq');
        seq.write(seqFilename);
        toc
                    
        % FG: run MR simulation
        tic
        [kList, gradMoms] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename, numSpins);
        toc
        
        kList(isnan(kList))=0;
        gradMomsScaled = (gradMoms+0.5)*resolution;  % calculate grad moms to FoV
        
        [X,Y] = meshgrid(1:resolution);
%         [Xq,Yq] = meshgrid(1:0.5:resolution);
%          kRefInterp = interp2(X,Y,kRef,gradMomsEnd(1), gradMomsEnd(2));
%         kRefInterp2 = interp2(X,Y,kRef,Xq, Yq);
               
        kReco = griddata(gradMomsScaled(1,:),gradMomsScaled(2,:),real(kList),X,Y) +1j*griddata(gradMomsScaled(1,:),gradMomsScaled(2,:),imag(kList),X,Y) ;
        kReco(isnan(kReco))=0;
         
        
        try kError_old=kError; end;
        
        kError = abs(kReco - kRef); 
        kError=mean(kError(:));
        
        if episodes>1
            Reward = kError_old-kError;  % positive reward when old larger than new.
            
            %if improved all get an reward 

            for ii=1:size(altered_Idx,1)
            aIdx=altered_Idx(ii,1);
            sIdx=altered_Idx(ii,2);
            Q(sIdx,aIdx) = Q(sIdx,aIdx) + learnRate * ( Reward + discount*max(Q(sIdx,:)) - Q(sIdx,aIdx) );
            end
            % if improved/worsened only the made changes get an extra reward/penalty

        figure(2), subplot(3,2,1), imagesc(abs(kReco),[0 200]);
        subplot(3,2,2), imagesc(abs(kRef),[0 200]);
        subplot(3,2,4), imagesc(abs(ifft2(fftshift(kRef))),[0 2]);
        subplot(3,2,3), imagesc(abs(fft2(fftshift(kReco))),[0 500]); title(sprintf('kError, %f, Reward %f: ',kError, Reward));
        subplot(3,2,5), plot(gradMomsScaled(1,:),gradMomsScaled(2,:));
        subplot(3,2,6),scatter3(gradMomsScaled(1,:),gradMomsScaled(2,:),real(kList))
        drawnow;
        clc   
        end
        
          
        
        
            % Lets break this down:
            %
            % We want to update our estimate of the global value of being
            % at our previous state s and taking action a. We have just
            % tried this action, so we have some information. Here are the terms:
            %   1) Q(sIdx,aIdx) AND later -Q(sIdx,aIdx) -- Means we're
            %      doing a weighting of old and new (see 2). Rewritten:
            %      (1-alpha)*Qold + alpha*newStuff
            %   2) learnRate * ( ... ) -- Scaling factor for our update.
            %      High learnRate means that new information has great weight.
            %      Low learnRate means that old information is more important.
            %   3) R(snewIdx) -- the reward for getting to this new state
            %   4) discount * max(Q(snewIdx,:)) -- The estimated value of
            %      the best action at the new state. The discount means future
            %      value is worth less than present value
            %   5) Bonus - I choose to give a big boost of it's reached the
            %      goal state. Optional and not really conventional.
            
            sIdx = sIdx + 1; %FG: dynamics step: increase state index
        end
        
        % Decay the odds of picking a random action vs picking the
        % estimated "best" action. I.e. we're becoming more confident in
        % our learned Q.
        epsilon = epsilon*epsilonDecay;
    
%     seq.plot();
        
end

if doVid
    close(writerObj);
end

end
