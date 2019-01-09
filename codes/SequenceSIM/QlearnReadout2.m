function QlearnPend
%% 
% for full readout

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
T2 = PD*0.35;
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
seqFilename = fullfile(pwd, 'gre.seq');
tic
for ii=1:1
    [kList, gradients] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename, numSpins);
end;
toc
kRef = abs(reshape(kList,[resolution resolution]));
kRefc = reshape(kList,[resolution resolution]);


%% SETTINGS

%%% What do we call good?
rewardFunc = @(x,xdot)(-(abs(x)).^2 + -0.25*(abs(xdot)).^2); % Reward is -(quadratic error) from upright position. Play around with different things!

%%% Confidence in new trials?
learnRate = 0.99; % How is new value estimate weighted against the old (0-1). 1 means all new and is ok for no noise situations.

%%% Exploration vs. exploitation
% Probability of picking random action vs estimated best action
epsilon = 0.1; % Initial value
epsilonDecay = 1; % Decay factor per iteration.

%%% Future vs present value
discount = 0.9; % When assessing the value of a state & action, how important is the value of the future states?

%%% Inject some noise?
successRate = 1; % How often do we do what we intend to do?
% E.g. What if I'm trying to turn left, but noise causes
% me to turn right instead. This probability (0-1) lets us
% try to learn policies robust to "accidentally" doing the
% wrong action sometimes.

winBonus = 100;  % Option to give a very large bonus when the system reaches the desired state (pendulum upright).

startPt = 1; % Start every episode at vertical down. %FG: start just at t=0 (no gradients applied yet)

maxEpi = 2000; % Each episode is starting with the pendulum down and doing continuous actions for awhile.
maxit = 2500; % Iterations are the number of actions taken in an episode.
substeps = 2; % Number of physics steps per iteration (could be 1, but more is a little better integration of the dynamics)
dt = 0.05; % Timestep of integration. Each substep lasts this long

% Gradient limits -- bang-bang control
gLim = 1;
actions_x = [0, -gLim, gLim, 0.5*gLim, -0.5*gLim]; % Only 3 options, Full blast one way, the other way, and off.
actions_y = [0, -gLim, gLim, 0.5*gLim, -0.5*gLim];

% Make the un-updated values on the value map transparent. If not, then
% we see the reward function underneath.
transpMap = true;

% Write to video?
doVid = false;

if doVid
    writerObj = VideoWriter('qlearnVid.mp4','MPEG-4');
    writerObj.FrameRate = 60;
    open(writerObj);
end

%% Discretize the state so we can start to learn a value map
% State1 is angle -- play with these for better results. Faster convergence
% with rough discretization, less jittery with fine.
states = 1:3000;
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

Q = randn(length(states), length(actions_x)*length(actions_y)); % Q is length(x1) x length(x2) x length(actions) - IE a bin for every action-state combination.

% A = zeros(length(actions_x), length(actions_y)); %FG: dirty trick not
% needed any more...


% V will be the best of the Q actions at every state. This is only
% important for my plotting. Not algorithmically significant. I leave
% values we haven't updated at 0, so they appear as blue (unexplored) areas
% in the plot.
V = zeros(size(states,1),1);
Vorig = max(Q,[],2);



%% Start learning!

% Number of episodes or "resets"
for episodes = 1:maxEpi
    
    
    % Number of actions we're willing to try before a reset
    for g = 1:maxit
        g
        
        
        
        % Stop if the figure window is closed.
        %         if ~ishandle(panel)
        %             break;
        %         end
        
        %% PICK AN ACTION
        
        % Interpolate the state within our discretization (ONLY for choosing
        % the action. We do not actually change the state by doing this!)
        %         [~,sIdx] = min(sum((states - repmat(z1,[size(states,1),1])).^2,2));
        
        % FG: initialize sequence
            seq = mr.Sequence();
            SeqOpts.FlipAngle = pi/2;
            rf = mr.makeBlockPulse(SeqOpts.FlipAngle,'Duration',1e-3);
            Nx = resolution;
            deltak = 1./resolution;
            riseTime = 5e-5; % use the same rise times for all gradients, so we can neglect them
            adc = mr.makeAdc(1,'Duration',dt,'Delay',riseTime); %FG: TODO: number?
            
            seq.addBlock(rf); % start sequence with 90deg excitation
            
            
        for sIdx=states
            
            
            
            % Choose an action:
            % EITHER 1) pick the best action according the Q matrix (EXPLOITATION). OR
            % 2) Pick a random action (EXPLORATION)
            if (abs(rand())>epsilon || episodes == maxEpi) && rand()<=successRate % Pick according to the Q-matrix it's the last episode or we succeed with the rand()>epsilon check. Fail the check if our action doesn't succeed (i.e. simulating noise)
                [~,aIdx] = max(Q(sIdx,:)); % Pick the action the Q matrix thinks is best!
            else
                aIdx = randi(length(actions_x)*length(actions_y),1); % Random action!
            end
            
            %         A(aIdx)=1;
            %         [ix,iy] = find(A); % FG: OPTIMIZE!
            %         A(aIdx)=0;
            
            % FG: better:
            ix = mod(aIdx-1, length(actions_x)) + 1; % wrap 1D action index aIdx to 2D indices (gx & gy)
            iy = fix((aIdx-1)/length(actions_y))+1; % fix(a/b) is integer division of a and b
            
            gx = actions_x(ix);
            gy = actions_y(iy);
            
            gradXevent=mr.makeTrapezoid('x','FlatArea',gx*deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
            gradYevent=mr.makeTrapezoid('y','FlatArea',gy*deltak,'FlatTime',dt-2*riseTime,'RiseTime', riseTime);
            
            adc = mr.makeAdc(1,'Duration',dt,'Delay',riseTime); %FG: TODO: number?
            %T = actions(aIdx);
            
            seq.addBlock(gradXevent,gradYevent,adc);
        end
            

            
            %% UPDATE Q-MATRIX
            
            % FG: write seq file to pass to simulation (not optimal to save on
            % disk for each iteration...)
            seqFilename = fullfile(pwd, 'QSeq.seq');
            seq.write(seqFilename);
            
            if (0)
                seq.plot();
            end
            % FG: do simulation
            [kList, gradMoms] = RunMRIzeroBlochSimulationNSpins(InVol, seqFilename, numSpins);
            
            kListEnd = abs(kList(end));
            gradMomsEnd = (gradMoms(:,end)+0.5)*resolution;
            
            [X,Y] = meshgrid(1:resolution);
            [Xq,Yq] = meshgrid(1:0.5:resolution);
            kRefInterp = interp2(X,Y,kRef,gradMomsEnd(1), gradMomsEnd(2),'nearest');
            kRefInterp2 = interp2(X,Y,kRef,Xq, Yq);
            
            if 1
                kList(isnan(kList))=0;
                gradMomsScaled = (gradMoms+0.5)*resolution;
                gradMomsScaled(:,end+1:end+resolution^2)= [X(1:end); Y(1:end)];
                kList(end+1:end+resolution^2)=0;
                kReco = griddata(gradMomsScaled(1,:),gradMomsScaled(2,:),real(kList),X,Y) +1j*griddata(gradMomsScaled(1,:),gradMomsScaled(2,:),imag(kList),X,Y) ;
                kReco(isnan(kReco))=0;
                figure(2), subplot(3,2,1), imagesc(abs(kReco));
                subplot(3,2,2), imagesc(abs(kRef));
                subplot(3,2,4), imagesc(abs(ifft2(fftshift(kRefc))));
                subplot(3,2,3), imagesc(abs(ifft2(fftshift(kReco))));
                subplot(3,2,5), plot(gradMomsScaled(1,:),gradMomsScaled(2,:));
                clc
            end
            
            
            % k space error
             kError = (kRef - abs(kReco)).^2;
             kError=sum(kError(:));
            
            % image space error
%              kError=(abs(ifft2(fftshift(kRefc)))-abs(ifft2(fftshift(kReco)))).^2;
%              kError=sum(kError(:));
            title(sprintf('Error %f',kError));
            
            if 1<0.01 % If we've reached upright with no velocity (within some margin), end this episode.
                success = true;
                bonus = winBonus; % Give a bonus for getting there.
            else
                bonus = 0;
                success = false;
            end
            
            
            if episodes ~= maxEpi % On the last iteration, stop learning and just execute. Otherwise...
                % Update Q
               for sIdx=states 
                % FG: max(Q(snewIdx,:)) -> max(Q(sIdx,:))
                Q(sIdx,aIdx) = Q(sIdx,aIdx) + learnRate * ( -kError + discount*max(Q(sIdx,:)) - Q(sIdx,aIdx) + bonus );
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
                
%                 sIdx = sIdx + 1; %FG: dynamics step: increase state index
            end
            
            % Decay the odds of picking a random action vs picking the
            % estimated "best" action. I.e. we're becoming more confident in
            % our learned Q.
            epsilon = epsilon*epsilonDecay;
            
            %% UPDATE PLOTS
            
            %         if episodes>0
            %             % Pendulum state:
            %             set(f,'XData',[0 -sin(z1(1))]);
            %             set(f,'YData',[0 cos(z1(1))]);
            %
            %             % Green tracer point:
            %             [newy,newx] = ind2sub([length(x2),length(x1)],snewIdx); % Find the 2d index of the 1d state index we found above
            %             set(pathmap,'XData',newx);
            %             set(pathmap,'YData',newy);
            %
            %             % The heat map of best Q values
            %             V = max(Q,[],2); % Best estimated value for all actions at each state.
            %             fullV = reshape(V,[length(x2),length(x1)]); % Make into 2D for plotting instead of a vector.
            %             set(map,'CData',fullV);
            %             if transpMap
            %                 set(map,'AlphaData',fullV~=Vorig); % Some spots have not changed from original. If not, leave them transparent.
            %             end
            %             drawnow;
            %
            %             % Take a video frame if turned on.
            %             if doVid
            %                 frame = getframe(panel);
            %                 writeVideo(writerObj,frame);
            %             end
            %         end
            
            % End this episode if we've hit the goal point (upright pendulum).
            if success
                break;
            end
            
        end
        seq.plot();
        
        
    end
    
    if doVid
        close(writerObj);
    end
    
end

    function zdot = Dynamics(z,T)
        % Pendulum with motor at the joint dynamics. IN - [angle,rate] & torque.
        % OUT - [rate,accel]
        g = 1;
        L = 1;
        z = z';
        zdot = [z(2) g/L*sin(z(1))+T];
    end