function plotSeqUpdate(fig, seq)
%plot Plot the sequence in a new figure.
            %   plot(seqObj) Plot the sequence
            %
            %   plot(...,'Type',type) Plot the sequence with gradients
            %   displayed according to type: 'Gradient' or 'Kspace'.
            %
            %   plot(...,'TimeRange',[start stop]) Plot the sequence
            %   between the times specified by start and stop.
            %
            %   plot(...,'TimeDisp',unit) Display time in:
            %   's', 'ms' or 'us'.
            %
            %   f=plot(...) Return the new figure handle.
            %
    validPlotTypes = {'Gradient','Kspace'};
    validTimeUnits = {'s','ms','us'};
    persistent parser
    if isempty(parser)
        parser = inputParser;
        parser.FunctionName = 'plot';
        parser.addParamValue('type',validPlotTypes{1},...
            @(x) any(validatestring(x,validPlotTypes)));
        parser.addParamValue('timeRange',[0 inf],@(x)(isnumeric(x) && length(x)==2));
        parser.addParamValue('timeDisp',validTimeUnits{1},...
            @(x) any(validatestring(x,validTimeUnits)));
    end
    parse(parser,varargin{:});
    opt = parser.Results;

    fig=figure;
    if nargout>0
        f=fig;
    end
    ax=zeros(1,6);
    for i=1:6
        ax(i)=subplot(3,2,i);
    end
    ax=ax([1 3 5 2 4 6]);   % Re-order axes
    arrayfun(@(x)hold(x,'on'),ax);
    arrayfun(@(x)grid(x,'on'),ax);
    labels={'ADC','RF mag (Hz)','RF ph (rad)','Gx (kHz/m)','Gy (kHz/m)','Gz (kHz/m)'};
    arrayfun(@(x)ylabel(ax(x),labels{x}),1:6);

    tFactorList = [1 1e3 1e6];
    tFactor = tFactorList(strcmp(opt.timeDisp,validTimeUnits));
    xlabel(ax(3),['t (' opt.timeDisp ')']);
    xlabel(ax(6),['t (' opt.timeDisp ')']);

    t0=0;
    %for iB=1:size(obj.blockEvents,1)
    for iB=1:length(obj.blockEvents)
        block = obj.getBlock(iB);
        isValid = t0>=opt.timeRange(1) && t0<=opt.timeRange(2);
        if isValid
            if ~isempty(block.adc)
                adc=block.adc;
                t=adc.delay + (0:adc.numSamples-1)*adc.dwell;
                plot(tFactor*(t0+t),zeros(size(t)),'rx','MarkerSize',10,'Parent',ax(1));
            end
            if ~isempty(block.rf)
                rf=block.rf;
                t=rf.t + rf.delay;
                plot(tFactor*(t0+t),abs(rf.signal),'LineWidth',5,'Parent',ax(2));
                plot(tFactor*(t0+t),angle(rf.signal),'LineWidth',5,'Parent',ax(3));
            end
            gradChannels={'gx','gy','gz'};
            for j=1:length(gradChannels)
                grad=block.(gradChannels{j});
                if ~isempty(block.(gradChannels{j}))
                    if strcmp(grad.type,'grad')
                        t=grad.delay + grad.t + (grad.t(2)-grad.t(1))/2;
                        waveform=1e-3*grad.waveform;
                    else
    %                                 t=cumsum([0 grad.riseTime grad.flatTime grad.fallTime]);
                        t=cumsum([0 grad.delay grad.riseTime grad.flatTime grad.fallTime]);
    %                                 waveform=1e-3*grad.amplitude*[0 1 1 0];
                        waveform=1e-3*grad.amplitude*[0 0 1 1 0];
                    end
                    plot(tFactor*(t0+t),waveform,'Parent',ax(3+j));
                end
            end                
        end
        t0=t0+mr.calcDuration(block);
    end

    % Set axis limits and zoom properties
    dispRange = tFactor*[opt.timeRange(1) min(opt.timeRange(2),t0)];
    arrayfun(@(x)xlim(x,dispRange),ax);
    linkaxes(ax(:),'x')
    h = zoom(fig);
    setAxesZoomMotion(h,ax(1),'horizontal');


end