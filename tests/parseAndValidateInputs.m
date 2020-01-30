%--------------------------------------------------------------------------
function [IMF,fs,T,TD,F,FRange,FResol,MinThres,Method,FreqLoc,isTT,isNF] = parseAndValidateInputs(IMF, varargin)
% input type checking
validateattributes(IMF,{'single','double','timetable'},{'2d','nonempty'},'hht','IMF');
isInMATLAB = coder.target('MATLAB');
isTT = isa(IMF,'timetable');

if isvector(IMF) && (~isTT)
    IMF = IMF(:);
end

if(size(IMF,1)<2)
    error(message('shared_signalwavelet:hht:general:notEnoughRows','IMF',1));
end


% check if Fs/Frequency location exist or not
fs = 2*pi;
FreqLoc = 'yaxis';
isNF = true;    % if it is normalized frequency
initVarargin = 1;
finalVarargin = length(varargin);

if(~isempty(varargin))
    if(~ischar(varargin{1}) && ~isstring(varargin{1}))
        fs = varargin{1};
        isNF = false;
        initVarargin = 2;
    end
    
    if((ischar(varargin{end}) || isstring(varargin{end}))...
            && mod(finalVarargin-initVarargin+1,2)==1)
        FreqLoc = varargin{end};
        finalVarargin = length(varargin)-1;
    end
end
validateattributes(fs,{'numeric'},{'nonnan','finite',...
    'positive','scalar'},'hht','fs');
validatestring(FreqLoc,{'yaxis','xaxis'},'hht','freqloc');

% handle timetable
if(isTT)
    signalwavelet.internal.util.utilValidateattributesTimetable(IMF, {'regular','sorted','multichannel'}, 'hht','IMF');
    [IMF, T, TD] = signalwavelet.internal.util.utilParseTimetable(IMF);
    validateattributes(T, {'single','double'},{'nonnan','finite','real'},'hht','T');
    
    % validate input frequency coincides with timetable
    if(~isNF)
        if(abs((T(2)-T(1))-1/fs)>eps)
            error(message('shared_signalwavelet:hht:general:notMatchedFreqTimetable','IMF'));
        end
    else
        fs = 1/(T(2)-T(1));
        isNF = false;
    end
else
    TD = [];
    T = (0:(size(IMF,1)-1))'/fs;
end

% data integrity checking
validateattributes(IMF,{'single','double'},{'real','finite','nonnan','nonsparse'},'hht','IMF');

% cast to double due to sparse matrix constraints
IMF = double(IMF);
T = double(T);

% parse and validate name-value pairs
defaultFRange = [];
defaultFResol = [];
defaultMinThres = -inf;
defaultMethod = 'HT';
if(isInMATLAB)
    p = inputParser;
    addParameter(p,'FrequencyLimits',defaultFRange);
    addParameter(p,'FrequencyResolution',defaultFResol);
    addParameter(p,'MinThreshold',defaultMinThres);
    addParameter(p,'Method',defaultMethod);
    parse(p,varargin{initVarargin:finalVarargin});
    FRange = p.Results.FrequencyLimits;
    FResol = p.Results.FrequencyResolution;
    MinThres = p.Results.MinThreshold;
    Method = p.Results.Method;
else
    coder.varsize('FRange',2);
    coder.varsize('FResol');
    parms = struct( 'FrequencyLimits',           uint32(0), ...
                    'FrequencyResolution',      uint32(0), ...
                    'MinThreshold',             uint32(0), ...
                    'Method',                   uint32(0));
    pstruct = eml_parse_parameter_inputs(parms,[],varargin{:});
    FRange = eml_get_parameter_value(pstruct.FrequencyLimits,defaultFRange,varargin{initVarargin:finalVarargin});
    FResol = eml_get_parameter_value(pstruct.FrequencyResolution,defaultFResol,varargin{initVarargin:finalVarargin});
    MinThres = eml_get_parameter_value(pstruct.MinThreshold,defaultMinThres,varargin{initVarargin:finalVarargin});
    Method = eml_get_parameter_value(pstruct.Method,defaultMethod,varargin{initVarargin:finalVarargin});
end
validateattributes(MinThres,{'numeric'},{'nonnan','scalar'},'hht','MinThreshold');
validatestring(Method,{'HT','DQ'},'hht','Method');

% compute frequency range and resolution when they are not specified
if(isempty(FRange))
    FRange = [0;fs/2];
end
validateattributes(FRange,{'numeric'},...
    {'nonnan','finite','numel',2,'>=',0,'<=',fs/2},...
    'hht','FrequencyLimits');
if(FRange(1)>=FRange(2))
    error(message('shared_signalwavelet:hht:general:invalidFreqRange', 'FrequencyLimits'));
end

if(isempty(FResol))
    FResol = (FRange(2)-FRange(1))/100;
end
validateattributes(FResol,{'numeric'},{'nonnan','finite','scalar','>',0},'hht','FrequencyResolution');

% set up frequency vector
F = (FRange(1):FResol:FRange(2))';
end