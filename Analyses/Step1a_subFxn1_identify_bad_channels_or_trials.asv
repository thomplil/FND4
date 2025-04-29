function [sub_bad, datasave] = identify_bad_channels_or_trials(d, condition, meanabs_thresh, num_trls, tt_t, i)
%% PURPOSE: Identify either bad trials or bad channels.
%% INPUTS:
% d: Data structure (FieldTrip format)
% condition: 'Channel' or 'Trial' (string)
% meanabs_thresh: Threshold for identifying bad channels or trials
% num_trls: Number of trials (set to 0 if identifying bad channels)
% tt_t: Not used explicitly (can be removed if unnecessary)
% i: Index for storing bad trials or channels

%% INITIALIZATION
sub_bad = {};  % Initialize as a cell array

% Define parameters based on condition
if strcmp(condition, 'Channel')
    cond = '_cont';
    ymin = -150000;
    ymax = 150000;
    num = 50;
else 
    cond = '_seg';
    ymin = -600000;
    ymax = 600000;
    num = num_trls;
end

data = d.trial{1,1};  % Extract trial data
num_chans = length(d.label);  % Get number of channels

%% COMPUTE MEAN ABSOLUTE VALUE FOR NOISE MEASUREMENT
if strcmp(condition, 'Channel')
    meanabs = nanmean(abs(data), 2);  % Compute mean absolute value per channel
else
    meanabs = [];
    for tt = 1:length(d.trial)
        mean_tt = nanmean(abs(d.trial{1, tt}(1:num_chans-1, :)), 2);
        mean_tt = nanmean(mean_tt);
        meanabs = [meanabs; mean_tt];
    end
end

% Convert to z-score for normalization
meanabs = zscore(meanabs);

%% VISUALIZATION AND IDENTIFYING BAD CHANNELS/TRIALS
[meanabs_sorted] = Visualize_NoiseMeasure_ChannelTrial(meanabs, num);
bad = meanabs_sorted(meanabs_sorted(:,1) > meanabs_thresh, 2);
good = meanabs_sorted(51:60, 2); % Sample of "good" channels/trials

% Warn if too many bad channels are detected
if length(bad) > 25
    disp('WARNING: More than 25 channels exceed the noise threshold. Consider removing the subject?');
    keyboard;
end

%% OPTIONAL MANUAL CHECK
run_semiauto = input('Would you like to manually check channel data? (0=N, 1=Y): ');
if run_semiauto == 1
    figure; hold on;
    cfg = [];
    cfg.blocksize = 300;
    cfg.ylim = [ymin ymax];
    vis = [bad', good'];
    
    if strcmp(condition, 'Channel')
        cfg.channel = d.label(vis);  % Set channels to visualize
    end
    
    ft_databrowser(cfg, d);
    disp('Modify bad based on visual inspection?');
    keyboard;
end

%% EXCLUDE BAD CHANNELS/TRIALS
cfg = [];
if strcmp(condition, 'Channel')
    if ~isempty(bad)
        cfg.channel = setdiff(d.label, d.label(bad));  % Exclude bad channels
        datasave = ft_selectdata(cfg, d);
    else
        datasave = d;
    end
else
    if ~isempty(bad)
        good_t = setdiff(1:num_trls, bad);  % Keep only good trials
        cfg.trials = good_t;
        datasave = ft_selectdata(cfg, d);
    else
        datasave = d;
    end
end

sub_bad = bad;  % Store bad channels or trials
if isempty(bad)
    sub_bad = [];  % Ensure an empty return instead of failing
end

end
