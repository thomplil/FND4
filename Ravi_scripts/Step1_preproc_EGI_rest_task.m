function Step1_preproc_EGI_rest_task(subjects,run_ica_inspect)

%Author: Ravi Mill, rdm146@rutgers.edu
%Last update: Aug 4th 2020

%DESCRIPTION: preprocesses EEG data (task and rest) for Dynamic Activity
%Flow project: Mill, XX, "Title"

%Preprocesses via the following steps:
%1. Imports EGI data into fieldtrip format
%2. Filters continuous data (using minimum phase causal filter): 60Hz notch + 1Hz highpass
%3. Segment into trial epochs (*differs for task and rest)
%4. Remove noisy channels using the continuous data - automated (absolute amplitude zscore); save bad channels in array
%5. Remove noisy trials using trial data - automated (absolute amplitude zscore); save bad trials in array
%6. ICA on trial data
%7. Identify and reject ICA components -  
%8. Baseline correct (*varies for task and rest) + lowpass filter (50Hz) + rereference to common average
%9. Save outputs from various approaches

%INPUTS
%subjects=numeric nx1 array, each row containing subject IDs 
%run_ica_inspect=numeric, determines which version of the function
    %is run; 1 (step1): preprocessing is run till ICA estimation and saved (run
    %using slurm batch), 2 (step2): loads in ICA results and allows visual
    %inspection of potential artifacts followed by IC rejection, lp
    %filtering etc

%OUTPUTS
%Writes results of steps leading up to and including ICA (run_ica_inspect=1), followed
%by final preproc results (after visually screening ICA results; lp filtering, baselining and common avg
%rereferencing (run_ica_inspect==2)

%% Set path and defaults for Fieldtrip
addpath '/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/';
ft_defaults;

%*more recent fieldtrip
% % Set path and defaults for Fieldtrip
% addpath /projectsn/f_mc1689_1/opm_meg_fc/docs/scripts/fieldtrip-20231025; 
% ft_defaults;

%need this for ica
addpath /projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/eeglab13_6_5b/; %need eeglab to run automated ICA methods

%path to egi electrode layout (in Fieldtrip templates directory)
elecfile = '/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/template/electrode/GSN-HydroCel-257.sfp';

%% Set inputs, parms etc

%total subjects in sample (830 excluded for many bad trials)
total_subjects = [803;804;805;806;807;809;810;811;813;...
    814;815;817;818;819;820;821;822;823;824;825;826;827;...
    828;829;831;832;833;834;835;836;837;838]; %total_subjects vs subjects might be redundant now actually;'total_subjects' and 'subjects' specification helps load in behav files

%set path to inputs
%/projects/f_mc1689_1/DynamicSensoryMotorEGI/data/rawdata/EEG/rest_pretask/
baseDir='/projects/f_mc1689_1/DynamicSensoryMotorEGI/';
input_dir = [baseDir,'/data/rawdata/'];

%REMOVE - now runs task and rest_pretask for all subjects
%**set to preproc 'task', 'rest_pretask', or 'rest_posttask';
%run_phase = 'rest_pretask';

%**set event name for task event preproc: 'stim', 'cue', 'resp'
task_event='stim';

%**set filter pass bands; note these are sequentially applied as this i) is
%advocated for ICA (from EEGLAB list), ii) yielded less delay relative to
%the raw/butterworth filtered signal than bandpass filtering (see Step0
%script)
filter_hp=1;
filter_lp=50;

%**set epoch length (in secs) for rest and task
%rest; rest is baselined based on pseudotrial mean
rest_epoch_prestim=0; %in s; i.e. pull out 0-600s continuous EEG data around 'start recording' marker sent by eprime to EGI
rest_epoch_poststim=600; %in s
rest_epoch_length=20000; %in samples, used to segment 600s rest segment into 20s pseudotrials
%task
task_epoch_prestim=0.5; %in s
task_epoch_poststim=1.5;
task_epoch_baseline=-0.5; %baseline will be -0.5 to 0

%set condition codes for task - used in MVPA analyses in later scripts, written here just for reference
condition_markers = 1:10;
condition_names = {'CueVis';'CueAud';'CueBiVis';'CueBiAud';'StimVis';'StimAud';'StimBiVisMatch';'StimBivisMismatch';'StimBiAudMatch';'StimBiAudMismatch'};
condition_array = [num2cell(condition_markers)',condition_names];
%number of task trials
trials = 480; %4 task conditions x 120 trials

%**name EOG electrode indices used for automated ICA eye artifact removal
%(i.e. eye signal timecourse that ICs are correlated with)
%Available EOG sensors: 
%Left VEOG = 241 (low i.e. below eye), 46 (high); Right VEOG =
%238 (l), 10 (r) - used for blinks
%More inferior HEOG = 234 (left), 244(right); More superior HEOG = 54 (left), 1 (right)
%Use right VEOG electrodes for blink, and more superior HEOG electrodes for eye movements
veog_chans=[238 10]; %lower, upper
heog_chans=[54 1]; %left, right

%**set zscore thresholds for identifying noisy channels, trials, ICs
meanabs_channel_thresh = 3; %if using raw abs amp = 95; if using z-scores = 3
meanabs_trial_thresh = 3; %if using raw abs amp = 30/40; if using z-scores = 3
ica_tempcorr_thresh = 3;

%**set whether you want to run artifact detection in 'semi-automated' mode
%i.e. if set to 1, will visualize the output of each automated detection
%step (for channels, trials and ICs), so the user can visually verify each
%automated detection before removal
run_semiauto = 0; %1=visually inspect outputs of bad channels and bad trials before rejecting
run_ica_semiauto = 1; %1=visually inspect ocular ICs before rejecting

%Set output directory
analysisName=['/Analysis3_causalFilter/'];
outputDir = [baseDir,'data/results/',analysisName];
if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

%REMOVE - bad channs + bad trials per subject should be saved in each
%subject's output file
% %create saved variable coding for bad subjects (based on high number of bad
% %channels + bad trials); rule of thumb for exclusion: bad chans > 25 (~10%); num
% %trials > 96 (20%)
% bad_sub_file=[outputDir,'/Subject_badchannels_badtrials.mat'];
% if ~exist(bad_sub_file)
%     sub_badchans_badtrials=[];
% else
%     load(bad_sub_file);
% end

%% 1. Import EGI .mff data into fieldtrip (start subject loop)

for x = 1:length(subjects)
    %select subject
    subject_num = subjects(x,1);
    %subject_order = find(total_subjects==subject_num);
    outputfile = [outputDir,num2str(subjects(x,1)),'_preproc_hp',num2str(filter_hp),'notch_seg_autochannelz',num2str(meanabs_channel_thresh),'_trialz',num2str(meanabs_trial_thresh),'_ICAz',num2str(ica_tempcorr_thresh),'_baselinelp',num2str(filter_lp),'avgref.mat'];
    ICA_output = [outputDir,num2str(subjects(x,1)),'_segICAcomps_ica_extended_channeltrialthresh',num2str(meanabs_channel_thresh),'.mat'];
    
    %init variables storing bad channs, trials, ICs
    sub_bad_channels={};
    sub_bad_trials={};
    sub_bad_ICs={};
    
    
    
    %specify subject input
    subject_rest_input = [input_dir,num2str(subject_num),'/EEG/',num2str(subject_num),'_rest_pretask.mff'];
    subject_task_input = [input_dir,num2str(subject_num),'/EEG/',num2str(subject_num),'_task.mff'];

%     %Load in data for visual inspection
%     cfg = [];
%     cfg.dataset = subject_input;
%     cfg.headerformat = 'egi_mmf_v2';
%     ft_databrowser(cfg);
%     
%     %Read all events - will help figure out which to use for segmentation
%     [event] = ft_read_event(subject_input,'dataformat','egi_mff_v2','headerformat','egi_mff_v2');

   %% 2. Filter continuous data - notch + highpass

    %TASK
    cfg = [];
    cfg.dataset = [subject_task_input];
    cfg.continuous = 'yes';
    %bs (notch)
    cfg.bsfilter='yes';
    cfg.bsfreq=[59 61; 119 121; 179 181];
    cfg.bsfilttype='firws';
    cfg.bsfiltdir='onepass-minphase';
    %hp
    cfg.hpfilter='yes';
    cfg.hpfreq=filter_hp;
    cfg.hpfilttype='firws';
    cfg.hpfiltdir='onepass-minphase';
    data_task_cont = ft_preprocessing(cfg);

    %REST
    cfg.dataset = [subject_rest_input];
    data_rest_cont = ft_preprocessing(cfg);

  %% 3. Segment into trials/epochs

    %TASK
    %define trials
    cfg = [];
    cfg.dataset = [subject_task_input]; 
    cfg.trialdef.eventtype = 'ECI_TCPIP_55513'; %same as Events_ .xml file
    cfg.trialdef.eventvalue = 'sti+'; %marks stim onset on each trial
    cfg.trialdef.prestim = task_epoch_prestim; %1s stim, 1-1.5-2s jitter (i.e. min 1s) = total trial duration=2s
    cfg.trialdef.poststim = task_epoch_poststim; 
    cfg = ft_definetrial(cfg); %outputs cfg.trl containing beginning, end and offset of each trial in datasamples

    % 3b. Insert check if more than 480 trials are segmented due to extra
    % triggers (can sometimes happen mistakenly); none present in this
    % dataset
    [num_trls,~] = size(cfg.trl);
    if num_trls > 480
        disp('***EXTRA TRIAL IN cfg.trl! Please adjust cfg.trl***')
        keyboard; %waits for keyboard/gui input - can therefore remove the extra trial before continuing the script
    end

    % 3c. Extract condition names from cfg.event
    %Use 'TRSP' event written at the end of each trial by EGI (containing all
    %trial info) to extract: task type, response key, accuracy, RT; use
    %accuracy and response key to identify stim type
    %*Only shortcoming here is that stim type cannot be written for trials
    %without a response (write NaN for these instead), although we will primarily be focusing on correct
    %trials, so don't think this will an issue 
    condition_code = [];
    for trls = 1:size(cfg.event,2)
        if strcmp(cfg.event(trls).value,'TRSP')==1
            trl_info = [];
            trl_task = str2num(cfg.event(trls).orig.keys(1).key.data.data);%see condition_array to check names for each codes
            trl_resp = str2num(cfg.event(trls).orig.keys(3).key.data.data); %115='s', 107 ='k'
            trl_acc = str2num(cfg.event(trls).orig.keys(4).key.data.data); %1 = correct, 0 = incorrect
            trl_RT = str2num(cfg.event(trls).orig.keys(5).key.data.data); %RT in ms

            %write stim_stype; 1 = horizontal/low, 2 = vertical/high
            if trl_resp == 0
                trl_stim = NaN;
            elseif trl_acc == 1
                if trl_resp == 115
                    trl_stim = 1;
                elseif trl_resp == 107
                    trl_stim = 2;
                end
            elseif trl_acc == 0
                if trl_resp == 115
                    trl_stim = 2;
                elseif trl_resp == 107
                    trl_stim = 1;
                end
            end

            trl_info = [trl_task,trl_stim,trl_resp,trl_acc,trl_RT];
            condition_code = [condition_code;trl_info];
        end
    end

    %830 had a task phase that ended early
    if subjects(x)==830
        trial_index = 1:292;
        trl = [cfg.trl,trial_index',condition_code]; %add to trial structure along with raw trial numbers (will be useful for identifying deleted trials later)
    else
        trial_index = 1:trials;
        trl = [cfg.trl,trial_index',condition_code]; %add to trial structure along with raw trial numbers (will be useful for identifying deleted trials later)
    end

    % 3d. Segment + preproc the filtered continuous data
    cfg = [];
    cfg.trl = trl;
    data_task_seg = ft_redefinetrial(cfg,data_task_cont);
    %data_seg_raw = data_seg; %need to save this raw version for later ERP comparisons


    %REST
    cfg = [];
    cfg.dataset = [subject_rest_input]; 
    %segment into stim epochs
    %cfg.trialdef.eventtype = '?'; %used to figure out event types
    cfg.trialdef.eventtype = 'ECI_TCPIP_55513'; %same as Events_ .xml file
    if subjects(x)==823
        %823 has missing 'bgin' marker
        cfg.trialdef.eventvalue = 'CELL'; %marks stim onset on each trial
    else
        cfg.trialdef.eventvalue = 'bgin'; %marks stim onset on each trial
    end
    cfg.trialdef.prestim = rest_epoch_prestim; %1s stim, 1-1.5-2s jitter (i.e. min 1s) = total 2s
    cfg.trialdef.poststim = rest_epoch_poststim; 
    cfg = ft_definetrial(cfg); %outputs cfg.trl containing beginning, end and offset of each trial in datasamples

    %update cfg.trl to create 30 trials * 20s epochs
    trl=cfg.trl;
    start_samp=trl(1);
    num_trls=(rest_epoch_poststim*1000)/rest_epoch_length;
    trl_final=[];
    for t=1:num_trls
        samps=[start_samp,start_samp+rest_epoch_length-1,0];
        trl_final=[trl_final;samps];
        start_samp=start_samp+rest_epoch_length;
    end

    % 3d. Segment + preproc the filtered continuous data
    cfg = [];
    cfg.trl = trl_final;
    data_rest_seg = ft_redefinetrial(cfg,data_rest_cont);
    %data_seg_raw = data_seg; %need to save this raw version for later ERP comparisons


    %% 4. Identify noisy channels
    %Automated procedure - for each channel, takes the absolute average across all samples
    %in the continuous data; threshold based on zscores

    %visualize continuous data - easier to spot bad channels here compared
    %to segmented data
%     cfg = [];
%     cfg.blocksize = 300;
%     cfg.ylim = [-150000 150000];
%     ft_databrowser(cfg, data_cont);

    %Extract continuous signal from all channels
    %loop through task and rest
    tt_types={'task' 'rest'};
    for ttt=1:length(tt_types)
        tt_t=tt_types{ttt};
        eval(['cont_data=data_',tt_t,'_cont.trial{1,1};']);

        %Compare different methods of identifying noisy channels - mean absolute
        %value performs best; variance works almost as well as meanabs but harder to threshold
        %in intuitive fashion, kurtosis performs poorly
        meanabs_channel = nanmean(abs(cont_data),2);
        %var_channel = var(cont_data,0,2);
        %kurt_channel = kurtosis(cont_data,1,2);

        %convert to zscore - accommodates 'baseline' differences between
        %subjects better than thresholding the raw abs amps
        meanabs_channel = zscore(meanabs_channel);

        %Send to function that plots channel- or trial-specific measures
        [meanabs_channel_sorted] = Visualize_NoiseMeasure_ChannelTrial(meanabs_channel,50);
        %[var_channel_sorted] = Visualize_NoiseMeasure_ChannelTrial(var_channel,50);

        %store above-threshold channelsin vector 'bad_channels'
        bad_channels = meanabs_channel_sorted(meanabs_channel_sorted(:,1)>meanabs_channel_thresh,2);
        good_channels = meanabs_channel_sorted(51:60,2); %supposed to load mid range channels for comparison with bad channels, but this isn't working apparently

        %insert check for high number of bad channels, in which case you should
%         %consider removing the subject...
%         if length(bad_channels) > 25
%             disp('WARNING: subject has more than 25 channels that exceed the noise threshold - consider removing subject?')
%             keyboard;
%         end

        %visualize the above-threshold channels in the continuous data - in
        %case you want to make sure visually that channels are worth throwing out
        if run_semiauto==1

            figure; hold on;
            cfg = [];
            cfg.blocksize = 300;
            cfg.ylim = [-150000 150000];
            %Set channels to visualize - above-threshold ones, and 10 mid-range normal
            %electrodes (for comparison); this isn't working - loads up all
            %channels rather than specified ones!
            channel_vis = [bad_channels',good_channels'];
            eval(['cfg.channels = data_',tt_t,'_cont.label(channel_vis,1);']);
            eval(['ft_databrowser(cfg, data_',tt_t,'_cont);']);

            %throw to keyboard in case you want to update the above-threshold artifacts
            disp('Modify bad_channels based on visual inspection?')
            keyboard; %waits for keyboard/gui input - modify bad_channels in command window before advancin script

        end

        %reject above-threshold channels
        if ~isempty(bad_channels) %if bad_channels is not empty willl do following steps
            eval(['channel_inds=data_',tt_t,'_seg.label;']);
            channel_inds(bad_channels)=[];
            eval(['data_',tt_t,'_seg=ft_selectdata(data_',tt_t,'_seg,''channel'',channel_inds);']);
        end

        %store in var
        sub_bad_channels{ttt}=bad_channels;

     %% 5. Identify noisy trials
        % Automated procedure - for each trial, takes the absolute average across all
        % channels for all samples

        %assign task/rest data 
        eval(['data_seg=data_',tt_t,'_seg;']);
        num_chans=size(data_seg.label,1); %used to exclude VREF from noisy trial computation, but don't think this is necessary

        % Loop through trials and take the absolute average across all channels and
        % samples
        meanabs_trial = [];
        for tt = 1:size(data_seg.trial,2)
            %mean_tt = nanmean(abs(data_seg.trial{1,tt}(1:num_chans-1,:)),2);
            mean_tt=nanmean(abs(data_seg.trial{1,tt}),2);
            mean_tt=nanmean(mean_tt);
            meanabs_trial = [meanabs_trial;mean_tt];
        end

        %Convert to zscores (as with noisy channels)
        meanabs_trial = zscore(meanabs_trial);

        %Send to function that plots channel- or trial-specific measures
        [meanabs_trial_sorted] = Visualize_NoiseMeasure_ChannelTrial(meanabs_trial,num_trls);

        %store above-threshold trials in vector 'bad_trials'
        bad_trials = meanabs_trial_sorted(meanabs_trial_sorted(:,1)>meanabs_trial_thresh,2);
        %good_trials = meanabs_trial_sorted(51:60,2);

        %visualize the above-threshold trials - in case you want to make sure 
        %visually that trials are worth throwing out
        if run_semiauto==1

            figure; hold on;
            cfg = [];
            %cfg.blocksize = 300;
            cfg.ylim = [-60000 60000];
            ft_databrowser(cfg, data_seg);

            %throw to keyboard in case you want to update the above-threshold artifacts
            disp('Modify bad_trials based on visual inspection?')
            keyboard; %waits for keyboard/gui input - modify bad_channels in command window before advancin script

        end

        %reject above-threshold trials
        if ~isempty(bad_trials)
            cfg = [];
            good_trials = 1:tt;
            good_trials(bad_trials)=[];
            cfg.trials = good_trials;
            data_seg = ft_selectdata(cfg,data_seg);
        end

        %data_seg_channels_trials = data_seg; %need to save this for ERP comparisons

        %*assign to task/rest
        eval(['data_',tt_t,'_seg=data_seg;']);

        %store in var
        sub_bad_trials{ttt}=bad_trials;
    end


   %% 6. Perform ICA
    %Using binica extended, which is identical to runica extended (infomax)
    %albeit a binary version so much faster

    %check if ICA has been run already
    if exist(ICA_output)
        load(ICA_output);
    else
        %loop through task types and compute ica
        tic;
        for ttt=1:length(tt_types)
            tt_t=tt_types{ttt};
            eval(['data_seg=data_',tt_t,'_seg;']);

            cfg = [];
            cfg.method = 'binica';
            cfg.binica.extended = 1;
            %exclude reference channel from ICA - otherwise matrix is rank deficient
            num_chans = size(data_seg.label,1);
            cfg.channel = data_seg.label(1:num_chans-1,1);
            %cfg.runica.pca = 150; %reduces input to specified number of PCs
            [comp] = ft_componentanalysis(cfg,data_seg);

            %*assign to task/rest
            eval(['comp_',tt_t,'=comp;']);
        end

        %save ICA results
        save(ICA_output,'comp_task','comp_rest','-v7.3');
        display(['time taken to run ICA for ',tt_t,' = ',num2str(toc)]);

    end
    
    
    %% 7a. Apply Automated ICA procedures - icablinkmetrics
        
    %check if run_ica_inspect == 1, in which case run pipeline up to ICA on task and rest data for this subject
    %otherwise if ==2 load up previous results for inspection, reject and finish up
    if run_ica_inspect==2
        
        % Based on icablinkmetrics developed by Pontifex et al., 2017,
        % Psychophys. Adapted for Fieldtrip (as converting from Fieldtrip to
        % EEGLAB was proving very annoying; also icablinkmetrics does not work 
        % on segmented data). Basic idea is to look at the absolute 
        % Pearson's correlation of the IC timecourses, with the
        % VEOG channels (for blink ICs) and the HEOG channels (for eye
        % movement ICs). Note that I'm borrowing a step from SASICA (Chaumon et al., 2015) 
        % by subtracting the two VEOG and HEOG channels from each other,
        % to create veog_diff and heog_diff vectors, that will have a higher
        % eyeartifact SNR

        %can transpose below to a function - takes comp, data_seg, VEOG + HEOG
        %channel definitions, threshold; *do this later...

        %1. Specify and prepare VEOG electroide
        %set veog and heog afresh for each subject (in case they had to be
        %changed due to noisy channels)
        veog=veog_chans;
        heog=heog_chans;

        %loop through task and rest
        for ttt=1:length(tt_types)
            tt_t=tt_types{ttt};
            %assign variables
            eval(['data_seg=data_',tt_t,'_seg;']);
            eval(['comp=comp_',tt_t,';']);
            bad_channels=sub_bad_channels{1,ttt};
            bad_trials=sub_bad_trials{1,ttt};
            
            %include check in case VEOG was removed during noisy channel step; if
            %so then swap with alternate (left low/hi VEOG electrodes)
            if any(ismember(veog,bad_channels))==1
                veog = [241 46];
                if any(ismember(veog,bad_channels))==1
                    disp('Error! All veog channels were removed during noisy channel rejection. Modify veog manually?')
                    keyboard;
                end
            end

            % Pick out VEOG channel indices; will use to reference data_seg and concatenate trials later
            low_index = find(strcmp(data_seg.label, ['E',num2str(veog(1,1))]));
            hi_index = find(strcmp(data_seg.label, ['E',num2str(veog(1,2))]));


            %2. Repeat for HEOG electrode...
            %include check in case VEOG was removed during noisy channel step...
            if any(ismember(heog,bad_channels))==1
                heog = [234 244];
                if any(ismember(heog,bad_channels))==1
                    disp('Error! All heog channels were removed during noisy channel rejection. Modify heog manually?')
                    keyboard;
                end
            end

            %pick out HEOG channel indices etc
            l_index = find(strcmp(data_seg.label, ['E',num2str(heog(1,1))]));
            r_index = find(strcmp(data_seg.label, ['E',num2str(heog(1,2))]));

            %3. Pull out IC timecourses (comp.trial), concatenate into one long
            %comp x samples array; do the same for the veog and heog channels in
            %data.seg...
            curr_trials = size(comp.trial,2);
            ic_timecourses = [];
            veog_tcs = [];
            heog_tcs = [];
            for tt=1:curr_trials
                ic_timecourses = [ic_timecourses,comp.trial{1,tt}];
                veog_tt = [data_seg.trial{1,tt}(low_index,:);data_seg.trial{1,tt}(hi_index,:)];
                veog_tcs = [veog_tcs,veog_tt];
                heog_tt = [data_seg.trial{1,tt}(l_index,:);data_seg.trial{1,tt}(r_index,:)];
                heog_tcs = [heog_tcs,heog_tt];
            end

            %create difference channels for veog and heog (as per SAUSICA)
            veog_diff = veog_tcs(1,:)-veog_tcs(2,:);
            heog_diff = heog_tcs(1,:)-heog_tcs(2,:);

            %lowpass filter the veog and heog channels - might make identifying eye
            %artifacts easier? Didn't help in practice...
        %     fc = 50; % Cut off frequency
        %     fs = 1000; % Sampling rate
        % 
        %     [b,a] = butter(6,fc/(fs/2)); % Butterworth filter of order 6
        %     veog_diff = filter(b,a,veog_diff); % Will be the filtered signal
        %     heog_diff = filter(b,a,heog_diff); % Will be the filtered signal

            %4. Loop through and correlate each IC timecourse with the VEOG timecourse
            %(to identify blinks)
            %Run blink corr first, and exclude any blink timecourses from
            %the eye movement timecourses

            %Blink (veog) correlation
            [tempcorr_blink_r tempcorr_blink_p]  = corr(veog_diff',ic_timecourses');


            %Sort ICs in descending order of abs corr and plot the results
            %Using zscore rather than raw absolute correlation as before
            %[tempcorr_blink_sorted] = Visualize_NoiseMeasure_ChannelTrial(abs(tempcorr_blink_r)',50);
            [tempcorr_blink_sorted_z] = Visualize_NoiseMeasure_ChannelTrial(zscore(abs(tempcorr_blink_r))',50);

            %reject based on zscore threshold and store in bad_
            bad_ICs_blinks_tempcorr = tempcorr_blink_sorted_z(tempcorr_blink_sorted_z(:,1)>ica_tempcorr_thresh,2);

            %5. Repeat for eye movements
            %Eye movement (heog) correlation
            [tempcorr_eyemov_r tempcorr_eyemov_p] = corr(heog_diff',ic_timecourses');
            tempcorr_eyemov_r_z = zscore(abs(tempcorr_eyemov_r));

            %Remove correlation with eyeblink ICs - replace with Nan. Blink ICs will also
            %correlate with heog, but are more likely to be blinks if already
            %correlated with veog...
            %tempcorr_eyemov_r(bad_ICs_blinks_tempcorr)=NaN;
            tempcorr_eyemov_r_z(bad_ICs_blinks_tempcorr)=NaN;

            %[tempcorr_eyemov_sorted] = Visualize_NoiseMeasure_ChannelTrial(abs(tempcorr_eyemov_r)',50);
            [tempcorr_eyemov_sorted_z] = Visualize_NoiseMeasure_ChannelTrial(tempcorr_eyemov_r_z',50);

            %reject based on zscore (or r) threshold
            bad_ICs_eyemov_tempcorr = tempcorr_eyemov_sorted_z(tempcorr_eyemov_sorted_z(:,1)>ica_tempcorr_thresh,2);


         %% 7c. Reject artifact ICs
            %Reject on the basis of 1) Full automation - tempcorr (blinks+eyemov), OR
            %2) Automation with final visual verification of auto ICs
            %(blinks+eyemov), this approach only visually confirms/rejects
            %components that have *already* been identified via the automated methods

            %1. Set up tempcorrvis vectors for visualisation
            bad_ICs_blinks_tempcorrvis = bad_ICs_blinks_tempcorr;
            bad_ICs_eyemov_tempcorrvis = bad_ICs_eyemov_tempcorr;

            if run_ica_semiauto == 1
                %Visualise topography of components - detect artifacts
                cfg = [];
                cfg.layout = elecfile; %easier to visualise on magnetometers
                cfg.colormap = 'jet';
                cfg.component = [1:40];
                cfg.comment = 'no';
                %cfg.interplimits = 'electrodes'; %vs default 'head'
                cfg.marker = 'off';
                ft_topoplotIC(cfg,comp);

                %Visualise timecourse of components - detect artifacts
                cfg = [];
                cfg.layout = elecfile;
                %cfg.zlim = [-10 10]
                cfg.colormap = 'jet';
                cfg.viewmode = 'component';
                ft_databrowser(cfg,comp);

                disp('Modify bad_ICs*tempcorrvis? based on visual inspection')
                keyboard;
            end

            %2. Reject components
            %collapse ICs
            bad_ICs_tempcorr = [bad_ICs_blinks_tempcorr',bad_ICs_eyemov_tempcorr'];
            bad_ICs_tempcorrvis = [bad_ICs_blinks_tempcorrvis',bad_ICs_eyemov_tempcorrvis'];

%             %Reject auto - tempcorr
%             cfg = [];
%             cfg.component = bad_ICs_tempcorr; %enter artifact components to be removed
%             data_seg_channels_trials_autoIC = ft_rejectcomponent(cfg,comp,data_seg);

            %Reject auto+vis - tempcorrvis
            cfg = [];
            cfg.component = bad_ICs_tempcorrvis; %enter artifact components to be removed
            data_seg_channels_trials_autovisIC = ft_rejectcomponent(cfg,comp,data_seg);

            clear data_seg;

          %% 8. Baseline correct + lowpass filter (100Hz) + rereference to common average
            %apply to all data_segs i.e. _raw, raw_channels_trials,
            %raw_channels_trials_autoIC, raw_channels_trials_autovisIC, 
            
            %baseline correct
            cfg = [];
            cfg.demean = 'yes';
            if ttt==1
                %task
                cfg.baselinewindow = [task_epoch_baseline 0];
            elseif ttt==2
                %rest
                cfg.baselinewindow = 'all';
            end

            %lowpass filter
            cfg.lpfilter = 'yes';
            cfg.lpfreq = filter_lp;
            cfg.lpfilttype='firws';
            cfg.lpfiltdir='onepass-minphase';

            %reref to common avg
            cfg.reref = 'yes';
            cfg.refchannel = 'all';

            %data_seg_raw = ft_preprocessing(cfg,data_seg_raw);
            %data_seg_channels_trials = ft_preprocessing(cfg,data_seg_channels_trials);
            %data_seg_channels_trials_autoIC = ft_preprocessing(cfg,data_seg_channels_trials_autoIC);
            data_seg_channels_trials_autovisIC = ft_preprocessing(cfg,data_seg_channels_trials_autovisIC);
            
            %*assign to task/rest vars
            sub_bad_ICs{ttt}=bad_ICs_tempcorrvis;
            eval([tt_t,'_data_seg_channels_trials_autovisIC=data_seg_channels_trials_autovisIC;']);
            
        end

       %% 9. Save final outputs (from various approaches i.e. automated procs, and visual + automated procs)
        
        save(outputfile,'sub_bad_ICs','sub_bad_channels','sub_bad_trials','task_data_seg_channels_trials_autovisIC','rest_data_seg_channels_trials_autovisIC','-v7.3');   
        
    end
     
end

end

