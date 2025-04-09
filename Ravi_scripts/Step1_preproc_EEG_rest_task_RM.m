function Step1_preproc_EEG_rest_task_RM(subjects, run_ica_inspect)

%Author: Ravi Mill, rdm146@rutgers.edu; Lily Thompson, let83@rutgers.edu
%Last update: Jan 30th 2025

% I take data from CPRO EEG folder and then save the preproc data in
% cpro2_eeg/data/participant/eeg/dataName right? --> nope cpro2_eeg/results/preproc1/rep_geom

%DESCRIPTION: preprocesses EEG data (task and rest) for CPRO2_EEG
%Flow project: Mill, XX, "Title" 


%REORGANIZED
%1a. Imports EGI data into fieldtrip format (checked)
%1.  Filters continuous data (using minimum phase causal filter): 60Hz
%notch + 1Hz highpass (checked)
%2.  Identify and remove noisy channels (checked)
%3.  Segment into trial epochs (checked)
%4.  ICA on trial data 
%5.  Identify and reject ICA components
%6.  Lowpass filter (125 Hz -- continuous data) (checked)
%7.  Identify and remove noisy trials (checked)
%8.  Segmentation 2 ~ split into events within trial (checked)
%9.  Baseline correct and find common average (I think this was checked)
%9b? Downsampling -- decide later
%10. Save outputs from various approaches 


%INPUTS
%subjects=numeric nx1 array, each row containing subject IDs 
%run_ica_inspect=numeric, determines which version of the function
    %is run; 1 (step1): preprocessing is run till ICA estimation and saved (run
    %using slurm batch), 2 (step2): loads in ICA results and allows visual
    %inspection of potential artifacts followed by IC rejection, lp
    %filtering etc
   %THIS MAY BE AN OLD FUNCTION 

%OUTPUTS
%Writes results of steps leading up to and including ICA (run_ica_inspect=1), followed
%by final preproc results (after visually screening ICA results; lp filtering, basebinicalining and common avg
%rereferencing (run_ica_inspect==2)

%% Set path and defaults for Fieldtrip
addpath /projectsn/f_mc1689_1/cpro2_eeg/docs/toolboxes/fieldtrip-20240704;
addpath /projectsn/f_mc1689_1/baby_actflow/docs/toolboxes/eeglab13_6_5b/;
ft_defaults;


%path to egi electrode layout (in Fieldtrip templates directory)
elecfile = '/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/template/electrode/GSN-HydroCel-257.sfp';

%% Set inputs, parms etc

%set path to inputs
%Behavioral data
path_to_behavior = '/projectsn/f_mc1689_1/cpro2_eeg/data/rawdata/CPRO2_fMRI_EEG_main.xlsx';
behavior = readtable(path_to_behavior);

%Path to EEG data
baseDir='/projects/f_mc1689_1/CPRO2_learning/';
input_dir = [baseDir,'data/rawdata/'];%participant/EEG/data_folder.mff

%REMOVE - now runs task and rest_pretask for all subjects
%**set to preproc 'task', 'rest_pretask', or 'rest_posttask';
%run_phase = 'rest_pretask';

%**set event name for task event preproc: 'stim', 'cue', 'resp' -- I don't
%THINK I need this
%task_event='stim';

%**set filter pass bands; note these are sequentially applied as this i) is
%advocated for ICA (from EEGLAB list), ii) yielded less delay relative to
%the raw/butterworth filtered signal than bandpass filtering (see Step0
%script)
filter_hp=.1;
filter_lp=125;

%**set epoch length (in secs) for rest and task
%rest; rest is baselined based on pseudotrial mean %DO I NEED TO CHANGE THE
%EPOCHS FOR REST DATA --> same measurements, not running on rest data yet
%rest_epoch_prestim=0; %in s; i.e. pull out 0-600s continuous EEG data around 'start recording' marker sent by eprime to EGI
%rest_epoch_poststim=600; %in s
%rest_epoch_length=20000; %in samples, used to segment 600s rest segment into 20s pseudotrials

%~~might help to provide fuller descriptions of the segmentations; have
%tried to do so below
%task segmentation 1
%~~Entire task block i.e. -90ms from enc1 through 1520ms after probe 2 on
%trial 3; ICA is done on this segmentation to ensure that removed
%components (and dimensionality) are matched across encoding/ITI/probe events
task_epoch_seg1_pretrial=0.09; %in s
task_epoch_seg1_posttrial = 1.52;
task_epoch_seg1_baseline=-0.09; %baseline will be -0.1 to 0

%task segmentation 2 - encoding screens
instruction_epoch_pre = 0.09; % Pre-instruction screen (100 ms before onset)
instruction_epoch_post = 0.9; % Post-instruction screen (900 ms after onset)
instruction_epoch_baseline = -0.09;

%task segmentation 2 - encoding-to-trial ITI
instruction_to_probe_offset = 1.4+.995; % Instruction to probe (1400 ms after instruction 3 offset (enc3+995))
instruction_to_probe_baseline = -0.09;
instruction_to_probe_pre = 0.09;

%task segmentation 2 - trial probes (probe 1 onset through probe 2 offset)
probe1_epoch_pre = 0.09; % Pre-probe 1 (100 ms before probe onset)
probe2_epoch_post = 1.52; % Probe 2 (from probe 1 to end of response window)
probe1_epoch_baseline = -0.09;

%set condition codes for task - used in MVPA analyses in later scripts, written here just for reference
%rule_markers = 1:4;
%logic_rules = {'Same';'Different'; 'Either';'Neither'};
%sensory_rules = {'Young face';'Indoor Scene';'Female Voice';'Word Phrase'};
%motor_rules = {'Left Middle';'Left Index';'Right Middle';'Right Index'};
%rules_array = [num2cell(rule_markers)',logic_rules,sensory_rules,motor_rules];

%~~ should be "task blocks" per CPRO terminology ("trials" are the 3 probe1-probe2 
%events within each block)
%number of task trials
total_blocks = 120; %15*4 practice trials and 60 novel mini blocks
total_trials = 360;

%**name EOG electrode indices used for automated ICA eye artifact removal
%(i.e. eye signal timecourse that ICs are correlated with)
%Available EOG sensors: 
%Left VEOG = 241 (low i.e. below eye), 46 (high); Right VEOG =
%238 (l), 10 (r) - used for blinks
%More inferior HEOG = 234 (left), 244(right); More superior HEOG = 54 (left), 1 (right)
%Use right VEOG electrodes for blink, and more superior HEOG electrodes for
%eye movements 
% are these values right?
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

%~~the outputDir will write results to CPRO2_learning; it should write to
%your cpro2_eeg directory, so you'll have to change how you assign the
%directory name below eg. change baseDir to rawDir, and set baseDir to the
%cpro2_eeg directory
%Set output directory
analysisName=['/preproc1_causalFilter/'];
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
disp('made it past initial naming stuff')
%% 1a. Import EGI .mff data into fieldtrip (start subject loop)
%there are some functions used throughout this code, I'm not exactly sure
%how running things on amarel works, but would I be able to access them, or
%do I need to write them?
%CHANGE AFTER PROOFING

%~~you can make more descriptive disp outputs by using the [] operators
%e.g.: disp(['number of subjects = ',num2str(length(subjects))]);
disp(length(subjects))
disp(subjects)
for x = 1:length(subjects)
    %select subject
    subject_num = subjects(1,x);
    disp(subject_num);
    subj_behavioral = behavior(behavior.Subject == subject_num, :);
    %subject_order = find(total_subjects==subject_num);
    
    %~~I would recommend adding 'sub' as a prefix to the individual files
    %being saved, as numeric prefixes filenames can confuse certain
    %programs (admittedly I made this error myself when naming the rawdata)
    outputfile = [outputDir,num2str(subjects(1,x)),'_preproc_hp',num2str(filter_hp),'notch_seg_autochannelz',num2str(meanabs_channel_thresh),'_trialz',num2str(meanabs_trial_thresh),'_ICAz',num2str(ica_tempcorr_thresh),'_baselinelp',num2str(filter_lp),'avgref.mat'];
    ICA_output = [outputDir,num2str(subjects(1,x)),'_segICAcomps_ica_extended_channeltrialthresh',num2str(meanabs_channel_thresh),'.mat'];
    
    %init variables storing bad channs, trials, ICs
    sub_bad_channels={};
    sub_bad_trials={};
    sub_bad_ICs={};    
    
    %specify subject input
    %subject_rest_input = [input_dir,num2str(subject_num),'/EEG/',num2str(subject_num),'_rest_pretask.mff'];
    subject_task_input_p1 = [input_dir,num2str(subject_num),'/EEG/'];
    subject_task_input_path = dir(fullfile(subject_task_input_p1, 'CPRO2_Task_*'));
    subject_task_input = fullfile(subject_task_input_path(1).folder, subject_task_input_path(1).name);
    disp(subject_task_input)
    %subject_task_input = ['/projectsn/f_mc1689_1/CPRO2_learning/data/rawdata/10/EEG/', 'CPRO2_Task_10_20180222_033003.mff'];
    disp('made it past naming the dataset')
   %% 1. Filter continuous data - notch + highpass

    %TASK
    tic
    cfg = [];
    cfg.dataset = subject_task_input;
    cfg.continuous = 'yes';
    %bs (notch)-- make configuration to store paramaters and path to data
    cfg.bsfilter='yes';
    cfg.bsfreq=[59 61; 119 121; 179 181];
    cfg.bsfilttype='firws';
    cfg.bsfiltdir='onepass-minphase';
    %hp -- make configuration to store paramaters and path to data
    cfg.hpfilter='yes';
    cfg.hpfreq=filter_hp;
    cfg.hpfilttype='firws';
    cfg.hpfiltdir='onepass-minphase';
    data_task_cont = ft_preprocessing(cfg);%actually does the filtering
    toc
    fprintf('Filtering the continuous data took %.2f seconds\n', toc);
    
    %~~I would advise using Matlab breakpoints (click on the dash next to
    %line so that a red circle pops up) rather than using the input()
    %commands below; when we parallelize subjects, the code has to
    %be able to run through without waiting for user input; 
    print_value = input('continue?: Y=1 N=0');
    full_time_info = [];
    full_time_info = [full_time_info;toc, 'filtering continuous data'];
    disp('made it past ft_preprocessing task')
    %%REST -- not doing yet
    %cfg.dataset = [subject_rest_input];
    %data_rest_cont = ft_preprocessing(cfg);
  
  %% 2. Identify and Remove Noisy Channels
    %Automated procedure - for each channel, takes the absolute average
    %across all samles in the continuous data; threshold based on zscores

    %visualize continuous data - spot bad channels in continuous data
    % cfg = [];
    % cfg.blocksize = 300;
    % cfg.ylim = [-150000 150000];
    % ft_databrowser(cfg, data_data_cont);
    
    %~~ ideally you would replace '3' in the inputs below with
    %meanabs_channel_thresh that was set at the start of the script, that
    %way you can change parameters efficiently if needed versus scrolling
    %through the entire script
    %~~ also, line 53 in the identify_bad* function below has a keyboard
    %command that will also mess up with the parallelization; I think we
    %can remove the keyboard call, and instead make it a practice to check
    %the outputs from parallelization for the WARNING that is posted via
    %disp, so that we can identify bad subjects with 25+ bad channels
    %~~ line 57 in the identify_bad* function also has an input() call that
    %should be removed
    
    %TEMPORARY
    [sub_bad_channels,data_task_cont] =  identify_bad_channels_or_trials(data_task_cont, "Channel",3, 0,'task',1);
    disp('made it past identifying and removing noisy channels')
    % enter in later
    % tt_types = {'task','rest'};
    % for i =1:length(tt_types)
    %     tt_t = tt_types(i);
    %     eval(['sub_bad_channels,data_',tt_t,'_cont =  identify_bad_channels_or_trials(data_',tt_t,'_cont, "Channel",meanabs_channel_thresh, 0,',tt_t,',',i,');']);
    % end
  %% 3. Segment into trials/epochs
    
    %TASK
    %define trials 
    %Find events
    
    %~~ the segmentation attempt below that I've now commented out is not correct,
    %due to 1) treating the first 'bgin' event as the first recorded sample, 2)
    %reconstructing the samples from the event.begintime information,
    %rather than using the directly recorded samples in event.sample, 3) not
    %using fieldtrip's ft_definetrial function which simplifies a lot of
    %this (I had emphasized starting with this approach in our meeting, but 
    %I guess you forgot) 
    
    %set trial_conds
    %1-3=enc screens
    %4=enc-to-trial ITI (written with reference to enc3)
    %5=trial probes (pr1+ through pr2+)
    trial_cond_evs={'enc1','enc2','enc3','enc3','pr1+'};%event markers written by EGI
    trial_cond_names={'enc1','enc2','enc3','enc3_ITI','probes'};%output subfields in the trial_defs struct
    
    %init out variable
    trial_defs=struct;
    
    %loop through trial_cond_evs and output the various definitions that we
    %will use to segment the data
    for tt=1:length(trial_cond_evs)
        %pick out event
        tt_event=trial_cond_evs{tt};
        tt_out=trial_cond_names{tt};
        
        %set pre/post epoch times for each event
        if tt<4
            %enc screen defs (used for final segmentation)
            %-90 to 900ms
            epoch_prestim=instruction_epoch_pre;
            epoch_poststim=instruction_epoch_post;
            
        elseif tt==4
            %enc-trial ITI
            %-90 from enc3 offset to 1400ms
            %i.e. set pre=0 for now, post=995ms+1400ms
            %use ft_redefinetrial later to re-epoch
            epoch_prestim=0;
            epoch_poststim=instruction_to_probe_offset;
        else
            %probe
            %-90 from pr1+ onset through 1520 after pr2+ offset
            %(above covers response window of 3850ms from pr1+ onset)
            %i.e. poststim from pr1+=1000+335+1000+1520=3855 
            epoch_prestim=probe1_epoch_pre;
            epoch_poststim=3.855;
            
        end
            
            
        
        %output trial definition using fieldtrip function
        cfg=[];
        cfg.dataset= subject_task_input;                   
        cfg.trialfun='ft_trialfun_general';             %default to read events 
        cfg.trialdef.eventtype= tt_event;        
        cfg.trialdef.eventvalue= tt_event;
        cfg.trialdef.prestim=epoch_prestim;        %
        cfg.trialdef.poststim= epoch_poststim;     %
        cfg_trial=ft_definetrial(cfg);
        
        %modify enc-trial  ITI
        if tt==4
            %begsample needs to be +905 samples (-90 from end of enc3 =
            %995-90=905 from start of enc3)
            cfg_trial.trl.begsample=cfg_trial.trl.begsample+905;
            
            %also update offset (relative to 0=event marker being written)
            vec=repmat(905,[size(cfg_trial.trl,1),1]);
            cfg_trial.trl.offset=vec;
            
            tt_out='enc3_ITI';
%         elseif tt==5
%             %matlab won't accept a '+' in the struct subfield name
%             tt_out='probes';
%         else
%             tt_out=tt_event;
            
        end
        
        %~~ add condition vars here - see my later notes for why the original 
        %approach was flawed 
        trl_info=table;
        %Block-level vars (add to enc1-3, and encITI)
        %1.EncScreenOrder: order of rules was counterbalanced across subjects;
        %   1-6, see behavioral excel sheet to figure out the order
        trl_info.EncScreenOrder=subj_behavioral.EncScreenOrder;

        %2. PracTaskOrder: order of which specific PRO tasks were assigned to
        %   prac vs novel was counterbalanced at group level; 1-16
        trl_info.PracTaskOrder=subj_behavioral.PracTaskOrder;

        %Subject condition vars
        %3.Cross-subject task index (will be useful if we end up aggregating
        %   analyses at the group level): TaskCode
        trl_info.TaskCode=subj_behavioral.TaskCode;

        %4.BlockCond: codes for practiced tasks (2 Switch, 2 NoSwitch) versus 
        %   novel tasks  
        trl_info.BlockCond=subj_behavioral.BlockCond;

        %5.Task Rules: LogicRule, MotorRule, SensoryRule
        trl_info.LogicRule=subj_behavioral.LogicRule;
        trl_info.MotorRule=subj_behavioral.MotorRule;
        trl_info.SensoryRule=subj_behavioral.SensoryRule;
        
        if tt==5
            %probe; 360 trials
            %Add to probes only
            %6. Specific probe stimuli: AudFileProbe1, AudFileProbe2, VisFileProbe1, 
            %   VisFileProbe2
            trl_info.AudFileProbe1=subj_behavioral.AudFileProbe1;
            trl_info.AudFileProbe2=subj_behavioral.AudFileProbe2;
            trl_info.VisFileProbe1=subj_behavioral.VisFileProbe1;
            trl_info.VisFileProbe2=subj_behavioral.VisFileProbe2;

            %7. Behavior: Probe1.ACC, Probe1.RESP, Probe1.RT
            trl_info.acc=subj_behavioral.Probe1_ACC;
            trl_info.resp=subj_behavioral.Probe1_RESP;
            trl_info.rt=subj_behavioral.Probe1_RT;
            
            %8. **add coding of whether task could be responded to on probe
            %1? i.e. for rules either/neither, referencing first stimulus
            %skip for now - can do this later based on information stored
            
        else
            %enc1-3, enc3_ITI
            num_trials=size(subj_behavioral,1);
            vec_ref=3:3:num_trials;
            
            %subsample from 360 trials to 120 blocks
            trl_info=trl_info(vec_ref,:);
        end
        
        %add trl_info to cfg_trial.trl
        cfg_trial.trl=cat(2,cfg_trial.trl,trl_info);
        
        %store
        trial_defs.(tt_out)=cfg_trial;
        
    end
    
    %*create 'task' segmentation for entire blocks from begsample enc1
    %(-90) through endsample of probes on trial 3 of each block;
    %just need to swap out endsample from enc1 seg with correct sample index 
    %from probes...
    num_blocks=size(trial_defs.enc1.trl,1);
    num_trials=size(trial_defs.probes.trl,1);
    
    %pull out vars - enc1 to replace its endsample col
    cfg_trial=trial_defs.enc1;
    %probe
    probes_endsamp=trial_defs.probes.trl.endsample;
    %generate vector of indices of every third trial
    vec_ref=3:3:num_trials;
    %ref to pull out endsamp for every third trial
    end_samp=probes_endsamp(vec_ref);
    %swap out endsamp
    cfg_trial.trl.endsample=end_samp;
    
    %segment data based on tone pairs 
    cfg = [];
    cfg.trl = cfg_trial.trl;                    
    data_task_seg_1 = ft_redefinetrial(cfg,data_task_cont); %stored segmented data 
    
%     
%     [event]=ft_read_event(subject_task_input);
%     %make list of event types and their start times
%     types={event.type};
%     begin_times = {event.begintime};
%     %Find initial time point to find number of samples that occur within the
%     %first 6 items of types and begin times 
%     k = string(begin_times(1, 1));
%     timestamp_str = extractBetween(k, 'T', '-'); % Extract hour to second
%     time_parts = split(timestamp_str,':'); %Separate hour min and sec to change into seconds
%     s = str2double(time_parts{3});
%     h = str2double(time_parts{1})*3600;
%     m = str2double(time_parts{2})*60;
%     start_time = h+m+s;
% 
%     k = string(begin_times(1, 6));
%     timestamp_str = extractBetween(k, 'T', '-'); % Extract hour to second
%     time_parts = split(timestamp_str,':'); %Separate hour min and sec to change into seconds
%     s = str2double(time_parts{3});
%     h = str2double(time_parts{1})*3600;
%     m = str2double(time_parts{2})*60;
%     end_time = h+m+s;
% 
%     time_diff = end_time - start_time;
%     starting_samples = time_diff*1000;
% 
%     %analyze actual experiment events
%     types = types(6:end);
%     begin_times = begin_times(6:end);
%     %convert timestamps into actual timecourse
%     initial_time=0;
%     initial_sample_number = 0;
%     times = [];
%     event_start_samples = [];
%     for i=1:length(types)
%         k = string(begin_times(1, i));
%         timestamp_str = extractBetween(k, 'T', '-'); % Extract hour to second
%         time_parts = split(timestamp_str,':'); %Separate hour min and sec to change into seconds
%         s = str2double(time_parts{3});
%         h = str2double(time_parts{1})*3600;
%         m = str2double(time_parts{2})*60;
%     
%         if initial_time==0
%             initial_time=h+m+s;
%         end
%         t_tot = h+m+s;
%         t=t_tot-initial_time; %find time passed since the enc1 events started
%         s = starting_samples +t*1000; %find total number of samples taken since experiment started
%         times = [times,t]; %make list of event time points in seconds 
%         event_start_samples = [event_start_samples, s]; %make list of what sample numbers events began
%     end
% 
%     %Find time points of enc1 and pr2+
%     enc1_times = [];
%     enc1_samples = [];
%     pr2_times = [];
%     pr2_samples = [];
%     pr2 = 0;
% 
%     for i = 1:length(times)
%         if strcmp(types{1, i}, 'enc1')  
%             enc1_times = [enc1_times, times(1, i)]; 
%             enc1_samples = [enc1_samples, event_start_samples(1,i)];
%         elseif strcmp(types{1, i}, 'pr2+')  
%             pr2 = pr2 + 1;
%             if pr2 == 3
%                 pr2_times = [pr2_times, times(1, i)];
%                 pr2_samples = [pr2_samples, event_start_samples(1,i)];
%                 pr2 = 0; 
%             end
%         end
%     end
%     %Define trials around enc1 and pr2+
%     % Initialize trial matrix
%     trl = [];
% 
%     % Define segments based on enc1 and pr2+ SAMPLE INDICES
%     for i = 1:length(enc1_samples)
%         %Define sample points of start and stop events
%         enc1_sample = enc1_samples(1,i);
%         pr2_sample = pr2_samples(1,i);
% 
%         % Define segment
%         start_sample = enc1_sample - task_epoch_seg1_pretrial*1000; % 100 ms before enc1
%         end_sample = pr2_sample + task_epoch_seg1_posttrial*1000; % 1520 ms after last pr2+ (frequency is samples per second, this needs to be in samples, second*freq=samples)
%     
%         % Add to trl matrix
%         trl = [trl; start_sample, end_sample,task_epoch_seg1_baseline*1000];
%     end
%     disp('Defined trials')
%     cfg = [];
%     cfg.trl = trl;
%     cfg.trl = round(cfg.trl);
%     data_task_seg_1 = ft_redefinetrial(cfg, data_task_cont); % Extract trials
%     disp('defined data_task_seg_1')

    % 3b. Insert check if more than 120 trials are segmented due to extra
    % triggers (can sometimes happen mistakenly); none present in this
    % dataset
    %[num_trls,~] = size(cfg.trl);
    
    %~~ changing this given that we create num_trials above
    if num_trials > total_trials % 120 mini blocks in 
        disp('***EXTRA TRIAL IN cfg.trl! Please adjust cfg.trl***')
        keyboard; %waits for keyboard/gui input - can therefore remove the extra trial before continuing the script
    end
    disp('trials do not have any extras')
    
    %~~ the following section will have to be reworked as it references the original
    %attempt at segmenting; also, for simplicity, you should just pull all 
    %the relevant info from the behavioral sheet (subj_behavioral) rather
    %than from the EGI event markers, especially as the importance of some 
    %variables created below is unclear (cel, eval)
    %you also didn't extract enough condition info for our analyses - see
    %my section above for ones that we are likely to need
    
%     % 3c. Extract condition names from cfg.event
%     %Use 'TRSP' event written at the end of each trial by EGI (containing all
%     %trial info) to extract: task type, response key, accuracy, RT;
%     
%     [event]=ft_read_event(subject_task_input);
%     condition_code = [];
%     rsp = {event.mffkey_rsp};
%     rsp = rsp(6:end);
%     cel = {event.mffkey_cel};
%     cel = cel(6:end);
%     %~~ note that 'eval' is a function name so you should avoid naming
%     %variables with the same name (it will confuse matlab)
%     eval = {event.mffkey_eval};
%     eval = eval(6:end);
%     rtim = {event.mffkey_rtim};
%     rtim = rtim(6:end);
%     pr2_times_EitherNeither = [];
%     pr1_times_EitherNeither = [];
%     sensory_rules = subj_behavioral.EncScreen1;
%     motor_rules = subj_behavioral.EncScreen2;
%     logic_rules = subj_behavioral.EncScreen3;
%     
%     for i = 1:length(times)%find the when in seconds each pr1 and pr2 event happens within the session
%         if strcmp(types{1, i}, 'pr1+')
%             pr1_times_EitherNeither = [pr1_times_EitherNeither, times(1, i)];
%             pr2_times_EitherNeither = [pr2_times_EitherNeither, 0];
%         elseif strcmp(types{1, i}, 'pr2+')
%             pr2_times_EitherNeither = [pr2_times_EitherNeither, times(1, i)];
%             pr1_times_EitherNeither = [pr1_times_EitherNeither, 0];
%         else
%             pr1_times_EitherNeither = [pr1_times_EitherNeither, 0];
%             pr2_times_EitherNeither = [pr2_times_EitherNeither, 0];
%         end
%     end
%     %Subtract the values and create a list the same length as rsp and rtim
%     time_btwn_pr1_pr2 = [];
%     for i=1:length(pr2_times_EitherNeither)
%         if pr2_times_EitherNeither(i)~=0
%             k = (pr2_times_EitherNeither(i) - pr1_times_EitherNeither(i-1)); %find difference
%             k = k*1000; %convert to ms
%             time_btwn_pr1_pr2 = [time_btwn_pr1_pr2; k];
%         end
%     end
%     index = 0;
%     index_stim = 0;
%     solved_pr1 = 0;
%     solved_pr2 = 0;
%     for trls = 1:size(types, 2)
%         % Ensure types{trls} is not a cell and extract its contents
%         if iscell(types{trls})
%             type_value = types{trls}{1}; % Extract string from nested cell
%         else
%             type_value = types{trls}; % Use directly if already string/char
%         end
%         
%         % Check if it matches 'TRSP'
%         if strcmp(type_value, 'TRSP')
%             trl_info = []; %add in novel vs practiced task, switching conditions (blockCond), taskCode, visFileProbe1 and 2, audfileProb1 and 2
%             trl_task = str2num(cel{trls});
%             trl_resp = str2num(rsp{trls});
%             trl_acc = str2num(eval{trls});
%             trl_RT = str2num(rtim{trls});
%             %~~ this is an imprecise way to code whether a task could be
%             %responded to during the probe1 events; 
%             %if the reaction time in choosing an option is less than the time
%             %between trials, then the logic rule must have been either or neither.
%             %Use indexing to compare reaction time (rtim) and the time between
%             %trials.
%             index = index +1;
%             if trl_RT>time_btwn_pr1_pr2(index)
%                 trl_rule = 0; %Participants could not complete the task before the second stimulus was shown
%                 solved_pr2 = solved_pr2+1;
%             else
%                 trl_rule = 1; %participants completed the task before the second stimulus was shown, the rule was either or neither
%                 solved_pr1 = solved_pr1+1;
%             end
%             index_stim=index_stim+1;
%             s = sensory_rules(index_stim,1);
%             m = motor_rules(index_stim,1);
%             l = logic_rules(index_stim,1);
%             trl_stim = [s,m,l];
% 
%             trl_info = [trl_task, trl_stim, trl_rule, trl_resp, trl_acc, trl_RT];
%             condition_code = [condition_code; trl_info];
%         end
%     end
%     disp('made list of condition codes')
%     fprintf('The percentage of trials completed before the second stimulus is presented is %.2f\n', solved_pr1 / (solved_pr2 + solved_pr1));

    % 3c. Segment + preproc the filtered continuous rest data -- NOT
    % CURRENTLY USING REST DATA
    %Assign segmentation to configuration 
    
    % %REST -- ADD IN LATER
    % cfg = [];
    % cfg.dataset = [subject_rest_input]; 
    % %segment into stim epochs
    % %cfg.trialdef.eventtype = '?'; %used to figure out event types
    % cfg.trialdef.eventtype = 'Event_ECI TC-PIP 55513'; %same as Events_ .xml file
    % cfg.trialdef.prestim = rest_epoch_prestim; %1s stim, 1-1.5-2s jitter (i.e. min 1s) = total 2s
    % cfg.trialdef.poststim = rest_epoch_poststim; 
    % cfg = ft_definetrial(cfg); %outputs cfg.trl containing beginning, end and offset of each trial in datasamples

    % %update cfg.trl to create 30 trials * 20s epochs
    % trl=cfg.trl;
    % start_samp=trl(1);
    % num_trls=(rest_epoch_poststim*1000)/rest_epoch_length;
    % trl_final=[];
    % for t=1:num_trls
    %     samps=[start_samp,start_samp+rest_epoch_length-1,0];
    %     trl_final=[trl_final;samps];
    %     start_samp=start_samp+rest_epoch_length;
    % end
    % 
    % % 3d. Segment + preproc the filtered continuous data
    % cfg = [];
    % cfg.trl = trl_final;
    % data_rest_seg = ft_redefinetrial(cfg,data_rest_cont);
    %data_seg_raw = data_seg; %need to save this raw version for later ERP comparisons

    
   %% 4. Perform ICA 
    %Using runica extended
    tic
    disp('starting ICA -- runica extended')
    cfg = [];
    cfg.method = 'runica';
    cfg.runica.extended = 1;
    %~~ removing reference channel is optional, but looking at my notes
    %from the EGI project it seems like pmitting this step leads to rank
    %deficiency in the ICA decomp; so we should include this
    num_chans = size(data_task_seg_1.label,1);
    cfg.channel = data_task_seg_1.label(1:num_chans-1); %This removes the last channel (Cz)
    
    %~~ this pca reduction speeds up the runtime, but can worsen the
    %quality of the ICA decomposition; did you estimate how long it takes 
    %without this included? Ideally we would avoid this step (as in the
    %sample script I sent you) 
    cfg.runica.pca = 150;%reduces number of specified PCs
    [comp] = ft_componentanalysis(cfg,data_task_seg_1);
    disp('got through ft_component analysis, attempting save output')
    save(ICA_output, 'comp','-v7.3');
    disp('saved output')
    toc
    fprintf('Running runica took %.2f seconds\n', toc);
    %~~ another input() command that should be removed; use breakpoints
    %instead
    print_value = input('continue?: Y=1 N=0');
    
    %~~ the ICA code is not setup to run separately for running the ica 
    %(automated; allows for parallelizing) and visualizing the ica and removing
    %components(manual; cannot be readily parallelized), as in my sample script 
    %via 'run_ica_inspect' parameter;
    %you should add this after making the other changes to your code
    
   %% 5a. Apply Automated ICA procedures - icablinkmetrics -- CHANGE TO FIT WITH FT ICA -- add tic and toc with this 
    disp('applying ICA procedures')
    tic
    if any(ismember(veog_chans,sub_bad_channels))==1
        veog_chans = [22 9];
        disp('Error! The vertical eye channel was remove during noisy channel rejection. Replace with alternate channels.');
    end
    if any(ismember(heog_chans,sub_bad_channels))==1
        heog_chans = [33 122];
        disp('Error! The vertical eye channel was remove during noisy channel rejection. Replace with alternate channels.');
    end
    left_index = find(strcmp(data_task_seg_1.label, ['E',num2str(heog_chans(1,1))]));
    right_index = find(strcmp(data_task_seg_1.label, ['E',num2str(heog_chans(1,2))]));
    low_index =find(strcmp(data_task_seg_1.label,['E',num2str(veog_chans(1,1))]));
    high_index =find(strcmp(data_task_seg_1.label,['E',num2str(veog_chans(1,2))]));
    % %Pull out IC timecourses (comp.trial), concatenate into one long
    % %comp x samples array; do the same for the veog and heog channels
    curr_trials = size(comp.trial,2);
    ic_timecourses = [];
    veog_tcs = [];
    heog_tcs = [];
    for tt=1:curr_trials
        ic_timecourses = [ic_timecourses,comp.trial{1,tt}];%THIS IS FUNCTIONING WITHIN THE LOOP, MAY NEED TO CHANGE
        veog_tt = [data_task_seg_1.trial{1,tt}(low_index,:);data_task_seg_1.trial{1,tt}(high_index,:)];
        veog_tcs = [veog_tcs,veog_tt];
        heog_tt = [data_task_seg_1.trial{1,tt}(left_index,:);data_task_seg_1.trial{1,tt}(right_index,:)];
        heog_tcs = [heog_tcs,heog_tt];
    end
    
    %GO THROUGH THIS LATER AND CHECK IF IT IS NECESSARY -- it should be
    %create difference channels for veog and heog 
    veog_diff = veog_tcs(1,:)-veog_tcs(2,:);
    heog_diff = heog_tcs(1,:)-heog_tcs(2,:);

    %~~ this section has errors because it departed from my sample
    %code referencing the function 'Visualize_NoiseMeasure_ChannelTrial';
    %in general, you should only depart from the sample code if there is a
    %good reason to do so; as it stands, with removal of that function, 
    %we do not now have a way of visualizing the sorted distribution of
    %blink and eye movement ICs...
    %e.g. of errors: tempcorr_blink_r is compared to ica_tempcorr_thresh 
    %being converted to zscored absolute values
    
%Blink (veog) correlation
    [tempcorr_blink_r tempcorr_blink_p]  = corr(veog_diff',ic_timecourses');
    %~~ correcting error
    tempcorr_blink_r_z = zscore(abs(tempcorr_blink_r));
    disp('blink correction')

%reject based on zscore threshold and store in bad_
    bad_ICs_blinks_tempcorr = tempcorr_blink_r(tempcorr_blink_r(:,1)>ica_tempcorr_thresh,2);
    disp('reject eye blinks based on zscore')
    
%Repeat for eye movements
%Eye movement (heog) correlation
    [tempcorr_eyemov_r tempcorr_eyemov_p] = corr(heog_diff',ic_timecourses');
    tempcorr_eyemov_r_z = zscore(abs(tempcorr_eyemov_r));

    %~~ no justification is provided for this step; pasting in the
    %justification from my sample code
    %Remove correlation with eyeblink ICs - replace with Nan. Blink ICs will also
            %correlate with heog, but are more likely to be blinks if already
            %correlated with veog...
    tempcorr_eyemov_r_z(bad_ICs_blinks_tempcorr)=NaN;
    
    %~~ pasting this from where it was mistakenly included below; the idea
    %is that we first identify the automated artifact ICs for both blinks
    %and eye movements before doing a final visual/manual check
    bad_ICs_eyemov_tempcorr = tempcorr_eyemov_r_z(tempcorr_eyemov_r(:,1)>ica_tempcorr_thresh,2);
    
    disp('identified bad eye movements ?')
    
    %~~ elecfile is already set at the start of the script
    %elecfile = '/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/template/electrode/GSN-HydroCel-257.sfp';
    
    %~~ pasting this from below as well ; the idea is that *tempcorr stores
    %the artifacts identified by the automated approach only, whereas
    %*tempcorrvis stores the final artifacts after visually inspecting the
    %automated results
    bad_ICs_tempcorr = [bad_ICs_blinks_tempcorr',bad_ICs_eyemov_tempcorr'];
    bad_ICs_tempcorrvis = [bad_ICs_blinks_tempcorr',bad_ICs_eyemov_tempcorr'];
    
    run_ica_semiauto = 1;
%VISUALIZE -- add topoplot or smth and databrowser here 
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
%reject based on zscore (or r) threshold
    %bad_ICs_eyemov_tempcorr = tempcorr_eyemov_r_z(tempcorr_eyemov_r(:,1)>ica_tempcorr_thresh,2);
    
    %~~ note you haven't rejected them yet, merely identified them
    disp('rejected bad eye movements')

%% REJECT ICs
    %collapse ICs
    %bad_ICs_tempcorr = [bad_ICs_blinks_tempcorr',bad_ICs_eyemov_tempcorr'];
    %bad_ICs_tempcorrvis = [bad_ICs_blinks_tempcorr',bad_ICs_eyemov_tempcorr'];
    %disp('collapsed the independent components')
    
%Reject auto+vis - tempcorrvis
    cfg = [];
    cfg.component = bad_ICs_tempcorrvis; %enter artifact components to be removed
    data_seg_channels_trials_autovisIC = ft_rejectcomponent(cfg,comp,data_task_seg_1);
    toc
    fprintf('rejecting the components  took %.2f seconds\n', toc);
    
    %~~ another input() value, although here it shouldn't impact running of
    %the code once you set up the run_ica_inspect parameter because we'll
    %be in the non-parallelized/manual part of the code
    print_value = input('continue?: Y=1 N=0');
    
    disp('rejected the components')
    %clear data_seg_final;
   %% 6. lowpass filter (100Hz)
    cfg = [];
    cfg.lpfilter = 'yes';
    cfg.lpfreq = filter_lp;
    cfg.lpfilttype='firws';
    cfg.lpfiltdir='onepass-minphase';
    %data_seg_channels_trials_autovisIC = ft_preprocessing(cfg,data_task_seg_channels_trials_autovisIC);
    data_task_seg_1 = ft_preprocessing(cfg,data_seg_channels_trials_autovisIC);
    disp('lowpass filter')
    
    %% 7. Throw out bad trials -- currently skipping 
    %Temporary 
    %[sub_bad_trials,data_task_seg_1] =  identify_bad_channels_or_trials(data_task_seg_1, "Trial",3, 120,'task',1);
    %disp('threw out the bad trials')
    % %USE THIS LATER WHEN REST DATA IS BEING USED AS WELL
    % tt_types = {'task','rest'};
    % for i =1:length(tt_types)
    %     tt_t = tt_types(i);
    %     eval(['sub_bad_trials,data_',tt_t,'_seg =  identify_bad_channels_or_trials(data_',tt_t,'_seg_1, "Trial",meanabs_trial_thresh, 0,',tt_t,',',i,');']);
    % end
    
    %% 8. Segmentation 2
    
    %~~ again, the following section needed to be updated to reflect the
    %correct segmentation approach
    %you basically loop through trial_defs and pass the appropriate trial
    %definition structures to ft_redefinetrial
    %*we should debug this once you've made the other corrections to make
    %sure it works as intended
    
    data_preproc_out=struct;
    for tt=1:length(trial_cond_names)
        tt_out=trial_cond_names{tt};
        tt_trial=trial_defs.(tt_out);
        
        %segment data based on tone pairs 
        cfg = [];
        cfg.trl = tt_trial.trl;                    
        data_out= ft_redefinetrial(cfg,data_task_seg_1); %stored segmented data
        
        %finish up remaining preproc within looping structure for
        %efficiency
        
        %baseline correct
        cfg = [];
        cfg.demean = 'yes';
        
        %note - baseline is currently the same for all events, but for clarity here
        %separating between encoding and probe events (will help if we ever
        %have to modify segmenting of events later)
        if tt<5
            %for enc1-3, and enc3_ITI
            cfg.baselinewindow = [instruction_epoch_baseline 0];
        else
            %for probes
            cfg.baselinewindow = [probe1_epoch_baseline 0];
        end
        
        %reref to common avg
        cfg.reref = 'yes';
        cfg.refchannel = 'all';
        
        %apply preproc
        data_out = ft_preprocessing(cfg,data_out);
        
        %store in struct
        data_preproc_out.(tt_out)=data_out;
    
        
    end
    
    
%     %Find time points of enc1 and enc3, enc3, and pr1+ and pr2+
%     enc_samples = [];
%     % enc2_samples = [];
%     enc3_samples = [];
%     pr1_samples = [];
%     %pr1 = 0;
%     pr2_samples = [];
%     %pr2 = 0;
% 
%     for i = 1:length(times)
%         if strcmp(types{1, i}, 'enc1')  
%             enc_samples = [enc_samples, event_start_samples(1,i)];
%         elseif strcmp(types{1,i}, 'enc2')
%             enc_samples = [enc_samples, event_start_samples(1,i)];
%         elseif strcmp(types{1,i}, 'enc3')
%             enc_samples = [enc_samples, event_start_samples(1,i)];
%             enc3_samples = [enc3_samples, event_start_samples(1,i)];
%         elseif strcmp(types{1, i}, 'pr1+')  
%             % CHANGE FROM 1 RESPONSE SEGMENT WITH THREE TRIALS...
%             % pr1 = pr1 + 1;
%             % if pr1 == 1
%             %     pr1_samples = [pr1_samples, event_start_samples(1,i)];
%             % elseif pr1==3
%             %     pr1=0;
%             % end
%             pr1_samples = [pr1_samples, event_start_samples(1,i)];
%         elseif strcmp(types{1, i}, 'pr2+')  
%             % CHANGE FROM 1 RESPONSE SEGMENT WITH THREE TRIALS...
%             % pr2 = pr2 + 1;
%             % if pr2 == 3
%             %     pr2_samples = [pr2_samples, event_start_samples(1,i)];
%             %     pr2 = 0; 
%             % end
%             % TO # RESPONSE SEGMENTS WITH ONE TRIAL EACH
%             pr2_samples = [pr2_samples, event_start_samples(1,i)];
%         end
%     end
%     %Define trials around enc1 and pr2+
%     % Initialize trial matrix 1
%     trl1 = [];
% 
%     % Define segments based on enc1 and enc3 SAMPLE INDICES
%     for i = 1:length(enc_samples)
%         %Define sample points of start and stop events
%         enc_sample = enc_samples(1,i);
%         % enc_sample_end = enc3_samples(1,i);
% 
%         % Define segment
%         start_sample = enc_sample - instruction_epoch_pre*1000; % 90 ms before enc1
%         end_sample = enc_sample + instruction_epoch_post*1000; % 900 ms after each encoding event (frequency is samples per second, this needs to be in samples, second*freq=samples)
%     
%         % Add to trl matrix
%         trl1 = [trl1; start_sample, end_sample, instruction_epoch_baseline*1000];
%     end
%     % Define waiting segment 
%     trl2 = [];
%     % Define segments based on enc3 SAMPLE INDICES
%     for i = 1:length(enc3_samples)
%         %Define event starting point
%         enc3_sample = enc3_samples(1,i);
% 
%         % Define segment
%         start_sample = enc3_sample; % 100 ms before enc1
%         end_sample = enc3_sample + instruction_to_probe_offset*1000; % 1520 ms after last pr2+ (frequency is samples per second, this needs to be in samples, second*freq=samples)
%     
%         % Add to trl matrix
%         trl2 = [trl2; start_sample, end_sample, instruction_to_probe_baseline];
%     end
% 
%     %Define response segment (trl3)
%     trl3=[];
% 
%     % Define segments based on pr2 and pr1 SAMPLE INDICES
%     for i = 1:length(enc1_samples)
%         %Define sample points of start and stop events
%         pr1_sample = pr1_samples(1,i);
%         pr2_sample = pr2_samples(1,i);
% 
%         % Define segment
%         start_sample = pr1_sample - probe1_epoch_pre*1000; % 90 ms before pr1+
%         end_sample = pr2_sample + probe2_epoch_post*1000; % 1520 ms after last pr2+ (frequency is samples per second, this needs to be in samples, second*freq=samples)
%     
%         % Add to trl matrix
%         trl3 = [trl3; start_sample, end_sample,probe1_epoch_baseline*1000];
%     end
% 
%     %Assign segmentation to configuration 
%     %PAST ERROR MESSAGE
%     %%Error using ft_redefinetrial
%     %%%you should specify only one of the options for redefining the data segments
%     %ERROR MEANING: trl has doubles and should only have integers 
%     cfg_1 = []; 
%     cfg_1.trl = trl1;
%     cfg_1.trl = round(cfg_1.trl);
%     data_task_seg_encode = ft_redefinetrial(cfg_1, data_task_seg_1); % Extract trials
% 
%     cfg_2 = [];
%     cfg_2.trl = trl2;
%     cfg_2.trl = round(cfg_2.trl);
%     data_task_seg_ITI = ft_redefinetrial(cfg_2, data_task_seg_1); % Extract trials
% 
%     cfg_3 = [];
%     cfg_3.trl = trl3;
%     cfg_3.trl = round(cfg_3.trl);
%     data_task_seg_trial  = ft_redefinetrial(cfg_3, data_task_seg_1); % Extract trials
%     
%     disp('got through the second segmentation')

%    %% 9. Baseline correct + rereference to common average
%     %apply to all data_segs i.e. _raw, raw_channels_trials,
%     %raw_channels_trials_autoIC, raw_channels_trials_autovisIC, 
%             
%     %baseline correct
%     cfg = [];
%     cfg.demean = 'yes';
%     cfg.baselinewindow = [probe1_epoch_baseline 0];
%     
%     % %USE AFTER REST IS ADDED BACK IN
%     % if ttt==1
%     %     %task
%     %     cfg.baselinewindow = [task_epoch_baseline 0];
%     % elseif ttt==2
%     %     %rest
%     %     cfg.baselinewindow = 'all';
%     % end
% 
%     %reref to common avg
%     cfg.reref = 'yes';
%     cfg.refchannel = 'all';
% 
%     %data_seg_raw = ft_preprocessing(cfg,data_seg_raw);
%     %data_seg_channels_trials = ft_preprocessing(cfg,data_seg_channels_trials);
%     %data_seg_channels_trials_autoIC = ft_preprocessing(cfg,data_seg_channels_trials_autoIC);
%     data_task_seg_trial = ft_preprocessing(cfg,data_task_seg_trial);
%     disp('got through the repsonse segmentation rereference')
%     %REPEAT FOR INSTRUCTION/ENCODE PERIOD
%     %baseline correct
%     cfg = [];
%     cfg.demean = 'yes';
%     cfg.baselinewindow = [instruction_epoch_baseline 0];
%     
%     % %USE AFTER REST IS ADDED BACK IN
%     % if ttt==1
%     %     %task
%     %     cfg.baselinewindow = [task_epoch_baseline 0];
%     % elseif ttt==2
%     %     %rest
%     %     cfg.baselinewindow = 'all';
%     % end
% 
%     %reref to common avg
%     cfg.reref = 'yes';
%     cfg.refchannel = 'all';
%     %ADD BACK IN WITH REST DATA
%     % %*assign to task/rest vars
%     % sub_bad_ICs{ttt}=bad_ICs_tempcorrvis;
%     % eval([tt_t,'_data_seg_channels_trials_autovisIC=data_seg_channels_trials_autovisIC;']) 
%     data_task_seg_encode = ft_preprocessing(cfg,data_task_seg_encode); 
%     disp('got through the encode segmentation rereference')
% 
%     %REPEAT FOR WAIT PERIOD
%     %baseline correct -- WORKS
%     cfg = [];
%     cfg.demean = 'yes';
%     cfg.baselinewindow = [instruction_to_probe_baseline 0];
%     
%     % %USE AFTER REST IS ADDED BACK IN
%     % if ttt==1
%     %     %task
%     %     cfg.baselinewindow = [task_epoch_baseline 0];
%     % elseif ttt==2
%     %     %rest
%     %     cfg.baselinewindow = 'all';
%     % end
% 
%     %reref to common avg
%     cfg.reref = 'yes';
%     cfg.refchannel = 'all';
%     %ADD BACK IN WITH REST DATA
%     % %*assign to task/rest vars
%     % sub_bad_ICs{ttt}=bad_ICs_tempcorrvis;
%     % eval([tt_t,'_data_seg_channels_trials_autovisIC=data_seg_channels_trials_autovisIC;']);
%     data_task_seg_ITI = ft_preprocessing(cfg,data_task_seg_ITI);
%     disp('got through the wait segmentation rereference')

    %% 10. Save final outputs (from various approaches i.e. automated procs, and visual + automated procs)
    %ERROR
%     Error using save
%       Unable to write file
%       /projects/f_mc1689_1/cpro2_eeg/data/results/Analysis3_causalFilter/1_preproc_hp0.1notch_seg_autochannelz3_trialz3_ICAz3_baselinelp125avgref.mat:
%       Disk quota exceeded.
 
    %~~ updating what is saved based on latest corrected version of the script
    %save(outputfile,'sub_bad_ICs','sub_bad_channels','sub_bad_trials','data_task_seg_trial','data_task_seg_encode','data_task_seg_ITI','-v7.3');   
    save(outputfile,'sub_bad_channels','bad_ICs_tempcorr','bad_ICs_tempcorrvis','data_preproc_out','-v7.3');
    disp('saved data')
     
end

end

