function Step1_preproc_EEG(subjects, run_ica_inspect)

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
%10. Save outputs from various approaches 


%INPUTS
%subjects=numeric nx1 array, each row containing subject IDs 
%run_ica_inspect=numeric, determines which version of the function
    %is run; 1 (step1): preprocessing is run till ICA estimation and saved (run
    %using slurm batch), 2 (step2): loads in ICA results and allows visual
    %inspection of potential artifacts followed by IC rejection, lp
    %filtering etc

%OUTPUTS
%Writes results of steps leading up to and including ICA (run_ica_inspect=1), followed
%by final preproc results (after visually screening ICA results; lp filtering, basebinicalining and common avg
%rereferencing (run_ica_inspect==2)

%% Set path and defaults for Fieldtrip
addpath /projectsn/f_mc1689_1/cpro2_eeg/docs/toolboxes/fieldtrip-20240704;
addpath /projectsn/f_mc1689_1/baby_actflow/docs/toolboxes/eeglab13_6_5b/;
ft_defaults;
addpath /projectsn/f_mc1689_1/CPRO2_learning/data/rawdata/;

%path to egi electrode layout (in Fieldtrip templates directory)
elecfile = '/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/template/electrode/GSN-HydroCel-257.sfp';

%% Set inputs, parms etc

%set path to inputs
%Behavioral data
path_to_behavior = '/projectsn/f_mc1689_1/cpro2_eeg/data/rawdata/CPRO2_fMRI_EEG_main.xlsx';
behavior = readtable(path_to_behavior);

%Path to EEG data
baseDirOut = '/projects/f_mc1689_1/cpro2_eeg';
baseDirRaw='/projects/f_mc1689_1/CPRO2_learning/';
input_dir = [baseDirRaw,'data/rawdata/'];%participant/EEG/data_folder.mff


%**set filter pass bands; note these are sequentially applied as this i) is
%advocated for ICA (from EEGLAB list), ii) yielded less delay relative to
%the raw/butterworth filtered signal than bandpass filtering (see Step0
%script)
filter_hp=.1;
filter_lp=125;

%**set epoch length (in secs) for rest and task
%rest; rest is baselined based on pseudotrial mean 
%rest_epoch_prestim=0; %in s; i.e. pull out 0-600s continuous EEG data around 'start recording' marker sent by eprime to EGI
%rest_epoch_poststim=600; %in s
%rest_epoch_length=20000; %in samples, used to segment 600s rest segment into 20s pseudotrials


%task segmentation 1
%~~Entire task block is -90ms from enc1 through 1520ms after probe 2 on
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

%number of tasks and trials
total_tasks = 120; %15*4 practice tasks and 60 novel mini blocks
total_trials = 360;

%**name EOG electrode indices used for automated ICA eye artifact removal
%(i.e. eye signal timecourse that ICs are correlated with)
%Available EOG sensors: 
%Left VEOG = 241 (low i.e. below eye), 46 (high); Right VEOG =
%238 (l), 10 (r) - used for blinks
%More inferior HEOG = 234 (left), 244(right); More superior HEOG = 54 (left), 1 (right)
%Use right VEOG electrodes for blink, and more superior HEOG electrodes for
%eye movements 
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

%set variable to determine if you need to make graphs comparing the middle 
% trial of seg1 and seg2 (probes). Set equal to 1 to compare, set equal to
% 0 to skip graphing. 
compare_seg1_seg2 = 1;

%Set output directory
analysisName= ['/preproc1_causalFilter/'];
outputDir = [baseDirOut,'/data/results/',analysisName];
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
%disp(['number of subjects = ',num2str(length(subjects))])
for x = 1:length(subjects)
    %select subject
    subject_num = subjects(1,x);
    disp(subject_num);
    subj_behavioral = behavior(behavior.Subject == subject_num, :);
    disp('past subj_behavioral')
    
    %name output files 
    timingfile = [outputDir, 'sub_',num2str(subjects(1,x)),'timing_info.mat'];
    %timingfile_NO_PCA = [outputDir, 'sub_',num2str(subjects(1,x)),'timing_info_PCA.mat'];

    outputfile = [outputDir,'sub_',num2str(subjects(1,x)),'_preproc_hp',num2str(filter_hp),'notch_seg_autochannelz',num2str(meanabs_channel_thresh),'_trialz',num2str(meanabs_trial_thresh),'_ICAz',num2str(ica_tempcorr_thresh),'_baselinelp',num2str(filter_lp),'avgref.mat'];
    %outputfile_NO_PCA = [outputDir,'sub_',num2str(subjects(1,x)),'_preproc_hp',num2str(filter_hp),'notch_seg_autochannelz',num2str(meanabs_channel_thresh),'_trialz',num2str(meanabs_trial_thresh),'_ICAz',num2str(ica_tempcorr_thresh),'_baselinelp',num2str(filter_lp),'avgref.mat'];

    ICA_output = [outputDir,'sub_',num2str(subjects(1,x)),'_segICAcomps_ica_extended_channeltrialthresh',num2str(meanabs_channel_thresh),'.mat'];
    %ICA_output_NO_PCA = [outputDir,'sub_',num2str(subjects(1,x)),'_segICAcomps_ica_extended_PCA_channeltrialthresh',num2str(meanabs_channel_thresh),'.mat'];
    
    %init variables storing bad channs, trials, ICs
    sub_bad_channels={};
    sub_bad_trials={};
    sub_bad_ICs={};    
    
    %specify subject input path
    %subject_rest_input = [input_dir,num2str(subject_num),'/EEG/',num2str(subject_num),'_rest_pretask.mff'];
    subject_task_input_p1 = [input_dir,num2str(subject_num),'/EEG/'];
    subject_task_input_path = dir(fullfile(subject_task_input_p1, 'CPRO2_Task_*'));
    disp(subject_task_input_path)
    subject_task_input = fullfile(subject_task_input_path(1).folder, subject_task_input_path(1).name);
    disp(subject_task_input)
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
    timing_info = table({'initial_preproc'}, toc, 'VariableNames', {'Section', 'seconds'});
    disp('made it past ft_preprocessing task')
    %%REST -- not doing yet
    %cfg.dataset = [subject_rest_input];
    %data_rest_cont = ft_preprocessing(cfg);
  
  %% 2. Identify and Remove Noisy Channels
    %Automated procedure - for each channel, takes the absolute average
    %across all samles in the continuous data; threshold based on zscores
    %TEMPORARY
    [sub_bad_channels,data_task_cont] =  identify_bad_channels_or_trials(subject_num, data_task_cont, "Channel",meanabs_channel_thresh, 0,'task',1, run_semiauto);
    disp('made it past identifying and removing noisy channels')
    % enter in later
    % tt_types = {'task','rest'};
    % for i =1:length(tt_types)
    %     tt_t = tt_types(i);
    %     eval(['sub_bad_channels,data_',tt_t,'_cont =  identify_bad_channels_or_trials(', subject_num, ', data_',tt_t,'_cont, "Channel",meanabs_channel_thresh, 0,',tt_t,',',i, run_semiauto,');']);
    % end
  %% 3. Segment into trials/epochs
    
    %TASK
    %define trials 
    %Find events
    
    %Segments events from enc1 to the last pr2 using redefinetrials. First,
    %segment each individual event, then redefine the trial by combining
    %the segments/events from enc1 to the last pr2.
    
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
        disp(tt_event);
        
        
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
        
        % Make a table of trial conditions/rules corresponding to each 
        % trial added to cfg_trial
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
            
        else
            %enc1-3, enc3_ITI
            num_trials=size(subj_behavioral,1);
            vec_ref=3:3:num_trials;
            disp(vec_ref)
            %subsample from 360 trials to 120 blocks
            trl_info=trl_info(vec_ref,:);
            disp(size(trl_info))
        end
        disp(tt_event);
        disp('trl_info');
        disp(size(trl_info));
        disp('cfg_trial.trl');
        disp(size(cfg_trial.trl));
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
    disp('breakpoint')

    
   %% 4. Perform ICA 
    %Using runica extended
    if exist(ICA_output)
        load(ICA_output);
    else
        tic
        disp('starting ICA -- runica extended')
        cfg = [];
        cfg.method = 'runica';
        cfg.runica.extended = 1;
        % removing reference to prevent rank deficiency in the ICA decomp
        num_chans = size(data_task_seg_1.label,1);
        cfg.channel = data_task_seg_1.label(1:num_chans-1); %This removes the last channel (Cz)

        %~~ this pca reduction speeds up the runtime, but can worsen the
        %quality of the ICA decomposition; did you estimate how long it takes 
        %without this included? Ideally we would avoid this step (as in the
        %sample script I sent you) -- NO PCA
        %cfg.runica.pca = 150;%reduces number of specified PCs
        [comp] = ft_componentanalysis(cfg,data_task_seg_1);
        disp('got through ft_component analysis, attempting save output')
        save(ICA_output, 'comp','-v7.3');
        disp('saved output')
        toc
        fprintf('Running runica no .pca = 150 took %.2f seconds\n', toc);
    end
    
    timing_info = [timing_info; {'ica_runica_pca',toc}];
    disp('breakpoint')
    %save(timingfile, 'timing_info')
    
   %% 5a. Apply Automated ICA procedures - icablinkmetrics -- CHANGE TO FIT WITH FT ICA -- add tic and toc with this 
    
    if run_ica_inspect ==2
        disp('applying ICA procedures')
        tic
        if any(ismember(veog_chans,sub_bad_channels))==1
            veog_chans = [241, 46];
            disp('Warning! The vertical eye channel was removed during noisy channel rejection. Have replaced with alternate channels.');
            if any(ismember(veog_chans,sub_bad_channels))==1
                disp('ERROR! All stated vertical eye channels removed during noisy channel rejection. Find alternatives manually?');
                keyboard;
            end               
        end
        if any(ismember(heog_chans,sub_bad_channels))==1
            heog_chans = [234, 244];
            disp('Error! The horizontal eye channel was removed during noisy channel rejection. Have replaced with alternate channels.');
            if any(ismember(heog_chans,sub_bad_channels))==1
                disp('ERROR! All stated horizontal eye channels removed during noisy channel rejection. Find alternatives manually?');
                keyboard;
            end
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
            ic_timecourses = [ic_timecourses,comp.trial{1,tt}];
            veog_tt = [data_task_seg_1.trial{1,tt}(low_index,:);data_task_seg_1.trial{1,tt}(high_index,:)];
            veog_tcs = [veog_tcs,veog_tt];
            heog_tt = [data_task_seg_1.trial{1,tt}(left_index,:);data_task_seg_1.trial{1,tt}(right_index,:)];
            heog_tcs = [heog_tcs,heog_tt];
        end
        
        %create difference channels for veog and heog 
        veog_diff = veog_tcs(1,:)-veog_tcs(2,:);
        heog_diff = heog_tcs(1,:)-heog_tcs(2,:);
    
        
    %Blink (veog) correlation
        [tempcorr_blink_r tempcorr_blink_p]  = corr(veog_diff',ic_timecourses');
        %previously contained naming error
        %tempcorr_blink_r_z = zscore(abs(tempcorr_blink_r));
        %disp('blink correction')
    
    %reject based on zscore threshold and store in bad_
    %~~ERROR here: the non-zscored r values were passed below which doesn't
    %   work because ica_tempcorr_thresh = zscored 3 threshold; have
    %   corrected it now by passing the zscored variable
    
        [tempcorr_blink_sorted_z] = Visualize_NoiseMeasure_ChannelTrial(zscore(abs(tempcorr_blink_r))',50);

        %reject based on zscore threshold and store in bad_
        bad_ICs_blinks_tempcorr = tempcorr_blink_sorted_z(tempcorr_blink_sorted_z(:,1)>ica_tempcorr_thresh,2);


    %Repeat for eye movements
    %Eye movement (heog) correlation
        [tempcorr_eyemov_r tempcorr_eyemov_p] = corr(heog_diff',ic_timecourses');
        tempcorr_eyemov_r_z = zscore(abs(tempcorr_eyemov_r));
   
        %Remove correlation with eyeblink ICs - replace with Nan. Blink ICs will also
                %correlate with heog, but are more likely to be blinks if already
                %correlated with veog...
        tempcorr_eyemov_r_z(bad_ICs_blinks_tempcorr)=NaN;
        
        %~~ updating the eye movement IC section to match the changes above    
        % First identified the automated artifact ICs for both blinks
        %and eye movements before doing a final visual/manual check
        [tempcorr_eyemov_sorted_z] = Visualize_NoiseMeasure_ChannelTrial(tempcorr_eyemov_r_z',50);
        %reject based on zscore (or r) threshold
        bad_ICs_eyemov_tempcorr = tempcorr_eyemov_sorted_z(tempcorr_eyemov_sorted_z(:,1)>ica_tempcorr_thresh,2);

        disp('identified bad eye movements ?')
        
            % Tempcorr stores the artifacts identified by the automated approach
            %only, and tempcorrvis stores the final artifacts after visually inspecting the
            %automated results
        bad_ICs_tempcorr = [bad_ICs_blinks_tempcorr',bad_ICs_eyemov_tempcorr'];
        bad_ICs_tempcorrvis = [bad_ICs_blinks_tempcorr',bad_ICs_eyemov_tempcorr'];
        
    %VISUALIZE -- add topoplot or smth and databrowser here 
        if run_ica_semiauto == 1 %change to 2
            %Visualise topography of components - detect artifacts
            figure;
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
        disp('identified bad eye movements')
    
    %% REJECT ICs
        
    %Reject auto+vis - tempcorrvis
        cfg = [];
        cfg.component = bad_ICs_tempcorrvis; %enter artifact components to be removed
        data_seg_channels_trials_autovisIC = ft_rejectcomponent(cfg,comp,data_task_seg_1);
        toc
        fprintf('rejecting the components  took %.2f seconds\n', toc);
        
        disp('rejected the components')
        %clear data_seg_final;
       %% 6. lowpass filter (100Hz)
        cfg = [];
        cfg.lpfilter = 'yes';
        cfg.lpfreq = filter_lp;
        cfg.lpfilttype='firws';
        cfg.lpfiltdir='onepass-minphase';
        data_task_seg_1 = ft_preprocessing(cfg,data_seg_channels_trials_autovisIC);
        disp('lowpass filter')
        
        %% 7. Throw out bad trials -- currently skipping 
        %Temporary 
        %[sub_bad_trials,data_task_seg_1] =  identify_bad_channels_or_trials(subject_num, data_task_seg_1, "Trial",3, 120,'task',1);
        %disp('threw out the bad trials')
        % %USE THIS LATER WHEN REST DATA IS BEING USED AS WELL
        % tt_types = {'task','rest'};
        % for i =1:length(tt_types)
        %     tt_t = tt_types(i);
        %     eval(['sub_bad_trials,data_',tt_t,'_seg =  identify_bad_channels_or_trials('subject_num, ',data_',tt_t,'_seg_1, "Trial",meanabs_trial_thresh, 0,',tt_t,',',i,');']);
        % end
        
        %% 8. Segmentation 2
        
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
            
            if compare_seg1_seg2==1 && strcmp(tt_out, 'probes')
                list_trials_seg2 = data_out.trial;
    
                %size list trials will be 255x 120 or 360
                shape_trials = size(list_trials_seg2);
                length_t = shape_trials(2);
    
                trials_to_plot = [1, length_t/2, length_t];
    
                figure; movegui('center');
                
                for i = 1:length(trials_to_plot)
                    trial_idx = trials_to_plot(i);
                    data = list_trials_seg2{trial_idx};
                
                    if trial_idx==180  % <- temporarily ignore outlier check
                        subplot(1, 2, 1);
                        plot(data);
                        %colorbar;
                        %caxis([-100 100]);  % or adjust based on expected EEG signal
                        title(['Preprocessed Trial 180']);
                        xlabel('Time');
                        ylabel('Channel');
                    else
                        disp(['Skipping trial ' num2str(trial_idx) ' due to outlier channel(s).'])
                    end
                end
                list_trials_seg1 = data_task_seg_1.trial;
                subplot(1, 2, 2);
                plot(list_trials_seg1{60});%change to data_task_seg_1.trials{180}
                %colorbar;
                %caxis([-100 100]);  % or adjust based on expected EEG signal
                title(['Nonprocessed Trial 60']);
                xlabel('Time');
                ylabel('Channel');
                
                drawnow;
                            
                subjNum = subjects(x);
                save_path = fullfile('/cache/home/let83/FND4/results/graphs/', [tt_out '_subj' subjNum '_preproc.png']);
                saveas(gcf, save_path);
            end

        end
        disp('breakpoint')
        
        %% 10. Save final outputs (from various approaches i.e. automated procs, and visual + automated procs)
        
        save(outputfile,'sub_bad_channels','bad_ICs_tempcorr','bad_ICs_tempcorrvis','data_preproc_out','-v7.3');
        disp('saved data')
    end   

end

end
function is_outlier = is_trial_outlier(data)
    % Define threshold for outlier detection
    zscore_threshold = 3;
       
    % Outlier check: Z-score threshold
    zdata = zscore(data, 0, 2);  % z-score across time for each channel
    if any(any(abs(zdata) > zscore_threshold, 2))
        is_outlier = true;
        return;
    end
        
    is_outlier = false;
end


