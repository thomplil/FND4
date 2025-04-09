%Step1_preprocess.m

%Author: Laura Milovic
%Last update: Nov 14th 2024

%DESCRIPTION: ~*just updating the formatting of this intro section to fit
%with norms in the field, otherwise the details you included are very helpful ~*

%Baby_actflow eeg preprocessing of activity for complex tone data t70 & t300
%t70 has a 70 ms ISI. this acts as the rapid rate condition where a merged repsonse is seen for the tone pair. 
%t300 has a 300 ms ISI which acts as the control rate condition where there is a clear response to each tone of the tone pair. 

%this script takes the original eeg files from EGI Net Station and preprocesses the data using the Fieldtrip Toolbox 

%~*I made some final edits here
%Pipeline steps: 
    %1. bandstop filter (60  Hz)
    %2. to make 400 system data comparable to other systems: highpass
    %   filter (0.1Hz), lowpass filter (100Hz)
    %3. for 400 system only: downsample to 250Hz to match other systems
    %   *note original intention was to downsample to 200Hz (Nyquist limit of upper frequency), 
    %   but with the non-filter downsampling approach, the new rate has to be an integer division of
    %   originaly, which is not the case for downsampling 250 to 200...
    %4. segment - define epochs w/ padding (-100 to 800ms, from trial
    %onset; *which marker defines this?)
    %5. set z-score thresholds for semi-automatic artifact detection (3 for now) 
    %6. identify bad channels and reject 
    %7. identify noisy trials and reject 
    %8. *ICA - create virtual V/HEOG channels (skipped for now)
    %9. *ICA - run ICA, then semi-auto eye artifact detection, visually
    %   inspect before finally removing artifact components (skipped for
    %   now)
    %10. baseline correct and re-reference to common average 
    %11. save outputs 
    
%important differences for the subsets of the dataset with different
%recording equipment/parameters 

%200 SYSTEM
    % collected 0.1-100Hz
    % sampling rate 250 Hz
    % GSN Net template 
    % event markers 
        % DIN - tone 
        % Bgin-Bgin - tone pair 
        % P7s1/P3s1 - standards 
        % P7sd/P3sd - predeviant standard 
        % P7dv/P3dv - deviant 
%300 SYSTEM 
    % collected 0.1- 100Hz
    % sampling rate 250 Hz 
    % HCL net template 
    % event markers 
        % DIN - tone 
        % Bgin-Bgin - tone pair 
        % P7s1/P3s1 - standard 
        % P7sd/P3sd - pre-deviant standard 
        % P7dv/P3dv - deviant 
%400 SYSTEM
    % collected w/o online filtering 
    % sampling rate 1000 Hz 
    % HCL net template 
    % event markers 
        % DIN2 - tone 
        % Bgin-Bgin - tone pair 
        % P7s1/P3s1 - standard 
        % P7sd/P3sd - pre-devaint standard 
        % P7dv/P3dv - devaint 
        
% Data is comprised of active intervention participants (pre (4m) and post (7m)) as well as cross-sectional age matched controls (7m)
        
        
% The goal is to preprocess all data to match prior aquisitions 
    % the 400 data will need to be filterd and downsampled 
    % all data will be segmented by tone pairs using bgin markers or condition specific markers 
        % there are 833 trials per session 
   % standard tone pairs Ps1 (standard-standard) 
   %pre-deviant tone pairs Psd (standard-standard) standard trial before deviant trial 
   %deviant tone pairs Pdv (standard- deviant)
  
% further filtering of the data (beyond 0.1-100Hz) will be determined after visualization 
% bad sessions or bad subjects will be removed from analysis 
% using z-score threshold of 3 bad channels and bad trials within sessions will be rejected

%UPDATES:
%Nov 2024 - % working revisions on step 1 script based on discussion of parameters 

%% SETTING UP THE WORKSPACE ENVIRONMENT 

    
%path for firldtrip toolbox 
addpath '/projectsn/f_mc1689_1/baby_actflow/docs/toolboxes/fieldtrip-20240704/';
ft_defaults;

%set path for ICA. Need EEGLAB. Need to determine if ICA is essential for baby data considering blink differences 
%addpath /projectsn/f_mc1689_1/baby_actflow/docs/toolboxes/eeglab2024.0/;
%this version did not work downloaded older version 13_6_5b
%added path at ICA step 
%addpath '/projectsn/f_mc1689_1/baby_actflow/docs/toolboxes/eeglab13_6_5b/';

%set path for EGI electrode layouts
%there are 2 channel montgaes used in the study. Old data used GSN nets (200 system) and more recent subjects used Hydrocel nets 
elecfile1 = '/projects/f_mc1689_1/baby_actflow/data/rawdata/Baby_EGI_Montages/Old/GSN-OldNet.sfp';
elecfile2 = '/projects/f_mc1689_1/baby_actflow/data/rawdata/Baby_EGI_Montages/New/GSN-Hydrocel-129.sfp';

%paths for inputs & base directory:
baseDir='/projectsn/f_mc1689_1/baby_actflow/';
%input_dir = [baseDir,'/data/rawdata/unfiltered_raw/complete_dataset/'];
input_dir = [baseDir, '/data/rawdata/actflow_eeg_data/data/'];

%Set output directory
analysisName=['/causal_filter_resegment/'];
outputDir = [baseDir,'data/results/',analysisName];
if ~exist(outputDir,'dir')
    mkdir(outputDir);
end

%~*you should avoid adding paths for directories where you just need to load
%in files, as this can cause unanticipated conflicts; I've made changes
%below accordingly~
%setting path to excel file with subject/session data information
% addpath '/projectsn/f_mc1689_1/baby_actflow/data/rawdata/actflow_details/';
% subj_info= readtable('subject_session_info.xlsx');
subj_info_file=[baseDir,'data/rawdata/actflow_details/subject_session_information.xlsx'];
subj_info=readtable(subj_info_file);


%% SETTING PARAMETERS 

%not actually filtering here just setting parameter so do not need to classify yet to just 400 system data 
filter_hp=0.1;  %400 data 
if filter_hp==0.1 %~* a period in file names can generally cause issues
    hp_pref='0pt1';
end
filter_lp=100;  %400 data
downsamp=250;   %400 data ~*orginally intended 200 (nyquist of 100Hz upper freq; but with the current no-filter downsampling, this generates issues (downsample rate has to be an integer of the original one)

%can determine if other filtering is required. should filter a few ways initially to see what best preserves most of the data but is not too noisy 
%comment out which one is not being used  

%filter_hp=100;
%filter_hp=50;
%filter_hp=35;

%filter_lp=0.1;
%filter_lp=0.5;
%filter_lp=1.0;

%set epoch lengths
%larger pre and post segments help with padding so the filter does not introduce edge artifacts 
%can make larger if needed (0.5)
%0.2s pre stimulus (200 ms)
%0.2 post simulus  (200ms) 
%remember ITI is 705 ms for t70; ITI for t300 is 700ms 

task_epoch_prestim=0.1; %in s
task_epoch_poststim=0.8;
task_epoch_baseline=-0.1; %baseline will be -0.2 to 0

%segment by tone pair for a trial       %this ended up not being helpful because some sessions contain resting or both t70 and t300. segmenting based on condition will get correct number of trials 
tone_pair='bgin';

%set the number of trials 
%*some subjects have fewer trials, but num_trials isn't used in the code,
%just for reference
num_trials= 833; 
% 125 pre-deviant/ deviant pairs ; 583 standard/standard pairs (708 standards all together if including pre-deviant trials) 

%conditions: P*s* =standard ; P*sd =pre-dviant standard ; P*dv =deviant tone
condition_names = {'Standard';'PreDeviant_Standard';'Deviant'};

%set event marker for segmenting the tone pairs 
t70_standard_tone= 'P7s1';
t70_pre_deviant_tone='P7sd';
t70_deviant_tone='P7dv';

t300_standard_tone='P3s1';
t300_pre_deviant_tone='P3sd';
t300_deviant_tone='P3dv';

%setting eye channels 
%baby nets do not have eye channels. Channels on the forehead are used in place

%HYDROCEL
% 21,14              both upper for VEOG (left, right)
% 32,1               left, right for HEOG 

%GSN OLD NET 
% 22, 14             both upper for VEOG (left, right) 
% 33, 1              left, right for HEOG 


%classifying eye channel variables here for each net 

veog_chans_GSN= [22 14];    
heog_chans_GSN= [33 1];     

veog_chans_HCL= [21 14];  
heog_chans_HCL= [32 1];  


%setting z-score for artifact detection
% 3 is the standard z-score threshold that is typically used 
% may need to modify to best fit infant data 
%this will be used for detecting noisy channels/trials 

meanabs_channel_thresh = 3; 
meanabs_trial_thresh = 3; 
ica_thresh = 3;


%artifact detection should be run in semiautomatic mode initially 
%can visualize what it is identifying and how the paraments are working 
%later it may be useful to run automatically %set to 0 to run automaticcaly

run_semiauto = 1; %1=visually inspect outputs of bad channels and bad trials before rejecting. 0 = no visual 
run_ica_semiauto = 1; %1=visually inspect ocular ICs before rejecting. 0=no visual 


%% LOADING DATA INTO FIELDTRIP 

% Select the subject session - *note that subject_info is organized with
% each session dataset listed in the rows (multiple day sessions only for active subjects)
num_subjects= size(subj_info,1);            % want to access all of column 1. num subjects includes pre and post active seperate 
sessions=subj_info.SUBJ;

%~*variables storing subjects that fail number of trials (833), bad_channel (>12) 
%and bad_trial (>208) checks below
bad_sub_ntrials={};
bad_sub_channel={};
bad_sub_trial={};

%time script
tic;
for x=1:length(sessions)
    session_ID=sessions{x};                 % x is the index which refers to the row number 
    
    %% identify which task types are present (t70 and t300)

    %~*updated this to make the looping between t70 and t300 more efficient  
    t70_session=subj_info.t70(x);           % want to determine if this subject row has a t70 session
    t300_session=subj_info.t300(x);
    num_runs={};
    if t70_session==1
        num_runs={'t70'};
    end
    if t300_session==1
        num_runs=[num_runs,'t300'];
    end
    
    %loop through runs
    for rr=1:length(num_runs)
        %if statement to check input data types
        if strcmp(num_runs{rr},'t70')
            %set t70 parms
            %~*changed outputfile to remove the period, and generally simplify
            %outputfile = [outputDir,subject_ID,'_t70','_preproc_hp',num2str(filter_hp),'_notch_seg_autochannelz',num2str(meanabs_channel_thresh),'_trialz',num2str(meanabs_trial_thresh),'_baselinelp',num2str(filter_lp),'avgref.mat'];
            outputfile = [outputDir,session_ID,'_t70','_preproc_notch_hp',hp_pref,'_lp',num2str(filter_lp),'_down',num2str(downsamp),'_segment_channel',...
                num2str(meanabs_channel_thresh),'_trial',num2str(meanabs_trial_thresh),'_baseline_avgref.mat'];
            ICA_output = [outputDir,session_ID, '_t70','_segICAcomps_ica_extended_channeltrialthresh',num2str(meanabs_channel_thresh),'.mat'];
            in_suff='_t70.mff';
            
            %markers (differ for t70 vs t300)
            trial_conds = {'P7s1';'P7sd';'P7dv'}; 
            
        elseif strcmp(num_runs{rr},'t300')
            %set t300 parms here
            outputfile = [outputDir,session_ID,'_t300','_preproc_notch_hp',hp_pref,'_lp',num2str(filter_lp),'_down',num2str(downsamp),'_segment_channel',...
                num2str(meanabs_channel_thresh),'_trial',num2str(meanabs_trial_thresh),'_baseline_avgref.mat'];
            ICA_output = [outputDir,session_ID, '_t300','_segICAcomps_ica_extended_channeltrialthresh',num2str(meanabs_channel_thresh),'.mat'];
            in_suff='_t300.mff';
            
            %markers (differ for t70 vs t300)
            trial_conds = {'P3s1';'P3sd';'P3dv'}; 
            
        end
        
        %load in data
        sub_in = [input_dir,session_ID, in_suff]; 
        
        %1. Bandstop filter line noise
        %*don't need to include harmonics, because data will be lowpassed (400 system) or has already been
        %lowpassed (other systems)
        cfg = [];
        cfg.dataset = sub_in;
        cfg.continuous= 'yes';
        cfg.bsfilter ='yes';            %need to notch filter for electrical noise centered around 60 Hz 
        cfg.bsfreq=[59 61];  
        %cfg.bsfreq=[59 61; 119 121; 179 181];
        cfg.bsfilttype= 'firws';
        cfg.bsfiltdir='onepass-minphase';
        
        %2. For 400 system only: highpass filter and lowpass filter
        %to make 400 system data comparable to other systems: highpass
        %filter (0.1Hz), lowpass filter (100Hz)
        system=subj_info.system(x);         %system is the column name. (x) because needs to relate back to that specific eeg file being system 400 
        if system== 400
            %highpass
            cfg.hpfilter='yes';
            cfg.hpfreq=filter_hp;            %high pass filter using filter_hp which is set to 0.1 
            cfg.hpfilttype='firws';
            cfg.hpfiltdir='onepass-minphase';
            
            %lowpass
            cfg.lpfilter = 'yes';
            cfg.lpfreq = filter_lp;
            cfg.lpfilttype='firws';
            cfg.lpfiltdir='onepass-minphase';
            
        end
    
        data_cont = ft_preprocessing(cfg);  %stored continuous filtered data
        
        %3. segment
        cfg=[];
        cfg.dataset= sub_in;                   
        cfg.trialfun='ft_trialfun_general';             %default to read events 
        cfg.trialdef.eventtype= trial_conds;        
        cfg.trialdef.eventvalue= trial_conds;
        cfg.trialdef.prestim=task_epoch_prestim;        %
        cfg.trialdef.poststim= task_epoch_poststim;     %
        cfg_trial=ft_definetrial(cfg);
        
        %segment data based on tone pairs 
        cfg = [];
        cfg.trl = cfg_trial.trl;                    
        data_seg = ft_redefinetrial(cfg,data_cont); %stored segmented data 
        
        %CHECK THAT EACH SESSION HAS 833 TRIALS
        %~*note that this will pause the script frequently as it loops over
        %subjects; consider commenting this out if you can verify that the
        %script can handle inconsistent number of trials per dataset...
        if size(data_seg.trial,2)~=833
            %add subject to counter
            bad_cell={session_ID,in_suff,size(data_seg.trial,2)};
            bad_sub_ntrials=[bad_sub_ntrials;bad_cell];
            
            disp(['dataset: ',session_ID,' does not have expected 833 number of trials!']);
            %keyboard;
        end
        
        %4. for 400 system only - downsample from 1000Hz to 250Hz
        %*downsample both data_cont (needed for bad_chan ID) and data_seg
        %(used for rest of preproc)
        if system==400
            cfg = [];
            cfg.resamplefs = downsamp;
            cfg.resamplemethod = 'downsample';
            cfg.detrend = 'no';
            cfg.demean = 'no';
            data_cont=ft_resampledata(cfg, data_cont);
            data_seg=ft_resampledata(cfg, data_seg);
        end
        
        %5. Remove "fake" channels i.e. those not connected in the baby nets (EOG channels etc)
        channel_inds=data_cont.label;
        fake_chans=[125;126;127;128];
        channel_inds(fake_chans)=[];
        cfg= [];
        cfg.channel =channel_inds;
        %remove from cont (needed for channel removal) and seg
        data_cont=ft_selectdata(cfg,data_cont);
        data_seg=ft_selectdata(cfg,data_seg);


%     %visualize segmented trial data 
%             cfg_view = [];
%             cfg.blocksize = 300;
%             cfg.ylim = [-150000 150000];
%             ft_databrowser(cfg_view, data_seg);


        %6. Noisy channel removal via automated approach
        cont_data=data_cont.trial{1};
        
        %identify noisy channels based on mean amplitude 
        meanabs_channel = nanmean(abs(cont_data),2); %the ,2 indicates to go by row and rows are channels. so takes the mean for each column by looking at row (that channel across time)
        %convert to zscore - accommodates 'baseline' differences between subjects better than thresholding the raw abs amplitude 
        meanabs_channel = zscore(meanabs_channel);
        
        %**updated approach: censor channels based on both zscore > 3 (high
        %amplitude deflection channels) and zscore < -3 (bad contact,
        %flatlined, bridged electrodes)
        bad_channels_pos = find(meanabs_channel>meanabs_channel_thresh); %change to meansabs_channel_sorted once function is developed 
        bad_channels_neg = find(meanabs_channel<meanabs_channel_thresh*-1); %change to meansabs_channel_sorted once function is developed 
        %sometimes the reference ends up in bad_channels_neg but we don't
        %want to remove this
        ix=find(bad_channels_neg==125);
        if ~isempty(ix)
            bad_channels_neg(ix)=[];
        end
        
        bad_channels = [bad_channels_pos;bad_channels_neg];

        %remove subject with too many bad channels 
        if length(bad_channels) > 12
            %add subject to counter
            bad_cell={session_ID,in_suff,length(bad_channels)};
            bad_sub_channel=[bad_sub_channel;bad_cell];
            
            disp('WARNING: subject has more than 12 channels that exceed the noise threshold - consider removing subject?');
            %keyboard;
        end
        
        execute=1;
        if execute==1
            %visualize continuous channel data before rejecting
            cfg = [];
            cfg.blocksize = 10;
            cfg.ylim = [-1000 1000];
            %Set channels to visualize - above-threshold ones, and 10 mid-range normal
            %electrodes (for comparison); this isn't working - loads up all
            %channels rather than specified ones!
            %channel_vis = [bad_channels',good_channels'];
            %eval(['cfg.channels = data_',tt_t,'_cont.label(channel_vis,1);']);
            %eval(['ft_databrowser(cfg, data_',tt_t,'_cont);']);
            ft_databrowser(cfg,data_cont);
        end

        %reject bad channels 
        if ~isempty(bad_channels)                      % ~ means it is NOT empty 
            channel_inds=data_cont.label;
            channel_inds(bad_channels)=[];             % removes all the bad chanels from this array 
            cfg= [];
            cfg.channel =channel_inds;
            data_seg=ft_selectdata(cfg,data_seg);   %replaces data_seg with only the good channels 
        end 

        %7. Identify and remove noisy trials
        %loop through and take trial average
        meanabs_trial = [];
        for tt = 1:size(data_seg.trial,2)
            %mean_tt = nanmean(abs(data_seg.trial{1,tt}(1:num_chans-1,:)),2);
            mean_tt=nanmean(abs(data_seg.trial{1,tt}),2);
            mean_tt=nanmean(mean_tt);
            meanabs_trial = [meanabs_trial;mean_tt];
        end

        %Convert to zscores (as with noisy channels)
        meanabs_trial = zscore(meanabs_trial);

        %store above-threshold trials in vector 'bad_trials'
        bad_trials = find(meanabs_trial > meanabs_trial_thresh);
        

        %reject above-threshold trials
        if ~isempty(bad_trials)
            cfg = [];
            good_trials = 1:size(data_seg.trial,2);
            good_trials(bad_trials)=[];
            cfg.trials = good_trials;
            data_seg = ft_selectdata(cfg,data_seg);
        end
        
        %*Check for bad subject - remove subject with too many bad trials 
        if length(bad_trials) > 208
            %add subject to counter
            bad_cell={session_ID,in_suff,length(bad_trials)};
            bad_sub_trial=[bad_sub_trial;bad_cell];
            
            disp('WARNING: subject has more than 25% of trials that exceed the noise threshold - consider removing subject?');
            %keyboard;
        end
        
        %8. *ICA - create virtual V/HEOG channels (skipped for now)
        
        %9. *ICA - run ICA, then semi-auto eye artifact detection, visually
        %   inspect before finally removing artifact components (skipped for
        %   now)
        
        %10. baseline correct and re-reference to common average 
        %baseline correct using first 200 ms (-0.2s) 
        cfg = [];
        cfg.demean = 'yes';
        cfg.baselinewindow = [task_epoch_baseline 0];
        %reref to common avg
        cfg.reref = 'yes';
        cfg.refchannel = 'all';
        data_seg = ft_preprocessing(cfg,data_seg);
        
        %11. Save dataset output
        save(outputfile,'data_seg','bad_channels','bad_trials','-v7.3'); 
        
        %print timing
        disp(['time taken to preproc session ',num2str(x),'=',num2str(toc)]);
        
    end
end
%save bad subject variables
out_file=[outputDir,'bad_subject_vars'];
save(out_file,'bad_sub_ntrials','bad_sub_channel','bad_sub_trial','-v7.3'); 
    
    
%     if t70_session==1
%         
%         %~*changed outputfile to remove the period, and generally simplify
%         %outputfile = [outputDir,subject_ID,'_t70','_preproc_hp',num2str(filter_hp),'_notch_seg_autochannelz',num2str(meanabs_channel_thresh),'_trialz',num2str(meanabs_trial_thresh),'_baselinelp',num2str(filter_lp),'avgref.mat'];
%         outputfile = [outputDir,session_ID,'_t70','_preproc_notch_hp',hp_pref,'_lp',num2str(filter_lp),'_down',num2str(downsamp),'_segment_channel',...
%             num2str(meanabs_channel_thresh),'_trial',num2str(meanabs_trial_thresh),'_baseline_avgref.mat'];
%         ICA_output = [outputDir,session_ID, '_t70','_segICAcomps_ica_extended_channeltrialthresh',num2str(meanabs_channel_thresh),'.mat'];
%         in_suff='_t70.mff';
% 
%         t70 = [input_dir,session_ID, '_t70.mff'];   %imports session file . renames as t70 for loop   
%         cfg =[];
%         cfg.dataset= [t70];
%         cfg.continuous= 'yes';
%         
%     %cfg.headerformat ='egi_mff';        
%     %ft_databrowser(cfg);
%     %data=ft_preprocessing(cfg);
%     %[event] = ft_read_event(complex_tones, 'dataformat', 'egi_mff', 'headerformat', 'egi_mff');
%  
%     
% %% Notch Filter 
% 
% %~*you need to include the higher harmonics here too, otherwise residual
% %line noise will interefere with the ICA
% %cfg.bsfreq=[59 61; 119 121; 179 181];cfg.bsfreq=[59 61; 119 121; 179 181];
% 
%         cfg.bsfilter ='yes';            %need to notch filter for electrical noise centered around 60 Hz 
%         %cfg.bsfreq=[59 61];  
%         cfg.bsfreq=[59 61; 119 121; 179 181];
%         cfg.bsfilttype= 'firws';
%         cfg.bsfiltdir='onepass-minphase';
%         
% 
%     
%  %% HIGHTPASS FILTER (400 only) 
%  
%         system=subj_info.system(x);         %system is the column name. (x) because needs to relate back to that specific eeg file being system 400 
%         if system== 400
%             cfg.hpfilter='yes';
%             cfg.hpfreq=filter_hp;            %high pass filter using filter_hp which is set to 0.1 
%             cfg.hpfilttype='firws';
%             cfg.hpfiltdir='onepass-minphase';
%         end
%     
%         data_cont = ft_preprocessing(cfg);  %stored continuous filtered data
%         
% %% DOWNSAMPLE 400 DATA 
% %400 data was collected at 1000Hz. To match previous acquisitions it will be downsampled to 250 Hz before further analysis 
% 
% %~*shifted this up here so that we have matched the different systems
% %earlier in the pipeline, therefore making artifact rejection more
% %comparable across them...
% 
%     if system== '400'
%         cfg = [];
%         cfg.resamplefs= 250;
%         cfg.resamplemethod= 'downsample';
%         cfg.detrend = 'no';
%         cfg.demean = 'no';
%         data_cont=ft_resample(cfg, data_cont);
%     end 
% 
% %% SEGMENT AS TONE PAIRS (bgin) OR BY SPECIFIC CONDITION
% 
%         t70_conditions = {'P7s1';'P7sd';'P7dv'};        %names of markers on eeg file 
%         cfg=[];
%         cfg.dataset= t70;                   
%         cfg.trialfun='ft_trialfun_general';             %default to read events 
%         cfg.trialdef.eventtype= t70_conditions ;        
%         cfg.trialdef.eventvalue= t70_conditions;
%         cfg.trialdef.prestim=task_epoch_prestim;        %0.2 pre tone pair
%         cfg.trialdef.poststim= task_epoch_poststim;     %0.2 post tone pair  
%         cfg_trial=ft_definetrial(cfg);
%         
%         %cfg=[];
%         %cfg.dataset=t70;     
%         %cfg.trialdef.eventtype = tone_pair;
%         %cfg.trialdef.eventvalue= tone_pair; 
%         %cfg.trialfun = 'ft_trialfun_general';               %to show all recorded events use ft_trialfun_show %general
%         %cfg.trialdef.prestim= task_epoch_prestim;
%         %cfg.trialdef.poststim= task_epoch_poststim;
%         %cfg_trial =ft_definetrial(cfg);                     %it is not able to read the headerfile. trialfun is error. added cfg.trialfun= 'ft_trialfun_general' 
%     %asign as new configuration strucutre so does not override others 
%     
%     
% %segment data based on tone pairs 
%         cfg_seg = [];
%         cfg_seg.trl = cfg_trial.trl;                    
%         data_seg = ft_redefinetrial(cfg_seg,data_cont); %stored segmented data 
%     
%       
% %visualize segmented trial data 
%         cfg_view = [];
%         cfg.blocksize = 300;
%         cfg.ylim = [-150000 150000];
%         ft_databrowser(cfg_view, data_seg);
% 
% % CHECK THAT EACH SESSION HAS 833 TRIALS 
% 
% 
%   %% ARTIFACTS 
%   
%      %initializing variables to store subjects' bad data 
%         sub_bad_channels={};
%         sub_bad_trials={};
%         sub_bad_ICs={};
% 
% 
% %% REJECT "FAKE" CHANS 
%    
%  % need to remove channels 125-128 as they are not active on baby nets
%     fake
%    
% 
%  %% IDENTIFY NOISY CHANNELS 
% 
% %This will be done by an automated procedure - that analyzes each channel, by taking the absolute average across all samples
% %the threshold is based on the z-score which we have set to 3 
%    
%         
%  
%      eval(['cont_data=data_cont.trial{1,1}']); 
%     %acess the continuous data saved in data_cont and use each trial (.trial in that data file). 
%     %{1,1} because info is stored in first column and first row of that variable data_cont.trial
%     %use continuous signal from each channel 
%     
%  
% 
%     %identify noisy channels based on mean amplitude 
%     meanabs_channel = nanmean(abs(cont_data),2); %the ,2 indicates to go by row and rows are channels. so takes the mean for each column by looking at row (that channel across time)
%     %convert to zscore - accommodates 'baseline' differences between subjects better than thresholding the raw abs amplitude 
%     meanabs_channel = zscore(meanabs_channel);
% 
%   
% %plot channel meaures 
% %need to create a function to visual the bad channels being removed 
% 
%    
% %store above-threshold channels in vector 'bad_channels'
%         bad_channels = find(meanabs_channel(:,1)>meanabs_channel_thresh,2); %change to meansabs_channel_sorted once function is developed 
% 
% 
% %remove subject with too many bad channels 
%         if length(bad_channels) > 12
%             disp('WARNING: subject has more than 12 channels that exceed the noise threshold - consider removing subject?');
%         end
%         
% %visually verify channels 
%  %need to work on this 
%  %   if run_semiauto==1
%  %      figure; hold on;
%  %      cfg = [];
%  %      cfg.blocksize = 300;
%  %      cfg.ylim = [-150000 150000];
%  %      ft_databrowser(cfg, bad_channels)
%    %end
% 
% %reject bad channels 
%         if ~isempty(bad_channels)                      % ~ measn it is NOT empty       
%            
%             channel_inds(bad_channels)=[];             % removes all the bad chanels from this array 
% 
%             cfg= [];
%             cfg.channel =channel_inds;
%             data_seg=ft_selectdata(cfg,data_seg);   %replaces data_seg with only the good channels 
%         end 
% 
% %store in variable sub_bad_channels 
%         sub_bad_channels=bad_channels; %rename to sub_bad_channels 
%     
%         
% 
% %% IDENTIFY NOISY TRIAL 
% 
% % Automated procedure - for each trial, takes the absolute average across all channels for all samples
% 
% % Loop through trials and take the absolute average across all channels and samples
%     
%         eval(['seg_data=data_seg.trial{1,:}']);
%         meanabs_trial = nanmean(abs(seg_data),2);
% %Convert to zscores (as with noisy channels)
%         meanabs_trial = zscore(meanabs_trial);
% 
% 
% % plots channel- or trial-specific measures
%   
% 
% %store above-threshold trials in vector 'bad_trials'
%         bad_trials = find(meanabs_trial(:,1)>meanabs_trial_thresh,2);
%      
% 
% %visualize the above-threshold trials to check 
%  %       if run_semiauto==1
%  %           figure; hold on;
%  %           cfg = [];
%  %           cfg.blocksize = 300; %for visualization purposes
%  %           cfg.ylim = [-60000 60000];  %for visualization 
%  %           ft_databrowser(cfg, data_seg)
%  %       end
% 
% %remove subject with too many bad trials 
%         if length(bad_trials) > 208
%             disp('WARNING: subject has more than 25% of trials that exceed the noise threshold - consider removing subject?');
%         end
% 
% %reject above-threshold trials
%         if ~isempty(bad_trials)
%             trials= 1:length(data_seg.trial);
%             trials(bad_trials)=[];
%             cfg = [];
%             cfg.trials=trials;
%             data_seg_final = ft_selectdata(cfg,data_seg); %replaces data_seg with only good trials (already has been replaced with only good channels) 
%         else 
%             data_seg_final=ft_selectdata(cfg,data_seg);
%         end
% 
% %store in var
%         sub_bad_trials=bad_trials;
% 
% 
% 
% 
% %% ICA WILL GO HERE IF NEEDED !!!
% 
% %% ICA 
% %testing to see on a few subjects how it looks 
% 
% %Using binica extended
% %testing this on just a few subjects before applying to all
% 
% addpath /projectsn/f_mc1689_1/baby_actflow/docs/toolboxes/eeglab13_6_5b/;
% 
% %loop through task types and compute ica
% %system=subj_info.system(x);
% %if system==400 
%    
%     cfg = [];
%     cfg.method = 'binica';
%     cfg.binica.extended = 1;
% %exclude reference channel from ICA  Cz is the reference channel and it is listed as E1001 in channel list 
%     num_chans = size(data_seg_final.label,1);
% 
%     cfg.channel = data_seg_final.label(1:num_chans-1);     %removing the last channel which is Cz
%     cfg.runica.pca = 150; %reduces input to specified number of PCs
%     [comp] = ft_componentanalysis(cfg,data_seg_final); %comp for component analysis 
% 
% %save ICA results
%     save(ICA_output,'comp','-v7.3');        %-v7.3 helps store large arrays. compresses data  
%     
%     
% 
% %% ICA FOR BLINKS 
% %not sure if this will be needed. might be too much for baby data. consider removing or an alternative approach 
% 
%   
%     net=subj_info.net(x);
%     net=string(net);
%     switch net 
%         case 'HCL'
%         veog=veog_chans_HCL;
%         heog=heog_chans_HCL;
%         case'GSN'           
%         veog=veog_chans_GSN;
%         heog=heog_chans_GSN;
%     end 
%     if any(ismember(veog,bad_channels))==1  %checking to see if veog channel was removed as a noisy channel 
%         veog= [22 9];
%         disp('Error! The vertical eye channel was removed during noisy channel rejection! Replaced with alternate channels.')
%     end
%     if any(ismember(heog,bad_channels))==1  %checking to see if heog removed as a noisy channel
%         heog= [33 122];
%         disp('Error! The horizontal eye channel was removed during noisy channel rejection! Replaced with alternate channels.')
%     end 
% 
% 
% %pick out V/HEOG channel indices 
%     left_index = find(strcmp(data_seg_final.label, ['E',num2str(heog(1,1))]));
%     right_index = find(strcmp(data_seg_final.label, ['E',num2str(heog(1,2))]));
%     low_index =find(strcmp(data_seg_final.label,['E',num2str(veog(1,1))]));
%     high_index =find(strcmp(data_seg_final.label,['E',num2str(veog(1,2))]));
% 
%     %not really low index. they are both high index but just left and right side 
%     
%     
% %Pull out IC timecourses (comp.trial), concatenate into one long
% %comp x samples array; do the same for the veog and heog channels
%     curr_trials = size(comp.trial,2);
%     ic_timecourses = [];
%     veog_tcs = [];
%     heog_tcs = [];
%     for tt=1:curr_trials
%         ic_timecourses = [ic_timecourses,comp.trial{1,tt}];
%         veog_tt = [data_seg_final.trial{1,tt}(low_index,:);data_seg_final.trial{1,tt}(high_index,:)];
%         veog_tcs = [veog_tcs,veog_tt];
%         heog_tt = [data_seg_final.trial{1,tt}(left_index,:);data_seg_final.trial{1,tt}(right_index,:)];
%         heog_tcs = [heog_tcs,heog_tt];
%     end
% 
% 
% 
% %create difference channels for veog and heog 
%     veog_diff = veog_tcs(1,:)-veog_tcs(2,:);
%     heog_diff = heog_tcs(1,:)-heog_tcs(2,:);
% 
% 
% %Blink (veog) correlation
%     [tempcorr_blink_r tempcorr_blink_p]  = corr(veog_diff',ic_timecourses');
% 
% 
% %reject based on zscore threshold and store in bad_
%     bad_ICs_blinks_tempcorr = tempcorr_blink_r(tempcorr_blink_r(:,1)>ica_thresh,2);
% 
% %Repeat for eye movements
% %Eye movement (heog) correlation
%     [tempcorr_eyemov_r tempcorr_eyemov_p] = corr(heog_diff',ic_timecourses');
%     tempcorr_eyemov_r_z = zscore(abs(tempcorr_eyemov_r));
% 
%     tempcorr_eyemov_r_z(bad_ICs_blinks_tempcorr)=NaN;
% 
% %VISUALIZE
% 
% %reject based on zscore (or r) threshold
%     bad_ICs_eyemov_tempcorr = tempcorr_eyemov_r_z(tempcorr_eyemov_r(:,1)>ica_thresh,2);
%  
% 
% %% REJECT ICs
%     %collapse ICs
%     bad_ICs_tempcorr = [bad_ICs_blinks_tempcorr',bad_ICs_eyemov_tempcorr'];
%     bad_ICs_tempcorrvis = [bad_ICs_blinks_tempcorr',bad_ICs_eyemov_tempcorr'];
% 
% %Reject auto+vis - tempcorrvis
%     cfg = [];
%     cfg.component = bad_ICs_tempcorrvis; %enter artifact components to be removed
%     data_seg_channels_trials_autovisIC = ft_rejectcomponent(cfg,comp,data_seg_final);
% 
%     %clear data_seg_final;
% 
% %% DETERMINE IF ICA WAS NEEDED AND REVIEW 
%     
% 
% 
% %% RESEGMENT w/o additional padding 
% 
%         cfg=[];
%         %cfg.dataset= data_seg_channels_trials_autovisIC;                   
%         %cfg.trialfun='ft_trialfun_general';             
%         cfg.trialdef.eventtype= t70_conditions ;        
%         cfg.trialdef.eventvalue= t70_conditions;
%         cfg.trialdef.prestim=0.2;        
%         cfg.trialdef.poststim= 0.8;      
%         cfg_trial=ft_definetrial(cfg);
% 
%         
% %% BASELINE CORRECT 
% 
% %baseline correct using first 200 ms (-0.2s) 
%         cfg = [];
%         cfg.demean = 'yes';
%         cfg.baselinewindow = [task_epoch_baseline 0];
% 
% 
% %% LOWPASS FILTER 
% %currently only the 400 sytstem data needs to be lowpassed. 
% %this is subject to change if smaller frequency range is of interest. perhaps starting with 50Hz 
% 
%     if system==400
%         cfg.lpfilter = 'yes';
%         cfg.lpfreq = filter_lp;
%         cfg.lpfilttype='firws';
%         cfg.lpfiltdir='onepass-minphase';
%     end 
%           
%     
% %% REREFERNCE FROM Cz TO COMMON AVERAGE 
% 
%     cfg.reref = 'yes';
%     cfg.refchannel = 'all';
%     
% 
% 
% 
% %% SAVE 
%     save(outputfile,'sub_bad_ICs','sub_bad_channels','sub_bad_trials','data_seg_channels_trials_autovisIC','-v7.3');   
%     %right now not saving ICA because still testing 
%     %end
%     
%     
%    
% %% 300 DATA for SUBJECT 
%     %same process now for 300 session 
%          
%     t300_session=subj_info.t300(x);         
%     if t300_session==1
%         
%         outputfile = [outputDir,session_ID,'_t300','_preproc_hp',num2str(filter_hp),'_notch_seg_autochannelz',num2str(meanabs_channel_thresh),'_trialz',num2str(meanabs_trial_thresh),'_baselinelp',num2str(filter_lp),'avgref.mat'];
%         ICA_output = [outputDir,session_ID, '_t300','_segICAcomps_ica_extended_channeltrialthresh',num2str(meanabs_channel_thresh),'.mat'];
%         
%         t300 = [input_dir,session_ID, '_t300.mff'];   
%         cfg =[];
%         cfg.dataset= [t300];
%         cfg.continuous= 'yes';
% 
%     
% %% Notch Filter 
% 
%         cfg.bsfilter ='yes';           
%         cfg.bsfreq=[59 61];                
%         cfg.bsfilttype= 'firws';
%         cfg.bsfiltdir='onepass-minphase';
% 
%     
%  %% HIGHTPASS FILTER (400 only) 
%  
%         system=subj_info.system(x);        
%         if system== 400
%             cfg.hpfilter='yes';
%             cfg.hpfreq=filter_hp;            
%             cfg.hpfilttype='firws';
%             cfg.hpfiltdir='onepass-minphase';
%         end
%     
%         data_cont = ft_preprocessing(cfg);  
% 
% %% SEGMENT AS TONE PAIRS (bgin) OR BY SPECIFIC CONDITION
% 
%         t300_conditions = {'P3s1';'P3sd';'P3dv'};         
%         cfg=[];
%         cfg.dataset= t300;                   
%         cfg.trialfun='ft_trialfun_general';            
%         cfg.trialdef.eventtype= t300_conditions ;        
%         cfg.trialdef.eventvalue= t300_conditions;
%         cfg.trialdef.prestim=task_epoch_prestim;       
%         cfg.trialdef.poststim= task_epoch_poststim;     
%         cfg_trial=ft_definetrial(cfg);
%         
%    
% %segment data based on tone pairs 
%         cfg_seg = [];
%         cfg_seg.trl = cfg_trial.trl;                    
%         data_seg = ft_redefinetrial(cfg_seg,data_cont); 
%     
%       
% %visualize segmented trial data 
%         cfg_view = [];
%         cfg.blocksize = 300;
%         cfg.ylim = [-150000 150000];
%         ft_databrowser(cfg_view, data_seg);
% 
% % CHECK THAT EACH SESSION HAS 833 TRIALS 
% 
% 
%   %% ARTIFACTS 
%   
%      %initializing variables to store subjects' bad data 
%         sub_bad_channels={};
%         sub_bad_trials={};
%         sub_bad_ICs={};
%         
%         
% %% REJECT "FAKE" CHANS 
%    
%     
%     channel_inds=data_cont.label;
%     fake_chans=[125;126;127;128];
%     channel_inds(fake_chans)=[];
%     cfg= [];
%     cfg.channel =channel_inds;
%     data_cont=ft_selectdata(cfg,data_cont);
% 
%     
% 
%  %% IDENTIFY NOISY CHANNELS 
%      
%         eval(['cont_data=data_cont.trial{1,1}']); 
%     %acess the continuous data saved in data_cont and use each trial (.trial in that data file). 
%     %{1,1} because info is stored in first column and first row of that variable data_cont.trial
%     %use continuous signal from each channel 
%    
% 
%         
% %identify noisy channels based on mean amplitude 
%         meanabs_channel = nanmean(abs(cont_data),2); 
%         meanabs_channel = zscore(meanabs_channel);
% 
%   
% %plot channel meaures 
% %need to create a function to visual the bad channels being removed 
% %    [meanabs_channel_sorted] = Visualize_NoiseMeasure_ChannelTrial(meanabs_channel,50);
% %this a custom function. need to develop this for my code. what is the 50 from 
%    
% %store above-threshold channels in vector 'bad_channels'
%         bad_channels = find(meanabs_channel(:,1)>meanabs_channel_thresh,2); %change to meansabs_channel_sorted once function is developed 
% 
% 
% %remove subject with too many bad channels 
%         if length(bad_channels) > 12
%             disp('WARNING: subject has more than 12 channels that exceed the noise threshold - consider removing subject?');
%         end
%         
% %visually verify channels 
%  %need to work on this so channel_vis has databrowser function 
%     if run_semiauto==1
%         figure; hold on;
%         cfg = [];
%         cfg.blocksize = 300;
%         cfg.ylim = [-150000 150000];
%         channel_vis = 'bad_channels';
%    end
% 
% %reject bad channels 
%         if ~isempty(bad_channels)                     
%             channel_inds=data_seg.label';       
%             channel_inds(bad_channels)=[];            
%             cfg= [];
%             cfg.channel =channel_inds;
%             data_seg=ft_selectdata(cfg,data_seg);  
%         end 
% 
% %store in variable sub_bad_channels 
%         sub_bad_channels=bad_channels; 
%     
%         
% 
% %% IDENTIFY NOISY TRIAL 
% 
% % Automated procedure - for each trial, takes the absolute average across all channels for all samples
% 
% % Loop through trials and take the absolute average across all channels and samples
%     
%         eval(['seg_data=data_seg.trial{1,:}']);
%         meanabs_trial = nanmean(abs(seg_data),2);
% %Convert to zscores (as with noisy channels)
%         meanabs_trial = zscore(meanabs_trial);
% 
% 
% %Send to function that plots channel- or trial-specific measures
%     %[meanabs_trial_sorted] = Visualize_NoiseMeasure_ChannelTrial(meanabs_trial,num_trls);
% 
% %store above-threshold trials in vector 'bad_trials'
%         bad_trials = find(meanabs_trial(:,1)>meanabs_trial_thresh,2);
%      
% 
% %visualize the above-threshold trials to check 
%         if run_semiauto==1
%             figure; hold on;
%             cfg = [];
%             cfg.blocksize = 300;
%             cfg.ylim = [-60000 60000]; 
%             ft_databrowser(cfg, data_seg)
%         end
% 
% %remove subject with too many bad trials 
%         if length(bad_trials) > 208
%             disp('WARNING: subject has more than 25% of trials that exceed the noise threshold - consider removing subject?');
%         end
% 
% %reject above-threshold trials
%         if ~isempty(bad_trials)
%             trials= 1:length(data_seg.trial);
%             trials(bad_trials)=[];
%             cfg = [];
%             cfg.trials=trials;
%             data_seg_final = ft_selectdata(cfg,data_seg);
%         else 
%             data_seg_final=ft_selectdata(cfg,data_seg);
%         end
% 
% %store in var
%         sub_bad_trials=bad_trials;
% 
% 
% 
% 
% %% ICA WILL GO HERE IF NEEDED !!!
% 
% %% ICA 
% %testing to see on a few subjects how it looks 
% 
% %Using binica extended
% %testing this on just a few subjects before applying to all
% 
% addpath /projectsn/f_mc1689_1/baby_actflow/docs/toolboxes/eeglab13_6_5b/;
% 
% %loop through task types and compute ica
% %system=subj_info.system(x);
% %if system==400 
%    
%     cfg = [];
%     cfg.method = 'binica';
%     cfg.binica.extended = 1;
% %exclude reference channel from ICA  Cz is the reference channel and it is listed as E1001 in channel list 
%     num_chans = size(data_seg_final.label,1);
% 
%     cfg.channel = data_seg_final.label(1:num_chans-1);     %removing the last channel which is Cz
%     cfg.runica.pca = 150; %reduces input to specified number of PCs
%     [comp] = ft_componentanalysis(cfg,data_seg_final); %comp for component analysis 
% 
% %save ICA results
%     save(ICA_output,'comp','-v7.3');        %-v7.3 helps store large arrays. compresses data  
%     
%     
% 
% %% ICA FOR BLINKS 
% %not sure if this will be needed. might be too much for baby data. consider removing or an alternative approach 
% 
%   
%     net=subj_info.net(x);
%     net=string(net);
%     switch net 
%         case 'HCL'
%         veog=veog_chans_HCL;
%         heog=heog_chans_HCL;
%         case'GSN'           
%         veog=veog_chans_GSN;
%         heog=heog_chans_GSN;
%     end 
%     if any(ismember(veog,bad_channels))==1  %checking to see if veog channel was removed as a noisy channel 
%         veog= [22 9];
%         disp('Error! The vertical eye channel was removed during noisy channel rejection! Replaced with alternate channels.')
%     end
%     if any(ismember(heog,bad_channels))==1  %checking to see if heog removed as a noisy channel
%         heog= [33 122];
%         disp('Error! The horizontal eye channel was removed during noisy channel rejection! Replaced with alternate channels.')
%     end 
% 
% 
% %pick out V/HEOG channel indices 
%     left_index = find(strcmp(data_seg_final.label, ['E',num2str(heog(1,1))]));
%     right_index = find(strcmp(data_seg_final.label, ['E',num2str(heog(1,2))]));
%     low_index=find(strcmp(data_seg_final.label,['E',num2str(veog(1,1))]));
%     high_index=find(strcmp(data_seg_final.label,['E',num2str(veog(1,2))]));
% 
% 
%    
% %Pull out IC timecourses (comp.trial), concatenate into one long
% %comp x samples array; do the same for the veog and heog channels
%     curr_trials = size(comp.trial,2);
%     ic_timecourses = [];
%     veog_tcs = [];
%     heog_tcs = [];
%     for tt=1:curr_trials
%         ic_timecourses = [ic_timecourses,comp.trial{1,tt}];
%         veog_tt = [data_seg_final.trial{1,tt}(low_index,:);data_seg_final.trial{1,tt}(high_index,:)];
%         veog_tcs = [veog_tcs,veog_tt];
%         heog_tt = [data_seg_final.trial{1,tt}(left_index,:);data_seg_final.trial{1,tt}(right_index,:)];
%         heog_tcs = [heog_tcs,heog_tt];
%     end
% 
% 
% %create difference channels for veog and heog 
%     veog_diff = veog_tcs(1,:)-veog_tcs(2,:);
%     heog_diff = heog_tcs(1,:)-heog_tcs(2,:);
% 
% 
% %Blink (veog) correlation
%     [tempcorr_blink_r tempcorr_blink_p]  = corr(veog_diff',ic_timecourses');
% 
% 
% %Sort ICs in descending order of abs corr and plot the results
% %Using zscore rather than raw absolute correlation as before
% %[tempcorr_blink_sorted] = Visualize_NoiseMeasure_ChannelTrial(abs(tempcorr_blink_r)',50);
%     %[tempcorr_blink_sorted_z] = Visualize_NoiseMeasure_ChannelTrial(zscore(abs(tempcorr_blink_r))',50);
% 
% %reject based on zscore threshold and store in bad_
%     bad_ICs_blinks_tempcorr = tempcorr_blink_r(tempcorr_blink_r(:,1)>ica_thresh,2);
% 
% %Repeat for eye movements
% %Eye movement (heog) correlation
%     [tempcorr_eyemov_r tempcorr_eyemov_p] = corr(heog_diff',ic_timecourses');
%     tempcorr_eyemov_r_z = zscore(abs(tempcorr_eyemov_r));
% 
% %Remove correlation with eyeblink ICs - replace with Nan. Blink ICs will also
% %correlate with heog, but are more likely to be blinks if already
% %correlated with veog...
% %tempcorr_eyemov_r(bad_ICs_blinks_tempcorr)=NaN;
%     tempcorr_eyemov_r_z(bad_ICs_blinks_tempcorr)=NaN;
% 
% %[tempcorr_eyemov_sorted] = Visualize_NoiseMeasure_ChannelTrial(abs(tempcorr_eyemov_r)',50);
%     %[tempcorr_eyemov_sorted_z] = Visualize_NoiseMeasure_ChannelTrial(tempcorr_eyemov_r_z',50);
% 
% %reject based on zscore (or r) threshold
%     bad_ICs_eyemov_tempcorr = tempcorr_eyemov_r_z(tempcorr_eyemov_r(:,1)>ica_thresh,2);
%  
% 
% %% REJECT ICs
%     %collapse ICs
%     bad_ICs_tempcorr = [bad_ICs_blinks_tempcorr',bad_ICs_eyemov_tempcorr'];
%     bad_ICs_tempcorrvis = [bad_ICs_blinks_tempcorr',bad_ICs_eyemov_tempcorr'];
% 
% %Reject auto+vis - tempcorrvis
%     cfg = [];
%     cfg.component = bad_ICs_tempcorrvis; %enter artifact components to be removed
%     data_seg_channels_trials_autovisIC = ft_rejectcomponent(cfg,comp,data_seg_final);
% 
%     %clear data_seg_final;
% 
% 
%  %DETERMINE IF ICA NEEDED    
%     
%  %% RESEGMENT w/o additional padding 
% 
%      cfg=[];
%      %cfg.dataset= data_seg_channels_trials_autovisIC;                   
%      %cfg.trialfun='ft_trialfun_general';             
%      cfg.trialdef.eventtype= t70_conditions ;        
%      cfg.trialdef.eventvalue= t70_conditions;
%      cfg.trialdef.prestim=0.2;        
%      cfg.trialdef.poststim= 0.8;      
%      cfg_trial=ft_definetrial(cfg);
%    
%     
% %% BASELINE CORRECT 
% 
% %baseline correct using first 200 ms (-0.2s) 
%         cfg = [];
%         cfg.demean = 'yes';
%         cfg.baselinewindow = [task_epoch_baseline 0];
% 
% 
% %% LOWPASS FILTER 
% %currently only the 400 sytstem data needs to be lowpassed. 
% %this is subject to change if smaller frequency range is of interest. perhaps starting with 50Hz 
% 
%     if system==400
%         cfg.lpfilter = 'yes';
%         cfg.lpfreq = filter_lp;
%         cfg.lpfilttype='firws';
%         cfg.lpfiltdir='onepass-minphase';
%     end 
%           
%     
% %% REREFERNCE FROM Cz TO COMMON AVERAGE 
% 
%     cfg.reref = 'yes';
%     cfg.refchannel = 'all';
%     
% 
% %% DOWNSAMPLE 400 DATA 
% %400 data was collected at 1000Hz. To match previous acquisitions it will be downsampled to 250 Hz before further analysis 
% 
%     if system== '400'
%         cfg.resamplefs= 250;
%         cfg.method= 'downsample';
%         data_400=ft_resample(cfg, cfg.dataset);
%     end 
% 
% %% SAVE 
%     
%     save(outputfile,'sub_bad_ICs','sub_bad_channels','sub_bad_trials','data_seg_channels_trials_autovisIC','-v7.3');   
%         
%     
%     end
%          
% 
% end 
% 
%   


