%Step4_DynamicDecoding_RawSensor_subject_pseudotrial_v2.m

%Author: Ravi Mill, rdm146@rutgers.edu, adapted by Lily Thompson,
%let83@rutgers.edu
%Last update: Apr 14th, 2025

%DESCRIPTION: runs dynamic decoding analysis for each subject, using raw sensors as features for Dynamic Activity
%Flow project: Mill, XX, "Title"

%Works for both stim-locked and resp-locked data (latter takes the stim-locked preproc'd data as input, and 
%firstly response-locks it using the RT info stored for each trial)

%Conducts sub-averaging of input trials into a smaller subset of
%'pseudotrials' - Grootswagers shows that this improves decoding
%accuracy.


%% Set path and defaults for Fieldtrip
addpath /projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/;
ft_defaults;

%add cosmomvpa toolbox
addpath /projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/CoSMoMVPA/mvpa/;
addpath /projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/CoSMoMVPA/external/;

%add libsvm (called by cosmomvpa)
addpath /projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/libsvm-3.22/matlab/;

%add path to linux scripts
addpath /projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/scripts/colelablinux_scripts/;

%% Set inputs, parms etc
%can use 5, 30, 31, 32, 34, 35
total_subjects = [1,2,3,5,10,16,17,21,34,44]; %total_subjects and subjects specification helps load in behav files
subjects = [1,2,3];
%baseDir='/home/let83/FND4/';

%path to egi electrode layout (in Fieldtrip templates directory)
elecfile = '/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/template/electrode/GSN-HydroCel-257.sfp';

%condition codes (written in substruct of stim event markers), written here
%for reference
condition_markers = {1:4};
condition_names = {'Left','Right'};
condition_array = [num2cell(condition_markers)',condition_names];

%task segmentation 2 - trial probes (probe 1 onset through probe 2 offset)
epoch_length = [-0.5 0.5]; %reduces multiple comparisons, as well as preventing edge artifacts from downsampling (which would appear at edges of original epochs)

%*set whether to downsample data prior to dynamic MVPA (downsampled using
%Nyquist theorem)
downsample_input=250;

%*confine to correct trials only
correct_only = 1;
if correct_only==1
    corr_suffix = 'correctOnly';
else
    corr_suffix = [];
end

%**run after creating pseudotrials (sub-averaging trials prior to running
%classification to improve SNR)
run_pseudotrials = 0; %0 = no pseudotrial averaging, 10 = loop over 10 repetitions of pseudotrial averaging
avg_pseudotrials = 14; %*set number of trials within each cond to average over
if run_pseudotrials == 0
    pseu_suffix = 'noPseudoTrials';
else
    pseu_suffix = [num2str(run_pseudotrials),'PseudoTrialsAvgOver',num2str(avg_pseudotrials),'Trials'];
end


%**choose condition classification
% BaselineMotorHand = baseline left (f,d) vs right (j,k) hand response
% BaselineMotorFinger = baseline left finger left hand vs right finger left
% hand vs. left finger right hand vs right finger right hand response

% Only study in correct trials
% identify trials using the key chosen, respMotorHand, respMotorFinger,
classify_cond = 'Left';

%Set task_cond values for the specified classification - will be used for
%timelockanalysis later
%.trialinfo cols: trial index, task condition (see condition_array),
%trial stim (1 = horizontal/low, 2 = vertical/high),response (115=s/1,107=k/2), 
%correct(1/0), RT (ms)

%Compares index and middle finger of left hand
if strcmp(classify_cond,'Left') == 1
    cond_code = [1 2];
%Compares index and middle of right hand
elseif strcmp(classify_cond,'Right') == 1
    cond_code = [3 4];
end

%**Set validation_type
%10fold = randomly separate the input trials into train and test sets
%(80/20 ratio), over 10 iterations/folds...
%leaveOneTrialOut = divide the input trials into n train and test sets,
%where the test set cycles through each individual trial...
%10-fold was used in Grootswagers and was comparable to leave-one-trial out...
validation_type = '10fold'; %'10fold', or 'leaveOneTrialOut'

% set whether to filter again
filter_source=[]; %[], or filter bands; change input_suffix
if ~isempty(filter_source)
    out_path = ['filter_',num2str(filter_source(1,1)),'to',num2str(filter_source(1,2)),'Hz'];
else
    out_path=[];
end

%*set response-lock parameters (epoch start and end) - only applies to
%resp-locked data i.e. BaselineMotorAll
resp_pre=0.5;
resp_post=0.5;

%*set analysisName and paths to input and output
data_lock = 'Resp_lock';
analysisName=['/Analysis1_DynamicDecoding_reproc/',data_lock,'/'];
baseDir = '/projectsn/f_mc1689_1/cpro2_eeg';
input_dir = [baseDir,'/data/results/'];
%*set input data suffix
input_suffix = '_preproc_hp0.1notch_seg_autochannelz3_trialz3_ICAz3_baselinelp125avgref'; %z3 contains visICA

%set outputs
output_suffix = ['_SubjectDecoding_',validation_type,'_',classify_cond,'_',corr_suffix,'_',pseu_suffix];
output_dir = [baseDir,'results/DynamicDecoding/RawSensor/',classify_cond,'/'];
if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

%% Start subject loop

for i = 1:length(subjects)
    tic;
    %~~ will have to change the sub_ to sub. Also change directory.
    subject_data_file = [input_dir, 'preproc1_causalFilter/','sub',num2str(subjects(i)),input_suffix,'.mat'];
    load(subject_data_file);
    
    %set data_input based on stim vs resp locked...
%     if strcmp(data_lock,'Stim_locked')==1
%         data_input = data_seg_channels_trials_autovisIC; %using autovisIC i.e. visual inspection of ICAz3 preproc
%         clear data_seg*
%     elseif strcmp(data_lock,'Resp_locked')==1
%         data_input = data_resp;
%     end
    %~~ works up to here
    data_input=data_preproc_out.probes;
   %% Check number of incorrect trials. If high percentage, consider removing subject
    if strcmp(classify_cond, 'Left')==1
        trialinfo = [data_input.trialinfo.acc;data_input.trialinfo.resp];
        accuracy_ = (sum(data_input.trialinfo.acc))/2;
    end
    if accuracy<130
         disp(['Subject' subjects(i) 'has only ' accuracy 'correct trials. Consider skipping?']);
         keyboard;
         %subject 2 has 114.5 correct trials
    end
   %% Response lock the data
    
    data_resp = data_input; %init output
    resp_secs = [];
    for t=1:size(data_input.trialinfo,1)
        start_resp=(data_input.trialinfo{t,15}/1000)-resp_pre; %RT-500
        end_resp=(data_input.trialinfo{t,15}/1000)+resp_post; %RT+500
        resp_secs=[resp_secs;[start_resp,end_resp]];
        %find start and end inds in .time (round to deal with floating
        %point discrepancies)
        start_ind=find(round(data_input.time{t}(1,:),3)==round(start_resp,3));
        end_ind=find(round(data_input.time{t}(1,:),3)==round(end_resp,3)); 

        %*seems like fieldtrip call to redefinetrial includes epoch end
        %timepoints for stim-locked data; hence for consistenchy not
        %removing the last ind for resp_locked

        %use inds to reference data_input.trial
        data_resp.trial{1,t}=data_input.trial{1,t}(:,start_ind:end_ind); %one less than last ind
        %replace data_resp.time with -0.5 to 0.4999
        data_resp.time{1,t}=(resp_pre*-1):0.001:resp_post;
    end

    %replace data_input with data_resp
    for j=length(data_resp.trial):-1:1
        if isempty(data_resp.trial{j})
            data_resp.trial(j)=[];
            data_resp.sampleinfo(j,:)=[];
            data_resp.time(j)=[];
            data_resp.trialinfo(j,:)=[];
        end
    end
  
    data_input=data_resp;

    % Convert key responses into numbers so that we can keep them in
    % trialinfo
    data_input.trialinfo.resp(strcmp(data_input.trialinfo.resp, 'k'))={4};
    data_input.trialinfo.resp(strcmp(data_input.trialinfo.resp, 'j'))={3};
    data_input.trialinfo.resp(strcmp(data_input.trialinfo.resp, 'f'))={2};
    data_input.trialinfo.resp(strcmp(data_input.trialinfo.resp, 'd'))={1};
    data_input.trialinfo.resp = cell2mat(data_input.trialinfo.resp);  % Now fully numeric
    %make trialinfo numeric
    vars = varfun(@isnumeric, data_input.trialinfo, 'OutputFormat', 'uniform');
    data_input.trialinfo = table2array(data_input.trialinfo(:, vars));
        
    %% Dynamic Decoding preproc
    
    %Downsample from 1000 to 200Hz
    %Going with nth downsampling approach below, which is clearly better than decimation (given causal filtering), and
    %equivalent to downaveraging; see Step6 mvar script for further
    %information
    cfg = [];
    cfg.resamplefs = downsample_input;
    cfg.resamplemethod='downsample'; %this skips the lp filter
    cfg.detrend = 'no';
    cfg.demean = 'no';
    data_preproc = ft_resampledata(cfg,data_input);
    
    %% Sort trials that will be classified
            
    %Select trials via ft_timelockanalysis - need to use timelockanalysis with keeptrials for
    cond_indx = find(ismember(data_preproc.trialinfo(:,5),cond_code')==1); %this format should work for both between and within classifications
    incorrect_indx = find(data_preproc.trialinfo(:,4)==0);
    incorrect_cond_indx = find(ismember(cond_indx,incorrect_indx));
    trls = cond_indx;
    
    % Constrain to correct trials only (if specified)
    if correct_only == 1
        trls(incorrect_cond_indx)=[];
    end
    
    %Create 'target' codes (1/2) for use in later classification
    %Hand differences: 1/2, Finger differences: 1/4
    cond_info = data_preproc.trialinfo;
    cond_info = cond_info(trls,:);
    if strcmp(classify_cond,'Left') == 1
        cond_targets = cond_info(:,5) == 1 | cond_info(:,5) == 2;
        cond_targets = cond_info(:,5);
        %cond_targets(cond_targets==1)=1;
        %cond_targets(cond_targets==2)=2;
    elseif strcmp(classify_cond,'Right') == 1
        cond_targets = cond_info(:,5);
        cond_targets(cond_targets==3)=1;
        cond_targets(cond_targets==4)=2;
    end
    
    %call timelock
    cfg = [];
    cfg.keeptrials = 'yes';
    cfg.trials = trls;
    data_trl = ft_timelockanalysis(cfg,data_preproc);
    %runs through here
    %% Conduct sub-averaging into pseudotrials over run_pseudotrials (10)
    %Create cols coding for subaverage inds ('chunks') in each cond; this
    %will be used to reference data_trl.trial to pick out EEG data for
    %sub-averaging...
    %cond1 -- don't have enough trials to do the pseudotrials
    cond1_inds=find(cond_targets==1);
    %add extratrial inds, if this exists
    %cond1_inds=[cond1_inds;repmat([cond1_pseudotrials+1],[cond1_extra,1])];
    %cond1=[cond1,cond1_inds];
    %cond2
    cond2_inds=find(cond_targets==2);
    
    %Pick out data_trl.trial (trial x channels x timepoints), create
    %pseudotrial indicies, average and replace data_trl.trial;; then run 
    %cosmomvpa setup on data_trl and see if this works...
    accuracy_over_folds=[]; %initilize var storing decoding accuracy across pseudofolds
    fold_structs={}; %var storing for struct output after each decoding fold - useful for storing time, feature info for use in group script
    
    %Randperm inds over required number of folds (in run_pseudotrials)
    xx_inds_cond1=cond1_inds(randperm(length(cond1_inds)));
    xx_inds_cond2=cond2_inds(randperm(length(cond2_inds)));
        
    %Average data in sub_trls - pick out inds in cond1 based on
    %randperm above, which sort trials in data_trl.trial for
    %averaging...
    cond1_trls = [];
    for ii=1:length(xx_inds_cond1)
        trlData=(data_trl.trial(xx_inds_cond1(ii),:,:));
        cond1_trls=cat(1, cond1_trls, trlData);
    end
        
    %cond2
    cond2_trls = [];
    for ii=1:length(xx_inds_cond2)
        trlData2=(data_trl.trial(xx_inds_cond2(ii),:,:));
        cond2_trls=cat(1, cond2_trls, trlData2);
    end
        
        
    %replace data_trl.trial with required trials...
    %data_trl_pseudo=data_trl;
    data_trl.trial=[cond1_trls;cond2_trls];
        
    %will also need to write new cond_targets
    cond_targets_pseudo = [repmat(1,[max(cond1_inds),1]);repmat(2,[max(cond2_inds),1])];
    
        
    
    %% Set up Dynamic Decoding via cosmomvpa

    %Convert fieldtrip struct to cosmomvpa format
    ds_tl = cosmo_meeg_dataset(data_trl); %ds_tl is the data struct required by cosmomvpa

    % set the target (trial conditions to be classified)
    ds_tl.sa.targets=cond_targets;

    % set the chunks (independent measurements - individual trials considered independent in M/EEG
    ds_tl.sa.chunks=[1:length(cond_targets)]';

    % just to check everything is ok
    cosmo_check_dataset(ds_tl);

    %Print number of channels, time points and trials
    fprintf('There are %d channels, %d time points and %d trials\n',...
            numel(unique(ds_tl.fa.chan)),numel(unique(ds_tl.fa.time)),...
            size(ds_tl.samples,1));

    %% Run dynamic decoding MVPA

    %1. SETUP TIMEPOINT BY TIMEPOINT CLASSIFICATION
    %This segments the ds_tl.samples (trials x (timepoints x channels)) into time_nbrhood.neighbors
    %radius = 0 performs classification on each timepoint (rather than over a sliding/searchlight window)
    time_nbrhood=cosmo_interval_neighborhood(ds_tl,'time','radius',0);

    %2. SET UP PARTITION SCHEME
    %set chunks for partition creation i.e. cross-validation setup
    if strcmp(validation_type,'10fold')==1
        nfolds=10; %10fold validation; 
        % Define the partitioning scheme using cosmo_independent_samples_partitioner.
        % Use 'fold_count' (10fold validation)
        % Use 'test_ratio',.2 to use 20% of the data for testing (and 80% for
        %training) in each fold; trials are randomly assigned to each fold
        %(with replacement across folds)..
        % *Note that this cosmo function automatically balances training and test 
        %sets for each fold, so that they contain equal number of targets 1 vs 2 
        partitions=cosmo_independent_samples_partitioner(ds_tl,...
                'fold_count',nfolds,...
                'test_ratio',.2);
    elseif strcmp(validation_type,'leaveOneTrialOut')==1
        nfolds=length(cond_targets); %leaveonetrialout
        partitions=cosmo_nchoosek_partitioner(ds_tl,1); %second parm if equals 1 = will make as many partitions as there are chunks
        %partitions=cosmo_balance_partitions(partitions, ds_tl);
    end

    %3. RUN DYNAMIC DECODING -- CHANGES WITH TEMPORAL GENERALIZATION
    measure=@cosmo_crossvalidation_measure; %set up sub function used to run analysis
    %Use the libsvm classifier and the partitions just defined.
    measure_args=struct();
    measure_args.partitions=partitions;
    measure_args.classifier=@cosmo_classify_libsvm;
    if strcmp(validation_type,'leaveOneTrialOut')==1
        measure_args.check_partitions = false; %need to set this to false to run leaveonetrialout (otherwise get an error message about unbalanced training and testing sets)
    end
    d_acc=cosmo_searchlight(ds_tl,time_nbrhood,measure,measure_args);

    %add to overall accuracy vars
    accuracy_over_folds=d_acc.samples; %initilize var storing decoding accuracy across pseudofolds
    fold_structs=d_acc;

   %% visualize
    decoding_accuracy = accuracy_over_folds; %compute mean accuracy for this subject across pseudo folds
    
    %Plot timecourse
    figure();
    plot(fold_structs.a.fdim.values{1,1},decoding_accuracy)
    subjNum = num2str(subjects(i));
    fullTitle = fullfile([classify_cond ' Hand Finger Decoding Accuracy of Subject ' subjNum]);
    title(fullTitle)
    xlabel('Time');
    ylabel('Decoding Accuracy')
    %ylim([0 1])
    print([output_dir,num2str(subjects(i)),output_suffix],'-dbmp','-r300'); %save figure

   %% Save subject output
    save([output_dir,num2str(subjects(i)),output_suffix,'.mat'],'decoding_accuracy','accuracy_over_folds','fold_structs','-v7.3');
                                      
end