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
total_subjects = [2]; %total_subjects and subjects specification helps load in behav files
subjects = total_subjects;
baseDir='/home/let83/FND4/';

%path to egi electrode layout (in Fieldtrip templates directory)
elecfile = '/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/template/electrode/GSN-HydroCel-257.sfp';

%condition codes (written in substruct of stim event markers), written here
%for reference
condition_markers = {1:4};
condition_names = {'LeftIndex','RightIndex','LeftMiddle','RightMiddle'};
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
run_pseudotrials = 10; %0 = no pseudotrial averaging, 10 = loop over 10 repetitions of pseudotrial averaging
avg_pseudotrials = 10; %*set number of trials within each cond to average over
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
classify_cond = 'RespMotorHand';

% %*set whether to run on Stim_locked or Resp_locked data - this depends on
% %classify_cond
% if strcmp(classify_cond,'RespMotorHand')==1
%     data_lock='Resp_locked'; %'Stim_locked'
% elseif strcmp(classify_cond,'RespMotorFinger')==1
%     data_lock='Resp_locked';
% elseif strcmp(classify_cond, 'RespIndexMiddle')==1
%     data_lock='Resp-locked';
% end

%Set task_cond values for the specified classification - will be used for
%timelockanalysis later
%.trialinfo cols: trial index, task condition (see condition_array),
%trial stim (1 = horizontal/low, 2 = vertical/high),response (115=s/1,107=k/2), 
%correct(1/0), RT (ms)

%Compares left vs right hand
if strcmp(classify_cond,'RespMotorHand') == 1
    cond_code = [1 2 3 4];
%Compares each individual finger (with left and right differentiation)
elseif strcmp(classify_cond,'RespMotorHandFinger') == 1
    cond_code = [1 2 3 4];
%Compares Index vs. Middle finger movements across hemispheres 
elseif strcmp(classify_cond,'RespIndexMiddle') == 1
    cond_code = [2 3 1 4]; % LeftIndex vs LeftMiddle and rightIndex vs rightMiddle
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
resp_pre=1.0;
resp_post=1.0;

%*set analysisName and paths to input and output
%data_lock = 'Resp_lock';
analysisName='/Analysis1_DynamicDecoding_reproc/Subj_TempGen/';
baseDirInput = '/projectsn/f_mc1689_1/cpro2_eeg';
input_dir = [baseDirInput,'/data/results/'];
%*set input data suffix
input_suffix = '_preproc_hp0.1notch_seg_autochannelz3_trialz3_ICAz3_baselinelp125avgref'; %z3 contains visICA

%set outputs
output_suffix = ['_SubjectDecoding_',validation_type,'_',classify_cond,'_',corr_suffix,'_',pseu_suffix];
output_dir = [baseDir,'results/DynamicDecoding/RawSensor/Subj_TempGen/',classify_cond,'/'];
if ~exist(output_dir,'dir')
    mkdir(output_dir);
end

%% Start subject loop

for i = 1:length(subjects)
    
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
    %find number of trials per condition
    cond_trl_num = [];
    if strcmp(classify_cond,'RespMotorHand') == 1
        cond_targets = cond_info(:,5);
        cond_targets(cond_targets==1)=1;
        cond_targets(cond_targets==2)=1;
        cond_trl_num = [cond_trl_num; sum(cond_targets==1)];
        cond_targets(cond_targets==3)=2;
        cond_targets(cond_targets==4)=2;
        cond_trl_num = [cond_trl_num; sum(cond_targets==2)];
    elseif strcmp(classify_cond,'RespMotorFinger') == 1
        cond_targets = cond_info(:,5); %need 4 classifications for this variable
        cond_trl_num = [cond_trl_num; sum(cond_targets==1)];
        cond_trl_num = [cond_trl_num; sum(cond_targets==2)];
        cond_trl_num = [cond_trl_num; sum(cond_targets==3)];
        cond_trl_num = [cond_trl_num; sum(cond_targets==4)];
    elseif strcmp(classify_cond,'RespIndexMiddle')
        cond_targets = cond_info(:,5);
        cond_targets(cond_targets==1)=1;
        cond_targets(cond_targets==3)=1;
        cond_trl_num = [cond_trl_num; sum(cond_targets==1)];
        cond_targets(cond_targets==2)=2;
        cond_targets(cond_targets==4)=2;
        cond_trl_num = [cond_trl_num; sum(cond_targets==2)];
    end
    numTrls = fullfile([output_dir,num2str(subjects(i)),'_SubjectDecoding_TrialCountPerCond.mat']);
    save(numTrls);
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
    %cond1
    cond1=find(cond_targets==1);
    cond1_pseudotrials=floor([length(cond1)/avg_pseudotrials]);
    cond1_extra=rem(length(cond1),avg_pseudotrials); %if this is nonzero then will need to add another pseudotrial with reps=cond1_extratrial
    cond1_inds=(1:cond1_pseudotrials)';
    cond1_inds=repmat(cond1_inds,[avg_pseudotrials,1]);
    % %add extratrial inds, if this exists
    % cond1_inds=[cond1_inds;repmat([cond1_pseudotrials+1],[cond1_extra,1])];
    %cond1=[cond1,cond1_inds];
    %cond2
    cond2=find(cond_targets==2);
    cond2_pseudotrials=floor([length(cond2)/avg_pseudotrials]);
    cond2_extra=rem(length(cond2),avg_pseudotrials); %if this is nonzero then will need to add another pseudotrial with reps=cond1_extratrial
    cond2_inds=(1:cond2_pseudotrials)';
    cond2_inds=repmat(cond2_inds,[avg_pseudotrials,1]);
    % %add extratrial inds, if this exists
    % cond2_inds=[cond2_inds;repmat([cond2_pseudotrials+1],[cond2_extra,1])];
    %cond2=[cond2,cond2_inds];
    
    %Pick out data_trl.trial (trial x channels x timepoints), create
    %pseudotrial indicies, average and replace data_trl.trial;; then run 
    %cosmomvpa setup on data_trl and see if this works...
    accuracy_over_folds=[]; %initilize var storing decoding accuracy across pseudofolds
    fold_structs={}; %var storing for struct output after each decoding fold - useful for storing time, feature info for use in group script
    
    %loop through pseudo folds
    for xx=1:run_pseudotrials
        tic;
        %Randperm inds over required number of folds (in run_pseudotrials)
        xx_inds_cond1=cond1_inds(randperm(length(cond1_inds)));
        xx_inds_cond2=cond2_inds(randperm(length(cond2_inds)));
        
        %Average data in sub_trls - pick out inds in cond1 based on
        %randperm above, which sort trials in data_trl.trial for
        %averaging...
        cond1_trls = [];
        for ii=1:1:max(cond1_inds)
            mean_ii=nanmean(data_trl.trial(cond1(find(xx_inds_cond1==ii)),:,:),1);
            cond1_trls=[cond1_trls;mean_ii];
        end
        
        %cond2
        cond2_trls = [];
        for ii=1:1:max(cond2_inds)
            mean_ii=nanmean(data_trl.trial(cond2(find(xx_inds_cond2==ii)),:,:),1);
            cond2_trls=[cond2_trls;mean_ii];
        end
        
        %replace data_trl.trial with required trials...
        data_trl_pseudo=data_trl;
        data_trl_pseudo.trial=[cond1_trls;cond2_trls];
        
        %will also need to write new cond_targets
        cond_targets_pseudo = [repmat(1,[max(cond1_inds),1]);repmat(2,[max(cond2_inds),1])];
    
        
    
        %% Set up Dynamic Decoding via cosmomvpa

        %Convert fieldtrip struct to cosmomvpa format
        ds_tl = cosmo_meeg_dataset(data_trl_pseudo); %ds_tl is the data struct required by cosmomvpa

        % set the target (trial conditions to be classified)
        ds_tl.sa.targets=cond_targets_pseudo;

        % set the chunks (independent measurements - individual trials considered independent in M/EEG
        ds_tl.sa.chunks=[1:length(cond_targets_pseudo)]';

        % just to check everything is ok
        cosmo_check_dataset(ds_tl);

        %Print number of channels, time points and trials
        fprintf('There are %d channels, %d time points and %d trials\n',...
            numel(unique(ds_tl.fa.chan)),numel(unique(ds_tl.fa.time)),...
            size(ds_tl.samples,1));

        %% Run dynamic decoding MVPA with TempGen

        %1. SETUP TIMEPOINT BY TIMEPOINT CLASSIFICATION
        %This segments the ds_tl.samples (trials x (timepoints x channels)) into time_nbrhood.neighbors
        %radius = 0 performs classification on each timepoint (rather than over a sliding/searchlight window)
        %time_nbrhood=cosmo_interval_neighborhood(ds_tl,'time','radius',0);

        %2. SET UP PARTITION SCHEME
        %temp generalization
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

        %3. RUN DYNAMIC DECODING WITH TEMPORAL GENERALIZATION
        %Loop over folds to assign train/test
        %trials based on 'chunk' call
        for nn=1:length(partitions.test_indices)
            %pick out test and train trials
            nn_test = partitions.test_indices{nn};
            nn_train = partitions.train_indices{nn};

            %set 'chunks: 1=train, 2=test
            ds_nn=ds_tl;
            ds_nn.sa.chunks(nn_train)=1;
            ds_nn.sa.chunks(nn_test)=2;

            %*insert check to remove extra trial (needed when there are
            %different numbers of trials in each condition)
            extra_trl=find(ds_nn.sa.chunks>2);
            if~isempty(extra_trl)
                %remove extra trl from all ds_nn structs
                ds_nn.samples(extra_trl,:)=[];
                ds_nn.sa.targets(extra_trl) = [];
                ds_nn.sa.chunks(extra_trl)=[];
            end

            %set up cosm
            measure_args=struct();
            measure_args.classifier=@cosmo_classify_libsvm;
            ds_time=cosmo_dim_transpose(ds_nn,'time',1);
            cosmo_disp(ds_time);
            measure_args.dimension='time';
            measure_args.measure=@cosmo_crossvalidation_measure;
            cdt_ds = cosmo_dim_generalization_measure(ds_time, measure_args);
            %unflatten to t_point x t_point matrix in data
            [data,labels,values]=cosmo_unflatten(cdt_ds,1);

            %add to counters
            %add to fold x pseudotrial counter
            accuracy_over_folds(:,:,nn,xx)=data; %initilize var storing decoding accuracy across pseudofoldss
            toc;
            disp(['time taken to compute tempGen matrix for fold',num2str(nn),' ptrial',num2str(xx),' subject',num2str(subjects(i)),nfolds '= ',num2str(toc)]);
        end
        
        %add to overall accuracy vars
        %accuracy_over_folds=[accuracy_over_folds;d_acc.samples]; %initilize var storing decoding accuracy across pseudofolds
        %fold_structs=[fold_structs;d_acc];
    
    end
    
   %% visualize
    %decoding_accuracy = nanmean(accuracy_over_folds,1); %compute mean accuracy for this subject across pseudo folds
    
    %Plot timecourse
    figure();
    imagesc(data);
    nticks=10;
    ytick=round(linspace(1, numel(values{1}), nticks));
    ylabel(strrep(labels{1}, '_',' '));
    set(gca, 'Ytick', ytick, 'YTickLabel',values{1}(ytick));

    xtick = round(linspace(1, numel(values{2}), nticks));
    xlabel(strrep(labels{2},'_',' '));
    tpoints=values{1};
    fullTitle = fullfile([classify_cond ' TG Decoding accuracy of Subject ' subjNum]);
    title(fullTitle)

    save([output_dir,num2str(subjects(i)),output_suffix,'.mat'],'accuracy_over_folds','tpoints','-v6.3)');

    %ylim([0 1])
    %print([output_dir,num2str(subjects(i)),output_suffix],'-dbmp','-r300'); %save figure

   %% Save subject output
    %save([output_dir,num2str(subjects(i)),output_suffix,'.mat'],'decoding_accuracy','accuracy_over_folds','fold_structs','-v7.3');
                                      
end