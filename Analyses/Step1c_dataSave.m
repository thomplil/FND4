function Step1c_dataSave(Subjects)
%Change how data is saved
%loop through subjects
%Run all analyses for 5 subjects, and then delete those subjects and run it for the next five
%Only keep one subject of data (have 1,2,3,5,10)
Subjects=[1];
addpath /projectsn/f_mc1689_1/cpro2_eeg/docs/toolboxes/fieldtrip-20240704;
addpath /projectsn/f_mc1689_1/baby_actflow/docs/toolboxes/eeglab13_6_5b/;
ft_defaults;
%**set filter pass bands
filter_hp=1;
filter_lp=50;
for subj=1:length(Subjects)
    %load in data
    inputDir = '/projectsn/f_mc1689_1/cpro2_eeg/data/results/preproc1_causalFilter/';
    inputFile = fullfile(['sub',num2str(Subjects(subj)),'_preproc_hp0.1notch_seg_autochannelz3_trialz3_ICAz3_baselinelp125avgref.mat']);
    subjDataFile = fullfile([inputDir,inputFile]);
    load(subjDataFile);
    %specifically save a probes data file and push that into cfg.dataset
    probes_old = data_preproc_out.probes;
    %Save trialinfo data
    trialinfo_old = probes_old.trialinfo;
    %Reorganize trialinfo so it can be read in python
    trialinfo_old.resp(strcmp(trialinfo_old.resp, 'k'))={4};
    trialinfo_old.resp(strcmp(trialinfo_old.resp, 'j'))={3};
    trialinfo_old.resp(strcmp(trialinfo_old.resp, 'f'))={2};
    trialinfo_old.resp(strcmp(trialinfo_old.resp, 'd'))={1};
    trialinfo_old.LogicRule(strcmp(trialinfo_old.LogicRule, '*EITHER**'))={1};
    trialinfo_old.LogicRule(strcmp(trialinfo_old.LogicRule, '*NEITHER*'))={1};
    trialinfo_old.LogicRule(strcmp(trialinfo_old.LogicRule, '**SAME***'))={2};
    trialinfo_old.LogicRule(strcmp(trialinfo_old.LogicRule, 'DIFFERENT'))={2};
    trialinfo = trialinfo_old(:, {'TaskCode','LogicRule','acc','resp','rt'});
    %Make all info a numeric array
    if istable(probes_old.trialinfo)
        vars = varfun(@isnumeric, probes_old.trialinfo, 'OutputFormat', 'uniform');
        probes_old.trialinfo = table2array(probes_old.trialinfo(:, vars));
    end
    %Bandpass from 1 to 50
    tic
    cfg = [];
    %hp -- make configuration to store paramaters and path to data
    cfg.hpfilter='yes';
    cfg.hpfreq=filter_hp;
    cfg.hpfilttype='firws';
    cfg.hpfiltdir='onepass-minphase';
    cfg.lpfilter = 'yes';
    cfg.lpfreq = filter_lp;
    cfg.lpfilttype='firws';
    cfg.lpfiltdir='onepass-minphase';
    probes = ft_preprocessing(cfg, probes_old);%actually does the filtering
    toc
    fprintf('Filtering the continuous data took %.2f seconds\n', toc);

    cfg = [];
    cfg.resamplefs = 125;
    cfg.resamplemethod='downsample'; %this skips the lp filter
    cfg.detrend = 'no';
    cfg.demean = 'no';
    probes = ft_resampledata(cfg,probes);
    %Name individual files to save
    fsample = probes.fsample;
    elec = probes.elec;
    hdr = probes.hdr;
    trial = probes.trial;
    time = probes.time;
    label = probes.label;
    cfg = probes.cfg;
    
    
    %List of file names to be saved
    fileNames = {'fsample','elec','hdr','trialinfo','trial','time','label','cfg'};
    outputDir = fullfile(['/projectsn/f_mc1689_1/cpro2_eeg/data/results/preproc1_causalFilter/sub',num2str(Subjects(subj)),'/']);
    if ~exist(outputDir,'dir')
        mkdir(outputDir);
    end
    for i=1:length(fileNames)
        fileSaving = fullfile([fileNames{i} '_hp0.1notch_seg_autochannelz3_trialz3_ICAz3_baselinelp125avgref.mat']);
        save([outputDir, fileSaving],fileNames{i},'-v7.3');
    end
end
end
