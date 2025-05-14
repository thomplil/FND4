function Step1c_dataSave(Subjects)
%Change how data is saved
%loop through subjects
%Run all analyses for 5 subjects, and then delete those subjects and run it for the next five
%Only keep one subject of data (have 1,2,3,5,10)
for subj=1:length(Subjects)
    %load in data
    inputDir = '/cache/home/let83/FND4/results/preproc1_causalFilter/';
    inputFile = fullfile(['sub',num2str(Subjects(subj)),'_preproc_hp0.1notch_seg_autochannelz3_trialz3_ICAz3_baselinelp125avgref.mat']);
    subjDataFile = fullfile([inputDir,inputFile]);
    load(subjDataFile);
    probes = data_preproc_out.probes;
    
    %Name individual files to save
    fsample = probes.fsample;
    elec = probes.elec;
    hdr = probes.hdr;
    trialinfo_old = probes.trialinfo;
    sampleinfo = probes.sampleinfo;
    trial = probes.trial;
    time = probes.time;
    label = probes.label;
    cfg = probes.cfg;
    
    %Reorganize trialinfo so it can be read in python
    trialinfo_old.resp(strcmp(trialinfo_old.resp, 'k'))={4};
    trialinfo_old.resp(strcmp(trialinfo_old.resp, 'j'))={3};
    trialinfo_old.resp(strcmp(trialinfo_old.resp, 'f'))={2};
    trialinfo_old.resp(strcmp(trialinfo_old.resp, 'd'))={1};
    trialinfo = trialinfo_old(:, {'TaskCode','acc','resp','rt'});
    
    %List of file names to be saved
    fileNames = {'fsample','elec','hdr','trialinfo','sampleinfo','trial','time','label','cfg'};
    outputDir = fullfile(['/cache/home/let83/FND4/results/preproc1_causalFilter/sub',num2str(Subjects(subj)),'/']);
    if ~exist(outputDir,'dir')
        mkdir(outputDir);
    end
    for i=1:length(fileNames)
        fileSaving = fullfile([fileNames{i} '_hp0.1notch_seg_autochannelz3_trialz3_ICAz3_baselinelp125avgref.mat']);
        save([outputDir, fileSaving],fileNames{i},'-v7.3');
    end
end
end