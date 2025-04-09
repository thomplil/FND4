function Step3_SourceModeling_bandLCMVbeamform_EGI_v5(subjects,input_type)

%Author: Ravi Mill, rdm146@rutgers.edu
%Last update: Aug 5th 2020

%DESCRIPTION: source models EEG data (task or rest) for Dynamic Activity
%Flow project: Mill, XX, "Title"

%Performs LCMV (time-domain) beamforming on EGI data
%1. Construct FEM head model (for realistic modeling of distortion of
    %electric fields through skull tissue) from subject T1 MRI.
    %Converts dicom T1 images to nifti (using
    %dicm2nii) via shell command
%2. Set up source model (source locations embedded in head model + sensors): 
    %uses Power et al 2011 atlas regions (264 source regions total)
%3. Compute leadfields and apply beamforming (in one step to save time) and save results; 
    %Optionally can reduce epoch length prior to this (as defined by epoch_length in parms below)
    %Note that for the task data, beamforming is done for stim-locked data
        %over -0.5 to 1.5 (this wider segmentation allows for later resp-locking as part
        %of the dynamic decoding script)
    %Can optionally apply bandpass filtering prior to this step, and/or
        %vary lamda parameter
    %Removes 'bad' channels removed during preproc from
        %the sensor array ('sens') used in the source  model

%INPUTS
%subjects=numeric nx1 array, each row containing subject IDs 
%input_type=numeric, whether 1= 'task' or 2='rest' data is to be source
    %modeled for this subject (having this as an extra parameter allows for
    %greater batched parallelization)
    
%OUTPUTS
%Writes source modeled timeseries

%% Set path and defaults for Fieldtrip
addpath /projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/;
ft_defaults;

%% Set path to SPM8 - required for MRI segmentation
addpath /projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/external/spm8/;

%% Set path to dicom converter
addpath /projects/f_mc1689_1/CPRO2_learning/docs/scripts/colelablinux_scripts/dicm2nii/

%% Set paths to input data

%*Set inputs for pre-processed data
input_dir='/projects/f_mc1689_1/DynamicSensoryMotorEGI/data/results/Analysis3_causalFilter/';

%path to egi electrode layout (in Fieldtrip templates directory)
elecfile='/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/template/electrode/GSN-HydroCel-257.sfp';

%Load power atlas coords - sorted into functional networks (to help visualisation later)
load('/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/scripts/power_atlas_coords_sortedByNetwork.mat'); %first 3 cols contain xyz in 'power_atlas_coords_sorted' (cell)

%*Set input suffix - refer to preproc for parms to enter here
input_suffix='_preproc_hp1notch_seg_autochannelz3_trialz3_ICAz3_baselinelp50avgref'; %z3 contains visICA

%Set path to subject MRIs and MRI template (used to create MNI source grid)
MRI_dir='/projects/f_mc1689_1/DynamicSensoryMotorEGI/data/rawdata/';
MRI_input='/MRI/T1.nii.gz'; %NIFTI not normalised, with skull

%now setting MNI_template_sourcemodel to regular 4mmgrid MNI created by Fieldtrip users
ftrip_MNI_template='/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/toolboxes/fieldtrip-20170716/template/sourcemodel/standard_sourcemodel3d4mm.mat'; %wholebrain grid (used when modeling wholebrain grid and then parcellating)


%% Set analysis parameters

%list of all subjects
% total_subjects = [803;804;805;806;807;809;810;811;813;...
%     814;815;817;818;819;820;821;822;823;824;825;826;827;...
%     828;829;831;832;833;834;835;836;837;838];

%This is redundant - varying this only works when creating a regularly spaced grid (but now I'm either using
%the Power template or a whole-brain grid created by fieldtrip); kept it in for
%ease i.e. not having to redo the output names
inwardshift_parm=0; %degree to shift threshold of brain interior; negative value = greater portion of MRI is considered exterior (i.e. outward shift of brain boundary)

%channel parms
channel_num='E*'; %excludes VREF from source model

%set source coordinates - from Power mat; already created this in
%fieldtrip, so now just loading in the result of that
%overall_ROI.coords=cell2mat(power_atlas_coords_sorted(:,2:4));
%[num_sources,~]=size(overall_ROI.coords);

%beamformer parms
leadfield_normalize='yes'; 
sourceanalysis_projectnoise='no';
sourceanalysis_lambda={'5%'};

%*set frequency bands to beamform in
input_freqs=[1 50]; %freqbands after preproc
output_freqs=[1 50]; %desired freqbands after preproc; *if different from input_freqs, then bandpass filtering is applied prior to source modeling

%*Set whether to demean_detrend prior to source analysis based on freq_bands
%1=demean each trial; 0=do not demean
%Note, detrending is potentially helpful for band-limited analyses (it is not useful for 
%broadband ERP analyses as it attenutates ERPs), but it is no longer the 
%default in fieldtrip (so it is excluded here)
demean_detrend=0;

%set whether to compute hilbert transform - not planning this for this project
%hilbert_apply=0;

%*automated way to set demean_detrend if cycling through frequency bands
% if freq_bands(1,1) == 1 && freq_bands(1,2) == 100 %broadband data
%     demean_detrend = 0; %no trial demeaning
%     hilbert_apply = 0;
% else
%     demean_detrend = 1; %perform trial demeaning prior to source analysis
%     hilbert_apply = 1;
% end

%*optionally change set epoch length prior to beamforming
epoch_length=[]; %empty = stick with preproc segments (-0.5 to 1.5); 

%*set whether to run source model (if 0 then only runs/loads head model)
run_source_model=1;

%*set what kind of source grid is to be used
source_grid='Power'; %Whole=whole-brain 4mm grid, Power=264 power MNI regions


%% Set output paths and variables

%create source output directories
output_dir=[input_dir,'SourceModel',input_suffix,'/'];
if ~exist(output_dir,'dir')
    mkdir(output_dir);
end
output_source_trial_prefix = 'SourceTrialbyTrial_';

%create head model output directories - path has to point to previously created head models
MRI_suffix = 'MRIheadmodel';
fieldtrip_processMRI='/projects/f_mc1689_1/DynamicSensoryMotorEGI/data/results/Analysis2_DynamicDecoding_reproc/Stim_locked/Fieldtrip_headmodels_MRI/';
if ~exist(fieldtrip_processMRI,'dir')
    mkdir(fieldtrip_processMRI);
end



%% Start subject loop

for i = 1:length(subjects)
    tic;
    
    subject_data_file = [input_dir,num2str(subjects(i,1)),input_suffix,'.mat'];
    disp(['Loading ',subject_data_file]);
    load(subject_data_file);
    
    %*assign data input based on input_type
    if input_type==1
        data_final=task_data_seg_channels_trials_autovisIC;
        input_name='task'; %used to write output
    elseif input_type==2
        data_final=rest_data_seg_channels_trials_autovisIC;
        input_name='rest';
    end
    %clear vars to free up memory
    clear task_data_seg_channels_trials_autovisIC rest_data_seg_channels_trials_autovisIC;
    

    %% Create Head Model from subject MRI 
    
   %% Align mri to EEG coordinate system - after checking if the alignment has already been saved
    %MRI and EEG needs to be in same coordinate space for source localisation
    
    MRI_aligned = [fieldtrip_processMRI,num2str(subjects(i,1)),'_',MRI_suffix,'_alignedtoCTF.mat'];
    if ~exist(MRI_aligned)
        %Specify subject mri path and read into fieldtrip format
        subject_mri = [MRI_dir,num2str(subjects(i,1)),MRI_input];

        %Check if T1 has been converted to nifti, otherwise run the dicom
        %conversion from here...
        if ~exist(subject_mri)
            %use dicm2nii to convert T1 to nifti
            outMRI=[MRI_dir,num2str(subjects(i,1)),'/MRI/'];
            rawMRI=[outMRI,'/MR t1_mpr_ns_sag_p2_iso_32Channel/'];
            dicm2nii(rawMRI,outMRI); %inputdir,outputdir
            %rename output to T1.nii.gz
            movefile([outMRI,'t1_mpr_ns_sag_p2_iso_32Channel.nii.gz'],subject_mri);
            movefile([outMRI,'t1_mpr_ns_sag_p2_iso_32Channel.json'],[outMRI,'T1.json']);
        end
        
        %load mri into fieldtrip
        input_mri = ft_read_mri(subject_mri);
    
        %Reslice MRI image - corrects up-down visualisation
        cfg = [];
        resliced_mri = ft_volumereslice(cfg,input_mri);

        %Convert to EEG sensor units - *important to do this before realigning given the later coord transform!
        resliced_mri = ft_convert_units(resliced_mri,'cm');

        %Mark nasion ('n'), left ('l') and right ('r') pre-auricular points (LPA/RPA) manually
        %'c' to toggle marked fiducial visibility, 'q' to finish
        cfg = [];
        cfg.method = 'interactive';
        cfg.coordsys = 'ctf'; %http://www.fieldtriptoolbox.org/faq/how_are_the_different_head_and_mri_coordinate_systems_defined/
        aligned_mri = ft_volumerealign(cfg,resliced_mri);
                
        save(MRI_aligned,'aligned_mri');
    else
        load(MRI_aligned);
        
    end
    
   %% Segment aligned_mri
    %after checking if the segmentation has already been done
    
    MRI_segmented = [fieldtrip_processMRI,num2str(subjects(i,1)),'_',MRI_suffix,'_alignedtoCTF_segmented.mat']; %segment the MRI aligned in Neuromag space
    %Neuromag_MRI_segmented = [fieldtrip_processMRI,num2str(subjects(i,1)),'_',MRI_label,'_alignedtoNIFTI_segmented.mat']; %segment the MRI still in Talairach space
    if ~exist(MRI_segmented)
    
        cfg = [];
        %need to segment into tissue types for FEM model
        cfg.output = {'gray','white','csf','skull','scalp'}; %{'brain','skull','scalp'}; %'brain' = binary representation of the brain (collapsing gm, wm and csf); 'tpm' = tissue probability map, probabilistic representation of gm, wm and csf (only required for more complex methods e.g. BEM, FEM)
        %cfg.write = 'no'; %do not write SPM format output 
        %cfg.downsample = 2;
        segmented_mri = ft_volumesegment(cfg,aligned_mri);
        save(MRI_segmented,'segmented_mri');
    else
        load(MRI_segmented);
    end
    
   %% Create FEM mesh
    %hexahedral representation of each segmented tissue type
    
    MRI_mesh = [fieldtrip_processMRI,num2str(subjects(i,1)),'_',MRI_suffix,'_alignedtoCTF_segmented_mesh.mat']; %segment the MRI aligned in Neuromag space
    if ~exist(MRI_mesh)
        cfg = [];
        cfg.shift = 0.3; %**range 0-0.49, shift parameter controlling boundaries between tissue types; default = 0.3
        cfg.method = 'hexahedral';
        mesh = ft_prepare_mesh(cfg,segmented_mri);

        %Create final headmodel
        cfg = [];
        cfg.method ='simbio'; %FEM method
        cfg.conductivity = [0.33 0.14 1.79 0.01 0.43];   %order follows mesh.tissuelabel; *note these values can be individualized for each subject if DWI data is collected; as such these are defaults based on post-mortem electrodes timulation studies
        vol = ft_prepare_headmodel(cfg, mesh);
        
        %save
        save(MRI_mesh,'mesh','vol');
    else
        load(MRI_mesh);
    end
    
    
   %% Align Electrodes to Mesh
    %*note that this has to be redone to reflect updated bad channels for
    %Analysis3; hence the previous manually adjusted elec alignment is
    %loaded in (Analysis2), with the new bad_channels (from Analysis3 with the new filters)
    %are removed
    
    % read in electrode coordinates file
    elec = ft_read_sens(elecfile); 
    
    %analysis2 - elec alignments
    elec_aligned_output_orig = [fieldtrip_processMRI,num2str(subjects(i,1)),'_',MRI_suffix,'_electrodesAlignedtoMRI.mat']; 
    %analysis3 - new bad_elecs
    elec_aligned_output = [fieldtrip_processMRI,num2str(subjects(i,1)),'_',MRI_suffix,'_electrodesAlignedtoMRI_analysis3.mat']; 
    if ~exist(elec_aligned_output)
    
        %*skip this
        % visualize electrodes on head mesh - see if they need aligning to the MRI mesh (**YES, INVARIABLY THEY DO)
%         figure
%         hold on
%         ft_plot_mesh(mesh,'surfaceonly','yes','vertexcolor','none','edgecolor','none','facecolor',[0.5 0.5 0.5],'face alpha',0.7)
%         camlight
%         ft_plot_sens(elec,'style', 'sr'); 
        
        if exist(elec_aligned_output_orig)
            %load in aligned elecs
            load(elec_aligned_output_orig);
        else

            %1. align mri fiducials to eeg coordinates (head system)
            %take fiducials from aligned_mri
            nas=aligned_mri.cfg.fiducial.nas;
            lpa=aligned_mri.cfg.fiducial.lpa;
            rpa=aligned_mri.cfg.fiducial.rpa;

            vox2head=aligned_mri.transform;

            nas=ft_warp_apply(vox2head, nas, 'homogenous');
            lpa=ft_warp_apply(vox2head, lpa, 'homogenous');
            rpa=ft_warp_apply(vox2head, rpa, 'homogenous');

            %2. Determine the transformation needed to get the position of the fiducials 
            %in the electrode structure (first three channels are nasion, LPA and RPA) to their counterparts 
            %in the CTF head coordinate system that we acquired from the anatomical mri (nas, lpa, rpa)

            % create a structure similar to a template set of electrodes
            fid.chanpos=[nas; lpa; rpa]; % CTF head coordinates of fiducials
            fid.elecpos=[nas; lpa; rpa]; 
            fid.label={'FidNz','FidT9','FidT10'}; % use the same labels as those in elec 
            fid.unit='cm'; % use the same units as those in mri

            % alignment
            cfg=[];
            cfg.method='fiducial';            
            cfg.elec=elec; % the electrodes we want to align
            %cfg.elecfile=elecfile;
            cfg.template=fid;                   % the template we want to align to
            cfg.fiducial={'FidNz', 'FidT9', 'FidT10'};  % labels of fiducials in fid and in elec
            elec_aligned=ft_electroderealign(cfg);

    %         %3. Check alignment by plotting mesh with overlaid sens
    %         figure;
    %         hold on;
    %         ft_plot_mesh(mesh,'surfaceonly','yes','vertexcolor','none','edgecolor','none','facecolor',[0.5 0.5 0.5],'face alpha',0.5)
    %         camlight
    %         ft_plot_sens(elec_aligned,'style','sr');


            %4. Improve alignment manually (often channels will end up inside the
            %brain, this is due to differences in fiducial definition conventions)
            cfg=[];
            cfg.method='interactive';
            cfg.elec=elec_aligned;
            cfg.headshape=vol;
            elec_aligned=ft_electroderealign(cfg);
        end
        
        %5. ***Remove bad channels from elec_aligned
        %set bad_channels based on input_type
        if input_type==1
            bad_channels=sub_bad_channels{1};
        elseif input_type==2
            bad_channels=sub_bad_channels{2};
        end
        num_bad=length(bad_channels);
        bad_channels_fids=bad_channels+3; %add 3 to indices, because first 3 channels are fiducials
        elec_aligned_removedBad=elec_aligned;
        
        %remove bad channels
        elec_aligned_removedBad.chanpos(bad_channels_fids,:)=[];
        elec_aligned_removedBad.chantype(bad_channels_fids)=[];
        elec_aligned_removedBad.chanunit(bad_channels_fids)=[];
        elec_aligned_removedBad.elecpos(bad_channels_fids,:)=[];
        elec_aligned_removedBad.label(bad_channels_fids)=[];

        %Save
        save(elec_aligned_output,'elec_aligned','elec_aligned_removedBad');
        
    else
        load(elec_aligned_output);
    end
    

   %% Set dipole location based on MNI transformations
    %Create template dipole grid based on template MRI image in MNI coordinates
    %Check if template source model has already been made
    
    %Load in appropriate source location template
    if strcmp(source_grid,'Whole')
        %use  fieldtrip's 4mm whole-brain MNI grid 
        MNI_template_sourcemodel=ftrip_MNI_template;
        load(MNI_template_sourcemodel);
        template_grid=sourcemodel;
        clear sourcemodel

        %Warp MNI grid to subject head model (in subject space i.e. anatomical MRI
        %warped to Neuromag)
        subject_grid = [fieldtrip_processMRI,num2str(subjects(i,1)),'_',MRI_suffix,'_gridWhole.mat']; 

        %Note: seems like Fieldtrip orients source coordinates in grid.pos as LPI 
        %even if the template/anatomical MRI is RPI, based on the
        %.transform and coordsys info; critically, the grid.pos info needs to be entered as LPI
        %and hence is currently comparable to the Power atlas coords (also defined as LPI)
        %see here for more info http://www.fieldtriptoolbox.org/faq/what_is_the_plotting_convention_for_anatomical_mris
        if ~exist(subject_grid)
            cfg = [];
            cfg.grid.warpmni = 'yes';
            cfg.grid.template = template_grid; %Note that even though template_grid is in mm, this step yields a grid in cm (that is the same as if we had converted units of template_grid to cm before)
            cfg.grid.nonlinear = 'no'; %*works way better with nonlinear transform set to no!!
            cfg.mri = aligned_mri;
            %cfg.inwardshift = inwardshift_parm;
            grid = ft_prepare_sourcemodel(cfg);
            %try out nonlinear warp
    %         cfg.grid.nonlinear='yes';
    %         grid_nonlinear=ft_prepare_sourcemodel(cfg);

            %Visualise results - too memory intensive!
    %         figure
    %         hold on
    %         ft_plot_vol(vol,'facecolor','cortex','edgecolor','none','surfaceonly','true');alpha 0.4;camlight;
    %         %ft_plot_mesh(grid.pos);
    %         ft_plot_mesh(grid.pos(grid.inside,:));
    %         ft_plot_sens(elec_aligned_removedBad,'style', 'sr'); 

            %plot grid on its own - less computationally intensive than plotting with vol
            figure; hold on;
            ft_plot_mesh(grid.pos(grid.inside,:));
            ft_plot_sens(elec_aligned_removedBad,'style', 'sr'); 
            %nonlinear version
    %         figure; hold on;
    %         ft_plot_mesh(grid_nonlinear.pos(grid_nonlinear.inside,:));
    %         ft_plot_sens(elec_aligned_removedBad,'style', 'sr');        
        
            %save
            save(subject_grid,'grid');
        else
            load(subject_grid);
        end
    
    elseif strcmp(source_grid,'Power')
        %load in Power 264 source atlas template
        MNI_template_sourcemodel = [fieldtrip_processMRI,'/WholeBrain_',num2str(inwardshift_parm),'inward_MNItemplate_sourcemodel_Power.mat'];
        load(MNI_template_sourcemodel);
        
        subject_grid = [fieldtrip_processMRI,num2str(subjects(i,1)),'_',MRI_suffix,'_grid.mat']; 
        if ~exist(subject_grid)
            cfg=[];
            cfg.grid.warpmni='yes';
            cfg.grid.template=template_grid; %Note that even though template_grid is in mm, this step yields a grid in cm (that is the same as if we had converted units of template_grid to cm before)
            cfg.grid.nonlinear='no'; %*works better with nonlinear transform set to no!!
            cfg.mri=aligned_mri;
            cfg.inwardshift=inwardshift_parm;
            grid=ft_prepare_sourcemodel(cfg);

            %Visualise results
%             figure
%             hold on
%             ft_plot_vol(vol,'facecolor','cortex','edgecolor','none');alpha 0.4;camlight;
%             ft_plot_mesh(grid.pos);
%             ft_plot_sens(elec_aligned_removedBad,'style', 'sr'); 

            %plot grid on its own - less computationally intensive than plotting with vol
            figure; hold on;
            ft_plot_mesh(grid.pos);
            ft_plot_sens(elec_aligned_removedBad,'style', 'sr'); 

            %save
            save(subject_grid,'grid');
        else
            load(subject_grid);

        end

    end
    
    
   %% Inverse model sources using LCMV beamformer   
    
    %whether to run the source model...
    if run_source_model==1
    
        %*Loop through frequency bands (code is setup to source model
        %multipl variations in freqbands, demean/detrend, lambda values)
        [num_bands,~] = size(output_freqs);

        for f=1:num_bands
            current_freq = output_freqs(f,:);

            for demean=1:length(demean_detrend)
                for lamb=1:length(sourceanalysis_lambda)

                    % Assign output suffix based on demean and lamb
                    output_suffix = [];
                    if demean_detrend(demean,1)==1
                        output_suffix='Demean';
                    end

                    if sourceanalysis_lambda{lamb,1}==0
                        output_suffix = [output_suffix,'lambda0'];
                    elseif strcmp(sourceanalysis_lambda{lamb,1},'1%')==1
                        output_suffix = [output_suffix,'lambda1percent'];
                    elseif strcmp(sourceanalysis_lambda{lamb,1},'5%')==1
                        output_suffix = [output_suffix,'lambda5percent'];
                    end


                %% 1. Source preproc
                    %Perform bandpass + redefine trial epoch_length if
                    %either is set at the top of the script

                    if current_freq(1)~=input_freqs(1) || current_freq(2)~=input_freqs(2) 
                        disp('Running bandpass filter');
                        cfg = [];
                        cfg.bpfilter = 'yes';
                        cfg.bpfilttype='firws';
                        cfg.bpfiltdir='onepass-minphase';
                        %change filtord for delta freq, given instablity with default
                        if current_freq(1,1)==0.5
                            cfg.bpfiltord = 3; %increased from default 4 to accommodate delta low freq band
                        end
                        cfg.bpfreq = current_freq; %bandpass filter into current frequency band

                        %Set whether demeaning+detrending is conducted
                        if demean_detrend(demean,1)==1
                            cfg.demean = 'yes';
                            cfg.baselinewindow = 'all';
                            %cfg.detrend = 'yes'; %not detrending anymore as fieldtrip discourages this
                        end

                        %Apply
                        data_final = ft_preprocessing(cfg,data_final);
                    end

                    if ~isempty(epoch_length) %data epochs require redefining
                        disp('Redefining epoch length');
                        cfg = [];
                        cfg.toilim = epoch_length;
                        data_final = ft_redefinetrial(cfg,data_final);
                    end

                %% 2. Run through beamformer covariance computation - Task + Rest common filter

                    %2a. Compute sensor covariance
                    cfg = [];
                    cfg.covariance = 'yes';
                    cfg.covariancewindow = 'all'; %[x y] = covariance computed within particular epochs within each trial  
                    cfg.keeptrials = 'no';
                    cfg.vartrllength = 2; %2 = use all available samples in each trial, in case trials are of variable length (not the case in this project)
                    data_covariance = ft_timelockanalysis(cfg, data_final);

                    %2b. Call timelockanalysis again to create trial
                    %structure required for subsequent timeseries reconstruction
                    cfg = [];
                    %cfg.covariance = 'yes';
                    %cfg.covariancewindow = 'all';
                    cfg.keeptrials = 'yes';
                    cfg.vartrllength = 2; %2 = use all available samples in each trial
                    task_freqband_avg = ft_timelockanalysis(cfg, data_final);
                    
                    %2c. Compute beamformer filter                  
                    cfg = [];
                    cfg.channel = channel_num;
                    cfg.elec = elec_aligned_removedBad;
                    %cfg.senstype = 'MEG';
                    cfg.method = 'lcmv';
                    cfg.grid = grid;
                    cfg.normalize = leadfield_normalize; %lf being computed during this step, so need to specify this
                    cfg.vol = vol;
                    cfg.rawtrial = 'no'; %no = compute beamformer on average data; 
                    cfg.keepfilter = 'yes'; %**yes = keep filter to apply in next stage to each trial
                    cfg.keeptrials = 'no'; %no = does not keep trials
                    cfg.fixedori = 'yes'; %%*constrains solution to dipole orientation with maximal power; ft_sourcedescriptives projectmom was generating an error, so this accomplishes the same thing
                    cfg.projectnoise = sourceanalysis_projectnoise;
                    cfg.keepleadfields='yes';
                    %*Set lambda value
                    cfg.lambda = sourceanalysis_lambda{lamb,1};
                    
                    %compute beamform filter, if not already saved
                    source_filt=[output_dir,num2str(subjects(i,1)),'_',input_name,'_sourceFilter_withlf',output_suffix,'_',num2str(current_freq(1,1)),'to',num2str(current_freq(1,2)),'Hz.mat'];
                    if ~exist(source_filt)
                        source_data_covariance = ft_sourceanalysis(cfg,data_covariance);

                        %save
                        save(source_filt,'source_data_covariance','-v7.3');
                    else
                        disp('Loading beamformer filter...');
                        load(source_filt);
                    end

                    %2d. Apply beamformer filter computed above to compute
                    %trial-by-trial source timeseries
                    cfg.grid.filter = source_data_covariance.avg.filter;
                    clear source_data_covariance data_covariance data_final; %trying to free up memory
                    cfg.rawtrial = 'yes'; %compute beamformer on raw trials rather than average
                    cfg.keeptrials = 'yes'; %this actually just keeps trialinfo (rawtrial is more important)
                    cfg.keepfilter = 'no';
                    sourcetimeseries_task_commonfilter = ft_sourceanalysis(cfg,task_freqband_avg);   
                    
                    %2e. Save sourcetimeseries (with covariance filter)
                    beam_filter = cfg.grid.filter;
                    if current_freq(1,1)==0.5 %treatment for '0.5' being written for delta output
                        save([output_dir,num2str(subjects(i,1)),'_',input_name,'_',output_source_trial_prefix,output_suffix,'_0pt5to',num2str(current_freq(1,2)),'Hz.mat'],'sourcetimeseries_task_commonfilter','beam_filter','-v7.3');

                    else
                        save([output_dir,num2str(subjects(i,1)),'_',input_name,'_',output_source_trial_prefix,output_suffix,'_',num2str(current_freq(1,1)),'to',num2str(current_freq(1,2)),'Hz.mat'],'sourcetimeseries_task_commonfilter','beam_filter','-v7.3');
                    end

                end
            end

        end
    end
    
    toc;
    disp(['Time to source localize subject',num2str(subjects(i,1)),' = ',num2str(toc)])
    close all;
    
end

end