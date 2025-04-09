%Step0_preproc_sourceModel.m

%R.Mill, Oct 2023

%Runs preprocessing and source modeling of OPM-MEG data release, 
%https://zenodo.org/record/7525341. Intended for OPM-MEG grant...

%NOTES:
%1.*decided to preproc each run separately (consistent with the Rier 2022 paper),
%meaning that each run will have separate bad channels removed, and
%separate sensor covariances (used to construct the beamformer filter);
%both factors could introduce differences between run 1 and 2 (which could
%reduce inter-session reliability/overlap); *but this is simplest approach
%that is unlikely to influence results...
%2. Omitting ICA from this analysis - Time-consuming: 2 runs * 10 subs = 20 ICAs
%but note this should be run for a proper analysis...
%3. Approach to source modeling: specifying 3d volume grid for the source
%model, then averaging the source timeseries into Glasser parcels (using
%the Horn volume version of the atlas); 
%*Alternative: use freesurfer to extract the cortical sheet, allowing use of 
%the fieldtrip surface version of the atlas; *this should be the ultimate approach
%described here https://www.fieldtriptoolbox.org/tutorial/networkanalysis/
%4. Treatment of 3 sensor types:
%i. zeroing out cross-terms in covariance matrix https://mailman.science.ru.nl/pipermail/fieldtrip/2017-May/037418.html
%ii. emphasis on units for all structures, and presence of grad.tra element, which
%   codes for scaling between different sensor types (*not output in the
%   created fieldtrip structure*); https://mailman.science.ru.nl/pipermail/fieldtrip/2017-May/037417.html,
%   see also definition of .tra here https://www.fieldtriptoolbox.org/faq/how_are_electrodes_magnetometers_or_gradiometers_described/
%iii. concatenate leadfields separately estimated for each sensor type,
%   with some outstanding problems highlighted https://mailman.science.ru.nl/pipermail/fieldtrip/2013-February/019045.html
%iv. Conversion code in Rier paper that might speak to this issue, see: 
%   "convert orientation of sources to polar"
%CURRENT APPROACH: just entering all sensor types as is into source model, 
%without scaling
%5. Fnirt registration worked poorly for sub6 and sub 10; for now just
%using the flirt result for those subjects (which worked ok); 
%in future look at variations to mprove fnirt 
%e.g. https://neurostars.org/t/fnirt-registration-problem-into-mni-1mm-space/25555/2
%6. *Consider using 10mm grid (currently using 4mm grid) as this was shown
%to be a spacing that excludes majority of field spread artifact (similar
%rationale employed for Power atlas in EGI paper, established in Schoffelen
%& Gross, 2009)

%% Set path and defaults for Fieldtrip

addpath /projectsn/f_mc1689_1/opm_meg_fc/docs/scripts/fieldtrip-20231025; 
ft_defaults;

%% Set paths, parameters

%add path to Rier functions
addpath /projectsn/f_mc1689_1/opm_meg_fc/docs/scripts/Rier2022_functions;

%set subject IDs
%subjects={'sub-001','sub-002','sub-003','sub-004','sub-005','sub-006',...
%    'sub-007','sub-008','sub-009','sub-010'};
%subjects={'sub-004','sub-005','sub-006'};
subjects={'sub-007','sub-008','sub-009','sub-010'};
num_subs=length(subjects);

%set path to data directory
%e.g. data: $datadir/$sub/meg/sub-001_task-movie_run-001_meg.mat
%T1: $datadir/$sub/anat/$sub.nii
%sensor locations: $datadir/$sub/meg/sub-001_task-movie_run-001_channels.tsv_new
%   %% how to extract sensor info:
%S.sensor_info.pos = [ch_table.Px,ch_table.Py,ch_table.Pz];
%S.sensor_info.ors = [ch_table.Ox,ch_table.Oy,ch_table.Oz];
datadir='/projects/f_mc1689_1/opm_meg_fc/data/rawdata/dog_day_afternoon_OPM/';

%2 10min runs of subjects watching the same movie clip were recorded
num_runs=2;

%% preproc parms
%set filter type - use causal filter in case we estimate MVAR
filter_type='causal_filter'; %noncausal_filter, broadband

%set filter bands
%run in 2 stages: hp first, then lp (based on eeglab reccs, see code below)
%*line noise in UK=50Hz, hence bandpass filter set to 1-45Hz
filt_bands=[1 45]; %broadband, excluding 50Hz line noise

%downsample HZ
%orig fs=1200Hz, need to choose downsample as 2*upper freq=90, using 100Hz
%as it allows for clearer presentation of timepoints
down_hz=100; 

%% source modeling parms

%recommended parms from prior source modeling analyses
leadfield_normalize = 'yes'; 
sourceanalysis_projectnoise = 'no';
sourceanalysis_lambda = '5%';
lambda_str='5percent';

%set path to fieldtrip 4mm grid source model
ftrip_MNI_template='/projects/f_mc1689_1/opm_meg_fc/docs/scripts/fieldtrip-20231025/template/sourcemodel/standard_sourcemodel3d4mm.mat';

%path to Fieldtrip's Glasser atlas - can only be used with freesurfer'd
%head model (2d cortical sheet)
%glasser_atlas='/projects/f_mc1689_1/opm_meg_fc/data/results/atlas_MMP1.0_4k.mat';

%load in MRI files - ROI, and MNI template
%path to Horn MMP (volume version of Glasser cortical atlas)
mmp_file='/projects/f_mc1689_1/opm_meg_fc/data/results/MMP_in_MNI_symmetrical_1_rpi_final.nii.gz';

%path to FSL MNI template
mni_file='/projects/f_mc1689_1/opm_meg_fc/data/rawdata/MNI152_T1_2mm.nii.gz';

%% set outputs 

%output directory
out_dir='/projectsn/f_mc1689_1/opm_meg_fc/data/results/grant_nov23/';
if ~exist(out_dir);mkdir(out_dir);end

%output suff
%*add relevant parms from above

%% Loop through subjects: load data, preproc, source model

tic;
for i=1:num_subs
    sub_i=subjects{i};
    
    for n=1:num_runs
        if n==1
            run_str='run-001';
        elseif n==2
            run_str='run-002';
        end
        
        %load data
        %data
        load([datadir,sub_i,'/meg/',sub_i,'_task-movie_',run_str,'_meg.mat']);
        %dat=data;clear data
        %sample_rate=fs;
        
        %anat - segmented mri or mesh?
        %**load in meshes for each subject
        %   $datadir/derivatives/sourcespace/$sub/sub-001_meshes.mat or
        %   sub-001_segmentedmri.mat (for segmented mri)
        %meshes, segmented
        load([datadir,'/derivatives/sourcespace/',sub_i,'/',sub_i,'_meshes.mat']);
        
        %sensor locations
        ch_table = readtable([datadir,sub_i,'/meg/',sub_i,'_task-movie_',run_str,'_channels.tsv_new'],...
            'Delimiter','tab','FileType','text'); 
        
        %ica - *not using prior Rier ICA outputs (would be safer to extract
        %these myself within my pipeline, but I don't have time atm)
%         %bad_comps
%         bad_comps=load([datadir,'/derivatives/ICA/',sub_i,'/',sub_i,'_task-movie_',run_str,'_bad_ICA_comps.mat'],'bad_comps');
%         %comp150
%         comp150=load([datadir,'/derivatives/ICA/',sub_i,'/',sub_i,'_task-movie_',run_str,'_ICA_data.mat'],'comp150');
%         

        %% Prep MRI info
        %load in MRIs, align Glasser ROIs to subect T1, specify 
        %region->grid overlap and save this for Step3
        
        %0. load
        %subject T1
        sub_mri_dir = [datadir,sub_i,'/anat/'];
        sub_mri = [sub_mri_dir,sub_i,'.nii'];
        mri = ft_read_mri(sub_mri);
        mri.coordsys = 'ars'; %identified by using 3dinfo afni command
        
        %load in horn MMP
        mmp_mri = ft_read_mri(mmp_file);
        mmp_mri.coordsys = 'rpi'; %identified by using 3dinfo afni command
        
        %1. align Glasser ROIs to subect T1 using FSL commands
        %check if output file has been created
        mmp_out=[sub_mri_dir,sub_i,'_MMP'];
        if ~exist([mmp_out,'.nii.gz'])
            %i. use flirt (linear transform) to warp T1 -> MNI
            flirt_out=[sub_mri_dir,'flirt_out'];
            aff_out=[sub_mri_dir,'my_aff'];
            eval(['!flirt -in ',sub_mri,' -ref ',mni_file,' -out ',flirt_out,' -omat ',aff_out,' -dof 12']);

            if strcmp(sub_i,'sub-006') || strcmp(sub_i,'sub-010')
                %fnirt performed poorly for these subjects
                
                %ii. invert the flirt transform
                aff_inv=[sub_mri_dir,'my_aff_inv'];
                eval(['!convert_xfm -omat ',aff_inv,' -inverse ',aff_out]);
                
                %iii. apply inverse transform to MMP -> native
                eval(['!flirt -in ',mmp_file,' -ref ',sub_mri,' -applyxfm -init ',aff_inv,' -o ',mmp_out]);
                
            else
                %ii. use flirt (nonlinear transform) to finetune warp T1 -> MNI
                warp_mni=[sub_mri_dir,'warp2mni'];
                eval(['!fnirt --ref=',mni_file,' --in=',sub_mri,' --aff=',aff_out,' --cout=',warp_mni,' --config=T1_2_MNI152_2mm']);

                %iii. invert the warp
                warp_struct=[sub_mri_dir,'warp2struct'];
                eval(['!invwarp --ref=',sub_mri,' --warp=',warp_mni,' --out=',warp_struct]);

                %iv. apply inverted warp to transform glasser ROIS from MNI -> native T1
                eval(['!applywarp --ref=',sub_mri,' --in=',mmp_file,' --warp=',warp_struct,' --out=',mmp_out,' --interp=nn']);
            end
        end

        %2. specify region->grid overlap and save this for Step3
        %i. Load in native mmp
        %mmp_dat = ft_read_mri([mmp_out,'.nii.gz']);
        %mmp_dat.coordsys = 'ars'; %identified by using 3dinfo afni command
        mmp_atlas = ft_read_atlas([mmp_out,'.nii.gz']);
        

        %% Channel preproc
        
        %1. remove bad channels - identified through visual inspection by Rier
        %*read from ch_table
        bad_chans_data = find(startsWith(ch_table.status,'bad'));
        %remove from data and ch_table
        ch_table(bad_chans_data,:) = [];
        data(:,bad_chans_data) = [];
        
        %1a. extract sensor info from good channels
        sensor_info.pos = [ch_table.Px,ch_table.Py,ch_table.Pz];
        sensor_info.ors = [ch_table.Ox,ch_table.Oy,ch_table.Oz];
        
        %2. get data into a fieldtrip format (using Rier et al function)
        %note that data needs to be transposed to chan x timepoinnts
        data_strct = makeFTstruct(data',fs,ch_table,sensor_info);
        clear data;
        
        %3. demean prior to frequency filters
        %not detrending here as it is not default in ft freqanalysis, and
        %also was not done by Rier
        cfg = [];
        cfg.demean = 'yes';
        %cfg.detrend = 'yes';
        cfg.baselinewindow = 'all';
        data_strct=ft_preprocessing(cfg,data_strct);
    
        %4. bandpass filter into freq bands
        %*use acausal filter, in case we need to use MVAR FC
        
        %*eeglab recommends doing separate stages so that filter
        %order can be optimized separately https://eeglab.org/others/Firfilt_FAQ.html#q-should-we-prefer-separate-high--and-low-pass-filters-over-bandpass-filters-09302020-updated
        %*also, separating the filters seems to reduce the latency delay
        %introduced by causal filtering

        %hp first
        cfg = [];
        cfg.hpfilter='yes';
        cfg.hpfreq=filt_bands(1);
        if strcmp(filter_type,'causal_filter')
            cfg.hpfilttype='firws';
            cfg.hpfiltdir='onepass-minphase';
        else
            %need to reduce order for low freqs for noncausal filter
            if f==1 || f==2
                cfg.hpfiltord=3;
            end
        end
        data_strct=ft_preprocessing(cfg,data_strct);
        
        %then lp
        cfg = [];
        cfg.lpfilter='yes';
        cfg.lpfreq=filt_bands(2);
        if strcmp(filter_type,'causal_filter')
            cfg.lpfilttype='firws';
            cfg.lpfiltdir='onepass-minphase';
        else
            %need to adjust order for low freqs for noncausal filter
            if f==1
                cfg.lpfiltord=3;
            end
        end
        data_strct=ft_preprocessing(cfg,data_strct);
        
        %5. field correction? recommended by Rier - seems to work based on
        %visual inspection
        %"attenuates interference from distal sources of magnetic field"
        %based on this paper: https://www.sciencedirect.com/science/article/pii/S1053811921007576
        N = sensor_info.ors; %orientation matrix (N_sens x 3)
        M = eye(length(N)) - N*pinv(N);
        data_strct.trial{1} = M*data_strct.trial{1};
        
        %6. Downsample
        %*withoug acausal filtering i.e. take nth sample
        cfg = [];
        cfg.resamplefs = down_hz;
        cfg.method='downsample'; %this skips the lp filter
        cfg.detrend = 'no';
        cfg.demean = 'no';
        %cfg.sampleindex = 'yes';
        data_strct = ft_resampledata(cfg,data_strct);
        
        %% Create singleshell headmodel
        
        %1. segment mri
        cfg = [];
        %For Nolte's singleshell method, it seems that output here has to be a
        %binary brain mask (collapsing across gm, wm and csf) - so even if
        %tpm is specified here, a binary brainmask will be created at the
        %ft_prepare_headmodel stage.
        cfg.output = 'tpm'; %{'brain','skull','scalp'}; %'brain' = binary representation of the brain (collapsing gm, wm and csf); 'tpm' = tissue probability map, probabilistic representation of gm, wm and csf (only required for more complex methods e.g. BEM, FEM)
        cfg.write = 'no'; %do not write SPM format output - **why not? investigate if this helps later visualisation via export to nifti
        %cfg.downsample = 2;
        segmented_mri = ft_volumesegment(cfg,mri);
        
        %2. create headmodel
        %BEM option = takes too long; generates error during
        %ft_sourceanalysis
%         cfg= [];
%         cfg.method = 'dipoli';%'singleshell';
%         %cfg.unit = 'mm';
%         headmodel = ft_prepare_headmodel(cfg, meshes);
        
        %Nolte singleshell
        cfg = [];
        cfg.method = 'singleshell'; %*Other MEG options - singlesphere (outdated), localspheres (Huang method), singleshell (Nolte realistic conductor - *preferred); various BEM and one FEM model (both suited to conductivity anisotropy which does not apply to MEG as much as EEG)
        headmodel = ft_prepare_headmodel(cfg,segmented_mri);
        
        
        %% Assemble source model (headmodel, sensors, sources)
        
        %Current approach: i) head model: regular volume grid, ii) sensor
        %type treatment: all entered without scaling (.tra=empty)
        
        %Adapting previous MEG and EGI code
        %Using released 'meshes'?
        
        %Info on coordinate systems: headmodel/mesh & sensors=native space
        %sources (template grid)=?
        
        %0. load in 4mm source grid
        %sourcemodel
        load(ftrip_MNI_template);
        
        %1. convert sensors in data_strct from 'cm' to 'mm' (for consistency
        %with headmodel i.e. mesh); also convert sourcemodel from 'cm' to
        %'mm'
        data_strct.grad=ft_convert_units(data_strct.grad,'mm');
        sourcemodel=ft_convert_units(sourcemodel,'mm');
        
        
        %2. warp source template (MNI) to subject mri (native)
        % create the subject specific grid, using the template grid that has just been created
        cfg           = [];
        cfg.warpmni   = 'yes';
        %cfg.basedonmni = 'yes'; %recommended now, but somehow gives different result?
        cfg.template  = sourcemodel;
        cfg.nonlinear = 'yes';
        cfg.mri       = mri;
        cfg.unit      ='mm';
        grid         = ft_prepare_sourcemodel(cfg);
        
        %3.interpolate mmp atlas onto source model
        %this specifices source -> glasser region mapping
        %source_to_regions.tissue = codes for sources that belong to Glasser regions
        cfg = [];
        cfg.interpmethod = 'nearest';
        cfg.parameter = 'tissue';
        source_to_regions = ft_sourceinterpolate(cfg, mmp_atlas, grid);
        
        %4. verify that sensors and brain are aligned
        %*note: mesh is in mm, sens is in cm - convert sens to mm?
        
        execute=1;
        if execute==1
            figure; hold on;
            %ft_plot_mesh(meshes,'edgecolor','skin','facealpha',0.2,'edgealpha',0.3);
            ft_plot_headmodel(headmodel,'edgecolor','skin','facealpha',0.2,'edgealpha',0.3);
            ft_plot_sens(data_strct.grad,'style', 'sr'); 
            
            %plot full mesh
            %ft_plot_mesh(grid.pos(grid.inside,:));
            
            %plot subset of sources corresponding to a Glasser region
            %(sanity check)
            ind = find(source_to_regions.tissue==78); %78 = left premotor
            ind2 = find(source_to_regions.tissue==322); %right visual/parietal
            ix = [ind;ind2];
            ft_plot_mesh(grid.pos(ix,:),'vertexcolor','red');
        end
        

        %% Run beamformer source modeling
        %adapting my egi code
        
        %1. compute sensor covariance matrix
        cfg = [];
        cfg.covariance = 'yes';
        cfg.covariancewindow = 'all'; %[x y] = covariance computed within particular epochs within each trial  
        cfg.keeptrials = 'no';
        %cfg.vartrllength = 2; %2 = use all available samples in each trial, in case trials are of variable length (not the case in this project)
        data_covariance = ft_timelockanalysis(cfg, data_strct);

        %2b. Call timelockanalysis again to create trial
        %structure required for subsequent timeseries reconstruction
        cfg = [];
        %cfg.covariance = 'yes';
        %cfg.covariancewindow = 'all';
        cfg.keeptrials = 'yes';
        cfg.vartrllength = 2; %2 = use all available samples in each trial
        data_trls = ft_timelockanalysis(cfg, data_strct);
                    
        
        %3. Compute beamformer filter                  
        cfg = [];
        %cfg.channel = channel_num;
        %cfg.elec = elec_aligned_removedBad;
        cfg.method = 'lcmv';
        %cfg.grid = grid;
        cfg.sourcemodel = grid;
        cfg.headmodel = headmodel;
        
        cfg.normalize = leadfield_normalize; %lf being computed during this step, so need to specify this
        cfg.rawtrial = 'no'; %no = compute beamformer on average data; 
        cfg.keepfilter = 'yes'; %**yes = keep filter to apply in next stage to each trial
        cfg.keeptrials = 'no'; %no = does not keep trials
        cfg.fixedori = 'yes'; %%*constrains solution to dipole orientation with maximal power; ft_sourcedescriptives projectmom was generating an error, so this accomplishes the same thing
        cfg.projectnoise = sourceanalysis_projectnoise;
        cfg.keepleadfields='yes';
        cfg.lambda = sourceanalysis_lambda;
        %tic;
        source_data = ft_sourceanalysis(cfg,data_covariance);
        %toc;
        
        %*below doesn't work, and is not necessary, given that only one
        %trial is present in the data
        %4. Compute beamformer timeseries
        %cfg.grid.filter = source_data_covariance.avg.filter;
%         cfg.sourcemodel.filter=source_data_covariance.avg.filter;
%         cfg.rawtrial = 'yes'; %compute beamformer on raw trials rather than average
%         cfg.keeptrials = 'yes'; %this actually just keeps trialinfo (rawtrial is more important)
%         cfg.keepfilter = 'no';
%         sourcetimeseries_task_commonfilter = ft_sourceanalysis(cfg,data_trls);
%         
        
        %extract source timeseries to save disk space
        source_timeseries = source_data.avg.mom;
        source_inside = grid.inside;
        
        %% Save
        %include source_to_regions, source_timeseries
        out_file=[out_dir,'SourceData_',sub_i,'_',run_str,'_',filter_type,num2str(filt_bands(1)),'to',...
            num2str(filt_bands(2)),'Hz_lambda',lambda_str,'.mat'];
        save(out_file,'source_to_regions','source_timeseries','source_inside','-v7.3');
        
        %print
        disp(['finished source modeling for ',sub_i,', ',run_str,', time: ',num2str(toc)]);
        
        
    end

end

