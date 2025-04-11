Project description: Compare the efficacy of to MVPAs in decoding left versus right hand movements. First decoder focuses only on spatial patterns, second decoder also accounts for temporal variation.

Project folders description:
  
  a. Analyses: contains all analysis scripts
     i.   fullBehavioralAnalysis: contains all behavioral analyses and visualizations of behavioral analyses. Should be run using jupyter notebook (python)
     ii.  will also contain decoder scripts and a script to analyze and compare results of decoders
     
  b. preproc: contains all preprocessing scripts
     i.   Step1_preproc_EEG_rest_task_RM_v2.m: does initial EGI preprocessing using FieldTrip. Written in matlab. Includes...
          1. task segmentation and encoding, ITI, and response/trial segmentation
          2. causal highpass, lowpass, and notch
          3. ICA eye movement cleaning
          5. Bad trial removal (no channels removed) 
     ii.  preproc_step1_source_batch.sh: Creates a job on amarel to run initial preprocessing for each subject in parallel. Written in bash. Still drafting.
     iii. Step2_VolumeSourceModeling_bandLCMVbeamform_EGI.m: Source models EGI data using the 264 Power Atlas. Still Drafting.
     iv.  others are self explanatory functions used in initial preprocessing. 
     
  c. Ravi_Scripts: Scripts written by my postdoc that I adapt to create (most) scripts in preproc.
          
          
