Project description:
1.  Question: Can small processes (like differentiating finger movements) be decoded as well as large processes (like differentiating hand movements) when using the whole brain signal?
2.  Methods: Use multivariate pattern analyses (SVM, LDA, and Random Forest) to decode a single timeseries and cross temporal generalization of a 256 channel whole-brain signal.
3.  Task: Participants were presented with a logic, sensory, and motor rule. Each one of these rule types was made up of a list of four possiblities such that when the rules were recombined, the participants could completed 60 novel tasks and 4 practice tasks that were repeated 15 times. Each task had three trials of responses. In this data, each trial was cleaned, separated, and stimulus locked. Because of the four rules, there were four different possible motor responses: left hand index finger, left hand middle finger, right hand index finger, and right hand middle finger. These responses were set into three conditions: Left (differentiating left index and middle finger movement), Right (differentiating right index and middle finger movement), and Hand (differentiating left and right hand movements).

Scripts descriptions:
1.  Step1a_preproc_EEG.m -- Preprocesses the data (filtering, segmentation, ICA removal of eye movements, baselining etc). Also graphs raw data.
2.  Step1a_subFxns -- mini functions that are needed to run Step1a_preproc_EEG.m
3.  Step1b_source_batch_preproc.sh -- make parallel jobs for each participant to run through the preprocessing in step1a
4.  Step1c_dataSave.m -- run additionaly bandwidth filtering specific to decoding analyses and downsampling on the data, save the motor info and logic rule info as numerical values in trialinfo, and save the necessary output in an easier format to load into python
5.  Step2a_run_decoder_fxns_v2.py -- list of functions needed to a) load in the data saved in .mat files, b) set up the data so there are easy inputs to the single timeseries decoder and temporal generalization decoder, c) run the single timeseries decoder at one time point, and d) run the temporal generalization at one training time point. 
		!!!!IMPORTANT!!!! To run these functions you may need to set up an environment that includes the package mne (along with all the other packages, but mne works with a specific numpy, so load that one in first when creating the environment)
6.  Step2b_Run_Decoders_Parallel.sh -- Creates a script for each participant and decoder type (LDA, SVM, or Random_Forest), and then depending on what decodingAnalysis is defined as ('DecodingAccuracy' or 'TempGen'), runs either the single timeseries decoding or temporal generalization decoding. 
    !!!!IMPORTANT!!!! 
        * There are several variables that need to be manually changed depending on what exactly you want to run (subjects, decoderType, decodingAnalysis, memory (for single decoder should not be more than 20GB and no more than 50GB for temporal generalization), numCPUs (for single decoder, no more than 10, for temporal generalization, lda = 15, svm = 27, rf = 27), time within batch needs to be increased for temporal generalization (10 hours)
        * If you are running either SVM or Random_Forest, you really should not submit more than 4 subjects at once because they will get pre-empted. In LDA, you can, but it works more reliably if you run it in the segments that I have defined.
7.  Step2c_DecodingAccuracy_Graphing.ipynb -- graph group results of single timeseries decoding.
8.  Step2d_Analyze_SingleTimeseries.ipynb -- Uses t-tests and a linear mixed model to analyze whether subject peak accuracy was significantly greater than 50% (t-tests) and whether decoder type, test time point, or condition predicted different accuracy levels. FDR corrections were applied as needed. Plots results.
9.  Step3e_Analyze_TempGen.ipynb -- Uses t-tests and a linear mixed model to determine whether accuracy time courses trained on each timepoint were significantly different from chance, and whether decoding accuracy changes across test timepoints and task conditions. FDR corrections were applied as needed. Plots results in three heatmaps (one for each condition), all accuracies that are not significantly greater than chance were set to 0. 







