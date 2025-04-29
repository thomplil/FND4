#!/bin/bash

##AMAREL computing cluster (at Rutgers University) batch script
##
##Author: Michael W. Cole (mwcole@mwcole.net) -- edited by Lily Thompson
##
##This script runs a MATLAB command on a large number of subjects using the SLURM queing system

##**** Modify these variables (use mkdir to create the indicated directory scheme, adjusted for the project): ****
##scriptDir is where scripts for these jobs are saved; all m-files and helper function files should be saved here before running
##I can try and move the preprocessing scripts to a scripts/preproc_causalFilter folder if you think I really need to
scriptDir="/projects/f_mc1689_1/cpro2_eeg/docs/scripts/"

##This method batches by subject; or creates a shell script for each subject and runs them separately (e.g., generates new jobs/tasks for each subject). The MATLAB functions herein do not initiate any parallel processing, so this method allows you to keep individual job specs to a minimum, but still run them simultaneously (e.g., have many non-demanding jobs running at once; less likely to get pre-empted). Note that for other uses/algorithms, parallelizing inside MATLAB might be a better option (but one will likely have to change the job spec options below).
##The following is a sub-directory to house the batched subject scripts (I like to keep up-to-date copies of the function scripts in here as well):
subjBatchScriptDir="/projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjbatch_preproc/"
if [ ! -d ${subjBatchScriptDir} ]; then mkdir -p $subjBatchScriptDir; fi 
jobNamePrefix="preproc1_cpro2eeg"

#subjects
listOfSubjects="1 2 3 5 6 7 8 9 10 11 13 14 16 17 18 21 22 23 24 25 26 27 29 30 31 32 34 35 36 37 38 39 40 41 42 43 44 45 48 49 50 51 53 54 55 56 57"  

#pass run_mode, 1=only run code up till ICA computation (rest has to be done in interactive session)
run_mode="1"

##Make and execute a batch script for each subject
for subjNum in $listOfSubjects
do
    #Only create and run script if the output file doesn't already exist; quality check for pre-empting
    #if [ ! -f ${outputDir}${outputFilePrefix}${subjNum} ]; then
    cd ${subjBatchScriptDir}

    batchFilename=${subjNum}_matlabBatch.sh


    #Job specs (resources) indicated for each batch process; memory, nodes, ntasks, time, and parititon are more likely to change depending on the project's needs. Other specs are more likely to be kept as-is (but double-check for your purposes). .Err files (also batched) are useful for debugging. See rutgers-oarc.github.io/amarel/ for more detailed info.
    echo "#!/bin/bash" > $batchFilename
    #These jobs are not low-impact, use partition Price
    echo "#SBATCH --partition=price" >> $batchFilename
    ##time for one subject + buffer 
    echo "#SBATCH --time=72:00:00" >> $batchFilename
    #1 Node indicated here to be at the minimum; modify for your purposes:
    echo "#SBATCH --nodes=1" >> $batchFilename
    #1 Task indicated here because each batch is just 1 job (no parallel processing); modify for your purposes:
    echo "#SBATCH --ntasks=1" >> $batchFilename
    #Job name created per batch:
    echo "#SBATCH --job-name=${jobNamePrefix}${subjNum}" >> $batchFilename
    #Output file created per batch; see .out files when jobs are completed (will open in a text editor):
    echo "#SBATCH --output=slurm.${jobNamePrefix}${subjNum}.out" >> $batchFilename
    #Error files create per batch; see .err files when jobs are completed (will open in a text editor); if no errors were thrown, these should be empty:
    echo "#SBATCH --error=slurm.${jobNamePrefix}${subjNum}.err" >> $batchFilename
    #2 CPU per task minimum needed to complete task
    echo "#SBATCH --cpus-per-task=2" >> $batchFilename
    #Mem choice of 90 GB, minimum needed to complete task
    echo "#SBATCH --mem=90000" >> $batchFilename
    #'Export all' purportedly exports all environmental variables to the job environment:
    echo "#SBATCH --export=ALL" >>$batchFilename


    #Running MATLAB. 'Module load' is required first in Amarel (for those adapting NM3 scripts; MATLAB was pre-written into NM3 path, but is not in Amarel, so you must load it for each batch process; purge is likely not required, but might be a good way to keep things clean).
    echo "module purge" >> $batchFilename
    #Use your version of MATLAB here
    echo "module load MATLAB/R2019a" >> $batchFilename
    echo "#Run the MATLAB command" >> $batchFilename
    echo "cd $scriptDir" >> $batchFilename
    #Call the MATLAB function: replace "compare_Params_FIR_NF" with your function's name. In this case, the only input is subjNum (as described above for listOfSubjects), but your function's required inputs might be set-up differently; everything inside curly brackets constitutes 1 input variable value; to call more than one input, use comma-separated curly-bracketed parameters inside the regular parentheses, or comma-separated values.
    echo "matlab -nodisplay -r \"Step1a_preproc_EEG(${subjNum},${run_mode}), exit\"" >> $batchFilename


    #Submit the job
    sbatch $batchFilename
done
