#!/bin/bash

##AMAREL computing cluster (at Rutgers University) batch script
##
##Author: Michael W. Cole (mwcole@mwcole.net)
##
##This script runs a MATLAB command on a large number of subjects using the SLURM queing system

##**** Modify these variables (use mkdir to create the indicated directory scheme, adjusted for the project): ****
##scriptDir is where scripts for these jobs are saved; all m-files and helper function files should be saved here before running
scriptDir="/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/scripts/Analysis3_causalFilter/"

##This method batches by subject; or creates a shell script for each subject and runs them separately (e.g., generates new jobs/tasks for each subject). The MATLAB functions herein do not initiate any parallel processing, so this method allows you to keep individual job specs to a minimum, but still run them simultaneously (e.g., have many non-demanding jobs running at once; less likely to get pre-empted). Note that for other uses/algorithms, parallelizing inside MATLAB might be a better option (but one will likely have to change the job spec options below).
##The following is a sub-directory to house the batched subject scripts (I like to keep up-to-date copies of the function scripts in here as well):
#subjBatchScriptDir="/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/scripts/Analysis3_causalFilter/subjbatch_source/"
#if [ ! -f ${subjBatchScriptDir} ]; then mkdir -p $subjBatchScriptDir; fi 

#jobNamePrefix="EGp"

#subjects
listOfSubjects="803 804 805 806 807 809 810 811 813 814 815 817 818 819 820 821 822 823 824 825 826 827 828 829 831 832 833 834 835 836 837 838"
listOfSubjects="810"

#pass run_mode, 1=task, 2=rest
run_mode="1"

#set batch script dir (and job prefix) based on run_mode
if [ $run_mode -eq 1 ]; then 
    subjBatchScriptDir="/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/scripts/Analysis3_causalFilter/subjbatch_source_task/"
    jobNamePrefix="EGt"
elif [ $run_mode -eq 2 ]; then 
    subjBatchScriptDir="/projects/f_mc1689_1/DynamicSensoryMotorEGI/docs/scripts/Analysis3_causalFilter/subjbatch_source_rest/"
    jobNamePrefix="EGr"
fi
if [ ! -f ${subjBatchScriptDir} ]; then mkdir -p $subjBatchScriptDir; fi 

##Make and execute a batch script for each subject
for subjNum in $listOfSubjects
do
    #Only create and run script if the output file doesn't already exist; quality check for pre-empting
    #if [ ! -f ${outputDir}${outputFilePrefix}${subjNum} ]; then
    cd ${subjBatchScriptDir}

    batchFilename=${subjNum}_matlabBatch.sh

    #Job specs (resources) indicated for each batch process; memory, nodes, ntasks, time, and parititon are more likely to change depending on the project's needs. Other specs are more likely to be kept as-is (but double-check for your purposes). .Err files (also batched) are useful for debugging. See rutgers-oarc.github.io/amarel/ for more detailed info.
    echo "#!/bin/bash" > $batchFilename
    #These jobs are low-impact, so there's no need to specify our own partition in this case; 'main' will allocate to any available compute node (pre-empting is possible though; but less likely with the low specs listed here):
    echo "#SBATCH --partition=nm3" >> $batchFilename
    #Requeue: Built-in quality check for jobs that may have been pre-empted:
    #echo "#SBATCH --requeue" >> $batchFilename
    echo "#SBATCH --time=336:00:00" >> $batchFilename
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
    #1 CPU per task indicated here because we created batched processes of 1 low-impact job per subject (with no parallelization in the MATLAB functions themselves); with the other specs considered, this means that on 1 node, each subject (which = 1 job/task) will use up 1 cpu; so you can have ~20 subjects running at once as separate jobs on the same node, depending on what's available at the time (or a variety of other allocation schemes; thus it's flexible, and many small-resource jobs are less likely to be pre-empted); modify for your purposes:
    echo "#SBATCH --cpus-per-task=4" >> $batchFilename
    #Mem choice of 10 GB: An interactive session with 1 subject was run & assessed; ~7GB was used, so 10GB seemend to be a reasonable number for mem. Note: I underestimated in a first-pass and this did not 'delimit' my job in any way (ex: if you underestime --time, the job will end at that time limit; but --mem does not appear to end jobs that exceed what is typed here; thus it's not clear how useful this is); see rutgers-oarc.github.io/amarel/ for how to run interactive sessions:
    echo "#SBATCH --mem=80000" >> $batchFilename
    #'Export all' purportedly exports all environmental variables to the job environment:
    echo "#SBATCH --export=ALL" >>$batchFilename


    #Running MATLAB. 'Module load' is required first in Amarel (for those adapting NM3 scripts; MATLAB was pre-written into NM3 path, but is not in Amarel, so you must load it for each batch process; purge is likely not required, but might be a good way to keep things clean).
    echo "module purge" >> $batchFilename
    echo "module load MATLAB/R2019a" >> $batchFilename
    echo "#Run the MATLAB command" >> $batchFilename
    echo "cd $scriptDir" >> $batchFilename
    #Call the MATLAB function: replace "compare_Params_FIR_NF" with your function's name. In this case, the only input is subjNum (as described above for listOfSubjects), but your function's required inputs might be set-up differently; everything inside curly brackets constitutes 1 input variable value; to call more than one input, use comma-separated curly-bracketed parameters inside the regular parentheses, or comma-separated values.
    echo "matlab -nodisplay -r \"Step3_SourceModeling_bandLCMVbeamform_EGI_v5(${subjNum},${run_mode}), exit\"" >> $batchFilename


    #Submit the job
    sbatch $batchFilename

done
