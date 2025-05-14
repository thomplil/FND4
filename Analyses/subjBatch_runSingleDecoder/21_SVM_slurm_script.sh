#!/bin/bash
#SBATCH --partition=price
#SBATCH --job-name=SVM_21
#SBATCH --time=06:00:00      # at increased
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8     # Number of parallel tasks, adjust as needed for tempGen
#SBATCH --mem=40000
#SBATCH --output=/projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/21_SVM.out
#SBATCH --error=/projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/21_SVM.err
#SBATCH --export=ALL
# Activate the virtual environment
source /home/let83/eeg_decoding_env/bin/activate

# Run the Python script for the subject
time python /projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/21_SVM_decoder_script.py
