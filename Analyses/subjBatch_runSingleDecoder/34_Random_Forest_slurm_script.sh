#!/bin/bash
#SBATCH --partition=price
#SBATCH --job-name=Random_Forest_34
#SBATCH --time=06:00:00      # at increased
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6     # Number of parallel tasks, adjust as needed for tempGen
#SBATCH --mem=20000
#SBATCH --output=/projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/34_Random_Forest.out
#SBATCH --error=/projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/34_Random_Forest.err
#SBATCH --export=ALL
# Activate the virtual environment
source /home/let83/eeg_decoding_env/bin/activate

# Run the Python script for the subject
time python /projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/34_Random_Forest_decoder_script.py
