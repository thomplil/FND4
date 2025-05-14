#!/bin/bash
#SBATCH --partition=price
#SBATCH --job-name=LDA_17
#SBATCH --time=06:00:00      # at increased
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5     # Number of parallel tasks, adjust as needed for tempGen
#SBATCH --mem=50000
#SBATCH --output=/projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/17_LDA.out
#SBATCH --error=/projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/17_LDA.err
#SBATCH --export=ALL
# Activate the virtual environment
source /home/let83/eeg_decoding_env/bin/activate

# Run the Python script for the subject
time python /projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/17_LDA_decoder_script.py
