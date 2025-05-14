#!/bin/bash
#SBATCH --partition=price
#SBATCH --job-name=DecodingSingleAcc_5
#SBATCH --time=03:00:00      # Will be increased when running tempGen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5    # Number of parallel tasks, adjust as needed for tempGen
#SBATCH --mem=80000
#SBATCH --output=/projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/5.out
#SBATCH --error=/projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/5.err
#SBATCH --export=ALL

# Activate the virtual environment
source /home/let83/eeg_decoding_env/bin/activate

# Run the Python script for the subject
time python /projects/f_mc1689_1/cpro2_eeg/docs/scripts/subjBatch_runSingleDecoder/5_single_decoder_script.py
