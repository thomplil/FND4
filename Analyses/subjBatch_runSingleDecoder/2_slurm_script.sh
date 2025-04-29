#!/bin/bash
#SBATCH --partition=price
#SBATCH --job-name=DecodingSingleAcc_2
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5    # Number of parallel tasks, adjust as needed
#SBATCH --mem=60000
#SBATCH --output=/home/let83/FND4/Analyses/subjBatch_runSingleDecoder/2.out
#SBATCH --error=/home/let83/FND4/Analyses/subjBatch_runSingleDecoder/2.err
#SBATCH --export=ALL

# Activate the virtual environment
source /home/let83/eeg_decoding_env/bin/activate

# Run the Python script for the subject
time python /home/let83/FND4/Analyses/subjBatch_runSingleDecoder/2_single_decoder_script.py
