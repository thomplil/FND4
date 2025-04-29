#!/bin/bash
#SBATCH --partition=price
#SBATCH --job-name=DecodingSingleAcc_16
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=60000
#SBATCH --output=/home/let83/FND4/Analyses/subjBatch_runSingleDecoder/16.out
#SBATCH --error=/home/let83/FND4/Analyses/subjBatch_runSingleDecoder/16.err
#SBATCH --export=ALL
source home/let83/eeg_decoding_env/bin/activate
time python /home/let83/FND4/Analyses/subjBatch_runSingleDecoder/16_single_decoder_script.py
