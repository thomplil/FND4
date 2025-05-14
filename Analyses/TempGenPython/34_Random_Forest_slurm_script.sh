#!/bin/bash
#SBATCH --partition=price
#SBATCH --job-name=Random_Forest_34
#SBATCH --time=06:00:00      # at increased
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12     # Number of parallel tasks, adjust as needed for tempGen
#SBATCH --mem=40000
#SBATCH --output=/projects/f_mc1689_1/cpro2_eeg/docs/scripts/TempGenPython/34_Random_Forest.out
#SBATCH --error=/projects/f_mc1689_1/cpro2_eeg/docs/scripts/TempGenPython/34_Random_Forest.err
#SBATCH --export=ALL
# Activate the virtual environment
source /home/let83/eeg_decoding_env/bin/activate

# Run the Python script for the subject
time python /projects/f_mc1689_1/cpro2_eeg/docs/scripts/TempGenPython/34_Random_Forest_decoder_script.py
