#!/bin/bash
#SBATCH --partition=price
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=preproc1_3
#SBATCH --output=slurm.preproc1_3.out
#SBATCH --error=slurm.preproc1_3.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=90000
#SBATCH --export=ALL
module purge
module load MATLAB/R2023a
#Run the MATLAB command
cd /home/let83/FND4/Analyses/
matlab -nodisplay -r "Step2b_Raw_MVPA_TempGen_subject(3,RespMotorHand), exit"
