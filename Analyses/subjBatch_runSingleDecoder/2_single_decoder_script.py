import import_ipynb
import Step2a_run_decoders_fxns
import os
import h5py, pkg_resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import sys
import glob, warnings, shutil
import mne
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from scipy.stats import sem  # Import sem for standard error

# Add the function directory to the Python path
sys.path.append('/home/let83/FND4/Analyses/')

# Import functions from Step2a_run_decoders_fxns
from Step2a_run_decoders_fxns import decode_single_timepoint, decode_TempGen

# Decoding Analysis Function
decodingAnalysis = 'DecodingAccuracy'  # or 'TempGen'
classify_cond = 'Hand'
subjNum = 2
decoderType = 'SVM'
nTimepoints=250
if decodingAnalysis == 'DecodingAccuracy':
    # Parallelize decoding of each timepoint
    avg_accuracy, sem_accuracy = Parallel(n_jobs=-1)(delayed(decode_single_timepoint)(classify_cond, subjNum,t, decoderType,decodingAnalysis) for t in range(nTimepoints))
    # Import functions from Step2a_run_decoders_fxns
    import Step2a_run_decoders_fxns
    from Step2a_run_decoders_fxns import decode_single_timepoint, decode_TempGen

    print(f'Ran timeseries decoding for subject {subjNum}')
    if classify_cond=='Left' or classify_cond=='Right':
        title = f'{classify_cond} Hand {decoderType} Finger Decoding\nAccuracy of Subject {subjNum}'
        outfile = f'{classify_cond}_Hand_{decoderType}_Finger_Decoding_Accuracy_of_Subject{subjNum}'
    elif classify_cond=='Hand':
        title = f'Hand {decoderType} Decoding Accuracy of Subject {subjNum}'
        outfile = f'Hand_{decoderType}_Decoding_Accuracy_of_Subject{subjNum}'

    timepoints = np.linspace(-0.5, 0.5, nTimepoints)
    plt.figure()
    plt.plot(timepoints, avg_accuracy, color='steelblue', linewidth=2)
    plt.fill_between(timepoints, avg_accuracy - sem_accuracy, avg_accuracy + sem_accuracy, alpha=0.3, color='lightskyblue')
    plt.title(title, fontsize=15) 
    plt.xlabel('Time (s)', fontsize=13)
    plt.ylabel('Decoding Accuracy', fontsize=13)
    plt.savefig(os.path.join(outputDir, outfile))
    plt.show()

    # Save output
    avg_accuracy_pd = pd.DataFrame(avg_accuracy)
    full_path = os.path.join(output_dir, f'{subjNum}{output_file}.csv')
    avg_accuracy_pd.to_csv(full_path, index=False)

elif decodingAnalysis == 'TempGen':
    # Parallelize temporal generalization decoding
    tempGenAcc = Parallel(n_jobs=-1)(delayed(decode_TempGen)(t, nTimepoints, classify_cond, subjNum, decoderType, decodingAnalysis) for t in range(nTimepoints))

    # Further processing for temporal generalization
    if classify_cond=='Left' or classify_cond=='Right':
        title = f'Temporal Generalization of {classify_cond} Hand\n{decoderType} Finger Decoding Accuracy of Subject {subjNum}'
        outfile = f'TempGen_{classify_cond}_Hand_{decoderType}_Finger_Decoding_Accuracy_of_Subject{subjNum}'
    elif classify_cond=='Hand':
        title = f'Temporal Generalization of Hand {decoderType} Decoding Accuracy of Subject {subjNum}'
        outfile = f'TempGen_Hand_{decoderType}_Decoding_Accuracy_of_Subject{subjNum}'

    nTimepoints = tempGenAcc.shape[0]
    timepoints = np.linspace(-0.5, 0.5, nTimepoints)
    plt.figure(figsize=(8,6))
    plt.imshow(tempGenAcc, aspect='auto', cmap='viridis', origin='lower',
               extent=[timepoints[0], timepoints[-1], timepoints[0], timepoints[-1]])
    plt.colorbar(label='Decoding Accuracy')
    plt.xlabel('Test Timepoints', fontsize=13)
    plt.ylabel('Train Timepoints', fontsize=13)
    plt.title(title, fontsize=15)
    plt.axhline(0, color='white', linestyle='--')
    plt.axvline(0, color='white', linestyle='--')
    plt.savefig(os.path.join(outputDir, outfile))
    plt.show()

    print('Ran Temporal Generalization decoding analysis')

