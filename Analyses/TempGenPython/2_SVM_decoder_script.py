import sys
sys.path.append('/projects/f_mc1689_1/cpro2_eeg/docs/scripts/')  # Ensure the directory is in the path
import Step3a_run_decoders_fxns_v2
import os
import h5py, pkg_resources
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import glob, warnings, shutil
import mne
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from scipy.stats import sem  # Import sem for standard error

# Add the function directory to the Python path
#sys.path.append('/projects/f_mc1689_1/cpro2_eeg/docs/scripts/')

# Import functions from Step3a_run_decoders_fxns_v2
from Step3a_run_decoders_fxns_v2 import DecodingAcc, decode_single_timeseries, decode_TempGen

# Decoding Analysis Function
decodingAnalysis = 'TempGen'  # or 'DecodingAccuracy' (done)
classify_cond = 'Left' # or 'Right' (needs "17 21 34 44", 5svm) or 'Hand' (lda needs last 4, svm has 1 10 16 (2 3 5)); left lda and svm needs "17 21 34 44", rf needs rest
subjNum = 2
decoderType = 'SVM'
nTimepoints=494
data_lock = 'stimulus' #can be stimulus or response
if decodingAnalysis == 'DecodingAccuracy':
    #Set up variables needed to run decoding 
    output_dir, output_file, cond_code, random_data_trl, folds_mat,random_cond_targets = DecodingAcc(classify_cond, decoderType, subjNum, decodingAnalysis,data_lock)

    # Parallelize decoding of each timepoint
    results = Parallel(n_jobs=-1)(delayed(decode_single_timeseries)(t,decoderType,cond_code, random_data_trl, folds_mat, random_cond_targets) for t in range(nTimepoints))
    #Extract results and turn them into arrays
    avg_accuracy, sem_accuracy = zip(*results)
    accuracy_avg = np.array(avg_accuracy)
    accuracy_sem = np.array(sem_accuracy)
    print('got decoding results')
    print(f'Ran timeseries decoding for subject {subjNum}')
    
    #Print results for each subject
    ##Specify title and saving name
    if classify_cond=='Left' or classify_cond=='Right':
        title = f'{classify_cond} Hand {decoderType} Finger Decoding\nAccuracy of Subject {subjNum}'
        outfile = f'{classify_cond}_Hand_{decoderType}_Finger_Decoding_Accuracy_of_Subject{subjNum}'
    elif classify_cond=='Hand':
        title = f'Hand {decoderType} Decoding Accuracy of Subject {subjNum}'
        outfile = f'Hand_{decoderType}_Decoding_Accuracy_of_Subject{subjNum}'
    ##Specify graph colors based on decoder type
    if decoderType == 'LDA':
        c = 'mediumorchid'
        c_fill = 'violet'
    elif decoderType == 'Random_Forest':
        c = 'seagreen'
        c_fill = 'mediumseagreen'
    elif decoderType == 'SVM':
        c = 'steelblue'
        c_fill = 'lightskyblue'
    ##Define x axis and plot
    timepoints = np.linspace(-0.09, 3.855, nTimepoints)
    plt.figure(figsize=(14,9))
    plt.plot(timepoints, accuracy_avg, color=c, linewidth=2)
    plt.fill_between(timepoints, accuracy_avg - accuracy_sem, accuracy_avg + accuracy_sem, alpha=0.3, color=c_fill)
    plt.title(title, fontsize=15) 
    plt.xlabel('Time (s)', fontsize=13)
    plt.ylabel('Decoding Accuracy', fontsize=13)
    plt.savefig(os.path.join(output_dir, outfile))
    #plt.show()
    print('saved graph')
    # Save output
    avg_accuracy_pd = pd.DataFrame(accuracy_avg)
    full_path = os.path.join(output_dir, f'{subjNum}{output_file}.csv')
    print(full_path)
    avg_accuracy_pd.to_csv(full_path, index=False)
    print('saved data')
    
    
    
elif decodingAnalysis == 'TempGen':
    def run_tempgen_decoding(classify_cond, decoderType, subjNum, decodingAnalysis, data_lock, nTimepoints):
        # Set up variables needed to run decoding
        output_dir, output_file, cond_code, random_data_trl, folds_mat, random_cond_targets = DecodingAcc(
            classify_cond, decoderType, subjNum, decodingAnalysis, data_lock)
        accMat = np.zeros((nTimepoints, nTimepoints)) #Make empty 2D array to add in decoding accuracy at each timepoint
        # Parallelize temporal generalization decoding
        print('started decoding')
        results = Parallel(n_jobs=-1, backend='loky')(
            delayed(decode_TempGen)(
                t_train, nTimepoints, decoderType, random_data_trl, folds_mat, random_cond_targets
            ) for t_train in range(nTimepoints)
        )
        # Stack all train_time rows to form the final accuracy matrix
        tempGenAcc = np.stack(results, axis=0)  # shape: (nTimepoints, nTimepoints)
        print('got temp gen decoding results')
        #Save the data
        print('Attempting to save data')
        tempGenAccMat = pd.DataFrame(tempGenAcc)
        full_path = os.path.join(output_dir, f'{subjNum}{output_file}.csv')
        tempGenAccMat.to_csv(full_path, index=False)
        print(f'saved data to {full_path}')

        return tempGenAcc, output_dir
    
    if __name__ == "__main__":
        # Define inputs
        # classify_cond = 'Hand'
        # decoderType = 'SVM'
        # subjNum = 2
        decodingAnalysis = 'TempGen'
        data_lock = 'stimulus'
        nTimepoints = 494
        
        tempGenAcc, output_dir = run_tempgen_decoding(classify_cond, decoderType, subjNum, decodingAnalysis, data_lock, nTimepoints)
    else: print('ERROR -- DID NOT RUN DECODING')
    
    print('Results:')
    print(tempGenAcc)
    print('')
    #Plot results for temporal generalization
    ##Specify graph title and saving 
    if classify_cond=='Left' or classify_cond=='Right':
        title = f'Temporal Generalization of {classify_cond} Hand\n{decoderType} Finger Decoding Accuracy of Subject {subjNum}'
        outfile = f'TempGen_{classify_cond}_Hand_{decoderType}_Finger_Decoding_Accuracy_of_Subject{subjNum}'
    elif classify_cond=='Hand':
        title = f'Temporal Generalization of Hand {decoderType} Decoding Accuracy of Subject {subjNum}'
        outfile = f'TempGen_Hand_{decoderType}_Decoding_Accuracy_of_Subject{subjNum}'
    
    ##Define graph dimensions based on number of timepoints within the session
    nTimepoints = tempGenAcc.shape[0]
    timepoints = np.linspace(-0.09, 3.855, nTimepoints)
    
    ##Plot heatmap of each timeseries of decoding accuracies (goes across x-axis) for each time point that decoding was trained on (each point on the y axis is a time point that the decoder was trained on)
    plt.figure(figsize=(14,9))
    plt.imshow(tempGenAcc, aspect='auto', cmap='viridis', origin='lower',
               extent=[timepoints[0], timepoints[-1], timepoints[0], timepoints[-1]])
    plt.colorbar(label='Decoding Accuracy')
    plt.xlabel('Test Timepoints', fontsize=13)
    plt.ylabel('Train Timepoints', fontsize=13)
    
    plt.grid()
    plt.title(title, fontsize=15)
    plt.axhline(0, color='white', linestyle='--')
    plt.axvline(0, color='white', linestyle='--')
    plt.savefig(os.path.join(output_dir, outfile))
    
    print('Ran Temporal Generalization decoding analysis')
