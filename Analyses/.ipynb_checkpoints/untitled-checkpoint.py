#import functions
import os
import numpy as np
import pandas as pd
# import statsmodels.api as sm
import matplotlib.pyplot as plt

print('got through first functions')
#Plotting
import h5py,pkg_resources,sys,scipy
print('impored h5py etc')
#general python use
import sys
import os,glob,warnings,shutil
import mne
print('imported mne')
import random
# from sklearn.ensemble import RandomForestClassifier
print('imported random')
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
print('imported sklearn scripts')
from scipy import signal, stats
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multitest import multipletests
import math
print('imported scipy sm and math')

#Decoders
from sklearn.svm import SVC

#read out matrices to files
import io

#Parallel loop
from joblib import Parallel, delayed
print('imported last modules')

def decode_single_timeseries(classify_cond, subjNum,t, decoderType,decodingAnalysis):
    cond_code, random_data_trl, folds_mat,random_cond_targets = DecodingAcc(classify_cond, decoderType, subjNum, decodingAnalysis)
    nTimepoints = random_data_trl.shape[2]
    X = random_data_trl[:, :, t]#trials x channels x 1 timepoint
    accuracies = []
    for train_idx, test_idx in folds_mat:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = random_cond_targets[train_idx], random_cond_targets[test_idx]
        
        # Initialize a NEW classifier each time (important!)
        if decodingAnalysis =='SVM':
            clf = SVC(kernel='linear')
        elif decodingAnalysis == 'LDA':
            clf = LinearDiscriminantAnalysis()
        elif decodingAnalysis == 'Random_Forest':
            clf = RandomForestClassifier(n_estimators=150, random_state=42)
            
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    return np.mean(accuracies), sem(accuracies)

# Make function to run temporal generalization 
def decode_TempGen(t_train, nTimepoints, classify_cond, subjNum, decoderType,decodingAnalysis):
    cond_code, random_data_trl, folds_mat,random_cond_targets = DecodingAcc(classify_cond, decoderType, subjNum, decodingAnalysis)
    nTimepoints = random_data_trl.shape[2]
    X_train_data = random_data_trl[:, :, t_train]
    for t in range(nTimepoints):
        # Extract data at time t across all trials
        # Shape: (n_trials, n_channels) --> perform classification on a time point instead a trial
        X = random_data_trl[:, :, t]
        # 10-fold cross-validation
        accuracies = []
        for train_idx, test_idx in folds:
            X_train, X_test = X_train_data[train_idx], X[test_idx]
            y_train, y_test = random_cond_targets[train_idx], random_cond_targets[test_idx]
            
            if decodingAnalysis =='SVM':
                clf = SVC(kernel='linear')
            elif decodingAnalysis == 'LDA':
                clf = LinearDiscriminantAnalysis()
            elif decodingAnalysis == 'Random_Forest':
                clf = RandomForestClassifier(n_estimators=150, random_state=42)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        # Mean cross-validated accuracy at this timepoint
        tempGenAcc[t_train,t]=np.mean(accuracies)

    return tempGenAcc

#Functions to load data
def read_dataset(ds,k):
    """Reads an HDF5 dataset, resolving references and decoding properly."""

    # If it's a string dataset
    str_info = h5py.check_string_dtype(ds.dtype)
    if str_info is not None:
        return ds.asstr()[()]

    # If it's a compound structured array
    if ds.dtype.names:
        data = {}
        for name in ds.dtype.names:
            field = ds[name][()]
            if np.issubdtype(field.dtype, np.bytes_):
                data[name] = field.astype(str)
            else:
                data[name] = field
        return pd.DataFrame(data)

    # If it's a dataset of references
    if ds.dtype.kind == 'O':
        refs = ds[()]
        if refs.ndim > 0:
            deref_data = []
            for r in refs.flat:
                if isinstance(r, h5py.h5r.Reference):
                    real_obj = ds.file[r]
                    if isinstance(real_obj, h5py.Dataset):
                        deref_data.append(read_dataset(real_obj,k))
                    elif isinstance(real_obj, h5py.Group):
                        deref_data.append(load_group(real_obj,k))
            return deref_data
        else:
            r = refs
            real_obj = ds.file[r]
            if isinstance(real_obj, h5py.Dataset):
                return read_dataset(real_obj,k)
            elif isinstance(real_obj, h5py.Group):
                return load_group(real_obj,k)

    # If it's a regular dataset
    return ds[()]

def load_group(group,k):
    
    """Recursively loads an HDF5 group into a nested dictionary"""
    result = {}
    try:
        for key, item in group.items():
            if isinstance(item, h5py.Dataset):
                result[key] = read_dataset(item,k)
            elif isinstance(item, h5py.Group):
                result[key] = load_group(item,k)
            else:
                print(f'Unknown item type: {type(item)}')
    except Exception as e:
        print(f'Error {e} occurred for file {k}')
        result[k]=read_dataset(group,k)
    return result

def loadData(dirc, file, fileName):

    """Loads structured HDF5 data safely into nested dictionaries and DataFrames"""
    datafile = f'{dirc}{file}'
    result = {}
    with h5py.File(datafile, 'r') as f:
        keys = list(f.keys())
        print(keys)
        for key in keys:
            result[key] = load_group(f[key],fileName)
    # with h5py.File(datafile, 'r') as f:
    #     result = load_group(f)
    print(f'Finished loading {datafile}')
    return result

def getTrialInfoCols(df, key, mk, lNum):

    """Extracts and reorders trialinfo as necessary"""
    col = df[key][mk][lNum]

    # If it's a list of arrays
    if isinstance(col, list):
        # Flatten and combine them
        new_col = []
        for c in col:
            if isinstance(c, np.ndarray):
                new_col.append(c.flatten()[0])  # Assuming each c is like array([[value]])
            else:
                new_col.append(c)
        return new_col

    # If it's a single array
    elif isinstance(col, np.ndarray):
        return col.flatten()

    else:
        raise TypeError(f"Unexpected type {type(col)} for column")
        
def DecodingAcc(classify_cond, decoderType, subjNum, decodingAnalysis):
    #Downsampling rate -- based on Nyquist
    downsample_rate =250

    #confine to correct trials only
    correct_only = 1
    if correct_only==1:
        corr_suffix = 'correctOnly'
    else:
        corr_suffix = []

    #**run after creating pseudotrials (sub-averaging trials prior to running
    #classification to improve SNR)
    run_pseudotrials = 0 #0 = no pseudotrial averaging, 10 = loop over 10 repetitions of pseudotrial averaging
    avg_pseudotrials = 14 #*set number of trials within each cond to average over
    if run_pseudotrials == 0:
        pseu_suffix = 'noPseudoTrials'
    else:
        pseu_suffix = [num2str(run_pseudotrials),'PseudoTrialsAvgOver',num2str(avg_pseudotrials),'Trials']

    #Set validation type --> how are you cross-validating it
    validation_type = '10fold'

    #Response lock info
    resp_pre = 0.5
    resp_post = 0.5

    #determine values used in classification 
    if classify_cond =='Left':
        cond_code = [1,2]
    elif classify_cond =='Right':
        cond_code = [3,4]
    else:
        cond_code=[1,2,3,4]

    #Name directories and make any necessary ones
    runFrom = 'FND4' #fnd4 or cpro2_eeg
    dataType = 'RawSensor' #sensor or source

    if runFrom =='FND4':
        baseDir = '/home/let83/FND4/'
    else:
        baseDir = '/projectsn/f_mc1689_1/cpro2_eeg/'
    output_dir = f'{baseDir}results/DynamicDecoding/{dataType}/{decoderType}/{classify_cond}/'
    output_file = f'_SubjectDecoding_{validation_type}_{classify_cond}_{corr_suffix}_{pseu_suffix}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    files2load = ['fsample','elec','hdr','trialinfo','sampleinfo','trial','time','label','cfg']
    for file in files2load:
        input_dir = f'{baseDir}results/preproc1_causalFilter/sub{subjNum}/'
        input_file = f'{file}_hp0.1notch_seg_autochannelz3_trialz3_ICAz3_baselinelp125avgref.mat'
        #globals()[f'left_{key}']
        print(file)
        
        try:
            if file == 'fsample':
                fsample = loadData(input_dir,input_file, file)
            elif file =='elec':
                elec = loadData(input_dir,input_file, file)
            elif file=='hdr':
                hdr = loadData(input_dir,input_file, file)
            elif file =='trialinfo':
                trialinfo = loadData(input_dir,input_file, file)
            elif file == 'sampleinfo':
                sampleinfo = loadData(input_dir,input_file, file)
            elif file == 'trial':
                trial =  loadData(input_dir,input_file, file)
            elif file=='time':
                time = loadData(input_dir,input_file, file)
            elif file=='label':
                label = loadData(input_dir,input_file, file)
                nChans = len(label['#refs#']) - 1
            elif file =='cfg':
                cfg = loadData(input_dir,input_file, file)
        except Exception as e:
            print(f'\nERROR: file {file} had error {e}/n')
    print('Loaded Data')
    #Actually Define necessary files (trialinfo, sampleinfo, time, trials)
    #Trialinfo
    subsystems = (trialinfo['#subsystem#'])
    ti = {'TaskCode':[],'acc':[],'resp':[],'rt':[]}
    colNames = ['TaskCode','acc','resp','rt']
    i=0
    for coln in colNames:
        ti[coln] = getTrialInfoCols(subsystems,'MCOS',2,i)
        i+=1
    trialinfo = pd.DataFrame(ti)
    print('Defined trialinfo')
    #Trials
    trial.keys()
    nTimepoints = 3945
    nTrial = 360
    trials_real = np.zeros((nTimepoints, nChans, nTrial))
    tkeys = list(trial['#refs#'].keys()) #Each key is a trial
    for tnum in range(360):
        try:
            tk = tkeys[tnum]
            trials_real[:,:,tnum] = trial['#refs#'][tk]
        except Exception as e:# one row should not have the right dimensions as there are 361 keys and only 360 trials
            print(f'ERROR {e} at {tkeys[tnum]},{tnum}!!!\n')
    print('Defined trials')
    #Define samples 
    samples = pd.DataFrame({'Start':sampleinfo['sampleinfo']['sampleinfo'][0],'End':sampleinfo['sampleinfo']['sampleinfo'][1]})
    print('defined samples')
    #Time and trialinfo need to be in the same format
    time_df = pd.DataFrame()
    for i in range(360):
        newTrial = pd.DataFrame(list(time['time']['time'][i]))
        nt = newTrial.T
        time_df = pd.concat([time_df,nt],ignore_index=False)
    print('Defined time')
    
    #Response Lock Data
    data_resp = {'trialinfo':trialinfo,
                 'sampleinfo':samples, 
                 'time' : time_df,
                 'trial': []}
    skipped_trials = []
    resp_secs = pd.DataFrame()
    time_new = pd.DataFrame()
    for t in range(len(trialinfo['rt'])):
        start_resp=(trialinfo['rt'][t]/1000)-resp_pre #RT-500
        end_resp=(trialinfo['rt'][t]/1000)+resp_post #RT+500
        newRow = pd.DataFrame({'Start':[start_resp], 'End':[end_resp]})
        # print(newRow)
        resp_secs=pd.concat([resp_secs,newRow],ignore_index=False)
        #find start and end inds in .time (round to deal with floating point discrepancies)
        times = time_df.iloc[t].values
        start_ind=np.where(np.round(times, 3) == np.round(start_resp, 3))[0]
        end_ind = np.where(np.round(times, 3) == np.round(end_resp, 3))[0]

        if len(start_ind) == 0 or len(end_ind) == 0:
            print(f"Warning: No matching start or end index for trial {t}. Skipping trial.")
            skipped_trials.append(t)
            continue  # skip trials that weren't answered

        time_row = pd.DataFrame(list(np.arange(-0.5, 0.5, 0.001))).T
        time_new = pd.concat([time_new,time_row],ignore_index=False)
        trl = trials_real[:, :, t]  # (channels, timepoints) for each trial
        trl = trl[start_ind[0]:end_ind[0]+1,:]  # +1 to include endpoint like MATLAB

        # Append to data_resp['trial']
        data_resp['trial'].append(trl)

    # Update the time field for each trial
    data_resp['time'] = time_new
    print('Response locked data')
        
    ##Dynamic Decoding Preproc
    # organize trial data so it is in the format (timepoints, channels) --> (n_trials, n_channels, n_timepoints)
    data_stack = np.stack(data_resp['trial'], axis=0) 
    data_for_resample = np.transpose(data_stack, (0, 2, 1))

    # resample trials
    resampled_data = mne.filter.resample(data_for_resample, up=1, down=4, axis=-1)

    # update data_resp to reflect downsampling
    data_resp['trial'] = resampled_data
    print(f'Downsampled data: {resampled_data.shape}')
    
    ##Sort trials that will be classified
    cond_info = trialinfo[(trialinfo['acc'] == 1) & (trialinfo['resp'].isin(cond_code))].reset_index(drop=True)
    cond_idx = cond_info.index.tolist()

    if classify_cond=='Left':
        cond_targets = cond_info[:]['resp']
    elif classify_cond=='Right':
        cond_targets = []
        for i in range(len(cond_info)):
            if cond_info.iloc[i]['resp'] == 3:
                cond_targets.append(1)
            elif cond_info.iloc[i]['resp'] == 4:
                cond_targets.append(2)
    elif classify_cond=='Hand':
        cond_targets = []
        for i in range(len(cond_info)):
            if cond_info.iloc[i]['resp'] == 3:
                cond_targets.append(1)
            elif cond_info.iloc[i]['resp'] == 4:
                cond_targets.append(2)
            elif cond_info.iloc[i]['resp']==1:
                cond_targets.append(1)
            else:
                cond_targets.append(2)
    print('defined cond_targets')
    
    #Time lock the data

    #make/organize data necessary for mne timelocking function 
    #data (trial x channels x time)
    #is resampled_data --> no reorganization needed

    #Info abt electrodes and data collection
    print(resampled_data.shape)
    n_channels = resampled_data.shape[1]

    print(n_channels)
    sfreq = 250  # sampling frequency after downsampling
    ch_names = [f'EEG {i:03d}' for i in range(n_channels)]
    #print(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    #events (trial number x dummy col x event ID)
    n_trials = resampled_data.shape[0]
    events = np.column_stack((np.arange(n_trials), np.zeros(n_trials, dtype=int), np.ones(n_trials, dtype=int)))
    #Timelock data
    data_trl = mne.EpochsArray(resampled_data, info, events=events, event_id={'response': 1}, tmin=-0.5)
    print(f'Resampled data, new shape: {data_trl.get_data().shape}')
    
    xx_cond_inds = random.sample(range(len(cond_targets)), len(cond_targets))
    
    #Make randomized cond targets
    random_cond_targets = []
    for i in xx_cond_inds:
        random_cond_targets.append(cond_targets[i])
    print('Randomized cond_target and its indices')
    
    # Loop over each randomized index
    nTimepoints = 250
    random_data_trl = np.zeros((len(xx_cond_inds), n_channels, nTimepoints))
    for idx in xx_cond_inds:
        trlData = data_trl.get_data()[idx]     # pull trial data
        random_data_trl[idx, :, :] = trlData   # final shape = (319, 251, 250)
    print('randomized trial order within a matrix of trial data')
    
    #Set up Dynamic Decoding
    #Set up basic variables
    nFolds = 10
    nTrials = len(xx_cond_inds)
    
    # Set up 10-fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=42)

    # matrix to collect accuracy info
    n_timepoints = np.array([i for i in range(250)])
    
    #Determine folds before entering loop
    X = random_data_trl[:, :, 1]
    folds_mat = []
    for train, test in cv.split(X, random_cond_targets):
        folds_mat.append((train,test))
    print('Completed setup for decoding, defined train and and test folds')
    return cond_code, random_data_trl, folds_mat,random_cond_targets
