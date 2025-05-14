#import functions
#General python use functions
import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import random

print('got through first functions')
#Importing data 
import h5py,pkg_resources,sys
print('impored h5py etc')
import os,glob,warnings,shutil
#Clean data
import mne
print('imported mne')

#Running decoders and classification 
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
print('imported sklearn scripts')
#Run any statistics
from scipy import signal, stats
from scipy.stats import sem
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.multitest import multipletests
import math
print('imported scipy sm and math')

#MAY NOT NEED
#read out matrices to files
# import io

#Parallel loop
# from joblib import Parallel, delayed
print('imported last modules')

def decode_single_timeseries(t,decoderType,cond_code, random_data_trl, folds_mat, random_cond_targets):
    #Define number of timepoints in the session
    nTimepoints = random_data_trl.shape[2]
    X = random_data_trl[:, :, t]#Extract data to run classificantion and decoding across one timepoint 
                                #(trials x channels x 1 timepoint)
    accuracies = []#make an empty list that classification accuracies at each fold can be added to
    for train_idx, test_idx in folds_mat:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = random_cond_targets[train_idx], random_cond_targets[test_idx]
        
        #Initialize a NEW classifier each time (important!)
        if decoderType =='SVM':
            clf = SVC(kernel='linear')
        elif decoderType == 'LDA':
            clf = LinearDiscriminantAnalysis()
        elif decoderType == 'Random_Forest':
            clf = RandomForestClassifier(n_estimators=200, random_state=42)#estimators is at 250 because I am working with high dimensional data, the standard starting point is 150, EEG data is typically between 100 and 300
            
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
    #Return the average and standard error of the accuracy across folds at each timepoint
    return np.mean(accuracies), sem(accuracies) 

# Make function to run temporal generalization 
def decode_TempGen(t_train, nTimepoints, decoderType, random_data_trl, folds_mat, random_cond_targets):
    # Define number of trials and channels in the data
    n_trials, n_channels, _ = random_data_trl.shape
    # Create training data set at specific timepoint
    X_train_data = random_data_trl[:, :, t_train]
    # Initialize empty array to hold decoding accuracy across test timepoints for a single train time
    row_acc = np.zeros(nTimepoints)

    # Loop over all test timepoints (one row per train timepoint)
    for t in range(nTimepoints):
        # Extract data at time t across all trials, shape: (n_trials, n_channels) 
        X = random_data_trl[:, :, t]
        # 10-fold cross-validation + decoding
        accuracies = []
        for train_idx, test_idx in folds_mat:
            X_train, X_test = X_train_data[train_idx], X[test_idx]
            y_train, y_test = random_cond_targets[train_idx], random_cond_targets[test_idx]
            
            # Define decoder types using if statements, initialize a new classifier at each timepoint
            if decoderType == 'SVM':
                clf = SVC(kernel='linear')
            elif decoderType == 'LDA':
                clf = LinearDiscriminantAnalysis()
            elif decoderType == 'Random_Forest':
                clf = RandomForestClassifier(n_estimators=250, random_state=42)

            # Fit model and evaluate
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)

        # At train_time and current test_timepoint of that timeseries,
        # add in the average accuracy at that timepoint into current row
        row_acc[t] = np.mean(accuracies)

    # Return the row of decoding accuracies (one train_time across all test_times)
    return row_acc

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
        
def DecodingAcc(classify_cond, decoderType, subjNum, decodingAnalysis, data_lock):
    print('starting DecodingAcc')

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

    #Define response window for response locking
    resp_pre = 0.75
    resp_post = 0.75

    #determine values used in classification 
    if classify_cond =='Left':
        cond_code = [1,2]
    elif classify_cond =='Right':
        cond_code = [3,4]
    else:
        cond_code=[1,2,3,4]
    
    #Name directories and make any necessary ones
    runFrom = 'cpro2_eeg' #fnd4 or cpro2_eeg
    dataType = 'RawSensor' #sensor or source
    print('got through basic naming')
    if runFrom =='FND4':
        baseDir = '/home/let83/FND4/'
    else:
        baseDir = '/projectsn/f_mc1689_1/cpro2_eeg/data/'
    output_dir = f'{baseDir}results/DynamicDecoding/{dataType}/{decoderType}/{classify_cond}/{decodingAnalysis}'
    output_file = f'_SubjectDecoding_{validation_type}_{classify_cond}_{corr_suffix}_{pseu_suffix}'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('made dir')
    files2load = ['fsample','elec','hdr','trialinfo','trial','time','label','cfg']
    for file in files2load:
        input_dir = f'{baseDir}results/preproc1_causalFilter/sub{subjNum}/'
        input_file = f'{file}_hp0.1notch_seg_autochannelz3_trialz3_ICAz3_baselinelp125avgref.mat'
        print('named input dir and file')
        print(file)
        
        try:
            if file == 'fsample':
                fsample = loadData(input_dir,input_file, file)
            elif file =='elec':
                elec = loadData(input_dir,input_file, file)
            elif file=='hdr':
                hdr = loadData(input_dir,input_file, file)
            elif file =='trialinfo':
                print('got trialinfo')
                trialinfo = loadData(input_dir,input_file, file)
                print('defined trialinfo')
                print(trialinfo.keys())
            elif file == 'trial':
                trial =  loadData(input_dir,input_file, file)
                print(trial.keys())
            elif file=='time':
                time = loadData(input_dir,input_file, file)
            elif file=='label':
                label = loadData(input_dir,input_file, file)
                nChans = len(label['#refs#']) - 1
                print(f'nChans is {nChans}')
            elif file =='cfg':
                cfg = loadData(input_dir,input_file, file)
        except Exception as e:
            print(f'\nERROR: file {file} had error {e}/n')
    print('Loaded Data')
    #Define necessary files as dataframes that only contains the necessary info and can be easily manipulated
    #Trialinfo (contains a code that describes the following:
    #               combination of sensory, logic, and motor rules, whether the logic rule was same/different (equal to 2) or                           either/neither (equal to 1)
    #               the accuracy (0 or 1)
    #               the response made (1= left hand, middle finger; 2= left hand, index; 3 = right hand, index; 4 = right hand, middle)
    #               the reaction time of the response in ms)
    subsystems = (trialinfo['#subsystem#'])
    ti = {'TaskCode':[],'LogicRule':[],'acc':[],'resp':[],'rt':[]}
    colNames = ['TaskCode','LogicRule','acc','resp','rt']
    i=0
    for coln in colNames:
        ti[coln] = getTrialInfoCols(subsystems,'MCOS',2,i)
        i+=1
    trialinfo = pd.DataFrame(ti)
    print('Defined trialinfo')
    print('Logic rule list:')
    print(trialinfo['LogicRule'])
    #Trials -- add trials into a numpy array
    trial.keys() #Figure out which keys are in trials to find data
    nTimepoints = 494 #Number of timepoints in the trial
    nTrial = 360 #Max number of trials
    trials_real = np.zeros((nTimepoints, nChans, nTrial))
    tkeys = list(trial['#refs#'].keys()) #Each key is a trial
    for tnum in range(360):
        try:
            tk = tkeys[tnum]
            trials_real[:,:,tnum] = trial['#refs#'][tk]
        except Exception as e:# one row should not have the right dimensions as there are 361 keys and only 360 trials
            print(f'ERROR {e} at {tkeys[tnum]},{tnum}!!!\n')
    print('Defined trials')
    #Time and trialinfo need to be in the same format
    time_df = pd.DataFrame()
    for i in range(360):
        newTrial = pd.DataFrame(list(time['time']['time'][i]))
        nt = newTrial.T
        time_df = pd.concat([time_df,nt],ignore_index=False)
    print('Defined time')
    
    #Lock Data to stimulus or response 
    if data_lock=='response':
        numCorrectTrls = 0
        for t in range(len(trialinfo['rt'])):
            if trialinfo['acc'][t]==1:
                numCorrectTrls+=1
        # print(numCorrectTrls)
        data_resp = {'trialinfo':trialinfo,
                     'time' : time_df,
                     'trial': []}
        skipped_trials = []
        resp_time = 1500
        indx = 0

        s_ind = []
        e_ind = []
        for t in range(len(trialinfo['rt'])):
            if trialinfo['acc'][t]==1: 
                start_resp=((trialinfo['rt'][t]/1000)-resp_pre) #start of window = RT-(trial window/2)
                end_resp=((trialinfo['rt'][t]/1000)+resp_post) #end of window = RT-(trial window/2)
                newRow = pd.DataFrame({'Start':[start_resp], 'End':[end_resp]})

                resp_secs=pd.concat([resp_secs,newRow],ignore_index=False)
                #find start and end inds in .time (round to deal with floating point discrepancies)
                times = time_df.iloc[t].values
                #use inds to reference data_input.trial
                start_ind=np.where(np.round(times, 3) == np.round(start_resp, 3))[0]
                end_ind = np.where(np.round(times, 3) == np.round(end_resp, 3))[0]

                #This skips any trials that do not line up exactly with window. Ensures that all trials are the same length 
                if len(start_ind) == 0 or len(end_ind) == 0: 
                    print(f"Warning: No matching start or end index for trial {t}. Skipping trial.")
                    skipped_trials.append(t)
                    continue  # skip this trial

                #*seems like fieldtrip call to redefinetrial includes epoch end timepoints for stim-locked data; 
                #hence for consistency not removing the last ind for resp_locked
                time_row = pd.DataFrame(list(np.arange(-1*resp_pre, resp_post, 0.001))).T
                time_new = pd.concat([time_new,time_row],ignore_index=False)

                #get start and end index lists 
                s_ind.append(start_ind[0])
                e_ind.append(end_ind[0])

                indx+=1

        # Update the time field for each trial -- optional
        data_resp['time'] = time_new

        # Get the number of channels and timepoints per trial
        n_trials = len(trialinfo)
        n_timepoints = trials_real.shape[0]
        n_chans = trials_real.shape[1]

        #Plot ERPs of response
        # Preallocate trial matrices
        valid_C3_full = []
        valid_C4_full = []

        valid_C3_resp = []
        valid_C4_resp = []

        valid_resp_trls = []
        valid_full_trls = []

        # Track accepted RTs
        valid_rts = []
        valid_trial_indices = []

        #Define timeseries during actual window, ERP in C3 and C4 electrodes 
        for t in range(n_trials):
            if trialinfo['acc'][t] != 1:
                continue  # skip incorrect trials

            times = time_df.iloc[t].values
            trl = trials_real[:, :, t]  # shape = (timepoints, channels)

            rt = trialinfo['rt'][t] / 1000  # ms → sec
            start_time = rt - resp_pre
            end_time = start_time + 1.5  # 1500 ms window

            start_ind = np.argmin(np.abs(times - start_time))
            end_ind = start_ind + 1500
            
            #Skip trials with the incorrect shape
            if end_ind > len(times):
                print(f"Skipping trial {t} due to short duration")
                continue

            # Slice trials and store response locked and full trial erps
            trl_new = trl[start_ind:end_ind, :]
            valid_resp_trls.append(trl_new)
            valid_full_trls.append(trl)

            valid_C3_full.append(trl[:, 54])   # C3
            valid_C4_full.append(trl[:, 182])  # C4

            valid_C3_resp.append(trl[start_ind:end_ind, 54])
            valid_C4_resp.append(trl[start_ind:end_ind, 182])

            valid_rts.append(rt)
            valid_trial_indices.append(t)

        # Convert to arrays: shape (timepoints, trials)
        C3_full = np.column_stack(valid_C3_full)
        C4_full = np.column_stack(valid_C4_full)
        C3_resp = np.column_stack(valid_C3_resp)
        C4_resp = np.column_stack(valid_C4_resp)

        # Stack full trial slices (with all channels)
        resp_trls = np.stack(valid_resp_trls, axis=2)  # shape: (1500, n_channels, n_trials)
        full_trls = np.stack(valid_full_trls, axis=2)  # shape: (1500, n_channels, n_trials)
        resp_ind_trls = full_trls[int(round(np.mean(s_ind),0)):int(round(np.mean(e_ind),0)),:]
        print("resp_trls shape:", resp_trls.shape)
        data_resp['trial'] = resp_trls
        data_input = data_resp

        print('Response locked data')
        #Plot all ERP graphs (full trial, based on average ERP window, specifically locked data)
        time_r = np.linspace(-1*resp_pre, resp_post, C3_resp.shape[0])

        plt.figure(figsize=(14,9))
        plt.xlabel('Time',fontsize = 13)
        plt.ylabel('Voltage', fontsize = 13)
        times = [i/1000 for i in range(len(np.mean(C3_full, axis=1)))]
        plt.plot(times, np.mean(C3_full, axis=1), color = 'mediumorchid',label='C3')
        plt.fill_between(times, np.mean(C3_full, axis=1) - sem(C3_full, axis=1), np.mean(C3_full, axis=1) + sem(C3_full, axis=1), alpha=0.3, color='plum')
        plt.axvline(times[int(round(np.mean(s_ind),0))], color='k', linestyle='--')
        plt.axvline(np.mean(rt), color='k', linestyle='--')
        plt.axvline(times[int(round(np.mean(e_ind),0))], color='k', linestyle='--')
        plt.plot(times, np.mean(C4_full, axis=1), color = 'steelblue',label='C4')
        plt.fill_between(times, np.mean(C4_full, axis=1) - sem(C4_full, axis=1), np.mean(C4_full, axis=1) + sem(C4_full, axis=1), alpha=0.3, color='lightskyblue')
        plt.grid()
        plt.title('Whole brain ERP across full trial', fontsize = 17)
        plt.legend()

        plt.figure(figsize=(14,9))
        plt.xlabel('Time',fontsize = 13)
        plt.ylabel('Voltage', fontsize = 13)
        times = [i/1000 for i in range(len(np.mean(C3_resp, axis=1)))]
        c3_ind = C3_full[int(round(np.mean(s_ind),0)):int(round(np.mean(e_ind),0)),:]
        c4_ind = C4_full[int(round(np.mean(s_ind),0)):int(round(np.mean(e_ind),0)),:]
        plt.plot(time_r, np.mean(c3_ind, axis=1), color = 'mediumorchid',label='C3')
        plt.fill_between(time_r, np.mean(c3_ind, axis=1) - sem(c3_ind, axis=1), np.mean(c3_ind, axis=1) + sem(c3_ind, axis=1), alpha=0.3, color='plum')
        plt.plot(time_r, np.mean(c4_ind, axis=1), color = 'steelblue',label='C4')
        plt.fill_between(time_r, np.mean(c4_ind, axis=1) - sem(c4_ind, axis=1), np.mean(c4_ind, axis=1) + sem(c4_ind, axis=1), alpha=0.3, color='lightskyblue')
        plt.grid()
        plt.legend()
        plt.title('Whole Brain ERP across Response Locked Data -- full indexing', fontsize = 17)

        plt.figure(figsize=(14,9))
        plt.xlabel('Time',fontsize = 13)
        plt.ylabel('Voltage', fontsize = 13)
        times = [i/1000 for i in range(len(np.mean(C3_resp, axis=1)))]
        # plt.plot(time_r, C3_resp[:, :10])  # plot first 10 trials

        plt.plot(time_r, np.mean(C3_resp, axis=1), color = 'mediumorchid',label='C3')
        plt.fill_between(time_r, np.mean(C3_resp, axis=1) - sem(C3_resp, axis=1), np.mean(C3_resp, axis=1) + sem(C3_resp, axis=1), alpha=0.3, color='plum')
        plt.plot(time_r, np.mean(C4_resp, axis=1), color = 'steelblue',label='C4')
        plt.fill_between(time_r, np.mean(C4_resp, axis=1) - sem(C4_resp, axis=1), np.mean(C4_resp, axis=1) + sem(C4_resp, axis=1), alpha=0.3, color='lightskyblue')
        plt.grid()
        # plt.ylim([-10,7.5])
        plt.legend()
        plt.title('Whole Brain ERP across Response Locked Data -- resp dataset', fontsize = 17)
        print('graphed response locked data')
        
    elif data_lock=='stimulus':
        #Stimulus Lock Data
        data_stim = {'trialinfo':trialinfo,
                     'time' : time_df,
                     'trial': []}
        print('defined data_stim')
        skipped_trials = []
        time_new = pd.DataFrame()
        # Preallocate trial matrices
        valid_full_trls = []

        # Track accepted RTs
        valid_rts = []
        valid_trial_indices = []
        print('defined initial variables')
        #Define the stimulus locked timeseries, actual data is not sliced
        for t in range(len(trialinfo['rt'])):
            if trialinfo['acc'][t]==1: 
                # print(start_ind)
                time_row = pd.DataFrame(list(np.linspace(-0.09, 3.855, nTimepoints))).T
                time_new = pd.concat([time_new,time_row],ignore_index=False)

                times = time_df.iloc[t].values
                trl = trials_real[:, :, t]  # shape = (timepoints, channels)

                # Slice trials and store response locked and full trial erps
                valid_full_trls.append(trl)
                valid_trial_indices.append(t)
        print('made it through loop')

        # Update the time 
        data_stim['time'] = time_new
        print('updated data_stim time')

        # Stack full trial slices (with all channels)
        full_trls = np.stack(valid_full_trls, axis=2)  # shape: (1500, n_channels, n_trials)
        print("full_trls shape:", full_trls.shape)
        data_stim['trial']= full_trls
        # Update data name
        data_input = data_stim
        print(data_input.keys())
        print('Updated data_input')
        
        # plot stim locked data
        #define time variable 
        times = np.linspace(-0.09,3.855, nTimepoints)
        print(len(times))

        #Averaged across trials
        #averaged across both trials and channels
        trl_avg = np.mean(data_input['trial'], axis=2)
        trl_sem = sem(data_input['trial'], axis=2)
        plt.figure(figsize=(14,9))
        plt.xlabel('Time',fontsize = 13)
        plt.ylabel('Voltage', fontsize = 13)
        plt.plot(times, trl_avg[:,54], color = 'mediumorchid')
        plt.fill_between(times, trl_avg[:,54] - trl_sem[:,54], trl_avg[:,54] + trl_sem[:,54], alpha=0.3, color='plum', label = 'C3')
        plt.plot(times, trl_avg[:,182], color = 'steelblue')
        plt.fill_between(times, trl_avg[:,182] - trl_sem[:,182], trl_avg[:,182] + trl_sem[:,182], alpha=0.3, color='lightskyblue', label = 'C4')
        plt.grid()
        plt.legend()
        plt.title(f'C3 and C4 ERP across Stimulus Locked Data\n of Subject {subjNum}-- averaged across trls', fontsize = 17)
        outfile = f'StimLocked_c3_c4_avgTrialData_of_Subject{subjNum}'
        output_dir_erp = f'{baseDir}results/DynamicDecoding/{dataType}/ERP_Graphs/{classify_cond}/'
        if not os.path.exists(output_dir_erp):
            os.makedirs(output_dir_erp)
        plt.savefig(os.path.join(output_dir_erp, outfile))
        print('plotted stim locked data')
    ##Dynamic Decoding Preproc 
    # Make sure the shape is (n_trials, n_channels, n_timepoints)
    data_trl = np.transpose(data_input['trial'], (2, 1, 0))  # (n_trials, n_channels, n_timepoints)

    # Sort trials that will be classified
    ##Skip those that should be dropped -- its a check there shouldn't be any trials dropped here
    trialinfo = trialinfo.drop(skipped_trials).reset_index(drop=True)
    #Filtering trialinfo to filter out inaccurate trials, trials with logic rules either or neither, and unanswered trials
    ##NOTE: Trials with logic rules either or neither were excluded because I am not supposed to response lock the trials and those trials can be answered at varying times depending on the stimuli presented
    cond_info = trialinfo[(trialinfo['acc'] == 1) & (trialinfo['LogicRule']==2) & (trialinfo['resp'].isin(cond_code))].reset_index(drop=True)
    cond_idx = cond_info.index.tolist()

    #Use response information to create a variable that targets and tries to differentiate the specific conditions I am trying to decode
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
    #Info abt electrodes and data collection
    print(data_trl.shape)
    n_channels = data_trl.shape[1]

    print(n_channels)
    sfreq = 125  # sampling frequency based on Nyquist theorem
    ch_names = [f'EEG {i:03d}' for i in range(n_channels)]
    #print(ch_names)
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')

    #Define events (trial number x dummy col x event ID)
    n_trials = data_trl.shape[0]
    events = np.column_stack((np.arange(n_trials), np.zeros(n_trials, dtype=int), np.ones(n_trials, dtype=int)))
    #Timelock data to -0.09 seconds
    data_trl = mne.EpochsArray(data_trl, info, events=events, event_id={data_lock: 1}, tmin=-0.09)
    print(f'data_trl, new shape: {data_trl.get_data().shape}')
    
    #Randomize indices of condition targets of trials
    xx_cond_inds = random.sample(range(len(cond_targets)), len(cond_targets))
    
    #Make randomized cond targets
    random_cond_targets = []
    for i in xx_cond_inds:
        random_cond_targets.append(cond_targets[i])
    random_cond_targets = np.array(random_cond_targets)
    print('Randomized cond_target and its indices')
    
    # Loop over each randomized index
    nTimepoints = data_trl.get_data().shape[2]
    random_data_trl = np.zeros((len(xx_cond_inds), n_channels, nTimepoints))
    for idx in xx_cond_inds:
        trlData = data_trl.get_data()[idx]     # pull trial data
        random_data_trl[idx, :, :] = trlData   # final shape = (319, n_electrodes, 494 timepoints)
    print('\nshape of random_data_trl')
    print(random_data_trl.shape)
    print('randomized trial order within a matrix of trial data')
    
    #Set up Dynamic Decoding
    #Set up basic variables
    nFolds = 10
    nTrials = len(xx_cond_inds)
    
    # Set up 10-fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=nFolds, shuffle=True, random_state=42)

    # matrix to collect accuracy info
    n_timepoints = np.array([i for i in range(nTimepoints)])
    
    #Determine folds before entering loop
    X = random_data_trl[:, :, 1]
    folds_mat = []
    for train, test in cv.split(X, random_cond_targets):
        folds_mat.append((train,test))
    print('Completed setup for decoding, defined train and and test folds')
    return output_dir, output_file, cond_code, random_data_trl, folds_mat,random_cond_targets
