import os;
import h5py, pkg_resources;
import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
from joblib import Parallel, delayed
import sys;
import glob,warnings,shutil;
import mne;
import random;
import io;
from sklearn.model_selection import train_test_split;
from sklearn.metrics import accuracy_score, confusion_matrix;
from sklearn.model_selection import StratifiedKFold;
from sklearn.svm import SVC;
sys.path.append('/home/let83/FND4/Analyses/')
from Step2a_run_decoders_fxns import DecodingAcc
subjNum = 34
"""
Runs single decoding analysis for a given subject.
"""
DecodingAcc(
    classify_cond='Hand',
    decoderType='SVM',
    subjNum=subjNum,
    decodingAnalysis='DecodingAccuracy',
)

