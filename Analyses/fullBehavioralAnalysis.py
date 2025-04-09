#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 20:13:09 2024

@author: lthompson
"""


#FULL BEHAVIORAL ANALYSIS CODE 

!pip install pandas numpy matplotlib statsmodels seaborn scipy openpyxl scikit-learn

'''
Import packages
'''
from scipy import signal,stats
from scipy.stats import wilcoxon, norm, ttest_rel, sem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LinearRegression

'''
Import behavioral dataset
'''
behavioralDfOG = pd.read_excel('~/projectsn/f_mc1689_1/cpro2_eeg/data/rawdata/CPRO2_fMRI_EEG_main.xlsx')#There might be an error here having to do with the path
session2OG = pd.read_excel('~/projectsn/f_mc1689_1/cpro2_eeg/data/results/CPRO2_main_fMRI_test.xlsx')

''' 
Remove the first (prac) and last (nov) 12 blocks of each participant
'''
index = 0 
#make an empty dataframe that rows will be added to
behavioralDf1 = pd.DataFrame()
for i in behavioralDfOG['Subject']:
    if behavioralDfOG.iloc[index, behavioralDfOG.columns.get_loc('BlockList')]<=12: 
        pass
    elif behavioralDfOG.iloc[index, behavioralDfOG.columns.get_loc('BlockList')]>108:
        pass
    else:
        newRow = behavioralDfOG.loc[[index]]
        behavioralDf1 = pd.concat([behavioralDf1, newRow], axis=0)
    index+=1
#Make list of subject accuracy percentages
subjs = list(behavioralDf1['Subject'])
endIndex = len(subjs)
percCorr = list(behavioralDf1['percentCorrectRun'])
blockNum = list(behavioralDf1['Block'])

#do the same with session 2
index = 0 
session2 = pd.DataFrame()
for i in session2OG['Subject']:
    if session2OG.iloc[index, session2OG.columns.get_loc('BlockList')]<=12: 
        pass
    elif session2OG.iloc[index, session2OG.columns.get_loc('BlockList')]>108:
        pass
    else:
        newRow = session2OG.loc[[index]]
        session2 = pd.concat([session2, newRow], axis=0)
    index+=1
session2=session2.dropna()
#Make list of subject accuracy percentages
subjs2 = list(session2['Subject'])
endIndex2 = len(subjs2)
percCorr2 = list(session2['percentCorrectRun'])
blockNum2 = list(session2['Block'])
'''
Block column spans 1-10 for each subject 
Each "Block" corresponds 1 accuracy percentage
Each "Block" contains 12 blocks from blockList (aka 109-120)
'''
index = 0
subjsAccuracy = []
subjAvgAcc = 0 
blocknums = 0
for i in subjs:
    if index+1!=endIndex:#Find and add average accuracy percentage at each participant excluding the last one
        if subjs[index+1]!=i:
            #Add in final average information
            blocknums+=1
            subjAvgAcc+=percCorr[index]
            #add average accuracy percentage to subjAvgAcc
            subjsAccuracy.append(round((subjAvgAcc/blocknums),4))
            #Reset average info
            blocknums=0
            subjAvgAcc=0
        else:
            if blockNum[index+1]!=blockNum[index]:#Update avg accuracy percentage info at each "Block" change within a participant
                blocknums+=1
                subjAvgAcc+=percCorr[index]
            else: pass
    else: #at the end of the list, add the average accuracy percentage for the last subject
        #Add in final average information
        blocknums+=1
        subjAvgAcc+=percCorr[index]
        #add average accuracy percentage to subjAvgAcc
        subjsAccuracy.append(round((subjAvgAcc/blocknums),4))
    index+=1

#Run Wilcoxon t test on whether subject accuracy is above chance
chance = 50
diff = [x for x in subjsAccuracy]
tStat, p = stats.ttest_1samp(diff, chance)
print(f'T Statistic: {tStat}\np-value: {p}\n')
if p<0.05: print('As the p-value is less than 0.05, we find significant\nevidence that the participants accuracy in the tasks\nis not due to chance, and that learning did occur.\n')

#do the same with session2
index = 0
subjsAccuracy2 = []
subjAvgAcc2 = 0 
blocknums2 = 0
for i in subjs2:
    try:
        if index+1!=endIndex2:#Find and add average accuracy percentage at each participant excluding the last one
            if subjs2[index+1]!=i:
                #Add in final average information
                blocknums2+=1
                subjAvgAcc2+=percCorr2[index]
                #add average accuracy percentage to subjAvgAcc
                subjsAccuracy2.append(round((subjAvgAcc2/blocknums2),4))
                #Reset average info
                blocknums2=0
                subjAvgAcc2=0
            else:
                if blockNum2[index+1]!=blockNum2[index]:#Update avg accuracy percentage info at each "Block" change within a participant
                    blocknums2+=1
                    subjAvgAcc2+=percCorr2[index]
                else: pass
        else: #at the end of the list, add the average accuracy percentage for the last subject
            #Add in final average information
            blocknums2+=1
            subjAvgAcc2+=percCorr2[index]
            #add average accuracy percentage to subjAvgAcc
            subjsAccuracy2.append(round((subjAvgAcc2/blocknums2),4))
    except Exception as e:
        print(f'unexpected error:   {e}')
        print(f'endIndex2 == {endIndex2}')
        print(f'index+1 == {index+1}')
    index+=1

#Run Wilcoxon t test on whether subject accuracy is above chance
diff2 = [x for x in subjsAccuracy2]
tStat2, p2 = stats.ttest_1samp(diff2, chance)
print(f'T Statistic: {tStat2}\np-value: {p2}\n')
if p2<0.05: print('As the p-value is less than 0.05, we find significant\nevidence that the participants accuracy in the tasks\nis not due to chance, and that learning did occur.\n')

'''
Plot accuracy above chance
'''
import random
#Plot as scatter with x as subject number and y as percent accuracy -- session3
plt.figure()
x = 1
plt.bar(x, sum(subjsAccuracy)/len(subjsAccuracy), width=.5, align='center', yerr = sem(subjsAccuracy, nan_policy='omit'), fill=False)
xs = [random.uniform(0.9, 1.1) for _ in range(len(subjsAccuracy))]
index=0
for i in subjsAccuracy:
    plt.scatter(xs[index],i,color='rebeccapurple')
    index+=1
plt.axhline(y = 50, color = 'plum', linestyle = '-', label='50% Accuracy')
plt.xlabel('Subject Average Accuracy', fontsize = 12)
plt.ylabel('Percent Accuracy', fontsize = 12)
plt.legend(loc="upper right", bbox_to_anchor=(1.51, 1), fontsize = 12)
plt.ylim(0,110)
plt.title('Accuracy (%) of Subjects During Testing -- Session 3', fontsize = 15)
plt.show
plt.savefig('~/projectsn/f_mc1689_1/CPRO2_learning/data/results/Accuracy_against_chance.png')

#Plot as scatter with x as subject number and y as percent accuracy -- session2
plt.figure()
x = 1
plt.bar(x, sum(subjsAccuracy2)/len(subjsAccuracy2), width=.5, align='center', yerr = sem(subjsAccuracy2, nan_policy='omit'), fill=False)
xs = [random.uniform(0.9, 1.1) for _ in range(len(subjsAccuracy2))]
index=0
for i in subjsAccuracy2:
    plt.scatter(xs[index],i,color='rebeccapurple')
    index+=1
plt.axhline(y = 50, color = 'plum', linestyle = '-', label='50% Accuracy')
plt.xlabel('Subject Average Accuracy', fontsize = 12)
plt.ylabel('Percent Accuracy', fontsize = 12)
plt.legend(loc="upper right", bbox_to_anchor=(1.51, 1), fontsize = 12)
plt.ylim(0,110)
plt.title('Accuracy (%) of Subjects During Testing -- Session 2', fontsize = 15)
plt.show
plt.savefig('~/projectsn/f_mc1689_1/CPRO2_learning/data/results/Accuracy_against_chance_s2.png')

'''
Write a function to make separate matrices to analyze subject averaged probe info,
separating novel and practice conditions 
'''
#Function writing
def organizeAvgREAL(dataC,df,col,rows):
    #make an empty matrix that you can add data value averages to per condition (rows) and subject (columns)
    rs = list(set(df[rows]))
    rn = 2*len(rs)
    emptyM = np.zeros((rn,3)) 
    index = 0
    #make values to calculate subjectCondition averages
    rn=0
    rln=0
    r2=0
    rl2=0
    indi=0
    for i in df[rows]: #loop through Subjects
        try:#get averages for all subjects except the last
            if df.iloc[index, df.columns.get_loc(col)]=='Novel': #novel condition averages of data value
                rn += df.iloc[index, df.columns.get_loc(dataC)]
                rln+=1
                blockCond='Novel'
            else: #practice condition averages of data value
                r2 += df.iloc[index, df.columns.get_loc(dataC)]
                rl2+=1
                blockCond='Practice'
                #make blockCond conditional 
            try:#at the transition from one subject to another add the condition averages to the matrix
                if  i<df.iloc[(index+1), df.columns.get_loc(rows)]:
                    emptyM[indi,1]=5
                    emptyM[indi,0]=round((rn/rln),4)
                    emptyM[indi,2]=i

                    indi+=1
                    emptyM[indi,0]=round((r2/rl2),4)
                    emptyM[indi,1]=1
                    
                    emptyM[indi,2]=i
                    #reset average variables for next subject
                    rn=0
                    r2=0
                    rln=0
                    rl2=0
                    indi+=1
                else: pass
            except:#add condition average for the last subject and data entry point
                emptyM[indi,1]=5
                emptyM[indi,0]=round((rn/rln),4)
                emptyM[indi,2]=i
                indi+=1
                emptyM[indi,0]=round((r2/rl2),4)
                emptyM[indi,1]=1
                emptyM[indi,2]=i
                indi+=1
            index+=1
        except:
            index+=1
    aNovel = []
    aPrac = []
    x = len(emptyM)
    for i in range(x):
        if emptyM[i,1]==5:
            aNovel.append(round(float(emptyM[i,0]),4))
        elif emptyM[i,1]==1:
            aPrac.append(round(float(emptyM[i,0]),4))
    return aNovel, aPrac, emptyM

#Function running
#Accuracy
aNovel1, aPrac1, aMatrix = organizeAvgREAL('Probe1.ACC',behavioralDf1,'BlockCond','Subject')
aNovel1S2, aPrac1S2, aMatrixS2 = organizeAvgREAL('Probe1.ACC',session2,'BlockCond','Subject')

#Session 3
#make in terms of 100%
aNovel = []
aPrac = []
for i in aNovel1:
    k = 100*i
    aNovel.append(round(k,4))
for i in aPrac1:
    k = 100*i
    aPrac.append(round(k,4))
print(f'\nNovel average accuracy:     {aNovel}')
print(f'\nPractice average accuracy:  {aPrac}')

#Session 2
aNovelS2 = []
aPracS2 = []
for i in aNovel1S2:
    k = 100*i
    aNovelS2.append(round(k,4))
for i in aPrac1S2:
    k = 100*i
    aPracS2.append(round(k,4))
print(f'\nNovel average accuracy - S2:     {aNovelS2}')
print(f'\nPractice average accuracy - S2:  {aPracS2}')

#Reaction Time
#If a participant doesn't chose an option it is counted as incorrect, and no reaction time is recorded
#this means that accuracy information is not impacted by NaN values, but reaction time is
#therefore, dropna is necessary for RT analyses, but would skew the accuracy analyses towards more accurate than reality
behavioralDf1 = behavioralDf1.dropna()
RTNovel, RTPrac, RTMatrix = organizeAvgREAL('Probe1.RT',behavioralDf1,'BlockCond','Subject')
print(f'\nNovel average reaction time:     {RTNovel}')
print(f'\nPractice average reaction time:  {RTPrac}')

#Find error
eNovel = []
ePrac = []
for i in aNovel:
    k = 100-i
    eNovel.append(round(k,4))
for i in aPrac:
    k = 100-i
    ePrac.append(round(k,4))
print(f'\nNovel average error:     {eNovel}')
print(f'\nPractice average error:  {ePrac}')

#Session 2
print('SESSION 2')
RTNovelS2, RTPracS2, RTMatrixS2 = organizeAvgREAL('Probe1.RT',session2,'BlockCond','Subject')
print(f'\nNovel average reaction time:     {RTNovelS2}')
print(f'\nPractice average reaction time:  {RTPracS2}')

#Find error
eNovelS2 = []
ePracS2 = []
for i in aNovelS2:
    k = 100-i
    eNovelS2.append(round(k,4))
for i in aPracS2:
    k = 100-i
    ePracS2.append(round(k,4))
print(f'\nNovel average error:     {eNovelS2}')
print(f'\nPractice average error:  {ePracS2}')


'''
Run t-tests for each probe to determine whether practice-novel
contrasts differ from 0
contrast against 0 and ttest_rel come up with the same exact values
'''

#Error Contrast
eContrast = []
for i in range(len(ePrac)):
    ec = ePrac[i]-eNovel[i]
    eContrast.append(ec)
t_eNP, p_eNP = stats.ttest_1samp(eContrast, 0)
t_Enp, p_Enp = ttest_rel(eNovel, ePrac, alternative= 'two-sided')
print('ERROR')
print(f'T-statistic: {t_eNP}\nP-Value: {p_eNP}')
if p_Enp<0.05: 
    print('As the p-value is less than 0.05, we find significant \nevidence that the participants error rates differ\nbetween novel and practice trials.\n')
else:
    print('As the p-value is not less than 0.05, we do not find \nsignificant evidence that the participants error rates differ\nbetween novel and practice trials.\n')

#Reaction Time
rtContrast = []
for i in range(len(RTNovel)):
    rtc = RTPrac[i]-RTNovel[i]
    rtContrast.append(rtc)
t_rtNP, p_rtNP = stats.ttest_1samp(rtContrast, 0)
t_RTnp, p_RTnp = ttest_rel(RTNovel, RTPrac, alternative= 'two-sided')
print('REACTION TIME')
print(f'T-statistic: {t_rtNP}\nP-Value: {p_rtNP}')
if p_RTnp>0.05: 
    print('As the p-value is not less than 0.05, we do not find \nsignificant evidence that the participants reaction time differs\nbetween novel and practice trials.\n')
else:
    print('As the p-value is less than 0.05, we find \nsignificant evidence that the participants reaction time differs\nbetween novel and practice trials.\n')

#Session 2
#Error Contrast
eContrastS2 = []
for i in range(len(ePracS2)):
    ec = ePracS2[i]-eNovelS2[i]
    eContrastS2.append(ec)
t_eNPS2, p_eNPS2 = stats.ttest_1samp(eContrastS2, 0)
t_EnpS2, p_EnpS2 = ttest_rel(eNovelS2, ePracS2, alternative= 'two-sided')
print('ERROR')
print(f'T-statistic: {t_eNPS2}\nP-Value: {p_eNPS2}')
if p_EnpS2<0.05: 
    print('As the p-value is less than 0.05, we find significant \nevidence that the participants error rates differ\nbetween novel and practice trials.\n')
else:
    print('As the p-value is not less than 0.05, we do not find \nsignificant evidence that the participants error rates differ\nbetween novel and practice trials.\n')

#Reaction Time
rtContrastS2 = []
for i in range(len(RTNovelS2)):
    rtc = RTPracS2[i]-RTNovelS2[i]
    rtContrastS2.append(rtc)
t_rtNPS2, p_rtNPS2 = stats.ttest_1samp(rtContrastS2, 0)
t_RTnpS2, p_RTnpS2 = ttest_rel(RTNovel, RTPrac, alternative= 'two-sided')
print('REACTION TIME')
print(f'T-statistic: {t_rtNPS2}\nP-Value: {p_rtNPS2}')
if p_RTnpS2>0.05: 
    print('As the p-value is not less than 0.05, we do not find \nsignificant evidence that the participants reaction time differs\nbetween novel and practice trials.\n')
else:
    print('As the p-value is less than 0.05, we find \nsignificant evidence that the participants reaction time differs\nbetween novel and practice trials.\n')

'''
Plot the practice and novel distributions as boxplots for error, accuracy, and reaction time (3 subplots)
'''
def findTots(Prac, Nov):
    Tot = []
    X = []
    for i in Prac:
        Tot.append(i)
        X.append('Practice')
    for i in Nov:
        Tot.append(i)
        X.append('Novel')
    return Tot, X
eTot, eX = findTots(ePrac, eNovel)
aTot, aX = findTots(aPrac, aNovel)
rtTot, rtX = findTots(RTPrac, RTNovel)
x_bar = ['Practice','Novel']
Ey_bar = [sum(ePrac)/len(ePrac),sum(eNovel)/len(eNovel)]
Eerrs = [sem(ePrac, nan_policy='omit'), sem(eNovel, nan_policy='omit')]
RTy_bar = [sum(RTPrac)/len(RTPrac),sum(RTNovel)/len(RTNovel)]
RTerrs = [sem(RTPrac, nan_policy='omit'), sem(RTNovel, nan_policy='omit')]

plt.figure()
fig, axes = plt.subplots(1,2, figsize=(11, 5))
fig.suptitle('Practice vs. Novel Distributions of Error and Reaction Time')
#Error subplot
axes[0].bar(x_bar,Ey_bar,width=0.5, align='center', yerr=Eerrs, fill=False, edgecolor='black')
axes[0].set_title('Error (%)')
axes[0].set_ylabel('Error (%)')
axes[0].set_ylim(0,100)
x_prac = [random.uniform(-0.1, 0.1) + 0 for _ in ePrac]  # Offset around x=0 ('Practice')
axes[0].scatter(x_prac, ePrac, color='darkorchid', alpha=.5)
x_novel = [random.uniform(-0.1, 0.1) + 1 for _ in eNovel]  # Offset around x=1 ('Novel')
axes[0].scatter(x_novel, eNovel, color='orchid', alpha=.5)
#Reaction time subplot
axes[1].bar(x_bar,RTy_bar,width=0.5, align='center', yerr=RTerrs, fill=False, edgecolor='black')
axes[1].set_title('Reaction Time (ms)')
axes[1].set_ylabel('Reaction Time (ms)')
x_prac = [random.uniform(-0.1, 0.1) + 0 for _ in RTPrac]  # Offset around x=0 ('Practice')
axes[1].scatter(x_prac, RTPrac, color='steelblue', alpha=.5)
x_novel = [random.uniform(-0.1, 0.1) + 1 for _ in RTNovel]  # Offset around x=1 ('Novel')
axes[1].scatter(x_novel, RTNovel, color='lightskyblue', alpha=.5)
plt.show()
plt.savefig('~/projectsn/f_mc1689_1/CPRO2_learning/data/results/Novel_vs_prac_RT_and_ERR.png')

eTotS2, eXS2 = findTots(ePracS2, eNovelS2)
rtTotS2, rtXS2 = findTots(RTPracS2, RTNovelS2)
x_barS2 = ['Practice','Novel']
Ey_barS2 = [sum(ePracS2)/len(ePracS2),sum(eNovelS2)/len(eNovelS2)]
EerrsS2 = [sem(ePracS2, nan_policy='omit'), sem(eNovelS2, nan_policy='omit')]
RTy_barS2 = [sum(RTPracS2)/len(RTPracS2),sum(RTNovelS2)/len(RTNovelS2)]
RTerrsS2 = [sem(RTPracS2, nan_policy='omit'), sem(RTNovelS2, nan_policy='omit')]

plt.figure()
fig, axes = plt.subplots(1,2, figsize=(11, 5))
fig.suptitle('Practice vs. Novel Distributions of Error and Reaction Time -- Session 2')
#Error subplot
axes[0].bar(x_barS2,Ey_barS2,width=0.5, align='center', yerr=EerrsS2, fill=False, edgecolor='black')
axes[0].set_title('Error (%)')
axes[0].set_ylabel('Error (%)')
axes[0].set_ylim(0,100)
x_prac = [random.uniform(-0.1, 0.1) + 0 for _ in ePracS2]  # Offset around x=0 ('Practice')
axes[0].scatter(x_prac, ePracS2, color='darkorchid', alpha=.5)
x_novel = [random.uniform(-0.1, 0.1) + 1 for _ in eNovelS2]  # Offset around x=1 ('Novel')
axes[0].scatter(x_novel, eNovelS2, color='orchid', alpha=.5)
#Reaction time subplot
axes[1].bar(x_bar,RTy_barS2,width=0.5, align='center', yerr=RTerrsS2, fill=False, edgecolor='black')
axes[1].set_title('Reaction Time (ms)')
axes[1].set_ylabel('Reaction Time (ms)')
x_prac = [random.uniform(-0.1, 0.1) + 0 for _ in RTPracS2]  # Offset around x=0 ('Practice')
axes[1].scatter(x_prac, RTPracS2, color='steelblue', alpha=.5)
x_novel = [random.uniform(-0.1, 0.1) + 1 for _ in RTNovelS2]  # Offset around x=1 ('Novel')
axes[1].scatter(x_novel, RTNovelS2, color='lightskyblue', alpha=.5)
plt.show()
plt.savefig('~/projectsn/f_mc1689_1/CPRO2_learning/data/results/Novel_vs_prac_RT_and_ERR.png')


'''
Make a function to organize each probe (RT or Acc) into a matrix showing each 
column as a block average (1-48), and each row as a subject
'''
#Function
#NOTE: within this function I mistakenly refer to blocks as trials. 
###### with how I have things labeled in the function, it is more logical to 
###### refer to the blocks as trials (why would t represent block number), so I 
###### will refer to the blocks as trials. I don't plan to change trials --> blocks 
###### because it doesn't matter functionally, and changing how several things  
###### are labeled is more likely to just screw up the entire code. 
def trialOrganization1(df, targetData, row, block):
    rowNumE = len(set(df[row]))
    r = list(df[row])
    novelM = np.zeros((rowNumE,48))#make empty matrix to add subject novel block data 
    pracM = np.zeros((rowNumE,48))#make empty matrix to add subject practice block data
    index = 0
    colNumNov = 0
    colNumPrac = 0
    rowNum = 0 
    Nsum = []
    Psum = []
    t = 1
    conditions = []

    for i in r: #loop through the subjects 
        if df.iloc[index,df.columns.get_loc("Probe1.RT")]==float('nan'):#Correct for any possible nan values
            if df.iloc[index,df.columns.get_loc(block)]=='Novel':
                conditions.append('Novel')
            else:
                conditions.append('Practice')
        else:
            try:
                if int(df.iloc[index,df.columns.get_loc('Trial')]) == t:
                    #Append necessary trial average information when the trial has not changed from last row of data
                    try:
                        if df.iloc[index,df.columns.get_loc(block)] == 'Novel':
                            #collect data from novel blocks and add them to the novel trial avg info using indices
                            conditions.append('Novel')
                            Nsum.append(df.iloc[index,df.columns.get_loc(targetData)])
                        else:
                            #collect data from practice blocks and add them to the practice trial avg info using indices
                            Psum.append(df.iloc[index,df.columns.get_loc(targetData)])
                            conditions.append('Practice')
                    except Exception as e:
                        print('This error happened at the first if/else block')
                        print(f"Sorry but the Practice and novel trials could not be\nseparated due to error {e}.")
                        print('No data was added to Nsum or Psum, and the condition\nfrom this index was not added to conditions.')
                        break
                else:
                    try:
                        #if current row is a different trial from the last row, reassign t to current trial
                        t = int(df.iloc[index,df.columns.get_loc('Trial')])
                    except Exception as e:
                        print('This error occurred within the first two lines of the else portion\nof the main if/else block')
                        print(f't could not be reassigned to the next trial value due\nto error {e}.')
                    try:
                        #as the new trial is starting, the previous trial average list info must be complete
                        # --> add previous trial average to matrix
                        if conditions[-1]=='Novel':#add to novel matrix
                            novelM[rowNum,colNumNov]= round((sum(Nsum)/len(Nsum)),2)
                            #reassign column values as needed
                            colNumNov+=1
                            if colNumNov>=48:
                                colNumNov=0
                            else: pass
                            #reassign novel trial average list
                            Nsum = []
                        else: #add to practice matrix
                            pracM[rowNum,colNumPrac]= round((sum(Psum)/len(Psum)),2)
                            #reassign column values as needed
                            colNumPrac+=1
                            if colNumPrac>=48:
                                colNumPrac=0
                            else: pass
                            #reassign practice trial average list
                            Psum = []
                    except Exception as e:
                        print(f'the tavg could not be added to the matrix due to the following error:\n    {e}')
                        print('this error occurred in the first block of the else section of the main if/else block')
                        print('Psum, Nsum, and colNum were also not reset')
                        print(f'this error occurred at subject {i}, trial {t}, and index {rowNum,colNumPrac}.')
                        print('')
                    try:#as we are starting the new trial, add in the first value for the trial to average lists
                        if df.iloc[index,df.columns.get_loc(block)] == 'Novel':
                            conditions.append('Novel')
                            Nsum.append(df.iloc[index,df.columns.get_loc(targetData)])
                        else:
                            Psum.append(df.iloc[index,df.columns.get_loc(targetData)])
                            conditions.append('Practice')
                    except Exception as e:
                        print(f'The first value in trial {t} was not added to the average\nin subject {i} of condition {conditions[-1]} due to the error\n    {e}')
                #add in final trial averages to practice and novel matrices
                if rowNum==rowNumE-1 and colNumNov==47:
                    try:
                        end = r[index+1]
                    except IndexError:
                        novelM[rowNum,47]=round((sum(Nsum)/len(Nsum)),2)
                elif rowNum==rowNumE-1 and colNumPrac==47:
                    try:
                        end = r[index+1]
                    except IndexError:
                        pracM[rowNum,colNumPrac]=round((sum(Psum)/len(Psum)),2)
                if index == 0: pass #avoid indexing error
                elif i!=r[index-1]:#add in nan values as needed for when a subject misses some trials within their session
                    if 0<colNumNov<48:
                        while colNumNov<48:
                            novelM[rowNum,colNumNov]='NaN'
                            colNumNov+=1
                    if 0<colNumPrac<48:
                        while colNumPrac<48:
                            pracM[rowNum,colNumPrac]='NaN'
                            colNumPrac+=1
                    #reset column and row matrix information for the next subject
                    rowNum+=1
                    colNumPrac = 0
                    colNumNov = 0
                else: pass
            except Exception as err:
                print('')
                print(f"Unexpected {err=}, {type(err)=}")
                print(f'Error occurred at Subject {i}, Trial {t}, index {index}.')
                print('')
        index+=1
    return novelM, pracM

#run for RT
novelBlockRT, pracBlockRT = trialOrganization1(behavioralDf1, 'Probe1.RT','Subject','BlockCond')

#run for Accuracy
novelBlockACC, pracBlockACC = trialOrganization1(behavioralDf1, 'Probe1.ACC','Subject','BlockCond')

#get error matrices
novelBlockErr = np.zeros((45,48))
r=-1
for row in novelBlockACC:
    c = -1
    r+=1
    for i in row:
        c+=1
        k = 1-i
        novelBlockErr[r,c]=k

pracBlockErr = np.zeros((45,48))
r=-1
for row in pracBlockACC:
    c = -1
    r+=1
    for i in row:
        c+=1
        k = 1-i
        pracBlockErr[r,c]=k

#repeat for session2

#run for RT
novelBlockRTS2, pracBlockRTS2 = trialOrganization1(session2, 'Probe1.RT','Subject','BlockCond')

#run for Accuracy
novelBlockACCS2, pracBlockACCS2 = trialOrganization1(session2, 'Probe1.ACC','Subject','BlockCond')

#get error matrices
novelBlockErrS2 = np.zeros((45,48))
r=-1
for row in novelBlockACCS2:
    c = -1
    r+=1
    for i in row:
        c+=1
        k = 1-i
        novelBlockErrS2[r,c]=k

pracBlockErrS2 = np.zeros((45,48))
r=-1
for row in pracBlockACCS2:
    c = -1
    r+=1
    for i in row:
        c+=1
        k = 1-i
        pracBlockErrS2[r,c]=k

'''
Write a function to find the subject mean and standard error of each block across a session 
Graph this information --> make conditional
'''
#Function
def graphBlocks(dfPrac, dfNov, dataType, units, signifsPN, YoN, sess):

    #add standard deviations to block points
    #find average AvgBlockRT
    meanPracRT = []
    stdPracRT = []
    meanNovRT = []
    stdNovRT = []
    
    TotAvgPracBlockRT = dfPrac.transpose()#avg prac block
    TotAvgNovBlockRT = dfNov.transpose()#avg novel block
    
    for row in TotAvgPracBlockRT:#get mean of participant practice block reaction time
        k = np.nanmean(row)
        meanPracRT.append(k)
        std = sem(row, nan_policy='omit')
        stdPracRT.append(std)
    
    for row in TotAvgNovBlockRT:#get mean of participant novel block reaction time
        k = np.nanmean(row)
        meanNovRT.append(k)
        std = sem(row, nan_policy='omit')
        stdNovRT.append(std)

    #graph each row as a line for reaction time practice
    if YoN == 'Y':
        #find number of blocks in a session will be plotted
        i=1
        for row in dfPrac:
            x = [x for x in range(len(row))]
            row = list(row)
            i+=1

        #plot subject averaged blocks across a session with error bars ~ standard error
        plt.figure()
        #Plot Practice Blocks
        plt.plot(x,meanPracRT, color='steelblue', linewidth=2)
        plt.fill_between(x, np.array(meanPracRT) - np.array(stdPracRT), np.array(meanPracRT) + np.array(stdPracRT), alpha=0.3, color='lightskyblue', label='Practice')
        #Plot Novel Blocks
        plt.plot(x,meanNovRT, color='mediumorchid',linewidth=2)
        plt.fill_between(x, np.array(meanNovRT) - np.array(stdNovRT), np.array(meanNovRT) + np.array(stdNovRT), alpha=0.3, color='violet', label='Novel')
        plt.title(f'Average {dataType} of Participants across Session {sess}',fontsize = 15)
             
        #Make Scatter points to identify novel blocks significantly different from prac blocks
        index=-1
        signifPN_label_added = False  # Flag to track label addition
        for index, i in enumerate(signifsPN):
            if i <= 0.05:
                if not signifPN_label_added:
                    plt.scatter(x[index], meanNovRT[index], color='lightpink', s=60, zorder=5, 
                                label='Practice Block ≠ Novel Block')
                    signifPN_label_added = True
                else:
                    plt.scatter(x[index], meanNovRT[index], color='lightpink', s=60, zorder=5)
                plt.scatter(x[index], meanPracRT[index], color='lightpink', s=60, zorder=5)
        plt.xlabel('Block Number', fontsize=13)
        plt.ylabel(f'{dataType} ({units})', fontsize = 13)
    
        plt.legend(loc="upper right", bbox_to_anchor=(1.75, 1), fontsize = 12)
        plt.savefig(f'~/projectsn/f_mc1689_1/CPRO2_learning/data/results/NvP_byBlock_{dataType}.png')
    else:
        pass
    return meanNovRT, stdNovRT, meanPracRT, stdPracRT
    
    
nada = [1]
#Reaction Time Block mean and standard error info
meanNovRT, stdNovRT, meanPracRT, stdPracRT = graphBlocks(pracBlockRT, novelBlockRT, 'Reaction Time', 'ms', nada, 'N',3)

#Error Block mean and standard error info
meanNovErr, stdNovErr, meanPracErr, stdPracErr = graphBlocks(pracBlockErr, novelBlockErr, 'Error', '%', nada, 'N',3)

#Session 2
#Reaction Time Block mean and standard error info
meanNovRTS2, stdNovRTS2, meanPracRTS2, stdPracRTS2 = graphBlocks(pracBlockRTS2, novelBlockRTS2, 'Reaction Time', 'ms', nada, 'N',2)

#Error Block mean and standard error info
meanNovErrS2, stdNovErrS2, meanPracErrS2, stdPracErrS2 = graphBlocks(pracBlockErrS2, novelBlockErrS2, 'Error', '%', nada, 'N',2)


#are the probe conditions different between sessions
print('Between session differences')
print('REACTION TIME PRACTICE')
tRTnov,pRTnov = stats.ttest_rel(meanPracRT, meanPracRTS2, alternative='two-sided')
p_values = np.array(pRTnov) if isinstance(pRTnov, (list, np.ndarray)) else np.array([pRTnov])

# Apply FDR correction using the Benjamini-Hochberg method
rejected, p_correctedRTnov, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
print(f'T-statistic: {tRTnov}\nP-Value: {p_correctedRTnov}')
if p_correctedRTnov<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe participants reaction times changes between sessions.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe participants reaction times changes between sessions.\n')

print('ERROR PRACTICE')
tERRnov,pERRnov = stats.ttest_rel(meanNovErr, meanNovErrS2, alternative='two-sided')
p_values = np.array(pERRnov) if isinstance(pERRnov, (list, np.ndarray)) else np.array([pERRnov])

# Apply FDR correction using the Benjamini-Hochberg method
rejected, p_correctedRTnov, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
print(f'T-statistic: {tERRnov}\nP-Value: {p_correctedRTnov}')
if p_correctedRTnov<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe participants error rate changes between sessions.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe participants error rate changes between sessions.\n')

'''
determine regression info, plot it
'''

#Function to find linear regressions
def regressions(y, X, condition): #y = target data, x=time equivalent
    #reshape X for sklearn if needed 
    X = X.reshape(-1, 1)
    # Create and fit the model
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict using the model
    predictions = model.predict(X)
    
    #r**2
    r = model.score(X, y)
    p = []
    if condition=='RT':
        #slope
        m = (predictions[-1]-predictions[0])/X[-1]
        return m, predictions, r
    else:
        for i in predictions:
            p.append(i*100)
        #slope
        m = (p[-1]-p[0])/X[-1]
        return m, p, r
#Actually run the regression
x = np.array([x for x in range(1,49)]) #define trials as 1 through 48

#AVERAGED SUBJECTS
#run for reaction time novel and prac 
mNovRT, predNovRT, rNovRT = regressions(meanNovRT, x, 'RT')
mPracRT, predPracRT, rPracRT = regressions(meanPracRT, x, 'RT')

#run for error novel and prac 
mNovERR, predNovERR, rNovERR = regressions(meanNovErr, x, 'ERR')
mPracERR, predPracERR, rPracERR = regressions(meanPracErr, x, 'ERR')

#Session 2
#run for reaction time novel and prac 
mNovRTS2, predNovRTS2, rNovRTS2 = regressions(meanNovRTS2, x, 'RT')
mPracRTS2, predPracRTS2, rPracRTS2 = regressions(meanPracRTS2, x, 'RT')

#run for error novel and prac 
mNovERRS2, predNovERRS2, rNovERRS2 = regressions(meanNovErrS2, x, 'ERR')
mPracERRS2, predPracERRS2, rPracERRS2 = regressions(meanPracErrS2, x, 'ERR')

#Plot the regressions --> turn error dataset from decimals to percentages
def turnPerc(dataSet):
    percs = [i*100 for i in dataSet]
    return percs

plt.figure()
plt.plot(x, predNovRTS2, color='seagreen', label =f'Novel S2 = {np.round(mNovRTS2,2)}(Trial)+{round(predNovRTS2[0],2)}')
plt.plot(x, predPracRTS2, color = 'mediumvioletred', label=f'Practice S2 = {np.round(mPracRTS2,2)}(Trial)+{round(predPracRTS2[0],2)}')
plt.scatter(x, meanPracRTS2, color = 'mediumvioletred')
plt.scatter(x,meanNovRTS2, color = 'seagreen')
plt.xlabel('Block Number',fontsize=13)
plt.ylabel('Predicted Reaction Time (ms)', fontsize = 13)
plt.plot(x, predNovRT, color='mediumorchid', label = f'Novel S3 = {np.round(mNovRT,2)}(Trial)+{round(predNovRT[0],2)}')
plt.plot(x, predPracRT, color = 'steelblue', label=f'Practice S3  = {np.round(mPracRT,2)}(Trial)+{round(predPracRT[0],2)}')
plt.scatter(x, meanPracRT, color = 'mediumorchid')
plt.scatter(x,meanNovRT, color = 'steelblue')
plt.ylim(2300,2700)
plt.title('Predicted Regression of Reaction Time across Trials', fontsize=15)
plt.legend(loc="upper right", bbox_to_anchor=(1.85, 1), fontsize = 12)
plt.savefig('~/projectsn/f_mc1689_1/CPRO2_learning/data/results/NvP_regression_RT.png')

y1 = turnPerc(meanNovErrS2)
y2 = turnPerc(meanPracErrS2)
y3 = turnPerc(meanNovErr)
y4 = turnPerc(meanPracErr)

plt.figure()
plt.plot(x, predNovERRS2, color='seagreen', label =f'Novel S2 = {np.round(mNovERRS2,2)}(Trial)+{round(predNovERRS2[0],2)}')
plt.scatter(x,y1, color='seagreen')
plt.plot(x, predPracERRS2, color = 'mediumvioletred', label=f'Practice S2 = {np.round(mPracERRS2,2)}(Trial)+{round(predPracERRS2[0],2)}')
plt.scatter(x,y2, color = 'mediumvioletred')
plt.xlabel('Block Number',fontsize=13)
plt.ylabel('Predicted Reaction Time (ms)', fontsize = 13)
plt.plot(x, predNovERR, color='mediumorchid', label = f'Novel S3 = {np.round(mNovERR,2)}(Trial)+{round(predNovERR[0],2)}')
plt.scatter(x,y3, color='mediumorchid')
plt.plot(x, predPracERR, color = 'steelblue', label=f'Practice S3  = {np.round(mPracERR,2)}(Trial)+{round(predPracERR[0],2)}')
plt.scatter(x,y4, color='steelblue')
plt.ylim(0,50)
plt.title('Predicted Regression of Error across Trials', fontsize=15)
plt.legend(loc="upper right", bbox_to_anchor=(1.8, 1), fontsize = 12)
plt.savefig('~/projectsn/f_mc1689_1/CPRO2_learning/data/results/NvP_regression_RT.png')

'''
run 2 regressions (novel and prac) for each subject to get 45 trends
add the practice and novel betas for each subject to a matrix
'''
#REACTION TIME
RTpaired = np.zeros((45,2)) #make empty matrix to add subject RT regression slopes to, 0=novel and 1=practice
subj = 0 
for row in novelBlockRT:
    row = row[~np.isnan(row)]
    x = np.array([x for x in range(len(row))])
    mNovRTs, predNovRTs, rNovRTs = regressions(row, x, 'RT')
    RTpaired[subj,0] = mNovRTs
    subj+=1
subjs = 0
for row in pracBlockRT:
    row = row[~np.isnan(row)]
    x = np.array([x for x in range(len(row))])
    mPracRTs, predPracRTs, rPracRTs = regressions(row, x, 'RT')
    RTpaired[subjs,1] = mPracRTs
    subjs+=1

#ERROR
ERRpaired = np.zeros((45,2)) #make empty matrix to add subject RT regression slopes to, 0=novel and 1=practice
subj = 0 
for row in novelBlockErr:
    row = row[~np.isnan(row)]
    x = np.array([x for x in range(len(row))])
    mNovERRs, predNovERRs, rNovERRs = regressions(row, x, 'ERR')
    ERRpaired[subj,0] = mNovERRs
    subj+=1
subjs = 0
for row in pracBlockErr:
    row = row[~np.isnan(row)]
    x = np.array([x for x in range(len(row))])
    mPracERRs, predPracERRs, rPracERRs = regressions(row, x, 'ERR')
    ERRpaired[subjs,1] = mPracERRs
    subjs+=1

#session 2
#REACTION TIME
RTpairedS2 = np.zeros((45,2)) #make empty matrix to add subject RT regression slopes to, 0=novel and 1=practice
subj = 0 
for row in novelBlockRTS2:
    row = row[~np.isnan(row)]
    x = np.array([x for x in range(len(row))])
    mNovRTsS2, predNovRTsS2, rNovRTsS2 = regressions(row, x, 'RT')
    RTpairedS2[subj,0] = mNovRTsS2
    subj+=1
subjs = 0
for row in pracBlockRTS2:
    row = row[~np.isnan(row)]
    x = np.array([x for x in range(len(row))])
    mPracRTsS2, predPracRTsS2, rPracRTsS2 = regressions(row, x, 'RT')
    RTpairedS2[subjs,1] = mPracRTsS2
    subjs+=1


#ERROR
ERRpairedS2 = np.zeros((45,2)) #make empty matrix to add subject RT regression slopes to, 0=novel and 1=practice
subj = 0 
for row in novelBlockErrS2:
    row = row[~np.isnan(row)]
    x = np.array([x for x in range(len(row))])
    mNovERRsS2, predNovERRsS2, rNovERRsS2 = regressions(row, x, 'ERR')
    ERRpairedS2[subj,0] = mNovERRsS2
    subj+=1
subjs = 0
for row in pracBlockErrS2:
    row = row[~np.isnan(row)]
    x = np.array([x for x in range(len(row))])
    mPracERRsS2, predPracERRsS2, rPracERRsS2 = regressions(row, x, 'ERR')
    ERRpairedS2[subjs,1] = mPracERRsS2
    subjs+=1
'''
run paired t-tests on the regression matrices to determine if there is a difference
in the slopes of practice and novel trends within subjects
'''
#RUN PAIRED T TESTS ON RTpaired - Is there a difference between novel and practice trends while analyzing reaction time?
RTnovSlopes = RTpaired[:,0]
RTpracSlopes = RTpaired[:,1]
RTslope_tStat, RTslope_p = ttest_rel(RTnovSlopes,RTpracSlopes)
print('REACTION TIME - P-I')
print(f'T-statistic: {RTslope_tStat}\nP-Value: {RTslope_p}')
if RTslope_p<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants reaction times across a session differs\nbetween novel and practice trials.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants reaction times across a session differs\nbetween novel and practice trials.\n')

#RUN PAIRED T TESTS ON ERRpaired- Is there a difference between novel and practice trends while analyzing accuracy?
ERRnovSlopes = ERRpaired[:,0]
ERRpracSlopes = ERRpaired[:,1]    
ERRslope_tStat, ERRslope_p = ttest_rel(ERRnovSlopes,ERRpracSlopes)
print('ERROR - P-I')
print(f'T-statistic: {ERRslope_tStat}\nP-Value: {ERRslope_p}')
if ERRslope_p<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants error across a session differs\nbetween novel and practice trials.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants error across a session differs\nbetween novel and practice trials.\n')


#session 2
#RUN PAIRED T TESTS ON RTpaired - Is there a difference between novel and practice trends while analyzing reaction time?
print('\nSESSION 2\n')
RTnovSlopesS2 = RTpairedS2[:,0]
RTpracSlopesS2 = RTpairedS2[:,1]
RTslope_tStatS2, RTslope_pS2 = ttest_rel(RTnovSlopesS2,RTpracSlopesS2)
print('REACTION TIME - P-I')
print(f'T-statistic: {RTslope_tStatS2}\nP-Value: {RTslope_pS2}')
if RTslope_pS2<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants reaction times across a session differs\nbetween novel and practice trials.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants reaction times across a session differs\nbetween novel and practice trials.\n')

#RUN PAIRED T TESTS ON ERRpaired- Is there a difference between novel and practice trends while analyzing accuracy?
ERRnovSlopesS2 = ERRpairedS2[:,0]
ERRpracSlopesS2 = ERRpairedS2[:,1]    
ERRslope_tStatS2, ERRslope_pS2 = ttest_rel(ERRnovSlopesS2,ERRpracSlopesS2)
print('ERROR - P-I')
print(f'T-statistic: {ERRslope_tStatS2}\nP-Value: {ERRslope_pS2}')
if ERRslope_pS2<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants error across a session differs\nbetween novel and practice trials.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants error across a session differs\nbetween novel and practice trials.\n')

#graph practice novel slope distributions for accuracy, reaction time, and error
#graph in subplots
eTot, eX = findTots(ERRpracSlopes, ERRnovSlopes)
aTot, aX = findTots(ACCpracSlopes, ACCnovSlopes)
rtTot, rtX = findTots(RTpracSlopes, RTnovSlopes)

x_bar = ['Practice','Novel']
Ey_bar = [sum(ERRpracSlopes)/len(ERRpracSlopes),sum(ERRnovSlopes)/len(ERRnovSlopes)]
Eerrs = [sem(ERRpracSlopes, nan_policy='omit'), sem(ERRnovSlopes, nan_policy='omit')]
RTy_bar = [sum(RTpracSlopes)/len(RTpracSlopes),sum(RTnovSlopes)/len(RTnovSlopes)]
RTerrs = [sem(RTpracSlopes, nan_policy='omit'), sem(RTnovSlopes, nan_policy='omit')]

plt.figure()
fig, axes = plt.subplots(1,2, figsize=(11, 5))
fig.suptitle('Distribution of Slopes from Probe Regressions S3')
#Error subplot
axes[0].bar(x_bar,Ey_bar,width=0.5, align='center', yerr=Eerrs, fill=False, edgecolor='black')
axes[0].set_title('Error (%)')
axes[0].set_ylabel('Slopes')
x_prac = [random.uniform(-0.1, 0.1) + 0 for _ in ERRpracSlopes]  # Offset around x=0 ('Practice')
axes[0].scatter(x_prac, ERRpracSlopes, color='darkorchid', alpha=.5)
x_novel = [random.uniform(-0.1, 0.1) + 1 for _ in ERRnovSlopes]  # Offset around x=1 ('Novel')
axes[0].scatter(x_novel, ERRnovSlopes, color='orchid', alpha=.5)
#Reaction time subplot
axes[1].bar(x_bar,RTy_bar,width=0.5, align='center', yerr=RTerrs, fill=False, edgecolor='black')
axes[1].set_title('Reaction Time (ms)')
axes[1].set_ylabel('Slopes')
x_prac = [random.uniform(-0.1, 0.1) + 0 for _ in RTpracSlopes]  # Offset around x=0 ('Practice')
axes[1].scatter(x_prac, RTpracSlopes, color='steelblue', alpha=.5)
x_novel = [random.uniform(-0.1, 0.1) + 1 for _ in RTnovSlopes]  # Offset around x=1 ('Novel')
axes[1].scatter(x_novel, RTnovSlopes, color='lightskyblue', alpha=.5)
plt.show()
plt.savefig(f'~/projectsn/f_mc1689_1/CPRO2_learning/data/results/NvP_SlopesDistribution_RTandERR.png')

#graph in one plot
x_bar = ['Error\nPractice','Error\nNovel','Reaction Time\nPractice','Reaction Time\nNovel']
y_bar = [sum(ERRpracSlopes)/len(ERRpracSlopes),sum(ERRnovSlopes)/len(ERRnovSlopes),sum(RTpracSlopes)/len(RTpracSlopes),sum(RTnovSlopes)/len(RTnovSlopes)]
errs = [sem(ERRpracSlopes, nan_policy='omit'), sem(ERRnovSlopes, nan_policy='omit'),sem(RTpracSlopes, nan_policy='omit'), sem(RTnovSlopes, nan_policy='omit')]

plt.figure()
plt.title('Distribution of Slopes from Probe Regressions')
#Error subplot
plt.bar(x_bar,y_bar,width=0.5, align='center', yerr=errs, fill=False, edgecolor='black')
x_prac = [random.uniform(-0.1, 0.1) + 0 for _ in ERRpracSlopes]  # Offset around x=0 ('Practice')
plt.scatter(x_prac, ERRpracSlopes, color='darkorchid', alpha=.5)
x_novel = [random.uniform(-0.1, 0.1) + 1 for _ in ERRnovSlopes]  # Offset around x=1 ('Novel')
plt.scatter(x_novel, ERRnovSlopes, color='orchid', alpha=.5)
x_prac = [random.uniform(-0.1, 0.1) + 2 for _ in RTpracSlopes]  # Offset around x=0 ('Practice')
plt.scatter(x_prac, RTpracSlopes, color='steelblue', alpha=.5)
x_novel = [random.uniform(-0.1, 0.1) + 3 for _ in RTnovSlopes]  # Offset around x=1 ('Novel')
plt.scatter(x_novel, RTnovSlopes, color='lightskyblue', alpha=.5)
plt.ylabel('Slopes')
plt.xlabel('Condition')
plt.show()
plt.savefig(f'~/projectsn/f_mc1689_1/CPRO2_learning/data/results/NvP_SlopesDistribution_RTandERR_type2.png')

#session 2
#graph practice novel slope distributions for accuracy, reaction time, and error
#graph in subplots
eTot, eX = findTots(ERRpracSlopesS2, ERRnovSlopesS2)
aTot, aX = findTots(ACCpracSlopesS2, ACCnovSlopesS2)
rtTot, rtX = findTots(RTpracSlopesS2, RTnovSlopesS2)

x_bar = ['Practice','Novel']
Ey_bar = [sum(ERRpracSlopesS2)/len(ERRpracSlopesS2),sum(ERRnovSlopesS2)/len(ERRnovSlopesS2)]
Eerrs = [sem(ERRpracSlopesS2, nan_policy='omit'), sem(ERRnovSlopesS2, nan_policy='omit')]
RTy_bar = [sum(RTpracSlopesS2)/len(RTpracSlopesS2),sum(RTnovSlopesS2)/len(RTnovSlopesS2)]
RTerrs = [sem(RTpracSlopesS2, nan_policy='omit'), sem(RTnovSlopesS2, nan_policy='omit')]

plt.figure()
fig, axes = plt.subplots(1,2, figsize=(11, 5))
fig.suptitle('Distribution of Slopes from Probe Regressions S2')
#Error subplot
axes[0].bar(x_bar,Ey_bar,width=0.5, align='center', yerr=Eerrs, fill=False, edgecolor='black')
axes[0].set_title('Error (%)')
axes[0].set_ylabel('Slopes')
x_prac = [random.uniform(-0.1, 0.1) + 0 for _ in ERRpracSlopes]  # Offset around x=0 ('Practice')
axes[0].scatter(x_prac, ERRpracSlopes, color='darkorchid', alpha=.5)
x_novel = [random.uniform(-0.1, 0.1) + 1 for _ in ERRnovSlopes]  # Offset around x=1 ('Novel')
axes[0].scatter(x_novel, ERRnovSlopes, color='orchid', alpha=.5)
#Reaction time subplot
axes[1].bar(x_bar,RTy_bar,width=0.5, align='center', yerr=RTerrs, fill=False, edgecolor='black')
axes[1].set_title('Reaction Time (ms)')
axes[1].set_ylabel('Slopes')
x_prac = [random.uniform(-0.1, 0.1) + 0 for _ in RTpracSlopes]  # Offset around x=0 ('Practice')
axes[1].scatter(x_prac, RTpracSlopes, color='steelblue', alpha=.5)
x_novel = [random.uniform(-0.1, 0.1) + 1 for _ in RTnovSlopes]  # Offset around x=1 ('Novel')
axes[1].scatter(x_novel, RTnovSlopes, color='lightskyblue', alpha=.5)
plt.show()
plt.savefig(f'~/projectsn/f_mc1689_1/CPRO2_learning/data/results/NvP_SlopesDistribution_RTandERR.png')


'''
Determine if each individual trend is different from 0
'''
print('NOVEL REACTION TIME')
t_RTnov, p_RTnov = stats.ttest_1samp(RTnovSlopes, 0)
# If p_RTnov is not an array (i.e., a single test result), convert it to a list for consistency in FDR correction
p_values = np.array(p_RTnov) if isinstance(p_RTnov, (list, np.ndarray)) else np.array([p_RTnov])

# Apply FDR correction using the Benjamini-Hochberg method
rejected, p_correctedRTnov, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
print(f'T-statistic: {t_RTnov}\nP-Value: {p_correctedRTnov}')
if p_RTnov<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants reaction times in novel \ntasks across a session differs from 0.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants reaction times in novel \ntasks across a session differs from 0.\n')

print('PRACTICE REACTION TIME')
t_RTprac, p_RTprac = stats.ttest_1samp(RTpracSlopes, 0)
p_values = np.array(p_RTprac) if isinstance(p_RTprac, (list, np.ndarray)) else np.array([p_RTprac])
rejected, p_correctedRTprac, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
print(f'T-statistic: {t_RTprac}\nP-Value: {p_correctedRTprac}')
if p_RTprac<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants reaction times in practice \ntasks across a session differs from 0.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants reaction times in practice \ntasks across a session differs from 0.\n')

print('NOVEL ERROR')
t_ERRnov, p_ERRnov = stats.ttest_1samp(ERRnovSlopes, 0)
p_values = np.array(p_ERRnov) if isinstance(p_ERRnov, (list, np.ndarray)) else np.array([p_ERRnov])
rejected, p_correctedACCnov, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
print(f'T-statistic: {t_ERRnov}\nP-Value: {p_correctedERRnov}')
if p_ERRnov<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants accuracy in novel tasks \nacross a session differs from 0.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants accuracy in novel tasks \nacross a session differs from 0.\n')

print('PRACTICE ERROR')
t_ERRprac, p_ERRprac = stats.ttest_1samp(ERRpracSlopes, 0)
p_values = np.array(p_ERRprac) if isinstance(p_ERRprac, (list, np.ndarray)) else np.array([p_ERRprac])
rejected, p_correctedERRprac, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
print(f'T-statistic: {t_ERRprac}\nP-Value: {p_correctedERRprac}')
if p_ERRprac<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants accuracy in practiced tasks \nacross a session differs from 0.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants accuracy in practiced tasks \nacross a session differs from 0.\n')


#Session 2
print('\nSESSION 2\n')
print('NOVEL REACTION TIME')
t_RTnovS2, p_RTnovS2 = stats.ttest_1samp(RTnovSlopesS2, 0)
# If p_RTnov is not an array (i.e., a single test result), convert it to a list for consistency in FDR correction
p_valuesS2 = np.array(p_RTnovS2) if isinstance(p_RTnovS2, (list, np.ndarray)) else np.array([p_RTnovS2])

# Apply FDR correction using the Benjamini-Hochberg method
rejectedS2, p_correctedRTnovS2, _, _ = multipletests(p_valuesS2, alpha=0.05, method='fdr_bh')
print(f'T-statistic: {t_RTnovS2}\nP-Value: {p_correctedRTnovS2}')
if p_RTnovS2<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants reaction times in novel \ntasks across a session differs from 0.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants reaction times in novel \ntasks across a session differs from 0.\n')

print('PRACTICE REACTION TIME')
t_RTpracS2, p_RTpracS2 = stats.ttest_1samp(RTpracSlopesS2, 0)
p_valuesS2 = np.array(p_RTpracS2) if isinstance(p_RTpracS2, (list, np.ndarray)) else np.array([p_RTpracS2])
rejectedS2, p_correctedRTpracS2, _, _ = multipletests(p_valuesS2, alpha=0.05, method='fdr_bh')
print(f'T-statistic: {t_RTpracS2}\nP-Value: {p_correctedRTpracS2}')
if p_RTpracS2<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants reaction times in practice \ntasks across a session differs from 0.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants reaction times in practice \ntasks across a session differs from 0.\n')

print('NOVEL ERROR')
t_ERRnovS2, p_ERRnovS2 = stats.ttest_1samp(ERRnovSlopesS2, 0)
p_valuesS2 = np.array(p_ERRnovS2) if isinstance(p_ERRnovS2, (list, np.ndarray)) else np.array([p_ERRnovS2])
rejectedS2, p_correctedERRnovS2, _, _ = multipletests(p_valuesS2, alpha=0.05, method='fdr_bh')
print(f'T-statistic: {t_ERRnovS2}\nP-Value: {p_correctedERRnovS2}')
if p_ERRnovS2<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants accuracy in novel tasks \nacross a session differs from 0.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants accuracy in novel tasks \nacross a session differs from 0.\n')

print('PRACTICE ERROR')
t_ERRpracS2, p_ERRpracS2 = stats.ttest_1samp(ERRpracSlopesS2, 0)
p_valuesS2 = np.array(p_ERRpracS2) if isinstance(p_ERRpracS2, (list, np.ndarray)) else np.array([p_ERRpracS2])
rejectedS2, p_correctedERRpracS2, _, _ = multipletests(p_valuesS2, alpha=0.05, method='fdr_bh')
print(f'T-statistic: {t_ERRpracS2}\nP-Value: {p_correctedERRpracS2}')
if p_ERRpracS2<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants accuracy in practiced tasks \nacross a session differs from 0.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants accuracy in practiced tasks \nacross a session differs from 0.\n')

#Is there a difference between session 2 and 3 novel reaction time trends 
print('\nSESSION 2 vs. SESSION 3\n')
print('NOVEL REACTION TIME')
t_RTnovs, p_RTnovs = stats.ttest_rel(RTnovSlopesS2, RTnovSlopes, alternative='two-sided')
# If p_RTnov is not an array (i.e., a single test result), convert it to a list for consistency in FDR correction
p_valuess = np.array(p_RTnovs) if isinstance(p_RTnovs, (list, np.ndarray)) else np.array([p_RTnovs])

# Apply FDR correction using the Benjamini-Hochberg method
rejecteds, p_correctedRTnovs, _, _ = multipletests(p_valuess, alpha=0.05, method='fdr_bh')
print(f'T-statistic: {t_RTnovs}\nP-Value: {p_correctedRTnovs}')
if p_RTnovs<0.05: print('As the p-value is less than 0.05, we find significant evidence that \nthe trend of participants reaction times in novel \ntasks across a session differs from 0.\n')
else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nthe trend of participants reaction times in novel \ntasks across a session differs from 0.\n')

print('AVERAGE DATA S2 VS. S3')
def s2vss3AvgData(novelBlockRT,novelBlockRTS2,dataType,condition):
    avgNovRTS3 = []
    avgNovRTS2 = []
    for row in novelBlockRT:
        k = np.nanmean(row)
        avgNovRTS3.append(k)
    for row in novelBlockRTS2:
        avgNovRTS2.append(np.nanmean(row))
    t_RTnovs, p_RTnovs = stats.ttest_ind(avgNovRTS3, avgNovRTS2, equal_var=False)
    # If p_RTnov is not an array (i.e., a single test result), convert it to a list for consistency in FDR correction
    p_valuess = np.array(p_RTnovs) if isinstance(p_RTnovs, (list, np.ndarray)) else np.array([p_RTnovs])
    
    # Apply FDR correction using the Benjamini-Hochberg method
    rejecteds, p_correctedRTnovs, _, _ = multipletests(p_valuess, alpha=0.05, method='fdr_bh')
    print(f'{condition} {dataType}')
    print(f'T-statistic: {t_RTnovs}\nP-Value: {p_correctedRTnovs}')
    if p_RTnovs<0.05: print(f'As the p-value is less than 0.05, we find significant evidence that \nparticipant {dataType} in {condition} \ntasks is different in session 2 vs 3.\n')
    else: print('As the p-value is greater than 0.05, we do not find significant evidence that \nparticipant {dataType} in {condition} \ntasks is different in session 2 vs 3.\n')
    return t_RTnovs, p_correctedRTnovs
t_RTnovs, p_correctedRTnovs = s2vss3AvgData(novelBlockRT,novelBlockRTS2,'Reaction Time','Novel')
t_RTprac, p_correctedRTprac = s2vss3AvgData(pracBlockRT,pracBlockRTS2,'Reaction Time','Practice')
t_ERRnovs, p_correctedERRnovs = s2vss3AvgData(novelBlockErr,novelBlockErrS2,'Error Rate','Novel')
t_ERRprac, p_correctedERRprac = s2vss3AvgData(pracBlockErr,pracBlockErrS2,'Error Rate','Practice')
t_RTpn, p_correctedRTpn = s2vss3AvgData(novelBlockRT,pracBlockRTS2,'Reaction Time','Practice2 vs. Novel3')

'''
Determine between block significant changes (ie: is block 1 significantly different
                                             from block 2)
#Run a test on how reaction time changes between one trial and the next 
#- Across participants have trial n and trial n+1, have 45 participants, run 60 tests
#- Group 1 = 45 p at trial n
#- Group 2 = 45 p at trial n+1
#- Get 59 fdr corrected t stats and p values
'''

def matrixTTest(matrix,tStats,pValues,conditionL,condType, probe, probeType):
    matrix = matrix.transpose()
    trialNum = 0
    for row in matrix:
        group1 = []
        group2 = []
        trialNum+=1
        if trialNum == 48:
            return tStats, pValues, conditionL, probe
        colNum = 0
        for i in row:
            n1 = matrix[trialNum,colNum]
            group1.append(i)
            group2.append(n1)
            colNum+=1
        index = 0
        for i in group1:
            if i==float('nan'):
                group1.remove(i)
                del group2[index]
            elif group2[index]==float('nan'):
                group1.remove(i)
                del group2[index]
            index+=1
        
        ts, pVal = ttest_rel(group2,group1)
        p_values = np.array(pVal) if isinstance(pVal, (list, np.ndarray)) else np.array([pVal])
        rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        for i in p_corrected:
            p = round(i,5)
        tStats.append(ts)
        pValues.append(p)
        conditionL.append(condType)
        probe.append(probeType)
    return tStats, pValues, conditionL, probe

def btwnTacrossS(matrixN, matrixP, p):
    tStats = []
    pValues = []
    tStatsN = []
    pValuesN = []
    tStatsP = []
    pValuesP = []
    condition= []
    probe = []
    tStatsN, pValuesN, condition, probe = matrixTTest(matrixN,tStatsN,pValuesN,condition,'Novel', probe, p)
    tStatsP, pValuesP, condition, probe = matrixTTest(matrixP,tStatsP,pValuesP,condition,'Practice', probe, p)
    for i in tStatsN:
        tStats.append(i)
    for i in tStatsP:
        tStats.append(i)
    for i in pValuesN:
        pValues.append(i)
    for i in pValuesP:
        pValues.append(i)
    finalM = pd.DataFrame({'T-Statistics':tStats, 'p-Values':pValues, 'Condition':condition, 'Probe':probe})
    return finalM, tStatsP, pValuesP, tStatsN, pValuesN

#Run function to find between block significance 
rtStats, RTtStatsP, RTpValuesP, RTtStatsN, RTpValuesN = btwnTacrossS(novelBlockRT,pracBlockRT, 'Reaction Time')
errStats, ERRtStatsP, ERRpValuesP, ERRtStatsN, ERRpValuesN = btwnTacrossS(novelBlockErr,pracBlockErr, 'Error')

#S2
#Run function to find between block significance 
rtStatsS2, RTtStatsPS2, RTpValuesPS2, RTtStatsNS2, RTpValuesNS2 = btwnTacrossS(novelBlockRTS2,pracBlockRTS2, 'Reaction Time')
errStatsS2, ERRtStatsPS2, ERRpValuesPS2, ERRtStatsNS2, ERRpValuesNS2 = btwnTacrossS(novelBlockErrS2,pracBlockErrS2, 'Error')

'''
At each block, determine whether the novel distribution is different from the 
practice distribution. 
- compare novelBlockRT vs. pracBlockRT columns 
idea: 
    transpose into rows
    for row in matrix (rowNum, indexing, tStat+pVal append to sep matrix)
    star which trials are significantly different on graphs
        
'''
#Write function
def compareBlocks(prac, novel, probe):
    
    prac = prac.transpose()
    nov = novel.transpose()
    rowNum = 0
    pVals = []
    tStats = []
    for row in prac:
        p = row
        n = nov[rowNum]
        index = 0
        valid_indices = ~np.isnan(p) & ~np.isnan(n)
        p = p[valid_indices]
        n = n[valid_indices]
        try:
            ts, pVal = ttest_rel(p,n, alternative='two-sided')
            p_values = np.array(pVal) if isinstance(pVal, (list, np.ndarray)) else np.array([pVal])
            rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            for i in p_corrected:
                p = round(i,5)
            tStats.append(ts)
            pVals.append(p_values)
        except Exception as e:
            print(f'Had unexpected error:   {e}')
        rowNum+=1
    return pVals, tStats

#Run function to determine blocks with practice-novel differences
pValRT, tStatRT = compareBlocks(pracBlockRT, novelBlockRT,'Reaction Time')
pValERR, tStatERR = compareBlocks(pracBlockErr, novelBlockErr,'Error')

#S2
#Run function to determine blocks with practice-novel differences
pValRTS2, tStatRTS2 = compareBlocks(pracBlockRTS2, novelBlockRTS2,'Reaction Time')
pValERRS2, tStatERRS2 = compareBlocks(pracBlockErrS2, novelBlockErrS2,'Error')

'''
Graph block by block trend over a session with next-current block and practice-novel block significance starred for reaction time, accuracy, and error
These graphs may not be needed
''' 
# #Reaction Time Block Graphing
# meanNovRT, stdNovRT, meanPracRT, stdPracRT = graphBlocks(pracBlockRT, novelBlockRT, 'Reaction Time', 'ms', pValRT,'Y',3)

# #Error Block Graphing 
# meanNovERR, stdNovERR, meanPracERR, stdPracERR = graphBlocks(pracBlockErr, novelBlockErr, 'Error', '%', pValERR,'Y',3)

# #Session 2
# #Reaction Time Block Graphing
# meanNovRTS2, stdNovRTS2, meanPracRTS2, stdPracRTS2 = graphBlocks(pracBlockRTS2, novelBlockRTS2, 'Reaction Time', 'ms', pValRTS2,'Y',2)

# #Error Block Graphing 
# meanNovERRS2, stdNovERRS2, meanPracERRS2, stdPracERRS2 = graphBlocks(pracBlockErrS2, novelBlockErrS2, 'Error', '%', pValERRS2,'Y',2)
'''
Trends and scatter graphed, block by block across a session (does contain both sessions)
'''
#Function
def graphBlocksS2vS3(dfPracS2, dfNovS2,dfPracS3, dfNovS3, dataType, units):

    #add standard deviations to block points
    #find average AvgBlockRT
    meanPracRTS2 = []
    stdPracRTS2 = []
    meanNovRTS2 = []
    stdNovRTS2 = []
    meanPracRTS3 = []
    stdPracRTS3 = []
    meanNovRTS3 = []
    stdNovRTS3 = []
    
    TotAvgPracBlockRTS2 = dfPracS2.transpose()#avg prac block
    TotAvgNovBlockRTS2 = dfNovS2.transpose()#avg novel block
    TotAvgPracBlockRTS3 = dfPracS3.transpose()#avg prac block
    TotAvgNovBlockRTS3 = dfNovS3.transpose()#avg novel block
    
    for row in TotAvgPracBlockRTS2:#get mean of participant practice block reaction time
        k = np.nanmean(row)
        meanPracRTS2.append(k)
        std = sem(row, nan_policy='omit')
        stdPracRTS2.append(std)
    
    for row in TotAvgNovBlockRTS2:#get mean of participant novel block reaction time
        k = np.nanmean(row)
        meanNovRTS2.append(k)
        std = sem(row, nan_policy='omit')
        stdNovRTS2.append(std)
        
    for row in TotAvgPracBlockRTS3:#get mean of participant practice block reaction time
        k = np.nanmean(row)
        meanPracRTS3.append(k)
        std = sem(row, nan_policy='omit')
        stdPracRTS3.append(std)
    
    for row in TotAvgNovBlockRTS3:#get mean of participant novel block reaction time
        k = np.nanmean(row)
        meanNovRTS3.append(k)
        std = sem(row, nan_policy='omit')
        stdNovRTS3.append(std)

    #graph each row as a line for reaction time practice
    #find number of blocks in a session will be plotted
    i=1
    for row in dfPracS2:
        x = [x for x in range(len(row))]
        row = list(row)
        i+=1

    #plot subject averaged blocks across a session with error bars ~ standard error
    plt.figure()
    #Plot Practice Blocks
    plt.plot(x,meanPracRTS2, color='steelblue', linewidth=2, alpha=.5)
    plt.fill_between(x, np.array(meanPracRTS2) - np.array(stdPracRTS2), np.array(meanPracRTS2) + np.array(stdPracRTS2), alpha=0.2, color='lightskyblue', label='Practice - Session 2')
    #Plot Novel Blocks
    plt.plot(x,meanNovRTS2, color='mediumorchid',linewidth=2, alpha=.5)
    plt.fill_between(x, np.array(meanNovRTS2) - np.array(stdNovRTS2), np.array(meanNovRTS2) + np.array(stdNovRTS2), alpha=0.2, color='violet', label='Novel - Session 2')
    #S3
    #Plot Practice Blocks
    plt.plot(x,meanPracRTS3, color='mediumvioletred', linewidth=2, alpha=.5)
    plt.fill_between(x, np.array(meanPracRTS3) - np.array(stdPracRTS3), np.array(meanPracRTS3) + np.array(stdPracRTS3), alpha=0.2, color='hotpink', label='Practice - Session 3')
    #Plot Novel Blocks
    plt.plot(x,meanNovRTS3, color='seagreen',linewidth=2, alpha=.5)
    plt.fill_between(x, np.array(meanNovRTS3) - np.array(stdNovRTS3), np.array(meanNovRTS3) + np.array(stdNovRTS3), alpha=0.2, color='mediumseagreen', label='Novel - Session 3')
    plt.title(f'Average {dataType} of Participants',fontsize = 15)
    plt.xlabel('Block Number', fontsize=13)
    plt.ylabel(f'{dataType} ({units})', fontsize = 13)

    plt.legend(loc="upper right", bbox_to_anchor=(1.5, 1), fontsize = 12)
    plt.savefig(f'~/projectsn/f_mc1689_1/CPRO2_learning/data/results/NvP_byBlock_{dataType}.png')
    
graphBlocksS2vS3(pracBlockRTS2, novelBlockRTS2,pracBlockRT, novelBlockRT, 'Reaction Time', 'ms')
graphBlocksS2vS3(pracBlockErrS2, novelBlockErrS2,pracBlockErr, novelBlockErr, 'Error', '%')

#TEST LAST MINUS FIRST PER SUBJECT RT SESSION 3
RTnovelLastStart = []
RTpracLastStart = []
for row in novelBlockRT:
    rows = row[~np.isnan(row)]
    diff = rows[-1]-rows[0]
    RTnovelLastStart.append(diff)
for row in pracBlockRT:
    rows = row[~np.isnan(row)]
    diff = rows[-1]-rows[0]
    RTpracLastStart = np.append(RTpracLastStart, diff)

#COMPARE LAST TO START DIFFERENCE AGAINST 0
t_RTnovLS, p_RTnovLS = stats.ttest_1samp(RTnovelLastStart, 0)
t_RTpracLS, p_RTpracLS = stats.ttest_1samp(RTpracLastStart, 0)
print('FIRST VS. LAST BLOCKS SESSION 3')
print('PRACTICE RT')
print(f'T-statistic: {t_RTpracLS}\nP-Value: {p_RTpracLS}')
if p_RTpracLS<0.05: 
    print('As the p-value is less than 0.05, we find significant \nevidence that the participants reaction time changes\nacross practice trials in a session.\n')
else:
    print('As the p-value is not less than 0.05, we do not find \nsignificant evidence that the participants reaction time changes\ncross practice trials in a session.\n')

print('NOVEL RT')
print(f'T-statistic: {t_RTnovLS}\nP-Value: {p_RTnovLS}')
if p_RTnovLS<0.05: 
    print('As the p-value is less than 0.05, we find significant \nevidence that the participants reaction time changes\nacross novel trials in a session.\n')
else:
    print('As the p-value is not less than 0.05, we do not find \nsignificant evidence that the participants reaction time changes\ncross novel trials in a session.\n')


#TEST LAST MINUS FIRST PER SUBJECT RT SESSION 2
RTnovelLastStartS2 = []
RTpracLastStartS2 = []
for row in novelBlockRTS2:
    rows = row[~np.isnan(row)]
    diff = rows[-1]-rows[0]
    RTnovelLastStartS2.append(diff)
for row in pracBlockRTS2:
    rows = row[~np.isnan(row)]
    diff = rows[-1]-rows[0]
    RTpracLastStartS2 = np.append(RTpracLastStartS2, diff)

#COMPARE LAST TO START DIFFERENCE AGAINST 0
t_RTnovLSS2, p_RTnovLSS2 = stats.ttest_1samp(RTnovelLastStartS2, 0)
t_RTpracLSS2, p_RTpracLSS2 = stats.ttest_1samp(RTpracLastStartS2, 0)
print('FIRST VS. LAST BLOCKS SESSION 2')
print('PRACTICE RT')
print(f'T-statistic: {t_RTpracLSS2}\nP-Value: {p_RTpracLSS2}')
if p_RTpracLSS2<0.05: 
    print('As the p-value is less than 0.05, we find significant \nevidence that the participants reaction time changes\nacross practice trials in a session.\n')
else:
    print('As the p-value is not less than 0.05, we do not find \nsignificant evidence that the participants reaction time changes\ncross practice trials in a session.\n')

print('NOVEL RT')
print(f'T-statistic: {t_RTnovLSS2}\nP-Value: {p_RTnovLSS2}')
if p_RTnovLSS2<0.05: 
    print('As the p-value is less than 0.05, we find significant \nevidence that the participants reaction time changes\nacross novel trials in a session.\n')
else:
    print('As the p-value is not less than 0.05, we do not find \nsignificant evidence that the participants reaction time changes\ncross novel trials in a session.\n')

x_axes = ['Practice - S2','Novel - S2','Practice - S3','Novel - S3']
y_start = [RTpracLastStartS2, RTnovelLastStartS2, RTpracLastStart, RTnovelLastStart]
y_axes = []
for i in y_start:
    y_axes.append(sum(i)/len(i))
    
yerr = [sem(RTpracLastStartS2, nan_policy='omit'),sem(RTnovelLastStartS2, nan_policy='omit'),sem(RTpracLastStart,nan_policy='omit'),sem(RTnovelLastStart,nan_policy='omit')]
plt.figure()
plt.bar(x_axes,y_axes,yerr=yerr,fill=False, edgecolor='black')
plt.xlabel('Condition and Session')
plt.ylabel('Reaction time (ms)')
plt.title('Difference in Reaction Time Between First and Last Block')
plt.savefig(f'~/projectsn/f_mc1689_1/CPRO2_learning/data/results/LastVersusFirstBlocks.png')

'''
The rest of the code in this script is included because I did write it, but I don't think that it is necessary/relevant to analyses
'''
#Plotting the distribution of t-statistics and p-values calculated when determining if the slope of each block is significantly different from 0

# rtStats = rtStats.dropna()
# accStats = accStats.dropna()
# #T-statistic plots
# fig, axes = plt.subplots(2, 2, figsize=(8, 6))
# rt_pal = {"Novel": "lightskyblue", "Practice": "steelblue"}
# acc_pal = {"Novel": "orchid", "Practice": "darkorchid"}

# sns.boxplot(ax=axes[0,0], x=rtStats['Condition'], y=rtStats['T-Statistics'], hue=rtStats['Condition'], palette=rt_pal)
# axes[0,0].set_title('RT Statistics')
# sns.boxplot(ax=axes[0,1], x=accStats['Condition'], y=accStats['T-Statistics'], hue=accStats['Condition'], palette=acc_pal)
# axes[0,1].set_title('Accuracy Statistics')
# sns.boxplot(ax=axes[1,0], x=rtStats['Condition'], y=rtStats['p-Values'], hue=rtStats['Condition'], palette=rt_pal)
# axes[1,0].set_title('RT Statistics')
# sns.boxplot(ax=axes[1,1], x=accStats['Condition'], y=accStats['p-Values'], hue=accStats['Condition'], palette=acc_pal)
# axes[1,1].set_title('Accuracy Statistics')
# fig.suptitle('Statistics of Intertrial Changes')
# plt.tight_layout()
# plt.show()
# index = -1
# rtNovTS = []
# rtPracTS = []
# for i in rtStats['Condition']:
#     index+=1
#     if i=='Novel':
#         rtNovTS.append(rtStats.iloc[index,rtStats.columns.get_loc("T-Statistics")])
#     else:
#         rtPracTS.append(rtStats.iloc[index,rtStats.columns.get_loc("T-Statistics")])
# xaxis = []
# rtNovP = []
# rtPracP = []
# index = -1
# n=0
# for i in rtStats['Condition']:
#     index+=1
#     if i=='Novel':
#         rtNovP.append(rtStats.iloc[index,rtStats.columns.get_loc("p-Values")])
#         n+=1
#         xaxis.append(n)
#     else:
#         rtPracP.append(rtStats.iloc[index,rtStats.columns.get_loc("p-Values")])

# index = -1
# accNovTS = []
# accPracTS = []
# xaxisP = []
# p=0
# for i in accStats['Condition']:
#     index+=1
#     if i=='Novel':
#         accNovTS.append(accStats.iloc[index,accStats.columns.get_loc("T-Statistics")])
#     else:
#         accPracTS.append(accStats.iloc[index,accStats.columns.get_loc("T-Statistics")])
#         p+=1
#         xaxisP.append(p)
# accNovP = []
# accPracP = []
# index = -1
# ap = 0
# xap = []
# for i in accStats['Condition']:
#     index+=1
#     if i=='Novel':
#         accNovP.append(accStats.iloc[index,accStats.columns.get_loc("p-Values")])
#         ap+=1
#         xap.append(ap)
#     else:
#         accPracP.append(accStats.iloc[index,accStats.columns.get_loc("p-Values")])

# plt.figure()
# plt.scatter(rtNovTS, xaxis, label='Novel - Reaction Time',color='lightskyblue')
# plt.scatter(rtPracTS, xaxisP, label='Practice - Reaction Time',color='steelblue')
# plt.scatter(accNovTS, xap, label='Novel - Accuracy',color='orchid')
# plt.scatter(accPracTS, xaxisP, label='Practice - Accuracy',color='darkorchid')
# plt.ylabel('First Trial Number',fontsize=13)
# plt.xlabel('T-statistic', fontsize = 13)
# plt.title('Intertrial change t-statistics', fontsize=15)
# plt.legend(loc="upper right", bbox_to_anchor=(1.75, 1), fontsize = 12)

# plt.figure()
# plt.scatter(rtNovP, xaxis, label='Novel - Reaction Time',color='lightskyblue')
# plt.scatter(rtPracP, xaxisP, label='Practice - Reaction Time',color='steelblue')
# plt.scatter(accNovP, xap, label='Novel - Accuracy',color='orchid')
# plt.scatter(accPracP, xaxisP, label='Practice - Accuracy',color='darkorchid')
# plt.ylabel('First Trial Number',fontsize=13)
# plt.xlabel('p-Value', fontsize = 13)
# plt.title('Intertrial change p-Values', fontsize=15)
# plt.legend(loc="upper right", bbox_to_anchor=(1.75, 1), fontsize = 12)
