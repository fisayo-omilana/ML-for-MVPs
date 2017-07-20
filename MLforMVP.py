import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import csv
import matplotlib.pyplot as plt
from sklearn import cluster
from __future__ import division
get_ipython().magic(u'matplotlib inline')


#Loading training data (2002/03-2012/13 Regular Seasons)
train_file = open('mvp-trnblank.csv','rU')
mvp_train =  pd.read_csv(train_file)
#Loading test data (2013/14-15/16 Regular Seasons)
test_file = open('mvp-tstblank.csv','rU')
mvp_test =  pd.read_csv(test_file)
#Loading respective labeled training data
Ltrain_file = open('mvp-trnlabeled.csv','rU')
Lmvp_train = pd.read_csv(Ltrain_file)
#Loading respective labeled test data
Ltest_file = open('mvp-tstlabeled.csv', 'rU')
Lmvp_test = pd.read_csv(Ltest_file)



def train_and_test_classifier(classifier, features, target, train_data, test_data, labeled_train, labeled_test, labels=True):  
    """
    fits a classifier to the provided training data, prints out and returns the prediction 
    accuracy on both training and test sets.
    
    Parameters:
    -----------
    classifier: The classifier object, e.g. KNN classifier
    features: a list of data frame columns to use as features. 
    target: the data frame column to use as target. E.g. "category"
    train_data: a Pandas data frame containing the training data
    test_data: a Pandas data frame containing the test data
    labels: If True, prints out the player name, predicted value, and the actual value 
                for every sample for both training and test sets.          
    Returns:
    --------
    train_acc: accuracy on training set
    test_acc: accuracy on test set
    train_prec: precision on training set
    test_prec: precision on test set
    train_rec: recall on training set
    test_rec: recall on test set
    """
#Sizes of training & test data...
    num_train = len(train_data)
    num_test = len(test_data)

#Training classifier model...
    classifier.fit(train_data[features], train_data[target])

#Predicting results from training data
    train_predictions = classifier.predict(train_data[features])

#Predicting results from test data
    test_predictions = classifier.predict(test_data[features])   

#Calculating training accuracy...
    train_correct = 0
    train_truepos = 0
    positives = 0
    train_false_neg = 0
    if labels == True: print "TRAINING..."
    for i in range(num_train):
        if labels == True: print labeled_train['PLAYER'][i], " || ", 'Predicted:', train_predictions[i], ' Actual:', train_data.ix[i][target]
        if train_predictions[i] == train_data.ix[i][target]: train_correct += 1
        if train_data.ix[i][target] != '0': positives += 1  
        if train_predictions[i] != '0' and train_data.ix[i][target] != '0': train_truepos += 1
        if train_predictions[i] != '0' and train_data.ix[i][target] == '0': train_false_neg += 1
    train_acc = float(train_correct)/float(num_train)
    train_prec = float(train_truepos)/float(positives)
    train_rec = float(train_truepos)/float(train_false_neg + train_truepos)
    print 'Training Accuracy:', train_acc
    print 'Training Precision:', train_prec
    print 'Training Recall:', train_rec

#Calculating test accuracy...
    test_correct = 0
    test_truepos = 0
    positives = 0
    test_false_neg = 0
    if labels == True: print "TESTING..."
    for i in range(num_test):
        if labels == True: print labeled_test['PLAYER'][i], "||", 'Predicted:', test_predictions[i], ' Actual:', test_data.ix[i][target]
        if test_predictions[i] == test_data.ix[i][target]: test_correct += 1
        if test_data.ix[i][target] != '0': positives += 1
        if test_predictions[i] != '0' and test_data.ix[i][target] != '0': test_truepos += 1
        if test_predictions[i] != '0' and test_data.ix[i][target] == '0': test_false_neg += 1
    test_acc = float(test_correct)/float(num_test)
    test_prec = float(test_truepos)/float(positives)
    test_rec = float(test_truepos)/float(test_false_neg + test_truepos)
    print 'Test Accuracy:', test_acc
    print 'Test Precision:', test_prec
    print 'Test Recall:', test_rec
    return train_acc, test_acc, train_prec, test_prec, train_rec, test_rec


# DETERMINING BEST CLASSIFIER

#Using Naive Bayes...
from sklearn.naive_bayes import GaussianNB
train_and_test_classifier(GaussianNB(), ['MPG', 'TS%', 'ASTR', 'USG', 'ORR', 'DRR', 'REBR', 'PER', 'VA', 'EWA', 'PPG', 'ORPG', 'DRPG', 'REB', 'RPG', 'RP48', 'FGMPG', 'FGAPG', 'FGM', '2P%', 'PPS', 'ADJFG%', 'FTMPG', 'FTAPG', 'FTM', 'FT%', '3PMPG', '3PAPG', '3PTM', '3P%', 'AST', 'APG', 'AP48M', 'AST/TO', 'STL', 'STPG', 'STP48M', 'ST/PF', 'BLK', 'BLKPG', 'BLKP48M', 'BLK/PF', 'DBLDBL', 'TRIDBL'], 'MVP', mvp_train, mvp_test, Lmvp_train, Lmvp_test, labels=False)


#Using Logistic Regression...
from sklearn.linear_model import LogisticRegression
train_and_test_classifier(LogisticRegression(), ['MPG', 'TS%', 'ASTR', 'USG', 'ORR', 'DRR', 'REBR', 'PER', 'VA', 'EWA', 'PPG', 'ORPG', 'DRPG', 'REB', 'RPG', 'RP48', 'FGMPG', 'FGAPG', 'FGM', '2P%', 'PPS', 'ADJFG%', 'FTMPG', 'FTAPG', 'FTM', 'FT%', '3PMPG', '3PAPG', '3PTM', '3P%', 'AST', 'APG', 'AP48M', 'AST/TO', 'STL', 'STPG', 'STP48M', 'ST/PF', 'BLK', 'BLKPG', 'BLKP48M', 'BLK/PF', 'DBLDBL', 'TRIDBL'], 'MVP', mvp_train, mvp_test, Lmvp_train, Lmvp_test, labels=False)


#Using K-Nearest Neighbors...
from sklearn.neighbors import KNeighborsClassifier
klist = [1, 2, 5, 10, 20]
for k in klist:
    print "{} NEIGHBOR(S)".format(k)
    train_and_test_classifier(KNeighborsClassifier(n_neighbors=k), ['MPG', 'TS%', 'ASTR', 'USG', 'ORR', 'DRR', 'REBR', 'PER', 'VA', 'EWA', 'PPG', 'ORPG', 'DRPG', 'REB', 'RPG', 'RP48', 'FGMPG', 'FGAPG', 'FGM', '2P%', 'PPS', 'ADJFG%', 'FTMPG', 'FTAPG', 'FTM', 'FT%', '3PMPG', '3PAPG', '3PTM', '3P%', 'AST', 'APG', 'AP48M', 'AST/TO', 'STL', 'STPG', 'STP48M', 'ST/PF', 'BLK', 'BLKPG', 'BLKP48M', 'BLK/PF', 'DBLDBL', 'TRIDBL'], 'MVP', mvp_train, mvp_test, Lmvp_train, Lmvp_test, labels=False)
    print ""


#Using Decision Trees...
from sklearn.tree import DecisionTreeClassifier
split_parameters = [2, 4, 10, 20, 40]
for split in split_parameters:
    print "{} SPLITS".format(split)
    train_and_test_classifier(DecisionTreeClassifier(min_samples_split=split), ['MPG', 'TS%', 'ASTR', 'USG', 'ORR', 'DRR', 'REBR', 'PER', 'VA', 'EWA', 'PPG', 'ORPG', 'DRPG', 'REB', 'RPG', 'RP48', 'FGMPG', 'FGAPG', 'FGM', '2P%', 'PPS', 'ADJFG%', 'FTMPG', 'FTAPG', 'FTM', 'FT%', '3PMPG', '3PAPG', '3PTM', '3P%', 'AST', 'APG', 'AP48M', 'AST/TO', 'STL', 'STPG', 'STP48M', 'ST/PF', 'BLK', 'BLKPG', 'BLKP48M', 'BLK/PF', 'DBLDBL', 'TRIDBL'], 'MVP', mvp_train, mvp_test, Lmvp_train, Lmvp_test, labels=False)
    print ""


#Using Random Forests...
from sklearn.ensemble import RandomForestClassifier
n_trees_list = [1, 2, 5, 10, 20]
for trees in n_trees_list:
    print "{} TREE(S)".format(trees)
    train_and_test_classifier(RandomForestClassifier(n_estimators=trees), ['MPG', 'TS%', 'ASTR', 'USG', 'ORR', 'DRR', 'REBR', 'PER', 'VA', 'EWA', 'PPG', 'ORPG', 'DRPG', 'REB', 'RPG', 'RP48', 'FGMPG', 'FGAPG', 'FGM', '2P%', 'PPS', 'ADJFG%', 'FTMPG', 'FTAPG', 'FTM', 'FT%', '3PMPG', '3PAPG', '3PTM', '3P%', 'AST', 'APG', 'AP48M', 'AST/TO', 'STL', 'STPG', 'STP48M', 'ST/PF', 'BLK', 'BLKPG', 'BLKP48M', 'BLK/PF', 'DBLDBL', 'TRIDBL'], 'MVP', mvp_train, mvp_test, Lmvp_train, Lmvp_test, labels=False)
    print ""


# PREDICTING THE 2016-17 MOST VALUALBE PLAYER

#Loading files to predict this year's NBA Regular Season MVP...
#Using all previous data as training data...
complete_file = open('mvp-allblank.csv','rU')
allmvp_train =  pd.read_csv(complete_file)
complete_labels = open('mvp-alllabeled.csv', 'rU')
allmvp_labels = pd.read_csv(complete_labels)
#Testing purely on this season's data...
test2017 = open('2016-17blank.csv', 'rU')
mvp2017predictions = pd.read_csv(test2017)
label2017 = open('2016-17labeled.csv', 'rU')
mvp2017labels = pd.read_csv(label2017)


#Using Random Forests with 10 trees input because it had the best overall performance of all models...
from sklearn.ensemble import RandomForestClassifier
train_and_test_classifier(RandomForestClassifier(n_estimators=10), ['MPG', 'TS%', 'ASTR', 'USG', 'ORR', 'DRR', 'REBR', 'PER', 'VA', 'EWA', 'PPG', 'ORPG', 'DRPG', 'REB', 'RPG', 'RP48', 'FGMPG', 'FGAPG', 'FGM', '2P%', 'PPS', 'ADJFG%', 'FTMPG', 'FTAPG', 'FTM', 'FT%', '3PMPG', '3PAPG', '3PTM', '3P%', 'AST', 'APG', 'AP48M', 'AST/TO', 'STL', 'STPG', 'STP48M', 'ST/PF', 'BLK', 'BLKPG', 'BLKP48M', 'BLK/PF', 'DBLDBL', 'TRIDBL'], 'MVP', allmvp_train, mvp2017predictions, allmvp_labels, mvp2017labels, labels=True)


# RESULTS FROM 10 RANDOM SAMPLE RUNS OF THE RANDOM FORESTS MODEL
# 
#     Westbrook 1st, 4th, 4th, 4th, 4th, 4th, 4th, 1st, 1st, 4th
#     Durant 2nd
#     Leonard 2nd, 2nd, 1st, 1st, 1st, 1st
#     Davis 1st, 2nd, 5th, 3rd
#     Harden 1st, 2nd, 2nd, 2nd, 1st, 1st, 2nd
#     James 1st, 1st, 1st, 2nd, 3rd, 1st, 1st, 3rd, 3rd, 2nd
#     Thomas 5th, 1st, 5th
#     Antetokounpo 3rd
#     Towns 1st, 3rd, 1st
#     Curry 1st, 5th
#     
# USING A TIER SYSTEM BASED ON 'NUMBER OF TOP 5 PREDICTIONS' AND 'AVERAGE PREDICTION' WITHIN EACH TIER
# 
#     Tier 10: 
#         James || avg = 2.4 || 1st
#         Westbrook || avg = 3.1 || 2nd
#     Tier 9:
#         None
#     Tier 8:
#         None
#     Tier 7:
#         Harden || avg = 2.1 || 3rd
#     Tie 6:
#         Leonard || avg = 1.3 || 4th
#     Tier 5:
#         None
#     Tier 4:
#         Davis || avg = 2.7 || 5th
#         
#     

# DETERMINING MOST TELLING STATISTICAL CATEGORIES
#     (repeat of function defintion to comment out print statements)

def train_and_test_classifier(classifier, features, target, train_data, test_data, labeled_train, labeled_test, labels=True):  
    """
    fits a classifier to the provided training data, prints out and returns the prediction 
    accuracy on both training and test sets.
    
    Parameters:
    -----------
    classifier: The classifier object, e.g. KNN classifier
    features: a list of data frame columns to use as features. 
    target: the data frame column to use as target. E.g. "category"
    train_data: a Pandas data frame containing the training data
    test_data: a Pandas data frame containing the test data
    labels: If True, prints out the player name, predicted value, and the actual value 
                for every sample for both training and test sets.          
    Returns:
    --------
    train_acc: accuracy on training set
    test_acc: accuracy on test set
    train_prec: precision on training set
    test_prec: precision on test set
    train_rec: recall on training set
    test_rec: recall on test set
    """
#Sizes of training & test data...
    num_train = len(train_data)
    num_test = len(test_data)

#Training classifier model...
    classifier.fit(train_data[features], train_data[target])

#Predicting results from training data
    train_predictions = classifier.predict(train_data[features])

#Predicting results from test data
    test_predictions = classifier.predict(test_data[features])   

#Calculating training accuracy...
    train_correct = 0
    train_truepos = 0
    positives = 0
    train_false_neg = 0
    if labels == True: print "TRAINING..."
    for i in range(num_train):
        if labels == True: print labeled_train['PLAYER'][i], " || ", 'Predicted:', train_predictions[i], ' Actual:', train_data.ix[i][target]
        if train_predictions[i] == train_data.ix[i][target]: train_correct += 1
        if train_data.ix[i][target] != '0': positives += 1  
        if train_predictions[i] != '0' and train_data.ix[i][target] != '0': train_truepos += 1
        if train_predictions[i] != '0' and train_data.ix[i][target] == '0': train_false_neg += 1
    train_acc = float(train_correct)/float(num_train)
    train_prec = float(train_truepos)/float(positives)
    train_rec = float(train_truepos)/float(train_false_neg + train_truepos)
#     print 'Training Accuracy:', train_acc
#     print 'Training Precision:', train_prec
#     print 'Training Recall:', train_rec

#Calculating test accuracy...
    test_correct = 0
    test_truepos = 0
    positives = 0
    test_false_neg = 0
    if labels == True: print "TESTING..."
    for i in range(num_test):
        if labels == True: print labeled_test['PLAYER'][i], "||", 'Predicted:', test_predictions[i], ' Actual:', test_data.ix[i][target]
        if test_predictions[i] == test_data.ix[i][target]: test_correct += 1
        if test_data.ix[i][target] != '0': positives += 1
        if test_predictions[i] != '0' and test_data.ix[i][target] != '0': test_truepos += 1
        if test_predictions[i] != '0' and test_data.ix[i][target] == '0': test_false_neg += 1
    test_acc = float(test_correct)/float(num_test)
    test_prec = float(test_truepos)/float(positives)
    test_rec = float(test_truepos)/float(test_false_neg + test_truepos)
#     print 'Test Accuracy:', test_acc
#     print 'Test Precision:', test_prec
#     print 'Test Recall:', test_rec
    return train_acc, test_acc, train_prec, test_prec, train_rec, test_rec


allfeat = ['MPG', 'TS%', 'ASTR', 'USG', 'ORR', 'DRR', 'REBR', 'PER', 'VA', 'EWA', 'PPG', 'ORPG', 'DRPG', 'REB', 'RPG', 'RP48', 'FGMPG', 'FGAPG', 'FGM', '2P%', 'PPS', 'ADJFG%', 'FTMPG', 'FTAPG', 'FTM', 'FT%', '3PMPG', '3PAPG', '3PTM', '3P%', 'AST', 'APG', 'AP48M', 'AST/TO', 'STL', 'STPG', 'STP48M', 'ST/PF', 'BLK', 'BLKPG', 'BLKP48M', 'BLK/PF', 'DBLDBL', 'TRIDBL']
allfeatset = set(['MPG', 'TS%', 'ASTR', 'USG', 'ORR', 'DRR', 'REBR', 'PER', 'VA', 'EWA', 'PPG', 'ORPG', 'DRPG', 'REB', 'RPG', 'RP48', 'FGMPG', 'FGAPG', 'FGM', '2P%', 'PPS', 'ADJFG%', 'FTMPG', 'FTAPG', 'FTM', 'FT%', '3PMPG', '3PAPG', '3PTM', '3P%', 'AST', 'APG', 'AP48M', 'AST/TO', 'STL', 'STPG', 'STP48M', 'ST/PF', 'BLK', 'BLKPG', 'BLKP48M', 'BLK/PF', 'DBLDBL', 'TRIDBL'])
differences_map = {}
for feature in allfeatset:
    removable = set([feature])
    testfeat = list(allfeatset - removable)
    #Using logistic regression as the model of choice because it is the best overall performing classifier that has no random variability for each run
    all_train_acc, all_test_acc, all_train_prec, all_test_prec, all_train_rec, all_test_rec = train_and_test_classifier(LogisticRegression(), allfeat, 'MVP', mvp_train, mvp_test, Lmvp_train, Lmvp_test, labels=False)
    diff_train_acc, diff_test_acc, diff_train_prec, diff_test_prec, diff_train_rec, diff_test_rec = train_and_test_classifier(LogisticRegression(), testfeat, 'MVP', mvp_train, mvp_test, Lmvp_train, Lmvp_test, labels=False)
    differences_map[feature] = [all_train_acc - diff_train_acc, all_test_acc - diff_test_acc, all_train_prec - diff_train_prec, all_test_prec - diff_test_prec, all_train_rec - diff_train_rec, all_test_rec - diff_test_rec]
for feature in differences_map:
    sum_diff = 0
    for difference in differences_map[feature]:
        sum_diff += abs(difference)
    print feature, sum_diff


# TOP 8 MOST TELLING STATISTICAL CATEGORIES
# 
#     MPG 32.0937885644% NET DIFFERENCE IN RESULTS
#     DBLDBL 17.2406417112% NET DIFFERENCE IN RESULTS
#     AST 13.541371974% NET DIFFERENCE IN RESULTS
#     PPG 13.4700789407% NET DIFFERENCE IN RESULTS
#     USG 9.61455270113% NET DIFFERENCE IN RESULTS
#     STL 9.1876114082% NET DIFFERENCE IN RESULTS
#     PER 7.97492163009% NET DIFFERENCE IN RESULTS
#     TRIDBL 7.06053418435% NET DIFFERENCE IN RESULTS



