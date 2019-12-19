#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:01:43 2019

@author: Alice
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 14:01:43 2019

@author: Alice
"""
import numpy as np
import pickle
import pandas as pd
from sklearn import svm, linear_model, preprocessing, decomposition
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectFromModel
from keras.models import Sequential, Model
from keras.layers import Reshape, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, LSTM, SpatialDropout1D, GlobalMaxPooling1D, Input, concatenate, Activation
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

def do_PCA(train_std, val_std = np.array([]),nb_pc = 2):
    '''
    DESCRIPTION: Perform Principal Component Analysis on training and validation sets

    INPUT:
        |--- train_std: [array] 2D array of standardized train feature vectors for each training sample
        |--- val_std: [array] 2D array of validation feature vectors for each validation sample standardized using training metrics; 
        |--- nb_pc: [int] integer which represents number of principal components to generate
    OUTPUT:
        |--- pc_train: [array] 2D array nb training samples x nb of principal components, stores principal components of training matrix
        |--- pc_val: [array] 2D array nb validation samples x nb of principal components, projection of validation matrix onto training PCs
    '''
    
    pca = PCA(n_components=nb_pc)
    pc_train = pca.fit_transform(train_std)
    if val_std.any() : pc_val = pca.transform(val_std)
    else : pc_val = []

    return pc_train, pc_val

def do_tsne(train_std,val_std = np.array([]),num_dim =2):
    '''
    DESCRIPTION: Perform tSNE dimensionality reduction on training and validation sets

    INPUT:
        |--- train_std: [array] 2D array of standardized train feature vectors for each training sample
        |--- val_std: [array] 2D array of validation feature vectors for each validation sample standardized using training metrics; 
        |--- nb_dim: [int] dimensions of final subspace
    OUTPUT:
        |--- tsne_train: [array] 2D array nb training samples x nb of final dimensions, stores principal components of training matrix
        |--- tsne_val: [array] 2D array nb validation samples x nb of final dimensions, projection of validation matrix onto training tSNE subspace
    '''

    tsne = TSNE(n_components=num_dim, random_state=0)
    tsne_train = tsne.fit_transform(train_std)
    if val_std.any() :tsne_val = tsne.transform(val_std)
    else: tsne_val = np.array([])
    
    return tsne_train, tsne_val

def plot_feature_selection(accuracies_train, accuracies_val, classifier='Bayes'):
    '''
        DESCRIPTION: Plot evoluation of train and validation accuracies for different numbers of features

        INPUT:
            |--- accuracies_train :[arr] 1D array containing all trainaccuracies for an increasing number of features
            |--- accuracies_val : [arr] 1D array containing all validation accuracies for an increasing number of features
            |--- classifier : [str] name of classifier to add to title

    '''
    plt.figure()
    feat_number = range(1,len(accuracies_val)+1)
    plt.plot(feat_number, accuracies_train)
    plt.plot(feat_number,accuracies_val)
    plt.axvline(x=np.argmax(accuracies_val)+1, color='r', linestyle='--')
    plt.xlabel("Number of features")
    plt.ylabel("Accuracy")
    plt.title("Feature selection "+classifier)
    plt.legend('Train Accuracy','Val Accuracy')
    plt.savefig("FeatureSelection_"+classifier)

def svm_classification(X_train, X_val, y_train, y_val):
    '''
        DESCRIPTION : Perform training classification using Support Vector Machine on training set, tested on validation set

        INPUT:
            |--- X_train: [arr] 2D array nb samples x nb features, with each row is the features vector for each sample in training set
            |--- X_val: [arr] 2D array nb samples x nb features, with each row is the features vector for each sample in validation set
            |--- y_train: [list] list of 0 and 1 as class labels for the training set
            |--- y_val: [list] list of 0 and 1 as labels for the validation set
        OUTPUT:
            |--- classifier: SVM classifier trained on training data
            |--- acc_train: [float] accuracy of classification on training set
            |--- acc_val: [float] accuracy of classification on validation set
    '''

    print('SVM')

    loss_ = 'hinge'
    intercept = False
    regulariser = 0.5
    nb_features = X_train.shape[1]
    classifier = svm.LinearSVC(fit_intercept= intercept ,loss=loss_,C=regulariser).fit(X_train,y_train)
    train_prediction = classifier.predict(X_train)
    val_prediction = classifier.predict(X_val)
    acc_train = accuracy_score(y_train,train_prediction)
    acc_val = accuracy_score(y_val,val_prediction)

    print("Train Accuracy using SVM linear (" + loss_ + ' and ' + str(regulariser)+ ' regul. and ' + str(nb_features) + ' features): {:0.3f}'.format(acc_train))
    print("Val Accuracy using SVM linear (" + loss_ + ' and ' + str(regulariser)+ ' regul. and ' + str(nb_features) + ' features): {:0.3f}'.format(acc_val))

    return classifier, acc_train, acc_val

def logistic_classification(X_train, X_val, y_train, y_val):
    '''
        DESCRIPTION : Perform training classification using Logistic Regression on training set, tested on validation set

        INPUT:
            |--- X_train: [arr] 2D array nb samples x nb features, with each row is the features vector for each sample in training set
            |--- X_val: [arr] 2D array nb samples x nb features, with each row is the features vector for each sample in validation set
            |--- y_train: [list] list of 0 and 1 as class labels for the training set
            |--- y_val: [list] list of 0 and 1 as labels for the validation set
        OUTPUT:
            |--- classifier: Logistic Regression classifier trained on training data
            |--- acc_train: [float] accuracy of classification on training set
            |--- acc_val: [float] accuracy of classification on validation set
    '''
    print('Logistic Regression')
    nb_features = X_train.shape[1]
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    acc_val = logreg.score(X_val, y_val)
    acc_train = logreg.score(X_train, y_train)

    print("Train Accuracy using Logistic Reg with " + str(nb_features)+ " features:{:0.3f}".format(acc_train))
    print("Val Accuracy using Logistic Reg with " + str(nb_features)+ " features:{:0.3f}".format(acc_val))

    return logreg, acc_train, acc_val

def naive_bayes(X_train, X_val, y_train, y_val):
    '''
        DESCRIPTION : Perform training classification using Naive Bayes on training set, tested in validation set

        INPUT:
            |--- X_train: [arr] 2D array nb samples x nb features, with each row is the features vector for each sample in training set
            |--- X_val: [arr] 2D array nb samples x nb features, with each row is the features vector for each sample in validation set
            |--- y_train: [list] list of 0 and 1 as class labels for the training set
            |--- y_val: [list] list of 0 and 1 as labels for the validation set
        OUTPUT:
            |--- classifier: Naive Bayes classifier trained on training data
            |--- acc_train: [float] accuracy of classification on training set
            |--- acc_val: [float] accuracy of classification on validation set
    '''

    print('Naive Bayes')
    nb_features = X_train.shape[1]
    bayes = GaussianNB()
    bayes.fit(X_train, y_train)
    acc_val = bayes.score(X_val, y_val)
    acc_train = bayes.score(X_train, y_train)

    print("Train Accuracy using Naive Bayes with " + str(nb_features)+ " features:{:0.3f}".format(acc_train))
    print("Val Accuracy using Naive Bayes with " + str(nb_features)+ " features:{:0.3f}".format(acc_val))

    return bayes, acc_train, acc_val

def no_selection(train_, val_, train_labels, val_labels, classifier='Bayes',poly_expansion=False, degree=0):
    '''
        DESCRIPTION : Perform classification with no feature selection using a specific classifier

        INPUT:
            |--- train_: [arr] 2D array nb samples x nb features, with each row is the features vector for each sample in training set
            |--- val_: [arr] 2D array nb samples x nb features, with each row is the features vector for each sample in validation set
            |--- train_labels: [list] list of 0 and 1 as class labels for the training set
            |--- val_labels: [list] list of 0 and 1 as class labels for the validation set
            |--- classifier: [std] name of classifier to used in the list ['Bayes','SVM','LogReg']
            |--- poly_expansion: [bool] boolean if yes or not features expansion should be performed before performing classification
            |--- degree: [int] power up to which feature expansion should be performed
        OUTPUT:
            |--- acc_train: [float] accuracy of classification on training set
            |--- acc_val: [float] accuracy of classification on validation set
    '''

    X_train = train_
    X_val = val_
    y_train = train_labels
    y_val = val_labels

    acc_train = np.nan
    acc_val = np.nan
    
    if poly_expansion:
        poly = PolynomialFeatures(degree)
        X_train = poly.fit_transform(X_train)
        X_val = poly.transform(X_val) # check that it's the right way to proceed

    if classifier=='SVM':
        _, acc_train, acc_val = svm_classification(X_train, X_val, y_train, y_val)
    elif classifier=='LogReg':
        _, acc_train, acc_val = logistic_classification(X_train, X_val, y_train, y_val)
    elif classifier=='Bayes':
        _, acc_train, acc_val = naive_bayes(X_train, X_val, y_train, y_val)
    elif classifier=='NN':
        print('Implement NN')

    return acc_train, acc_val


def feature_selection(train_, val_, train_labels, val_labels, classifier='Bayes', RF_size=3000, only_RF=True, feat_select=False, poly_expansion=False, degree=0):
    """
        DESCRIPTION : Function designed in order to see the impact of iteratively increasing the number of features before the training. 
        The order chosen is based on the feature importances computed through the training of a random forest.
        Additionnally, one can decide to do feature augmentation by adding polynomial expansion.

        INPUTS:
            |--- tX : [arr] 2D array nb samples x nb features, with each row is the features vector for each sample in training set
            |--- y : [list] target labels
            |--- classifier : [str] name of the linear classifier chosen {'Bayes', 'LogReg'}
            |--- RF_size : [int] number of trees in the random forest
            |--- transformation : [bool] boolean. True : keep most important features, train and compute unique score. False: keep all feature and show score for each subset
            |--- poly_expansion : [bool] boolean indicating whether a polynomial expansion is performed
            |--- degree : [int] degree of the polynomial expansion
        OUPUTS: 
            |--- acc_train: [int] value of the training accuracy for this classifier with no feature selection
            |--- acc_val: [int] value of the validation accuracy for this classifier with no feature selection
            |--- accuracies_train: [list] list of training accuracy for each number of features added to model
            |--- accuracies_val: [list] list of validation accuracy for each number of features added to model
        """

    X_train = train_
    X_val = val_
    y_train = train_labels
    y_val = val_labels
    
    if poly_expansion:
        poly = PolynomialFeatures(degree)
        X_train = poly.fit_transform(X_train)
        X_val = poly.transform(X_val) # check that it's the right way to proceed

    # Initialise results variables
    accuracies_train = []
    accuracies_val = []
    acc_train= np.nan
    acc_val= np.nan

    print('\n Starting the Random Forest')
    X_train_transform, X_val_transform, ranked_index = random_forest_selection(X_train, y_train, X_val, RF_size)
    print('\n Random Forest Completed')

    if only_RF:
        if classifier=='SVM':
            model, acc_train, acc_val = svm_classification(X_train_transform, X_val_transform, y_train, y_val)
        elif classifier=='LogReg':
            model, acc_train, acc_val = logistic_classification(X_train_transform, X_val_transform, y_train, y_val)
        elif classifier=='Bayes':
            model, acc_train, acc_val = naive_bayes(X_train_transform, X_val_transform, y_train, y_val)
        elif classifier=='NN':
            print('Implement NN')

    if feat_select:
        X_train = X_train[:, ranked_index] # sort features from the more important to the least one
        X_val = X_val[:, ranked_index]
        print('Number features',X_train.shape[1])
        for i in range(X_train.shape[1]):

            X_train_trunc = X_train[:, :i+1]
            X_val_trunc = X_val[:, :i+1]

            if classifier=='SVM':
                model, acc_train, acc_val = svm_classification(X_train_trunc, X_val_trunc, y_train, y_val)
                accuracies_train.append(acc_train)
                accuracies_val.append(acc_val)
            elif classifier=='LogReg':
                model, acc_train, acc_val = logistic_classification( X_train_trunc, X_val_trunc, y_train, y_val)
                accuracies_train.append(acc_train)
                accuracies_val.append(acc_val)
            elif classifier=='Bayes':
                model, acc_train, acc_val = naive_bayes( X_train_trunc, X_val_trunc, y_train, y_val)
                accuracies_train.append(acc_train)
                accuracies_val.append(acc_val)
            elif classifier=='NN':
                print('Implement NN')

        print(accuracies_val,accuracies_train)
        if np.any(accuracies_val) : print('Best accuracy reached for {} features : {:0.3f}'.format(np.argmax(accuracies_val)+1, np.max(accuracies_val)))
        plot_feature_selection(accuracies_train,accuracies_val, classifier) #ascending order
    
    return acc_train, acc_val, accuracies_train, accuracies_val

def random_forest_selection(X_train, y_train, X_val, RF_size=3000):
    """
        DESCRIPTION : Function training a random forest on the dataset.
        INPUTS:
            |--- X_train, X_val : [arr] 2D array nb samples x nb features, with each row is the features vector for each sample in training/validation set
            |--- y_train : [list] list of labels of the training dataset
            |--- RF_size : [int] number of trees in the forest.
        OUTPUTS:
            |--- X_important_train, X_important_test : [arr] 2D reduced feature matrices, keeping only important features
            |--- ranked_index: [list] list of features indices ranked in descending importance order
            |--- transform : [bool] boolean to choose between transformation or feature ranking
    """

    X_important_train = []
    X_important_val = []
    ranked_index = []

    # Reduce number of features, keeping the ones with importance greater than the mean
    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=RF_size, random_state=0, n_jobs=-1) # TODO : check other parameters
    # Train the classifier
    clf.fit(X_train, y_train)

    sfm = SelectFromModel(clf)
    sfm.fit(X_train, y_train)
    print("Number of conserved features after RF : {}".format(np.sum(sfm.get_support())))
    X_important_train= X_train[:, sfm.get_support()] # get_support returns an array of boolean values. True for the features whose importance is greater than the mean importance and False for the rest.
    X_important_test = X_val[:, sfm.get_support()]

    # Print feature importances and get the ascending order index
    print("Feature importances : ",clf.feature_importances_)
    ranked_index = np.array(clf.feature_importances_).argsort() #ascending order
    ranked_index = ranked_index[::-1] #descending order
    
    return X_important_train, X_important_val, ranked_index

def training_pipeline(train_features,val_features,train_labels,val_labels):
    '''
        DESCRIPTION :

        INPUT:
            |--- train_features: [arr] 
            |--- val_features: [arr]
            |--- train_labels: [list]
            |--- val_labels: [list]
        OUTPUT:
            |--- results_noselection: [DataFrame]
            |--- results_selection: [DataFrame]
    '''
      
    # No feature selection, no polonomial expansion
    print('Processing : No feature selection')
    #acc_train_bayes, acc_val_bayes = no_selection(train_features, val_features, train_labels, val_labels, classifier='Bayes',poly_expansion=False, degree=0)
    #acc_train_log, acc_val_log = no_selection(train_features, val_features, train_labels, val_labels, classifier='LogReg',poly_expansion=False, degree=0)
    acc_train_svm, acc_val_svm = no_selection(train_features, val_features, train_labels, val_labels, classifier='SVM',poly_expansion=False, degree=0)
    #acc_train_NN1, acc_val_NN1 = no_selection(train_features, val_features, train_labels, val_labels, classifier='NN1',poly_expansion=False, degree=0)
    # No feature selection, polynomial expansion degree 2
    print('Processing : No feature selection, polynomial expansion')
    #acc_train_bayes_exp, acc_val_bayes_exp = no_selection(train_features, val_features, train_labels, val_labels, classifier='Bayes',poly_expansion=True, degree=2)
    #acc_train_log_exp, acc_val_log_exp = no_selection(train_features, val_features, train_labels, val_labels, classifier='LogReg',poly_expansion=True, degree=2)
    #acc_train_svm_exp, acc_val_svm_exp = no_selection(train_features, val_features, train_labels, val_labels, classifier='SVM',poly_expansion=True, degree=2)
    #acc_train_NN1_exp, acc_val_NN1_exp = no_selection(train_features, val_features, train_labels, val_labels, classifier='NN1',poly_expansion=True, degree=2)
    # Feature selection - RF + RF on its own, no polynomail expansion
    print('Processing : RF + Feature selection')
    #acc_train_bayes_RF, acc_val_bayes_RF, acc_train_bayes_featselect, acc_val_bayes_featselect = feature_selection(train_features, val_features, train_labels, val_labels, 
    #classifier='Bayes', RF_size=3000, only_RF=True, feat_select=True, poly_expansion=False, degree=0)
    #acc_train_log_RF, acc_val_log_RF, acc_train_log_featselect, acc_val_log_featselect = feature_selection(train_features, val_features, train_labels, val_labels, 
    #classifier='LogReg', RF_size=3000, only_RF=True, feat_select=True, poly_expansion=False, degree=0)
    #acc_train_svm_RF, acc_val_svm_RF, acc_train_svm_featselect, acc_val_svm_featselect = feature_selection(train_features, val_features, train_labels, val_labels, 
    #classifier='SVM', RF_size=3000, only_RF=True, feat_select=True, poly_expansion=False, degree=0)
    #acc_train_NN1_RF, acc_val_NN1_RF, acc_train_NN1_featselect, acc_val_NN1_featselect = feature_selection(train_features, val_features, train_labels, val_labels, 
    #classifier='NN1', RF_size=3000, only_RF=True, feat_select=True, poly_expansion=False, degree=0)

    # Creation of results dataframe
    #results_noselection = pd.DataFrame(data = {'Bayes':[acc_train_bayes, acc_val_bayes],'Logistic':[acc_train_log, acc_val_log],
    #                                'SVM':[acc_train_svm, acc_val_svm],'NN1':[acc_train_NN1, acc_val_NN1],'Bayes Exp':[acc_train_bayes_exp, acc_val_bayes_exp],
    #                                'Logistic Exp':[acc_train_log_exp, acc_val_log_exp],'SVM Exp':[acc_train_svm_exp, acc_val_svm_exp],
    #                                'NN Exp':[acc_train_NN1_exp, acc_val_NN1_exp],'Bayes RF':[acc_train_bayes_RF, acc_val_bayes_RF],
    #                                'Logistic RF':[acc_train_log_RF, acc_val_log_RF],'SVM RF':[acc_train_svm_RF, acc_val_svm_RF],
    #                                'NN RF':[acc_train_NN1_RF, acc_val_NN1_RF]}, index = ['Train','Validation'],dtype =np.float64)
    #results_selection = pd.DataFrame(data = {'Bayes Train':acc_train_bayes_featselect,'Bayes Val':acc_val_bayes_featselect,'Log Train':acc_train_log_featselect,
    #                                'Log val':acc_val_log_featselect,'SVM Train':acc_train_svm_featselect,'SVM Val':acc_val_svm_featselect,
    #                                'NN Train':acc_train_NN1_featselect,'NN Val':acc_val_NN1_featselect}, 
    #                                index = [str(x+1) + ' features' for x in range(len(acc_train_bayes_RF))])
    results_noselection = pd.DataFrame(data = {'SVM':[acc_train_svm, acc_val_svm]})
    results_selection = pd.DataFrame(data = [])
    return results_noselection, results_selection
                            