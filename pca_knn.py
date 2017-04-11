#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 21:37:36 2017

@author: steven
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

def pca(X_tr, X_ts, teste, n):
    pca = PCA(n)
    pca.fit(X_tr)
    X_tr_pca = pca.transform(X_tr)
    X_ts_pca = pca.transform(X_ts)
    teste = pca.transform(teste)    
    return X_tr_pca, X_ts_pca, teste

os.chdir("/Users/steven/Documents/dataMining/Kaggle/digitRecognizer")
cwd = os.getcwd()
print("setting working directory to...", cwd)

print("Loading data...")
train = pd.read_csv("input/train.csv")
test  = pd.read_csv("input/test.csv")

print("Training set has {0[0]} rows and {0[1]} columns".format(train.shape))
print("Test set has {0[0]} rows and {0[1]} columns".format(test.shape))

Y_train = train["label"]
X_train = train.drop("label",1)
header = X_train.columns

print("Starting to split the training data set into 70/30")
X_70_train, X_30_train, Y_70_train, Y_30_train = train_test_split(X_train, Y_train, test_size=0.30, random_state=2)

print("Starting Normalization...")
norm = Normalizer().fit(X_70_train)
X_70_tr_norm = norm.transform(X_70_train)
X_30_tr_norm = norm.transform(X_30_train)
test = norm.transform(test)

print("Starting PCA analysis...")
X_70_tr_norm_pca, X_30_tr_norm_pca, test = pca(X_70_tr_norm, X_30_tr_norm, test, 50)
X_70_tr_norm_pca = pd.DataFrame(X_70_tr_norm_pca) 
X_30_tr_norm_pca = pd.DataFrame(X_30_tr_norm_pca)
test = pd.DataFrame(test)

print("Starting KNN analysis...")
model = KNeighborsClassifier(n_neighbors = 3, weights='distance')
model.fit(X_70_tr_norm_pca, Y_70_train)
score = model.score(X_30_tr_norm_pca, Y_30_train)
print ('KNN ', score)
pred = model.predict(test)

file_name = "3nn_50pca.csv"    
print("Exporting data to "+file_name)
submission = pd.DataFrame({
    "ImageId": np.arange(1, pred.shape[0] + 1),
    "Label": pred
})
submission.to_csv(file_name, index=False)

