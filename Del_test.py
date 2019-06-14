# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:48:15 2019

@author: masoudg
"""


import pandas as pd
import numpy as np
import os


# lets try a bunch of classifiers to see which is works better
from sklearn.decomposition import PCA  
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# import some cross-validation and reporting stuff
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# import libraries for visualization
import matplotlib.pyplot as plt
import seaborn as sns


# data path and file name
data_path   = 'C:/MyFolder/Git/rock_mine_test'
result_path = 'C:/MyFolder/Git/rock_mine_test'
file_name   = 'sonar.all-data'

# first lets read the tsv data file
df           = pd.read_csv(os.path.join(data_path, file_name), header=None, prefix='X')

# the last column are labels, so lets rename it 
df.rename(columns={'X60':'Label'}, inplace=True)
df.Label     = df.Label.astype('category')
data, labels = df.ix[:, :-1], df.ix[:, -1]

# lets normalize the features before doing anything
data_norm = pd.DataFrame(StandardScaler().fit_transform(data), columns=data.columns)

# split the data to train and test
X_train, X_test, y_train, y_test = \
  train_test_split(data_norm, labels, test_size=0.2, random_state=18)

# just trying a buch of classifiers to see which one is better
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="rbf", C=0.025, probability=True),
    NuSVC(probability=True),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]
   
# Logging for Visual Comparison
log_cols=["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)


for clf in classifiers:
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print("*********** Results *********")
    train_prediction = clf.predict(X_test)
    acc = accuracy_score(y_test, train_prediction)
    print("Accuracy: {:.4%}".format(acc))
    
    train_prediction = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_prediction)
    print("loss: {}".format(ll))
    
    
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols)
    log = log.append(log_entry)
    
print("="*30)


# visualization
result     = pd.DataFrame(log)
result_acc = result.groupby(["Classifier"])['Accuracy'].aggregate(np.median).reset_index().sort_values('Accuracy')
result_lss = result.groupby(["Classifier"])['Log Loss'].aggregate(np.median).reset_index().sort_values('Log Loss')

fig, ax    = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))
ax         = plt.subplot(2, 1, 1)
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b", order=result_acc['Classifier'])
plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')

ax         = plt.subplot(2, 1, 2)
sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g", order=result_lss['Classifier'])
plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.savefig(result_path + "/"  + "classification_results", dpi=300)
plt.show()

# do some dimension reudction and then classify
pca             = PCA()
data_norm_pca   = pd.DataFrame(pca.fit_transform(data_norm), columns=data.columns)

# we can only keek the most important dimensions and remove the rest
# keeping those that explain at least 95% of total variance
variance        = pca.explained_variance_ratio_
cum_variance    = variance.cumsum()
n_comps         = 1 + np.argmax(cum_variance > 0.95)
data_norm_pca   = data_norm_pca.ix[:, :n_comps]

X_train_pca, X_test_pca, y_train_pca, y_test_pca = \
  train_test_split(data_norm_pca, labels, test_size=0.2, random_state=18)

# doing the classification again
log_cols_pca=["Classifier", "Accuracy", "Log Loss"]
log_pca = pd.DataFrame(columns=log_cols_pca)

for clf in classifiers:
    clf.fit(X_train_pca, y_train_pca)
    name = clf.__class__.__name__
    
    print("="*30)
    print(name)
    
    print("*********** Results after dimension reduction *********")
    train_prediction = clf.predict(X_test_pca)
    acc = accuracy_score(y_test_pca, train_prediction)
    print("Accuracy: {:.4%}".format(acc))
    
    train_prediction = clf.predict_proba(X_test_pca)
    ll = log_loss(y_test_pca, train_prediction)
    print("loss: {}".format(ll))
      
    log_entry = pd.DataFrame([[name, acc*100, ll]], columns=log_cols_pca)
    log_pca = log_pca.append(log_entry)
    
print("="*30)


result_pca     = pd.DataFrame(log_pca)
result_acc_pca = result.groupby(["Classifier"])['Accuracy'].aggregate(np.median).reset_index().sort_values('Accuracy')
result_lss_pca = result.groupby(["Classifier"])['Log Loss'].aggregate(np.median).reset_index().sort_values('Log Loss')

fig, ax    = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))
ax         = plt.subplot(2, 1, 1)
sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b", order=result_acc_pca['Classifier'])
plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy after PCA')

ax         = plt.subplot(2, 1, 2)
sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g", order=result_lss_pca['Classifier'])
plt.xlabel('Log Loss')
plt.title('Classifier Log Loss after PCA')
plt.savefig(result_path + "/"  + "classification_results_PCA", dpi=300)
plt.show()

# as the best classfier turns out to be NuSVC, we can do some grid search to 
# find the best parameters
parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'nu': [0.25, 0.5]},
               {'kernel': ['linear'],'nu': [0.25, 0.5]}]

clf = GridSearchCV(NuSVC(nu=0.5), parameters, cv=5, scoring='accuracy', n_jobs=-1)
clf.fit(X_train_pca, y_train_pca)

clf.score(X_test_pca, y_test_pca)
y_pred = clf.predict(X_test_pca)

print(classification_report(y_test_pca, y_pred))









