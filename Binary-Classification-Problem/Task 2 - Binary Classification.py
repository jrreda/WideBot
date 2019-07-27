#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

# Import Datasets
train = pd.read_csv('training.csv', sep=';', decimal=',')
validation = pd.read_csv('validation.csv', sep=';', decimal=',')


# pre-processing

# 1st, Impute missing Numerci values
numeric_features = train.select_dtypes(np.number)
for col in numeric_features.columns:
    # Train dataset
    if train[col].isnull().values.any():
        train[col] = train[col].fillna(train[col].mean())
    # Validation dataset
    if validation[col].isnull().values.any():
        validation[col] = validation[col].fillna(validation[col].mean())

# 2nd, Impute categorical missing values
categorical_features = train.select_dtypes(np.object)
for col in categorical_features.columns:
    # Train dataset
    if train[col].isnull().values.any():
        train[col] = train[col].fillna(train[col].value_counts().index[0])
    # Validation dataset
    if validation[col].isnull().values.any():
        validation[col] = validation[col].fillna(validation[col].value_counts().index[0])


# Feature Transformation

le = LabelEncoder()
# Train dataset
for col in categorical_features.columns:
    train[col] = le.fit_transform(train[col])
# Validation dataset
for col in categorical_features.columns:
    validation[col] = le.fit_transform(validation[col])


# Specifying the features (X) and the target (y) FOR Train dataset
X = train.loc[:, train.columns != 'classLabel']
y = train.loc[:, 'classLabel']

# Specifying the features (X) and the target (y) FOR Validation dataset
X_val = validation.loc[:, validation.columns != 'classLabel']
y_val = validation.loc[:, 'classLabel']

# Split the X, y train set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# Train the Model
DC = DecisionTreeClassifier(max_features=2, min_samples_split=9, random_state=0).fit(X_train, y_train)
DC.score(X_test, y_test)


# predict and evaluate the Model using: Accuracy_score
DC_pred = DC.predict(X_val)

print(metrics.confusion_matrix(y_val, DC_pred))
print(metrics.accuracy_score(y_val, DC_pred))



# Plot the evaluation result using roc_curve
fpr, tpr, thresholds = metrics.roc_curve(y_val, DC_pred)

plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for diabetes classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()
