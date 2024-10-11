#!/usr/bin/env python
# coding: utf-8

# 
# #  NEURAL NETWORKS AND DEEP LEARNING FOR FINANCE
#      Autumn 2020
# 
# ## Credit scoring use case
# 
# ####   Use different deep learning architectures to build a credit scoring model
# 
# - Load data and impute missing values
# - Model 1: First basic linear model
# - Model 2: Deep model with 2 layers
# - Model 3: Deep model with 5 layers
# - Model 4: Final model with regularization and early stopping
# 
# - Downloaded from: https://www.kaggle.com/c/GiveMeSomeCredit
#  ******************************************************************       
#         
# ## Your Task
# Find optimum articture for deep learning model,
# 
# For this task, please keet the model architecture simple. You need to create a function which allows to parameterise the
# choice of hyperparameters in the neural network ((similar to Titanic model) ). Use a formal procedure for tuning parameters. The output layer has a sigmoid activation function, which is used to 'squash' all our outputs to be between 0 and 1.
# Evaluate your results with ROC plot and confusion matrix.
# 
# ### Report
# 
# Report your results with Python code in this nootbook file. Please explain the result in details. Reporting codes and graphs without explainations is not enough.

# In[2]:


from __future__ import print_function

import numpy as np
import pandas as pd
import time
import os

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10, 10)

import tensorflow as tf
print('Tensorflow version: ', tf.__version__)


# In[4]:


# Import data
#data_path='/Users/jorge/data/training/finance/credit_scoring/'

#data = pd.read_csv(os.path.join('cs-training.csv'), index_col=0)
data = pd.read_csv('cs-training.csv', index_col=0)

data


# In[5]:


# Check missing values

print('Count of missing values \n-----------------------')
print(data.isnull().sum(axis=0))


# In[6]:


# Generate missing imputators variables.
data['MonthlyIncomeMissing'] = data.apply(lambda row: 1 if pd.isnull(row['MonthlyIncome']) else 0, axis=1)
data['NumberOfDependentsMissing'] = data.apply(lambda row: 1 if pd.isnull(row['NumberOfDependents']) else 0, axis=1)

# Impute missing values by the mean
data = data.fillna(data.mean())


# In[ ]:


# Analyze distributions
data.describe()


# In[ ]:


# Transform by log
data['DebtRatioLog'] = np.log(data['DebtRatio']+1)
data['MonthlyIncomeLog'] = np.log(data['MonthlyIncome']+1)
data['RevolvingUtilizationOfUnsecuredLinesLog'] = np.log(data['RevolvingUtilizationOfUnsecuredLines']+1)

data['NumberOfTime30-59DaysPastDueNotWorseLog'] = np.log(data['NumberOfTime30-59DaysPastDueNotWorse']+1)
data['NumberOfOpenCreditLinesAndLoansLog'] = np.log(data['NumberOfOpenCreditLinesAndLoans']+1)
data['NumberOfTimes90DaysLateLog'] = np.log(data['NumberOfTimes90DaysLate']+1)
data['NumberRealEstateLoansOrLinesLog'] = np.log(data['NumberRealEstateLoansOrLines']+1)
data['NumberOfTime60-89DaysPastDueNotWorseLog'] = np.log(data['NumberOfTime60-89DaysPastDueNotWorse']+1)
data['NumberOfDependentsLog'] = np.log(data['NumberOfDependents']+1)

data


# In[ ]:


# Separate train test and convert to numpy arrays
from sklearn.model_selection import train_test_split

input_vars = ['RevolvingUtilizationOfUnsecuredLinesLog',
             'age',
             'NumberOfTime30-59DaysPastDueNotWorseLog',
             'DebtRatioLog',
             'MonthlyIncomeLog',
             'NumberOfOpenCreditLinesAndLoansLog',
             'NumberOfTimes90DaysLateLog',
             'NumberRealEstateLoansOrLinesLog',
             'NumberOfTime60-89DaysPastDueNotWorseLog',
             'NumberOfDependentsLog',
             'MonthlyIncomeMissing',
             'NumberOfDependentsMissing']

target_var = ['SeriousDlqin2yrs']

X_train, X_test, y_train, y_test = train_test_split(data[input_vars], data[target_var], test_size=0.2, random_state=42)
X_train = np.array(X_train, dtype=np.float32)
X_test  = np.array(X_test,  dtype=np.float32)
y_train = np.array(y_train, dtype=np.uint8)
y_test  = np.array(y_test,  dtype=np.uint8)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train = np.array(X_train, dtype=np.float32)
X_val  = np.array(X_val,  dtype=np.float32)
y_train = np.array(y_train, dtype=np.uint8)
y_val  = np.array(y_val,  dtype=np.uint8)


# # Model 1: Linear model standard parameters

# In[ ]:


# First model - Linear model

import tensorflow as tf

print('Linear model...')
input_data =  tf.keras.layers.Input(batch_shape=(None, 12), dtype='float32', name='Input_data')

output_data =  tf.keras.layers.Dense(2, activation='softmax', name='Dense_output')(input_data)

# Model Architecture defined
model_linear = tf.keras.models.Model(inputs=input_data, outputs=output_data)
model_linear.summary()


# In[ ]:


# Select optimizer and compile model
model_linear.compile(loss='sparse_categorical_crossentropy',  optimizer='Adam', metrics=['accuracy'])
#mymodel.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[ ]:


# Train the model
#start = time.time()

#tb_callback_ln = callbacks.TensorBoard(log_dir='/tmp/tensorboard/credit/linear/')

history_linear = model_linear.fit(X_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(X_val, y_val))

#print('Seconds to train: ', time.time() - start)


# In[ ]:


# Plot histograms
plt.rcParams['figure.figsize'] = (15, 20)
data.hist(data.columns)


# In[30]:


from sklearn import metrics

p_test = model_linear.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, p_test[:,1], pos_label=1)
print('ROC area Linear model:', metrics.auc(fpr, tpr))


# In[31]:


#Evaluate the confusion matrix
from sklearn.metrics import confusion_matrix

pred_test = np.argmax(p_test, axis=1)
print(confusion_matrix(y_test, pred_test))


# # Model 2: Add 2 dense layers

# In[14]:


# Second model - Deep nn model
print('dense 1 model...')
input_data = tf.keras.layers.Input(batch_shape=(None, 12), dtype='float32', name='Input_data')

dense1  = tf.keras.layers.Dense(256, activation='relu', name='Dense_layer_1')(input_data)
dense2  = tf.keras.layers.Dense(256, activation='relu', name='Dense_layer_2')(dense1)

output_data = tf.keras.layers.Dense(2, activation='softmax', name='Dense_output')(dense2)

# Model Architecture defined
model_dense1 = tf.keras.models.Model(inputs=input_data, outputs=output_data)
model_dense1.summary()



# In[15]:


# Select optimizer and compile model
model_dense1.compile(loss='sparse_categorical_crossentropy',  optimizer='Adam', metrics=['accuracy'])


# In[16]:


# Train the model
start = time.time()

history_dense1 = model_dense1.fit(X_train, y_train, batch_size=128, epochs=40, verbose=1, validation_data=(X_val, y_val))

print('Seconds to train: ', time.time() - start)


# In[17]:


p_test = model_dense1.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, p_test[:,1], pos_label=1)
print('ROC area dense 1:', metrics.auc(fpr, tpr))
#0.86116818072


# # Model 3: A more dense architecture

# In[18]:


input_data = tf.keras.layers.Input(batch_shape=(None, 12), dtype='float32', name='Input_data')

dense1  = tf.keras.layers.Dense(256, activation='relu')(input_data)
dense2  = tf.keras.layers.Dense(128, activation='relu')(dense1)
dense3  = tf.keras.layers.Dense(64, activation='relu')(dense2)
dense4  = tf.keras.layers.Dense(64, activation='relu')(dense3)
dense5  = tf.keras.layers.Dense(32, activation='relu')(dense4)

output_data = tf.keras.layers.Dense(2, activation='softmax', name='Dense_output')(dense4)

model_dense2 = tf.keras.models.Model(inputs=input_data, outputs=output_data)
model_dense2.summary()



# In[19]:


model_dense2.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[20]:


# Train the model
start = time.time()

history_dense2 = model_dense2.fit(X_train, y_train, batch_size=128, epochs=40,
                                  verbose=1, validation_data=(X_val, y_val))

print('Seconds to train: ', time.time() - start)


# In[21]:


# Evaluate ROC
p_test = model_dense2.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, p_test[:,1], pos_label=1)
print('ROC area dense 2:', metrics.auc(fpr, tpr))
#0.860878284242


# # Model 4: Add regularization and early stopping

# In[32]:


input_data = tf.keras.layers.Input(batch_shape=(None, 12), dtype='float32', name='Input_data')

dense1  = tf.keras.layers.Dense(256, activation='relu')(input_data)

dense2  = tf.keras.layers.Dense(128, activation='relu')(dense1)

dense3  = tf.keras.layers.Dense(64, activation='relu')(dense2)

dense4  = tf.keras.layers.Dense(64, activation='relu')(dense3)
dense4 = tf.keras.layers.Dropout(0.5)(dense4)

dense5  = tf.keras.layers.Dense(32, activation='relu')(dense4)
dense5 = tf.keras.layers.Dropout(0.5)(dense5)

output_data = tf.keras.layers.Dense(2, activation='sigmoid', name='Dense_output')(dense5)


model_dense3 = tf.keras.models.Model(inputs=input_data, outputs=output_data)
model_dense3.summary()




# In[33]:


model_dense3.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[34]:


# Train the model
start = time.time()

history_dense3 = model_dense3.fit(X_train, y_train, batch_size=128, epochs=100, verbose=1, validation_data=(X_val, y_val))

print('Seconds to train: ', time.time() - start)


# In[ ]:


p_test = model_dense3.predict(X_test)
fpr, tpr, thresholds = metrics.roc_curve(y_test, p_test[:,1], pos_label=1)
print('ROC area dense 3:', metrics.auc(fpr, tpr))


# 
# 
