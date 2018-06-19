# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 13:49:49 2018

@author: VaSrivastava
"""

import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
attributes=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset=pd.read_csv('irisdata.csv',names=attributes) 
X=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4].values
y_label=LabelEncoder()
ylab=y_label.fit_transform(y)
y=np_utils.to_categorical(ylab)
from sklearn.model_selection import train_test_split
X_train, X_testVal, y_train, y_testVal = train_test_split(X, y, test_size = 0.33, random_state = 1)

#X_val, X_test, y_val, y_test = train_test_split(X_testVal, y_testVal, test_size = 0.50, random_state = 1)

model =Sequential()
model.add(Dense(7,input_dim=4,activation='relu',init='uniform'))
model.add(Dense(3,activation='softmax',init='uniform'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(X_train,y_train,epochs=500,batch_size=20)

y_pred=model.predict(X_testVal)
y_pred = (y_pred > 0.5)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_testVal, y_pred)
print ('confusion matrix :')
print (cm)
print ('f-score(weighted for test) is : ')
print (f1_score(y_testVal,y_pred,average='weighted'))
#print("\nAccuracy: %.2f%%" % (scores[1]*100))


