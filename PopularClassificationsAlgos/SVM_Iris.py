import pandas as pd
import numpy as np
attributes=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset=pd.read_csv("irisdata.csv",names=attributes)
X=dataset.iloc[:,0:4].values
y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train, X_testVal, y_train, y_testVal = train_test_split(X, y, test_size = 0.40, random_state = 1)

X_val, X_test, y_val, y_test = train_test_split(X_testVal, y_testVal, test_size = 0.50, random_state = 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val)
X_test = sc.transform(X_test)


#Trying to tune logistic regression as per our need
#By changing regularization parameter
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred_val = classifier.predict(X_val)
from sklearn.metrics import confusion_matrix
print 'For validation set :'
print 'confusion matrix :'
cm = confusion_matrix(y_val, y_pred_val)
print cm
print 'f-score(weighted for test) is : '
from sklearn.metrics import f1_score
print f1_score(y_val,y_pred_val,average='weighted')

y_pred_test=classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred_test)
print 'confusion matrix :'
print cm
print 'f-score(weighted for test) is : '
print f1_score(y_test,y_pred_test,average='weighted')
