#importing all the libraries and dataset
import pandas as pd
from sklearn.neural_network import MLPClassifier

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

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(3), random_state=1)
clf.fit(X_train, y_train)
y_pred_val=clf.predict(X_val)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred_val)
print 'confusion matrix :'
print cm
from sklearn.metrics import f1_score
print 'f-score(weighted) is : '
print f1_score(y_val,y_pred_val,average='weighted')

y_pred_test=clf.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_test)
print 'confusion matrix :'
print cm
from sklearn.metrics import f1_score
print 'f-score(weighted) is : '
print f1_score(y_test,y_pred_test,average='weighted')
