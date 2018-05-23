import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
x = np.array([0, 1, 2, 3, 4])
y = np.array([1, 3, 2, 5, 7])
x2=np.array([5, 6, 7, 8, 9])
y2=np.array([8, 8, 9, 10, 12])
x=x.reshape((1,5))
y=y.reshape((1,5))
x2=x2.reshape((1,5))
y2=y2.reshape((1,5))
regr = linear_model.LinearRegression()
regr.fit(x,y)
y_pred=regr.predict(x2)
coef=regr.coef_
inter=regr.intercept_
print coef[0,0]
print inter[0]
print "=============================="
print y2
print y_pred[0]
plt.scatter(x, y, color = "m",marker = "o", s = 30)
plt.plot(x[0],y_pred[0])
plt.show()
