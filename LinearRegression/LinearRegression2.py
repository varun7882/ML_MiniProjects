#Linear Regression
#Gradient descent algorithm
def costFunc(x,y,t0,t1):
    ans=0.0
    for i in range(len(x)):
        ans=ans+(t0+(t1*x[i])-y[i])**2
        print ans
    ans=ans/(2.0*len(x))
    return ans
import numpy as np
from matplotlib import pyplot as plt
#x=[int(i) for i in raw_input("Input x elements\n").split()]
#y=[int(i) for i in raw_input("Input y elements\n").split()]
#x=np.array(x)
#y=np.array(y)
#print x
#print y
x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])
print np.corrcoef(x,y)
theta1=np.corrcoef(x,y)[0,1]*(np.std(y)/np.std(x))
print np.cov(x,y)[0,1]/np.var(x)," ",theta1
theta0=np.mean(y)-(theta1*np.mean(x))
print theta0
print theta1
plt.scatter(x, y, color = "m",marker = "o", s = 30)
y_pred=theta0+(theta1*x)
plt.plot(x,y_pred)
plt.show()
