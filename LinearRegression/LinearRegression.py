#Linear Regression
#Gradient descent algorithm
import numpy
x=[int(i) for i in raw_input("Input x elements\n").split()]
y=[int(i) for i in raw_input("Input y elements\n").split()]
#print x
#print y
theta0=float(raw_input("Input theta0\n"))
theta1=float(raw_input("Input theta1\n"))
alpha=float(raw_input("learning rate\n"))
ptheta0=0.0
ptheta1=0.0
thd=.01
def sumterm0(x,y):
    print "s0\n"
    ans=0.0
    for i in range(len(x)):
        ans=ans+theta0+(theta1*x[i])-y[i]
    return ans

def sumterm1(x,y):
    print "s1\n"
    ans=0.0
    for i in range(len(x)):
        ans=ans+(theta0+(theta1*x[i])-y[i])*x[i]
    return ans
#print abs(ptheta0-theta0)
while(abs(ptheta0-theta0)>=thd and abs(ptheta1-theta1)>=thd):
    ptheta0=theta0
    ptheta1=theta1
    theta0=theta0-((alpha*sumterm0(x,y))//len(x))
    theta1=theta1-((alpha*sumterm1(x,y))//len(x))
    print "theta0 ",theta0
    print "theta1 ",theta1
print theta0
print theta1
