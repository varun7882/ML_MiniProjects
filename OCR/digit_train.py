import numpy as np
import cv2
img=cv2.imread("digits.png")
cv2.imshow("hey",img)
cv2.waitKey(10)
cv2.destroyAllWindows()
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print np.shape(gray)
train=[]
train_labels=[]
for i in range(0,1000,20):
    for j in range(0,2000,20):
        t1=gray[i:i+20,j:j+20]
        #cv2.imshow("hey",t1)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        train.append(t1.reshape(-1,400))
        train_labels.append(i%5)
knn=cv2.KNearest()
train=np.array(train)
train=train.astype(np.float32)
train_l=np.array(train_labels)
train_l=train_l.astype(np.float32)
knn.train(train,train_l)
test=cv2.imread("test1.jpg")
test=cv2.cvtColor(test,cv2.COLOR_BGR2GRAY)
ret,result,nb,dist=knn.find_nearest(test,k=5)
print result
        
