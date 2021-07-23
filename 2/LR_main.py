import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
def sigmoid(x):
    res=1/(1+np.exp(-x))
    #print("sigmod sucess")
    return res

def initialize(d): #d is the diamension
    W=np.zeros(d)
    b=0
    return W,b

def logistic(X,y,W,b):
    num_train=X.shape[0]
    #num_feature=X.shape[1]
    temp=np.dot(X,W)+b
    z=sigmoid(temp)
    #cost = -1/num_train * np.sum(y*np.log(z) + (1-y)*np.log(1-z))  #loss function
    dW = np.dot(X.T, (z-y))/num_train 
    db = np.sum(z-y)/num_train
    #cost = np.squeeze(cost)
    return dW,db

def logistic_train(X,y,test,learning_rate,epochs):
    W,b=initialize(X.shape[1])
    
    for i in range(epochs):
        dW,db=logistic(X,y,W,b)
        W=W-learning_rate*dW
        b=b-learning_rate*db 
    # save the result
    params = { 'W': W, 'b': b }
    res=predict(test,params)
    return res

def predict(X, params): 
    y_prediction = sigmoid(np.dot(X, params['W']) + params['b']) 
    print("successfully predict!")
    return y_prediction

def OvR(X,y,test,rate,epochs):
    #get the result answer
    res=[]
    for i in range(26):
        tmp_res=[]
        y_i=[]
        for j in range(y.shape[0]):
            if(y[j]==i+1):
                y_i.append(1)
            else :
                y_i.append(0)
        npyi=np.array(y_i)
        tmp_res=logistic_train(X,npyi,test,rate,epochs)
        res.append(tmp_res)
        print("sucessfully done class %d" %(i+1))
    return res

    

def clean(res):
    y=[]
    for i in range(len(res[0])):
        temp = np.array(res)
        temp=temp[...,i]
        temp=temp.tolist()
        index=temp.index(max(temp))
        y.append(index+1)
    return y

def performance(yres,testy):
    FP=[]
    FN=[]
    TP=[]
    TN=[]
    num=26
    for i in range(26):       
        fp=0
        fn=0
        tp=0
        tn=0
        for j in range(len(yres)):
            if yres[j]==i+1 and testy[j]==i+1 :
                tp=tp+1
            elif yres[j]==i+1 and testy[j]!=i+1 :
                fp=fp+1
            elif yres[j]!=i+1 and testy[j]!=i+1 :
                tn=tn+1
            else :
                fn=fn+1
        FP.append(fp)
        FN.append(fn)
        TP.append(tp)
        TN.append(tn)
    FP=np.array(FP)
    FN=np.array(FN)
    TP=np.array(TP)
    TN=np.array(TN)
    P=np.true_divide(TP,TP+FP)
    R=np.true_divide(TP,TP+FN)
    F1=np.true_divide(2*np.multiply(P,R),P+R)
    macro_P=np.sum(P)/num
    macro_R=np.sum(R)/num
    macro_F1=np.sum(F1)/num
    micro_P=np.sum(TP)/(np.sum(TP)+np.sum(FP))
    micro_R=np.sum(TP)/(np.sum(TP)+np.sum(FN))
    micro_F1=micro_P*micro_R*2/(micro_R+micro_P)
    return macro_P,macro_R,macro_F1,micro_P,micro_R,micro_F1
# params
rate=0.088
epochs=10000

ALL = np.loadtxt('train_set.csv', dtype='double',delimiter=',',skiprows=1)
test= np.loadtxt('test_set.csv', dtype='double',delimiter=',',skiprows=1)
testX=test[...,0:test.shape[1]-1]
testy=test[...,test.shape[1]-1] 
# get the data from the scv
# get the X and the y
X=ALL[...,0:ALL.shape[1]-1]
y=ALL[...,ALL.shape[1]-1]  
res=OvR(X,y,testX,rate,epochs)
yres=clean(res)
rescmp=[testy,yres]
pres = pd.DataFrame.from_records(rescmp)
pres.to_csv("res.csv")
print("write in csv sucessfully!")
num=0
for i in range(len(yres)):
    if yres[i]==testy[i]:
        num=num+1
accuracy=num/len(yres)
print("accuracy is ",accuracy)
# start culculating F1 P R
macro_P,macro_R,macro_F1,micro_P,micro_R,micro_F1=performance(yres,testy)
print("micro_P is ",micro_P)
print("micro_R is ",micro_R)
print("micro_F1 is ",micro_F1)
print("macro_P is ",macro_P)
print("macro_R is ",macro_R)
print("macro_F1 is ",macro_F1)


