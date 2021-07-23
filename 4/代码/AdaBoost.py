import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
train_num=15
columns = ['Age','Workclass','fnlgwt','Education','EdNum','MaritalStatus',
           'Occupation','Relationship','Race','Sex','CapitalGain',
           'CapitalLoss','HoursPerWeek','Country','Income']
data=pd.read_csv('adult.data',names=columns)
#lable encoder
for col in columns:
    le = preprocessing.LabelEncoder()
    if(data[col].dtype=='object'):
        le.fit(data[col].values.tolist())
        data[col]=le.transform(data[col])

# random permutation
data_x=np.array(data.iloc[:,0:14])
indices = np.random.permutation(data_x.shape[0])
ram_data=data_x[indices]
data_x=ram_data
data_y=np.array(data.iloc[:,-1:])
ram_data=data_y[indices]
data_y=ram_data



# 5-fold cross validation
kf=KFold(n_splits=5)
auc_draw=[]
for num in range(train_num):
    accur_e=0
    auc_e=0

    for train_index,test_index in kf.split(data_x):
        train_x,train_y=data_x[train_index],data_y[train_index]
        test_x,test_y=data_x[test_index],data_y[test_index]
        train_x= np.squeeze(train_x)
        train_y = np.squeeze(train_y)
        test_x = np.squeeze(test_x)
        test_y = np.squeeze(test_y)
        D=np.ones(train_x.shape[0])
        D=D/train_x.shape[0]
        alpha=[]
        tree_pr=[]
        # adaboost
        for i in range(0, num):
            # create dicision tree
            clf = tree.DecisionTreeClassifier(max_depth=5)
            clf = clf.fit(train_x, train_y,sample_weight=D)
            pr = clf.predict(test_x)
            tree_pr.append(pr)
            #renew the parameters
            pr_y=clf.predict(train_x)
            accuracy = clf.score(train_x, train_y,sample_weight=D)
            er=1-accuracy
            if(er>0.5):
                break
            alpha_i=0.5*math.log((1-er)/er)
            #print(alpha_i)
            alpha.append(alpha_i)
            D=D*np.exp(-alpha_i*train_y*pr_y)
            D=D/np.sum(D)
        prdict=np.zeros(test_x.shape[0])
        for i in range(len(alpha)):
            prdict+=alpha[i]*tree_pr[i]
        for i in range(prdict.size):
            if prdict[i] >= 0.5:
                prdict[i] = 1
            else :
                prdict[i] = 0
        fpr, tpr, thresholds = metrics.roc_curve(test_y, prdict)
        auc=metrics.auc(fpr, tpr)
        accuracy = accuracy_score(test_y, prdict)
        accur_e+=accuracy
        auc_e+=auc
    accur_e=accur_e/5
    auc_e=auc_e/5
    print(num,'train sucessfully!')
    print("accuracy is ",accur_e," AUC is ",auc_e)
    auc_draw.append(auc_e)
plt.xlabel('the number of base leaner',fontsize=14)
plt.ylabel('AUC indicator',fontsize=14)
plt.title('AdaBoost',fontsize=20)
plt.plot(range(0,train_num),auc_draw,linewidth=3)
plt.show()