import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
train_num=50
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

m=50000

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

        tree_pr=[]
        # adaboost
        for i in range(0, num):
            # create dicision tree
            # ramdle and chose the first m
            indices = np.random.permutation(train_x.shape[0])
            ram_train_x = train_x[indices]
            x_m=ram_train_x[0:m]
            ram_train_y = train_y[indices]
            y_m=ram_train_y[0:m]
            #print(y_m)
            clf = tree.ExtraTreeClassifier(max_features="log2")
            clf = clf.fit(x_m, y_m)
            #accuracy = clf.score(train_x, train_y)
            #print(accuracy)
            #print(test_x.shape)
            pr = clf.predict(test_x)
            tree_pr.append(pr)

        #print(len(tree_pr))
        predict = np.zeros(test_x.shape[0])
        for i in range(len(tree_pr)):
            predict +=tree_pr[i]
        if len(tree_pr)!=0:
            predict=predict/len(tree_pr)
        for i in range(predict.size):
            if predict[i] >= 0.5:
                predict[i] = 1
            else:
                predict[i] = 0
        fpr, tpr, thresholds = metrics.roc_curve(test_y, predict)
        auc=metrics.auc(fpr, tpr)
        accuracy = accuracy_score(test_y, predict)
        accur_e+=accuracy
        auc_e+=auc

    accur_e=accur_e/5
    auc_e=auc_e/5
    print(num,'train sucessfully!')
    print("accuracy is ",accur_e," AUC is ",auc_e)
    auc_draw.append(auc_e)
plt.xlabel('the number of base leaner',fontsize=14)
plt.ylabel('AUC indicator',fontsize=14)
plt.title('RandomForest',fontsize=20)
plt.plot(range(0,train_num),auc_draw,linewidth=3)
plt.show()