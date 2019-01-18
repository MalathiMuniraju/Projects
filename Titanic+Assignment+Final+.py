
# coding: utf-8

# In[39]:


#Import Packages
#Linear Algebra
import numpy as np
#Data Processing
import pandas as pd
import sklearn
#Data Visualization
import seaborn as sns
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve,auc,recall_score,precision_score,accuracy_score,f1_score
from sklearn.metrics import confusion_matrix,average_precision_score,recall_score
from sklearn import svm
#Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import AdaBoostClassifier


# In[40]:


#To print many statements at the same time using below command
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity='all'


# In[41]:


# Import datset and create a dataframe
df=pd.read_csv("Titanic_Raw.csv")
df.head()
df.tail()


# In[42]:


df.info()


# In[43]:


df.describe()


# In[44]:


#Missing Value Count
total=df.isnull().sum().sort_values(ascending=False)
per=df.isnull().sum()/df.isnull().count()*100
percentage=(round(per,2)).sort_values(ascending=False)
missing_data=pd.concat([total,percentage],axis=1,keys=['Total','%'])
missing_data


# In[45]:


#Replacing missing age value
median=df.groupby(['pclass'])['age'].median()
median
df=df.set_index(['pclass'])
df['age']=df['age'].fillna(median)
df=df.reset_index()
pd.isnull(df["age"]).any()
pd.isnull(df["age"]).sum()


# In[46]:


#Replacing missing value of embarkment 
common_value='S'
df['embarked']=df['embarked'].fillna(common_value)
pd.isnull(df['embarked']).any()
pd.isnull(df['embarked']).sum()


# In[47]:


#Replacing missing fare value
median_fare =df.groupby(['pclass'])['fare'].median()
median_fare
df=df.set_index(['pclass'])
df['fare']=df['fare'].fillna(median)
df=df.reset_index()
pd.isnull(df["fare"]).any()
pd.isnull(df["fare"]).sum()


# In[48]:


#Univariate Analysis
pd.crosstab(index=df.pclass,columns='Frequency')
sns.countplot(x=df.pclass)


# In[49]:


pd.crosstab(index=df.survived,columns='Frequency')
sns.countplot(x=df.survived)


# In[50]:


pd.crosstab(index=df.sex,columns='Frequency')
sns.countplot(x=df.sex)


# In[51]:


sns.swarmplot(x=df.age)


# In[52]:


pd.crosstab(index=df.sibsp,columns='Frequency')
sns.countplot(x=df.sibsp)


# In[53]:


pd.crosstab(index=df.parch,columns='Frequency')
sns.countplot(x=df.parch)


# In[54]:


sns.boxplot(x=df.fare)


# In[55]:


pd.crosstab(index=df.embarked,columns='Frequency')
sns.countplot(x=df.embarked)


# In[56]:


#Multivariate Analysis
survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))
women = df[df['sex']=='female']
men = df[df['sex']=='male']
ax = sns.distplot(women[women['survived']==1].age, bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['survived']==0].age, bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['survived']==1].age, bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['survived']==0].age, bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')


# In[57]:


pd.crosstab(index=df.pclass,columns=df.survived)
sns.countplot(x=df.pclass,hue=df.survived)


# In[58]:


c=pd.pivot_table(df,index=(df.embarked,df.pclass),columns=df.survived,aggfunc='size')
c
c.plot.bar(stacked=False)


# In[59]:


FacetGrid = sns.FacetGrid(df, row='embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'pclass', 'survived', 'sex', palette=None,  order=None, hue_order=None )
FacetGrid.add_legend()


# In[60]:


grid = sns.FacetGrid(df, col='survived', row='pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'age', alpha=0.5, bins=20)
grid.add_legend();


# In[61]:


#Extracting new feature from the variable cabin
import re
deck={"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7,"T":8,"U":9}
data=[df]

for dataset in data:
    dataset['cabin']=dataset['cabin'].fillna("U0")
    dataset['deck']=dataset['cabin'].map(lambda x:re.compile("([a-zA-Z]+)").search(x).group())
    dataset['deck']=dataset['deck'].map(deck)
    dataset['deck']=dataset['deck'].fillna(0)
    dataset['deck']=dataset['deck'].astype(int)


# In[62]:


df.deck.value_counts().sort_index()


# In[63]:


a=pd.crosstab(index=df.deck,columns=df.survived);a
a.plot.bar(stacked=False)


# In[64]:


#Converting fare from floating vaariable into integer
data=[df]
for dataset in data:
    dataset['fare']= dataset['fare'].astype(int)


# In[65]:


#Converting categorical variable in to numeric 
genders = {"male": 0, "female": 1}
data = [df]

for dataset in data:
    dataset['sex'] = dataset['sex'].map(genders)


# In[66]:


ports = {"S": 0, "C": 1, "Q": 2}
data = [df]

for dataset in data:
    dataset['embarked'] = dataset['embarked'].map(ports)


# In[67]:


data=[df]
for dataset in data:
    dataset['age']=dataset['age'].astype(int)
    dataset.loc[dataset['age']<=11,'age'] = 0
    dataset.loc[(dataset['age']>11) & (dataset['age']<=18),'age']=1
    dataset.loc[(dataset['age']>18) & (dataset['age']<=22),'age']=2
    dataset.loc[(dataset['age']>22) & (dataset['age']<= 27),'age']=3
    dataset.loc[(dataset['age']>27) & (dataset['age']<= 33),'age']=4
    dataset.loc[(dataset['age']>33) & (dataset['age']<= 40),'age']=5
    dataset.loc[(dataset['age']>40) & (dataset['age']<= 66),'age']=6
    dataset.loc[dataset['age']>66,'age']=7


# In[68]:


data=[df]
for dataset in data:
    dataset.loc[dataset['fare']<=7.91, 'fare']=0
    dataset.loc[(dataset['fare']>7.91) & (dataset['fare']<=14.454),'fare']=1
    dataset.loc[(dataset['fare']>14.454) & (dataset['fare']<=31),'fare']=2
    dataset.loc[(dataset['fare']>31) & (dataset['fare']<=99),'fare']=3
    dataset.loc[(dataset['fare']>99) & (dataset['fare']<=250),'fare']=4
    dataset.loc[dataset['fare']>250,'fare']=5
    dataset['fare']=dataset['fare'].astype(int)


# In[69]:


#Dropping variabels from the dataframe
df=df.drop(['name','ticket','boat','body','home.dest','cabin'],axis=1)
df.info()


# In[70]:


df.describe()


# In[71]:


df.head()


# In[72]:


a=df.corr()
a


# In[38]:


sns.heatmap(a)


# In[73]:


#Creating independent and dependent variable
x=df.drop('survived',axis=1)
y=df['survived']
x.head()
x.shape
y.head()
y.shape


# In[74]:


#Creating train and test data 75% and 25% split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.15,random_state=1)
train_x.shape
test_x.shape
train_y.shape
test_y.shape


# In[75]:


test_y.value_counts()


# # LOGISTIC REGRESSION

# In[76]:


#Build a Logistic regression model
from sklearn.linear_model import LogisticRegression
log=LogisticRegression()


# In[77]:


log.fit(train_x,train_y)


# In[78]:


log.coef_


# In[79]:


#Generate Model Diagnostics
classes=log.predict(test_x)
print(classes.size)
print('Positive Cases in Test Data:',test_y[test_y==1].shape[0])
print('Negative Cases in Test Data:',test_y[test_y==0].shape[0])


# In[80]:


acc_log = round(log.score(test_x, test_y)*100,2)
acc_log


# In[81]:


#Precision and Recall
print("Accuracy Score")
print(metrics.accuracy_score(test_y,classes))
print("Precision/Recall Metrics")
print(metrics.classification_report(test_y,classes))
print("AUC")
auc_log=metrics.roc_auc_score(test_y,classes)
auc_log


# In[82]:


prec_log=round(precision_score(test_y,classes),2)
prec_log
recall_log=round(recall_score(test_y,classes),2)
recall_log


# In[83]:


#ROC Chart
fpr,tpr,th=roc_curve(test_y,classes)
roc_auc=metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('ROCR CHART')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[84]:


#Confusion Matrix
print("Confusion Matrix")
cf=metrics.confusion_matrix(test_y,classes)
lbl1=["Predicted 0","Predicted 1"]
lbl2=["True 0","True 1"]
sns.heatmap(cf,annot=True,cmap="Greens",fmt="d",xticklabels=lbl1,yticklabels=lbl2)
plt.show();


# In[85]:


#Find out key predictor of Defaulter
coeff=pd.concat([pd.DataFrame(x.columns),pd.DataFrame(np.transpose(log.coef_))],axis=1)
coeff.columns=("Variable","Coeff")
coeff.sort_index(ascending=True)


# In[86]:


#Feature Importance
coeff.plot(kind='barh')
plt.show();


# # SUPPORT VECTOR CLASSIFIER

# In[87]:


#Build the model
linear_svc = LinearSVC()


# In[88]:


#Fitting Model
linear_svc.fit(train_x,train_y)


# In[89]:


#Applying our learnt model on test data
y_pred_test = linear_svc.predict(test_x)
y_pred_test


# In[90]:


#Generating accuracy score
metrics.accuracy_score(test_y,y_pred_test)


# In[91]:


acc_svc = round(linear_svc.score(test_x, test_y)*100,2)
acc_svc


# In[92]:


auc_svc=metrics.roc_auc_score(test_y,y_pred_test)
auc_svc
prec_svc=round(precision_score(test_y,y_pred_test),2)
prec_svc
recall_svc=round(recall_score(test_y,y_pred_test),2)
recall_svc


# In[93]:


#Create Confusion Matrix
conf = metrics.confusion_matrix(test_y,y_pred_test)
conf


# In[94]:


#Plotting confusion matrix
cmap=sns.cubehelix_palette(50, hue=0.05, rot=0,light=0.9,dark=0,as_cmap=True)
sns.heatmap(conf,cmap=cmap,xticklabels=['0','1'],yticklabels=['0','1'],annot=True,fmt="d")
plt.show();


# In[95]:


#ROC Chart
fpr,tpr,th=roc_curve(test_y,y_pred_test)
roc_auc=metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('ROCR CHART')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[96]:


#Find out key predictor of Defaulter
coeff=pd.concat([pd.DataFrame(x.columns),pd.DataFrame(np.transpose(linear_svc.coef_))],axis=1)
coeff.columns=("Variable","Coeff")
coeff.sort_index(ascending=True)


# In[97]:


coeff.plot(kind='barh')
plt.show();


# # Stochastic Gradient Descent (SGD)

# In[98]:


#Build the model
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)


# In[99]:


#Fit the model
sgd.fit(train_x, train_y)


# In[100]:


sgd.coef_


# In[101]:


#Find out key predictor of Defaulter
coeff=pd.concat([pd.DataFrame(x.columns),pd.DataFrame(np.transpose(sgd.coef_))],axis=1)
coeff.columns=("Variable","Coeff")
coeff.sort_values("Variable",ascending=True)


# In[102]:


sgd.intercept_


# In[103]:


#Generating Model Diagnostics
y_pred = sgd.predict(test_x)


# In[104]:


#Generating Accuracy Score
acc_sgd = round(sgd.score(test_x, test_y)*100,2)
acc_sgd


# In[105]:


acc_sgd = round(sgd.score(train_x, train_y) * 100, 2)
acc_sgd


# In[106]:


#Precision and Recall
print("Accuracy Score")
print(metrics.accuracy_score(test_y,y_pred))
print("Precision/Recall Metrics")
print(metrics.classification_report(test_y,y_pred))
print("AUC")
auc_sgd=metrics.roc_auc_score(test_y,y_pred)
auc_sgd


# In[107]:


prec_sgd=round(precision_score(test_y,y_pred),2)
prec_sgd
recall_sgd=round(recall_score(test_y,y_pred),2)
recall_sgd


# In[108]:


#ROC Chart
fpr,tpr,th=roc_curve(test_y,y_pred)
roc_auc=metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('ROCR CHART')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[109]:


#Find out key predictor of Defaulter
coeff=pd.concat([pd.DataFrame(x.columns),pd.DataFrame(np.transpose(sgd.coef_))],axis=1)
coeff.columns=("Variable","Coeff")
coeff.sort_index(ascending=True)


# In[110]:


coeff.plot(kind='barh')
plt.show();


# In[111]:


#Confusion Matrix
print("Confusion Matrix")
cf=metrics.confusion_matrix(test_y,y_pred)
lbl1=["Predicted 0","Predicted 1"]
lbl2=["True 0","True 1"]
sns.heatmap(cf,annot=True,cmap="Greens",fmt="d",xticklabels=lbl1,yticklabels=lbl2)
plt.show();


# In[112]:


sgd.coef_


# # K Nearest Neighbor

# In[113]:


#Build a model
knn = KNeighborsClassifier(n_neighbors = 3)


# In[114]:


#Fit a model
knn.fit(train_x,train_y) 


# In[115]:


knn.get_params


# In[116]:


#Model Diagnosis
knn_pred = knn.predict(test_x)


# In[117]:


#Genearting Accuracy Score
acc_knn=round(knn.score(test_x,test_y)*100,2)
acc_knn


# In[118]:


#Precision and Recall
print("Accuracy Score")
print(metrics.accuracy_score(test_y,knn_pred))
print("Precision/Recall Metrics")
print(metrics.classification_report(test_y,knn_pred))
print("AUC")
auc_knn=metrics.roc_auc_score(test_y,knn_pred)
auc_knn


# In[119]:


prec_knn=round(precision_score(test_y,knn_pred),2)
prec_knn
recall_knn=round(recall_score(test_y,knn_pred),2)
recall_knn


# In[120]:


#ROC Chart
fpr,tpr,th=roc_curve(test_y,knn_pred)
roc_auc=metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('ROCR CHART')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[121]:


#Confusion Matrix
print("Confusion Matrix")
cf=metrics.confusion_matrix(test_y,knn_pred)
lbl1=["Predicted 0","Predicted 1"]
lbl2=["True 0","True 1"]
sns.heatmap(cf,annot=True,cmap="Greens",fmt="d",xticklabels=lbl1,yticklabels=lbl2)
plt.show();


# # Gaussian Naive Bayes

# In[122]:


#Build a model
gaussian = GaussianNB()


# In[123]:


#Fit a model
gaussian.fit(train_x,train_y)


# In[124]:


#Model Diagnosis
gaus_pred=gaussian.predict(test_x)


# In[125]:


#Generating accuarcy score
acc_gaussian=round(gaussian.score(test_x,test_y)*100,2)
acc_gaussian


# In[126]:


#Precision and Recall
print("Accuracy Score")
print(metrics.accuracy_score(test_y,gaus_pred))
print("Precision/Recall Metrics")
print(metrics.classification_report(test_y,gaus_pred))
print("AUC")
auc_gaussian=metrics.roc_auc_score(test_y,gaus_pred)
auc_gaussian


# In[127]:


prec_gaussian=round(precision_score(test_y,gaus_pred),2)
prec_gaussian
recall_gaussian=round(recall_score(test_y,gaus_pred),2)
recall_gaussian


# In[128]:


#ROC Chart
fpr,tpr,th=roc_curve(test_y,gaus_pred)
roc_auc=metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('ROCR CHART')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[129]:


#Confusion Matrix
print("Confusion Matrix")
cf=metrics.confusion_matrix(test_y,gaus_pred)
lbl1=["Predicted 0","Predicted 1"]
lbl2=["True 0","True 1"]
sns.heatmap(cf,annot=True,cmap="Greens",fmt="d",xticklabels=lbl1,yticklabels=lbl2)
plt.show();


# # DECISION TREE

# In[130]:


#Build a model
decision_tree = DecisionTreeClassifier()


# In[131]:


#Fit the model
decision_tree.fit(train_x,train_y)


# In[132]:


#Predict on test data
decision_pred= decision_tree.predict(test_x)


# In[133]:


#Generating accuracy score
acc_decision_tree=round(decision_tree.score(test_x,test_y)*100,2)
acc_decision_tree


# In[134]:


#Precision and Recall
print("Accuracy Score")
print(metrics.accuracy_score(test_y,decision_pred))
print("Precision/Recall Metrics")
print(metrics.classification_report(test_y,decision_pred))
print("AUC")
auc_decision=metrics.roc_auc_score(test_y,decision_pred)
auc_decision


# In[135]:


prec_decision=round(precision_score(test_y,decision_pred),2)
prec_decision
recall_decision=round(recall_score(test_y,decision_pred),2)
recall_decision


# In[136]:


#ROC Chart
fpr,tpr,th=roc_curve(test_y,decision_pred)
roc_auc=metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('ROCR CHART')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[137]:


#Confusion Matrix
print("Confusion Matrix")
cf=metrics.confusion_matrix(test_y,decision_pred)
lbl1=["Predicted 0","Predicted 1"]
lbl2=["True 0","True 1"]
sns.heatmap(cf,annot=True,cmap="Greens",fmt="d",xticklabels=lbl1,yticklabels=lbl2)
plt.show();


# In[138]:


decision_tree.set_params


# # RANDOM FOREST

# In[139]:


# Build a model
random_forest = RandomForestClassifier(n_estimators=100)


# In[140]:


#Fit a model
random_forest.fit(train_x,train_y)


# In[141]:


#Predicting on test data
random_pred= random_forest.predict(test_x)


# In[142]:


#Generating accuracy score
acc_random_forest=round(random_forest.score(test_x,test_y)*100,2)
acc_random_forest


# In[143]:


random_forest.score(train_x, train_y)


# In[144]:


random_forest.score(test_x, test_y)


# In[145]:


#Precision and Recall
print("Accuracy Score")
print(metrics.accuracy_score(test_y,random_pred))
print("Precision/Recall Metrics")
print(metrics.classification_report(test_y,random_pred))
print("AUC")
auc_random=metrics.roc_auc_score(test_y,random_pred)
auc_random


# In[146]:


prec_random=round(precision_score(test_y,random_pred),2)
prec_random
recall_random=round(recall_score(test_y,random_pred),2)
recall_random


# In[147]:


#ROC Chart
fpr,tpr,th=roc_curve(test_y,random_pred)
roc_auc=metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('ROCR CHART')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[148]:


#Confusion Matrix
print("Confusion Matrix")
cf=metrics.confusion_matrix(test_y,random_pred)
lbl1=["Predicted 0","Predicted 1"]
lbl2=["True 0","True 1"]
sns.heatmap(cf,annot=True,cmap="Greens",fmt="d",xticklabels=lbl1,yticklabels=lbl2)
plt.show();


# # PERCEPTRON

# In[149]:


#Build a model
perceptron = Perceptron(max_iter=5)


# In[150]:


#Fit a model
perceptron.fit(train_x,train_y)


# In[151]:


#Predicting on test data
percep_pred = perceptron.predict(test_x)


# In[152]:


perceptron.score(train_x,train_y)


# In[153]:


#Generating accuracy score
acc_perceptron=round(perceptron.score(test_x,test_y)*100,2)
acc_perceptron


# In[154]:


#Precision and Recall
print("Accuracy Score")
print(metrics.accuracy_score(test_y,percep_pred))
print("Precision/Recall Metrics")
print(metrics.classification_report(test_y,percep_pred))
print("AUC")
auc_perceptron=metrics.roc_auc_score(test_y,percep_pred)
auc_perceptron


# In[155]:


prec_perceptron=round(precision_score(test_y,percep_pred),2)
recall_perceptron=round(recall_score(test_y,percep_pred),2)


# In[156]:


#ROC Chart
fpr,tpr,th=roc_curve(test_y,percep_pred)
roc_auc=metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('ROCR CHART')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[157]:


#Confusion Matrix
print("Confusion Matrix")
cf=metrics.confusion_matrix(test_y,percep_pred)
lbl1=["Predicted 0","Predicted 1"]
lbl2=["True 0","True 1"]
sns.heatmap(cf,annot=True,cmap="Greens",fmt="d",xticklabels=lbl1,yticklabels=lbl2)
plt.show();


# # Linear Discriminant Analysis

# In[158]:


#Build a model
lda=LinearDiscriminantAnalysis()


# In[159]:


#Fit a model
lda.fit(train_x,train_y)


# In[160]:


#Predicting on test data
lda_predict=lda.predict(test_x)


# In[161]:


#Generating accuracy score
acc_lda=round(lda.score(test_x,test_y)*100,2)
acc_lda


# In[162]:


#Precision and Recall
print("Accuracy Score")
print(metrics.accuracy_score(test_y,lda_predict))
print("Precision/Recall Metrics")
print(metrics.classification_report(test_y,lda_predict))
print("AUC")
auc_lda=metrics.roc_auc_score(test_y,lda_predict)
auc_lda


# In[163]:


prec_lda=round(precision_score(test_y,lda_predict),2)
prec_lda
recall_lda=round(recall_score(test_y,lda_predict),2)
recall_lda


# In[164]:


#ROC Chart
fpr,tpr,th=roc_curve(test_y,lda_predict)
roc_auc=metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('ROCR CHART')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[165]:


#Confusion Matrix
print("Confusion Matrix")
cf=metrics.confusion_matrix(test_y,percep_pred)
lbl1=["Predicted 0","Predicted 1"]
lbl2=["True 0","True 1"]
sns.heatmap(cf,annot=True,cmap="Greens",fmt="d",xticklabels=lbl1,yticklabels=lbl2)
plt.show();


# # AdaBoost Classifier

# In[166]:


#Build a model

ada=AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))


# In[167]:


#Fit a model
ada.fit(train_x,train_y)


# In[168]:


#Predict on the test data
ada_predict=ada.predict(test_x)


# In[169]:


#Generating accuracy score
acc_ada=round(ada.score(test_x,test_y)*100,2)
acc_ada


# In[170]:


#Precision and Recall
print("Accuracy Score")
print(metrics.accuracy_score(test_y,ada_predict))
print("Precision/Recall Metrics")
print(metrics.classification_report(test_y,ada_predict))
print("AUC")
auc_ada=metrics.roc_auc_score(test_y,ada_predict)
auc_ada


# In[171]:


prec_ada=round(precision_score(test_y,ada_predict),2)
prec_ada
recall_ada=round(recall_score(test_y,ada_predict),2)
recall_ada


# In[172]:


#ROC Chart
fpr,tpr,th=roc_curve(test_y,ada_predict)
roc_auc=metrics.auc(fpr,tpr)
import matplotlib.pyplot as plt
plt.title('ROCR CHART')
plt.plot(fpr,tpr,'b',label='AUC=%0.2f'% roc_auc)
plt.legend(loc="lower right")
plt.plot([0,1],[0,1],'o--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show();


# In[173]:


#Confusion Matrix
print("Confusion Matrix")
cf=metrics.confusion_matrix(test_y,ada_predict)
lbl1=["Predicted 0","Predicted 1"]
lbl2=["True 0","True 1"]
sns.heatmap(cf,annot=True,cmap="Greens",fmt="d",xticklabels=lbl1,yticklabels=lbl2)
plt.show();


# # Results

# In[174]:


results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 
              'Decision Tree','Linear Discriminant Analysis','AdaBoost Classifier'],
    'Accuracy': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_decision_tree, acc_lda,acc_ada],
    'AUC':[auc_svc,auc_knn,auc_log,auc_random,auc_gaussian,
                 auc_perceptron,auc_sgd,auc_decision, auc_lda,auc_ada],
    'Precision': [prec_svc,prec_knn,prec_log,prec_random,prec_gaussian,
                  prec_perceptron,prec_sgd,prec_decision,prec_lda,prec_ada],
    'Recall':[recall_svc,recall_knn,recall_log,recall_random,recall_gaussian,
              recall_perceptron,recall_sgd,recall_decision,recall_lda,recall_ada]})
result_df = results.sort_values(by='AUC', ascending=False)
result_df = result_df.set_index('Model')
result_df.head(15)

