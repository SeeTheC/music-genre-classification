
# coding: utf-8

# In[1]:

import pandas;
import numpy as np;
import ast;
import sklearn.tree as tree;
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier


# In[3]:

def readCSVFile(file):
    data=pandas.read_csv(file,",",header=0, na_values='?', skipinitialspace=True);
    return data;
    pass;
def readTrainData(dataset):    
    return dataset.ix[:,6:], dataset.ix[:,4:5].astype(int),dataset.ix[:,5:6];
    pass;

def readTestData(dataset):    
    return dataset.ix[:,6:], dataset.ix[:,4:5].astype(int),dataset.ix[:,5:6];
    pass;

def normalizePhi(unNormalizedPhi,last_col_bias=False):    
    #assuming last column as bias column
    no_of_column=len(unNormalizedPhi[0]);
    phi=np.array(unNormalizedPhi);
    std=phi.std(0);
    mean=phi.mean(0);    
    #std[no_of_column-1]=1;
    #mean[no_of_column-1]=0;
    #phi_normalize=(phi-mean)/std;    
    
    max_vector=phi.max(axis=0)
    phi_normalize=phi/max_vector;    
    
    return phi_normalize;
    pass;

def categoryToNumber(dataset,categoryList):
    for c in categoryList:
        if (c in dataset):            
            dataset[c]=pandas.get_dummies(dataset[c]).values.argmax(1);        
    return dataset;
    pass;
    

def handleCategoryData(dataset,categoryList):    
        return categoryToNumber(dataset,categoryList)


# In[5]:

dir="data/"
trainFile=dir+"train.csv";
testFile=dir+"test.csv";
trained_dataset=readCSVFile(trainFile);
test_dataset=readCSVFile(testFile);
trained_data,trained_y,trained_y_vector=readTrainData(trained_dataset);
test_data,test_y,test_y_vector=readTestData(test_dataset);

mtx_train =trained_data.as_matrix(columns=None)
mtx_train_y  =trained_y.as_matrix(columns=None)
mtx_trained_y_vector=trained_y_vector.as_matrix(columns=None);

mtx_train_norm=normalizePhi(mtx_train);
mtx_train_y=np.array(list((e[0] for e in mtx_train_y)));
mtx_trained_y_vector=np.array(list((ast.literal_eval(e[0]) for e in mtx_trained_y_vector)));

mtx_test=test_data.as_matrix(columns=None);
mtx_test_y=test_y.as_matrix(columns=None);
mtx_test_y_vector=test_y_vector.as_matrix(columns=None);

mtx_test_norm=normalizePhi(mtx_test);
mtx_test_y=np.array(list((e[0] for e in mtx_test_y)));
mtx_test_y_vector=np.array(list((ast.literal_eval(e[0]) for e in mtx_test_y_vector)));


# In[186]:

#trainedModel=OneVsRestClassifier(LinearSVC()).fit(mtx_train, mtx_train_y);
trainedModel=LinearSVC().fit(mtx_train, mtx_train_y);
y_predict=trainedModel.predict(mtx_train);
print(y_predict);
print("done");


# In[51]:

y_predict=trainedModel.predict(mtx_test);
print(y_predict);


# In[6]:

x = trained_data
y = np.ravel(trained_y)
y_test=np.ravel(test_y);

index = ["Decision Tree", "Random Forest", "K-Neighbors", "Gradient Boosting",
         "Logistic Regression", "Support Vector", "Bernoulli NB", "Gaussian NB","Adaboost"]
columns = ["Trained Misclassified Points", "Trained Accuracy", "Test Misclassified Points", "Test Accuracy",]
modelComparision = pandas.DataFrame(index=index, columns=columns)

dtc = 0, DecisionTreeClassifier(min_samples_split=20)
rfc = 1, RandomForestClassifier(min_samples_split=10)
knn = 2, KNeighborsClassifier()
gbc = 3, GradientBoostingClassifier(max_depth=3)
lgr = 4, LogisticRegression(max_iter=300)
svc = 5, LinearSVC(max_iter=4000)
bnb = 6, BernoulliNB()
gnb = 7, GaussianNB()
ada = 8, AdaBoostClassifier()

tr_size=len(x.index);
tt_size=len(y_test);
for i in lgr, gnb, bnb, dtc, svc, knn, gbc, ada, rfc:    
    model=i[1].fit(x, y);
    y_pred = model.predict(x)
    tr_misclassifedPoints = (y_pred != y).sum()  
    tr_accuracy = ((tr_size - tr_misclassifedPoints)*100) / tr_size;
    
    y_pred = model.predict(test_data)
    tt_misclassifedPoints = (y_pred != y_test).sum()  
    tt_accuracy = ((tt_size - tt_misclassifedPoints)*100) / tt_size;
    if(i[0]==0):
        tree.export_graphviz(model,out_file='tree.dot');        
    #print(misclassifedPoints,":",size - misclassifedPoints,':',accuracy);   
    modelComparision.ix[i[0]] = [tr_misclassifedPoints, tr_accuracy, tt_misclassifedPoints, tt_accuracy,];
    
print("Done");
modelComparision


# In[ ]:




# In[25]:

#Neural network
x=mtx_train_norm;
y=mtx_trained_y_vector;
model = MLPClassifier(solver='sgd', alpha=1e-7, activation="logistic",hidden_layer_sizes=(2, 100),
                        max_iter=10000,learning_rate_init=0.00000001, learning_rate='constant',
                        random_state=1);
model.fit(x,y);
tt_size=len(y);
print("Training done");
y_pred = model.predict(x)
print(y_pred);
tt_misclassifedPoints = (y_pred != y).sum()  
tt_accuracy = ((tt_size - tt_misclassifedPoints)*100) / tt_size;
print(misclassifedPoints,":",accuracy);

x=mtx_test_norm;
y=mtx_test_y_vector;
y_pred = model.predict(x)
size=len(y);
print(y_pred);
tt_misclassifedPoints = (y_pred != y).sum()  
tt_accuracy = ((size - tt_misclassifedPoints)*100) / size;
print(misclassifedPoints,":",accuracy)


# In[163]:




# In[ ]:



