#!/usr/bin/env python
# coding: utf-8

# In[254]:


import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# In[255]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# In[256]:


data = pd.read_csv('input.csv')
data.columns = [ x.lower().strip() for x in data.columns]
data.columns


# In[257]:


num_cols = data._get_numeric_data().columns
num_cols


# In[258]:


grouping_dict = {}

grouping = ['concom','comorb','risk']

for val in grouping:
    for col in data.columns:
        if col.startswith(val):
            grouping_dict[col] = val


# In[259]:


grouping_dict


# In[260]:


del grouping_dict['risk_segment_prior_ntm']
del grouping_dict['risk_segment_during_rx']

combined_columns = [x for x in grouping_dict.keys()]


# In[261]:


combined_columns


# In[262]:


le = LabelEncoder()
for val in combined_columns:
    data[val] = le.fit_transform(data[val])


# In[263]:


data.describe()


# In[264]:


data = data.drop(['change_risk_segment','risk_segment_during_rx','tscore_bucket_during_rx','change_t_score'],axis = 1)


# In[265]:


binary_cols = ['persistency_flag','gender','ntm_specialist_flag','gluco_record_prior_ntm','gluco_record_during_rx','dexa_during_rx','frag_frac_during_rx','risk_segment_prior_ntm','tscore_bucket_prior_ntm','adherent_flag','idn_indicator','injectable_experience_during_rx','frag_frac_prior_ntm']


# In[267]:


for col in binary_cols:
    data[col] = le.fit_transform(data[col])


# In[268]:


data = data.drop(['race','region','ethnicity'],axis =1)


# In[269]:


data["age_bucket"] = data["age_bucket"].astype('category')
data['age_bucket'] = le.fit_transform(data['age_bucket'])
data["ntm_speciality"] = data["ntm_speciality"].astype('category')
data['ntm_speciality'] = le.fit_transform(data['ntm_speciality'])
data["ntm_speciality_bucket"] = data["ntm_speciality_bucket"].astype('category')
data['ntm_speciality_bucket'] = le.fit_transform(data['ntm_speciality_bucket'])


# In[270]:


corrmat = data.corr()


# In[271]:


k = 10 
cols = corrmat.nlargest(k, 'persistency_flag')['persistency_flag'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[272]:


most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr


# In[274]:


features = data[cols]
target = features[['persistency_flag']]
features = features.drop(['persistency_flag'],axis=1)


# In[275]:


x_train, x_test,y_train, y_test = train_test_split(features,target,test_size = 0.2, random_state = 0)


# In[276]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
log_y_pred = logreg.predict(x_test)
acc_logreg = accuracy_score(y_test,log_y_pred)
acc_logreg


# In[324]:


confusion_matrix(y_test,log_y_pred)


# In[282]:


knn = KNeighborsClassifier(n_neighbors = 50)
knn.fit(x_train,y_train)
knn_y_pred = knn.predict(x_test)
acc_knn = accuracy_score(y_test,knn_y_pred)
acc_knn


# In[325]:


confusion_matrix(y_test,knn_y_pred)


# In[326]:


f1_score(y_test,knn_y_pred)


# In[294]:


classifier = KerasClassifier(build_fn = classifier, batch_size = 32 , epochs = 1000 )


# In[295]:


classifier.fit(x_train,y_train)


# In[ ]:





# In[ ]:





# In[ ]:





# In[173]:


test = data.set_index('ptid').groupby(grouping_dict,axis=1).sum()

test.columns = ['concomitancy_count','comorbidity_count','risk_factors_count']

test = test.reset_index()

test.head(5)


# In[174]:


data = data.drop(combined_columns,axis=1)

data = pd.merge(data, test, on='ptid', how = 'inner')

data.pop('count_of_risks')


# In[175]:


data['persistency_flag'].value_counts()


# In[176]:


data.describe(include=['O'])


# In[177]:


sns.catplot(x="change_risk_segment", hue="persistency_flag", kind="count",palette="pastel", edgecolor=".6",data=data)


# In[178]:


sns.catplot(x="risk_segment_during_rx", hue="persistency_flag", kind="count",palette="pastel", edgecolor=".6",data=data)


# In[179]:


sns.catplot(x="tscore_bucket_during_rx", hue="persistency_flag", kind="count",palette="pastel", edgecolor=".6",data=data)


# In[180]:


sns.catplot(x="change_t_score", hue="persistency_flag", kind="count",palette="pastel", edgecolor=".6",data=data)


# In[181]:


data = data.drop(['change_risk_segment','risk_segment_during_rx','tscore_bucket_during_rx','change_t_score'],axis = 1)


# In[182]:


data = data.drop(['race','region','ethnicity'],axis =1)


# In[183]:


data.describe(include='O')


# In[184]:


binary_cols = ['persistency_flag','gender','ntm_specialist_flag','gluco_record_prior_ntm','gluco_record_during_rx','dexa_during_rx','frag_frac_during_rx','risk_segment_prior_ntm','tscore_bucket_prior_ntm','adherent_flag','idn_indicator','injectable_experience_during_rx','frag_frac_prior_ntm']


# In[185]:


for col in binary_cols:
    data[col] = le.fit_transform(data[col])


# In[186]:


data.describe(include='O')


# In[187]:


data.dtypes


# In[188]:


data.shape


# In[189]:


data["age_bucket"] = data["age_bucket"].astype('category')
data['age_bucket'] = le.fit_transform(data['age_bucket'])
data["ntm_speciality"] = data["ntm_speciality"].astype('category')
data['ntm_speciality'] = le.fit_transform(data['ntm_speciality'])
data["ntm_speciality_bucket"] = data["ntm_speciality_bucket"].astype('category')
data['ntm_speciality_bucket'] = le.fit_transform(data['ntm_speciality_bucket'])


# In[190]:


data["ntm_speciality"] = data["ntm_speciality"].astype('category')
data['ntm_speciality'] = le.fit_transform(data['ntm_speciality'])


# In[191]:


data["ntm_speciality_bucket"] = data["ntm_speciality_bucket"].astype('category')
data['ntm_speciality_bucket'] = le.fit_transform(data['ntm_speciality_bucket'])


# In[192]:


corrmat = data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)


# In[193]:


k = 10 
cols = corrmat.nlargest(k, 'persistency_flag')['persistency_flag'].index
cm = np.corrcoef(data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()


# In[194]:


most_corr = pd.DataFrame(cols)
most_corr.columns = ['Most Correlated Features']
most_corr
#try one hot encoding for ntm speciality bucket


# In[195]:


features = data[cols]
target = features[['persistency_flag']]
features = features.drop(['persistency_flag'],axis=1)


# In[196]:


features.head(5)


# In[197]:


target.head(5)


# In[198]:


x_train, x_test,y_train, y_test = train_test_split(features,target,test_size = 0.2, random_state = 0)
y_train.shape


# - Logistic Regression
# - KNN or k-Nearest Neighbors
# - Support Vector Machines
# - Naive Bayes classifier
# - Decision Tree
# - Random Forrest
# - Perceptron
# - Artificial neural network
# - RVM or Relevance Vector Machine

# In[200]:


logreg = LogisticRegression()
logreg.fit(x_train, y_train)
log_y_pred = logreg.predict(x_test)
acc_logreg = accuracy_score(y_test,log_y_pred)
acc_logreg


# In[206]:


knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(x_train,y_train)
knn_y_pred = knn.predict(x_test)
acc_knn = accuracy_score(y_test,knn_y_pred)
acc_knn


# In[207]:


gnb = GaussianNB()
gnb.fit(x_train,y_train)
gnb_y_pred = gnb.predict(x_test)
acc_gnb = accuracy_score(y_test,gnb_y_pred)
acc_gnb


# In[283]:


dtc = DecisionTreeClassifier()
dtc.fit(x_train,y_train)
dtc_y_pred = dtc.predict(x_test)
acc_dtc = accuracy_score(y_test,dtc_y_pred)
acc_dtc


# In[215]:


random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
rf_y_pred = random_forest.predict(x_test)
acc_rf= accuracy_score(y_test,rf_y_pred)
acc_rf


# In[217]:


classifier = Sequential()


# In[218]:


from keras.models import Sequential


# In[303]:


from keras.layers import Dense,Dropout
from keras.wrappers.scikit_learn import KerasClassifier


# In[297]:


x_train.shape


# In[313]:


def classifier():
    model = Sequential()
    model.add(Dense(units=64 , kernel_initializer = 'glorot_uniform' , activation = 'relu' , input_dim = 9))
    model.add(Dropout(p=0.2))
    model.add(Dense(units=32 , kernel_initializer = 'glorot_uniform' , activation = 'relu'))
    model.add(Dropout(p=0.2))
    model.add(Dense(units=1 , kernel_initializer = 'glorot_uniform' , activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model


# In[314]:


classifier = KerasClassifier(build_fn = classifier, batch_size = 32 , epochs = 1000 )


# In[315]:


classifier.fit(x_train,y_train)


# In[317]:


predicted = classifier.predict(x_test)


# In[318]:


classifier.score(x_test,y_test)


# In[319]:


accuracy_score(y_test,predicted)


# In[322]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score


# In[321]:


confusion_matrix(y_test,predicted)


# In[323]:


f1_score(y_test,predicted)


# In[ ]:




