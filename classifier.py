#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
import re
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt


# In[2]:


###GET DATA###
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)
print("shapeof data:",data.shape)


# In[3]:


data = data.replace('',np.nan)
data = data.dropna()
print("shape after dropping",data.shape)


# In[4]:


nltk.download()


# In[5]:


###PREPROCESSING####

def preprocess(dats):
    
    entries = data.copy()
    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    for index,entry in entries.iterrows():
        #print(entry['description'])
        soup = BeautifulSoup(entry['description'])
        text = soup.get_text()
        #print(text.lower())
        text.lower()
        text = re.sub(r'^\w+[\s,\t]*:[\s,\t]*https?:\/\/.*.\w*', '', text, flags=re.MULTILINE)
        entry['description']= [word_tokenize(entry['description'])]
        
        #print(entry['description'])
        temp_list = entry['description']
        
        for index,henry in enumerate(temp_list):
            #print(index,henry)
            index = int(index)
            Final_words = []
            word_Lemmatized = WordNetLemmatizer()
            for word, tag in pos_tag(henry):
                if word not in stopwords.words('english') and word.isalpha():
                    word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
                    Final_words.append(word_Final)
                temp_list[index] = str(Final_words[1:-1])        
        entry['description'] = temp_list
    return entries

data_pre = preprocess(data)


# In[6]:


for i in range(len(data_pre.description)):
    data_pre.description.iloc[i] = data_pre.description.iloc[i][0]
    print(data_pre.description.iloc[i])
    data_pre.description.iloc[i] = data_pre.description.iloc[i][1:-1].replace(",","")
    print(data_pre.description.iloc[i])
    data_pre.description.iloc[i] = data_pre.description.iloc[i].replace("'","")
    print(data_pre.description.iloc[i])


# In[7]:


X = data_pre.description
y = data_pre.category
X_train, X_test,y_train,y_test = train_test_split(X,y)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)

print(X_train.shape,X_test.shape)
print(y_train.shape,y_test.shape)


# vector = CountVectorizer()
# tfdf = TfidfTransformer()
# 
# X_train_counts = vector.fit_transform(X_train)
# print(X_train_counts.shape)
# X_train_tfidf = tfdf.fit_transform(X_train_counts)
# print(X_train_tfidf.shape)
# clf = naive_bayes.MultinomialNB().fit(X_train_tfidf, y_train)
# 
# X_test_count = vector.transform(X_test)
# print(X_test_count.shape)
# X_test_tfidf = tfdf.fit_transform(X_test_count)
# print(X_test_tfidf.shape)
# pred = clf.predict(X_test_tfidf)
# np.mean(pred == y_test)

# In[8]:


pipe_clf_svm = Pipeline([('vect',CountVectorizer()),('tfdf',TfidfTransformer()),('clf',LinearSVC(dual=False))])
parameters= {
    'vect__ngram_range':[(1,1),(1,2)],
    'tfdf__use_idf': (True,False),
    'clf__penalty' : ('l1','l2'),
    'clf__loss': ('squared_hinge','hinge')
}

gc_clf_svm = GridSearchCV(pipe_clf_svm, parameters,n_jobs=1,error_score=0.0)
gc_clf_svm = gc_clf_svm.fit(X_train,y_train)
print(gc_clf_svm.best_score_)
print(gc_clf_svm.best_params_)


# In[9]:


pred_svm = gc_clf_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test,pred_svm)
precision_svm = precision_score(y_test,pred_svm,average='weighted')
f1_score_svm = f1_score(y_test, pred_svm,average='weighted')
recall_scaore_svm = recall_score(y_test,pred_svm, average='weighted')
print("#####FOR SVM#####")
print("Accuracy: ",accuracy_svm)
print("Precision:",precision_svm)
print("F1 Score:", f1_score_svm)
print("Recall Score:", recall_scaore_svm)


# In[10]:


pipe_clf = Pipeline([('vect',CountVectorizer()),('tfdf',TfidfTransformer()),('clf-multinb',naive_bayes.MultinomialNB())])
parameters= {
    'vect__ngram_range':[(1,1),(1,2)],
    'tfdf__use_idf': (True,False),
    'clf-multinb__alpha': (1e-2,2e-3),
}

gc_clf_nb = GridSearchCV(pipe_clf,parameters, n_jobs=1)
gc_clf_nb = gc_clf_nb.fit(X_train,y_train)
print(gc_clf_nb.best_score_)
print(gc_clf_nb.best_params_)


# In[13]:


pred_nb = gc_clf_nb.predict(X_test)
accuracy_nb = accuracy_score(y_test,pred_nb)
precision_nb = precision_score(y_test,pred_nb,average='weighted')
f1_score_nb = f1_score(y_test, pred_nb,average='weighted')
recall_scaore_nb = recall_score(y_test,pred_nb, average='weighted')
print("####FOR NB######")
print("Accuracy: ",accuracy_nb)
print("Precision:",precision_nb)
print("F1 Score:", f1_score_nb)
print("Recall Score:", recall_scaore_nb)


# In[14]:


pipe_bag = Pipeline([('vect',CountVectorizer()),('tfdf',TfidfTransformer()),('boost', BaggingClassifier(base_estimator=naive_bayes.MultinomialNB()))])

parameters= {
    'vect__ngram_range':[(1,1),(1,2)],
    'tfdf__use_idf': (True,False),
}
gc_clf_bc = GridSearchCV(pipe_bag,parameters, n_jobs=1)
gc_clf_bc = gc_clf_bc.fit(X_train,y_train)
print(gc_clf_bc.best_score_)
print(gc_clf_bc.best_params_)


# In[15]:


pred_bc = gc_clf_bc.predict(X_test)
accuracy_bc = accuracy_score(y_test,pred_bc)
precision_bc = precision_score(y_test,pred_bc,average='weighted')
f1_score_bc = f1_score(y_test, pred_bc,average='weighted')
recall_scaore_bc = recall_score(y_test,pred_bc, average='weighted')
print("#####FOR Bagging with NB")
print("Accuracy: ",accuracy_bc)
print("Precision:",precision_bc)
print("F1 Score:", f1_score_bc)
print("Recall Score:", recall_scaore_bc)


# In[16]:


pipe_boost = Pipeline([('vect',CountVectorizer()),('tfdf',TfidfTransformer()),('boost', AdaBoostClassifier(base_estimator=naive_bayes.MultinomialNB()))])

parameters= {
    'vect__ngram_range':[(1,1),(1,2)],
    'tfdf__use_idf': (True,False),
}

gc_clf_boost = GridSearchCV(pipe_boost,parameters, n_jobs=1)
gc_clf_boost = gc_clf_boost.fit(X_train,y_train)
print(gc_clf_boost.best_score_)
print(gc_clf_boost.best_params_)


# In[17]:


pred_boost = gc_clf_boost.predict(X_test)
accuracy_boost = accuracy_score(y_test,pred_boost)
precision_boost = precision_score(y_test,pred_boost,average='weighted')
f1_score_boost = f1_score(y_test, pred_boost,average='weighted')
recall_scaore_boost = recall_score(y_test,pred_boost, average='weighted')
print("#####FOR BOOSTING WITH NB######")
print("Accuracy: ",accuracy_boost)
print("Precision:",precision_boost)
print("F1 Score:", f1_score_boost)
print("Recall Score:", recall_scaore_boost)


# In[18]:


pipe_bag_svm = Pipeline([('vect',CountVectorizer()),('tfdf',TfidfTransformer()),('boost', BaggingClassifier(base_estimator=LinearSVC(dual=False,penalty="l2",loss="squared_hinge")))])

parameters= {
    'vect__ngram_range':[(1,1),(1,2)],
    'tfdf__use_idf': (True,False),
}

gc_clf_bag_svm = GridSearchCV(pipe_bag_svm, parameters,n_jobs=1,error_score=0.0)
gc_clf_bag_svm = gc_clf_bag_svm.fit(X_train,y_train)
print(gc_clf_bag_svm.best_score_)
print(gc_clf_bag_svm.best_params_)


# In[19]:


pred_bag_svm = gc_clf_bag_svm.predict(X_test)
accuracy_bag_svm = accuracy_score(y_test,pred_bag_svm)
precision_bag_svm = precision_score(y_test,pred_bag_svm,average='weighted')
f1_score_bag_svm = f1_score(y_test, pred_bag_svm,average='weighted')
recall_scaore_bag_svm = recall_score(y_test,pred_bag_svm, average='weighted')
print("#####FOR BAGGING WITH SVC#######")
print("Accuracy: ",accuracy_bag_svm)
print("Precision:",precision_bag_svm)
print("F1 Score:", f1_score_bag_svm)
print("Recall Score:", recall_scaore_bag_svm)


# In[20]:


pipe_boost_svm = Pipeline([('vect',CountVectorizer()),('tfdf',TfidfTransformer()),('boost', AdaBoostClassifier(base_estimator=LinearSVC(dual=False,penalty="l2",loss="squared_hinge"),algorithm="SAMME"))])

parameters= {
    'vect__ngram_range':[(1,1),(1,2)],
    'tfdf__use_idf': (True,False),
}

gc_clf_boost_svm = GridSearchCV(pipe_boost_svm, parameters,n_jobs=1,error_score=0.0)
gc_clf_boost_svm = gc_clf_boost_svm.fit(X_train,y_train)
print(gc_clf_boost_svm.best_score_)
print(gc_clf_boost_svm.best_params_)


# In[21]:


pred_boost_svm = gc_clf_boost_svm.predict(X_test)
accuracy_boost_svm = accuracy_score(y_test,pred_boost_svm)
precision_boost_svm = precision_score(y_test,pred_boost_svm,average='weighted')
f1_score_boost_svm = f1_score(y_test, pred_boost_svm,average='weighted')
recall_scaore_boost_svm = recall_score(y_test,pred_boost_svm, average='weighted')
print("#######FOR BOOSTING WITH SVC######")
print("Accuracy: ",accuracy_boost_svm)
print("Precision:",precision_boost_svm)
print("F1 Score:", f1_score_boost_svm)
print("Recall Score:", recall_scaore_boost_svm)


# In[23]:


plt.figure(figsize=(30,30))
names = ["Naive-Bayes","LinearSVC","Bagging with NB", "Bagging with SVM","Boosting with NB","Boosting with SVM"]
ind = np.arange(1,60,10)
print(ind)
width = 1.5
fig, ax = plt.subplots(figsize=(20,20))
#fig.fisize = (20,20)
def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{0:.3g}'.format(height), ha=ha[xpos], va='bottom')

rect1 = ax.bar(ind - width, (accuracy_nb,accuracy_svm,accuracy_bc,accuracy_bag_svm,accuracy_boost,accuracy_boost_svm), width, color='SkyBlue',label="accuracy" )
rect2 = ax.bar(ind, (precision_nb,precision_svm,precision_bc,precision_bag_svm,precision_boost,precision_boost_svm), width, color="IndianRed", label="precision")
rect3 = ax.bar(ind - width*2, (f1_score_nb,f1_score_svm,f1_score_bc,f1_score_bag_svm,f1_score_boost,f1_score_boost_svm),width,color="red",label="f1_score")
rect4 = ax.bar(ind + width, (recall_scaore_nb,recall_scaore_svm,recall_scaore_bc,recall_scaore_bag_svm,recall_scaore_boost,recall_scaore_boost_svm),width,color="green",label="recall_score")
autolabel(rect1)
autolabel(rect2)
autolabel(rect3)
autolabel(rect4)
ax.set_xticks(ind)
ax.set_xticklabels((names))
ax.legend()


# In[ ]:




