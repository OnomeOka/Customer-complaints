#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import numpy as np


# In[24]:


pro = pd.read_csv('complaints_processed.csv')


# In[25]:


pro.head()


# In[26]:


# checking for any missing value
pro.isnull().sum()


# In[27]:


#checking for duplicates
pro.duplicated()


# In[28]:


# droping any columns with missing value
pro.dropna(axis=0, inplace=True)


# In[29]:


#checking for the occurance of each unique values in the column
pro['product'].value_counts()


# In[32]:


import re
import string
import nltk 

def text_clean(text):
    clean_words = []
    
    word_list = text.split() 
    for word in word_list:
        word_l = word.lower().strip()
        if word_l.isalpha():
            if len(word_l) > 3:
                if word_l not in nltk.corpus.stopwords.words('english'):
                    clean_words.append(word_l)
                else:
                    continue
    return clean_words     


# In[33]:


#checking for words that appears frequently
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(analyzer=text_clean)
X=tfidf.fit_transform(pro['narrative'][:5000])


# In[35]:


#To split the words
def tokenizer(text):
    return text.split()
tokenizer('forwarded like forword and they have forward')


# In[36]:


# checking for imbalance in the dataset
from imblearn.over_sampling import SMOTE
X_smote, Y_smote = SMOTE().fit_resample(X, pro['product'][:5000])


# In[38]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X_smote, Y_smote, test_size=0.20, random_state=42)


# In[39]:


from sklearn import linear_model


# In[41]:


logr = linear_model.LogisticRegression()
#train the model
logr.fit(X_train,Y_train)


# In[44]:


# make prediction on the test set 
Y_pred = logr.predict(X_test)


# In[47]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc


# In[48]:


# Evaluate teh accuracy of the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")


# In[49]:


# print confusion matrix and classification report
print('confusion Matrix:')
print(confusion_matrix(Y_test, Y_pred))


# In[51]:


print('\nclassification Report')
print(classification_report(Y_test, Y_pred))


# In[54]:


from sklearn.ensemble import RandomForestClassifier


# In[62]:


rf = RandomForestClassifier()
#train the model
rf.fit(X_train, Y_train)


# In[63]:


#make prediction on the test set
Y_pred=rf.predict(X_test)


# In[64]:


# evalute model performace
print(classification_report(Y_test, Y_pred))


# In[ ]:




