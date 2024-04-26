# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 20:31:08 2024

@author: pratik
"""

#Import pandas 
import pandas as pd 

#Import the dataset 

sms = pd.read_csv("/Users/prati/OneDrive/Documents/SMSSpamCollection.txt",sep='\t',names=("label","message"))

#Data Cleaning and Preprocessing 
import re 
import nltk
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
ps = PorterStemmer()
corpus=[]


for i in range(0,len(sms)):
    review = re.sub('[^a-zA-Z]',' ',sms['message'][i])
    review = review.lower()
    review = review.split()

    review = [ps.stem(word) for word in review if not word in stopwords.words("english")]
    review = ''.join(review)
    corpus.append(review)
    
    
#Creating Bag of words Model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()
    
y = pd.get_dummies(sms['label'])
y = y.iloc[:,1].values


#Train Test Split 

from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Model Training Using Naive bayes Classifier 

from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train,y_train)

#Prediction 
y_pred = NB.predict(X_test)

#Accuracy 
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))

#TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
tfvector = TfidfVectorizer(max_features=2500)
X = tfvector.fit_transform(corpus).toarray()

y = pd.get_dummies(sms['label'])
y = y.iloc[:,1].values

#Train Test Split 
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X,y,test_size=0.2,random_state=0)

#Model Training Using Naive bayes Classifier 

from sklearn.naive_bayes import MultinomialNB
NB = MultinomialNB()
NB.fit(X_train,y_train)


#Prediction 
y_pred = NB.predict(X_test)

#Accuracy 
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))