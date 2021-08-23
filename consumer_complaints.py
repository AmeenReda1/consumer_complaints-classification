import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
from textblob import TextBlob
from nltk.stem import PorterStemmer
from textblob import Word
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import sklearn.feature_extraction.text as text
from sklearn import model_selection, preprocessing,linear_model, naive_bayes, metrics, svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from io import StringIO
from sklearn.metrics import confusion_matrix
import seaborn as sns
# this is the link of the dataset 
# https://www.kaggle.com/subhassing/exploring-consumer-complaint-data/data
Data = pd.read_csv("./consumer_complaints/consumer_complaints.csv",encoding='latin-1')
# Selecting required columns and rows
Data = Data[['product', 'consumer_complaint_narrative']]
print(Data.shape)
Data = Data[pd.notnull(Data['consumer_complaint_narrative'])]
#print(Data.shape)
#print(Data.head())
# Factorizing the category column
Data['category_id'] = Data['product'].factorize()[0]
#print(Data.head())
# Check the distriution of complaints by category
num_complaintsby_category=Data.groupby('product').consumer_complaint_narrative.count()
#print("The num_complaintsby_category is : ",num_complaintsby_category)
# Lets plot it and see
fig = plt.figure(figsize=(8,6))
Data.groupby('product').consumer_complaint_narrative.count().plot.bar(ylim=0)
#plt.show()
#split data to train and test
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(Data['consumer_complaint_narrative'], Data['product'])
# Feature engineering using Tf_idf
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
tfidf_vect = TfidfVectorizer(analyzer='word',
token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(Data['consumer_complaint_narrative'])
xtrain_tfidf = tfidf_vect.transform(train_x)
xvalid_tfidf = tfidf_vect.transform(valid_x)
print("xtrain_tfidf: ",xtrain_tfidf)
print("xvalid_tfidf: ",xvalid_tfidf)
# Model summary
model = linear_model.LogisticRegression().fit(xtrain_tfidf, train_y)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, max_iter=100, multi_class='ovr',n_jobs=1,
                    penalty='l2', random_state=None, solver='liblinear',
                    tol=0.0001,
                    verbose=0, warm_start=False)
# Checking accuracy
accuracy = metrics.accuracy_score(model.predict(xvalid_tfidf),
valid_y)
print ("Accuracy: ", accuracy)
# Classification report
#print(metrics.classification_report(valid_y, model.predict(xvalid_tfidf),target_names=Data['product'].unique()))
#confusion matrix
conf_mat = confusion_matrix(valid_y, model.predict(xvalid_tfidf))
# Vizualizing confusion matrix
category_id_df = Data[['product', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id','product']].values)
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap="BuPu",
 xticklabels=category_id_df[['product']].values,
yticklabels=category_id_df[['product']].values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()
# Prediction example
texts = ["This company refuses to provide me verification andvalidation of debt"+ "per my right under the FDCPA.I do not believe this debt is mine."]
text_features = tfidf_vect.transform(texts)
predictions = model.predict(text_features)
print(texts)
print(" - Predicted as: '{}'".format(id_to_category[predictions[0]]))