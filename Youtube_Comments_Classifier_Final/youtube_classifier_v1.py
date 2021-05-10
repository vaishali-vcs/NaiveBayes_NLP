# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 02:01:14 2021

@authors: [Shivam Verma, Divyanshu Johar, Gagandeep Singh]
"""

# Imports
import numpy as np 
import pandas as pd
import os
import pickle
import matplotlib.pyplot as plt
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)
import nltk
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics

import utils
import DataAnalysis

# Load dataset into pandas dataframe
dir_name = r'E:\shivam_centennial\AI_237\assigmnets\final_eval_project_NLP\Youtube_Comments_Classifier_Final'
input_file = 'Input_Files'
file_name = 'Youtube02-KatyPerry.csv'

input_path = os.path.join(dir_name,input_file,file_name)
input_data = pd.read_csv(input_path)


# Create Output Folder 
output_file = 'Output_Files'
output_path = os.path.join(dir_name,output_file)
output_img_path = os.path.join(dir_name,output_file,"Images")
if not os.path.exists(output_path):
    os.makedirs(output_path)

if not os.path.exists(output_img_path):
    os.makedirs(output_img_path)

#set seed
np.random.seed(1)

DataAnalysis.get_data_analysis(input_data, input_path, output_img_path)


# Define the category map
category_map = {0 : 'Ham', 1 : 'Spam'}

# Plot the number of values in each classes given.
input_data['CLASS'].value_counts().plot(kind='bar')

# Shuffle the dataset
input_data = input_data.sample(frac=1)

# Pre-Processing of the input data
vectorized_pre_processed_input_data, count_vectorizer = utils.data_preprocessing(input_data)

# define dependent variable
y = input_data['CLASS']

#This downscaling is called tf–idf for “Term Frequency times Inverse Document Frequency”.
# Create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(vectorized_pre_processed_input_data)
# train_tfidf = tfidf.fit_transform(train_tc)


# create random train/test split
indices = range(train_tfidf.shape[0])
num_training_instances = int(0.75 * train_tfidf.shape[0])
train_indices = indices[:num_training_instances]
test_indices = indices[num_training_instances:]

# split the actual data
X_train, X_test = train_tfidf[train_indices], train_tfidf[test_indices]
y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

# Train Test Split
# X_train, X_test, y_train, y_test = train_test_split(train_tfidf, y, test_size = 0.25, random_state = 1)

# Train a Naive Bayes classifier
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)


# Metrices Classification of Model
print(metrics.classification_report(y_test, y_pred, target_names=category_map.values()),"\n")


# Accuracy Score
print("Accuracy Score is ", round(accuracy_score(y_test, y_pred)*100,2),"%")

# Confussion Matrix
confussion_matrix = pd.DataFrame(
    confusion_matrix(y_test, y_pred),
    index = [['actual', 'actual'], ['ham', 'spam']],
    columns = [['predicted', 'predicted'], ['ham', 'spam']])

print("\nconfussion_matrix is ->   \n",confussion_matrix, "\n")

# Scoring functions
num_folds = 5
accuracy_values = cross_val_score(spam_detect_model, X_train, y_train, scoring='accuracy', cv=num_folds)

for i in range(len(accuracy_values)):
  print('Accuracy for k-fold '+str(i+1)+' is ',round(accuracy_values[i]*100,2),'%')

print("Mean Accuracy for K-fold is : " + str(round(100*accuracy_values.mean(), 2)) + "%")

# Save the model as .pkl file
model_file = 'youtube_comment_pridictor.pkl'
pickle.dump(spam_detect_model, open(os.path.join(output_path,model_file), 'wb'))


## Testing the model with custom test inputs

# Load the saved model

model_file = 'youtube_comment_pridictor.pkl'
youtube_pred_model = pickle.load(open(os.path.join(output_path,model_file),"rb"))

data = ['It was a good movie',
        'I really love this video.. http://www.bubblews.com/account/389088-sheilcenï»¿',
        'i need money in bitcoin this is my url https://abc.com/sdhiuwq/!@#$%%',
        'I love katy perry very much, i love her songs',
        'idk why she is famous',
        'please like me facebook page www.facebook.com/asiuwdgwdigwd'
        ]

input_test_data = count_vectorizer.transform(data)
input_test_data = tfidf.transform(input_test_data)

test_prediction = youtube_pred_model.predict(input_test_data)

# Print the outputs
for sent, category in zip(data, test_prediction):
    print('\nInput:', sent, '\nPredicted category:', \
            category_map[category])