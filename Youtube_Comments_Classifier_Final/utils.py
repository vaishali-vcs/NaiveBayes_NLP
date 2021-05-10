# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 02:20:24 2021

@author: admin
"""

# Imports

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = nltk.PorterStemmer()

def data_preprocessing(input_data):
    
    
    # defining the independent column
    data_col = input_data["CONTENT"]
    
    # Replace money symbols with ''
    processed = data_col.str.replace(r'£|\$', '')
    
    # Replace 10 digit phone numbers (formats include paranthesis, spaces, no spaces, dashes) with 'phonenumber'
    processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$','')
    
    # Replace whitespace between terms with a single space
    processed = processed.str.replace(r'\s+', ' ')
    
    # Remove leading and trailing whitespace
    processed = processed.str.replace(r'^\s+|\s+?$', '')    
    
    # Processed data into lowercase
    processed = processed.str.lower()
    
    processed = processed.apply(lambda x: ' '.join(ps.stem(w) for w in x.split() if w not in stop_words))
    
        
    # Build a count vectorizer and extract term counts 
    count_vectorizer = CountVectorizer()
    train_tc = count_vectorizer.fit_transform(processed)
    
    print("\nDimensions of training data -> ", train_tc.shape, "\n")
    
    return train_tc, count_vectorizer        
           