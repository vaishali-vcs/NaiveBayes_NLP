# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 02:20:24 2021

@author: admin
"""

# Imports
import numpy as np 
import pandas as pd
import os
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
import nltk
ps = nltk.PorterStemmer()

def data_preprocessing(data_col):
    
    
    # use regular expressions to replace email addresses, URLs, phone numbers, other numbers
    
    # # Replace email addresses with 'email'
    # processed = data_col.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','')
    
    # # Replace URLs with 'web_address'
    # processed = processed.str.replace(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*','')
    
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
    
    return processed          
           