#!/usr/bin/env python
# coding: utf-8
#generate requirements: pip freeze > requirements.txt


import numpy as np
import pandas as pd
from pandas import DataFrame

import sys, os

import json
from pandas.io.json import json_normalize

from sklearn.preprocessing import normalize
import csv

import gensim
from gensim.models.word2vec import Word2Vec
from collections import defaultdict

import re
import operator
from operator import itemgetter, attrgetter, add


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
import string

import pickle
from clean_text import vectorize_and_store_existing_titles, vectorize_new_title

def rank_existing_titles(json_vectorized_title_df):
    '''
    Load the current titles in the Pangea database,
    and then rank them by similarity to the latest user query

    Input: vectorized titles (dictionary .pkl)
    Output: sorted title vectors (dict)
    '''
    ranked_titles = {}
    other_titles = pd.read_pickle("./vectorized_titles.pkl")
    for index,row in other_titles.iterrows():
        ranked_titles[row['Titles']] = sum(row['Vectors'][0]*json_vectorized_title_df['Vectors'][0][0])
    sorted_title_vecs = sorted(ranked_titles.items(), key=operator.itemgetter(1), reverse=True)
    return(sorted_title_vecs)

def generate_recommendations(title, model):
    '''
    Final function call API that puts together the
    prior 3 functions in a neat mega-function

    Input:
    - User inputted titles via curl command (str)
    - Google News Vectors Model (gensim model .bin)

    Output:
    - ranked titles (dict)
    note that this will print in the terminal on the client side
    '''
    vectorized_title = vectorize_new_title(title,model)
    ranked_titles = rank_existing_titles(vectorized_title)
    other_titles = pd.read_pickle("./vectorized_titles.pkl")
    other_titles.append({"Titles": title, "Vectors": vectorized_title}, ignore_index=True)

    with open("./ranked_titles.csv", "w", newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for title in ranked_titles:
            wr.writerow([ranked_titles, title])
    print("*COMPLETE")
    return(ranked_titles)
