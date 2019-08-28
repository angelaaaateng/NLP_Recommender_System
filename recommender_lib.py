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




def get_titles(raw):
    titles = raw['title'];
    post_titles = [title for title in titles];
    post_titles = set(post_titles);
    return(post_titles);

def tokenize_title(title):
    tokens = [word for word in title.lower().split()];
    return(tokens);

def remove_punctuation(tokens):
    clean_words = [word.translate(str.maketrans('', '', string.punctuation)) for word in tokens];
    return(clean_words);

def clean_text(model, clean_words):
    stoplist = set(stopwords.words('english'));
    titles_nostopwords = [[word for word in title if word not in stoplist] for title in clean_words];
    filtered_word_list = [[word for word in title if word in model.vocab] for title in titles_nostopwords];
    return(filtered_word_list);

def vectorize_filtered_words(post_titles, filtered_word_list):
    dictionary = dict(zip(post_titles, filtered_word_list))
    vectorized_titles = pd.DataFrame(columns=["Titles", "Vectors"])
    for title in post_titles:
        word_vecs = [model[word] for word in dictionary[title]]
        if len(word_vecs) == 0:
            title_vec = [np.zeros(300)]
        else:
            title_vec = normalize(sum(word_vecs).reshape(1, -1))
        vectorized_titles = vectorized_titles.append({'Titles': title, 'Vectors': title_vec}, ignore_index=True)
    return(vectorized_titles);

def pickle_titles(vectorized_titles):
    vectorized_titles.to_pickle("./vectorized_titles.pkl")
    return(vectorized_titles)

# function 2
def json_clean_text(model, json_clean_words):
    stoplist = set(stopwords.words('english'));
    json_titles_nostopwords = [word for word in json_clean_words if word not in stoplist];
    json_preprocessed = [word for word in json_titles_nostopwords if word in model.vocab];
    return(json_preprocessed)

def normalize_title_vecs(model, json_preprocessed, title):
    json_title_vectors = {}
    normalized_title = pd.DataFrame(columns=["Titles", "Vectors"])
    json_word_vecs = [model[word] for word in json_preprocessed]
    #manually normalizing the word vectors here since normalize command here didn't work
    if len(json_preprocessed) == 0:
        json_title_vec = [np.zeros(300)]
    else:
        json_title_vec = normalize(sum(json_word_vecs).reshape(1, -1))
    normalized_title = normalized_title.append({'Titles': title, 'Vectors': json_title_vec}, ignore_index = True)
    return(normalized_title);

def store_title_csv(normalized_title):
    if not os.path.isfile('./ranked_titles.csv'):
        normalized_title.to_csv (r'./ranked_titles.csv', index = None, header=True)
    else:
        normalized_title.to_csv (r'./ranked_titles.csv', mode='a', index = None, header=False)
    return(normalized_title)

'''
Compilation of Mini-Functions into Larger Functions
'''

def vectorize_and_store_existing_titles(model):
    '''
    Vectorize and store existing titles in legacy Pangea database

    Input: Word2Vec Model (gensim model - dict object with .bin extension)
    Output: Vectorized Titles (titles.dict, which is saved as a .pkl)

    Note: can split this up into two different functions later
    '''
    raw = pd.read_csv("allPostData.csv", header=0);
    post_titles = get_titles(raw);
    #tokenization
    tokens = [tokenize_title(title) for title in post_titles];
    #clean_words
    clean_words = remove_punctuation(tokens);
    #clean_text
    clean_text = clean_text(clean_words);
    #vectorize_filtered_words
    vectorized_titles = vectorize_filtered_words(post_titles, clean_text);
    #pickle_titles
    pickled_titles = pickle_titles(vectorized_titles);
    return(pickled_titles)


def vectorize_new_title(title, model):
    '''
    Vectorize each new title as a user/student/company creates a new post

    Input:
    - title from user query curl command (str)
    - model (.bin)

    Output: json_vectorized_title_df (dict)

    '''
    #tokenization of one new title
    json_tokens = tokenize_title(title);
    #clean_words
    json_clean_words = remove_punctuation(json_tokens);
    #clean_text
    json_preprocessed = json_clean_text(model, json_clean_words);
    #vectorize_new_title
    normalized_title = normalize_title_vecs(model, json_preprocessed, title);
    #pickle_titles
    json_vectorized_title_df = store_title_csv(normalized_title);
    return(json_vectorized_title_df)


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
