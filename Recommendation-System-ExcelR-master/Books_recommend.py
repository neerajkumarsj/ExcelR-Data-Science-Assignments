# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 00:20:26 2020

@author: HP
"""

import numpy as np
import pandas as pd
import chardet

#To prevent unicode error
with open(r'D:\\Data Analytics Assignments\\Recommendation System\\books.csv','rb') as f:
    data=chardet.detect(f.read())
    books=pd.read_csv(r'D:\\Data Analytics Assignments\\Recommendation System\\books.csv', encoding=data['encoding'])     

from sklearn.feature_extraction.text import TfidfVectorizer
books.columns
tfidf=TfidfVectorizer(stop_words='english')
books['Book.Title'].isnull().sum()

tfidf_matrix=tfidf.fit_transform(books['Book.Title'])
tfidf_matrix.shape

from sklearn.metrics.pairwise import linear_kernel

cosine_sin_matrix=linear_kernel(tfidf_matrix,tfidf_matrix)

book_index=pd.Series(books.index, index=books['Book.Title']).drop_duplicates()

book_index['Jane Doe']

def books_recommendation(Name, TopN):
    book_id=book_index[Name]   #using arguments
    cosine_scores=list(enumerate(cosine_sin_matrix[book_id]))
    cosine_scores=sorted(cosine_scores, key=lambda x:x[1], reverse=True)
    cosine_scores_10=cosine_scores[0:TopN+1]
    
    book_idx=[i[0] for i in cosine_scores_10 ]
    book_scores=[i[1] for i in cosine_scores_10]
    
    book_similar_show = pd.DataFrame(columns=["Book.Title"])
    book_similar_show["Book.Title"] = books.loc[book_idx,"Book.Title"]    
    book_similar_show["rating"] = books.loc[book_idx,"ratings"]
    book_similar_show["score"] = book_scores
    book_similar_show.reset_index(inplace=True)  
    book_similar_show.drop(["index"],axis=1,inplace=True)
    print(book_similar_show)
    
books_recommendation('Jane Doe', TopN=5)
    