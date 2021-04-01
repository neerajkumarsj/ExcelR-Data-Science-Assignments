# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 17:25:31 2020

@author: HP
"""
import pandas as pd

import requests
from bs4 import BeautifulSoup as bs
import re
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud

kindle_reviews=[]
for i in range(1,200):
    ip=[]
    respons=requests.get('https://www.amazon.in/All-New-Kindle-reader-Glare-Free-Touchscreen/product-reviews/B0186FF45G/ref=cm_cr_getr_d_paging_btm_3?showViewpoints=1&pageNumber='+str(i))
    soup=bs(respons.content,'html.parser')
    reviews=soup.findAll('span', attrs={"class","a-size-base review-text review-text-content"})
    
    for i in range(len(reviews)):
        ip.append(reviews[i].text)
        kindle_reviews=kindle_reviews+ip
    
reviews_strings=" ".join(kindle_reviews)
#reviews_strings = " ".join(kindle_reviews)

reviews_strings = re.sub("[^A-Za-z" "]+"," ",reviews_strings).lower()
reviews_strings = re.sub("[0-9" "]+"," ",reviews_strings)

reviews_words=reviews_strings.split(" ")

with open("D:\\Data Analytics Assignments\\Text Mining\\stop.txt","r") as sw:
    stopwords = sw.read()
stopwords=stopwords.split('\n')

reviews_words=[w for w in reviews_words if w not in stopwords]

reviews_strings=" ".join(reviews_words)

wordcloud_ip=WordCloud(
        background_color=None,
        mode='RGBA',
        width=1800,
        height=1600
        ).generate(reviews_strings)
plt.imshow(wordcloud_ip)

help(WordCloud)

with open("D:\\Data Analytics Assignments\\Text Mining\\positive-words.txt","r") as pos:
    positive=pos.read().split('\n')
                    
    
positive=positive[36:]
pos_reviews=[w for w in reviews_words if w in positive]
pos_strings=" ".join(pos_reviews)

wordcloud_ip=WordCloud(
        background_color=None,
        mode='RGBA',
        width=1800,
        height=1600
        ).generate(pos_strings)
plt.imshow(wordcloud_ip)

