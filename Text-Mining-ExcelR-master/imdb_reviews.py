# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 22:50:55 2020

@author: HP
"""

import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re # regular expressions 

import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud

imdb_review=[]
for i in range(1,200):
    ip=[]
    respons=requests.get('https://www.imdb.com/title/tt7286456/reviews?ref_=tt_urv')
    soup=bs(respons.content, 'html.parser')
    imdb_rev=soup.findAll('div', attrs={"class","content"})
    
    for i in range(len(imdb_rev)):
        ip.append(imdb_rev[i].text)
        imdb_review=imdb_review+ip
        
joker_rev_string=" ".join(imdb_review)
joker_rev_string=re.sub("[^A-Za-z" "]+"," ",joker_rev_string).lower()
joker_rev_string=re.sub("[0-9" "]+"," ",joker_rev_string)

joker_rev_words=joker_rev_string.split(" ")
with open("D:\\Data Analytics Assignments\\Text Mining\\stop.txt","r") as sw:
    stopwords = sw.read()
stopwords=stopwords.split("\n")
joker_reviews_words=[w for w in joker_rev_words if w not in stopwords]
joker_reviews_strings=" ".join(joker_reviews_words)

wordcloud_ip=WordCloud(
        background_color='black',
        mode='RGB',
        width=1800,
        height=1600
        ).generate(joker_reviews_strings)
plt.imshow(wordcloud_ip)


with open("D:\\Data Analytics Assignments\\Text Mining\\negative-words.txt","r") as neg:
    negative=neg.read().split('\n')
                    
    
negative=negative[36:]

joker_neg_reviews=[w for w in joker_reviews_words if w in negative]
joker_neg_strings=" ".join(joker_neg_reviews)

wordcloud_ip=WordCloud(
        background_color='black',
        mode='RGB',
        width=1800,
        height=1600
        ).generate(joker_neg_strings)
plt.imshow(wordcloud_ip)
