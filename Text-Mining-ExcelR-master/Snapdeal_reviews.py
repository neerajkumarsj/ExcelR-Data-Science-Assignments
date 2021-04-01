# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 19:43:14 2020

@author: HP
"""

import requests   # Importing requests to extract content from a url
from bs4 import BeautifulSoup as bs # Beautifulsoup is for web scrapping...used to scrap specific content 
import re # regular expressions 

import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
from wordcloud import WordCloud

flipkart_reviews=[]

for i in range(1,100):
    ip=[]
    respons=requests.get('https://www.flipkart.com/philips-qt3310-15-runtime-30-min-trimmer-men/product-reviews/itmc7efd5003ea13?pid=SHVEHHFBMHJZUZWS&lid=LSTSHVEHHFBMHJZUZWSKCNLNB&marketplace=FLIPKART&page='+str(i))
    soup=bs(respons.content,'html.parser')
    reviews=soup.findAll('div', attrs={"class","qwjRop"})
    
    for i in range(len(reviews)):
        ip.append(reviews[i].text)
        flipkart_reviews=flipkart_reviews+ip
        
trimmer_rev_string=" ".join(flipkart_reviews)
trimmer_rev_string=re.sub("[^A-Za-z" "]+"," ",trimmer_rev_string).lower()
trimmer_rev_string=re.sub("[0-9" "]+"," ",trimmer_rev_string)

trimmer_rev_words=trimmer_rev_string.split(" ")
with open("D:\\Data Analytics Assignments\\Text Mining\\stop.txt","r") as sw:
    stopwords = sw.read()
stopwords=stopwords.split("\n")


trimmer_reviews_words=[w for w in trimmer_rev_words if w not in stopwords]

trimmer_reviews_strings=" ".join(trimmer_reviews_words)

wordcloud_ip=WordCloud(
        background_color=None,
        mode='RGBA,
        width=1800,
        height=1600
        ).generate(trimmer_reviews_strings)
plt.imshow(wordcloud_ip)


with open("D:\\Data Analytics Assignments\\Text Mining\\positive-words.txt","r") as pos:
    positive=pos.read().split('\n')
                    
    
positive=positive[36:]

trimmer_pos_reviews=[w for w in trimmer_reviews_words if w in positive]
trimmer_pos_strings=" ".join(trimmer_pos_reviews)

wordcloud_ip=WordCloud(
        background_color=None,
        mode='RGBA',
        width=1800,
        height=1600
        ).generate(trimmer_pos_strings)
plt.imshow(wordcloud_ip)






