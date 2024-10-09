# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("C:/Users/print/Downloads/online_retail.csv", dtype={'CustomerID': str,'InvoiceID': str}, encoding="ISO-8859-1")

df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format="%m/%d/%Y %H:%M")

df = df.dropna()
df.head()



df['Description'].tolist()



import collections

corpus = " ".join(df['Description'].tolist()).split(" ")
count = collections.Counter(corpus)
print(count)

import random
import pytagcloud
import webbrowser

ranked_tags = count.most_common(40)
taglist = pytagcloud.make_tags(ranked_tags, maxsize=200)
pytagcloud.create_tag_image(taglist, 'wordcloud_example.jpg', size=(900, 600), rectangular=False)

from IPython.display import Image
Image(filename='wordcloud_example.jpg')



