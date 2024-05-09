from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
link = 'https://finviz.com/quote.ashx?t=' 
companies = ['AMZN', 'GOOG', 'AMD']
data_table = {}

for comp in companies:
    url = link + comp + '&p=d'
    req = Request(url=url, headers={'user-agent': 'my-app'})
    response = urlopen(req)
    html = BeautifulSoup(response, "lxml")

    table = html.find(id='news-table')
    data_table[comp] = table



all_data = []


for comp, data_item in data_table.items():
    for row in data_item.find_all('tr'):
        try:
            title = row.a.text
            date = row.td.text.split()
            if len(date) == 1:
                time = date[0]
            else:
                day = date[0]
                time = date[1]
            if day == 'Today':
                day = datetime.today()
            all_data.append([comp, day, time, title])
        except:
            pass 


df = pd.DataFrame(all_data, columns=['comp', 'day', 'time', 'title'])
vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
df['score'] = df['title'].apply(f)
df['day'] = pd.to_datetime(df.day).dt.date

plt.figure(figsize=(10, 8))

mean_data = df.groupby(['comp', 'day']).mean(numeric_only=True).unstack()
mean_data = mean_data.xs('score', axis="columns").transpose()
mean_data.plot(kind='bar')
plt.show()