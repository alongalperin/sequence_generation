# Workshop for Data Scientist Final Project  
## Sequence Generation with Keras 
## By Alon Galperin & Matan Yeshurun  
The subject of our mission is songs lyrics.

# Part A - Data Collection
In this part we need to get at least 50 songs for each artist, where the number of artists is 3.  
At first we wanted to build crawler what will crawl lyrics sites and get us the lyrics.  
After a good session of searching the internet we got to Kaggle lyrics songs dataset csv which is really great.  
https://www.kaggle.com/gyani95/380000-lyrics-from-metrolyrics/data  
  
The corpus is in format CSV and has lyrics for 380,000+ songs.  
__Structure:__   

| Attribute | type |
|:------------- |:------------- |
| index      | numeric |
| song      | string |
| year      | numeric |
| artist      | string |
| genre      | string |
| lyrics      | string |  
  
At first we will load the csv to dataframe  
```
lyrics = pd.read_csv('./lyrics.csv')  
```
