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
  
At first we load the csv to dataframe  
```
lyrics = pd.read_csv('./lyrics.csv')  
```
We want to see how the data looks, we will print the first 5 records:  
```
print lyrics.head()
```
![head raw](https://github.com/alongalperin/sequence_generation/blob/master/images/head.JPG)  
  
We will check for how many missing values are there. The columns that important for us the most are: artist and lyrics.  
```
lyrics.apply(lambda x: sum(x.isnull()),axis=0)
```
![missing values](https://github.com/alongalperin/sequence_generation/blob/master/images/missing.JPG)  
  
We see that we have a lot of records with missing lyrics. We will drop these rows and check with how many records we remained:  
```
lyrics.dropna(subset=['lyrics'], inplace=True)
print('now we have %d records' %len(lyrics))
```
now we have 266557 records

