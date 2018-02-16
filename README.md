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
now we have 266,557 records  
  
we can see that we have enoght songs lyrics.  
  
We saw that in the dataset there are songs of Eminem, Beyonce-Knowles and Arctic Monkeys.  
We will check if we have at least 50 songs for each artist:  
```
print ('Eminem has %d songs' % len(lyrics.loc[lyrics['artist'] == 'eminem']))
print ('Beyonce has %d songs' % len(lyrics.loc[lyrics['artist'] == 'beyonce-knowles']))
print ('Arctic-Monkeys has %d songs' % len(lyrics.loc[lyrics['artist'] == 'arctic-monkeys']))
```
Eminem has 578 songs  
Beyonce has 248 songs  
Arctic-Monkeys has 134 songs  
  
We can see that the lowest number of songs is the number of Arctic Monkeys songs, so we will take **130** songs from each artist.   
We will concat all the 3 artists lyrics to 1 dataset. 
```
eminem_songs = lyrics.loc[lyrics['artist'] == 'eminem'][:130]

beyonce_songs = lyrics.loc[lyrics['artist'] == 'beyonce-knowles'][:130]

arctic_monkeys_songs = lyrics.loc[lyrics['artist'] == 'arctic-monkeys'][:130]

total_songs = eminem_songs.append([beyonce_songs,arctic_monkeys_songs])
print('our total songs has %d songs' % len(total_songs))
```
our total songs has 390 songs
