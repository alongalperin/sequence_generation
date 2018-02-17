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
print('We have total of %d songs' % len(total_songs))
```
We have total of 390 songs  
  
# Part 2: Building Classifier for Classifying Lyrics and artists  
### Data Preparation
In this step we need to prepare and clean the data. First we will clean the data.  
We dont need to check for missing values since we handled this on part 1.  
  
The steps that we are going to apply are:  
1. Change the text format to uff-8.
2. Change the text to lower-case.
3. Tokenize the sentences to words token.
4. Stop words removal. We also remove puncuation (using python puncuation list string.punctuation)
5. Stem the words using Porter Stemmer
Last step we will concat the sentence together  

word_punct_tokenizer = WordPunctTokenizer() # from nltk.tokenize
stop_words = stopwords.words('english') + list(string.punctuation)
ps = PorterStemmer()
```
def clean_and_prepare(text):
    text = text.decode("utf-8")
    text = text.lower()
    text_tokens = word_punct_tokenizer.tokenize(text)
    meaningful_words  = [w for w in text_tokens if not w in stop_words]
    stemmed_text = []
    for word in meaningful_words :
        stem = ps.stem(word)
        stemmed_text.append(stem)
    text = ' '.join(stemmed_text)
    return text
```
We will print the 5 lyrics before and after the cleaning:  
Before cleaninig:  
![beforecleaninig](https://github.com/alongalperin/sequence_generation/blob/master/images/text_before.JPG)  

After cleaning:  
We can see that all the text is lower case, the stop words and pucnchuation are not there, and the text is stemeed.  
![aftercleaning](https://github.com/alongalperin/sequence_generation/blob/master/images/text_after.JPG)  
Additional inforamtion that we want see check is what is the average song length for Eminem, Beyonce-Knowles and Arctic Monkeys. Maybe it will be usefull in the future.  
![avg;ength](https://github.com/alongalperin/sequence_generation/blob/master/images/avg_length.JPG)  

