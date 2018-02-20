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
``` python
lyrics = pd.read_csv('./lyrics.csv')  
```
We want to see how the data looks, we will print the first 5 records:  
``` python
print lyrics.head()
```
![head raw](https://github.com/alongalperin/sequence_generation/blob/master/images/head.JPG)  
  
We will check for how many missing values are there. The columns that important for us the most are: artist and lyrics.  
``` python
lyrics.apply(lambda x: sum(x.isnull()),axis=0)
```
![missing values](https://github.com/alongalperin/sequence_generation/blob/master/images/missing.JPG)  
  
We see that we have a lot of records with missing lyrics. We will drop these rows and check with how many records we remained:  
``` python
lyrics.dropna(subset=['lyrics'], inplace=True)
print('now we have %d records' %len(lyrics))
```
output: now we have 266,557 records  
  
Summary so far: We started with corpus of 380,000 songs. After we dropped songs that had no lyrics  
we have 266,557 songs.  
we can see that we have enoght songs lyrics.  
  
We saw that in the dataset there are songs of Eminem, Beyonce-Knowles and Arctic Monkeys.  
We will check if we have at least 50 songs for each artist:  
``` python
print ('Eminem has %d songs' % len(lyrics.loc[lyrics['artist'] == 'eminem']))
print ('Beyonce has %d songs' % len(lyrics.loc[lyrics['artist'] == 'beyonce-knowles']))
print ('Arctic-Monkeys has %d songs' % len(lyrics.loc[lyrics['artist'] == 'arctic-monkeys']))
```
Eminem has 578 songs  
Beyonce has 248 songs  
Arctic-Monkeys has 134 songs  
  
We can see that the lowest number of songs is the number of Arctic Monkeys songs, so we will take **130** songs from each artist.   
We will concat all the 3 artists lyrics to 1 dataset. 
``` python
eminem_songs = lyrics.loc[lyrics['artist'] == 'eminem'][:130]

beyonce_songs = lyrics.loc[lyrics['artist'] == 'beyonce-knowles'][:130]

arctic_monkeys_songs = lyrics.loc[lyrics['artist'] == 'arctic-monkeys'][:130]

total_songs = eminem_songs.append([beyonce_songs,arctic_monkeys_songs])
print('We have total of %d songs' % len(total_songs))
```
We have total of 390 songs  
In part B we will check the avg length of each song, we will do it in part B after stopwords removal.

# Part B: Building Classifier for Classifying Lyrics and artists  
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

``` python
word_punct_tokenizer = WordPunctTokenizer() # from nltk.tokenize
stop_words = stopwords.words('english') + list(string.punctuation)
ps = PorterStemmer()

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
![avg;ength](https://github.com/alongalperin/sequence_generation/blob/master/images/avg_length2.JPG)  
  
We shuffle the records in the dataset. So we have mix of artists in the train and test set
``` python
total_songs = total_songs.reindex(np.random.permutation(total_songs.index))
```
### Split the dataset to train and test sets
We split the dataset to 80% and 20% train and test set  
The data (x) is the lyrics and the target (y) is the artist name  

``` python
df_train, df_test = train_test_split(total_songs, test_size=0.2)

x_train = df_train['lyrics']
y_train = df_train['artist']

x_test = df_test['lyrics']
y_test = df_test['artist']  
```
  
| Train set | Test set |
|:------------- |:------------- |
| 312 records     | 78 records |
  
### Prepare the data for classification algorithms
__We will transform the data from domain of words(strings) to numric domain.__
The reason is that almost all of the ML algorithms we learned - can handle only numric input and cant handle string.  
The method that we are going to use called TFIDF (as learned in lesson 12).  
TFIDF will convert the domain from world of string and words - to world of **document frequency**. The number in  
the matrix will represent statistic that will reflect how important a word is to the songs collection.  
  
**Also**  the matrix will be sparse, that mean that it be effient in space and it won't have entries for values  
that are null in the frequency table.  
We won't define configurations for this step. Since we read in the internet that the basic configurations are working OK.  
The defualts are: use_idf = true,  smooth_idf  = true,  sublinear_tf  = false, norm = none  
``` python
TfidfVectorizer = TfidfVectorizer()

x_train = TfidfVectorizer.fit_transform(x_train)
x_test = TfidfVectorizer.transform(x_test)
```
  
Change target attribute from words to numric representation.  
In this step we will change the values in the target attribute from words (strings)  
to numric.  
Example: If we have target attribute of singers so we will change each singer name with a number  
arctic-monkeys -> 0  
beyonce-knowles -> 1  
eminem -> 2  
  
We will use LabelEncoder of NLTK to do this task (as learned in lesson 8).  
``` python
le = preprocessing.LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
```

## Train the Models:
### Random Forest:
We will maintain array called algorithm_results that will contain the result of the algorithms.  
We checked and saw that n_estimators=100 gets us the best results  
In the code below we create the model and fit it according to our train data
``` python
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
```
  
Predict and get the score of Random Forest:  
``` python
score = random_forest.score(x_test, y_test)
print ('the score is: %f'  %score)
```
output: the score is: 0.871795  
  
### LinearSVC:
This is variation of Support Vector Machine, considered to work well on text classification  
We searched the internet and saw that the most common configurations are penalty of type "l1" and tol=le-3 
``` python
svc = LinearSVC(penalty="l1", dual=False, tol=1e-3)
svc.fit(x_train, y_train)
```
Fit the model:
``` python
svc = LinearSVC(penalty="l1", dual=False, tol=1e-3)
svc.fit(x_train, y_train)
```
Predict and get the score of LinearSVC:  
``` python
score = svc.score(x_test, y_test)
print ('the score is: %f'  %score)
```

output: the score is: 0.769231  
  
### Naive Bayes using Gaussian function (GaussianNB)
Fit the model:  
``` python
gaussianNB = GaussianNB()
x_train_not_dense = x_train.toarray() # Naive Bayes not working on dense matries
gaussianNB.fit(x_train_not_dense, y_train)
```
Predict and get the score of Naive Bayes:  
``` python
x_test_not_dense = x_test.toarray() # Naive Bayes not working on dense matries

score = gaussianNB.score(x_test_not_dense, y_test)
print ('the score is: %f'  %score)
```
output: the score is: 0.602564  
  
### Nural Network (not Keras)
The configurations are after 2-3 tries with diffrent hidden_layer_sizes sizes  
Fit the model:  
``` python
mlpClassifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlpClassifier.fit(x_train, y_train)
```
Predict and get the score of LinearSVC:  
``` python
x_test_not_dense = x_test.toarray() #  mlpClassifier not working on dense matries so we enter the original x_test

score = mlpClassifier.score(x_test_not_dense, y_test)
print ('the score is: %f'  %score)
```
output: the score is: 0.807692  
  
### Plot results comparison:
We used python matplotlib to plot the following graph on data saved in algorithm_results array:  
The code for this plot is in the notebook  

![results](https://github.com/alongalperin/sequence_generation/blob/master/images/result_plot.JPG)  
  
We can see that **Random Forest** is leading with the best score.
  
# Part C: Songs Generating
In this part we will show the creation of songs only for Arctic-Monkeys. We did the same proccess  
for Eminem songs and Beyonce also.  
First we import all the packeges we need from Keras:
``` python
from keras.models import Sequential
from keras.layers import Activation,LSTM,Dense
from keras.optimizers import Adam
```
Create array of 50 songs  
``` python
ds = all_songs_original.loc[all_songs_original['artist'] == 'arctic-monkeys']
ds = ds['lyrics']
ds = ds [:50]
songs_lyrics_array = np.array(ds)
```
  
concat all lyrics to one text:  
  
``` python
txt=''
for ix in range(len(songs_lyrics_array)):
    try:
        txt+=songs_lyrics_array[ix]
    except:
        print ('cant read: ')
        print (songs_lyrics_array[ix])
```

The set of targets is stored in next_chars which is the next character after the window of 40. There will be lots of overlap in each window.

We are going to train our model to predict the next character, based on the previous 40 characters
``` python
vocab=list(set(txt))
char_ix={c:i for i,c in enumerate(vocab)}
ix_char={i:c for i,c in enumerate(vocab)}

maxlen=40
vocab_size=len(vocab)

sentences=[]
next_char=[]
for i in range(len(txt)-maxlen-1):
    sentences.append(txt[i:i+maxlen])
    next_char.append(txt[i+maxlen])
```

A 1 hot vector for a character, is a vector that is the size of the number of characters in the corpus.  
The index of the given character is set to 1, while all others are set to 0.  
``` python
X=np.zeros((len(sentences),maxlen,vocab_size))
y=np.zeros((len(sentences),vocab_size))
for ix in range(len(sentences)):
    y[ix,char_ix[next_char[ix]]]=1
    for iy in range(maxlen):
        X[ix,iy,char_ix[sentences[ix][iy]]]=1
```

The shape of the input is the window length of 1 hot vectors  
The number of LSTM units is 128  
Lastly we have dense layer with a softmax output which can predict the possible target character  
``` python
model=Sequential()
model.add(LSTM(128,input_shape=(maxlen,vocab_size)))
model.add(Dense(vocab_size))
model.add(Activation('softmax'))
model.summary()
model.compile(optimizer=Adam(lr=0.01),loss='categorical_crossentropy')
```
  
fit the model
``` python
model.fit(X,y,epochs=20,batch_size=128)
```
  
We create function that generate song using the Keras network we complied.  
We will create 130 songs for each of the 3 artists:  
``` python
df_songs_generated = pd.DataFrame(columns=['lyrics'])
counter = 0

while (counter < 130):
    generated_song = generate_song()
    df_songs_generated.loc[counter] = generated_song
    counter += 1
```
  
Save  the songs to csv  
``` python
df_songs_generated.to_csv('arctic-monkeys_song_generated.csv', sep=',')
```

# Part D: Classifying the Generated Songs
  
In this part will run the algorithm that gave us the maximum accuracy in part B on out generated songs.  
The algorithm that was the most accurate in Part B was Random Forest algorithm.  
  
The steps are:  
1. Read all the generated songs.  
2. Preform preprocess to the songs text, by using the same preproccess we did in part B.  
3. Create dataframe for lyrics (x_test) and dataframe for targer (y_test).  
4. Change the lyrics to numeric representation using tfidfVectorizer from part B.  
5. Change the target representation from singers names to number using LabelEncoder from part B.  
6. Run the algorithm and check for accuracy.  
  
step 1:
``` python
eminem_generated_songs = pd.read_csv('./eminem_song_generated.csv')
beyonce_generated_songs = pd.read_csv('./beyonce-knowles_song_generated.csv')
arctic_monkeys_generated_songs = pd.read_csv('./arctic-monkeys_song_generated.csv')

total_generated_songs = eminem_generated_songs.append([beyonce_generated_songs,arctic_monkeys_generated_songs])

# re-arrange the indices
total_generated_songs.reset_index(drop=True, inplace=True)
```
step 2:
``` python
for index, row in total_generated_songs.iterrows():
    total_generated_songs.loc[index, "lyrics"] = clean_and_prepare(row['lyrics'])
```
step 3:
``` python
x_test = total_generated_songs['lyrics']
y_test = total_generated_songs['artist']
```
step 4:
``` python
x_test = tfidfVectorizer.transform(x_test)
```
step 5:
``` python
y_test = le.transform(y_test)
```
step 6:
``` python
score = random_forest.score(x_test, y_test)
print ('the score is: %f'  %score)
```
output: the score is: 0.953333  
We reached to 95% precenetage of accuracy.  
We will print confusion matrix (code in the notebook)  
![alt text](https://github.com/alongalperin/sequence_generation/blob/master/images/confusion_martix2.jpg)  
  
**Conclusions:**  
**Data Collection:** We saw that Kaggle has many datasets that can be useful for learning purposes. This time Kaggle  
saved us a lot of time, we didn't have to build scraper for getting the lyrics from lyrics sites.  
We Saw that more data (songs) means more accuracy. This is the reason we choose artists with large number of songs.  
**Data Cleaning:** There is a connection between the quality of the cleaning process to the accuracy. Data cleaning is an  
important step that can really impact the learning and preformance of the learning algorithms.  
It is recommended to write function for data cleaning so that it can be called many time from every section in the program.  
**Data Classification:** Random Forest is the big winner for us. With good accuracy and not many tunning. Random Forest also was mentioned in the course as strong algorithm.  
**Sequence Generations:** Working with Keras, a very easy package for working with Nueral Network. In out case Keras was a like a wrapper for Theano.  
The step of Sequence Generations was the most challenging in all of the assignments in the course..  
The difficulties was to plan the input to the algorithm and to tune the parameters.  
**Data Classification of the Generated Songs:** The good results was realy surprising at first, but after thinkning about it and  
realizing that the songs was generated from "close" corpus - the great results maked sense.  
  
# Alon & Matan
