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
  
We shuffle the records in the dataset. So we have mix of artists in the train and test set
```
total_songs = total_songs.reindex(np.random.permutation(total_songs.index))
```
### Split the dataset to train and test sets
We split the dataset to 80% and 20% train and test set  
The data (x) is the lyrics and the target (y) is the artist name  

```
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
```
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
```
le = preprocessing.LabelEncoder()

y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)
```

## Train the Models:
### Random Forest:
We will maintain array called algorithm_results that will contain the result of the algorithms.  
We checked and saw that n_estimators=100 gets us the best results  
In the code below we create the model and fit it according to our train data
```
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(x_train, y_train)
```
  
Predict and get the score of Random Forest:  
```
score = random_forest.score(x_test, y_test)
print ('the score is: %f'  %score)
```
output: the score is: 0.871795  
  
### LinearSVC:
This is variation of Support Vector Machine, considered to work well on text classification  
We searched the internet and saw that the most common configurations are penalty of type "l1" and tol=le-3 
```
svc = LinearSVC(penalty="l1", dual=False, tol=1e-3)
svc.fit(x_train, y_train)
```
Fit the model:
```
svc = LinearSVC(penalty="l1", dual=False, tol=1e-3)
svc.fit(x_train, y_train)
```
Predict and get the score of LinearSVC:  
```
score = svc.score(x_test, y_test)
print ('the score is: %f'  %score)
```

output: the score is: 0.769231  
  
### Naive Bayes using Gaussian function (GaussianNB)
Fit the model:  
```
gaussianNB = GaussianNB()
x_train_not_dense = x_train.toarray() # Naive Bayes not working on dense matries
gaussianNB.fit(x_train_not_dense, y_train)
```
Predict and get the score of Naive Bayes:  
```
x_test_not_dense = x_test.toarray() # Naive Bayes not working on dense matries

score = gaussianNB.score(x_test_not_dense, y_test)
print ('the score is: %f'  %score)
```
output: the score is: 0.602564  
  
### Nural Network (not Keras)
The configurations are after 2-3 tries with diffrent hidden_layer_sizes sizes  
Fit the model:  
```
mlpClassifier = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
mlpClassifier.fit(x_train, y_train)
```
Predict and get the score of LinearSVC:  
```
x_test_not_dense = x_test.toarray() #  mlpClassifier not working on dense matries so we enter the original x_test

score = mlpClassifier.score(x_test_not_dense, y_test)
print ('the score is: %f'  %score)
```
output: the score is: 0.807692  
  
### Plot results comparison:
We used python matplotlib to plot the following graph on data saved in algorithm_results array:  
The code for this plot is in the notebook  

![results](https://github.com/alongalperin/sequence_generation/blob/master/images/result_plot.JPG)  
  
