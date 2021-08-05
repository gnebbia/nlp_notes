# Text Classification

We can apply classification and regression problems to textual data.
Classification problems to textual data are much more common so we will keep our
discussion on classification.
Examples of text classifications are:
* Spam filters
* Sentiment Analysis
* ...
* Anywhere we want to apply a label to a set of sentences/words


Let's start by talking about text in an NLP context, text can be thought as a
sequence. It can be a sequence of different things, depending on how we want to
analyze it, indeed it can be analyzed as:
* characters (very low level representation)
* words
* phrases and named entities
* sentences
* paragraphs (high levl representation)
* ... 

Commonly we represent text with words, a word is a meaningful sequence of
characters. 

A first problem could be, how do we identify words, which are the boundaries?
In English words are separated by whitespaces or punctuation.

The process of splitting an input sequence into meaningful semantic unit that
can be used for further processing is called "tokenization".
Depending on our analysis a token can be a word, a paragraph or a sentence and
so on.

Once we have our tokens, we have to associate the same variation of a token to
the same token, this process is called **"normalization"** for example:

wolf, wolves should be wolf
leaf, leaves should be leaf
running, ran, run should be run

and so on, in order to do this we have two approaches:
* Stemming, application of simple rules/heuristics, this process is generally
    very useful at removing and replacing suffixes to get to the root form of
    the word, which is called **stem**
* Lemmatization, much more complex process to be able to induce ran to run or
    ate to eat and so on, so simple rules here are not enough, so the need for a
    vocabulary/database is needed with a morphological analysis, this process
    returns the base or dictionary form of a word, which is known as the
    **lemma**

A famous stemming example is the Porter's stemmer, which use 5 heuristic phases
of word reductions, applied sequentially.
The problem with stemming is that not all the words can be reduced with a mere
application of the rules.

A famouse lemmatization example is provided by WordNet lemmatizer which use he
WordNet database to lookup lemmas.
The problem with lemmatization is that not all forms are reduced.

Further normalization other then stemming and lemmatizing is to use take into
account the following things:
* capital letters, be careful when they change the meaning
* acronyms, they can be reduced to the same word if we use regexes, but this
    becomes a hard problem (e.g., e.t.a., E.T.A., eta)

To sum up:
* We can think of text as a sequence of tokens
* Tokenization is a process of extracting those tokens
* We can normalize our tokens using **stemming** or **lemmatization**, we should
    try both and see which works best usually
* We can also further normalize casing and acronyms
* Tokens will be transformed into features for a model


## Transforming tokens into features

How can we transform tokens into features? There are different ways, one of
these is the Bag of Words (BOW), which count the occurrences of a particular
token in our text.
For each token we will have a feature column, and this process is called text
vectorization, while each row will represent a document/review/title we are
analyzing.
So in general a BOW model will involve a **text vectorization** which will
produce a matrix which has size numberofdocuments x totalnumberofwordsinalldocuments.

We have two problems with the BOW approach:
* we loose word order, hence the name "bag of words"
* counters are not normalized

Let's see how to solve these problems, we can solve partially the word ordering
problem by using **n-grams** (also called **shingles**), so instead of having
columns only for single words, we also add columns for the couples of words
found close, so if a review contains "good restaurant" we will create a column
called "good", one called "restaurant" and another one called "good restaurant".
The problem related to this approach is that we will have too many features.

Notice that by 1-grams we mean a list composed by 1 token (again, a token can be
a char, a word, a sentence and so on).
A 2-grams is composed by a pair of tokens, while a 3-grams is composed by three
tokens and so on, we could go on infinitely.
In order to overcome this problem, we have to **remove** some n-grams.

Generally what we remove is:
* High frequency n-grams, which for example may be constituted by articles, and
    grammatical objects which do not give significant meaning, such as "the",
    "a", "an" and so on. These words are called "stop-words" and do not allow us
    to discriminate text
* Low frequency n-grams, these may be represented by for example typos, so we 
    also don't need them since we may overfit
* Medium frequency n-grams: these are the good n-grams


The thing is that there are a lot of medium frequency of n-grams, so we want
to understand what is the impact of each n-gram and give a ranking, in order to
do this, we have to introduce two concepts:

* term frequency (TF)
* inverse document frequency (IDF)

### Term frequency (TF)

We can talk about $tf(t,d)$ as the frequency for term (or n-gram) $t$ in
document $d$.

We can count term frequency in different ways:
* Binary weighting scheme (0 or 1, depending on the presence of the token in a
    document)
* Raw count (count of the token in a document)
* Term Frequency (count divided by the total number of words)
* Log Normalization $1 + log(f_t, d)$


### Inverse Document Frequency (IDF)

$idf(t,D) = log \frac{N}{|d\in D: t\in d|}$

where N or |D| is the total number of documents in our corpus, and the
denominator is the number of documents where the term t appears.

Notice that:
$tfidf(t,d,D) = tf(t,d) * idf(t,D)

A high weight in TF-IDF is reached by a high term frequency (in the given
document) and a low document frequency of the term in the whole collection of
documents.

### Better BOW

A better Bag of words model consists of a text vectorization which replaces
simple counts with TF-IDF values, and then we can normalize these values
row-wise (for example with a simple normalization of by dividing by the L2
norm).


### Linear Models for Sentiment Analysis

We will talk about text classification models.
In order to tackle text classification problems, as with any other
classification problems, we need a labeled dataset.

In this case, IMDB reviews or ebay reviews with star could represent a dataset,
since we have a set of words and some stars which represent how positive
something is.


Generally text vectorization as we may notice will give us matrices with a huge
amount of columns, and generally algorithms such as decision trees or/and
gradient boosted trees do not work well, since thay have to do an exhaustive
search over all features for the next tree split.

Good algorithms in text classification problems are:
* Linear Models
* Naive Bayes

A common linear model is the logistic regression, which can handle sparse data,
it is fast to train and weights can be interpreted.

Actually all linear models have the advantage of being interpretable.
If we use logistic regression, we can look at weights to see how relevant each
word was and which were the top positive and which the top negative.

So how can we also make it better:

* We can play around with tokenization, for example adding particular sequences
    like :) or !!! can help, we may consider to add these to the set of valid
    tokens
* Try to normalize tokens, by using stemming or lemmatization
* Try different models, like SVM, Naive Bayes, and other classification
    algorithms which can handle sparse features
* Throw BOW away and use Deep Learning, deep learning algorithms achieved on a
    famous dataset in 2016 2.5% more of accuracy with respect to BOW model with
    2-grams.

To sum up:
* BOW and linear models actually work for texts
* The accuracy gain from deep learning models is not mind blowing for sentiment
    classification, but it's still there in case it is important to also gain 1%
    of accuracy



### Feature Hashing

Feature Hashing is a feature engineering technique used in any case in which we
have a large corpora and hence the number of features would be huge. For example
we could have a corpora with 40 milion of unique words, this means that each row
of the dataaset will have at least 40 milions of columns for each document of
the corpus.

In order to solve this problem a feature engineering technique called "hash
trick" or "feature hashing" is used, this helps us since allows to fix the
number of features to a specific number.

For example in our 40 milion unique words corpus, what would happen with the
hashing trick is that, we take a word, we apply to it a hash function and then
take the modulo with a huge number like 2^22 or something like that, well now
the number of features is fixed to 2^22.
It turns out that this mechanism does not hurt the quality of the model.
So the feature hashing formula to recap is:

$\phi(h) = hash(x) % 2^b$

We can also use the so called **personalized tokens trick** which before hashing
a token what it does is adding to that string some additional info, like in a
spam filtering system it may be the username of the mail received, so with the
personalized feature hashing we have:

$\phi_p(h) = hash(u + "_" + token) % 2^b$

Persnalized tokens have generally better performance with respect to a baseline
BOW approach this is because it captures also other features inside its process.

To sum up:
* We can use feature hashing, especially with lots of features, feature hashing
    is a technique related to feature engineering and a way to represent a huge
    number of features as a fixed number of features
* Personalized features constitute a nice trick to have better performances
* Linear models over bag of words scale well for production



## Neural Networks for Text

A BOW model can also be seen as a sum of sparse one-hot-encoded vectors.
Since bag of words represent a sparse model, this is not a good representation
for Neural networks.
Neural networks indeed work well with dense representations, so in order to 
transform text into features for neural networks, a more efficient way 
is to use the **word2vec** model.

Word2Vec model assigns vector of numbers to each word, in an unsupervised way,
so every word will get its values automatically, we will study this in detail
later.

We will see how word2vec embeddings work, anyway the main property is that:
*Words that have similar context tend to have collinear vectors*.
This is a very nice property.

In order to represent a document/review we can sum all the vectors representing
the words, and this will give us a unique vector which is a result of the entire
sum.

Anyway a better way is to use **convolutions**, for example in order to include
2-grams we could apply a convolutional filter on two rows, which is a
convolutional filter on two words of the document.

So Word2Vec vectors for similar words are similar in terms of cosine distance
(similar to dot product).

Convolutions are very cool, since we do not have anymore to look at all the
pairs built with 2-grams or lists of 3-grams and so on, we can just apply some
convolutional filters, and we won't have all those columns anymore.

These are 1D convolutions, since text is 1-Directions, we can use 3-grams, or
4-grams and so on. We generally indeed use many n-grams.

Now the results of convolutions are variable sized vectors, and the length
depends on the input length, so clearly these canno be used for classifications,
so what we do is applying "max pooling" so getting the maximum value of the
vector, in this way we lose information about the ordering of the tokens, but we
still can capture the meaning, or the fact that a specific meaning was in the
text.

An example scenario indeed could be represented by an architecture using
3,4,5-grams windows with 100 different filters each, so in this case we get a
vector with a length of 300 after pooling is applied for each filter, in this
way we have transformed a sentence/text/review into a fixed size vector which
can be used as the feature list in our problem, so given as input to a neural
network.
This word2vec embedding works very well, and generally better than BOW.

So until now we have seen different ways to text embeddings:
1. BOW
2. BOW with hashed features
3. For DNN we use Word2Vec

To sum up:
We can average pre traine word2vec vectores for yout text
We can have better results by applying convolutions


### Neural Network Learning from Characters

Notice that we can use convolutional networks on top of characters, this
technique is called **learning from scratch**, this works best for large
datasets, where it beats classical approaches like BOW, sometimes this approach
also beats DNN like LSTM that work on word level.


We should learn from scratch whenever we have very huge datasets.











