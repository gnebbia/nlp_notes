# Natural Language Processing with NLTK

Natural language processing (NLP) is an area of computer science and artificial
intelligence concerned with the interactions between computers and human
(natural) languages, in particular how to program computers to process and
analyze large amounts of natural language data.

Challenges in natural language processing frequently involve speech recognition,
natural language understanding, and natural language generation. 

Natural Language Processing can be viewed as the application of computational
linguistics.

A thourough theoritical reference can be:
`Speech and Language Processing by Daniel Jurafsky and James H. Martin`


## EDA with Text

### Getting text from websites

```python
import urllib2
import nltk

# with any framework we just need the response html text
response = urllib2.urlopen('http://python.org')
html - response.read()

# removes all tags and useless stuff
clean = nltk.clean_html(html)

tokens = [tok for tok in clean.split()]
print(tokens)
```

### Getting the frequency distribution of terms


```python
import nltk

# Once we have the tokens
# ... 
freq_dist = nltk.FreqDist(tokens)
for k,v in freq_dist.items():
    print(str(k) + ':' + str(v))
```

We can plot the distribbution of words with:

```python
freq_dist.plot(50, cumulative=False)
```

In EDA, we should get rid of the useless words such as:
`the`,`a`,`of`,`for`,`=`, and so on, this is noise in text and is generally 
referred to as **stop words**. These stop words are considered noise because 
they are not discriminative enough to be informative.
People generally removes stopwords.

We should have a list of stopwords for the language we are analyzing, on
the web we can find such lists and then we can clean our text from noise by
doing something like:


```python
stopwords = [word.strip().lower() for word in open("english-stop-words.txt")]
clean_tokens = [to for tok in tokens if len(tok.lower())>1 and (tok.lower() not in stopwords)]
```

Now we can plot the distribution without stop words:

```python
freq_dist = nltk.FreqDist(clean_tokens)
freq_dist.plot(50, cumulative=False)
```

Another human friendly way to visualize distributions of text is to use **word
clouds**, which help in visualizing the topics in a large amount of unstructured
text.



## Preprocessing in Detail

In the previous step we looked at preprocessing in a very inaccurate and fast
paced manner, now we will go into every step of the so called 'text wrangling'.
Text wrangling is related to all the pre-processing and actions/transformations
we do before we have a readable and formatted text from raw data.
This process involves the following steps:

* Data Munging
* Text Cleansing
* Specific Preprocessing
* Tokenization
* Stemming
* Lemmatization
* Stop Word Removal

Notice that there are no clear boundaries between the terms **data munging**,
**text cleansing** and **data wrangling**, they can be used interchangeably in a
similar context.

### Sentence Splitter

To split paragraphs or corpus of text into sentences, by default the dot '.' is
used.

In nltk we can use a sentence boundary detection algorithm which is builtin
and do something like this:
```python
inputstring =''' a large text. another phrase. another sentence again!!!'''
from nltk.tokenize import sent_tokenize
all_sent = sent_tokenize(inputstring)
print(all_sent)
```

### Tokenization

A word or token is the minimal unit that a machine can understand and process.
Any text string which we want to process with NLP should go through the
tokenization process which given as input a text returns the tokens.

The complexity of tokenization varies according to the need we have with respect
to a specific problem and also the considered language.

For example if we consider tokenization for chinese or japanese language, it can
be a very difficult task.

Let's see some examples of tokenization done with nltk:

```python
from nltk.tokenize import word_tokenize
from nltk.tokenize import regexp_tokenize, wordpunct_tokenize, blankline_tokenize

s = 'Hello all! this is l33t language'


# The most basic tokenizer is the split python method
# basically it uses whitespace/s to extract tokens
print(s.split()) # returns ['Hello', 'all!', 'this', 'is', 'l33t', 'language']

# a more robust version of tokenization is given by the word_tokenize nltk method
word_tokenize(s) # returns 'Hello', 'all', '!', 'this', 'is', 'l33t','language']

regexp_tokenize(s, pattern='\w+') # returns  ['Hello', 'all', 'this', 'is', 'l33t', 'language']
regexp_tokenize(s, pattern='\d+') # returns  ['33']

wordpunct_tokenize(s) # returns ['Hello', 'all', '!', 'this', 'is', 'l33t', 'language']
blankline_tokenize(s) # returns ['Hello all! this is l33t language']
```
Consider that the two most common tokenizers are:
1. work_tokenize
2. regexp_tokenize (which can actually be used to derive the other tokenizers or
   more custom tokenizers)

In the basic/common scenario stick to `work_tokenize` method of nltk.

### Stemming

In linguistic morphology and information retrieval, stemming is the process of
reducing inflected (or sometimes derived) words to their word stem, base or root
form—generally a written word form. [Wikipedia]

Basically stemming is used to take into account the variations of the same
semantic word meaning. For example we want that:
* `drink`
* `drinks`
* `drank`
* `drinking`
* `drink.*`
be processed as the same word.

Stemming is a simple process and it is generally used just to eliminate all the
grammatical variations provided by the same concept and just get the root word.
For more complex tasks, stemming can be not enough and e should use
**lemmatization**. Lemmatization is a more robust process which combines
grammatical variations to the root of a word (as we will see in the next section).


TODO: Check Differences between Lemmatization and Stemming


Let's see a example of stemming in nltk:
```python
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem.Snowball import SnowballStemmer

pst = PorterStemmer()
lst = LancasterStemmer()
lst.stem("drinks") #returns "drink"
pst.stem("running") #returns "run"
```

Consider that basic stemmers just looking for s/es or ed and few other rules
achieves an accuracy of approximately 70%.
Porter stemmer uses a bunch of rules which achieves a very good accuracy
generally.

The rule of thumb here is:
* There are many stemmers we can find online for different languages
* When using English, PorterStemmer is good enough
* Snowball stemmers are also good and there are versions for many languages
    (including Italian)

Consider that stemming is not always used, it really depends on the application,
for example when using NLP taggers, like Part of Speech tagger also referred to
as (POS) or dependency parsers (NER) we want to avoid stemming since this could
lead to different output results.


### Lemmatization
Lemmatization may appear similar to stemming, but it is a much more complex
process, where grammar of a specific language is taken into account to infer the
meaning behind a word.
To make an example let's consider english irregular verbs, with stemming we are
able to induce the root of the word if we get a word like `running` by applying
simple rules, but how can we get the root of the word `ran` `ate` or `laid` or `were`
or many other irregular verbs?
This is where lemmatization comes up; consider that lemmatization uses context
and part of speech to be able to determine the inflected form of the word also
applying different normalization rules.
In this context the root of the word is also called **lemma**.
Let's see an example in python:

```python
from nltk.stem import WordNetLemmatizer
wlem = WordNetLemmatizer()
wlem.lemmatize("ran", v) # shows run
```
Notice that we should also specify to the lemmatization system what part of the
speech we are referring to in order to get the right word, because some of the
words can be used as verbs or as nouns or as adjectives and so on. Anyway there
are algorithms which help us in detecting and identifying the various parts of
the speech.


### Stop Word Removal

The process of stop word removal consists in removing the words that occur
commonly across all the documents in the corpus. 
Generally in English and may other languages at least articles and pronouns are
considered stop words. Also because these stop words have no significance in
some NLP tasks like information retrieval or classification.

Stop words can also be induced by looking at words which appear frequently in
the entire corpus of documents, more specifically we should look at the word-s
document frequency, for example taking each word in the entire corpus and then
check which words appear in all documents and with which frequency.


The software framework nltk already comes with lists of stop words for many
languages.
Let's see an example of stop word removal with nltk:

```python
from nltk.corpus import stopwords
stoplist = stopwords.words('english')
text = ' this is just some random string of text'
cleanwordlist = [w for w in text.split() if w not in stoplist]
# cleanwordlist is equal to ['random', 'string', 'text'] 
```

### Rare word removal

Another form of noise in text is represented by very rare words like names or
brands or names of specific products.
Or for example very short words or very long words are also generally undesired.

We can remove the 50 rarer words in nltk with the following snippet of code:

```python
freq_dist = nltk.FreqDist(token)
rarewords = freq_dist.keys()[-50:]
after_rare_words = [ word for word in token not in rarewords ]
```

### Spell correction

Generally we do not need to use spell checking for NLP applications, anyway in
some cases it is useful to use a basic spellcheck.
A basic spellchecker can be designed by using a dictionary lookup. One of the
most common algorithms used in this applications is the `edit-distance`
algorithm. 

let's see an example in nltk:
```python
from nltk.metrics import edit_distance
edit_distance("mext", "next")
```

## Part of Speech (POS) Tagging
By part of speech we refer to grammatical components of language, like adverbs,
adjectives, verbs, pronouns and so on.
Making a machine detect POS was in the past a very hard problem, with the
current state of the art we have reached approximately an accuracy of 97%, but
there is still a lot of research in this area.


When talking about POS, the common POS notation used is called **Penn
Treebank**, we can look on the web what every tag means, but examples for demo
purposes may be `NNP`:Proper noun, singular, or `RB`: Adverb, or again
`RP`:Particle and so on.


Let's see an example of tokenization with nltk which comes with a pretrained POS
tagger:
```python
import nltk
from nltk import word_tokenize
s = 'I was eating an apple'

print(nltk.pos_tag(word_tokenize(s)))
# prints this:
# [('I', 'PRP'), ('am', 'VBP'), ('eating', 'VBG'), ('an', 'DT'), ('apple', 'NN')]
```

This POS tagger is internally using the `maxent` classifier to predict to which
class a specific word belongs to.

Another exaample may be the one of giving us all the nouns in a given sentence:
```python
tagged = nltk.pos_tag(word_tokenize(s))
allnoun = [word for word,pos in tagged if pos in ['NN','NNP']]
# returns all the nouns in text in the allnoun variable
```

Notice that if we have to deal with POS tagging, it is better to defer the stop
word removal phase since this can confuse the POS tagging algorithm.

Another common tagger used with nltk is the Stanford tagger, it may be useful to
also check out this.

To summarize, there are generally two ways to achieve tagging tasks in nltk:
1. using a pre-trained tagger (also the nltk one)
2. building and training a customized tagger, we can start from datasets found
   online where people spent a lot of time tagging corpus of text manually

If we want to train our own POS tagger, we have to manually tag corpus of
documents related to our domain, in general indeed tagging requires domain
experts knowledge.

In the domain of Machine Learning the POS tagging tasks is viewed as a sequence
labeling problems or classification problems.


### Evaluating Taggers

Let's start to see how to evaluate taggers which we may eventually build, for
example we can build our own tagger and then evaluate them by using some corpus
of text included in nltk such as the `brown` corpus.
Indeed generally when evaluating taggers we use these common corpus of
texts and use default taggers as comparison elements.

A default tagger is a tagger which predicts always the same tag for any token in
the corpus of text, and is a good comparison element when we have to evaluate
our taggers.


Let's see how to evaluate the accuracy of a default tagger which predicts always
'NN' for every word in nltk:

```python
brown_tagged_sents = brown.tagged_sents(categories='news')
default_tagger = nltk.DefaultTagger('NN')
print(default_tagger.evaluate(brown_tagged_sents)
# This gives 0.130894842572 as accuracy, of course accuracies of default taggers
# are quite poor
```

#### Sequential Taggers

##### N-gram Tagger
An n-gram tagger is a type of SequentialTagger, where the tagger takes the
previous **n** words in the context, to predict the POS tag for a specific
token.
Variations of the n-gram general approach are in nltk represented by
UnigramTagger, BigramTagger, TrigramTagger.

Let's see how to test and evaluate these taggers:

```python
from nltk.tag import UnigramTagger
from nltk.tag import DefaultTagger
from nltk.tag import BigramTagger
from nltk.tag import TrigramTagger

# we are dividing the data into a test and train to evaluate our taggers.
train_data = brown_tagged_sents[:int(len(brown_tagged_sents) * 0.9)]
test_data = brown_tagged_sents[int(len(brown_tagged_sents) * 0.9):]

unigram_tagger = UnigramTagger(train_data,backoff=default_tagger)
print (unigram_tagger.evaluate(test_data))
# 0.826195866853
bigram_tagger = BigramTagger(train_data, backoff=unigram_tagger)
print(bigram_tagger.evaluate(test_data))
# 0.835300351655
trigram_tagger = TrigramTagger(train_data,backoff=bigram_tagger)
print(trigram_tagger.evaluate(test_data))
# 0.83327713281
```

Unigram just considers the conditional frequency of tags and predicts the most
frequent tag for every given token. The bigram_tagger parameter will consider
both the current word and the previous one as a tuple, while with a similar process
a trigram will consider the current word and the previous two words.


Of course although the trigram may perform better, the coverage will be less,
because we have to find a tuple with three elements which have the same values.
This is way it is important the concept of BackoffTagger, which is a strategy
which allow us to attempt with a trigram tagger, if the lookup is unsuccessful
try with a bigram and again if the lookup fails try with unigram and if this
fails again try with the DeafultTagger using it with a Noun, so NN tag.

TODO: Explain better this process, i think these taggers are not using ML, but
just lookup tables as far as I understood ).


##### Regex Tagger
This is another class of sequential tagger based on regular expressions. Here
instead of looking for the exact word as with N-gram taggers we look for matches
with respect to a regex.
Let's see an example with nltk:

```python
from nltk.tag.sequential import RegexpTagger
regexp_tagger = RegexpTagger(
    [( r'^-?[0-9]+(.[0-9]+)?$', 'CD'), # cardinal numbers
    ( r'(The|the|A|a|An|an)$', 'AT'), # articles
    ( r'.*able$', 'JJ'), # adjectives
    ( r'.*ness$', 'NN'), # nouns formed from adj
    ( r'.*ly$', 'RB'), # adverbs
    ( r'.*s$', 'NNS'), # plural nouns
    ( r'.*ing$', 'VBG'), # gerunds
    (r'.*ed$', 'VBD'), # past tense verbs
    (r'.*', 'NN') # nouns (default)
    ])

print(regexp_tagger.evaluate(test_data))
# 0.303627342358
```

We can use easily regex taggers to tag date and money for example and things like that.

##### Brill Tagger

The Brill tagger is a transformation based tagger, where the idea is to start
with a guess for the given tag and, in next iteration, go back and fix the
errors based on the next set of rules the tagger learned. It's a supervised way
of tagging, but unlike N-gram tagging where we count the N-gram patterns in
training data, we look for transformation rules.

Let's take an example, a Brill tagger could say that if the tagger starts with a
Unigram/Bigram tagger with an acceptable accuracy, instead of looking for a
trigram tuple it will be looking for rules based on tags, position and the word
itself.

For example, an example of rule could be replace NN with VB when the previous
word is of tag TO.

The brill tagger can outperform some of the N-gram taggers. The advice is of
course always to not overfit the tagger on the training set.


### Named Entity Recognition (NER)

One of the other most common labeling probems is finding entities in the text.
Typically NER constitutes name, location, and organizations. There are NER
systems that tag more entities than just three of these.
