# Language Models

Here we will learn about language models and models that work with sequences of
words such as:
* Part of Speech Tagging
* Named Entities Recognition

These concepts compose building blocks of many NLP applications.

Language modeling is about computing the probability for the next word, given
some previous words.

For example if we have a corpus of text where we see the string "this is the" 4
times and only once "this is the house", then the probability of getting "this
is the house" is:

p(house|this is the) = 1/4


In this case we are considering language modeling with 4-grams, anyway we will
see later how to analyze which n-grams are good for these tasks.

So whre do we use language modeling? Actually everywhere, for example:
* Suggestions in messengers
* Spelling correction
* Machine Translation
* Speech Recognition
* Handwriting Recognition
* ...

Language modeling also allow us to predict the probability of long sequences of
words, like, what is the probability of getting the phrase "I am here today" ?
We want to understand how to compute p("I am here today") also if this specific
phrase never occurs in text.

In order to do this, we can use the **chain rule**, and hence do:
p(w) = p(w_1)p(w_2|w_1)...p(w_k|w1...w_{k-1})

Anyway this is still too long, well we can use the **Markov Assumption** which
tell us:
p(w_i|w_1 ... w_i-1) = p(w_i|w_i-1+1 ... w_i-1

Markov assumption just tell us to not care about all the history, we hust have
to care about the last n -1 terms, and condition on them.
Let's see an example, let's say that we want to build a "Bigram Language Model",
in this case we have a toy corpus which is:

```text
This is the malt
That lay in the house that Jack built
```

at this point in order to compute p(this is the house) we can do:
p(this is the house) = p(this)p(is|this)p(the|is)p(house|the)

and from the toy corpus we can deduce that:
p(this) = 1/12
p(is|this) = 1
p(the|is) = 1
p(house|the) = 1/2


As w can notice, we can compute probabilities for previously unseen cases.

We have two problems at this point:
1. the initial word in our corpus is always either "This" or "That" we can use
   this information to improve our model, so that both words should be assigned
   a probability of 1/2
2. The summ of all the probabilities does not sum up to 1, so in order to do
   this, we have to split probabilities values among all the possible
   combinations

So here we are, we can have the definition of **Bigram Language Model**

p(w) = \product_{i=1}{k+1} p (w_i | w_{i-1}

the only difference with an n-gram language model is that we condition with a
longer history, specifically with a history long *n*.


Language model are useful to generate text of data, we generally evaluate the
model on the test set, and use the **perplexity** metric, which if lower it will
be the better.
Perplexity is just likelihood to the power of (-1/N) where N is the length of
text corpus, so all words concatenated.

We may have problems wth the current definition of perplexity, because never
encountered words may give an infinite perplexity, so what we do, is to build a
vocabulary of words (e.g., by word frequencies) and then substitute OOV  (out of
vocabulary) words by <UNK> both in trin and test and then we count as usual 
for all tokens.

Anyway althoug we may have solved the problem for unknown words, we still may
have infinite perplexity on bigrams which are not present in the corpus, in
order to fix this we can:

* Laplactian Smoothing
* Katz Backoff
* Interpolation Smoothing
* Absolute Discounting
* Kneser-Ney Smoothing

Again, these are all methods to give more probabilities to infrequent n-grams
and removing these probabilities to frequent n-grams.

### Laplacian Smoothing
We can use laplacian smoothing, so the idea is to pull some probability from
frequent bigrams to infrequent ones:
* Just adding 1 to the counts (**add-one smoothing**)
* Or tune a parameter (**add-k smoothing**), we can tune k on test data

### Interpolation Smoothing
It uses a mixture of several n-gram models, and assigns to each n-gram model a
weight. Then the weights are optimized on a test (dev) set. Optionally these
weights can also depend on the context.

### Katz Backoff
Here we basically say, try at first with long n-grams, e.g., if we are working
with 3-grams we start evaluating all 3-grams, but if a specific 3-grams is not
there, backoff to 2-grams and if this is still not there, try 1-grams.

In the Katz Backoff we also have to choose p tilde and alpha probabilities to
ensure normalization.

### Absolute Discounting
This technique tells us that if we subtract a specific constant on test set we
should have more or less the same count of the training set, so the two counts
should be correlated.

### Kneser-Ney Smoothing
It takes into account the context of a word, like how many contexts can the word
appear in? 
For example the word Kong may be very popular but only appears after Hong
generally while the word "malt" is not very popular but may appear in many
contexts.

Notice that n-grams models + Kneser-Ney smoothing is a strong baseline in
Language Modeling.

Also with neural networks model it is not so easy to beat this baseline.


