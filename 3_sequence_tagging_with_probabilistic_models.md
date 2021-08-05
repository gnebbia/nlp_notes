# Sequence Labeling

The problem here is: Given a sequence of tokens, infer the most probable
sequence of lables for these tokens.

Examples are:
* Part of Speech Tagging (POS Tagging)
* Named Entity Recognition (NER)

For POS tagging we can use tags from the "Universal Dependencies Project".
So here we want to understand for example who is the subject, what is the verb,
what is the object, and this is in practice part of speech tagging, so giving a
label to parts of the speech.

On the other hand, named entities recognition is related to the detection of any
real-world object which may have a proper name, like:
* Persons
* Organizations
* Locations
* ...

Sometimes they also usually include:
* Dates and Times
* Amounts
* Units


In order to solve the sequence labeling tasks either POS or NER we have
different approaches:

1. Rule-Based Models (example: EngCG tagger)
2. Separate label classifiers for each token (e.g., Naive Bayes, or Logistic
   Regression and so on used at each position)
3. Sequence Models (HMM, MEMM, CRF)
4. Neural Networks



## Neural Language Modeling

Neural networks are the state of the art for the previously discussed tasks such
as Language Modeling, POS tagging and NER.

Here we use RNN, specifically LSTM generally.


