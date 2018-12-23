## Intro to NLP


We can say that there are three main groups of approaches to NLP problems:

1. Rule Based Methods
    * Regular Expressions
    * Context-Free Grammars
    * ...
2. Machine Learning Methods
    * Likelihood Maximization
    * Linear Classifiers
    * ...
3. Deep Learning Methods
    * Recurrent Neural Networks
    * Convolutional Neural Networks
    * ...

Rule based approach are generally very accurate but have a low recall in many
contexts for natural language and are not able to generalize on unseen elements.

Machine learning model are able to generalize but need a feature engineering
phase and performance of the model will depend on the choice of features.

Deep learning models do not need a feature engineering phase, but generally they
need big corpus of documents.

It is useful to study both traditional (rule based + machine learning) and deep 
learning approaches in order to be proficient in NLP. Indeed although deep 
learning techniques seem to perform better in most application, the knowledge
about traditional methods is still useful for some problems (e.g., POS tagging)
and will also help us tune and improve deep learning models on all the other
applications.

#### Linguistic Knowledge in NLP
[IMAGE] pyramid

Given a sentence in a language, we must remember that there are different stages
of analysis for the same sentence:

* **Morphology**, different forms of words, such as parts of speech,
  different genders, tenses, cases and so on, this is something related to
  single words in a sentence
* **Syntax**, this takes into account the different relations between words in
    the sentence, for example we may have subjects and then objects and so on
* **Semantics**, if morphology and syntax are right, sentence are grammatically
    correct but they may not make sense. Semantics is about the meaning of parts
    of the sentence
* **Pragmatics**, this is an high level abstraction, putting together the
    semantic meanings of the sentence


For what concerns linguistics there are different relationships between words,
some examples are:
* Hyponym and Hypernym (one is a specific type of an object, like "apple" and the
    other is an abstraction, like "fruit")
* Meronyms, this is the part-whole relationships, like wheel and car have a
    meronym relationship

We can use in our applications this knowledge through external resources which
is product of the work of linguists, resources like *WordNet* or *BabelNet* are
indeed very useful.

So what we have to take home from this subsection is that there is some useful
linguistics knowledge that could be useful in order to improve the performances
of our future models.

