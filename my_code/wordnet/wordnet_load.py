from nltk.corpus import wordnet as wn
#import nltk
#nltk.download('omw-1.4')
#nltk.download('wordnet')
from nltk.corpus.reader import Synset


def wordnet(word):
    relatedWords = wn.synsets(word)
    return [word._name.split('.',2)[0] for word in relatedWords]

#print(wordnet('badger'))

