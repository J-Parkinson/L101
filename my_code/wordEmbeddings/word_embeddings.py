import numpy.random as nprand
from my_code.word2vec.word2vec_load import getWord2Vector
from my_code.wordnet.wordnet_load import wordnet
from my_code.conceptnet.conceptnet_load import conceptnet
import pickle

def checkInOrder(wordList):
    return next((item for item in wordList if item is not None), None)

def getConceptNet(word):
    allWords = conceptnet(word)
    allVectors = [getWord2Vector(wo) for wo in allWords]
    return checkInOrder(allVectors)

def getWordNet(word):
    allWords = wordnet(word)
    allVectors = [getWord2Vector(wo) for wo in allWords]
    return checkInOrder(allVectors)

def convertWordToWordEmbedding(word, customWords):
    if word in customWords:
        return customWords[word]
    wordVec = getWord2Vector(word.lower())
    if wordVec is not None:
        customWords[word] = wordVec
        return wordVec
    wordNet = getWordNet(word.lower())
    if wordNet is not None:
        customWords[word] = wordNet
        return wordNet
    conceptNet = getConceptNet(word.lower())
    if conceptNet is not None:
        customWords[word] = conceptNet
        return conceptNet
    customWords[word] = (-1 + (nprand.rand(300) * 2))
    return customWords[word]

def convertTextToWordEmbeddings(sentenceSplit, customWords):
    returnVal = [convertWordToWordEmbedding(word, customWords) for word in sentenceSplit]
    return returnVal

#Need to split into words first
def convertDataToWordVectors(data, customWords={}, name=None, save=False, columnName='sequence'):
    print(f'Length of {name}:', len(data['sequence']))
    data[columnName] = data['sequence'].apply(lambda x: convertTextToWordEmbeddings(x, customWords))
    if save:
        with open(name, 'wb') as loc:
            pickle.dump(customWords, loc)
    return data, customWords