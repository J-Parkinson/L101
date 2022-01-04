import pandas as pd

from my_code.CNN_LSTM.convertToPyTorchDataset import splitIntoSentencesAndPad, biasesForLossFunctionUnbalanced
from my_code.helpers.datasets import Datasets
from my_code.load_and_tokenize.preprocess_genspam import loadGenspam
from my_code.load_and_tokenize.preprocess_lingspam import loadLingspam
from my_code.load_and_tokenize.preprocess_sms_spam import loadSMSSpam
import numpy as np

from my_code.wordEmbeddings.word_embeddings import convertDataToWordVectors

def convertDataToIndices(*data):
    allData = pd.concat(data)
    allWords = np.unique(np.array([np.array(x) for x in allData['sequence'].to_numpy()]).flatten())
    allWordsByIndex = {word: i for i,word in enumerate(allWords)}
    for datum in data:
        datum['sequence'] = datum['sequence'].apply(lambda x: np.array(list(map(lambda y: allWordsByIndex[y], x))))
    return data

#(number of sentences) x (padded sentence length)
# We DO NOT use embeddings here. The padded sentence length * 300 would give us an inordinate amount of dimensions to solve SVM over, and instead hence we use simple indices.
def loadSMSSpamSVM():
    smstrn = loadSMSSpam(Datasets.train)
    smsdev = loadSMSSpam(Datasets.dev)
    smstst = loadSMSSpam(Datasets.test)
    smstrn, smsdev, smstst = splitIntoSentencesAndPad(smstrn, smsdev, smstst)
    print('Loaded sms')

    weight = biasesForLossFunctionUnbalanced(smstrn, smsdev, smstst)

    smstrn, smsdev, smstst = convertDataToIndices(smstrn, smsdev, smstst)

    return {Datasets.train: smstrn, Datasets.dev: smsdev, Datasets.test: smstst, 'weight': weight}

def loadLingspamSVM(restrictLength=300):
    lsmtrn = loadLingspam(Datasets.train)
    lsmdev = loadLingspam(Datasets.dev)
    lsmtst = loadLingspam(Datasets.test)
    lsmtrn, lsmdev, lsmtst = splitIntoSentencesAndPad(lsmtrn, lsmdev, lsmtst, restrictLength=restrictLength)
    print('Loaded ling')

    weight = biasesForLossFunctionUnbalanced(lsmtrn, lsmdev, lsmtst)

    lsmtrn, lsmdev, lsmtst = convertDataToIndices(lsmtrn, lsmdev, lsmtst)

    return {Datasets.train: lsmtrn, Datasets.dev: lsmdev, Datasets.test: lsmtst, 'weight': weight}

def loadGenspamSVM(restrictLength=300):
    gsmtrn = loadGenspam(Datasets.train)
    gsmdev = loadGenspam(Datasets.dev)
    gsmtst = loadGenspam(Datasets.test)
    gsmtrn, gsmdev, gsmtst = splitIntoSentencesAndPad(gsmtrn, gsmdev, gsmtst, restrictLength=restrictLength)
    print('Loaded gen')

    weight = biasesForLossFunctionUnbalanced(gsmtrn, gsmdev, gsmtst)

    gsmtrn, gsmdev, gsmtst = convertDataToIndices(gsmtrn, gsmdev, gsmtst)

    return {Datasets.train: gsmtrn, Datasets.dev: gsmdev, Datasets.test: gsmtst, 'weight': weight}

