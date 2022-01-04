from my_code.load_and_tokenize.loading_helpers import getSplits
from my_code.helpers.datasets import Datasets
import pandas as pd
import re

regexToUse = re.compile('[ -~]\w*')

def tokenizer(text):
    tokenized = ' '.join([x.replace(' ', '') for x in list(regexToUse.findall(text))])
    return re.sub(' +', ' ', tokenized)

def loadSMSSpam(dataset=Datasets.train):
    spamDataCSV = pd.read_csv("../../data/smsspamcollection/SMSSpamCollection.txt", delimiter='\t', names=["type", "sequence"])
    length = len(spamDataCSV.index)
    splits = getSplits(length, dataset)
    spamDataCSV['type'][spamDataCSV['type'] == 'ham'] = 0
    spamDataCSV['type'][spamDataCSV['type'] == 'spam'] = 1
    spamDataCSV['sequence'] = spamDataCSV['sequence'].apply(lambda x: tokenizer(x))
    return spamDataCSV[splits[0]:splits[1]]


def getStatistics():
    train = loadSMSSpam()
    trainspam = len(train.loc[train['type'] == 'spam'].index)
    trainham = len(train.loc[train['type'] == 'ham'].index)
    trainfull = trainham + trainspam

    dev = loadSMSSpam(Datasets.dev)
    devspam = len(dev.loc[dev['type'] == 'spam'].index)
    devham = len(dev.loc[dev['type'] == 'ham'].index)
    devfull = devham + devspam

    test = loadSMSSpam(Datasets.test)
    testspam = len(test.loc[test['type'] == 'spam'].index)
    testham = len(test.loc[test['type'] == 'ham'].index)
    testfull = testspam + testham

    print(trainspam/trainfull, trainham/trainfull, devspam/devfull, devham/devfull, testspam/testfull, testham/testfull)

#print(loadSMSSpam())