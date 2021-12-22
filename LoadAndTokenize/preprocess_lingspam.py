from LoadAndTokenize.loading_helpers import getSplits
from Reusables.datasets import Datasets
import pandas as pd

def loadLingspam(dataset=Datasets.train):
    data = pd.read_csv('../../lingspamcollection/messages.csv')

    #Note that the ling spam dataset has both subjects and messages - we concatenate for our purposes
    data['sequence'] = data['subject'] + '\f' + data['message']
    data = data.rename(columns={'label': 'type'})

    lingspamDataCSV = data[['type', 'sequence']]
    length = len(lingspamDataCSV.index)
    splits = getSplits(length, dataset)
    return lingspamDataCSV[splits[0]:splits[1]].dropna()

def getStatistics():
    train = loadLingspam()
    trainspam = len(train.loc[train['type'] == 1].index)
    trainham = len(train.loc[train['type'] == 0].index)
    trainfull = trainham + trainspam

    dev = loadLingspam(Datasets.dev)
    devspam = len(dev.loc[dev['type'] == 1].index)
    devham = len(dev.loc[dev['type'] == 0].index)
    devfull = devham + devspam

    test = loadLingspam(Datasets.test)
    testspam = len(test.loc[test['type'] == 1].index)
    testham = len(test.loc[test['type'] == 0].index)
    testfull = testspam + testham

    print(trainspam/trainfull, trainham/trainfull, devspam/devfull, devham/devfull, testspam/testfull, testham/testfull)

print(loadLingspam())