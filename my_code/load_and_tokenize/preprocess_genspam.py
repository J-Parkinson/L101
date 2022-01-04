from my_code.helpers.datasets import Datasets
import pandas as pd
import re

random_seed = 923940644 #Random seed in [0, 10^9] from random.org

parseText = re.compile('<TEXT_[A-Z]*>([\s\S]*?)</TEXT_[A-Z]*>')

def parse(data):
    data = data.replace("<MESSAGE>\n", "").replace('\r', '')
    data = data.split("</MESSAGE>")
    getBits = [parseText.findall(dataBit.replace('\n', '')) for dataBit in data]
    tokenized = ['\f'.join(dataBit) for dataBit in getBits]
    return tokenized


def loadGenspam(dataset=Datasets.train):
    data = {}
    dataLoad = {Datasets.train: ['train_GEN.ems', 'train_SPAM.ems'], Datasets.dev: ['adapt_GEN.ems', 'adapt_SPAM.ems'], Datasets.test: ['test_GEN.ems', 'test_SPAM.ems']}
    for file in dataLoad[dataset]:
        if 'GEN' in file:
            fileType = 'ham'
        else:
            fileType = 'spam'
        with open(f"../../data/genspamcollection/{file}", 'rb') as fileData:
            fileDataRead = fileData.read().decode("utf-8", 'backslashreplace')
            data[fileType] = parse(fileDataRead)

    genspamDataCSVHam = pd.Series(data['ham'], name='sequence')
    genspamDataCSVHam = genspamDataCSVHam.to_frame()
    genspamDataCSVHam['type'] = 0 #ham

    genspamDataCSVSpam = pd.Series(data['spam'], name='sequence')
    genspamDataCSVSpam = genspamDataCSVSpam.to_frame()
    genspamDataCSVSpam['type'] = 1 #spam

    genspamDataCSV = genspamDataCSVHam.append(genspamDataCSVSpam, ignore_index=True)
    #genspamDataCSV['sequence'] = genspamDataCSV['sequence'].apply(lambda x: x.replace('^ ', ''))
    genspamDataCSV = genspamDataCSV.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    return genspamDataCSV


def getStatistics():
    train = loadGenspam()
    trainspam = len(train.loc[train['type'] == 1].index)
    trainham = len(train.loc[train['type'] == 0].index)
    trainfull = trainham + trainspam

    dev = loadGenspam(Datasets.dev)
    devspam = len(dev.loc[dev['type'] == 1].index)
    devham = len(dev.loc[dev['type'] == 0].index)
    devfull = devham + devspam

    test = loadGenspam(Datasets.test)
    testspam = len(test.loc[test['type'] == 1].index)
    testham = len(test.loc[test['type'] == 0].index)
    testfull = testspam + testham

    print(trainspam/trainfull, trainham/trainfull, devspam/devfull, devham/devfull, testspam/testfull, testham/testfull)

#print(loadGenspam())