from my_code.helpers.datasplit import DataSplit, dataSplits
import numpy as np

def getSplits(length, dataset):
    splitSizes = [dataSplits[DataSplit.train], dataSplits[DataSplit.dev], dataSplits[DataSplit.test]]
    splitStartEnd = [sum(splitSizes[:i]) for i in range(len(splitSizes) + 1)]
    splitStartProportion = splitStartEnd[dataset.value - 1:dataset.value + 1]
    return [int(np.floor(val * length)) for val in splitStartProportion]

#From https://www.activecampaign.com/blog/spam-words
def loadGazetteer():
    with open("../../gazetteer/spam.txt", 'r') as gazetteer:
        gazetteerRead = gazetteer.read()
    return gazetteerRead.split('\n')