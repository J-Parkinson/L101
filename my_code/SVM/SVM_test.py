from sklearn.svm import SVC
import pickle

from my_code.SVM.SVM_preprocess import loadSMSSpamSVM, loadGenspamSVM, loadLingspamSVM
from my_code.helpers.datasplit import DataSplit
import numpy as np

cSMS = 1
gammaSMS = 0.0000001

def loadSVM(location):
    with open(location, 'rb') as file:
        svm = pickle.load(file)
    return svm


def safeDivision(a,b):
    return a/b if b else 0

def testModel(data, svm):
    total = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    trainData = data[DataSplit.test]
    actual = list(trainData['type'].to_numpy())
    sentences = list(trainData['sequence'].to_numpy())
    # calculate outputs by running images through the network
    predicted = svm.predict(sentences)
    #print(predicted)
    # the class with the highest energy is what we choose as prediction
    predicted = np.round(predicted.data)
    TP += np.sum(np.logical_and(actual, predicted)).item()
    FP += np.sum(np.logical_and(np.logical_not(actual), predicted)).item()
    TN += np.sum(np.logical_and(np.logical_not(actual), np.logical_not(predicted))).item()
    FN += np.sum(np.logical_and(actual, np.logical_not(predicted))).item()
    total += len(actual)

    TP = safeDivision(TP,total)
    TN = safeDivision(TN,total)
    FP = safeDivision(FP,total)
    FN = safeDivision(FN,total)

    recall = safeDivision(TP,(TP+FN))
    selectivity = safeDivision(TN,(TN+FP))
    precision = safeDivision(TP,(TP+FP))
    NPV = safeDivision(TN,(TN+FN))
    accuracy = safeDivision((TP+TN),(TP+TN+FP+FN))
    F1 = safeDivision(2 * (precision * recall),(precision + recall))

    print(f'TP:{TP}')
    print(f'FP:{FP}')
    print(f'TN:{TN}')
    print(f'FN:{FN}')
    print(f'recall     :{recall}')
    print(f'selectivity:{selectivity}')
    print(f'precision  :{precision}')
    print(f'NPV        :{NPV}')
    print(f'accuracy   :{accuracy}')
    print(f'F1         :{F1}')

classifier = loadSVM('svmSpam')

testModel(loadSMSSpamSVM(), classifier)