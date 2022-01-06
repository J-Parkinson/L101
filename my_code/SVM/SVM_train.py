from sklearn.svm import SVC
import pickle

from my_code.SVM.SVM_preprocess import loadSMSSpamSVM, loadGenspamSVM, loadLingspamSVM
from my_code.helpers.datasplit import DataSplit

cSMS = 1
gammaSMS = 0.0000001

def trainSVM(data, save=None):
    trainData = data[DataSplit.train]
    weight = data['weight']
    outputs = list(trainData['type'].to_numpy())
    sequences = list(trainData['sequence'].to_numpy())
    spamClassifier = SVC(class_weight={0: 1/(1-(1/weight)), 1: weight})
    spamClassifier.fit(sequences, outputs)
    if save:
        with open(save, 'wb') as file:
            pickle.dump(spamClassifier, file)
    return spamClassifier

spamClassifier = trainSVM(loadSMSSpamSVM(), save='svmSpam')