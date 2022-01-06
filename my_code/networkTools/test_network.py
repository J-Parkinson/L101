import torch

from my_code.CNN_LSTM.convertToPyTorchDataset import loadSMSSpamPyTorch as SSCLSMS, loadGenspamPyTorch as SSCLGenspam, loadLingspamPyTorch as SSCLLingspam
from my_code.CNN_LSTM.network import SSCL
from my_code.CNN_LSTM.network_ADAPTED import SSCLAdapted
from my_code.biLSTM.convertToPyTorchDataset import loadSMSSpamPyTorch as BiLSTMSMS, loadGenspamPyTorch as BiLSTMGenspam, loadLingspamPyTorch as BiLSTMLingspam
from my_code.biLSTM.network import BiLSTM

from my_code.helpers.datasplit import DataSplit

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

import numpy as np

def safeDivision(a,b):
    return a/b if b else 0

def testModel(model, dataLoader):
    total = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    allPredicted = torch.tensor([]).to(device, dtype=torch.float)

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataLoader:
            sentences, actual = data
            # calculate outputs by running images through the network
            outputs = model(sentences)
            # the class with the highest energy is what we choose as prediction
            predicted = torch.round(outputs.data)
            allPredicted = torch.concat((allPredicted, predicted))
            TP += torch.sum(torch.logical_and(actual, predicted)).item()
            FP += torch.sum(torch.logical_and(torch.logical_not(actual), predicted)).item()
            TN += torch.sum(torch.logical_and(torch.logical_not(actual), torch.logical_not(predicted))).item()
            FN += torch.sum(torch.logical_and(actual, torch.logical_not(predicted))).item()
            total += len(outputs)

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

    return np.array([TP, FP, TN, FN, recall, selectivity, precision, NPV, accuracy, F1]), allPredicted


def loadNetwork(location, model):
    model.load_state_dict(torch.load(location))
    model.eval()
    return model

#data = SSCLSMS(deviceToUse='cpu')
#dataToUse = data[DataSplit.test]
#noWords = data['words']

#networkToTest = loadNetwork('models/SSCLA_SMSmodel_20220104_144633', SSCLAdapted(noWords))
#print('Model initialised')

#testModel(networkToTest, dataToUse)