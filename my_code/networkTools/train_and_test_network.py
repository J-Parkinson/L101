from my_code.helpers.data import Dataset
from my_code.helpers.datasplit import DataSplit
from my_code.helpers.model import Model

from my_code.CNN_LSTM.convertToPyTorchDataset import loadSMSSpamPyTorch as SSCLSMS, loadGenspamPyTorch as SSCLGenspam, loadLingspamPyTorch as SSCLLingspam
from my_code.biLSTM.convertToPyTorchDataset import loadSMSSpamPyTorch as BiLSTMSMS, loadGenspamPyTorch as BiLSTMGenspam, loadLingspamPyTorch as BiLSTMLingspam
from my_code.networkTools.test_network import testModel
from my_code.networkTools.train_network import trainSSCL, trainSSCLAdapted, trainBiLSTM

import torch
import numpy as np

from my_code.results.resultsToPandas import saveTestResult, saveScores


def train_and_test(dataset, model):
    location = ''
    if model == Model.BiLSTM:
        trainingCall = trainBiLSTM
        location += 'BiLSTM/'
        if dataset == Dataset.SMS:
            dataset = BiLSTMSMS
            location += 'SMS/'
        elif dataset == Dataset.Lingspam:
            dataset = BiLSTMLingspam
            location += 'Lingspam/'
        else:
            dataset = BiLSTMGenspam
            location += 'Genspam/'
    elif model == Model.SSCLA:
        trainingCall = trainSSCLAdapted
        location += 'SSCLA/'
        if dataset == Dataset.SMS:
            dataset = SSCLSMS
            location += 'SMS/'
        elif dataset == Dataset.Lingspam:
            dataset = SSCLLingspam
            location += 'Lingspam/'
        else:
            dataset = SSCLGenspam
            location += 'Genspam/'
    else:
        trainingCall = trainSSCL
        location += 'SSCL/'
        if dataset == Dataset.SMS:
            dataset = SSCLSMS
            location += 'SMS/'
        elif dataset == Dataset.Lingspam:
            dataset = SSCLLingspam
            location += 'Lingspam/'
        else:
            dataset = SSCLGenspam
            location += 'Genspam/'

    testDataset, testDatasetPandas = dataset(shuffle=False, pytorchOnly=False)
    testDatasetPandas = testDatasetPandas[DataSplit.test]
    testDataset = testDataset[DataSplit.test]

    results = []
    for i in range(3):
        datasetToUse = dataset()
        model = trainingCall(datasetToUse, location)
        print(f'MODEL {i}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        testResult, predicted = testModel(model, testDataset)
        results.append(testResult)
        saveTestResult(testResult, predicted, testDatasetPandas, i)
        del model
        del datasetToUse
        torch.cuda.empty_cache()
    results = np.array(results)
    resultMean = list(np.mean(results, axis=0))
    resultVariance = list(np.var(results, axis=0))
    saveScores(resultMean, variance=resultVariance)



#train_and_test(Dataset.SMS, Model.SSCL)
#train_and_test(Dataset.Lingspam, Model.SSCL)
#train_and_test(Dataset.Genspam, Model.SSCL)
#train_and_test(Dataset.SMS, Model.SSCLA)
#train_and_test(Dataset.Lingspam, Model.SSCLA)
#train_and_test(Dataset.Genspam, Model.SSCLA)
#train_and_test(Dataset.SMS, Model.BiLSTM)
#train_and_test(Dataset.Lingspam, Model.BiLSTM)
train_and_test(Dataset.Genspam, Model.BiLSTM)