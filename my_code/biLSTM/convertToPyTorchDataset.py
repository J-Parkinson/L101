from torch import tensor
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
import numpy as np

from my_code.glove.word_embeddings import addGloveVectors
from my_code.wordEmbeddings.word_embeddings import convertDataToWordVectors
from my_code.load_and_tokenize.preprocess_genspam import loadGenspam
from my_code.load_and_tokenize.preprocess_lingspam import loadLingspam
from my_code.load_and_tokenize.preprocess_sms_spam import loadSMSSpam
from my_code.helpers.datasets import Datasets

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


def biasesForLossFunctionUnbalanced(*argv):
    totalLength = 0
    spam = 0
    for dataset in argv:
        totalLength += len(dataset)
        spam += dataset['type'].value_counts()[1]
    return totalLength / spam

def splitIntoSentencesAndPad(*argv, restrictLength=None):
    max_lens = set()
    for data in argv:
        data['seq_len'] = data['sequence'].apply(lambda x: len(x.split(' ')))
        max_lens.add(data['seq_len'].max())
    len_to_pad_to = max(max_lens)
    if restrictLength is not None:
        len_to_pad_to = max(len_to_pad_to, restrictLength)
    print('Padded length:', len_to_pad_to)
    for data in argv:
        data['sequence'] = data['sequence'].apply(lambda val: val + ' ' * (len_to_pad_to - len(val.split(' '))))
        data['sequence'] = data['sequence'].apply(lambda val: val.split(' '))
        if restrictLength is not None:
            data['sequence'] = data['sequence'].apply(lambda val: val[:restrictLength])
    return argv


def pyTorchDataset(data, name='', deviceToUse=device):
    print(f'converting {name}to pyTorch')
    sequenceData = np.array(list(map(lambda x: np.array(x), data['sequence'].to_numpy())))
    typeData = np.array(data['type'].to_numpy(), dtype='uint8')
    print('Vectorised input size:', sequenceData.shape)
    print('Expected output size:', typeData.shape)
    dataInput = tensor(sequenceData).to(deviceToUse, dtype=torch.float)
    dataOutput = tensor(typeData).to(deviceToUse, dtype=torch.float)
    print('Loaded into CUDA')
    train = TensorDataset(dataInput, dataOutput)
    train_loader = DataLoader(train, batch_size=32, shuffle=True)
    return train_loader

with open('../wordEmbeddings/genspam_test', 'rb') as genspam_test:
    customWords = pickle.load(genspam_test)
    customWords[''] = np.zeros(300)

print('Loaded custom words')

#(number of sentences) x (padded sentence length) x (embedding length = 300)
def loadSMSSpamPyTorch(deviceToUse=device):
    smstrn = loadSMSSpam(Datasets.train)
    smsdev = loadSMSSpam(Datasets.dev)
    smstst = loadSMSSpam(Datasets.test)
    smstrn, smsdev, smstst = splitIntoSentencesAndPad(smstrn, smsdev, smstst)
    print('Loaded sms')

    weight = tensor(biasesForLossFunctionUnbalanced(smstrn, smsdev, smstst)).to(deviceToUse, dtype=torch.float)
    words = len(smstrn['sequence'][0])

    smstrn, _ = convertDataToWordVectors(smstrn, customWords, 'smsspam_train', columnName='sequenceVector')
    print('smsspam_train done')
    smsdev, _ = convertDataToWordVectors(smsdev, customWords, 'smsspam_dev', columnName='sequenceVector')
    print('smsspam_dev done')
    smstst, _ = convertDataToWordVectors(smstst, customWords, 'smsspam_test', columnName='sequenceVector')
    print('smsspam_test done')

    smstrn = addGloveVectors(smstrn, 'smsspam_train')
    print('smsspam_train glove done')
    smsdev = addGloveVectors(smsdev, 'smsspam_dev')
    print('smsspam_dev glove done')
    smstst = addGloveVectors(smstst, 'smsspam_test')
    print('smsspam_test glove done')

    smstrn['sequence'] = smstrn.apply(lambda row: np.concatenate((row['sequenceVector'], row['sequenceGlove']), axis=1), axis=1)
    smsdev['sequence'] = smsdev.apply(lambda row: np.concatenate((row['sequenceVector'], row['sequenceGlove']), axis=1), axis=1)
    smstst['sequence'] = smstst.apply(lambda row: np.concatenate((row['sequenceVector'], row['sequenceGlove']), axis=1), axis=1)

    smstrn = pyTorchDataset(smstrn, 'smstrn ', deviceToUse=deviceToUse)
    smsdev = pyTorchDataset(smsdev, 'smsdev ', deviceToUse=deviceToUse)
    smstst = pyTorchDataset(smstst, 'smstst ', deviceToUse=deviceToUse)

    return {Datasets.train: smstrn, Datasets.dev: smsdev, Datasets.test: smstst, 'weight': weight, 'words': words}

def loadLingspamPyTorch(restrictLength=300, deviceToUse=device):
    lsmtrn = loadLingspam(Datasets.train)
    lsmdev = loadLingspam(Datasets.dev)
    lsmtst = loadLingspam(Datasets.test)
    lsmtrn, lsmdev, lsmtst = splitIntoSentencesAndPad(lsmtrn, lsmdev, lsmtst, restrictLength=restrictLength)
    print('Loaded ling')

    weight = tensor(biasesForLossFunctionUnbalanced(lsmtrn, lsmdev, lsmtst)).to(deviceToUse, dtype=torch.float)

    lsmtrn, _ = convertDataToWordVectors(lsmtrn, customWords, 'lingspam_train', columnName='sequenceVector')
    print('lingspam_train done')
    lsmdev, _ = convertDataToWordVectors(lsmdev, customWords, 'lingspam_dev', columnName='sequenceVector')
    print('lingspam_dev done')
    lsmtst, _ = convertDataToWordVectors(lsmtst, customWords, 'lingspam_test', columnName='sequenceVector')
    print('lingspam_test done')

    lsmtrn = pyTorchDataset(lsmtrn, 'lsmtrn ', deviceToUse=deviceToUse)
    lsmdev = pyTorchDataset(lsmdev, 'lsmdev ', deviceToUse=deviceToUse)
    lsmtst = pyTorchDataset(lsmtst, 'lsmtst ', deviceToUse=deviceToUse)

    return {Datasets.train: lsmtrn, Datasets.dev: lsmdev, Datasets.test: lsmtst, 'weight': weight}

def loadGenspamPyTorch(restrictLength=175, deviceToUse=device):
    gsmtrn = loadGenspam(Datasets.train)
    gsmdev = loadGenspam(Datasets.dev)
    gsmtst = loadGenspam(Datasets.test)
    gsmtrn, gsmdev, gsmtst = splitIntoSentencesAndPad(gsmtrn, gsmdev, gsmtst, restrictLength=restrictLength)
    print('Loaded gen')

    weight = tensor(biasesForLossFunctionUnbalanced(gsmtrn, gsmdev, gsmtst)).to(deviceToUse, dtype=torch.float)

    gsmtrn, _ = convertDataToWordVectors(gsmtrn, customWords, 'genspam_train', columnName='sequenceVector')
    print('genspam_train done')
    gsmdev, _ = convertDataToWordVectors(gsmdev, customWords, 'genspam_dev', columnName='sequenceVector')
    print('genspam_dev done')
    gsmtst, _ = convertDataToWordVectors(gsmtst, customWords, 'genspam_test', columnName='sequenceVector')
    print('genspam_test done')

    gsmtrn = pyTorchDataset(gsmtrn, 'gsmtrn ', deviceToUse=deviceToUse)
    gsmdev = pyTorchDataset(gsmdev, 'gsmdev ', deviceToUse=deviceToUse)
    gsmtst = pyTorchDataset(gsmtst, 'gsmtst ', deviceToUse=deviceToUse)

    return {Datasets.train: gsmtrn, Datasets.dev: gsmdev, Datasets.test: gsmtst, 'weight': weight}

trainingData = loadSMSSpamPyTorch()
