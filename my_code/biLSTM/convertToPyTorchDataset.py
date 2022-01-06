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
from my_code.helpers.datasplit import DataSplit

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


def pyTorchDataset(data, name='', deviceToUse=device, shuffle=True):
    print(f'converting {name}to pyTorch')
    sequenceData = np.array(list(map(lambda x: np.array(x), data['sequence'].to_numpy())))
    typeData = np.array(data['type'].to_numpy(), dtype='uint8')
    print('Vectorised input size:', sequenceData.shape)
    print('Expected output size:', typeData.shape)
    dataInput = tensor(sequenceData).to(deviceToUse, dtype=torch.float)
    dataOutput = tensor(typeData).to(deviceToUse, dtype=torch.float)
    print('Loaded into CUDA')
    train = TensorDataset(dataInput, dataOutput)
    train_loader = DataLoader(train, batch_size=32, shuffle=shuffle)
    return train_loader

with open('../wordEmbeddings/genspam_test', 'rb') as genspam_test:
    customWords = pickle.load(genspam_test)
    customWords[''] = np.zeros(300)

print('Loaded custom words')

#(number of sentences) x (padded sentence length) x (embedding length = 300)
def loadSMSSpamPyTorch(deviceToUse=device, shuffle=True, pytorchOnly=True):
    smstrn = loadSMSSpam(DataSplit.train)
    smsdev = loadSMSSpam(DataSplit.dev)
    smstst = loadSMSSpam(DataSplit.test)

    smstrn['sequenceOriginal'] = smstrn['sequence']
    smsdev['sequenceOriginal'] = smsdev['sequence']
    smstst['sequenceOriginal'] = smstst['sequence']

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

    smstrnPT = pyTorchDataset(smstrn, 'smstrn ', deviceToUse=deviceToUse, shuffle=shuffle)
    smsdevPT = pyTorchDataset(smsdev, 'smsdev ', deviceToUse=deviceToUse, shuffle=shuffle)
    smststPT = pyTorchDataset(smstst, 'smstst ', deviceToUse=deviceToUse, shuffle=shuffle)

    if pytorchOnly:
        return {DataSplit.train: smstrnPT, DataSplit.dev: smsdevPT, DataSplit.test: smststPT, 'weight': weight, 'words': words}
    else:
        return ({DataSplit.train: smstrnPT, DataSplit.dev: smsdevPT, DataSplit.test: smststPT, 'weight': weight,
                'words': words}, {DataSplit.train: smstrn, DataSplit.dev: smsdev, DataSplit.test: smstst, 'weight': weight, 'words': words})

def loadLingspamPyTorch(restrictLength=300, deviceToUse=device, shuffle=True, pytorchOnly=True):
    lsmtrn = loadLingspam(DataSplit.train)
    lsmdev = loadLingspam(DataSplit.dev)
    lsmtst = loadLingspam(DataSplit.test)

    lsmtrn['sequenceOriginal'] = lsmtrn['sequence']
    lsmdev['sequenceOriginal'] = lsmdev['sequence']
    lsmtst['sequenceOriginal'] = lsmtst['sequence']

    lsmtrn, lsmdev, lsmtst = splitIntoSentencesAndPad(lsmtrn, lsmdev, lsmtst, restrictLength=restrictLength)
    print('Loaded ling')

    weight = tensor(biasesForLossFunctionUnbalanced(lsmtrn, lsmdev, lsmtst)).to(deviceToUse, dtype=torch.float)
    words = len(lsmtrn['sequence'][0])

    lsmtrn, _ = convertDataToWordVectors(lsmtrn, customWords, 'lingspam_train', columnName='sequenceVector')
    print('lingspam_train done')
    lsmdev, _ = convertDataToWordVectors(lsmdev, customWords, 'lingspam_dev', columnName='sequenceVector')
    print('lingspam_dev done')
    lsmtst, _ = convertDataToWordVectors(lsmtst, customWords, 'lingspam_test', columnName='sequenceVector')
    print('lingspam_test done')

    lsmtrn = addGloveVectors(lsmtrn, 'lingspam_train')
    print('lingspam_train glove done')
    lsmdev = addGloveVectors(lsmdev, 'lingspam_dev')
    print('lingspam_dev glove done')
    lsmtst = addGloveVectors(lsmtst, 'lingspam_test')
    print('lingspam_test glove done')

    lsmtrn['sequence'] = lsmtrn.apply(lambda row: np.concatenate((row['sequenceVector'], row['sequenceGlove']), axis=1),
                                      axis=1)
    lsmdev['sequence'] = lsmdev.apply(lambda row: np.concatenate((row['sequenceVector'], row['sequenceGlove']), axis=1),
                                      axis=1)
    lsmtst['sequence'] = lsmtst.apply(lambda row: np.concatenate((row['sequenceVector'], row['sequenceGlove']), axis=1),
                                      axis=1)

    lsmtrnPT = pyTorchDataset(lsmtrn, 'lsmtrn ', deviceToUse=deviceToUse, shuffle=shuffle)
    lsmdevPT = pyTorchDataset(lsmdev, 'lsmdev ', deviceToUse=deviceToUse, shuffle=shuffle)
    lsmtstPT = pyTorchDataset(lsmtst, 'lsmtst ', deviceToUse=deviceToUse, shuffle=shuffle)

    if pytorchOnly:
        return {DataSplit.train: lsmtrnPT, DataSplit.dev: lsmdevPT, DataSplit.test: lsmtstPT, 'weight': weight, 'words': words}
    else:
        return ({DataSplit.train: lsmtrnPT, DataSplit.dev: lsmdevPT, DataSplit.test: lsmtstPT, 'weight': weight, 'words': words},
                {DataSplit.train: lsmtrn, DataSplit.dev: lsmdev, DataSplit.test: lsmtst, 'weight': weight,
                 'words': words})

def loadGenspamPyTorch(restrictLength=175, deviceToUse=device, shuffle=True, pytorchOnly=True):
    gsmtrn = loadGenspam(DataSplit.train)
    gsmdev = loadGenspam(DataSplit.dev)
    gsmtst = loadGenspam(DataSplit.test)

    gsmtrn = gsmtrn[:len(gsmtrn) // 2]
    gsmdev = gsmdev[:len(gsmdev) // 2]

    gsmtrn['sequenceOriginal'] = gsmtrn['sequence']
    gsmdev['sequenceOriginal'] = gsmdev['sequence']
    gsmtst['sequenceOriginal'] = gsmtst['sequence']

    gsmtrn, gsmdev, gsmtst = splitIntoSentencesAndPad(gsmtrn, gsmdev, gsmtst, restrictLength=restrictLength)
    print('Loaded gen')

    weight = tensor(biasesForLossFunctionUnbalanced(gsmtrn, gsmdev, gsmtst)).to(deviceToUse, dtype=torch.float)
    words = len(gsmtrn['sequence'][0])

    gsmtrn, _ = convertDataToWordVectors(gsmtrn, customWords, 'genspam_train', columnName='sequenceVector')
    print('genspam_train done')
    gsmdev, _ = convertDataToWordVectors(gsmdev, customWords, 'genspam_dev', columnName='sequenceVector')
    print('genspam_dev done')
    gsmtst, _ = convertDataToWordVectors(gsmtst, customWords, 'genspam_test', columnName='sequenceVector')
    print('genspam_test done')

    gsmtrn = addGloveVectors(gsmtrn, 'genspam_train')
    print('genspam_train glove done')
    gsmdev = addGloveVectors(gsmdev, 'genspam_dev')
    print('genspam_dev glove done')
    gsmtst = addGloveVectors(gsmtst, 'genspam_test')
    print('genspam_test glove done')

    gsmtrn['sequence'] = gsmtrn.apply(lambda row: np.concatenate((row['sequenceVector'], row['sequenceGlove']), axis=1),
                                      axis=1)
    gsmdev['sequence'] = gsmdev.apply(lambda row: np.concatenate((row['sequenceVector'], row['sequenceGlove']), axis=1),
                                      axis=1)
    gsmtst['sequence'] = gsmtst.apply(lambda row: np.concatenate((row['sequenceVector'], row['sequenceGlove']), axis=1),
                                      axis=1)

    gsmtrnPT = pyTorchDataset(gsmtrn, 'gsmtrn ', deviceToUse=deviceToUse, shuffle=shuffle)
    gsmdevPT = pyTorchDataset(gsmdev, 'gsmdev ', deviceToUse=deviceToUse, shuffle=shuffle)
    gsmtstPT = pyTorchDataset(gsmtst, 'gsmtst ', deviceToUse=deviceToUse, shuffle=shuffle)

    if pytorchOnly:
        return {DataSplit.train: gsmtrnPT, DataSplit.dev: gsmdevPT, DataSplit.test: gsmtstPT, 'weight': weight, 'words': words}
    else:
        return ({DataSplit.train: gsmtrnPT, DataSplit.dev: gsmdevPT, DataSplit.test: gsmtstPT, 'weight': weight, 'words': words},
                {DataSplit.train: gsmtrn, DataSplit.dev: gsmdev, DataSplit.test: gsmtst, 'weight': weight,
                 'words': words})
#trainingData = loadSMSSpamPyTorch()
