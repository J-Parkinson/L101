from datetime import datetime

import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch

from my_code.CNN_LSTM.convertToPyTorchDataset import loadSMSSpamPyTorch as SSCLSMS, loadGenspamPyTorch as SSCLGenspam, loadLingspamPyTorch as SSCLLingspam
from my_code.CNN_LSTM.network import SSCL
from my_code.CNN_LSTM.network_ADAPTED import SSCLAdapted
from my_code.biLSTM.convertToPyTorchDataset import loadSMSSpamPyTorch as BiLSTMSMS, loadGenspamPyTorch as BiLSTMGenspam, loadLingspamPyTorch as BiLSTMLingspam
from my_code.biLSTM.network import BiLSTM

#Cobbled together from https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
#hence the intentional mismatch in variable name formats
from my_code.helpers.datasplit import DataSplit


def trainOneEpoch(trainingLoader, model, optimizer, lossFunction, epoch_index, tb_writer):
    runningLoss = 0.0
    lastLoss = 0.0
    noDataPerEpoch = len(trainingLoader)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(trainingLoader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = lossFunction(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        runningLoss += float(loss.item())
        if i % (noDataPerEpoch//10) == ((noDataPerEpoch//10)-1):
            lastLoss = runningLoss / 1000  # loss per batch
            print(f'  batch {i + 1} loss: {lastLoss}')
            tb_x = epoch_index * len(trainingLoader) + i + 1
            tb_writer.add_scalar('Loss/train', lastLoss, tb_x)
            runningLoss = 0.

    return lastLoss


def trainNetwork(trainingLoader, validationLoader, model, weight, EPOCHS=15, location=""):
    #print(weight)
    # binary loss function
    lossFunction = nn.BCELoss(weight=weight)

    # gradient descent optimiser
    optimizer = optim.Adagrad(model.parameters())

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'runs/CNN_LSTM_{timestamp}')
    epoch_number = 0
    best_model=None

    best_vloss = 1_000_000.
    model_path = f'models/{location}model_{timestamp}'
    torch.save(model.state_dict(), model_path)

    for epoch in range(EPOCHS):
        print(f'EPOCH {epoch_number + 1}:')

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = trainOneEpoch(trainingLoader, model, optimizer, lossFunction, epoch_number, writer)

        # We don't need gradients on to do reporting
        model.train(False)

        running_vloss = 0.0
        i=0
        for i, vdata in enumerate(validationLoader):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = lossFunction(voutputs, vlabels)
            running_vloss += float(vloss)

        avg_vloss = running_vloss / (i + 1)
        print(f'LOSS train {avg_loss} valid {avg_vloss}')

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                           {'Training': avg_loss, 'Validation': avg_vloss},
                           epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f'models/{location}model_{timestamp}'
            best_model = model
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

    return best_model


def trainSSCL(data, loc='_'):
    numberWords = data['words']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model = SSCL(numberWords).to(device)
    print(f'Loaded model with {numberWords} words')
    model = trainNetwork(data[DataSplit.train], data[DataSplit.dev], model, data['weight'], location=f'{loc}')
    return model

def trainSSCLAdapted(data, loc='_'):
    numberWords = data['words']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model = SSCLAdapted(numberWords).to(device)
    print(f'Loaded model with {numberWords} words')
    model = trainNetwork(data[DataSplit.train], data[DataSplit.dev], model, data['weight'], location=f'{loc}')
    return model

def trainBiLSTM(data, loc='_'):
    numberWords = data['words']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    model = BiLSTM(numberWords).to(device)
    print(f'Loaded model with {numberWords} words')
    model = trainNetwork(data[DataSplit.train], data[DataSplit.dev], model, data['weight'], location=f'{loc}')
    return model

#trainSSCL(SSCLSMS(), '_SMS')
#trainSSCLAdapted(SSCLSMS(), '_SMS')
#trainBiLSTM(BiLSTMSMS(), '_SMS')