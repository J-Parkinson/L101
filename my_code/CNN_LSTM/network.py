import torch
from torch import nn

class SSCL(nn.Module):
    def __init__(self, numberWords, numberFilters=128, filterSize=5, embeddingSize=300):
        super().__init__()

        # LSTM COPIED
        self.hidden_dim = 2
        self.embedding_dim = embeddingSize
        self.number_words = numberWords - filterSize + 1
        self.filter_dim = filterSize
        self.number_filters = numberFilters

        self.convStack = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(in_channels=1, out_channels=self.number_filters, kernel_size=(self.filter_dim, self.embedding_dim)),
            nn.ReLU(inplace=True),
        )

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.number_words, self.number_filters // 2, batch_first=True, proj_size=1)

        self.softmax = nn.Sigmoid()

    def forward(self, x):
        #print(x.shape)
        xReshape = x.view((x.shape[0], 1, x.shape[1], x.shape[2]))
        #print(xReshape.shape)
        xConv = self.convStack(xReshape)
        #print(xConv.shape)
        xConvReshape = xConv.squeeze()
        #print(xConvReshape.shape)
        xLSTM = self.lstm(xConvReshape)
        xLSTMFinal = xLSTM[1][1].squeeze()[:, -1]
        xSoftmaxed = self.softmax(xLSTMFinal)
        return xSoftmaxed

#model = SSCL().to(device)
#print('Loaded model')