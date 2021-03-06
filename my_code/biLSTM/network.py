import torch
from torch import nn

class BiLSTM(nn.Module):
    def __init__(self, numberWords, numberFilters=64, filterSize=5, poolFilters=4, embeddingSize=500):
        super().__init__()

        # LSTM COPIED
        self.embedding_dim = embeddingSize
        self.number_words = numberWords - filterSize + 1
        self.filter_dim = filterSize
        self.number_filters = numberFilters
        self.pool_filters = poolFilters

        self.convStack = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=self.number_filters, kernel_size=(self.filter_dim, self.embedding_dim)),
            nn.ReLU(inplace=True),
        )

        self.poolLayer = nn.MaxPool1d(self.number_filters//self.pool_filters, stride=self.number_filters//self.pool_filters)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.pool_filters, self.pool_filters, batch_first=True, bidirectional=True)

        self.dense = nn.Linear(self.number_words * self.pool_filters * 2, 1)

        self.softmax = nn.Sigmoid()

    def forward(self, x):
        #print(x.shape)
        xReshape = x.view((x.shape[0], 1, x.shape[1], x.shape[2]))
        #print(xReshape.shape)
        xConv = self.convStack(xReshape)
        #print(xConv.shape)
        xConvReshape = xConv.squeeze().swapaxes(1, 2)
        #print(xConvReshape.shape)
        xPool = self.poolLayer(xConvReshape)
        #print(xPool.shape)
        xLSTM = self.lstm(xPool)[0]
        #print(xLSTM.shape)
        xLSTMReshape = torch.flatten(xLSTM, start_dim=1)
        #print(xLSTMReshape.shape)
        xDense = self.dense(xLSTMReshape).squeeze()
        #print(xDense.shape)
        xSoftmaxed = self.softmax(xDense)
        #print(xSoftmaxed.shape)
        return xSoftmaxed

#model = SSCL().to(device)
#print('Loaded model')