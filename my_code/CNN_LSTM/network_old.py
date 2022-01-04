import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

class SSCL(nn.Module):
    def __init__(self, numberFilters=128, filterSize=5, embeddingSize=300):
        super().__init__()

        # LSTM COPIED
        self.hidden_dim = 2
        self.embedding_dim = embeddingSize
        self.number_filters = numberFilters
        self.filter_dim = filterSize


        self.convStack = nn.Sequential(
            nn.Dropout(0.1),
            #Even though it is called a 'CNN', we implement using dense layers.
            nn.Linear(embeddingSize*filterSize, numberFilters),
            nn.ReLU(),
        )

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.number_filters, 1, batch_first=True)

        self.softmax = nn.Sigmoid()

    def forward(self, x):
        #xJoin = torch.dstack([torch.roll(x,-i,1) for i in range(self.filter_dim)])[:,:-self.filter_dim+1]
        #xJoin = torch.tensor(np.array([xSwap[i:i+self.filter_dim] for i in range(len(xSwap)-self.filter_dim+1)]))
        #xMove = torch.moveaxis(xJoin, (0,1,2), (2,0,1))
        #xCNN = torch.reshape(xMove, (xMove.shape[0], xMove.shape[1], xMove.shape[2] * xMove.shape[3]))
        convolved = self.convStack(xJoin)
        LSTMd = self.lstm(convolved)[0][:,-1].squeeze()
        softmaxed = self.softmax(LSTMd)
        return softmaxed

model = SSCL().to(device)
print('Loaded model')