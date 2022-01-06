from enum import Enum

class DataSplit(Enum):
    train = 1
    dev = 2
    test = 3

dataSplits = {DataSplit.train: 0.7, DataSplit.dev: 0.15, DataSplit.test: 0.15}