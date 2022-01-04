from enum import Enum

class Datasets(Enum):
    train = 1
    dev = 2
    test = 3

dataSplits = {Datasets.train: 0.7, Datasets.dev: 0.15, Datasets.test: 0.15}