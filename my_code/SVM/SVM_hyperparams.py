from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import numpy as np
from my_code.helpers.datasplit import DataSplit

def determineHyperparams(data):
    params = {'C': np.logspace(-2, 3, 6), 'gamma': np.logspace(-7, 1, 9)}
    dataToUse = data[DataSplit.dev]
    outputs = list(dataToUse['type'].to_numpy())
    sequences = list(dataToUse['sequence'].to_numpy())
    grid = GridSearchCV(SVC(), param_grid=params)
    grid.fit(sequences, outputs)
    print(
        "The best parameters are %s with a score of %0.2f"
        % (grid.best_params_, grid.best_score_)
    )
