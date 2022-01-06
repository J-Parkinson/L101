from datetime import datetime

def saveScores(scores, i='', timestamp = datetime.now().strftime('%Y%m%d_%H%M%S'), variance=None):
    TP, FP, TN, FN, recall, selectivity, precision, NPV, accuracy, F1 = scores
    if variance:
        TPVar, FPVar, TNVar, FNVar, recallVar, selectivityVar, precisionVar, NPVVar, accuracyVar, F1Var = variance
    with open(f'{timestamp}_{i}_scores.txt', 'w') as scoresFile:
        if variance:
            scoresFile.write('MEAN MODEL RESULTS~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            scoresFile.write(f'TP:{TP} var:{TPVar}\n')
            scoresFile.write(f'FP:{FP} var:{FPVar}\n')
            scoresFile.write(f'TN:{TN} var:{TNVar}\n')
            scoresFile.write(f'FN:{FN} var:{FNVar}\n')
            scoresFile.write(f'recall     :{recall} var:{recallVar}\n')
            scoresFile.write(f'selectivity:{selectivity} var:{selectivityVar}\n')
            scoresFile.write(f'precision  :{precision} var:{precisionVar}\n')
            scoresFile.write(f'NPV        :{NPV} var:{NPVVar}\n')
            scoresFile.write(f'accuracy   :{accuracy} var:{accuracyVar}\n')
            scoresFile.write(f'F1         :{F1} var:{F1Var}\n')
        else:
            scoresFile.write(f'MODEL {i}~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            scoresFile.write(f'TP:{TP}\n')
            scoresFile.write(f'FP:{FP}\n')
            scoresFile.write(f'TN:{TN}\n')
            scoresFile.write(f'FN:{FN}\n')
            scoresFile.write(f'recall     :{recall}\n')
            scoresFile.write(f'selectivity:{selectivity}\n')
            scoresFile.write(f'precision  :{precision}\n')
            scoresFile.write(f'NPV        :{NPV}\n')
            scoresFile.write(f'accuracy   :{accuracy}\n')
            scoresFile.write(f'F1         :{F1}\n')

def saveTestResult(scores, predicted, dataset, i):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dataset['predicted'] = predicted.cpu().numpy()
    dataset = dataset[['sequenceOriginal', 'type', 'predicted']]
    dataset.to_csv(f'{timestamp}_{i}_predicted.csv', columns=['sequenceOriginal', 'type', 'predicted'])
    saveScores(scores, i, timestamp)