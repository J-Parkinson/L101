from my_code.glove.loadGlove import getGlove
import numpy as np

def convertTextToWordEmbeddings(sentenceSplit):
    returnVal = np.array([getGlove(word.lower()) for word in sentenceSplit])
    return returnVal

#Need to split into words first
def addGloveVectors(data, name=None):
    print(f'Calculating GloVe vectors for {name}')
    data['sequenceGlove'] = data['sequence'].apply(lambda x: convertTextToWordEmbeddings(x))
    return data