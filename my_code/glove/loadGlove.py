import gensim
import numpy as np

# Load Glove
glove = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\jrp32\Documents\Cambridge University\Year III\L101\MainProject\my_code/glove/glove.twitter.27B.200d.txt')

def getGlove(word):
    if word in glove:
        return glove[word]
    return np.zeros(200)