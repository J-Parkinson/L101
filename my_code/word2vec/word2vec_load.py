import gensim

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('C:\\Users\jrp32\Documents\Cambridge University\Year III\L101\MainProject\my_code\word2vec\GoogleNews-vectors-negative300.bin', binary=True)

def findSimilarWords(word):
    return model.most_similar(positive=[word])

def getWord2Vector(word):
    if word in model:
        return model[word]
    return None

#print(findSimilarWords('.'))
#print(getWord2Vector('carrot'))