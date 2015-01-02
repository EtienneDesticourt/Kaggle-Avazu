from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import avuzuScorer as ascor
from datetime import datetime

#CONSTANTS
PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
#PATH = "train.csv" 
SEP = ","
TRAINRANGE = 40000000 #Number of examples to include in training
LENGTHBATCH = 100000 #Number of examples per batch
NUMBATCH = int(TRAINRANGE / LENGTHBATCH)
TESTRANGE = [0, 100000]


#DATA GENERATION
def generator(path, numBatches, lengthBatch, sep):
    trainFile = open(path, "r")
    trainFile.readline() #discard column names
    for batchIndex in range(numBatches):
        xBatch = []
        yBatch = []
        for exampleIndex in range(lengthBatch):
            line = trainFile.readline()
            line = line[:-1] #discard line break char
            array = np.array(line.split(sep))
            example = np.hstack([array[2:5], array[14:24]]) #get relevant features
            example = example.astype(np.float)
            target = array[1].astype(np.float) #get click status    
            xBatch.append(example)
            yBatch.append(target)
        xBatch = np.array(xBatch)
        yBatch = np.array(yBatch)
        yield xBatch, yBatch


data = generator(PATH, NUMBATCH, LENGTHBATCH, SEP) ; print("Done generating training set.")
#MODEL TRAINING
Classifier = SGDClassifier(loss='log')#n_iter=100)
i=0
for x, y in data:
    Classifier.partial_fit(x, y, [0,1])
    i+=1
    if (i % (NUMBATCH/100)) == 0: print(datetime.now(), "example:", i*LENGTHBATCH)

#TEST MODEL

x, y = ascor.loadData(PATH, SEP, TESTRANGE) ; print("Done loading test set.")
p = Classifier.predict_proba(x)
p = p.T[1].T #Keep column corresponding to probability of class 1

weights = ascor.calcWeights(y)
ascor.showAccuracy(Classifier, x, y, p, weights)


#TEST MODEL
##data = generator(PATH, TRAINRANGE, SEP)
##array = np.array(list(data))
##
##
##y = array.T[1].astype(np.bool)
###print(sum(y))
##weights = np.zeros(len(y))
##weights[y==0] = 0.5 * sum(y) / TRAINRANGE
##weights[y==1] = 1 - 0.5 * sum(y) / TRAINRANGE
###print(weights)
##x = np.zeros((TRAINRANGE,14))
##for e in range(len(array.T[0])):
##    x[e] = array.T[0][e]
##    
##p = Classifier.score(x, y, weights)
##print(p)
