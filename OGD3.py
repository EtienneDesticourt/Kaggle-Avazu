from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import avazuScorer as ascor
from datetime import datetime

#CONSTANTS
PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
PATH2 = "trainNormalized.csv"
#PATH = "train.csv"
NUMROWS = 40428968 #not used
SEP = ","
TRAINRANGE = 100000 #Number of examples to include in training
LENGTHBATCH = 10000 #Number of examples per batch
NUMBATCH = int(TRAINRANGE / LENGTHBATCH)
TESTRANGE = [0, 100000]
EPOCHS = 10

#DATA GENERATION
def generator(path, path2, numBatches, lengthBatch, sep):
    trainFile = open(path, "r")
    trainFile.readline() #discard column names
    trainFile2 = open(path2, "r")
    odd = False
    for batchIndex in range(numBatches):
        xBatch = []
        yBatch = []
        for exampleIndex in range(lengthBatch):
            line = trainFile.readline()
            line = line[:-1] #discard line break char
            array = np.array(line.split(sep))
##            example = np.hstack([array[2:5], array[14:24]]) #get relevant features
##            example = example.astype(np.float)
            target = array[1].astype(np.float) #get click status
            if (target != odd): continue
            line2 = trainFile2.readline()[:-1]
            example = np.array(line2.split(sep)).astype(np.float)
            xBatch.append(example)
            yBatch.append(target)
            odd = not odd
        xBatch = np.array(xBatch)
        np.random.shuffle(xBatch)
        yBatch = np.array(yBatch)
        np.random.shuffle(yBatch)
        yield xBatch, yBatch
        



#MODEL TRAINING
Classifier = SGDClassifier(loss='log', alpha=0.00000003)#, class_weight={0:0.85, 1:0.15})#n_iter=100)

for epoch in range(EPOCHS):
    data = generator(PATH, PATH2, NUMBATCH, LENGTHBATCH, SEP) ; print("Done generating training set.")
    i=0
    print("New epoch:", epoch)
    ones = 0
    zeros = 0
    for x, y in data:
        Classifier.partial_fit(x, y, [0,1])
        ones += sum(y)
        zeros += len(y) - sum(y)
        i+=1
        if (i % (NUMBATCH/10)) == 0: print(datetime.now(), "example:", i*LENGTHBATCH)
    print(ones, zeros)

#TEST MODEL

x, y = ascor.loadData(PATH, SEP, TESTRANGE) ; print("Done loading test set.")
p = Classifier.predict_proba(x)
p = p.T[1].T #Keep column corresponding to probability of class 1

weights = ascor.calcWeights(y)
ascor.showAccuracy(Classifier, x, y, p, weights)


##array([[  1.76287972e+03,  -1.55851733e+02,   3.67080832e+01,
##         -7.56592354e+01,  -1.63434418e+02,  -8.86253253e+04,
##         -2.13243974e+03,   2.12654111e+04,   4.39138358e+03,
##          6.93522679e+01,  -1.24492380e+04,  -2.27176750e+05,
##         -8.53628818e+03]])

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
