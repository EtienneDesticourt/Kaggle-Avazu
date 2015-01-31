from math import sqrt, exp
import numpy as np
from itertools import islice
import random

#DATA PREPARATION
class Generator():
    def __init__(self, path, separator, numLines):
        self.path = path
        self.sep = separator
        self.numLines = numLines
        self.open()
    def open(self):
        self.file = open(self.path, "r")
        #islice(self.file, 0, 3)
        self.file.readline()
    def close(self):
        self.file.close()
    def reset(self):
        self.close()
        self.open()
    def getNext(self):
        line = self.file.readline()
        features = np.array(line.split(self.sep)).T #get csv row in array
        target = features[1].astype(np.int) #extract label for this row
        features0 = features[14:24].T #keep relevant features
        features1 = features[2:5].T #keep relevant features
        features = np.hstack(([1], features0, features1)).astype(np.float) #stack into single feature vector
        return (features, target)
    def getNew(self):
        r = random.randrange(1,self.numLines)
        for i in islice(self.file, r-1, r): pass
        i = i[:-1] #remove line break
        i = i.split(self.sep)
        slices = [[1], i[14:24], i[2:5]] #Bias and relevant features
        #slices = [[1], i[0]]
        features = np.hstack(slices).astype(np.float)
        target = int(i[1])
        self.reset()
        return (features, target)
        

#MODEL TESTING
def calcAccuracy(weights, Generator, testRange):
    if testRange[0]>=testRange[1]: print("Wrong test range") ; return 0
    lenRange = testRange[1]-testRange[0]
    yDistribution = [0,0]
    #Ready generator
    Generator.reset()
    for t in range(testRange[0]):
        Generator.getNext()
    #Calc accuracy
    accurate = [0,0]
    for t in range(*testRange):
        x, y = Generator.getNext()
        yDistribution[y] += 1
        p = predict(x, weights)
        p = p >= 0.5
        if p == y:
            accurate[y] += 1
    weights = [lenRange/(2.*yDistribution[0]), lenRange/(2.*yDistribution[1])]
    accuracy = accurate[0]*weights[0] + accurate[1]*weights[1]
    return accuracy/(lenRange)


#ONLINE GRADIENT DESCENT
def predict(x, w):
    inner  = x.dot(w.T)
    if inner < - 100: return 0 #to avoid math overflow errors
    return 1. / (1 + exp(-inner))


def train(x, y, w, numExamples, numFeats, Gen):
    for t in range(numExamples):
        p = predict(x, w)
        w -= 2. * ALPHA * (p-y) * x
    return w


#EXECUTE
#CONSTANTS
ALPHA = 0.0001
BETA = 1.
LAMBDA1 = 1.
LAMBDA2 = 1.

PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
SEP = ","
TRAINRANGE = 5



Gen = Generator(PATH, SEP, 1000)
lengthFeatVect = len(Gen.getNext()[0])
Gen.reset()

ERRORTHRESHOLD = 0.1
MAXITER = 100000

w = np.zeros(lengthFeatVect)
for i in range(MAXITER):
    x, y = Gen.getNext()
    w = train(x, y, w, TRAINRANGE, lengthFeatVect, Gen)    
    Gen.reset()
    i += 1
    if  i % (MAXITER/10) == 0: print(i)
    
Gen.reset()
print(w)
print(calcAccuracy(w, Gen, (0,100)))
