import numpy as np
from math import exp, sqrt


#DATA PREPARATION
class Generator():
    def __init__(self, path, separator):
        self.path = path
        self.sep = separator
        self.file = open(path, "r")
        self.file.readline() #skip column names
    def close(self):
        self.file.close()
    def reset(self):
        self.close()
        self.file = open(self.path, "r")
        self.file.readline() #skip column names
    def getNext(self):
        line = self.file.readline()
        features = np.array(line.split(self.sep)).T #get csv row in array
        target = features[1].astype(np.int) #extract label for this row
        features0 = features[14:24].T #keep relevant features
        features1 = features[2:5].T #keep relevant features
        features = np.hstack((features0, features1)).astype(np.float) #stack into single feature vector
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


#FTRL PROXIMAL
def predict(x, w):
    inner  = x.dot(w.T)
    if inner < - 100: return 0 #to avoid math overflow errors
    return 1. / (1 + exp(-inner))

def getUpdate(zi, ni): 
    if abs(zi) < LAMBDA1: return 0
    sign = abs(zi)/zi
    return (sign * LAMBDA1 - zi) / ( (BETA + sqrt(ni)) / ALPHA + LAMBDA2)

def train(numExamples, numFeats, Gen):
    w = np.zeros(numFeats) #Initialize weights
    z = np.zeros(numFeats)
    n = np.zeros(numFeats)
    #g and sigma created and used in same iteration -> no need to put in vect
    
    for t in range(numExamples):
        x, y = Gen.getNext()
        for i in range(numFeats):
            w[i] = getUpdate(z[i], n[i]) #update weights
        p = predict(x, w)
        #store for next example weights calc
        for i in range(numFeats):
            g = (p-y)*x[i]
            sigma = ( sqrt(n[i] + g**2) - sqrt(n[i]) ) / ALPHA #Learning rate
            z[i] += g - sigma * w[i]
            n[i] += g**2
    return w

#CONSTANTS
ALPHA = 1.
BETA = 1.
LAMBDA1 = 1.
LAMBDA2 = 1.

PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
SEP = ","
TRAINRANGE = 20



Gen = Generator(PATH, SEP)
lengthFeatVect = len(Gen.getNext()[0])
Gen.reset()

w = train(TRAINRANGE, lengthFeatVect, Gen)

print(calcAccuracy(w, Gen, (0,1000)))
