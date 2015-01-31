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
        self.file.readline()
    def getNext(self):
        line = self.file.readline()
        features = np.array(line.split(self.sep)).T #get csv row in array
        target = features[1].astype(np.int)
        features = features[16:17].T.astype(np.float)
        #features0 = features[14:24].T #keep relevant features
        #features1 = features[2:5].T #keep relevant features
        #features = np.hstack(([1],features0, features1)).astype(np.float) 
        return (features, target)
        

#MODEL TESTING
def calcAccuracy(Classifier, Generator, testRange):
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
        p = Classifier.predict(x, Classifier.w)
        #print(p)
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



    
class FTRLprox:
    def __init__(self, alpha, beta, delta1, delta2, Generator):
        self.alpha = alpha
        self.beta = beta
        self.delta1 = delta1
        self.delta2 = delta2
        self.Gen = Generator
        self.z = np.zeros(lengthFeatVect) #storage for weight update
        self.n = np.zeros(lengthFeatVect)
        self.w = np.zeros(lengthFeatVect)
    def getX(self):
        return self.Gen.getNext()
    def predict(self, x, w):
        #print(x.shape,w.shape)
        #print(x)
        #print(a)
        z = self.z
        n = self.n
        I = len(x)
        for i in range(I):
            w[i] = self.getUpdatedWeights(z[i], n[i])
        a = x.dot(w.T)
        if a < -100: return 0.
        return 1. / (1. + exp(-x.dot(w.T)))
    def getUpdatedWeights(self, zi, ni):
        if abs(zi) < self.delta1: return 0.
        else:
            sign = abs(zi)/zi
            result = (sign * self.delta1 - zi) / ( (self.beta + sqrt(ni)) / self.alpha + self.delta2)
            return result
    def train(self, numExamples, lengthFeatVect):
        z = self.z
        n = self.n
        w = self.w
        for t in range(numExamples): #for each training example
            x,y = self.getX() #get example and label from generator
            self.z = z
            self.n = n
            p = self.predict(x, w) #
            I = len(x)
            for i in range(I):# for each feature          
                g = (p-y)*x[i]
                sigma = ( sqrt(n[i] + g**2) - sqrt(n[i]) ) / self.alpha #Learning rate
                z[i] += g - sigma * w[i]
                n[i] += g**2
            print(x)
            print(w)
            print(x.dot(w.T))
            print("-----------------------------------")
        self.w = w
        print(w)


ALPHA = 1.
BETA = 1.
DELTA1 = 1.
DELTA2 = 1.


PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
SEP = ","
TRAINRANGE = 100

Gen = Generator(PATH, SEP)
lengthFeatVect = len(Gen.getNext()[0])
train(



##Classifier = FTRLprox(ALPHA, BETA, DELTA1, DELTA2, Gen)
##Classifier.train(TRAINRANGE, lengthFeatVect)
print(calcAccuracy(Classifier, Gen, (0,1000)))
