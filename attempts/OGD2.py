import numpy as np
import random
from itertools import islice
import warnings
warnings.filterwarnings('error')
from math import exp, sqrt

#Generator
class Generator:
    def __init__(self, path, sep):
        self.path = path
        self.sep = sep
        self.open()
        self.numLines = self.countLines()
    def countLines(self):
##        for i, l in enumerate(self.file): pass
##        return i+1
        return 1000
    def open(self):
        self.file = open(self.path, "r")        
    def close(self):
        self.file.close()
    def reset(self):
        self.close()
        self.open()
    def getNew(self):
        r = random.randrange(1,self.numLines)
        for i in islice(self.file, r-1, r): pass
        i = i[:-1] #remove line break
        i = i.split(self.sep)
        #slices = [[1], i[14:24], i[2:5]] #Bias and relevant features
        slices = [[1], i[0]]
        features = np.hstack(slices).astype(np.float)
        target = int(i[1])
        self.reset()
        return (features, target)
        


#MODEL
y = np.array(range(10)) + 5
x = np.array(range(10))

x2 = np.array(range(20))
def predict(x, w):
    return x.dot(w.T)

def predictBin(x, w):
    a = x.dot(w.T)
    if a < -30: return 0.
    return 1. / (1 + exp(-a))  

def train(w, numExamples, ALPHA=0.0000000000001):
    for e in range(numExamples):
        x, y = Gen.getNew()
        p = predictBin(x, w)
##        try:            
        w -= 2 * ALPHA * (p-y) * x
##        print(p)
##        except Warning:
##            print(x, w)
    return w
        
#RUN
PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
#PATH = "train2.csv"
SEP = ","

Gen = Generator(PATH, SEP)
w = np.zeros(2)

N = 1
for i in range(N):
    w = train(w, 1000)
    if (i % (N/10)) == 0:
        print(i)
print(w)

##for i in x2:
##    print(i,":",predictBin(np.array([1, i]), w)>0.5)
