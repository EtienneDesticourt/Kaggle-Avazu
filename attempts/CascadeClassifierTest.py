import random
import numpy as np

def loadRows(path, separator, numberOfRows):
    f = open(path,"r")
    data = []
    for ri in range(numberOfRows):
        data.append(f.readline().split(separator))
    f.close()
    return data[1:]

class Classifier():
    def __init__(self):
        self.threshold = 0
    def predict(self, value):
        return value >= self.threshold
    def predictArray(self,data):
        return self.mod*data >= self.mod*self.threshold
    def calcAccuracy(self,data,target):
        weights = [len(target)/(2.0*np.sum(target==0)),  len(target)/(2.0*np.sum(target==1))]
##        print(weights)
        size = len(data)
        predicted = self.predictArray(data)
        result = predicted == target
        result = result.astype(np.float)
        result[target==0] *= weights[0]
        result[target==1] *= weights[1]        
        accuracy = np.sum(result)/size
##        print(accuracy)
        return accuracy
    def calcFalseNegative(self, data, target):
        prediction = self.predictArray(data)
        correct = prediction == target
        noClick = correct[target==0]
        return 1- float(sum(noClick))/len(noClick)
    def calcFalsePositive(self, data, target):
        prediction = self.predictArray(data)
        correct = prediction == target
        noClick = correct[target==1]
        return 1- float(sum(noClick))/len(noClick)
    def train(self, data, target, alpha):
        mini = min(data)
        maxi = max(data)
##        start = random.randrange(mini,maxi)
##        way = random.choice([1,-1])
        leastError = 1
        leastErrorThres = mini
        bestMod = 1
        for mod in [1,-1]:
            curr = mini
            while curr < maxi:
                self.threshold = curr
                self.mod = mod
                error = 1 - self.calcAccuracy(data, target)
                if error < leastError:
##                    print("e:",error)
##                    print("le:",leastError)
##                    print("t:",curr)
##                    print("----------------------")
                    leastError = error
                    leastErrorThres = curr
                    bestMod = mod
                curr += alpha
        self.threshold = leastErrorThres
        self.error = leastError
        self.mod = bestMod
    def trainFN(self, data, target, alpha, index):
##        f = open("C:\\Users\\Etienne\\Downloads\\avazu\\train\\Cdata"+str(index)+".csv","w")
        mini = min(data)
        maxi = max(data)
        leastFN = 1
        leastFNThres = mini
        bestMod = 1
        t=""
        for mod in [1,-1]:
            curr = mini
            while curr < maxi:
                self.threshold = curr
                self.mod = mod
                falseNegative = self.calcFalseNegative(data, target)
                falsePositive = self.calcFalsePositive(data, target)
##                t += str(falseNegative)+","+str(falsePositive)+","+str(curr)+"\n"
                if falseNegative < leastFN:
                    leastFN = falseNegative
                    leastFNThres = curr
                    bestMod = mod
                curr += alpha
##        f.write(t)        
##        f.close()
        self.threshold = leastFNThres
        self.error = leastFN
        self.mod = bestMod
        
        
PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
NOR = 100
SEP = ","

fileData = np.array(loadRows(PATH, SEP, NOR)).T

Classifiers = []
target = fileData[1].astype(np.float)
alphas = [1,1,1,1,1,1,1,1]
for i in range(16,24):
    C = Classifier()
    data = fileData[i].astype(np.float)
    C.trainFN(data, target, alphas[i-16],i)
    #if i==1: C.threshold *= -1
    print("A:",C.calcAccuracy(data, target))
    print("FN:",C.calcFalseNegative(data, target))
    print("FP:",C.calcFalsePositive(data, target))
    print("------------------------------")
##    print("T:",C.threshold)
    Classifiers.append(C)
    

##Classifiers.remove(Classifiers[1])
class CascadeClassifier():
    def __init__(self, classifiers):
        self.classifiers = classifiers
    def predict0(self,data):
        result = 0
        for i in range(len(self.classifiers)):
            c = self.classifiers[i]
            if c.predict(data[i]):
                result += (1-c.error)
            else:
                result -= (1-c.error)
        if result > 0:print(result)
        return result < - 1
    def predict(self, data):
        result = 0
        for i in range(len(self.classifiers)):
            c = self.classifiers[i]
            result += c.predict(data[i])
        return result >= (len(self.classifiers)-1 or 1)
            


C = CascadeClassifier(Classifiers)

def testClassifier(C, data, target):
    accurate = 0
    total = 0
    weights = [len(target)/(2.0*np.sum(target==0)), len(target)/(2.0*np.sum(target==1))]
    for i in range(len(data)):
        value = data[i]
        total += 1
        if (C.predict(value)==target[i]):
            accurate += 1*weights[int(target[i])]
    print(accurate/total*100)

def solve(C, data):
    pass
    
newData = fileData[16:24].T.astype(np.float)
testClassifier(C, newData, target)

TESTPATH = "C:\\Users\\Etienne\\Downloads\\avazu\\test\\test.csv"
fileData = np.array(loadRows(PATH, SEP, NOR)).T
newData = fileData[16:24].T.astype(np.float)


    
