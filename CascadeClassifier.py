import random
import numpy as np
from avuzuScorer import *

#DATA LOADING 
def loadRows(path, separator, numberOfRows):
    f = open(path,"r")
    data = []
    for ri in range(numberOfRows):
        data.append(f.readline().split(separator))
    f.close()
    return data[1:]

#TRAINING 
def predictArray(data, threshold, mod):
    return mod*data >= mod*threshold

def calcFalseNegative(prediction, targets):
    incorrect = prediction != targets
    falseNegative = incorrect[prediction==0]
    return float(sum(falseNegative))/len(targets[targets==1])
    
def calcFalsePositive(prediction, targets):    
    incorrect = prediction != targets
    falseNegative = incorrect[prediction==1]
    return float(sum(falseNegative))/len(targets[targets==0])

def genClassifiers(feature, trainData, targets, classifiers, alpha):
    mini = min(trainData)
    maxi = max(trainData)
    print(mini,maxi)
    #Difference checking variables
    lastFN = 10
    lastFP = 10
    #############
    for mod in [1,-1]:
        threshold = mini
        while threshold < maxi:
            prediction = predictArray(trainData, threshold, mod)
            falseNegative = calcFalseNegative(prediction, targets)
            falsePositive = calcFalsePositive(prediction, targets)
            threshold += alpha
            if falseNegative != lastFN or falsePositive != lastFP:
                classifiers.append([feature, threshold, mod, falseNegative, falsePositive])
                lastFN = falseNegative
                lastFP = falsePositive
    print(lastFN)
            
#EXECUTE TRAINING
PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
NOR = 10000
SEP = ","

fileData = np.array(loadRows(PATH, SEP, NOR)).T
targets = fileData[1].astype(np.float)#[:0.75*len(fileData[1])]

print("Done loading.")

def train():
    classifiers = []
    alpha = [10,1,1,1,1,1,10,1]
    for i in range(16,24):
        trainData = fileData[i].astype(np.float)#[:0.75*len(fileData[i])]
        genClassifiers(i, trainData, targets, classifiers, alpha[i-16])
        print("Trained",str(i),"th classifier.")
    print(len(classifiers),"classifiers.")
    return classifiers

def removeUseless(classifiers):
    for c in classifiers:
        if c[4] == 1: #100% false positive
            classifiers.remove(c)            

def pickBestClassifiers(classifiers, n):
    nc = []
    for i in range(n):
            t = classifiers[i*len(classifiers)/n:(i+1)*len(classifiers)/n] #select each 
            nc.append(t[t[:,4].argmin()])
    nc = np.array(nc)

##    while nc[:,3].max() > 0.49 or nc[:,4].min() < 0.51:
##        nc = nc[:nc[:,3].argmax()]
        
    print(nc[:,3:])
    return nc
##
##classifiers = train()
##removeUseless(classifiers)
##classifiers = np.array(classifiers)
##np.save("avuzu.npy", classifiers)
###################

#EXECUTE TESTING
classifiers = np.load("avuzu.npy")
classifiers = classifiers[classifiers[:,3].argsort()]
classifiers = pickBestClassifiers(classifiers, 100)

##print(classifiers[:,3:])

NOR = 600000
fileData = np.array(loadRows(PATH, SEP, NOR)).T
targets = fileData[1].astype(np.float)#[0.75*len(fileData[1]):]


def calcAccuracy(prediction ,target):
    weights = [len(target)/(2.0*np.sum(target==0)),  len(target)/(2.0*np.sum(target==1))]
    result = prediction == target
    result = result.astype(np.float)
    result[target==0] *= weights[0]
    result[target==1] *= weights[1]        
    accuracy = np.sum(result)/len(target)
    return accuracy


accuracy = np.ones((len(classifiers), len(targets)))
accuracy2 = accuracy.copy()

##nc = []
##for i in range(10):
##	t = classifiers[i*len(classifiers)/10:(i+1)*len(classifiers)/10]
##	nc.append(t[t[:,4].argmin()])
##nc = np.array(nc)
##
##while nc[:,3].max() > 0.49 or nc[:,4].min() < 0.51:
##    nc = nc[:nc[:,3].argmax()]
##    
##print(nc[:,3:])
##classifiers = nc

def test():
    fullPrediction = np.ones(len(targets))
    for cIndex in range(len(classifiers)):
        classifier = classifiers[cIndex]
        feature = classifier[0]
        testData = fileData[feature].astype(np.float)#[0.75*len(fileData[feature]):]
        threshold = classifier[1]
        mod = classifier[2]
        falseNegative = classifier[3]
        falsePositive= classifier[4]
        prediction = predictArray(testData, threshold, mod)
        #print(prediction)
        accuracy[cIndex][prediction==0] = min([falseNegative,1-falsePositive])
        accuracy2[cIndex][prediction==0] = falsePositive
        #break
    #print(accuracy[:,0])
    return accuracy


a = test()

b = a.min(axis=0)
print(b)
c=b.copy()
c[b>0.49] = 1
c[b<=0.49] = 0
d = accuracy2.max(axis=0)


print(llfun(targets, b+(1-d)))
##c = int((15./100)*len(b))
##b[b==b.max()] = 1
##b[b!=1]=0
##print(b)
##
print(calcAccuracy(c, targets))
##print(targets)

    
