import scipy as sp
import numpy as np

PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
RANGE = [10000,20000]
SEP = ","


#DATA LOADING 
def loadRowsR(path, separator, rowRange):
    f = open(path,"r")
    data = []
    for ri in range(rowRange[0]):
        f.readline() #discard lines before start of range
    for ri in range(rowRange[1]):
        data.append(f.readline().split(separator))
    f.close()
    return data[1:]

def genFeatures(data):
    training0 = data[14:24].T
    training1 = data[2:5].T
    training = np.column_stack((training0, training1)).astype(np.float) #; print(training.shape)
    newFeatures = training
##    p = PolynomialFeatures(2)
##    newFeatures = p.fit_transform(training) ; print(newFeatures.shape)
    return newFeatures

def loadData(path, separator, rowRange):
    rows = loadRowsR(path, separator, rowRange)
    rows = np.array(rows).T
    x = genFeatures(rows)
    y = rows[1].astype(np.float)
    return x, y
#PREDICTION SCORING

def calcWeights(y):    
    weightOnes = float(sum(y))/len(y)
    weightZeros = 1-weightOnes
    weights = np.zeros(len(y))
    weights[y==0] = weightOnes
    weights[y==1] = weightZeros
    return weights

def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll   

def showAccuracy(Classifier, x, y, p, weights):
    print("Accuracy:",Classifier.score(x, y, weights))
    print("Score:",llfun(y, p))
    


