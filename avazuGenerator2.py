import numpy as np
import csv
from FeatureEngineering import splitDateFeature, addWorkingFeature, createPolynomialFeatures

#DATA GENERATION
def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b


def generator(path, numBatches, lengthBatch, sep=","):
    trainFile = open(path, "r")
    print(trainFile.readline()) #discard column names
    odd = False #to have evenly distributed classes
    for batchIndex in range(numBatches):
        xBatch = []
        yBatch = []
        for exampleIndex in range(lengthBatch):
            line = trainFile.readline()
            line = line[:-1] #discard line break char
            example = line.split(sep)
            target = float(example[1]) #get click status
            example.pop(1) #remove target from features
##            if (target != odd): continue
            xBatch.append(np.array(example))
            yBatch.append(target)
            odd = not odd
        xBatch = np.array(xBatch)        
        yBatch = np.array(yBatch)
        xBatch, yBatch = shuffle_in_unison(xBatch, yBatch)
        yield xBatch, yBatch

def generator2(path, numBatches, lengthBatch):
    trainFile = csv.DictReader(open(path))
    for batchIndex in range(numBatches):
        xBatch = []
        yBatch = []
        for exampleIndex in range(lengthBatch):
            row = next(trainFile)
            target = float(row["click"]) #get click status
            del row["click"]
            xBatch.append(row)
            yBatch.append(target)
        #xBatch, yBatch = shuffle_in_unison(xBatch, yBatch)
        yield xBatch, yBatch
    trainFile.close()

def generator3(path, numBatches, lengthBatch, sep=","):
    trainFile = open(path, "r")
    trainFile.readline() #discard column names
    odd = False #to have evenly distributed classes
    for batchIndex in range(numBatches):
        xBatch = []
        yBatch = []
        negPerPos = 0
        exampleIndex = 0
        while exampleIndex < lengthBatch:
            line = trainFile.readline()
            if line == "": break
            line = line[:-1] #discard line break char
            example = line.split(sep)
            target = float(example[1]) #get click status
            example.pop(1) #remove target from features
            date = example.pop(1) #remove date
            year = date[:2]
            month = date[2:4]
            day = date[4:6]
            hour = date[6:8]
            example.insert(1, year)
            example.insert(1, month)            
            example.insert(1, day)            
            example.insert(1, hour)
            pairs = [(i,1.0) for i in example]
            example = pairs
##            if (target != odd): continue
##            if (target==0):
##                negPerPos += 1
##            else:
##                negPerPos = 0
##            if (negPerPos >= 10): continue
            xBatch.append(example)
            yBatch.append(target)
            odd = not odd
            exampleIndex +=  1
        if len(xBatch) != 0:
            yield xBatch, yBatch
    trainFile.close()

def generatorWithFreq(path, numBatches, lengthBatch, sep=","):
    trainFile = open(path, "r")
    trainFile.readline() #discard column names
    for batchIndex in range(numBatches):
##        xBatch = np.array([]).reshape(0,23)
##        yBatch = np.array([]).reshape(0,1)
        xBatch = []
        yBatch = []
        exampleIndex = 0
        while exampleIndex < lengthBatch:
            #READ ROW
            line = trainFile.readline()
            if line == "": break #stop if end of file
            line = line[:-1] #discard line break char
            if "True" in line: print(line)
            example = line.split(sep)
            
##            example = [float(feat) for feat in line.split(sep)]
            #REMOVE TARGET
            target = float(example[1]) #get click status
            example.pop(0)
            example.pop(0) #remove target from features
            #ADD TO BATCH
            pairs = [(i,1.0) for i in example] #format for the hasher
            example = pairs
##            xBatch = np.append(xBatch, [example], axis=0)
##            yBatch = np.append(yBatch, [[target]], axis=0)
            xBatch.append(example)
            yBatch.append(target)
            exampleIndex +=  1
        if len(xBatch) != 0:
            yield xBatch, yBatch
    trainFile.close()



def noHashGenerator(path, numBatches, lengthBatch, sep=","):
    trainFile = open(path, "r")
    trainFile.readline() #discard column names
    for batchIndex in range(numBatches):
        xBatch = np.array([]).reshape(0,23)
        yBatch = np.array([]).reshape(0,1)
        exampleIndex = 0
        while exampleIndex < lengthBatch:
            #READ ROW
            line = trainFile.readline()
            if line == "": break #stop if end of file
            line = line[:-1] #discard line break char
            if "True" in line: print(line)
            example = line.split(sep)            
            example = [float(feat) for feat in line.split(sep)]
            #REMOVE TARGET
            target = float(example[1]) #get click status
            example.pop(0) #remove id from features
            example.pop(0) #remove target from features
            #ADD TO BATCH
            xBatch = np.append(xBatch, [example], axis=0)
            yBatch = np.append(yBatch, [[target]], axis=0)
            exampleIndex +=  1
        if len(xBatch) != 0:
            yield xBatch, yBatch
    trainFile.close()



def testGenerator(testPath, numBatches, sep=","):
    testFile = open(testPath,"r")
    line = testFile.readline() #discard header row
    perbatch = 5000000/numBatches
    for batchIndex in range(numBatches):
        xBatch = []
        idBatch = []
        exampleIndex = 0
        while exampleIndex < perbatch:
            line = testFile.readline()
            if line == "": break
            line = line[:-1]
            example = line.split(sep)
            ID = example.pop(0)
##            example = [float(i) for i in example]
##            date = example.pop(1) #remove date
##            year = date[:2]
##            month = date[2:4]
##            day = date[4:6]
##            hour = date[6:8]
##            example.insert(1, year)
##            example.insert(1, month)            
##            example.insert(1, day)            
##            example.insert(1, hour)
            pairs = [(i, 1.) for i in example]
            example = pairs
            exampleIndex += 1
            xBatch.append(example)
            idBatch.append(ID)
        if len(xBatch) != 0:
            yield xBatch, idBatch
    testFile.close()

        
def rawGenerator(path, sep=","):
    "Yields raw split rows with no feature engineering or target separation."
    file = open(path, "r")
    file.readline() #discard column names
    row = file.readline()
    while row != "":
        row = row[:-1] #discard line break char
        features = row.split(sep) #split into features
        yield features
        row = file.readline()
    file.close()

    
