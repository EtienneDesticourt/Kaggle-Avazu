import numpy as np
import csv

#DATA GENERATION


def baseGenerator(path, numBatches, lengthBatch, sep=","):
    trainFile = open(path, "r")
    trainFile.readline() #discard column names
    
    for batchIndex in range(numBatches):
        xBatch = []
        yBatch = []
        exampleIndex = 0
        while exampleIndex < lengthBatch:
            #READ ROW
            line = trainFile.readline()
            if line == "": break #break at end of file
            line = line[:-1] #discard line break char
            example = line.split(sep)
            #REMOVE TARGET AND ID
            target = float(example[1]) #get click status
            example.pop(1) #remove target from features            
            example.pop(0) #remove ID
            #PREPARE FOR HASHER
            pairs = [(i,1.0) for i in example]
            example = pairs
            #ADD TO BATCH
            xBatch.append(example)
            yBatch.append(target)
            exampleIndex +=  1
            
        if len(xBatch) != 0: #in case of end of file
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

    
