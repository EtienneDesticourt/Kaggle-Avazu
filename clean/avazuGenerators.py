import numpy as np
import csv
import time
#DATA GENERATION


def baseGenerator(path, numBatches, lengthBatch, holdout=[], polynomial=False, polyRange=[], sep=","):
    trainFile = open(path, "r")
    trainFile.readline() #discard column names
    
    for batchIndex in range(numBatches):
        xBatch = []
        yBatch = []
        exampleIndex = 0

        timeReadRow = 0
        timeHoldout = 0
        timePop = 0
        timePoly = 0
        timeHash = 0
        timeBatch = 0
        
        while exampleIndex < lengthBatch:
            start = time.time()
            #READ ROW
            line = trainFile.readline()
            if line == "": break #break at end of file
            line = line[:-1] #discard line break char
            example = line.split(sep)
            start1 = time.time()
            #MUST HOLDOUT ?
            holdoutFlag = False
            for i in holdout:
                if exampleIndex % i == 0:
                    exampleIndex += 1
                    holdoutFlag = True
                    break
            if holdoutFlag: continue
            start2 = time.time()
            #REMOVE TARGET AND ID
            target = float(example[1]) #get click status
            example.pop(1) #remove target from features            
            example.pop(0) #remove ID
            start3 = time.time()
            #REMOVE SCALING FEATURES
            example.pop(8) #device_id
            example.pop(9) #device_ip
            example.pop(10)#device_model            
            #CREATE POLYNOMIAL FEATURES
##            example.append(example[polyRange[0]]+"_"+example[polyRange[1]])
##            if polynomial:
##                for i in polyRange:
##                    example.append(example[i[0]]+"_"+example[i[1]])
##                for i in polyRange:
##                    for j in range(i+1,polyRange[1]):
##                        example.append(example[i] + "_" + example[j])
            start4 = time.time()
            #PREPARE FOR HASHER
            pairs = [(i,1.0) for i in example]
            example = pairs
            start5 = time.time()
            #ADD TO BATCH
            xBatch.append(example)
            yBatch.append(target)
            exampleIndex +=  1
            start6 = time.time()
            timeReadRow += start1 - start
            timeHoldout += start2 - start1
            timePop += start3 - start2
            timePoly += start4 - start3
            timeHash += start5 - start4
            timeBatch += start6 - start5
##        print("TimeReadRow:", timeReadRow)
##        print("TimeHoldout:", timeHoldout)
##        print("TimePop:", timePop)
##        print("TimePoly:", timePoly)
##        print("TimeHash:", timeHash)
##        print("TimeBatch:", timeBatch)
        if len(xBatch) != 0: #in case of end of file
            yield xBatch, yBatch
            
    trainFile.close()
    


def testGenerator(testPath, numBatches, polynomial=False, sep=","):
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
            #REMOVE SCALING FEATURES
            example.pop(8) #device_id
            example.pop(9) #device_ip
            example.pop(10)#device_model  
            #CREATE POLYNOMIAL FEATURES
            if polynomial:
                originalLength = len(example)
                for i in range(originalLength):
                    for j in range(i+1,originalLength):
                        example.append(example[i] + "_" + example[j])
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

    
    
