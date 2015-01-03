import numpy as np

def generator(path, numBatches, lengthBatch, sep):
    trainFile = open(path, "r")
    trainFile.readline() #discard column names
    for batchIndex in range(numBatches):
        xBatch = []
        yBatch = []
        for exampleIndex in range(lengthBatch):
            line = trainFile.readline()
            line = line[:-1] #discard line break char
            array = np.array(line.split(sep))
            example = np.hstack([array[2:5], array[14:24]]) #get relevant features
            example = example.astype(np.float)
            target = array[1].astype(np.float) #get click status    
            xBatch.append(example)
            yBatch.append(target)
        xBatch = np.array(xBatch)
        yBatch = np.array(yBatch)
        yield xBatch, yBatch
