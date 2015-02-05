




from Model import Model
from avazuGenerators import baseGenerator, testGenerator
from featureEngineering import crossSiteApp, replaceHour, createInteractionFunc
import pickle
from math import ceil

ALPHA = 0.0000028
NFEATURES = 2**23
NEPOCHS = 1


PATH = "..\\data\\trainIPIID.csv"
TEST_PATH = "..\\data\\testIPID2930.csv"

SIZE_BATCH = 500000
NUM_EXAMPLES_TRAIN = (40428968 - 8051546) / 1
NUM_BATCH_TRAIN = int(NUM_EXAMPLES_TRAIN / SIZE_BATCH)
NUM_EXAMPLES_TEST = 8051546 / 1
NUM_BATCH_TEST = ceil(NUM_EXAMPLES_TEST / SIZE_BATCH)

def save(classifier):
    fid = open("class.pkl","wb")
    pickle.dump(classifier, fid)
    fid.close()
    
COUNTS = [22,89,172,227,52,153,125,207,49,142,21,216,90,199,68,59,97,80,212,29,128]
def testModel(nfeatures, alpha, path, numBatchTrain, batchSize, funcs, testPath, numBatchTest, mustTest, numEpochs, penalty="l1"):
    Class = Model(nfeatures, alpha, numEpochs)
    
    generator = baseGenerator(path, numBatchTrain, batchSize, featCreators=funcs)
    Class.train(generator, v=True)

    if mustTest:
        generator = baseGenerator(testPath, numBatchTest, batchSize, featCreators=funcs)
        y, p = Class.test(generator, v=True)

        score = Class.score(y, p)
    else:
        save(Class)
        score = -1
    return score


def findIndexes(count):
    c=0
    for i in range(23):
        for j in range(i+1,23):
            if c==count: return (i,j)
            c+=1

scores = []
numEpochs = 10
##i, j = findIndexes(89)


##f = open("scoresSaved.csv","w")
##f.write("Scores,i,j\n")
##f.close()
##for i in range(23):
##    for j in range(i+1, 23):
##        numEpochs = 10
##        ALPHA = 0.000025
##        NFEATURES = 2**22
##        funcs = [createInteractionFunc(i,j)]
##        score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
##        scores.append(score)
##        f = open("scoresSaved.csv","a")
##        f.write(str(scores)+","+str(i)+","+str(j)+"\n")
##        f.close()

##alphas = [0.000025,0.0001,0.000075]
##



i = 10
for j in range(1,23):
    
    numEpochs = 10
    ALPHA = 0.000025
    NFEATURES = 2**26
    funcs = [createInteractionFunc(i, j)]
    score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
    scores.append(score)










input()
numEpochs = 10
ALPHA = 0.000025
NFEATURES = 2**28
funcs = []
score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
scores.append(score)



numEpochs = 10
ALPHA = 0.15
NFEATURES = 2**22
funcs = []
score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
scores.append(score)


numEpochs = 10
ALPHA = 0.015
NFEATURES = 2**22
funcs = []
score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
scores.append(score)


numEpochs = 10
ALPHA = 0.0015
NFEATURES = 2**22
funcs = []
score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
scores.append(score)

numEpochs = 10
ALPHA = 0.000025
NFEATURES = 2**26
funcs = []
score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
scores.append(score)


numEpochs = 10
ALPHA = 0.000025
NFEATURES = 2**24
funcs = []
score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
scores.append(score)









##numEpochs = 10
##ALPHA = 0.00005
##NFEATURES = 2**22
##funcs = [createInteractionFunc(i,j)]
##score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
##scores.append(score)



##numEpochs = 10
##ALPHA = bestAlpha
##NFEATURES = 2**22
##funcs = [createInteractionFunc(i,j)]
##score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
##scores.append(score)
##
##numEpochs = 10
##ALPHA = bestAlpha
##NFEATURES = 2**21
##funcs = [createInteractionFunc(i,j)]
##score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
##scores.append(score)
##
##numEpochs = 10
##ALPHA = bestAlpha
##NFEATURES = 2**20
##funcs = [createInteractionFunc(i,j)]
##score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
##scores.append(score)

##numEpochs = 10
##k, l = findIndexes(43)
##funcs = [createInteractionFunc(i,j), createInteractionFunc(k,l)]
##score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
##scores.append(score)
##
##numEpochs = 10
##m, n = findIndexes(21)
##funcs = [createInteractionFunc(i,j), createInteractionFunc(k,l), createInteractionFunc(m, n)]
##score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True, numEpochs)
##scores.append(score)
   
##f = open("scoreSave.csv","w")
##f.write("Score,Count\n")
##f.close()
##for c in COUNTS[8:]:
##    i,j = findIndexes(c)
##    funcs = [replaceHour, createInteractionFunc(i,j)]
##    score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True)
##    scores.append(score)
##    f = open("scoreSave.csv","a")    
##    f.write(str(score)+","+str(c)+"\n")
##    f.close()
##    
##min1 = scores.index(min(scores))
##temp = [i for i in scores]
##temp.pop(min1)
##min2 = scores.index(min(temp))
##
##i1,j1 = findIndexes(COUNTS[min1])
##i2,j2 = findIndexes(COUNTS[min2])
##
##funcs = [replaceHour, createInteractionFunc(i1,j1), createInteractionFunc(i2,j2)]
##score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True)
##scores.append(score)
##f = open("scoreSave.csv","a")    
##f.write(str(score)+","+str(COUNTS[min1])+";"+str(COUNTS[min2])+"\n")
##f.close()
##
##minimum = min(scores)
##if minimum == scores[-1]:
##    funcs = [replaceHour, createInteractionFunc(i1,j1), createInteractionFunc(i2,j2)]
##else:
##    index = scores.index(minimum)
##    i,j = findIndexes(COUNTS[index])
##    funcs = [replaceHour, createInteractionFunc(i,j)]
##
##NUM_EXAMPLES_TRAIN = 40428968
##NUM_BATCH_TRAIN = int(NUM_EXAMPLES_TRAIN / SIZE_BATCH)
##testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, False)
##
##









