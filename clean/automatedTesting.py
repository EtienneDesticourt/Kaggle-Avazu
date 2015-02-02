




from Model import Model
from avazuGenerators import baseGenerator, testGenerator
from featureEngineering import crossSiteApp, replaceHour, createInteractionFunc
import pickle
from math import ceil

ALPHA = 0.0000028
NFEATURES = 2**23
NEPOCHS = 1


PATH = "..\\data\\train.csv"
TEST_PATH = "..\\data\\test2930.csv"

SIZE_BATCH = 1500000
NUM_EXAMPLES_TRAIN = (40428968 - 8051546) / 1
NUM_BATCH_TRAIN = int(NUM_EXAMPLES_TRAIN / SIZE_BATCH)
NUM_EXAMPLES_TEST = 8051546 / 1
NUM_BATCH_TEST = ceil(NUM_EXAMPLES_TEST / SIZE_BATCH)

def save(classifier):
    fid = open("class.pkl","wb")
    pickle.dump(classifier, fid)
    fid.close()
    
COUNTS = [22,89,172,227,52,153,125,207]
def testModel(nfeatures, alpha, path, numBatchTrain, batchSize, funcs, testPath, numBatchTest, mustTest):
    Class = Model(nfeatures, alpha)
    
    generator = baseGenerator(path, numBatchTrain, batchSize, featCreators=funcs)
    Class.train(generator, NEPOCHS, v=True)

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
f = open("scoreSave.csv","w")
f.write("Score,Count\n")
f.close()
for c in COUNTS:
    i,j = findIndexes(c)
    funcs = [replaceHour, createInteractionFunc(i,j)]
    score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True)
    scores.append(score)
    f = open("scoreSave.csv","a")    
    f.write(str(score)+","+str(c))
    f.close()
    
min1 = scores.index(min(scores))
temp = [i for i in scores]
temp.pop(min(scores))
min2 = scores.index(min(temp))

i1,j1 = findIndexes(COUNTS[min1])
i2,j2 = findIndexes(COUNTS[min2])

funcs = [replaceHour, createInteractionFunc(i1,j1), createInteractionFunc(i2,j2)]
score = testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, True)
scores.append(score)
f = open("scoreSave.csv","a")    
f.write(str(score)+","+str(COUNTS[min1])+";"+str(COUNTS[min2]))
f.close()

minimum = min(scores)
if minimum == scores[-1]:
    funcs = [replaceHour, createInteractionFunc(i1,j1), createInteractionFunc(i2,j2)]
else:
    index = scores.index(minimum)
    i,j = findIndexes(COUNTS[index])
    funcs = [replaceHour, createInteractionFunc(i,j)]

NUM_EXAMPLES_TRAIN = 40428968
NUM_BATCH_TRAIN = int(NUM_EXAMPLES_TRAIN / SIZE_BATCH)
testModel(NFEATURES, ALPHA, PATH, NUM_BATCH_TRAIN, SIZE_BATCH, funcs, TEST_PATH, NUM_BATCH_TEST, False)











