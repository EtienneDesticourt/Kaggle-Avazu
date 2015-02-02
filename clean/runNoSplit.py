from Model import Model
from avazuGenerators import baseGenerator, testGenerator
from featureEngineering import crossSiteApp, replaceHour, createInteractionFunc
import pickle
from math import ceil
#CONSTANTS
##ALPHA = 0.0000028
##NFEATURES = 2**23
##NEPOCHS = 15
ALPHA =0.000025
NFEATURES = 2**22
NEPOCHS = 10

PATH = "..\\data\\train.csv"
TEST_PATH = "..\\data\\test2930.csv"
SUB_PATH = "..\\data\\test.csv"
OUT_PATH = "..\\data\\submission.csv"

SIZE_BATCH = 1500000
NUM_EXAMPLES_TRAIN = 40428968#(40428968 - 8051546) / 1
NUM_BATCH_TRAIN = ceil(NUM_EXAMPLES_TRAIN / SIZE_BATCH)
NUM_EXAMPLES_TEST = 8051546 / 1
NUM_BATCH_TEST = ceil(NUM_EXAMPLES_TEST / SIZE_BATCH)

NUM_TEST_BATCH = 40

def save(classifier):
    fid = open("class.pkl","wb")
    pickle.dump(classifier, fid)
    fid.close()

load = input("Load or train?(l/t)")

#CREATE INTERACTING FEATURES
feat1 = [8, 2, 1, 4, 9, 15, 2, 8, 6, 12, 2]
feat2 = [16, 3, 2, 12, 20, 18, 12, 14, 15, 22, 9]


def findIndexes(count):
    c=0
    for i in range(23):
        for j in range(i+1,23):
            if c==count: return (i,j)
            c+=1
            
##funcs = []
##for i in range(1):
##    i = 3
##    print(feat1[i],feat2[i])
##    funcs.append(createInteractionFunc(feat1[i],feat2[i]))

i, j = findIndexes(89)
funcs = [createInteractionFunc(i,j)]


if load == "t":
    Class = Model(NFEATURES, ALPHA, NEPOCHS, mustShuffle=True)
    #TRAINING
    print("Starting training.")
    generator = baseGenerator(PATH, NUM_BATCH_TRAIN, SIZE_BATCH, featCreators=funcs)
    Class.train(generator, v=True)

##    #TESTING
##    print("Starting testing.")
##    generator = baseGenerator(TEST_PATH, NUM_BATCH_TEST, SIZE_BATCH, featCreators=funcs)
##    Class.test(generator, v=True)


else:
    classFile = open("class.pkl","rb")
    Class = pickle.load(classFile)


#WRITE SUBMISSION
input("Go on with submission?")

outputFile = open(OUT_PATH, "w")
outputFile.write("id,click\n")

i = 0
gen = testGenerator(SUB_PATH, NUM_TEST_BATCH, featCreators = funcs)
predictionGen = Class.generatePrediction(gen)

for pBatch, idBatch in predictionGen:
    outputText = ""
    for example in range(len(pBatch)):
        ID = idBatch[example]
        click = pBatch[example][1] #Class 1
        outputText += str(ID) + "," + str(click) + "\n"
    outputFile.write(outputText)
    print(100 * i /  NUM_TEST_BATCH, "%")
    i += 1

outputFile.close()
