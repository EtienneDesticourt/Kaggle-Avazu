from math import ceil
import pickle
from LightModel import LightModel
from avazuGenerators import visitsGenerator

ALPHA =0.000025
NFEATURES = 2**26
NEPOCHS = 10

PATH = "..\\data\\train1IPID.csv"
TEST_PATH = "..\\data\\test1IPID.csv"
SUB_PATH = "..\\data\\test.csv"
OUT_PATH = "..\\data\\submission.csv"

SIZE_BATCH = 500000
NUM_EXAMPLES_TRAIN = (40428968 - 8051546) / 1
NUM_BATCH_TRAIN = ceil(NUM_EXAMPLES_TRAIN / SIZE_BATCH)
NUM_EXAMPLES_TEST = 8051546 / 1
NUM_BATCH_TEST = ceil(NUM_EXAMPLES_TEST / SIZE_BATCH)

NUM_TEST_BATCH = 40

def save(classifier):
    fid = open("class.pkl","wb")
    pickle.dump(classifier, fid)
    fid.close()

load = input("Load or train?(l/t)")


if load == "t":
    Class = LightModel(ALPHA, NEPOCHS, mustShuffle=True)
    #TRAINING
    print("Starting training.")
    generator = visitsGenerator(PATH, NUM_BATCH_TRAIN, SIZE_BATCH)
    Class.train(generator, v=True)

    #TESTING
    print("Starting testing.")
    generator = visitsGenerator(TEST_PATH, NUM_BATCH_TEST, SIZE_BATCH)
    Class.test(generator, v=True)


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
