from Model import Model
from avazuGenerators import baseGenerator, testGenerator
from featureEngineering import crossSiteApp, replaceHour, createInteractionFunc
import pickle
from math import ceil
#CONSTANTS
ALPHA = 0.0000028
NFEATURES = 2**23
NEPOCHS = 1

PATH = "..\\data\\train.csv"
TEST_PATH = "..\\data\\test2930.csv"
SUB_PATH = "..\\data\\test.csv"
OUT_PATH = "..\\data\\submission.csv"

SIZE_BATCH = 1500000
NUM_EXAMPLES_TRAIN = 40428968 #(40428968 - 8051546) / 1
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

funcs = [replaceHour]
for i in range(1):
    i = 1
    print(feat1[i],feat2[i])
    funcs.append(createInteractionFunc(feat1[i],feat2[i]))

##f = open("..\\data\\testScores.csv","a")
##f.write("i,j,score,c\n")
if load == "t":
##    c = 0
##    for i in range(11,23):
##        for j in range(i+1,23):
##            print("Iteration:",c)
    Class = Model(NFEATURES, ALPHA)
    #TRAINING
    print("Starting training.")
    generator = baseGenerator(PATH, NUM_BATCH_TRAIN, SIZE_BATCH, featCreators=funcs)
    Class.train(generator, NEPOCHS, v=True)

    #TESTING
##    print("Starting testing.")
##    generator = baseGenerator(TEST_PATH, NUM_BATCH_TEST, SIZE_BATCH, featCreators=funcs)
##    y, p = Class.test(generator, v=True)


##    score = Class.score(y, p)

##    f.write(str(i)+","+str(j)+","+str(score)+","+str(c)+"\n")            
##    c += 1
    
##    del y,p
else:
    classFile = open("class.pkl","rb")
    Class = pickle.load(classFile)

##f.close()
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
