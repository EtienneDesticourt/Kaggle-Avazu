from Model import Model
from avazuGenerators import baseGenerator, testGenerator
from featureEngineering import crossSiteApp, replaceHour
import pickle
#CONSTANTS
ALPHA = 0.0000028
NFEATURES = 2**23
NEPOCHS = 1

PATH = "..\\data\\train10th.csv"
TEST_PATH = "..\\data\\test293010th.csv"
SUB_PATH = "..\\data\\test.csv"
OUT_PATH = "..\\data\\submission.csv"

SIZE_BATCH = 1000000
NUM_EXAMPLES_TRAIN = (40428968 - 8051546) / 10
NUM_BATCH_TRAIN = int(NUM_EXAMPLES_TRAIN / SIZE_BATCH)
NUM_EXAMPLES_TEST = 8051546 / 10
NUM_BATCH_TEST = int(NUM_EXAMPLES_TEST / SIZE_BATCH)

NUM_TEST_BATCH = 40


load = input("Load or train?(l/t)")

f = open("..\\data\\testScores.csv","w")
f.write("i,j,score,c\n")
if load == "t":
    c = 0
    for i in range(23):
        for j in range(i+1,23):
            print("Iteration:",c)
            
            Class = Model(NFEATURES, ALPHA)
            #TRAINING
            print("Starting training.")
            generator = baseGenerator(PATH, NUM_BATCH_TRAIN, SIZE_BATCH, featCreators=[crossSiteApp, replaceHour])
            Class.train(generator, NEPOCHS, v=True)

            #TESTING
            print("Starting testing.")
            generator = baseGenerator(TEST_PATH, NUM_BATCH_TEST, SIZE_BATCH, featCreators=[crossSiteApp, replaceHour])
            y, p = Class.test(generator, v=True)


            score = Class.score(y, p)

            f.write(str(i)+","+str(j)+","+str(score)+","+str(c)+"\n")            
            c += 1
            
            del y,p
else:
    classFile = open("class.pkl","rb")
    Class = pickle.load(classFile)

#WRITE SUBMISSION
input("Go on with submission?")

outputFile = open(OUT_PATH, "w")
outputFile.write("id,click\n")

i = 0
gen = testGenerator(SUB_PATH, NUM_TEST_BATCH)
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
