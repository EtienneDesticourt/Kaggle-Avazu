from Model import Model
from avazuGenerators import baseGenerator

#CONSTANTS
ALPHA = 0.0000028
NFEATURES = 2**23
NEPOCHS = 5

COMP_PATH = "..\\data\\trainsetComp.csv"
MOBILE_PATH = "..\\data\\trainsetPhone.csv"
MOBILE_PATH_CV = "..\\data\\trainsetPhone2930.csv"
TEST_PATHS = ["..\\data\\testsetComp.csv", "..\\data\\testsetPhone.csv"]
OUT_PATH = "..\\data\\submission.csv"

SIZE_BATCH = 75000

NUM_EXAMPLES_PHONE_TOTAL = 14596138
NUM_EXAMPLES_PHONE_TRAIN = 14596138#11242146
NUM_EXAMPLES_PHONE_TEST = 3353992
NUM_BATCH_PHONE_TRAIN = int(NUM_EXAMPLES_PHONE_TRAIN / SIZE_BATCH)
NUM_BATCH_PHONE_TEST = int(NUM_EXAMPLES_PHONE_TEST / SIZE_BATCH)

NUM_EXAMPLES_COMP = 25832831
NUM_BATCH_COMP = int(NUM_EXAMPLES_COMP / SIZE_BATCH)

NUM_TEST_BATCH = 40



#TRAIN CLASSIFIER WITH MOBILE DATA
ClassPhone = Model(NFEATURES, ALPHA)
generator = baseGenerator(MOBILE_PATH, NUM_BATCH_PHONE_TRAIN, SIZE_BATCH, polynomial=True)
ClassPhone.train(generator, NEPOCHS, v=True)

##generator = baseGenerator(MOBILE_PATH_CV, NUM_BATCH_PHONE_TEST, SIZE_BATCH)#, polynomial=True)
##ClassPhone.test(generator, v=True)





#TRAIN CLASSIFIER WITH COMPUTER DATA
#input("Go on with computer training?")
ClassComp = Model(NFEATURES, ALPHA)
generator = baseGenerator(COMP_PATH, NUM_BATCH_COMP, SIZE_BATCH, polynomial=True)
ClassComp.train(generator, NEPOCHS, v=True)


##generator = baseGenerator(COMP_PATH, NUM_BATCH_COMP, SIZE_BATCH, holdout=[2,3,4,5,6,7,8,9], polynomial=True)
##ClassComp.test(generator, v=True)


Classifiers = [ClassComp, ClassPhone]



#WRITE SUBMISSION

#input("Go on with submission?")

outputFile = open(OUT_PATH, "w")


i = 0
for j in range(len(Classifiers)):
    Class = Classifiers[j]
    gen = testGenerator(TEST_PATHS[j], NUM_TEST_BATCH, polynomial = True)
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
    
