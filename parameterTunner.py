from Model import Model
import matplotlib.pyplot as plt
import time

TRAINPATH = "E:\\Users\\Etienne2\\Downloads\\subset4.csv"
TRAINNUMEXAMPLES = 420000
TRAINSIZEBATCH = 210000
TRAINNUMBATCH = int(TRAINNUMEXAMPLES / TRAINSIZEBATCH)

TESTPATH = "E:\\Users\\Etienne2\\Downloads\\subset5.csv"
TESTNUMEXAMPLES = 42000
TESTSIZEBATCH = 42000
TESTNUMBATCH = int(TESTNUMEXAMPLES / TESTSIZEBATCH)



##learningRate = 0.000005
##numEpochs = 5
##scores = []
##for featExp in range(15,26):
##
##    start = time.time()
##    
##    numFeatures = 2**featExp
##    M = Model(numFeatures, learningRate)
##    print("Training.")
##    M.train(TRAINPATH, TRAINNUMBATCH, TRAINSIZEBATCH, numEpochs)
##    print("Testing.")
##    target, prediction = M.test(TESTPATH, TESTNUMBATCH, TESTSIZEBATCH)
##    score = M.score(target, prediction)
##    scores.append(score)
##
##    end = time.time()
##    print("Time spent this iteration:", end-start, "seconds.")
##
##
##plt.plot(list(range(15,26)), scores)
##plt.show()

numEpochs = 5
scores = []
featExp = 23
for alpha in range(20,40):
    learningRate = alpha / 10000000
    start = time.time()
    
    numFeatures = 2**featExp
    M = Model(numFeatures, learningRate)
    print("Training.")
    M.train(TRAINPATH, TRAINNUMBATCH, TRAINSIZEBATCH, numEpochs)
    print("Testing.")
    target, prediction = M.test(TESTPATH, TESTNUMBATCH, TESTSIZEBATCH)
    score = M.score(target, prediction)
    scores.append(score)

    end = time.time()
    print("Time spent this iteration:", end-start, "seconds.")


plt.plot(list(range(1,100,10)), scores)
plt.show()
