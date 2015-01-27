import avazuGenerator2 as ag
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from datetime import datetime
from avazuScorer import llfun, logloss


#IDEAS
#hour separation DID NOT WORK
#feature combinations
#parameter generation with genetic algorithm MEH
#gpu implementation cuda


#CONSTANTS
##PATH = "E:\\Users\\Etienne2\\Downloads\\subset3.csv"
PATH = "E:\\Users\\Etienne2\\Downloads\\train.csv"
LOCALTESTPATH = "subset2.csv"
OUTPATH = "submission.csv"
TESTPATH = "test.csv"

NUMEXAMPLES = 42000000
BATCHSIZE = 100000
NUMBATCHES = int(NUMEXAMPLES / BATCHSIZE)
NUMFEATS = 23

TESTSIZE = 4000000
TESTBATCHSIZE = 100000 
NUMTESTBATCHES = int(TESTSIZE / TESTBATCHSIZE)

EPOCHS = 5
ALPHA = 0.000005
NFEATURES = 2**24
##ALPHA = 0.0000028
##NFEATURES = 2**23
##########


#MODEL TRAINING
FH = FeatureHasher(n_features=NFEATURES, input_type='pair')
Classifier = SGDClassifier(loss='log', alpha=ALPHA, shuffle=True)#, class_weight={0:0.45, 1:0.55})#, class_weight={0:0.85, 1:0.15})#n_iter=100)

gen = ag.generator3(PATH, NUMBATCHES, BATCHSIZE) ; print("Done generating training set.")

i=0
for x, y in gen: 
    xHash = FH.transform(x) #hash trick
    y = np.array(y)
    
    for epoch in range(EPOCHS):
        #xHash, y = shuffle(xHash, y)
        Classifier.partial_fit(xHash, y, [0,1])
    i+=1
    if (i % (NUMBATCHES/10)) == 0: print(datetime.now(), "example:", i*BATCHSIZE)


del x, y

#TEST MODEL
gen = ag.generator3(LOCALTESTPATH, NUMTESTBATCHES, TESTBATCHSIZE) ; print("Done generating testing set.")


ytot = np.array([])
ptot = np.array([])
for b in gen:
    data = list(b)
    x = data[0]

    x = FH.transform(x)
    y = np.array(data[1])
    p = Classifier.predict_proba(x)
    p = p.T[1].T #Keep column corresponding to probability of class 1
    ytot = np.hstack((ytot, y))
    ptot = np.hstack((ptot, p))

    
tp = sum(ptot[ytot==1] > 0.5) 
fp = sum(ptot[ytot==0] > 0.5)
fn = sum(ptot[ytot==1] < 0.5)

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("\n")
print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:",Classifier.score(x, y))
print("Score:",llfun(ytot, ptot))

del x, y, tp, fp, fn, p, data


#WRITE SUBMISSION
input("Submission follows.")
f = open(OUTPATH, "w")
f.write("id,click\n") #header row
NUMBATCH = 40
gen = ag.testGenerator(TESTPATH, NUMBATCH) ; print("Done generating submission set.")

i = 0
for x in gen:
    xHash = FH.transform(x)
    p = Classifier.predict_proba(xHash)
    text = ""
    for j in range(len(x)):
        ID = x[j][0][0]
        click = p[j][1]
        text += str(ID)+","+str(click)+"\n"
    f.write(text)
    i += 1
    print(100*i/NUMBATCH,"%")
f.close()



