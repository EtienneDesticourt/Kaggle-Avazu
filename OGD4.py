import avazuGenerator2 as ag
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
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
#Feature engineering ideas:
#   binary working feature : week end or week <9am >5pm:0 else:1
#SGD with no hashing
#polynomial features
#one hot encoding instead of hash trick
#smoothing
#polynomial features without continuous features

#CONSTANTS
##PATH = "E:\\Users\\Etienne2\\Downloads\\subset3.csv"
PATH = "tenth.csv"
LOCALTESTPATH = "hundredth.csv"
OUTPATH = "submission.csv"
TESTPATH = "testfreq.csv"

NUMEXAMPLES = 4200000
BATCHSIZE = 500000
NUMBATCHES = int(NUMEXAMPLES / BATCHSIZE)
NUMFEATS = 23

TESTSIZE = 420000
TESTBATCHSIZE = 105000 
NUMTESTBATCHES = int(TESTSIZE / TESTBATCHSIZE)

EPOCHS = 5
##ALPHA = 0.000005
##NFEATURES = 2**24
ALPHA = 0.0000028
NFEATURES = 2**23
##########


#MODEL TRAINING
FH = FeatureHasher(n_features=NFEATURES, input_type='pair')
Classifier = SGDClassifier(loss='log', alpha=ALPHA, shuffle=True)#, class_weight={0:0.45, 1:0.55})#, class_weight={0:0.85, 1:0.15})#n_iter=100)

##gen = ag.generator3(PATH, NUMBATCHES, BATCHSIZE) ; print("Done generating training set.")
gen = ag.generatorWithFreq(PATH, NUMBATCHES, BATCHSIZE) ; print("Done generating training set.")
PF = PolynomialFeatures(interaction_only = True, include_bias=False)
i=0
for x, y in gen:
##    print(x.shape)
##    print(x[0])
##    x = PF.fit_transform(x)
##    print(len(x), len(x[0]))
    xHash = FH.transform(x) #hash trick
##    y = np.array(y)
##    xHash = x
    for epoch in range(EPOCHS):
        #xHash, y = shuffle(xHash, y)
        Classifier.partial_fit(xHash, y, [0,1])
    i+=1
    print(datetime.now(), "example:", i*BATCHSIZE)


del x, y

#TEST MODEL
##gen = ag.generator3(LOCALTESTPATH, NUMTESTBATCHES, TESTBATCHSIZE) ; print("Done generating testing set.")
gen = ag.generatorWithFreq(LOCALTESTPATH, NUMTESTBATCHES, TESTBATCHSIZE) ; print("Done generating testing set.")


ytot = np.array([])
ptot = np.array([])
for b in gen:
    data = list(b)
    x = data[0]
##    x = PF.fit_transform(x)

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
##
##print(ytot.shape, ptot.shape)

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
for x, idBatch in gen:
    xHash = FH.transform(x)
##    xHash = x
    p = Classifier.predict_proba(xHash)
    text = ""
    for j in range(len(x)):
        ID = idBatch[j]
        click = p[j][1]
        text += str(ID)+","+str(click)+"\n"
    f.write(text)
    i += 1
    print(100*i/NUMBATCH,"%")
f.close()



