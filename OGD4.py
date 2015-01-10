import avazuGenerator2 as ag
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from datetime import datetime
from avazuScorer import llfun, logloss
#CONSTANTS
PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
NUMEXAMPLES = 1000000
BATCHSIZE = 200000
NUMBATCHES = int(NUMEXAMPLES / BATCHSIZE)
NUMFEATS = 23
TESTSIZE = 400000


EPOCHS = 5
ALPHA = 0.000001
NFEATURES = 2**24
##########


#MODEL TRAINING
FH = FeatureHasher(n_features=NFEATURES, input_type='pair')
Classifier = SGDClassifier(loss='log', alpha=ALPHA, shuffle=True)#, class_weight={0:0.45, 1:0.55})#, class_weight={0:0.85, 1:0.15})#n_iter=100)

gen = ag.generator3(PATH, NUMBATCHES, BATCHSIZE) ; print("Done generating training set.")

i=0
for x, y in gen:    
    xHash = FH.transform(x)
    y = np.array(y)
    
    for epoch in range(EPOCHS):
        #xHash, y = shuffle(xHash, y)
        Classifier.partial_fit(xHash, y, [0,1])
    i+=1
    if (i % (NUMBATCHES/10)) == 0: print(datetime.now(), "example:", i*BATCHSIZE)


del x
del y
#TEST MODEL
gen = ag.generator3(PATH, 1, TESTSIZE) ; print("Done generating training set.")
data = list(gen)
x = data[0][0]

x = FH.transform(x)
y = np.array(data[0][1])
p = Classifier.predict_proba(x)
p = p.T[1].T #Keep column corresponding to probability of class 1

tp = sum(p[y==1] > 0.5) 
fp = sum(p[y==0] > 0.5)
fn = sum(p[y==1] < 0.5)

precision = tp / (tp + fp)
recall = tp / (tp + fn)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:",Classifier.score(x, y))
print("Score:",llfun(y, p))

