from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import PolynomialFeatures
from avuzuScorer import loadRowsR, PATH, SEP, llfun
import numpy as np


#PREPROCESSING
def genFeatures(data):
    training0 = data[14:24].T
    training1 = data[2:5].T
    training = np.column_stack((training0, training1)).astype(np.float) #; print(training.shape)
    newFeatures = training
##    p = PolynomialFeatures(2)
##    newFeatures = p.fit_transform(training) ; print(newFeatures.shape)
    return newFeatures



#LOAD TRAINING DATA
data = loadRowsR(PATH, SEP, (0,600000)) ; print("Done loading.")
data = np.array(data).T
training = genFeatures(data)
target = data[1].astype(np.float)


#TRAIN MODEL
model  = LR(penalty='l2')
model.fit(training, target) ; print("Done training")


#TEST MODEL
data = loadRowsR(PATH, SEP, (600000,630000)) ; print("Done loading.")
data = np.array(data).T
training = genFeatures(data)
target = data[1].astype(np.float)

prediction = model.predict_proba(training).T[1].T

#CHECK SCORE AND ACCURACY
weightOnes = float(sum(target))/len(target)
weightZeros = 1-weightOnes
weights = np.zeros(len(target))
weights[target==0] = weightZeros
weights[target==1] = weightOnes
print("Accuracy:",model.score(training, target, weights))
print("Score:",llfun(target, prediction))
