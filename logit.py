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
data = loadRowsR(PATH, SEP, (0,100000)) ; print("Done loading.")
data = np.array(data).T
training = genFeatures(data)
target = data[1].astype(np.float)


#TRAIN MODEL
model  = LR(penalty='l2',class_weight={0:0.16,1:0.84})
model.fit(training, target) ; print("Done training")


#TEST MODEL
data = loadRowsR(PATH, SEP, (100000,130000)) ; print("Done loading.")
data = np.array(data).T
training = genFeatures(data)
target = data[1].astype(np.float)

prediction = model.predict_proba(training).T[1].T

#CHECK SCORE AND ACCURACY
weightOnes = float(sum(target))/len(target)
weightZeros = 1-weightOnes
weights = np.zeros(len(target))
weights[target==0] = weightOnes
weights[target==1] = weightZeros
print("Accuracy:",model.score(training, target, weights))
print("Score:",llfun(target, prediction))


y = target
p = prediction > 0.5
tn = sum(p[y==0] == y[y==0])
tp = sum(p[y==1] == y[y==1])
print(tn,tp)

