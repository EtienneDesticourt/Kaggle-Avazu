import avazuGenerator2 as ag
import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from avazuScorer import llfun
from datetime import datetime

class Model:
    def __init__(self,numFeatures, learningRate, mustShuffle=True):
        #Init scikit models
        self.FH = FeatureHasher(n_features=numFeatures, input_type='pair')
        self.Classifier = SGDClassifier(loss='log', alpha=learningRate, shuffle=mustShuffle)
    def train(self, path, numBatches, sizeBatch, numEpochs,  v=False):
        gen = ag.generator3(path, numBatches, sizeBatch)
        if v: print("Done generating training set.")

        i = 0
        for x, y in gen: #For each batch
            xHash = self.FH.transform(x) #hash trick
            y = np.array(y)            
            for epoch in range(numEpochs):
                self.Classifier.partial_fit(xHash, y, [0,1])
                
            if v and (i % (numBatches/60)) == 0: print(datetime.now(), "example:", i*sizeBatch)
            i+=1
    def test(self, path, numBatches, sizeBatch,  v=False):
        gen = ag.generator3(path, numBatches, sizeBatch)
        if v: print("Done generating testing set.")

        #init target and prediction arrays
        ytot = np.array([])
        ptot = np.array([])
        #Get prediction for each batch
        for batch in gen:
            data = list(batch) #store batch in memory for prediction
            x, y = data[0], np.array(data[1])
            x = self.FH.transform(x)
            p = self.Classifier.predict_proba(x)
            p = p.T[1].T #Keep column corresponding to probability of class 1
            #Stack target and prediction for later analysis
            ytot = np.hstack((ytot, y)) 
            ptot = np.hstack((ptot, p))

        if v: print("Score:", self.score(ytot, ptot))
        
        return (ytot, ptot)
    def score(self, target, prediction):
        return llfun(target, prediction)
                
