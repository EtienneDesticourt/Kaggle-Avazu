import numpy as np
from sklearn.feature_extraction import FeatureHasher
from sklearn.linear_model import SGDClassifier
from avazuScorer import llfun
from datetime import datetime
import scipy


class LightModel:
    def __init__(self,learningRate, numEpochs, ppenalty="l1", mustShuffle=True):
        #Init scikit models
        self.Classifier = SGDClassifier(penalty=ppenalty, loss='log', alpha=learningRate, n_iter = numEpochs, shuffle=mustShuffle)
    def train(self, gen,  v=False):
        i = 0
        for x, y in gen: #For each batch
            self.Classifier.partial_fit(x, y, [0,1])
            i += len(x)
            if v : print(str(datetime.now())[:-7] , "example:", i)
            
    def test(self, gen,  v=False):

        #init target and prediction arrays
        ytot = np.array([])
        ptot = np.array([])
        #Get prediction for each batch
        i = 0
        for x,y in gen:
            p = self.Classifier.predict_proba(x)
            p = p.T[1].T #Keep column corresponding to probability of class 1
            #Stack target and prediction for later analysis
            ytot = np.hstack((ytot, y)) 
            ptot = np.hstack((ptot, p))
            i += y.shape[0]
            if v : print(str(datetime.now())[:-7] , "example:", i)
        if v: print("Score:", self.score(ytot, ptot))
        
        return (ytot, ptot)
    def score(self, target, prediction):
        return llfun(target, prediction)
                
