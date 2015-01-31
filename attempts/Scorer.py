

class Scorer:
    def __init__(self):
        pass
    def testModel(self, model, testArgs):
        target, prediction = model.test(*testArgs)
        return (target, prediction)
    def getScore(self, target, prediction):
        return llfun(target, prediction)
    def showResults(self, target, prediction):        
        tp = sum(prediction[target==1] > 0.5) 
        fp = sum(prediction[target==0] > 0.5)
        fn = sum(prediction[target==1] < 0.5)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        print("Precision:", precision)
        print("Recall:", recall)
        print("Score:",llfun(ytot, ptot))
        
