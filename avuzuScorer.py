import scipy as sp

#DATA LOADING 
def loadRowsR(path, separator, rowRange):
    f = open(path,"r")
    data = []
    for ri in range(rowRange[0]):
        f.readline()
    for ri in range(rowRange[1]):
        data.append(f.readline().split(separator))
    f.close()
    return data[1:]


#PREDICTION SCORING
def llfun(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1-epsilon, pred)
    ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
    ll = ll * -1.0/len(act)
    return ll   

PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
RANGE = [10000,20000]
SEP = ","



