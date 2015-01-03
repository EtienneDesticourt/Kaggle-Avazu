import numpy as np

NUMFEATS = 13
INPATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
OUTPATH = "trainNormalized.csv"
SEP = ","

def findExtrema(genFunc, args):
    data = genFunc(*args)
    mini =  np.ones(NUMFEATS)*1000000000
    maxi = -np.ones(NUMFEATS)*1000000000
    i = 0
    for feats in data:
        feats = feats[0][0] # generator returns [[x], [y]]
        #Update extrema
##        print(mini)
##        print(feats)
        mini[mini > feats] = feats[mini > feats] 
        maxi[maxi < feats] = feats[maxi < feats]
        if (i%500000)==0: print(i/404000)
        i += 1
    return (mini, maxi)
        

def normalize(genFunc, args, PATH):
    mini, maxi = findExtrema(genFunc, args)
    data = genFunc(*args)
    f = open(PATH, "w")
    i = 0
    for line in data:
        line  = line[0][0]
        #Normalize 
        dif = line - mini
        length = maxi - mini
        length[length==0] = 1.
        newFeats = dif / length
        f.write(",".join(newFeats.astype(np.str))+"\n")
        if (i%500000)==0: print(i/404000)
        i += 1
        
        
    f.close()

if __name__ == "__main__":
    from avazuGenerator import generator
    NUMBATCH = 40400000
    LENGTHBATCH = 1
    normalize(generator, [INPATH, NUMBATCH, LENGTHBATCH, SEP], OUTPATH)
