import numpy



def loadCSV(path):
    a = []
    f = open(path, "r")
    for i in range(13):
        f.readline()
    while 1:
        r=f.readline()
        if r=="": break
        feats = r.split(":")
        feats[2] = float(feats[2])
        a.append(feats)
    f.close()
    return numpy.array(a)
        
    
def saveSorted(array, outpath):
    f = open(outpath,"w")
    sortedArray = array[array[:,2].argsort()]
    for row in sortedArray:
        string = ",".join(str(row)[1:-1].split(" "))+"\n"
        f.write(string)
    f.close()

pathIn = "readable.model"
pathOut = "bestFeatures.csv"
array = loadCSV(pathIn)
saveSorted(array, pathOut)
    
