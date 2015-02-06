






p = "train.vw"
pTrain = "train80.vw"
pTest = "test20.vw"


f = open(p,"r")
fTrain = open(pTrain,"w")
fTest = open(pTest,"w")

i=0
while 1:
    r = f.readline()
    if r =="": break
    if i < 0.8*40000000:
        fTrain.write(r)
    else:
        fTest.write(r)
    if i % 1000000 == 0:
        print(i)
    i += 1

f.close()
fTrain.close()
fTest.close()
    
