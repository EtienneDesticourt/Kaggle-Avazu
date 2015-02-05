##
##
##f = open("trainIPIID.csv","r")
##RANGE = 40428968
##r = f.readline()
##max0 = 0
##max1 = 0
##while r != "":
##    r = f.readline()
##    if r =="": break
##    l = r.split(',')
##    a = int(l[-2])
##    b = int(l[-1])
##    if a > max0:
##        max0 = a
##    if b > max1:
##        max1 = b
##    
##f.close()
f = open("train.csv","r")


r = f.readline()
dicip = {}
dicid = {}

while r != "":
    r = r.split(",")
    ID = r[11]
    ip = r[12]
    if ID not in dicid:
        dicid[ID] = 0
    if ip not in dicip:
        dicip[ip] = 0

    dicip[ip] += 1
    dicid[ID] += 1
    r = f.readline()

f.close()

f = open("test.csv","r")
f2 = open("testIPID.csv","w")
r = f.readline()
f2.write(r)
RANGE = 40428968
RANGE2 = (40428968 - 8051546)

for i in range(RANGE):
    r = f.readline()
    if r == "": break

    feats = r.split(",")
    ID = feats[11]
    IP = feats[12]
    try:
        countID = dicid[ID]
    except KeyError:
        countID = 0
    try:
        countIP = dicip[IP]
    except KeyError:
        countIP = 0

    countID = str(min(countID, 8))
    countIP = str(min(countIP, 8))
    string = r[:-1]+","+countID+","+countIP+"\n"

    
##    feats = r.split(",")
##    day = feats[2][4:6]
##    if day == "29" or day == "30":
##        f2.write(r)
    
##    if i % 10 == 0:
##        f2.write(r)
##    if i > RANGE2:
##        f2.write(r)
##    day = l[2][4:6]
##    if day == "29" or day == "30" :
##        f2.write(r)
##        #print(i)
    if i % 40000 == 0:
        print(i/404289)
        

f.close()
f2.close()
