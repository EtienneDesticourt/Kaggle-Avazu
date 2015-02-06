


f0 = open("0.csv","r")
f1 = open("1.csv","r")
f2 = open("2.csv","r")
f3 = open("3.csv","r")
f4 = open("sub.csv","w")

fs = [f0,f1,f2,f3]
for i in fs:
    r = i.readline()

f4.write(r)



while 1:
    rs = []
    for i in fs:
        rs.append(i.readline()[:-1].split(","))

    ID = rs[0][0]

    tot = 0
    for i in rs:
        tot += float(i[1])
    tot = tot / 4

    f4.write(ID+","+str(tot)+"\n")

    
