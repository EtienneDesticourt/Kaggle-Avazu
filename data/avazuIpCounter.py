

f = open("train.csv","r")
f2 = open("trainIPIID.csv","w")
r = f.readline()
f2.write(r[:-1]+",IP Count,ID Count\n")
RANGE = 40428968



ips = {}
ids = {}
i = 0
while r != "":
    r = f.readline()
    ip = r.split(",")[12]
    if ip not in ips:
        ips[ip] = 0    
    r = r[:-1]+","+str(ips[ip])
    ID = r.split(",")[11]
    if ID not in ids:
        ids[ID] = 0    
    r += ","+str(ids[ID])+"\n"
    f2.write(r)
    ips[ip] += 1
    ids[ID] += 1
    i += 1
    if i % 40000 == 0:
        print(i/40428968)
        

f.close()
f2.close()
