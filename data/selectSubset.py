

f = open("test2930.csv","r")
f2 = open("test293010th.csv","w")
r = f.readline()
f2.write(r)
RANGE = 40428968

for i in range(RANGE):
    r = f.readline()
    if r == "": break
    if i % 10 == 0:
        f2.write(r)
##    l = r.split(",")
##    day = l[2][4:6]
##    if day == "29" or day == "30" :
##        f2.write(r)
        #print(i)
    if i % 40000 == 0:
        print(i/40428968)
        

f.close()
f2.close()
