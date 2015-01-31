

f = open("trainsetPhone.csv","r")

r = f.readline()
RANGE = 40428968

for i in range(RANGE):
    r = f.readline()
    if r == "": break
    l = r.split(",")
    day = l[2][4:6]
    if day == "29" or day == "30" :
        print(i)
    if i % 40000 == 0:
        print(i/40428968)
        

f.close()
f2.close()
