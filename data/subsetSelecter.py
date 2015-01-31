PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
RANGE = 40428968

f = open(PATH,"r")
f2 = open("subset.csv","w")
f.readline()
for i in range(RANGE):
    l = f.readline()
    if i % 10 == 0 :
        f2.write(l)
    if i % 400000 == 0:
        print(i/RANGE)

f.close()
f2.close()
