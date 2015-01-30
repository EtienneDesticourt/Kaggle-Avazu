

f = open("E:\\Users\\Etienne2\\Downloads\\train.csv","r")
f2 = open("tenth.csv","w")

r = f.readline()
f2.write(r)
RANGE = 40428968
for i in range(RANGE):
    r = f.readline()
    if i % 10  == 0 and i % 100 != 0:
        f2.write(r)
    if i % 40000 == 0:
        print(i/40428968)
        

f.close()
f2.close()
