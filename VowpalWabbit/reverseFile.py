
pathOut = "bestFeatures.csv"

f = open(pathOut, "r")
f2 = open("bestFeatures2.csv", "w")
i = 0

a = []
while 1:
    r = f.readline()
    if r == "" : break
    a.append(r)

f.close()



print(a[len(a)-1])

##
##
##for i in range(len(a)):
##    f2.write(a[len(a)-1-i])
##
##f2.close()
