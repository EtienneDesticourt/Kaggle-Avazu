import math

def zygmoid(x):
    #I know it's a common Sigmoid feature, but that's why I probably found
    #it on FastML too: https://github.com/zygmuntz/kaggle-stackoverflow/blob/master/sigmoid_mc.py
    return 1 / (1 + math.exp(-x))

with open("submission.csv","wb") as outfile:
    outfile.write(bytes("id,click\n", "UTF-8"))
    i=0
    for line in open("avazu.preds.txt"):
##        print(line)
        row = line.strip().split(" ")
##        print(row)
        outfile.write(bytes("%s,%f\n"%(row[1],zygmoid(float(row[0]))), "UTF-8"))
        i+=1
        if i%100000==0:print(i)
