from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np

#CONSTANTS
PATH = "C:\\Users\\Etienne\\Downloads\\avazu\\train\\train.csv"
SEP = ","
TRAINRANGE = 10000

#DATA GENERATION
def generator(path, number, sep):
    trainFile = open(path, "r")
    trainFile.readline() #discard column names
    for i in range(number):
        line = trainFile.readline()
        line = line[:-1] #discard line break char
        array = np.array(line.split(sep))
        example = np.hstack([[1], array[2:5], array[14:24]]) #get relevant features
        example = example.astype(np.float)
        target = array[1].astype(np.float) #get click status
        yield example, target


data = generator(PATH, TRAINRANGE, SEP)

#MODEL TRAINING
Classifier = SGDClassifier()#n_iter=100)
i=0
for x, y in data:
    Classifier.partial_fit([x], [y], [0,1])
    i+=1
    if (i % (TRAINRANGE/10)) == 0: print(i)


#TEST MODEL
data = generator(PATH, TRAINRANGE, SEP)
array = np.array(list(data))


y = array.T[1].astype(np.bool)
print(sum(y))
weights = np.zeros(len(y))
weights[y==0] = 0.5 * sum(y) / TRAINRANGE
weights[y==1] = 1 - 0.5 * sum(y) / TRAINRANGE
#print(weights)
x = np.zeros((TRAINRANGE,14))
for e in range(len(array.T[0])):
    x[e] = array.T[0][e]
    
p = Classifier.score(x, y, weights)
print(p)
