import random
#GENETIC ALGORITHM

#0: 0000
#1: 0001
#2: 0010
#3: 0011
#4: 0100
#5: 0101
#6: 0110
#7: 0111
#8: 1000
#9: 1001
#+: 1010
#-: 1011
#*: 1100
#/: 1101


#Chromosom size: 7 symbols
CSIZE = 5
#Symbol size: 4 digits
SSIZE = 4
#Population
N = 2500
#Number of generations
G = 100000
#Crossover rate
CROSSRATE = 0.9
#Mutation rate
MUTRATE = 0.005
#Target number
TARGET = 53



def weightedChoice(choices):
   total = sum(w for c, w in choices)
   r = random.uniform(0, total)
   upto = 0
   for c, w in choices:
      if upto + w > r:
         return c
      upto += w
   assert False, "Shouldn't get here"



def binary(number):
    r = bin(number)[2:]
    while len(r) < SSIZE:
        r = "0" + r
    return r

    
def createRandomChromosome():
    chrom = ""
    for i in range(CSIZE):
        r = random.randrange(14)
        chrom += binary(r)
    return chrom


def decodeChromosome(chrom,csize=CSIZE, ssize=SSIZE):
    dic = {0:"0", 1:"1", 2:"2", 3:"3", 4:"4", 5:"5", 6:"6", 7:"7", 8:"8", 9:"9", 10:"+", 11:"-", 12:"*", 13:"/"}

    #split chromosome in symbols
    symbols = []
    for i in range(csize):
        symbol = chrom[ssize*i:ssize*(i+1)]
        symbols.append(int(symbol ,2))

    #keep meaningful parts
    newSymbols = []
    last = 0 #0:operator, 1:digit
    for s in symbols:
        if s > 13 or s == 0: continue
        if (s < 10 and not last) or (s >= 10 and last):
            newSymbols.append(dic[s])
            last = not last
    if len(newSymbols)==0: return "0+0"
    if newSymbols[-1] in ["-","+","*","/"]: newSymbols.pop()
    #join into evaluable text
    text = "".join(newSymbols)
    return text

def getFitness(c):

    try:
        result = eval(decodeChromosome(c))
    except ZeroDivisionError:
        result = 10000000

    try:
        fitness = 1 / (TARGET - result)
    except ZeroDivisionError:
        fitness = 10000000
    return fitness

def crossChromosomes(c1, c2, bit):
    c3 = c1[:bit] + c2[bit:]
    c4 = c2[:bit] + c1[bit:]
    return (c3, c4)

def mutateChromosome(c, i):
    bit = c[i]
    if bit == "1": invBit = "0"
    else: invBit = "1"
    
    if i==0: c = invBit + c[1:]
    else: c = c[:i-1] + invBit + c[i:]

    return c
    
population = []
newGen = []
for i in range(N):
    c = createRandomChromosome()
    fitness = getFitness(c)
    population.append((c, fitness))

G = 1000000000
lastfitness = 0
for j in range(G):
    while len(newGen) < N:
        c1 = weightedChoice(population)
        c2 = weightedChoice(population)
##        print(c1)
##        print(decodeChromosome(c1))
##        print(c2)
##        print(decodeChromosome(c2))
        crossover = random.uniform(0,1)
##        print(crossover)
        if crossover < CROSSRATE:
##            print("Crossing over.")
            bit = random.randrange(1, SSIZE*CSIZE)
            c1, c2 = crossChromosomes(c1, c2, bit)
##            print(c1)
##            print(decodeChromosome(c1))
##            print(c2)
##            print(decodeChromosome(c2))
        for i in range(SSIZE*CSIZE):
            mutation1 = random.uniform(0, 1)
##            print(mutation1)
            if mutation1 < MUTRATE:
##                print("Mutating.")
                c1 = mutateChromosome(c1, i)
##                print(c1)
            mutation2 = random.uniform(0, 1)
            if mutation2 < MUTRATE:
                c2 = mutateChromosome(c2, i)
        fitness1 = getFitness(c1)
        fitness2 = getFitness(c2)
##        print(fitness1)
##        print(fitness2)
        newGen.append((c1, fitness1))
        newGen.append((c2, fitness2))
        break
    population = newGen
    temp = [f for c, f in population]
    i = temp.index(max(temp))
    if lastfitness != population[i][1]:
        print("Generation",j)
        print("Best solution:", decodeChromosome(population[i][0]), "Fitness:", population[i][1])
        print("-----------------------------------------------------")
        lastfitness = population[i][1]
    
        
        

