from avazuGenerators import rawGenerator

PATH = "..\\data\\test2930.csv"
gen = rawGenerator(PATH)

rowNum = 1 #header
for row in gen:
    rowNum += 1

print(rowNum)



