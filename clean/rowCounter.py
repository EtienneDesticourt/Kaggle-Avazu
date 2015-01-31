from avazuGenerators import rawGenerator

PATH = "..\\data\\trainsetPhone.csv"
gen = rawGenerator(PATH)

rowNum = 1 #header
for row in gen:
    rowNum += 1

print(rowNum)



