from FTRLProx import Generator, PATH, SEP
import numpy as np


Gen = Generator(PATH, SEP)
a = Gen.getNext()[0]
Gen.reset()
b = Gen.getNext()[0]
print(a)
print(a[0])
print("-------------------")
for i in range(len(a)):
    print(a[i],":",b[i])
