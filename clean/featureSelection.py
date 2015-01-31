
from avazuGenerators import baseGenerator
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
SIZE_BATCH = 50000

NUM_EXAMPLES_PHONE_TOTAL = 14596138
NUM_EXAMPLES_PHONE_TRAIN = 14596138#11242146
NUM_EXAMPLES_PHONE_TEST = 3353992
NUM_BATCH_PHONE_TRAIN = int(NUM_EXAMPLES_PHONE_TRAIN / SIZE_BATCH)
NUM_BATCH_PHONE_TEST = int(NUM_EXAMPLES_PHONE_TEST / SIZE_BATCH)


MOBILE_PATH = "..\\data\\trainsetPhone.csv"
generator = baseGenerator(MOBILE_PATH, NUM_BATCH_PHONE_TRAIN, SIZE_BATCH, polynomial=True)
SKBest = SelectKBest(chi2,40)
for x, y in generator:
    SKBest.fit(x, np.array(y))
    print("New")

