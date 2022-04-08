import pandas as pd
import numpy as np
from random import shuffle

df = pd.read_csv("0SelectedSMILES_QM9.txt")
smiles_list = df.to_numpy()
print(smiles_list)
np.random.shuffle(smiles_list)
print(smiles_list)
validation = smiles_list[0:10000]
training = smiles_list[10000:]
with open("validation_set.txt", "w") as f1:
    #lines = f1.readlines
    for elem in validation:
        f1.writelines(elem[0] + "," + elem[1] + "\n")
        
with open("training_set.txt", "w") as f2:
    #lines = f1.readlines
    for elem in training:
        f2.writelines(elem[0] + "," +elem[1] + "\n")
