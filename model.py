import numpy as np
import pandas as pd

smiles_npz = np.load("Data_Base_Building/smiles.npz")
print(smiles_npz)
df = pd.read_csv('Data_Base_Building/drug_target_interactions_IC50.csv')
allsmiles = df["SMILES"]


max_shape = 0
for smiles in smiles_npz:
    max_shape = max(max_shape, (smiles_npz[smiles].shape)[0])

print(max_shape)

X = []

for smiles in allsmiles:
    embedding = smiles_npz[smiles]
    shape_embedding = max_shape-embedding.shape[0] 
    padded_embedding = (np.pad(embedding, [(0, shape_embedding), (0, 0)], mode='constant'))
    X.append(padded_embedding)

X = np.array(X)
y = np.array(df["Affinity"])

print(X.shape)  # Should give you (number of 4753, 512, 193)
print(y.shape)  # Should give you (number of 4753,)


