import numpy as np
import pandas as pd
import pickle as pkl
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import tqdm

data = pd.read_csv('./bace.csv',sep=',')
print(len(data))
data.head()
data = data.fillna(-1.0)
data.head()

data_smiles = data['smiles'].values.tolist()
split_num = 15
distributions = np.zeros(split_num)
max_length = -1
for smile in data_smiles:
    mol = Chem.MolFromSmiles(smile)
    try:
        Chem.SanitizeMol(mol)
    except:
        continue
    atom_num = mol.GetNumAtoms() // 10
    if atom_num < split_num - 1:
        distributions[atom_num] += 1
    else:
        distributions[-1] += 1
    max_length = mol.GetNumAtoms() if mol.GetNumAtoms() > max_length else max_length

for i in range(split_num):
    if i < split_num - 1:
        print(f'{i*10} - {(i+1) * 10}: {distributions[i]}')
    else:
        print(f'> {i * 10}: {distributions[i]}')
print(f'max length: {max_length}')
print(f'source: {len(data_smiles)}, after sanitize: {np.sum(distributions)}')

element_dict = {}

data_smiles = data['smiles'].values.tolist()
data_labels = data.iloc[:, 1:].values.tolist()

data_san_mol, data_san_label = [], []
for smile, label in zip(data_smiles, data_labels):
    # check the sanitizemol
    mol = Chem.MolFromSmiles(smile)
    try:
        Chem.SanitizeMol(mol)
    except:
        continue

    # delete the molecule number >= 150
    if mol.GetNumAtoms() >= 200:
        continue

    data_san_mol.append(mol)
    data_san_label.append(label)

print(len(data_san_mol), np.array(data_san_label).shape)
with open('preprocess/bace.pickle', 'wb') as fw:
    pkl.dump([data_san_mol, data_san_label],fw)