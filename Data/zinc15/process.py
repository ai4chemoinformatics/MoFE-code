import pandas as pd
from rdkit import Chem
import pickle
import torch
import csv
import numpy
from torch_geometric.data import Data
path='./zinc15_250K.csv'
skip_smiles = set()
with open(path) as f:
    reader = csv.reader(f)
    next(reader)  # skip header

    lines = []
    for line in reader:
        smiles = line[0]
        if smiles in skip_smiles:
            continue
        lines.append(line)

node_feature=numpy.load('./node_new_feature.npy')
molecules=[]
data_list=[]
for smile in lines:
    mol = Chem.MolFromSmiles(smile[0])
    molecules.append(mol)

with open('./preprocess/zinc15_250K.pickle', 'wb') as f:
     pickle.dump(molecules,f)

