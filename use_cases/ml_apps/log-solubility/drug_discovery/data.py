# data.py - load, preprocess, split, tokenize, etc. data.
import os
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem

import network.AttentiveFP.AttentiveLayers, network.AttentiveFP.AttentiveLayers_viz
import network.AttentiveFP.Featurizer, network.AttentiveFP.getFeatures
import network.AttentiveFP.Featurizer_aromaticity_rm, network.AttentiveFP.getFeatures_aromaticity_rm

from network.AttentiveFP.getFeatures import save_smiles_dicts, get_smiles_array
from network.AttentiveFP.AttentiveLayers import Fingerprint

import config

raw_filename = "dataset/delaney-processed.csv"
feature_filename = raw_filename.replace('.csv','.pickle')

filename = raw_filename.replace('.csv','')
prefix_filename = raw_filename.split('/')[-1].replace('.csv','')

smiles_tasks_df = pd.read_csv(raw_filename)
smilesList = smiles_tasks_df.smiles.values

if os.path.isfile(feature_filename):
    feature_dicts = pickle.load(open(feature_filename, "rb" ))
else:
    feature_dicts = save_smiles_dicts(smilesList,filename)

# print("number of all smiles: ",len(smilesList))

atom_num_dist = []
remained_smiles = []
canonical_smiles_list = []

for smiles in smilesList:
    try:        
        mol = Chem.MolFromSmiles(smiles)
        atom_num_dist.append(len(mol.GetAtoms()))
        remained_smiles.append(smiles)
        canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
    except:
        # print(smiles)
        pass

# print("number of successfully processed smiles: ", len(remained_smiles))
smiles_tasks_df = smiles_tasks_df[smiles_tasks_df["smiles"].isin(remained_smiles)]

# print(smiles_tasks_df)
smiles_tasks_df['cano_smiles'] = canonical_smiles_list

plt.figure(figsize=(5, 3))
sns.set(font_scale=1.5)
ax = sns.distplot(atom_num_dist, bins=28, kde=False)

plt.tight_layout()
plt.savefig("atom_num_dist_"+prefix_filename+".png",dpi=200)