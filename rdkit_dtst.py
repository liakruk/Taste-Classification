import pandas as pd
import numpy as np
import kagglehub as kghub
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Lipinski


def calcular_descritores(df, coluna_smiles):
    smiles_array = df[coluna_smiles].values
    num_mols = len(smiles_array)

    descritores = {
        'MW': np.zeros(num_mols),
        'NumTotalAtoms': np.zeros(num_mols),
        'NumHeavyAtoms': np.zeros(num_mols),
        'NumHeteroAtoms': np.zeros(num_mols),
        'NumOHCount': np.zeros(num_mols),
        'NumRotatableBonds': np.zeros(num_mols),
        'NumRings': np.zeros(num_mols),
        'NumSaturatedRings': np.zeros(num_mols),
        'NumAromaticRings': np.zeros(num_mols),
        'NumAromaticHeterocycles': np.zeros(num_mols),
        'NumAliphaticHeterocycles': np.zeros(num_mols),
        # 'NumSingleBonds': np.zeros(num_mols),
        # 'NumDoubleBonds': np.zeros(num_mols),
        # 'NumTripleBonds': np.zeros(num_mols),
        # 'NumFunctionalGroups': np.zeros(num_mols),
        'NumSP2Carbons': np.zeros(num_mols),
        'NumSP3Carbons': np.zeros(num_mols),
        'NumAliphaticCarbons': np.zeros(num_mols),
        'NumAliphaticRings': np.zeros(num_mols),
        'NumAromaticCarbocycles': np.zeros(num_mols),
        'NumSaturatedCarbocycles': np.zeros(num_mols),
        'NumHDonors': np.zeros(num_mols),
        'NumHAcceptors': np.zeros(num_mols)
    }

    for i, smiles in enumerate(smiles_array):
        try:
            mol = Chem.MolFromSmiles(smiles)
            descritores['MW'][i] = Descriptors.MolWt(mol)
            descritores['NumTotalAtoms'][i] = mol.GetNumAtoms()
            descritores['NumHeavyAtoms'][i] = rdMolDescriptors.CalcNumHeavyAtoms(mol)
            descritores['NumHeteroAtoms'][i] = Lipinski.NumHeteroatoms(mol)
            descritores['NumOHCount'][i] = Lipinski.NHOHCount(mol)
            descritores['NumRotatableBonds'][i] = Lipinski.NumRotatableBonds(mol)
            descritores['NumRings'][i] = Lipinski.RingCount(mol)
            descritores['NumSaturatedRings'][i] = Lipinski.NumSaturatedRings(mol)
            descritores['NumAromaticRings'][i] = Lipinski.NumAromaticRings(mol)
            descritores['NumAromaticHeterocycles'][i] = Lipinski.NumAromaticHeterocycles(mol)
            descritores['NumAliphaticHeterocycles'][i] = Lipinski.NumAliphaticHeterocycles(mol)
            # descritores['NumSingleBonds'][i] = Descriptors.NumSingleBonds(mol)
            # descritores['NumDoubleBonds'][i] = Descriptors.NumDoubleBonds(mol)
            # descritores['NumTripleBonds'][i] = Descriptors.NumTripleBonds(mol)
            # descritores['NumFunctionalGroups'][i] = Descriptors.NumFunctionalGroups(mol)
            descritores['NumSP2Carbons'][i] = sum((x.GetHybridization() == Chem.HybridizationType.SP2) for x in mol.GetAtoms())
            descritores['NumSP3Carbons'][i] = sum((x.GetHybridization() == Chem.HybridizationType.SP3) for x in mol.GetAtoms())
            descritores['NumAliphaticCarbons'][i] = Lipinski.NumAliphaticCarbocycles(mol)
            descritores['NumAliphaticRings'][i] = Lipinski.NumAliphaticRings(mol)
            descritores['NumAromaticCarbocycles'][i] = Lipinski.NumAromaticCarbocycles(mol)
            descritores['NumSaturatedCarbocycles'][i] = Lipinski.NumSaturatedCarbocycles(mol)
            descritores['NumHDonors'][i] = Lipinski.NumHDonors(mol)
            descritores['NumHAcceptors'][i] = Lipinski.NumHAcceptors(mol)
        except:
            descritores['MW'][i] = np.nan
            descritores['NumTotalAtoms'][i] = np.nan
            descritores['NumHeavyAtoms'][i] = np.nan
            descritores['NumHeteroAtoms'][i] = np.nan
            descritores['NumOHCount'][i] = np.nan
            descritores['NumRotatableBonds'][i] = np.nan
            descritores['NumRings'][i] = np.nan
            descritores['NumSaturatedRings'][i] = np.nan
            descritores['NumAromaticRings'][i] = np.nan
            descritores['NumAromaticHeterocycles'][i] = np.nan
            descritores['NumAliphaticHeterocycles'][i] = np.nan
            # descritores['NumSingleBonds'][i] = np.nan
            # descritores['NumDoubleBonds'][i] = np.nan
            # descritores['NumTripleBonds'][i] = np.nan
            # descritores['NumFunctionalGroups'][i] = np.nan
            descritores['NumSP2Carbons'][i] = np.nan
            descritores['NumSP3Carbons'][i] = np.nan
            descritores['NumAliphaticCarbons'][i] = np.nan
            descritores['NumAliphaticRings'][i] = np.nan
            descritores['NumAromaticCarbocycles'][i] = np.nan
            descritores['NumSaturatedCarbocycles'][i] = np.nan
            descritores['NumHDonors'][i] = np.nan
            descritores['NumHAcceptors'][i] = np.nan

    df_descritores = pd.DataFrame(descritores)
    df_final = pd.concat([df[['Name', 'Canonical_SMILES']], df_descritores], axis=1)
    return df_final

#path = kghub.dataset_download("fanconic/smiles-toxicity")

#data_smiles= pd.read_csv(path+'/data/NR-ER-train/names_smiles.csv')
#data_smiles.columns = ['name', 'smiles']


df = pd.read_csv("bitter_sweet_pre-cleanup.csv")
data = df[df['Target'] != 'Non_Bitter_Sweet'][["Name", "Canonical_SMILES"]].reset_index(drop=True)

rdkit = calcular_descritores(data, "Canonical_SMILES")
rdkit = rdkit.drop(columns=["Canonical_SMILES", 'Name'])
rdkit.insert(0, 'cid', range(1, len(rdkit) + 1))

#rdkit.to_csv('rdkit.csv', index=False)

import rdkit
print(rdkit.__version__)