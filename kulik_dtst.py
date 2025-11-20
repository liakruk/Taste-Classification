import pandas as pd
import numpy as np
import kagglehub as kghub
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Lipinski

df = pd.read_csv("bitter_sweet_pre-cleanup.csv")
#print(df['Target'].value_counts())
data = df[df['Target'] != 'Non_Bitter_Sweet']
#print(data['Target'].value_counts())
data_smiles = data[["Name", 'Canonical_SMILES']]

def create_property_descriptors(smiles, depth, prop, prop_index=0):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    # Preparar listas de resultados (separação por depth=0, 1, 2, 3...)
    result_list = [0] * (depth+1) # Se depth = 2, precisamos de 3 itens (para d=[0,1,2])

    # Obter a matriz de distâncias entre cada um dos átomos e a lista de átomos
    atom_list = [ atom.GetSymbol() for atom in mol.GetAtoms() ]
    n_atoms = len(atom_list)
    dist_matrix = np.tril(Chem.rdmolops.GetDistanceMatrix(mol, force=True))

    # Iterar toda a matriz, extraindo os dados de interesse e separando por valor de depth
    for i in range(0, n_atoms):
        for j in range(0, n_atoms):
            d = int(dist_matrix[i][j])

            if d <= depth and d == 0 and i == j:
                a_i = atom_list[i]
                p_i = 0

                if prop[a_i] is list or tuple:
                    p_i = prop[a_i][prop_index]   # caso haja + de 1 propriedade p/ calcular

                else:
                    p_i = prop[a_i]            # caso haja somente 1 propriedade p/ calcular

                result_list[d] += round(p_i * p_i, 3)

            if d <= depth and d > 0:
                a_i = atom_list[i]
                a_j = atom_list[j]
                p_i, p_j = 0, 0

                if prop[a_i] is list or tuple:
                    p_i = prop[a_i][prop_index]   # caso haja + de 1 propriedade p/ calcular
                else:
                    p_i = prop[a_i]            # caso haja somente 1 propriedade p/ calcular
                if prop[a_j] is list or tuple:
                    p_j = prop[a_j][prop_index]
                else:
                    p_j = prop[a_j]
                result_list[d] += round(p_i * p_j, 3)    # posição [d] na lista de resultados é incrementada

    # Finalizando
    return result_list

dataElem = pd.read_csv('PDorig.csv')

# ⚠️ Corrige espaços e formatação dos símbolos químicos
dataElem["Symbol"] = dataElem["Symbol"].astype(str).str.strip().str.capitalize()

# ✅ Usa os símbolos como índice
dataElem = dataElem.set_index("Symbol")

# 🔄 Converte valores para numérico
dataElem = dataElem.apply(pd.to_numeric, errors='coerce')

# 🔧 Cria o dicionário de propriedades com símbolos químicos como chave
properties_dict = dataElem.T.to_dict('list')
print(dataElem.columns.tolist())

RACs_result = []

data_smiles['cid'] = range(1, len(data_smiles) + 1)


for Smiles, cid in zip(data_smiles["Canonical_SMILES"], data_smiles["cid"]):
    try:
        mass = create_property_descriptors(Smiles, 3, properties_dict, 1)
        EN = create_property_descriptors(Smiles, 3, properties_dict, 2)
        In = create_property_descriptors(Smiles, 3, properties_dict, 3)
        aRadius = create_property_descriptors(Smiles, 3, properties_dict, 4)
        VdW = create_property_descriptors(Smiles, 3, properties_dict, 5)
        covRadius = create_property_descriptors(Smiles, 3, properties_dict, 6)
        valence = create_property_descriptors(Smiles, 3, properties_dict, 7)

        dict_RACs = {
            'cid': cid,
            'mass dZero': mass[0],
            'mass dOne': mass[1],
            'mass dTwo': mass[2],
            'mass dThree': mass[3],
            'EN dZero': EN[0],
            'EN dOne': EN[1],
            'EN dTwo': EN[2],
            'EN dThree': EN[3],
            'In dZero': In[0],
            'In dOne': In[1],
            'In dTwo': In[2],
            'In dThree': In[3],
            'aRadius dZero': aRadius[0],
            'aRadius dOne': aRadius[1],
            'aRadius dTwo': aRadius[2],
            'aRadius dThree': aRadius[3],
            'VdW dZero': VdW[0],
            'VdW dOne': VdW[1],
            'VdW dTwo': VdW[2],
            'VdW dThree': VdW[3],
            'covRadius dZero': covRadius[0],
            'covRadius dOne': covRadius[1],
            'covRadius dTwo': covRadius[2],
            'covRadius dThree': covRadius[3],
            'valence dZero': valence[0],
            'valence dOne': valence[1],
            'valence dTwo': valence[2],
            'valence dThree': valence[3]
        }

        RACs_result.append(dict_RACs)

    except:

        dict_RACs = {
            'cid': cid,
            'mass dZero': np.nan,
            'mass dOne': np.nan,
            'mass dTwo': np.nan,
            'mass dThree': np.nan,
            'EN dZero': np.nan,
            'EN dOne': np.nan,
            'EN dTwo': np.nan,
            'EN dThree': np.nan,
            'In dZero': np.nan,
            'In dOne': np.nan,
            'In dTwo': np.nan,
            'In dThree': np.nan,
            'aRadius dZero': np.nan,
            'aRadius dOne': np.nan,
            'aRadius dTwo': np.nan,
            'aRadius dThree': np.nan,
            'VdW dZero': np.nan,
            'VdW dOne': np.nan,
            'VdW dTwo': np.nan,
            'VdW dThree': np.nan,
            'covRadius dZero': np.nan,
            'covRadius dOne': np.nan,
            'covRadius dTwo': np.nan,
            'covRadius dThree': np.nan,
            'valence dZero': np.nan,
            'valence dOne': np.nan,
            'valence dTwo': np.nan,
            'valence dThree': np.nan
        }

        RACs_result.append(dict_RACs)

RACs = pd.DataFrame(RACs_result)

RACs.to_csv('kulik.csv', index=False)