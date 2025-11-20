from math import log
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import kagglehub as kghub

path = kghub.dataset_download("fanconic/smiles-toxicity")

#Load data
data_smiles = pd.read_csv(path + "/data/NR-ER-train/names_smiles.csv")
data_labels = pd.read_csv(path + "/data/NR-ER-train/names_labels.csv")
rdkit = pd.read_csv("rdkit.csv")
kulik = pd.read_csv("kulik.csv")

#cleaning and normalizing the data
print(kulik.columns)
kulik.drop(["Unnamed: 0", "cid", "name"], axis="columns", inplace=True)
print(kulik.columns)
kulik = kulik.map(lambda x: log(x + sys.float_info.epsilon))

data_labels.columns = ["name", "target"]
data_smiles.columns = ["name", "smiles"]
rdkit = pd.DataFrame({**rdkit, **kulik})
rdkit = rdkit.merge(data_labels, on="name", how="inner")


rdkit.drop(["Unnamed: 0", "name"], axis="columns", inplace=True)
rdkit["MW"] = rdkit["MW"].map(lambda x: log(x + sys.float_info.epsilon))


print(rdkit.shape)
rdkit = rdkit.dropna()
print(rdkit.shape)
print(rdkit.columns)

print(rdkit['target'].sum())

rdkit.to_csv("dataset.csv")
