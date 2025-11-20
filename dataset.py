from math import log
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import kagglehub as kghub

df = pd.read_csv("bitter_sweet_pre-cleanup.csv")
data = df[df['Target'] != 'Non_Bitter_Sweet']
data['cid'] = range(1, len(data) + 1)

rdkit = pd.read_csv("rdkit.csv")
kulik = pd.read_csv("kulik.csv")
kulik.drop(columns=['cid'], errors='ignore')

data = data[['cid', 'Target']]
kulik = kulik.map(lambda x: log(x + sys.float_info.epsilon))
kulik["cid"] = rdkit["cid"]

mix = rdkit.merge(kulik, on='cid').merge(data, on='cid')
if "MW" in mix.columns:
    mix["MW"] = mix["MW"].map(lambda x: log(x + sys.float_info.epsilon))

#cleaning and normalizing the data
print(kulik.columns)
kulik.drop(["cid"], axis="columns", inplace=True)
print(kulik.columns)

mix.to_csv("dataset.csv")