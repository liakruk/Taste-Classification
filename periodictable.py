from mendeleev import element
import pandas as pd

pd1 = pd.read_csv("PD1.csv")
pd1['NumberofValence'] = pd1['NumberofValence'].fillna(0)
data = pd1['NumberofValence']

elements = []
for Z in range(1, 119):  # Elementos até 118
    e = element(Z)
    elements.append({
        "Symbol": e.symbol,
        "AtomicNumber": e.atomic_number,
        "AtomicMass": e.atomic_weight,
        "Electronegativity": e.en_pauling,
        "IonizationEnergy": e.ionenergies.get(1),
        "AtomicRadius": e.atomic_radius,
        "CovalentRadius": e.covalent_radius_cordero,
        "VanDerWaalsRadius": e.vdw_radius
    })

df = pd.DataFrame(elements)
mix = pd.merge(df, pd1[['AtomicNumber', 'NumberofValence']], on='AtomicNumber', how='left')

mix.to_csv("tabela_periodica.csv", index=False)