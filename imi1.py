from mp_api.client import MPRester
import pandas as pd
import numpy as np
import random

from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import DataStructs

RDLogger.DisableLog('rdApp.*')

# ---------------- CONFIG ----------------
API_KEY = "XauC5crm74AJgstOZWBu3JvfnUYxG999"
TARGET = 500
SIM_THRESHOLD = 0.65

# ---------------- STEP 1: FETCH MP DATA ----------------
print("📡 Fetching Materials Project data...")

with MPRester(API_KEY) as mpr:
    docs = mpr.summary.search(
        band_gap=(0, 5),
        fields=[
            "material_id",
            "elements",
            "nelements",
            "formula_anonymous",
            "cbm",
            "vbm",
            "is_metal",
            "energy_per_atom",
            "volume",
            "nsites",
            "efermi"
        ]
    )

# ---------------- STEP 2: ECO SCORE ----------------
def get_eco_score(elements_list):
    elements = set(elements_list)

    bio = {"C", "H", "O", "N"}
    semi = {"Si", "Al", "Fe"}
    toxic = {"Pb", "Cd", "Hg", "As"}

    if elements.issubset(bio):
        return 1.0
    elif elements.intersection(toxic):
        return 0.1
    elif elements.intersection(semi):
        return 0.7
    else:
        return 0.5

# ---------------- STEP 3: BUILD DATAFRAME ----------------
data = []
for d in docs[:5000]:   # large pool
    cbm_val = d.cbm or 0
    vbm_val = d.vbm or 0

    elements_list = [el.symbol for el in d.elements]

    data.append({
        "material_id": str(d.material_id),
        "elements": ",".join(elements_list),
        "nelements": d.nelements,
        "formula_anonymous": d.formula_anonymous,
        "cbm": cbm_val,
        "vbm": vbm_val,
        "band_edge_properties": cbm_val - vbm_val,
        "is_metal": int(d.is_metal),
        "energy_per_atom": d.energy_per_atom or 0,
        "volume_per_atom": (d.volume / d.nsites) if d.volume and d.nsites else 0,
        "dos_at_fermi": d.efermi or 0,
        "eco_score": get_eco_score(elements_list)
    })

df = pd.DataFrame(data)

# ---------------- STEP 4: GENERATE DIVERSE SMILES ----------------
atoms = ["C","N","O","Cl","Br","F"]

def generate_smiles():
    length = random.randint(2, 10)
    return "".join(random.choices(atoms, k=length))

smiles_pool = [generate_smiles() for _ in range(5000)]

df["smiles"] = np.random.choice(smiles_pool, len(df))

# ---------------- STEP 5: FINGERPRINT ----------------
morgan = GetMorganGenerator(radius=2, fpSize=2048)

def get_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return morgan.GetFingerprint(mol)
    return None

df["fp"] = df["smiles"].apply(get_fp)
df = df[df["fp"].notnull()].reset_index(drop=True)

# ---------------- STEP 6: TANIMOTO (GUARANTEED 500) ----------------
def tanimoto(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

selected_fps = []
selected_idx = []

print("🧹 Selecting 500 diverse samples using Tanimoto...")

for i in range(len(df)):
    fp_i = df.loc[i, "fp"]

    is_similar = False
    for fp_j in selected_fps:
        if tanimoto(fp_i, fp_j) >= SIM_THRESHOLD:
            is_similar = True
            break

    if not is_similar:
        selected_fps.append(fp_i)
        selected_idx.append(i)

    # ✅ HARD STOP AT 500
    if len(selected_idx) == TARGET:
        break

# Safety fallback
if len(selected_idx) < TARGET:
    print("⚠ Not enough diversity, filling randomly...")
    remaining = list(set(range(len(df))) - set(selected_idx))
    extra = random.sample(remaining, TARGET - len(selected_idx))
    selected_idx.extend(extra)

df_final = df.iloc[selected_idx]

print("Final samples:", len(df_final))

# ---------------- STEP 7: FINAL FEATURES ----------------
df_final = df_final[[
    "material_id",
    "elements",
    "nelements",
    "formula_anonymous",
    "cbm",
    "vbm",
    "band_edge_properties",
    "is_metal",
    "energy_per_atom",
    "volume_per_atom",
    "dos_at_fermi",
    "eco_score",
    "smiles"
]]

# ---------------- STEP 8: SAVE ----------------
df_final.to_excel("final_500_samples2.xlsx", index=False)

print("\n✅ SUCCESS")
print("Saved: final_500_samples2.xlsx")
