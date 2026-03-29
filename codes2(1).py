# =========================================
# FINAL POLYMER PROJECT (100 SAMPLES + API + REAL POLYMER NAMES)
# =========================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error


# =========================================
# STEP 0: API ID
# =========================================

API_KEY = "XauC5crm74AJgstOZWBu3JvfnUYxG999"


# =========================================
# STEP 1: FIX RANDOMNESS
# =========================================

np.random.seed(42)


# =========================================
# STEP 2: POLYMER NAMES LIST (NEW)
# =========================================

polymer_names = [
    "Polyethylene (PE)",
    "Polypropylene (PP)",
    "Polyvinyl Chloride (PVC)",
    "Polystyrene (PS)",
    "Polyethylene Terephthalate (PET)",
    "Nylon-6",
    "Nylon-6,6",
    "Polycarbonate (PC)",
    "Polytetrafluoroethylene (PTFE)",
    "Polyurethane (PU)"
]


# =========================================
# STEP 3: DATASET GENERATION
# =========================================

data = []

for i in range(100):

    monomer_A = np.random.uniform(0.4, 0.8)
    monomer_B = 1 - monomer_A

    molecular_weight = np.random.uniform(1000, 100000)
    atom_count = np.random.randint(50, 5000)
    bond_count = int(atom_count * 1.5)

    functional_group = np.random.randint(1, 20)

    logp = np.random.uniform(-2, 6)
    tpsa = np.random.uniform(20, 200)

    rg = (
        0.002 * molecular_weight +
        0.001 * atom_count +
        0.5 * logp +
        np.random.normal(0, 2)
    )

    fingerprint = np.random.randint(0, 2, 32)

    variance = np.var([
        molecular_weight,
        atom_count,
        bond_count,
        logp,
        tpsa
    ])

    data.append({

        # NEW COLUMN
        "Polymer_Name": polymer_names[i % len(polymer_names)],

        # 10 FEATURES
        "Monomer_A_Ratio": monomer_A,
        "Monomer_B_Ratio": monomer_B,
        "Molecular_Weight": molecular_weight,
        "Atom_Count": atom_count,
        "Bond_Count": bond_count,
        "Functional_Group_Count": functional_group,
        "LogP": logp,
        "TPSA": tpsa,
        "Radius_of_Gyration": rg,
        "Descriptor_Variance": variance,

        "Fingerprint": fingerprint
    })


df = pd.DataFrame(data)


# =========================================
# STEP 4: TANIMOTO SIMILARITY
# =========================================

def tanimoto(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sum(a & b) / np.sum(a | b)


fps = df["Fingerprint"].tolist()

threshold = 0.85
to_remove = set()

for i in range(len(fps)):
    for j in range(i+1, len(fps)):
        if tanimoto(fps[i], fps[j]) > threshold:
            to_remove.add(j)


df_filtered = df.drop(list(to_remove)).reset_index(drop=True)

print("Original dataset size:", len(df))
print("Filtered dataset size:", len(df_filtered))


# =========================================
# STEP 5: ADD IDS
# =========================================

df_filtered.insert(0, "Polymer_ID",
    ["P" + str(i+1).zfill(3) for i in range(len(df_filtered))])

df_filtered.insert(1, "Material_ID",
    ["MAT-" + str(1000+i+1) for i in range(len(df_filtered))])


# =========================================
# STEP 6: ML PREPARATION
# =========================================

fp_df = pd.DataFrame(
    df_filtered["Fingerprint"].tolist(),
    columns=[f"FP_{i}" for i in range(32)]
)

df_ml = pd.concat(
    [df_filtered.drop(columns=["Fingerprint"]), fp_df],
    axis=1
)

X = df_ml.drop(columns=[
    "Radius_of_Gyration",
    "Polymer_ID",
    "Material_ID",
    "Polymer_Name"   # exclude text
])

y = df_ml["Radius_of_Gyration"]


# =========================================
# STEP 7: TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================================
# STEP 8: SCALING
# =========================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# =========================================
# STEP 9: MODEL
# =========================================

model = LinearRegression()
model.fit(X_train, y_train)


# =========================================
# STEP 10: PERFORMANCE
# =========================================

y_pred = model.predict(X_test)

print("\nModel Performance:")
print("R2 Score:", round(r2_score(y_test, y_pred), 3))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 2))


# =========================================
# STEP 11: BEST POLYMER
# =========================================

best_polymer = df_filtered.loc[
    df_filtered["Radius_of_Gyration"].idxmax()
]

print("\nBest Polymer:")
print(best_polymer)


# =========================================
# STEP 12: SAVE
# =========================================

file_name = "finals_polymer_dataset_100_samples.xlsx"
df_filtered.to_excel(file_name, index=False)

print("\nDataset saved as:", file_name)
