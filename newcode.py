from mp_api.client import MPRester
import pandas as pd
import numpy as np
import random

API_KEY = "LbmeAmx3Sqkb9AblV2DEbDJ4kyQuh6C1"

# -----------------------------
# Tanimoto Similarity
# -----------------------------

def tanimoto_similarity(v1, v2):
    v1, v2 = np.array(v1), np.array(v2)
    return np.dot(v1, v2) / (np.dot(v1, v1) + np.dot(v2, v2) - np.dot(v1, v2) + 1e-8)

# -----------------------------
# Fetch Data
# -----------------------------

materials_data = []

with MPRester(API_KEY) as mpr:
    docs = mpr.materials.summary.search(
        num_elements=(1, 5),  # more diversity
        fields=[
            "material_id", "formula_pretty",
            "band_gap", "efermi", "density",
            "ordering",
            "energy_per_atom",
            "homogeneous_poisson",
            "formation_energy_per_atom",
            "energy_above_hull",
            "nelements",
            "num_magnetic_sites",
            "elements"
        ],
        chunk_size=800
    )

    for doc in docs:
        try:
            # Skip Actinium-heavy bias
            if "Ac" in [str(e) for e in doc.elements]:
                continue

            row = {
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "band_gap": doc.band_gap,
                "fermi_energy": doc.efermi,
                "density": doc.density,
                "magnetic_ordering": str(doc.ordering) if doc.ordering else "Unknown",
                "total_energy": doc.energy_per_atom,
                "poisson_ratio": doc.homogeneous_poisson,
                "formation_energy": doc.formation_energy_per_atom,
                "energy_above_hull": doc.energy_above_hull,
                "num_elements": doc.nelements,
                "num_magnetic_sites": doc.num_magnetic_sites
            }

            materials_data.append(row)

        except:
            continue

# -----------------------------
# Convert to DataFrame
# -----------------------------

df = pd.DataFrame(materials_data)

# -----------------------------
# Handle Missing Values
# -----------------------------

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# -----------------------------
# Balance Dataset (IMPORTANT)
# -----------------------------

# Split metals & non-metals
metals = df[df["band_gap"] == 0]
non_metals = df[df["band_gap"] > 0]

# Take balanced samples
metals_sample = metals.sample(min(150, len(metals)), random_state=42)
non_metals_sample = non_metals.sample(min(150, len(non_metals)), random_state=42)

df = pd.concat([metals_sample, non_metals_sample])

# -----------------------------
# Encode Categorical
# -----------------------------

df = pd.get_dummies(df, columns=["magnetic_ordering"])

# -----------------------------
# Remove Redundancy (Tanimoto)
# -----------------------------

numeric_df = df.drop(columns=["material_id", "formula"])

selected_indices = []
vectors = []

for i, row in numeric_df.iterrows():
    vec = row.values
    duplicate = False

    for v in vectors:
        if tanimoto_similarity(vec, v) > 0.95:
            duplicate = True
            break

    if not duplicate:
        selected_indices.append(i)
        vectors.append(vec)

    if len(selected_indices) >= 100:
        break

final_df = df.loc[selected_indices]

# -----------------------------
# Save to Excel
# -----------------------------

final_df.to_excel("final_high_quality_dataset.xlsx", index=False)

print("✅ Final dataset created with", len(final_df), "materials")

