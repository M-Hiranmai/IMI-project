from mp_api.client import MPRester
import pandas as pd
import numpy as np

API_KEY = "LbmeAmx3Sqkb9AblV2DEbDJ4kyQuh6C1"
TARGET_SIZE = 100
THRESHOLD = 0.5  # stricter = less redundancy

# ---------- Tanimoto using composition ----------
def tanimoto(comp1, comp2):
    keys = set(comp1.keys()).union(set(comp2.keys()))
    v1 = np.array([comp1.get(k, 0) for k in keys])
    v2 = np.array([comp2.get(k, 0) for k in keys])

    dot = np.dot(v1, v2)
    denom = np.dot(v1, v1) + np.dot(v2, v2) - dot

    return dot / denom if denom != 0 else 0


# ---------- Step 1: Fetch summary ----------
with MPRester(API_KEY) as mpr:

    docs = mpr.materials.summary.search(
        num_elements=(1, 5),
        fields=[
            "material_id",
            "formula_pretty",
            "nsites",
            "volume",
            "formation_energy_per_atom",
            "energy_above_hull",
            "symmetry",
            "structure",
            "composition",
            "elements",
            "total_magnetization"
        ]
    )

    print(f"Total materials fetched: {len(docs)}")

    # ---------- Step 2: Non-redundant selection ----------
    selected = []
    fingerprints = []

    for doc in docs:
        comp_dict = doc.composition.fractional_composition.as_dict()

        is_duplicate = False
        for fp in fingerprints:
            if tanimoto(comp_dict, fp) > THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            selected.append(doc)
            fingerprints.append(comp_dict)

        if len(selected) >= TARGET_SIZE:
            break

    print(f"Selected non-redundant materials: {len(selected)}")

    # ---------- Step 3: Fetch dielectric (ONCE) ----------
    material_ids = [doc.material_id for doc in selected]

    diel_docs = mpr.materials.dielectric.search(
        material_ids=material_ids,
        fields=["material_id", "total"]
    )

    dielectric_dict = {
        d.material_id: d.total if d.total else None
        for d in diel_docs
    }

# ---------- Step 4: Create dataset ----------
data = []

for doc in selected:
    structure = doc.structure
    lattice = structure.lattice

    row = {
        "material_id": doc.material_id,
        "formula": doc.formula_pretty,
        "nsites": doc.nsites,
        "volume": doc.volume,
        "formation_energy_per_atom": doc.formation_energy_per_atom,
        "energy_above_hull": doc.energy_above_hull,
        "alpha": lattice.alpha,
        "beta": lattice.beta,
        "gamma": lattice.gamma,
        "spacegroup": doc.symmetry.symbol if doc.symmetry else None,
        "crystal_system": doc.symmetry.crystal_system if doc.symmetry else None,
        "total_magnetization": doc.total_magnetization,
        "dielectric_constant": dielectric_dict.get(doc.material_id, None)
    }

    # ---------- Element fractions ----------
    comp = doc.composition.fractional_composition.as_dict()
    for el, frac in comp.items():
        row[f"frac_{el}"] = frac

    data.append(row)

df = pd.DataFrame(data)

# ---------- Step 5: Final redundancy check ----------
similarities = []

for i in range(len(fingerprints)):
    for j in range(i + 1, len(fingerprints)):
        similarities.append(tanimoto(fingerprints[i], fingerprints[j]))

print("\n🔍 FINAL REDUNDANCY CHECK")
print("Average:", np.mean(similarities))
print("Max:", np.max(similarities))
print("Min:", np.min(similarities))

# ---------- Save ----------
df.to_csv("final_materials_100.csv", index=False)

print("\n✅ FINAL DATASET READY (100 points, FAST, NON-REDUNDANT)")
