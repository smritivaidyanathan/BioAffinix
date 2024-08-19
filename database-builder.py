from chembl_webresource_client.new_client import new_client
import csv
from tqdm import tqdm
import pandas as pd


molecule = new_client.molecule
activity = new_client.activity

target_id = "CHEMBL4235"

activities = activity.filter(target_chembl_id=target_id, standard_type="IC50")
print("Filtered Activites")
sm_data = []
for record in tqdm(activities, desc="Processing Records and Adding SMILES"):
    chembl_id = record['molecule_chembl_id']
    molecule_obj = molecule.get(chembl_id)
    if (molecule_obj):
        smiles = molecule_obj['molecule_structures']['canonical_smiles']
        affinity = record['standard_value']
        sm_data.append((chembl_id, smiles, affinity))
    else:
        print("Molecule NOT FOUND. Skipping")

with open('Data_Base_Building/drug_target_interactions_IC50.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["CHEMBL_ID", "SMILES", "Affinity"])
    writer.writerows(sm_data)

print("Wrote CSV")

df = pd.read_csv('Data_Base_Building/drug_target_interactions_IC50.csv')

with open('Data_Base_Building/smiles.txt', 'w') as file:
    for item in df["SMILES"]:
        file.write(f"{item}\n")