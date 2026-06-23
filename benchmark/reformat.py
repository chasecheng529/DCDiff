import argparse
import os
import pandas as pd
import subprocess
import json
from rdkit import Chem
import glob
from tqdm import tqdm
import csv
import numpy as np
import torch

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') # Disable RDKit logs.

def load_rdkit_molecule(xyz_path, obabel_path, scaf_sdf_path, true_sdf_path, true_scaf_smi_ori, true_mol_smi_ori):
    # Check whether the file exists.
    if not os.path.exists(obabel_path):
        print(f"File does not exist, skipping: {obabel_path}")
        return None, None, None, None, None

    try:
        # Load molecule.
        supp = Chem.SDMolSupplier(obabel_path, sanitize=False)
        mol = list(supp)[0] if len(supp) > 0 else None
        if mol is None:
            print(f"Failed to load molecule, skipping: {obabel_path}")
            return None, None, None, None, None
    except Exception as e:
        print(f"Error while loading {obabel_path}: {e}")
        return None, None, None, None, None

    # Process molecule.
    mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    mol_filtered = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    try:
        mol_smi = Chem.MolToSmiles(mol_filtered, canonical=True)
    except RuntimeError:
        mol_smi = Chem.MolToSmiles(mol_filtered, canonical=False)

    # Load scaffold file.
    supp = Chem.SDMolSupplier(scaf_sdf_path, sanitize=False)
    true_scaf = list(supp)[0]
    true_scaf_smi = Chem.MolToSmiles(true_scaf)

    # Load true molecule file.
    supp = Chem.SDMolSupplier(true_sdf_path, sanitize=False)
    true_mol = list(supp)[0]
    true_mol_smi = Chem.MolToSmiles(true_mol)
    
    # Check whether the scaffold matches.
    match = mol_filtered.GetSubstructMatch(true_scaf)
    if len(match) == 0: 
        true_scaf = Chem.MolFromSmiles(true_scaf_smi_ori, sanitize=False)
        try:
            Chem.SanitizeMol(mol_filtered)
        except Exception as e:
            print(e)
        mol_filtered_ = Chem.MolFromSmiles(mol_smi, sanitize=True)
        if type(mol_filtered_) != Chem.rdchem.Mol:
            mol_filtered_ = Chem.MolFromSmiles(mol_smi, sanitize=False)

        match_ = mol_filtered_.GetSubstructMatch(true_scaf)
        if len(match_) == 0:
            rgroup_smi = ''
            mol_smi = true_scaf_smi_ori
        else:
            rgroup = Chem.DeleteSubstructs(mol_filtered, true_scaf)
            try:
                Chem.Kekulize(rgroup, clearAromaticFlags=True)
            except Exception:
                pass
            try:
                rgroup_smi = Chem.MolToSmiles(rgroup)
            except RuntimeError:
                rgroup_smi = Chem.MolToSmiles(rgroup, canonical=False)
            
        return mol_filtered, mol_smi, rgroup_smi, true_scaf_smi, true_mol_smi
    else:
        rgroup = Chem.DeleteSubstructs(mol_filtered, true_scaf)
        try:
            Chem.Kekulize(rgroup, clearAromaticFlags=True)
        except Exception:
            pass
        try:
            rgroup_smi = Chem.MolToSmiles(rgroup)
        except RuntimeError:
            rgroup_smi = Chem.MolToSmiles(rgroup, canonical=False)

    return mol_filtered, mol_smi, rgroup_smi, true_scaf_smi, true_mol_smi

def todel_load_molecules(folder, true_scaf_smi_ori, true_mol_smi_ori):
    pred_mols = []
    pred_mols_smi = []
    pred_rgroup_smi = []
    sample_num = 100
    
    scaf_xyz_path = f'{folder}/scaf_.xyz'
    scaf_sdf_path = f'{folder}/scaf_.sdf'
    true_xyz_path = f'{folder}/true_.xyz'
    true_sdf_path = f'{folder}/true_.sdf'

    if not os.path.exists(scaf_sdf_path):
        subprocess.run(f'obabel {scaf_xyz_path} -O {scaf_sdf_path} 2> /dev/null', shell=True)
    if not os.path.exists(true_sdf_path):
        subprocess.run(f'obabel {true_xyz_path} -O {true_sdf_path} 2> /dev/null', shell=True)

    for i in range(sample_num):
        pred_xyz_path = f'{folder}/{str(i)}_.xyz'
        pred_sdf_path = f'{folder}/{str(i)}_.sdf'
        mol, mol_smi, rgroup_smi, true_scaf_smi, true_mol_smi = load_rdkit_molecule(pred_xyz_path, pred_sdf_path, scaf_sdf_path, true_sdf_path, true_scaf_smi_ori, true_mol_smi_ori)

        
        if mol is None:  # Skip this sample if loading fails.
            continue

        pred_mols.append(mol)
        pred_mols_smi.append(mol_smi)
        pred_rgroup_smi.append(rgroup_smi)

    
    return pred_mols, pred_mols_smi, pred_rgroup_smi, true_scaf_smi, true_mol_smi

def load_sampled_dataset(folder, idx2true_mol_smi, idx2true_scaf_smi, idx2true_protein_filename):
    pred_mols = []
    pred_mols_smi = []
    pred_rgroup_smi = []
    true_mols_smi = []
    true_scafs_smi = []
    true_mols_smi_ori = []
    true_scafs_smi_ori = []
    protein_filename_list = []
    max_num = 0

    # Find the largest sample index.
    for fname in os.listdir(folder):
        if fname.isdigit():
            max_num = max(max_num, int(fname))

    for i in range(max_num + 1):
        try:
            true_mol_smi = idx2true_mol_smi[str(i)]
            true_scaf_smi = idx2true_scaf_smi[str(i)]
            protein_filename = idx2true_protein_filename[str(i)]
        except KeyError:
            print(f"Missing sample data, skipping sample index {i}")
            continue

        mols, mols_smi, rgroup_smi, true_scaf_smi_, true_mol_smi_ = load_molecules(f'{folder}/{str(i)}', true_scaf_smi, true_mol_smi)
        
        if not mols:  # Skip if loading fails.
            continue

        pred_mols += mols
        pred_mols_smi += mols_smi
        pred_rgroup_smi += rgroup_smi
        true_mols_smi += [true_mol_smi_] * len(mols)
        true_scafs_smi += [true_scaf_smi_] * len(mols)
        true_mols_smi_ori += [true_mol_smi] * len(mols)
        true_scafs_smi_ori += [true_scaf_smi] * len(mols)
        protein_filename_list += [protein_filename] * len(mols)
    
    return pred_mols, pred_mols_smi, pred_rgroup_smi, true_mols_smi, true_scafs_smi, true_mols_smi_ori, true_scafs_smi_ori, protein_filename_list



def reformat(samples, formatted, true_smiles_path):
    # Read the true_smiles table.
    true_smiles_table = pd.read_csv(true_smiles_path, names=['uuid','molecule_name','molecule','scaffold','rgroups','anchor','pocket_full_size','pocket_bb_size','molecule_size','scaffold_size','rgroup_size', 'protein_filename','affinity'])
    #true_smiles_table = pd.read_csv(true_smiles_path, names=['uuid','molecule_name','molecule','scaffold','rgroups','anchor','pocket_full_size','pocket_bb_size','molecule_size','scaffold_size','rgroup_size', 'protein_filename'])
    # Build mappings from UUID to SMILES and filenames.
    idx2true_mol_smi = dict(zip(true_smiles_table.uuid.values, true_smiles_table.molecule.values))
    idx2true_scaf_smi = dict(zip(true_smiles_table.uuid.values, true_smiles_table.scaffold.values))
    idx2true_protein_filename = dict(zip(true_smiles_table.uuid.values, true_smiles_table.protein_filename.values))

    # Load predicted data.
    pred_mols, pred_mols_smi, pred_rgroup_smi, true_mols_smi, true_scafs_smi, true_mols_smi_ori, true_scafs_smi_ori, protein_filename_list = load_sampled_dataset(
        folder=samples,
        idx2true_mol_smi=idx2true_mol_smi,
        idx2true_scaf_smi=idx2true_scaf_smi,
        idx2true_protein_filename=idx2true_protein_filename,
    )

    # Create the target directory if it does not exist.
    formatted_output_dir = formatted
    if not os.path.exists(formatted_output_dir):
        os.makedirs(formatted_output_dir)

    # Output file paths.
    metric_out_smi_path = os.path.join(formatted_output_dir, 'bingdingnet_test_metric.smi')
    vina_out_smi_path = os.path.join(formatted_output_dir, 'bingdingnet_test_vina.smi')
    out_sdf_path = os.path.join(formatted_output_dir, 'bingdingnet_test_out.sdf')

    # Write metric_out_smi.
    with open(metric_out_smi_path, 'w') as f:
        for i in range(len(pred_mols)):
            f.write(f'{true_scafs_smi[i]} {true_mols_smi[i]} {pred_mols_smi[i]} {pred_rgroup_smi[i]} {protein_filename_list[i]}\n')

    # Write vina_out_smi.
    with open(vina_out_smi_path, 'w') as f:
        for i in range(len(pred_mols)):
            f.write(f'{true_scafs_smi_ori[i]} {true_mols_smi_ori[i]} {pred_mols_smi[i]} {pred_rgroup_smi[i]} {protein_filename_list[i]}\n')

    # Write .sdf file.
    with Chem.SDWriter(out_sdf_path) as writer:
        for mol in pred_mols:
            if mol is not None:
                writer.write(mol)

def create_pred_gt_csv(samples_path, split_file_path = None, gt_csv_path=None, out_pred_gt_csv_path=None):
    # need 'true_molecules','group_id','pred_molecule','core'
    # Return early if the file already exists.
    if os.path.exists(out_pred_gt_csv_path):
        print("File already exists.")
        return
    # Get sample nums
    sample_num = 0
    for fname in os.listdir(samples_path):
        if fname.isdigit():
            sample_num = max(sample_num, int(fname))
    
    # Use test group ids to map sample folders.
    with open(split_file_path, 'r', encoding='utf-8') as f:
            data_split = json.load(f)
    group_id_list = data_split.get('test_group', [])


    all_data_rows = []
    # Read each group.
    for i in range(sample_num + 1):
        group_id = group_id_list[i]
        group_sample_folder = f'{samples_path}/{str((i))}'
        group_pred_smi_list, group_core_smi, ref_smi = load_molecules(group_sample_folder) # Convert sampled SDF files to SMILES.
        gt_smi_list, target_path = load_gt_smi_from_csv(gt_csv_path, group_id)
        target_path = os.path.join('../SAR-DRG', target_path)

        gt_smi_str = ",".join(gt_smi_list)

        for sample_id, pred_smi in enumerate(group_pred_smi_list):
            # Build one CSV row.
            row = {
                'group_id': group_id,
                'core': group_core_smi,
                'true_molecules': gt_smi_str, # Merged ground-truth molecule string.
                'pred_molecule': pred_smi,     # Single predicted molecule.
                'protein_path' : target_path, # Corresponding protein pocket.
                'ref_molecules': ref_smi, # Reference SMILES may not be needed because docking rebuilds 3D structures from SMILES.
                'sample_id': sample_id
            }
            all_data_rows.append(row)

    df = pd.DataFrame(all_data_rows)
    # Save CSV without row indices.
    df.to_csv(out_pred_gt_csv_path, index=False)
    print(f"Saved {len(df)} rows.")

def load_molecules(folder):
    pred_smi_list = []
    sample_num = 100

    # Load group core.
    core_sdf_path = f'{folder}/core_.sdf'
    core_smi = load_mol_from_sdf(core_sdf_path)
    if core_smi is None:
        print("CORE SMI None")
        return None, None, None
    
    # Load group high-reference molecules.
    ref_smi_list = []
    gt_sdf_files = glob.glob(os.path.join(folder, 'gt_*.sdf'))
    for gt_sdf_file in gt_sdf_files:
        ref_smi = load_mol_from_sdf(gt_sdf_file)
        if ref_smi is not None:
            ref_smi_list.append(ref_smi)
    ref_smi = ''
    if len(ref_smi_list) == 0:
        print("REF SMI None")
        return None, None, None
    else:
        for each_ref_smi in ref_smi_list:
            ref_smi = ref_smi + each_ref_smi + ','
        ref_smi = ref_smi[:-1]  # Remove the trailing comma.

    # Load SMILES for all predicted molecules.
    for i in range(sample_num):
        pred_sdf_path = f'{folder}/{str(i)}_.sdf'
        # pred_sdf_path = f'{folder}/{str(i)}.sdf'
        pred_smi = load_mol_from_sdf(pred_sdf_path)
        if pred_smi is None:  # Skip this sample if loading fails.
            continue
        pred_smi_list.append(pred_smi)

    return pred_smi_list, core_smi, ref_smi

def load_mol_from_sdf(sdf_file_path):
    if not os.path.exists(sdf_file_path):
        print(f"File does not exist, skipping: {sdf_file_path}")
        return None
    try:
        # Load molecule.
        supp = Chem.SDMolSupplier(sdf_file_path, sanitize=False)
        if not supp:
            print(f"Failed to load molecule, skipping: {sdf_file_path}")
            return None
        mol = supp[0]
        if mol is None:
            print(f"Failed to load molecule, skipping: {sdf_file_path}")
            return None
    except Exception as e:
        print(f"Error while loading {sdf_file_path}: {e}")
        return None

    # Process molecule.
    try:
        mol_smi = Chem.MolToSmiles(mol, canonical=True)
    except RuntimeError:
        mol_smi = Chem.MolToSmiles(mol, canonical=False)
    return mol_smi

def load_gt_smi_from_csv(gt_csv_path, group_id):
    df = pd.read_csv(gt_csv_path)
    # Convert group ids to strings to avoid int-vs-str mismatches.
    df['group_id'] = df['group_id'].astype(str)
    group_id = str(group_id)
    
    # Select matching rows and extract their SMILES values.
    smi_list = df[df['group_id'] == group_id]['smi'].tolist()
    target_path_list = df[df['group_id'] == group_id]['target_path'].tolist()
    return smi_list, target_path_list[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', action='store', type=str, required=True, default=None)
    parser.add_argument('--formatted', action='store', type=str, required=True, default=None)
    parser.add_argument('--split_config', action='store', type=str, default='../SAR-DRG/dataset_split.json')
    parser.add_argument('--metadata', action='store', type=str, required=False, default='../SAR-DRG/metadata.csv')

    args = parser.parse_args()
    split_config = args.split_config
    metadata = args.metadata 
    samples = args.samples
    formatted = args.formatted
    print("Reference : {}".format(split_config))
    create_pred_gt_csv(samples, split_config, metadata, formatted)
    
