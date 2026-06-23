import numpy as np
import pandas as pd
import sys

import utils.sascore as sascore  # 确保路径正确
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem.QED import qed
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit import Chem
import os
import torch

np.random.seed(0)
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
logger = rkl.logger()
logger.setLevel(rkl.ERROR)
rkrb.DisableLog('rdApp.error')

# -------------- Validity -------------- #
def is_valid(pred_mol_smiles, scaf_smiles):
    if pred_mol_smiles == '':
        return False
    # In the case of generated mol is two individual parts
    if '.' in pred_mol_smiles:
        return False
    pred_mol = Chem.MolFromSmiles(pred_mol_smiles)
    core = Chem.MolFromSmiles(scaf_smiles)
    if core is None:
        core = Chem.MolFromSmiles(scaf_smiles, sanitize=False)
    if pred_mol is None:
        pred_mol = Chem.MolFromSmiles(pred_mol_smiles, sanitize=False)
        if pred_mol is None:
            return False
    if len(pred_mol.GetSubstructMatch(core)) != core.GetNumAtoms():
        return False
    return True

def get_valid_score(data,  group_id_list):
    val_dict = {}
    for i in range(len(group_id_list)):
        val_dict[group_id_list[i]] = []
        
    for obj in data:
        valid = is_valid(obj['pred_molecule'], obj['core'])
        obj['valid'] = valid
        val_dict[obj['group_id']].append(valid)

    avg_tmp = []
    for k, v in val_dict.items():
        if len(v) == 0:
            continue
        avg_tmp.append(sum(v) / len(v))
    validity = sum(avg_tmp) / len(avg_tmp) * 100
    return validity

# -------------- Uniqueness -------------- #
def get_unique_score(data,  group_id_list):
    #创建字典，映射group-id和所属的预测出来的分子
    uni_dict = {}
    for i in range(len(group_id_list)):
        uni_dict[group_id_list[i]] = []
    
    group_smi_dict = dict()
    for obj in data:
        if not obj['valid']:
            continue
        group_smi_dict.setdefault(obj["group_id"], []).append(obj['pred_molecule'])

    for k, samples in group_smi_dict.items():
        uni_dict[k].append(len(set(samples)) / len(samples))

    avg_tmp = []
    for k, v in uni_dict.items():
        if len(v) == 0:
            continue
        avg_tmp.append(sum(v) / len(v))
    if len(avg_tmp) == 0:
        uniqueness = 0
    else:
        uniqueness = sum(avg_tmp) / len(avg_tmp) * 100

    return uniqueness

# -------------- Similarity ---------------#
def get_smi_score(data,  group_id_list):
    sim_dict = {}
    group_gt_dict = {}
    for i in range(len(group_id_list)):
        sim_dict[group_id_list[i]] = []

    for obj in data: #遍历每一个生成的分子
        # 首先获取这个group的所有GT
        if obj['group_id'] not in group_gt_dict:
            gt_fingerprint_list = []
            #构建gt的fingerprint
            for true_smi in obj['true_molecules']:
                true_mol = Chem.MolFromSmiles(true_smi)
                if true_mol is None:
                    true_mol = Chem.MolFromSmiles(true_smi, sanitize=False)
                if true_mol is None:
                    print("ERROR REF MOL: {}, ID:{}".format(true_smi,obj['group_id']))
                    exit()
                    
                gt_fingerprint_list.append(Chem.RDKFingerprint(true_mol))
            group_gt_dict[obj['group_id']] = gt_fingerprint_list
        
        #从cache里面拿gt的fingerprint
        group_gt_fingerprints = group_gt_dict[obj['group_id']]

        gt_similarities = []
        if not obj['valid']:
            continue
        pred_mol = Chem.MolFromSmiles(obj['pred_molecule'])
        if pred_mol is None:
            pred_mol = Chem.MolFromSmiles(obj['pred_molecule'], sanitize=False)
        pred_fingerprint = Chem.RDKFingerprint(pred_mol)
        for gt_fingerprint in group_gt_fingerprints:
            sim = DataStructs.FingerprintSimilarity(pred_fingerprint, gt_fingerprint)
            gt_similarities.append(sim)
        sim_dict[obj['group_id']].append(max(gt_similarities))

    avg_tmp = []
    for k, v in sim_dict.items():
        if len(v) == 0:
            continue
        avg_tmp.append(sum(v) / len(v))
    
    if len(avg_tmp) == 0:
        similarity = 0
    else:
        similarity = sum(avg_tmp) / len(avg_tmp) * 100
    return similarity

# ----------------- Recovery --------------#
def get_recovery_score(data, group_id_list):
    group_gt_dict = {}
    recovery_dict = {}
    for i in range(len(group_id_list)):
        recovery_dict[group_id_list[i]] = False
    for obj in data:
        if obj['group_id'] not in group_gt_dict:
            gt_smiles_set = set()
            #构造所有的gt的mol
            for gt_smiles in obj['true_molecules']:
                try:
                    true_mol = Chem.MolFromSmiles(gt_smiles)
                    Chem.RemoveStereochemistry(true_mol)
                    true_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(true_mol))
                except:
                    true_mol = Chem.MolFromSmiles(gt_smiles, sanitize=False)
                    Chem.RemoveStereochemistry(true_mol)
                    true_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(true_mol, sanitize=False))
                gt_smiles_set.add(true_mol_smi)
            group_gt_dict[obj['group_id']] = gt_smiles_set
        group_gt_smiles_set = group_gt_dict[obj['group_id']]
        if not obj['valid']:
            obj['recovered'] = False
        else:
            #构建预测的分子的SMILES
            try:
                pred_mol = Chem.MolFromSmiles(obj['pred_molecule'])
                Chem.RemoveStereochemistry(pred_mol)
                pred_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(pred_mol))
            except:
                pred_mol = Chem.MolFromSmiles(obj['pred_molecule'], sanitize=False)
                Chem.RemoveStereochemistry(pred_mol)
                pred_mol_smi = Chem.MolToSmiles(Chem.RemoveHs(pred_mol, sanitize=False))
            #命中一个就算recovery
            obj['recovered'] = pred_mol_smi in group_gt_smiles_set
        recovery_dict[obj['group_id']] = recovery_dict[obj['group_id']] or obj['recovered']
    recovered_count = 0
    group_count = 0
    for k, v in recovery_dict.items():
        if v:
            recovered_count += 1
        group_count += 1
    recovery_socre = recovered_count / group_count * 100
    return  recovery_socre


# -------------- SAS Score ---------------- #
def compute_sas(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        return sascore.calculateScore(mol)
    except Exception as e:
        print("ERROR",e)
        return None

def get_sas_score(data, group_id_list):
    sas_dict = {gid: [] for gid in group_id_list}

    for obj in data:
        if not obj['valid']:
            obj['sas'] = None
            continue
        sas = compute_sas(obj['pred_molecule'])
        if sas is not None:
            sas = (10 - sas) / 9
            sas_dict[obj['group_id']].append(sas)
            obj['sas'] = sas
        else:
            obj['sas'] = None
    avg_tmp = []
    for k, v in sas_dict.items():
        if len(v) == 0:
            continue
        avg_tmp.append(sum(v) / len(v))
    sas_avg = sum(avg_tmp) / len(avg_tmp)
    return sas_avg

# -------------- QED Score ---------------- #
def compute_qed(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        return qed(mol)
    except:
        return None

def get_qed_score(data, group_id_list):
    qed_dict = {gid: [] for gid in group_id_list}
    for obj in data:
        if not obj['valid']:
            obj['qed'] = None
            continue
        qed_score = compute_qed(obj['pred_molecule'])
        if qed_score is not None:
            qed_dict[obj['group_id']].append(qed_score)
            obj['qed'] = qed_score
        else:
            obj['qed'] = None

    avg_tmp = []
    for k, v in qed_dict.items():
        if len(v) == 0:
            continue
        avg_tmp.append(sum(v) / len(v))
    qed_avg = sum(avg_tmp) / len(avg_tmp)
    return qed_avg
    # print(f'Avg QED (Quantitative Estimate of Drug-likeness): {qed_avg:.3f}')

# -------------- Lipinski's Rule of Five ---------------- #
def compute_lipinski(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    try:
        # Lipinski's Rule of Five criteria:
        # 1. Molecular weight < 500 Da
        # 2. LogP <= 5
        # 3. Number of hydrogen bond donors <= 5
        # 4. Number of hydrogen bond acceptors <= 10
        # 5. Number of rotatable bonds <= 10
        mw = Descriptors.ExactMolWt(mol) < 500
        logp = Descriptors.MolLogP(mol) <= 5
        hbd = Lipinski.NumHDonors(mol) <= 5
        hba = Lipinski.NumHAcceptors(mol) <= 10
        rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
        
        # Count how many criteria are met
        lipinski_count = sum([mw, logp, hbd, hba, rotatable])
        return lipinski_count
    except:
        return None

def get_avg_lipinski_score(data, group_id_list):
    lipinski_dict = {gid: [] for gid in group_id_list}

    for obj in data:
        if not obj['valid']:
            obj['lipinski'] = None
            continue
        lipinski_score = compute_lipinski(obj['pred_molecule'])
        if lipinski_score is not None:
            lipinski_dict[obj['group_id']].append(lipinski_score)
            obj['lipinski'] = lipinski_score
        else:
            obj['lipinski'] = None

    avg_tmp = []
    for k, v in lipinski_dict.items():
        if len(v) == 0:
            continue
        avg_tmp.append(sum(v) / len(v))
    lipinski_avg = sum(avg_tmp) / len(avg_tmp)
    return lipinski_avg
    # print(f'Avg Lipinski (Rule of Five criteria met): {lipinski_avg:.3f}')

# -------------- Lipinski Full Compliance (5/5) rate ---------------- #
def get_full_lipinski_rate(data, group_id_list):
    lipinski_full_dict = {gid: [] for gid in group_id_list}

    for obj in data:
        if obj.get('lipinski') is not None:
            # 如果完全满足5个规则，记录为True，否则为False
            full_compliance = (obj['lipinski'] == 5)
            lipinski_full_dict[obj['group_id']].append(full_compliance)

    avg_tmp = []
    for k, v in lipinski_full_dict.items():
        if len(v) == 0:
            continue
        # 计算该core-pocket组中完全满足规则的分子占比
        compliance_ratio = sum(v) / len(v)
        avg_tmp.append(compliance_ratio)

    lipinski_full_ratio = sum(avg_tmp) / len(avg_tmp) * 100 if avg_tmp else 0
    return lipinski_full_ratio
    # print(f'Lipinski Full Compliance (5/5): {lipinski_full_ratio:.2f}%')

# -------------- High Affinity ---------------- #
def get_high_affinity(data, ref_data, group_id_list):
    high_affinity_count_dict = {gid: [] for gid in group_id_list}
    for each_data in data:
        ref_affinity = ref_data[each_data['group_id']]
        high_affinity_count_dict[each_data['group_id']].append(each_data['affinity'] <= ref_affinity)

    total_count = sum([len(v) for v in high_affinity_count_dict.values()])
    high_affinity_count = sum([sum(v) for v in high_affinity_count_dict.values()])
    high_affinity_ratio = high_affinity_count / total_count if total_count > 0 else 0
    return high_affinity_ratio

# -------------- MPBG ---------------- #
def get_mpbg(data, ref_data, group_id_list):
    mpbg_dict = {gid: [] for gid in group_id_list}
    for each_data in data:
        ref_affinity = ref_data[each_data['group_id']]
        each_mpbg = (ref_affinity - each_data['affinity']) / ref_affinity if ref_affinity != 0 else 0
        mpbg_dict[each_data['group_id']].append(each_mpbg)

    total_count = sum([len(v) for v in mpbg_dict.values()])
    mpbg_sum = sum([sum(v) for v in mpbg_dict.values()])
    mpbg_ratio = mpbg_sum / total_count if total_count > 0 else 0
    return mpbg_ratio

# -------------- Prepare Reference Molecular Docking Scores ---------------- #
def read_vina_dict_from_file(group_id_list, results_path):
    vina_metric_dict = {group_id_list[i]: [] for i in range(len(group_id_list))}
    results = torch.load(results_path, weights_only=False)
    for i in range(len(results)):
        if results[i]['vina'] and len(results[i]['vina']) != 0:
            key = int(results[i]['group'])
            vina_metric_dict[key].append(results[i]['vina'][0]['affinity'])
    return vina_metric_dict

def get_reference_vina_scores(group_id_list, results_ref_path = './utils/docking_result_ref.pt'):
    if os.path.exists(results_ref_path):
        ref_vina_metric_dict = read_vina_dict_from_file(group_id_list, results_ref_path)
        vina_metric_dict_high = {}
        for key, value in ref_vina_metric_dict.items():
            if len(value) == 0:
                continue
            vina_metric_dict_high[key] = min(value) #最高的affinity

        ref_high_affinities = vina_metric_dict_high.values()
        
        if ref_high_affinities:
            ref_high_affinity_value = sum(ref_high_affinities) / len(ref_high_affinities)
            print(f"{'Reference High Vina Score':<25} | {ref_high_affinity_value:.4f}")
        else:
            print('Could not calculate reference vina score. No valid docking results.')
        return vina_metric_dict_high


import json
if __name__ == "__main__":
    gen_smi_file = sys.argv[1]
    df = pd.read_csv(gen_smi_file)
    df['true_molecules'] = df['true_molecules'].fillna('').astype(str).apply(lambda x: [item.strip() for item in x.split(',') if item.strip()])
    data = df.to_dict(orient='records')

    with open('../SAR-DRG/dataset_split.json', 'r', encoding='utf-8') as f:
            data_split = json.load(f)
    group_id_list = data_split.get('test_group', [])
    group_id_list = [int(i) for i in group_id_list]

    # Calculate Metrics for all generated molecules
    valid_score = get_valid_score(data, group_id_list)
    unique_score = get_unique_score(data, group_id_list)
    similarity_score = get_smi_score(data, group_id_list)
    recovery_score = get_recovery_score(data, group_id_list)
    print("="*50)
    print("-" * 50)
    print(f"{'EVALUATION METRICS For All Generated Molecules':^50}") # 居中标题
    print("-" * 50)
    print(f"{'Validity':<25} | {valid_score:.2f}%")
    print(f"{'Uniqueness':<25} | {unique_score:.2f}%")
    print(f"{'Similarity':<25} | {similarity_score:.2f}%")
    print(f"{'Recovery':<25} | {recovery_score:.2f}%")
    print("-" * 50)

    valid_data = [obj for obj in data if obj['valid']]
    df = pd.DataFrame(valid_data)

    # ref_data = get_reference_vina_scores(group_id_list, results_ref_path = './utils/docking_result_ref_case.pt')
    ref_data = get_reference_vina_scores(group_id_list, results_ref_path = './utils/docking_result_ref.pt')
    for topk in [1, 3, 5, 10]:
        df['affinity'] = pd.to_numeric(df['affinity'], errors='coerce')
        df = df.dropna(subset=['affinity'])
        result_df = df.sort_values('affinity', ascending=True).groupby('group_id').head(topk)
        data = result_df.to_dict(orient='records')
        average_affinity = sum([d['affinity'] for d in data]) / len(data) if data else 0
        high_affinity = get_high_affinity(data, ref_data, group_id_list)
        mpbg = get_mpbg(data, ref_data, group_id_list)
        sas_score = get_sas_score(data, group_id_list)
        qed_score = get_qed_score(data, group_id_list)
        lipinski_avg = get_avg_lipinski_score(data, group_id_list)
        lipinski_rate = get_full_lipinski_rate(data, group_id_list)

        print(f"{f'EVALUATION METRICS For Top {topk} Generated Molecules':^50}") # 居中标题
        print("-" * 50)
        print(f"{'Average Affinity':<25} | {average_affinity:.2f}")
        print(f"{'High Affinity':<25} | {high_affinity * 100:.2f}%")
        print(f"{'MPBG':<25} | {mpbg * 100:.2f}%")
        print(f"{'SAS (Synthesizability)':<25} | {sas_score:.4f}")
        print(f"{'QED (Drug-likeness)':<25} | {qed_score:.4f}")
        print(f"{'Lipinski Rule (Avg)':<25} | {lipinski_avg:.4f}")
        print(f"{'Lipinski Pass Rate':<25} | {lipinski_rate:.2f}%")
        print("-" * 50)