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
RDLogger.DisableLog('rdApp.*')# 屏蔽 RDKit 日志

def load_rdkit_molecule(xyz_path, obabel_path, scaf_sdf_path, true_sdf_path, true_scaf_smi_ori, true_mol_smi_ori):
    # 检查文件是否存在
    if not os.path.exists(obabel_path):
        print(f"文件不存在，跳过: {obabel_path}")
        return None, None, None, None, None

    try:
        # 加载分子
        supp = Chem.SDMolSupplier(obabel_path, sanitize=False)
        mol = list(supp)[0] if len(supp) > 0 else None
        if mol is None:
            print(f"加载分子失败，跳过: {obabel_path}")
            return None, None, None, None, None
    except Exception as e:
        print(f"加载 {obabel_path} 时出错: {e}")
        return None, None, None, None, None

    # 对分子进行处理
    mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    mol_filtered = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
    try:
        mol_smi = Chem.MolToSmiles(mol_filtered, canonical=True)
    except RuntimeError:
        mol_smi = Chem.MolToSmiles(mol_filtered, canonical=False)

    # 加载 scaffold 文件
    supp = Chem.SDMolSupplier(scaf_sdf_path, sanitize=False)
    true_scaf = list(supp)[0]
    true_scaf_smi = Chem.MolToSmiles(true_scaf)

    # 加载 true 分子文件
    supp = Chem.SDMolSupplier(true_sdf_path, sanitize=False)
    true_mol = list(supp)[0]
    true_mol_smi = Chem.MolToSmiles(true_mol)
    
    # 检查 scaffold 是否匹配
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

        
        if mol is None:  # 如果文件加载失败，跳过当前样本
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

    # 找到最大样本编号
    for fname in os.listdir(folder):
        if fname.isdigit():
            max_num = max(max_num, int(fname))

    for i in range(max_num + 1):
        try:
            true_mol_smi = idx2true_mol_smi[str(i)]
            true_scaf_smi = idx2true_scaf_smi[str(i)]
            protein_filename = idx2true_protein_filename[str(i)]
        except KeyError:
            print(f"缺少样本数据，跳过样本编号 {i}")
            continue

        mols, mols_smi, rgroup_smi, true_scaf_smi_, true_mol_smi_ = load_molecules(f'{folder}/{str(i)}', true_scaf_smi, true_mol_smi)
        
        if not mols:  # 如果加载失败，跳过
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
    # 读取 true_smiles 表
    true_smiles_table = pd.read_csv(true_smiles_path, names=['uuid','molecule_name','molecule','scaffold','rgroups','anchor','pocket_full_size','pocket_bb_size','molecule_size','scaffold_size','rgroup_size', 'protein_filename','affinity'])
    #true_smiles_table = pd.read_csv(true_smiles_path, names=['uuid','molecule_name','molecule','scaffold','rgroups','anchor','pocket_full_size','pocket_bb_size','molecule_size','scaffold_size','rgroup_size', 'protein_filename'])
    # 建立 UUID 到 SMILES 和文件名的映射
    idx2true_mol_smi = dict(zip(true_smiles_table.uuid.values, true_smiles_table.molecule.values))
    idx2true_scaf_smi = dict(zip(true_smiles_table.uuid.values, true_smiles_table.scaffold.values))
    idx2true_protein_filename = dict(zip(true_smiles_table.uuid.values, true_smiles_table.protein_filename.values))

    # 加载预测数据
    pred_mols, pred_mols_smi, pred_rgroup_smi, true_mols_smi, true_scafs_smi, true_mols_smi_ori, true_scafs_smi_ori, protein_filename_list = load_sampled_dataset(
        folder=samples,
        idx2true_mol_smi=idx2true_mol_smi,
        idx2true_scaf_smi=idx2true_scaf_smi,
        idx2true_protein_filename=idx2true_protein_filename,
    )

    # 创建目标目录，如果不存在
    formatted_output_dir = formatted
    if not os.path.exists(formatted_output_dir):
        os.makedirs(formatted_output_dir)

    # 输出文件路径
    metric_out_smi_path = os.path.join(formatted_output_dir, 'bingdingnet_test_metric.smi')
    vina_out_smi_path = os.path.join(formatted_output_dir, 'bingdingnet_test_vina.smi')
    out_sdf_path = os.path.join(formatted_output_dir, 'bingdingnet_test_out.sdf')

    # 写入 metric_out_smi
    with open(metric_out_smi_path, 'w') as f:
        for i in range(len(pred_mols)):
            f.write(f'{true_scafs_smi[i]} {true_mols_smi[i]} {pred_mols_smi[i]} {pred_rgroup_smi[i]} {protein_filename_list[i]}\n')

    # 写入 vina_out_smi
    with open(vina_out_smi_path, 'w') as f:
        for i in range(len(pred_mols)):
            f.write(f'{true_scafs_smi_ori[i]} {true_mols_smi_ori[i]} {pred_mols_smi[i]} {pred_rgroup_smi[i]} {protein_filename_list[i]}\n')

    # 写入 .sdf 文件
    with Chem.SDWriter(out_sdf_path) as writer:
        for mol in pred_mols:
            if mol is not None:
                writer.write(mol)

def create_pred_gt_csv(samples_path, split_file_path = None, gt_csv_path=None, out_pred_gt_csv_path=None):
    # need 'true_molecules','group_id','pred_molecule','core'
    # 判断文件是否已经存在
    if os.path.exists(out_pred_gt_csv_path):
        print("文件已经保存")
        return
    # Get sample nums
    sample_num = 0
    for fname in os.listdir(samples_path):
        if fname.isdigit():
            sample_num = max(sample_num, int(fname))
    
    # 获得group_id的list作为映射sample的folder
    with open(split_file_path, 'r', encoding='utf-8') as f:
            data_split = json.load(f)
    group_id_list = data_split.get('test_group', [])


    all_data_rows = []
    #读取每一个group
    for i in range(sample_num + 1):
        group_id = group_id_list[i]
        group_sample_folder = f'{samples_path}/{str((i))}'
        group_pred_smi_list, group_core_smi, ref_smi = load_molecules(group_sample_folder) #将采样文件夹中的sdf文件转换为SMILES
        gt_smi_list, target_path = load_gt_smi_from_csv(gt_csv_path, group_id)
        target_path = os.path.join('../SAR-DRG', target_path)

        gt_smi_str = ",".join(gt_smi_list)

        for sample_id, pred_smi in enumerate(group_pred_smi_list):
            # 构建一个字典，代表 CSV 中的一行
            row = {
                'group_id': group_id,
                'core': group_core_smi,
                'true_molecules': gt_smi_str, # 这里放合并后的字符串
                'pred_molecule': pred_smi,     # 这里放单个预测分子
                'protein_path' : target_path, # 这里放对应的蛋白质口袋
                'ref_molecules': ref_smi, # 由于对接都是从SMILES转为3D，所以ref SMILES可能并不需要
                'sample_id': sample_id
            }
            all_data_rows.append(row)

    df = pd.DataFrame(all_data_rows)
    # 保存 CSV，index=False 表示不保存行号
    df.to_csv(out_pred_gt_csv_path, index=False)
    print(f"保存完成，共 {len(df)} 行数据。")

def load_molecules(folder):
    pred_smi_list = []
    sample_num = 100

    #加载group的core
    core_sdf_path = f'{folder}/core_.sdf'
    core_smi = load_mol_from_sdf(core_sdf_path)
    if core_smi is None:
        print("CORE SMI None")
        return None, None, None
    
    #加载group的high ref mol
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
        ref_smi = ref_smi[:-1]  # 去掉最后一个逗号

    #加载所有的预测的分子的SMILES
    for i in range(sample_num):
        pred_sdf_path = f'{folder}/{str(i)}_.sdf'
        # pred_sdf_path = f'{folder}/{str(i)}.sdf'
        pred_smi = load_mol_from_sdf(pred_sdf_path)
        if pred_smi is None:  # 如果文件加载失败，跳过当前样本
            continue
        pred_smi_list.append(pred_smi)

    return pred_smi_list, core_smi, ref_smi

def load_mol_from_sdf(sdf_file_path):
    if not os.path.exists(sdf_file_path):
        print(f"文件不存在，跳过: {sdf_file_path}")
        return None
    try:
        # 加载分子
        supp = Chem.SDMolSupplier(sdf_file_path, sanitize=False)
        if not supp:
            print(f"加载分子失败，跳过: {sdf_file_path}")
            return None
        mol = supp[0]
        if mol is None:
            print(f"加载分子失败，跳过: {sdf_file_path}")
            return None
    except Exception as e:
        print(f"加载 {sdf_file_path} 时出错: {e}")
        return None

    # 对分子进行处理
    try:
        mol_smi = Chem.MolToSmiles(mol, canonical=True)
    except RuntimeError:
        mol_smi = Chem.MolToSmiles(mol, canonical=False)
    return mol_smi

def load_gt_smi_from_csv(gt_csv_path, group_id):
    df = pd.read_csv(gt_csv_path)
    # 2. 转换数据类型，确保 target_group_id 和 csv 里的格式一致 (防止 int vs str 匹配失败)
    # 我们统一把 group_id 列转成字符串来比较，这样最稳
    df['group_id'] = df['group_id'].astype(str)
    group_id = str(group_id)
    
    # 3. 筛选 + 提取
    # 逻辑：找出 id_col 等于 target_group_id 的所有行，取 smi_col 列，转为 list
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
    