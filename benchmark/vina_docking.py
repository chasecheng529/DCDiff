from rdkit.Chem.rdForceFieldHelpers import UFFOptimizeMolecule
import subprocess
from rdkit.Chem.rdMolAlign import CalcRMS
import numpy as np
from easydict import EasyDict
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
from rdkit import Geometry
from rdkit.Chem.QED import qed
from rdkit.Chem import rdMolDescriptors
from rdkit.six.moves import cPickle
from rdkit.six import iteritems
import math
import os.path as op
import os
import string
import random
from rdkit import Chem
from copy import deepcopy
import csv
import torch
from Bio.PDB import PDBParser
import argparse
import shutil
import warnings
warnings.filterwarnings("ignore")
from multiprocessing.dummy import Pool as ThreadPool
import time
import sys
from tqdm import tqdm
import pandas as pd

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # 禁用所有 RDKit 日志

_fscores = None

def get_logp(mol):
    return Crippen.MolLogP(mol)

def obey_lipinski(mol):
    mol = deepcopy(mol)
    Chem.SanitizeMol(mol)
    rule_1 = Descriptors.ExactMolWt(mol) < 500
    rule_2 = Lipinski.NumHDonors(mol) <= 5
    rule_3 = Lipinski.NumHAcceptors(mol) <= 10
    rule_4 = (logp:=Crippen.MolLogP(mol)>=-2) & (logp<=5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])

def readFragmentScores(name='fpscores'):
  import gzip
  global _fscores
  # generate the full path filename:
  if name == "fpscores":
    name = op.join(op.dirname(__file__), name)
  _fscores = cPickle.load(gzip.open('%s.pkl.gz' % name))
  outDict = {}
  for i in _fscores:
    for j in range(1, len(i)):
      outDict[i[j]] = float(i[0])
  _fscores = outDict

def numBridgeheadsAndSpiro(mol, ri=None):
  nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
  nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
  return nBridgehead, nSpiro

def calculateScore(m):
  if _fscores is None:
    readFragmentScores()

  fp = rdMolDescriptors.GetMorganFingerprint(m,
                                             2)  #<- 2 is the *radius* of the circular fingerprint
  fps = fp.GetNonzeroElements()
  score1 = 0.
  nf = 0
  for bitId, v in iteritems(fps):
    nf += v
    sfp = bitId
    score1 += _fscores.get(sfp, -4) * v
  score1 /= nf

  # features score
  nAtoms = m.GetNumAtoms()
  nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
  ri = m.GetRingInfo()
  nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
  nMacrocycles = 0
  for x in ri.AtomRings():
    if len(x) > 8:
      nMacrocycles += 1

  sizePenalty = nAtoms**1.005 - nAtoms
  stereoPenalty = math.log10(nChiralCenters + 1)
  spiroPenalty = math.log10(nSpiro + 1)
  bridgePenalty = math.log10(nBridgeheads + 1)
  macrocyclePenalty = 0.
  # ---------------------------------------
  # This differs from the paper, which defines:
  #  macrocyclePenalty = math.log10(nMacrocycles+1)
  # This form generates better results when 2 or more macrocycles are present
  if nMacrocycles > 0:
    macrocyclePenalty = math.log10(2)

  score2 = 0. - sizePenalty - stereoPenalty - spiroPenalty - bridgePenalty - macrocyclePenalty

  # correction for the fingerprint density
  # not in the original publication, added in version 1.1
  # to make highly symmetrical molecules easier to synthetise
  score3 = 0.
  if nAtoms > len(fps):
    score3 = math.log(float(nAtoms) / len(fps)) * .5

  sascore = score1 + score2 + score3

  # need to transform "raw" value into scale between 1 and 10
  min = -4.0
  max = 2.5
  sascore = 11. - (sascore - min + 1) / (max - min) * 9.
  # smooth the 10-end
  if sascore > 8.:
    sascore = 8. + math.log(sascore + 1. - 9.)
  if sascore > 10.:
    sascore = 10.0
  elif sascore < 1.:
    sascore = 1.0

  return sascore

def compute_sa_score(rdmol):
    rdmol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
    sa = calculateScore(rdmol)
    sa_norm = round((10-sa)/9,2)
    return sa_norm

def get_random_id(length=30):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length)) 

def parse_qvina_outputs(docked_sdf_path, ref_mol):

    suppl = Chem.SDMolSupplier(docked_sdf_path, sanitize=False)
    results = []
    for i, mol in enumerate(suppl):
        if mol is None:
            print(f"Parsing Qvina Out, Mol is None, skip. File:{docked_sdf_path}")
            continue
        line = mol.GetProp('REMARK').splitlines()[0].split()[2:]
        try:
            rmsd = CalcRMS(ref_mol, mol)
        except Exception as e:
            rmsd = np.nan
        results.append(EasyDict({
            'rdmol': mol,
            'mode_id': i,
            'affinity': float(line[0]),
            'rmsd_lb': float(line[1]),
            'rmsd_ub': float(line[2]),
            'rmsd_ref': rmsd
        }))

    return results

class BaseDockingTask(object):

    def __init__(self, pdb_block, ligand_rdmol):
        super().__init__()
        self.pdb_block = pdb_block
        self.ligand_rdmol = ligand_rdmol

    def run(self):
        raise NotImplementedError()
    
    def get_results(self):
        raise NotImplementedError()


class QVinaDockingTask(BaseDockingTask):

    @classmethod
    def from_data(cls, ligand_mol, protein_path):
        with open(protein_path, 'r') as f:
            pdb_block = f.read()

        struct = PDBParser().get_structure('', protein_path)

        return cls(pdb_block, ligand_mol, struct)

    def __init__(self, pdb_block, ligand_rdmol, struct, conda_env='MolGen', tmp_dir='./tmp', use_uff=True, center=None):
        super().__init__(pdb_block, ligand_rdmol)

        residue_ids = []
        atom_coords = []

        for residue in struct.get_residues():
            resid = residue.get_id()[1]
            for atom in residue.get_atoms():
                atom_coords.append(atom.get_coord())
                residue_ids.append(resid)

        residue_ids = np.array(residue_ids)
        atom_coords = np.array(atom_coords)
        center_pro = (atom_coords.max(0) + atom_coords.min(0)) / 2

        #给输入的分子加H原子并计算坐标
        params = AllChem.ETKDGv3()
        params.randomSeed = 1  # 固定随机种子
        ligand_rdmol = Chem.AddHs(ligand_rdmol, addCoords=True)
        AllChem.EmbedMolecule(ligand_rdmol, params)

        self.conda_env = conda_env
        self.tmp_dir = os.path.realpath(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        self.task_id = get_random_id()
        self.receptor_id = self.task_id + '_receptor'
        self.ligand_id = self.task_id + '_ligand'

        self.receptor_path = os.path.join(self.tmp_dir, self.receptor_id + '.pdb')
        self.ligand_path = os.path.join(self.tmp_dir, self.ligand_id + '.sdf')

        with open(self.receptor_path, 'w') as f:
            f.write(pdb_block)

        #UFF立场优化
        if use_uff:
            try:
                not_converge = 10
                while not_converge > 0:
                    flag = UFFOptimizeMolecule(ligand_rdmol)
                    not_converge = min(not_converge - 1, flag * 10)
            except RuntimeError:
                pass
        sdf_writer = Chem.SDWriter(self.ligand_path)
        sdf_writer.write(ligand_rdmol)
        sdf_writer.close()

        self.ligand_rdmol = ligand_rdmol
        self.noH_rdmol = Chem.RemoveHs(ligand_rdmol)
        self.center = center_pro
        self.proc = None
        self.results = None
        self.output = None
        self.docked_sdf_path = None
    
    def run(self, exhaustiveness=16):
        commands = """
eval "$(conda shell.bash hook)"
conda activate {env}
cd {tmp}
# Prepare receptor (PDB->PDBQT)
../utils/autodocktools-prepare/prepare_receptor4.py -r {receptor_id}.pdb -o {receptor_id}.pdbqt
# Prepare ligand
obabel {ligand_id}.sdf -O{ligand_id}.pdbqt
qvina \
    --receptor {receptor_id}.pdbqt \
    --ligand {ligand_id}.pdbqt \
    --center_x {center_x:.4f} \
    --center_y {center_y:.4f} \
    --center_z {center_z:.4f} \
    --size_x 40 --size_y 40 --size_z 40 \
    --exhaustiveness {exhaust} \
    --cpu 32 \
    --seed 1 
obabel {ligand_id}_out.pdbqt -O{ligand_id}_out.sdf -h
        """.format(
            receptor_id = self.receptor_id,
            ligand_id = self.ligand_id,
            env = self.conda_env, 
            tmp = self.tmp_dir, 
            exhaust = exhaustiveness,
            center_x = self.center[0],
            center_y = self.center[1],
            center_z = self.center[2],
        )

        self.docked_sdf_path = os.path.join(self.tmp_dir, '%s_out.sdf' % self.ligand_id)

        # Use shell=True for simplicity with conda commands, and capture output
        self.proc = subprocess.Popen(
            commands, 
            shell=True,
            executable='/bin/bash', # Explicitly use bash
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,  # 捕获错误输出 (关键!)
            encoding='utf-8',        # 自动解码为字符串
            errors='ignore'          # 防止编码报错
        )
    
    def wait(self):
        """Waits for the process to complete and returns stdout/stderr."""
        if self.proc:
            stdout, stderr = self.proc.communicate()
            if self.proc.returncode != 0:
                # Log errors for debugging without stopping the entire pool
                print(f"Error docking {self.ligand_id}. Return code: {self.proc.returncode}")
                print(f"Stderr: {stderr.decode('utf-8', errors='ignore')}")
            return stdout, stderr
        return None, None

    def run_sync(self):
        self.run()
        while self.get_results() is None:
            pass
        results = self.get_results()
        return results

    def get_results(self):
        if self.proc is None:  # 任务未开始
            return None
            
        # 如果结果已经解析过，直接返回
        if self.results is not None:
            return self.results
            
        # 检查进程是否还在运行
        if self.proc.poll() is None: # In progress
            return None

        # 进程已结束，开始解析结果文件
        if self.proc.returncode != 0:
            # print(f"Docking process for {self.ligand_id} failed with non-zero exit code.")
            # 在这种情况下，结果文件可能不存在或不完整
            return [] # 返回一个空列表表示失败但程序可继续

        try:
            # 核心逻辑：直接解析输出的SDF文件
            self.results = parse_qvina_outputs(self.docked_sdf_path, self.noH_rdmol)
        except FileNotFoundError:
            # print(f'[Error] Vina output file not found: {self.docked_sdf_path}')
            self.results = [] # 文件不存在，返回空列表
        except Exception as e:
            # print(f'[Error] Vina output parsing error for: {self.docked_sdf_path}')
            # print(f'*get_results_exception*: {e}')
            self.results = [] # 其他解析错误，返回空列表
                
        return self.results


import multiprocessing as mp

def read_vina_dict_from_file(gt_gen_csv_path, results_path):
    group_id_list = set()
    # Step 1: Load test CSV data
    with open(gt_gen_csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            group_id_list.add(row[0])  # group_id
    group_id_list = list(group_id_list)

    # Step 2: Initialize vina metric dictionary
    vina_metric_dict = {group_id_list[i]: [] for i in range(len(group_id_list))}
    # Step 3: Load results and populate the dictionary
    results = torch.load(results_path, weights_only=False)
    for i in range(len(results)):
        if results[i]['vina'] and len(results[i]['vina']) != 0:
            key = results[i]['group']
            vina_metric_dict.setdefault(key, []).append(results[i]['vina'][0]['affinity'])
            # vina_metric_dict.setdefault(key, []).append((results[i]['vina'][0]['affinity'], results[i]['mol']))
    return vina_metric_dict

def run_docking_for_molecule(args):
    """
    A worker function for the multiprocessing pool.
    It takes a tuple of arguments, runs a single docking task, and returns the result.
    """
    group_id, mol_index, pred_smi, protein_path = args
    
    # Handle invalid or empty SMILES
    if not pred_smi:
        # print(f"Group: {group_id}: Invalid or empty SMILES, skipping.")
        return {'mol': None, 'vina': None, 'group': group_id, 'index': mol_index}

    pred_mol = Chem.MolFromSmiles(pred_smi, sanitize=True)
    if pred_mol is None:
        # print(f"Group: {group_id} RDKit could not parse SMILES '{pred_smi}', skipping.")
        return {'mol': None, 'vina': None, 'group': group_id, 'index': mol_index}

    try:
        # Each worker creates its own task object
        vina_task = QVinaDockingTask.from_data(pred_mol, protein_path)
        vina_task.run()
        vina_task.wait() # Efficiently wait for the single subprocess to finish
        vina_results = vina_task.get_results()
        stdout, stderr = vina_task.wait()
        if not vina_results or len(vina_results) == 0:
            if stderr and len(stderr.strip()) > 0:
                # 截取最后 300 个字符的报错信息，避免太长
                print(f"Vina/Obabel Error: {stderr.strip()}")
        return {
            'mol': pred_mol,
            'vina': vina_results,
            'group': group_id,
            'index': mol_index
        }
    except Exception as e:
        print(f'Error processing molecule index {mol_index}: {e}')
        # Return a failed result so we don't lose the index
        return {
            'mol': pred_mol,
            'vina': None,
            'group': group_id,
            'index': mol_index
        }

#计算生成分子的vina对接指数,并返回结果
def cal_pred_mol_vina_docking(gt_gen_csv_path='', results_path=''):
    if os.path.exists(results_path):
        pred_vina_metric_dict = read_vina_dict_from_file(gt_gen_csv_path, results_path)
        return pred_vina_metric_dict

    # 1. Prepare a list of tasks (arguments for each molecule)
    tasks = []
    with open(gt_gen_csv_path, 'r') as f:
        csv_reader = csv.reader(f)
        next(csv_reader) # Skip header
        for col_index, row in enumerate(csv_reader):
            pred_smi = row[3]
            group_id = row[0]
            protein_path = row[4]
            tasks.append((group_id, col_index, pred_smi, protein_path)) #使用group_id作为标记

    # 2. Determine the number of parallel processes
    num_processes = mp.cpu_count() // 4  # Adjust as needed
    print(f"Starting docking generated mols with {num_processes} parallel processes...")

    results = []
    # 3. Create a pool and run the tasks in parallel
    with mp.Pool(processes=num_processes) as pool:
        # Use imap_unordered for efficiency and tqdm for a progress bar
        # imap_unordered returns results as they are completed
        for result in tqdm(pool.imap_unordered(run_docking_for_molecule, tasks), total=len(tasks), desc="Docking Molecules"):
            results.append(result)
            # Optional: Save results periodically
            if len(results) % 200 == 0:
                # tqdm.write(f"\nSaving intermediate results... ({len(results)} completed)")
                # Sort results by original index 'i' before saving
                sorted_results = sorted(results, key=lambda x: x['index'])
                torch.save(sorted_results, results_path)

    # Final sort and save
    print("All docking tasks completed. Performing final save.")
    print("Success {} of total {}".format(len(results), len(tasks)))
    final_sorted_results = sorted(results, key=lambda x: x['index'])
    torch.save(final_sorted_results, results_path)

    pred_vina_metric_dict = read_vina_dict_from_file(gt_gen_csv_path, results_path)
    return pred_vina_metric_dict

#计算测试集中生成的100组分子中（每组100个分子）的指标（均值与大于阈值的百分比）
def cal_pred_mol_average_metric(pred_mol_vina, topk=10):
    # Step 1: Calculate metrics
    avg_tmp = []  # Store average scores for each group
    total_scores = 0  # Total number of scores

    for v in pred_mol_vina.values():
        if len(v) == 0:
            continue
        v = sorted(v)[:topk]
        avg_tmp.append(sum(v) / len(v))  # Average score for this group
        total_scores += len(v)  # Update total score count

    # Overall metrics
    vina_score = sum(avg_tmp) / len(avg_tmp) if avg_tmp else 0  # Overall average vina score

    # Print results
    print(f"{'Vina Score':<25} | {vina_score:.4f}")

#计算基准的参考分子的对接分数
# from rdkit.Chem.QED import qed
# import utils.sascore as sascore 
def cal_gt_mol_vina(gt_gen_csv_path = '', results_ref_path = ''):
    if os.path.exists(results_ref_path):
        ref_vina_metric_dict = read_vina_dict_from_file(gt_gen_csv_path, results_ref_path)
        vina_metric_dict_high = {}
        # vina_high_mol_obj = []
        vina_metric_dict_low = {}
        for key, value in ref_vina_metric_dict.items():
            if len(value) == 0:
                continue
            # best_tuple = max(value, key=lambda x: x[0])
            # vina_metric_dict_high[key] = best_tuple[0]
            # vina_high_mol_obj.append(best_tuple[1])
            vina_metric_dict_high[key] = min(value) #最高的affinity
            vina_metric_dict_low[key] = max(value) #最低的affinity
            # vina_metric_dict_low[key] = 0

        ref_high_affinities = vina_metric_dict_high.values()
        ref_low_affinities = vina_metric_dict_low.values()
        
        if ref_high_affinities and ref_low_affinities:
            ref_high_affinity_value = sum(ref_high_affinities) / len(ref_high_affinities)
            ref_low_affinity_value = sum(ref_low_affinities) / len(ref_low_affinities)
            print(f"{'Reference High Vina Score':<25} | {ref_high_affinity_value:.4f}")
            print(f"{'Reference Low Vina Score':<25} | {ref_low_affinity_value:.4f}")
        else:
            print('Could not calculate reference vina score. No valid docking results.')
        return vina_metric_dict_high, vina_metric_dict_low
    
    print(f"Test results file not found. Starting docking for the test set...")
    # 1. 准备任务列表
    tasks = []
    group_id_list = []

    df = pd.read_csv(gt_gen_csv_path, dtype={'group_id': str})
    unique_groupid_refmol = df.drop_duplicates(subset=['group_id'], keep='first')
    for _, row in unique_groupid_refmol.iterrows():
        smi = row['true_molecules']
        ref_smi_list = smi.split(',')
        protein_path = row['protein_path']
        group_id = row['group_id']
        for each_ref_smi in ref_smi_list:
            tasks.append((group_id, 0, each_ref_smi, protein_path)) #按照每个参考分子创建task进行对接,使用group_id来标记
        group_id_list.append(row['group_id'])

    # 2. 设置并行进程数并执行
    num_processes = mp.cpu_count() // 2 # 同样，可以根据需要调整
    print(f"Starting test set docking with {num_processes} parallel processes...")
    
    results = []
    with mp.Pool(processes=num_processes) as pool:
        for result in tqdm(pool.imap_unordered(run_docking_for_molecule, tasks), total=len(tasks), desc="Docking Test Set"):
            results.append(result)

    # 3. 保存结果并计算指标
    final_sorted_results = sorted(results, key=lambda x: x['index'])
    torch.save(final_sorted_results, results_ref_path)
    print(f"Test set docking completed. Results saved to {results_ref_path}.")

    ref_vina_metric_dict = read_vina_dict_from_file(gt_gen_csv_path, results_ref_path)
    vina_metric_dict_high = {}
    vina_metric_dict_low = {}
    
    for key, value in ref_vina_metric_dict.items():
        if len(value) == 0:
            continue
        vina_metric_dict_high[key] = min(value) #最高的affinity
        vina_metric_dict_low[key] = max(value) #最低的affinity

    ref_high_affinities = vina_metric_dict_high.values()
    ref_low_affinities = vina_metric_dict_low.values()
    
    if ref_high_affinities and ref_low_affinities:
        ref_high_affinity_value = sum(ref_high_affinities) / len(ref_high_affinities)
        ref_low_affinity_value = sum(ref_low_affinities) / len(ref_low_affinities)
        print(f"{'Reference High Vina Score':<25} | {ref_high_affinity_value:.4f}")
        print(f"{'Reference Low Vina Score':<25} | {ref_low_affinity_value:.4f}")
    else:
        print('Could not calculate reference vina score. No valid docking results.')
    print('='*30)

    return vina_metric_dict_high, vina_metric_dict_low

def cal_high_aff(pred_mol_vina, high_ref_vina, base_ref_vina, topk):
    high_aff_count_to_high = {}
    high_aff_count_to_base = {}
    # 1. 遍历预测结果字典
    for key, pred_scores in pred_mol_vina.items():
        # 安全检查：确保该 key 在两个参考字典中都存在
        if key not in high_ref_vina or key not in base_ref_vina:
            continue
        
        # 2. 获取参考值
        high_ref_val = high_ref_vina[key]
        base_ref_val = base_ref_vina[key]

        current_pred_scores = sorted(pred_scores)[:topk]  # 只考虑前 topk 个分数        
        total_num = len(current_pred_scores)
        if total_num == 0:
            continue

        # 3. 统计比例
        # Vina score 越小越好，所以统计 pred < ref 的数量
        
        # 统计优于 High Ref 的比例
        better_than_high_count = sum(1 for s in current_pred_scores if s <= high_ref_val)
        high_aff_count_to_high[key] = better_than_high_count / total_num
        
        # 统计优于 Base Ref 的比例
        better_than_base_count = sum(1 for s in current_pred_scores if s <= base_ref_val)
        high_aff_count_to_base[key] = better_than_base_count / total_num

    # 4. 计算并打印均值
    def get_mean_from_dict(d):
        return sum(d.values()) / len(d) if len(d) > 0 else 0.0

    mean_to_high = get_mean_from_dict(high_aff_count_to_high)
    mean_to_base = get_mean_from_dict(high_aff_count_to_base)

    print(f"{'HA High/Base':<25} | {mean_to_high*100:.2f}% / {mean_to_base*100:.2f}%")

def cal_mpbg(pred_mol_vina, high_ref_vina, base_ref_vina, topk):
    mpbg_to_high = {}
    mbpg_to_ref = {}

    # 1. 遍历预测字典
    for key, pred_scores in pred_mol_vina.items():
        # 确保 Key 在两个参考字典中都存在
        if key not in high_ref_vina or key not in base_ref_vina:
            continue
            
        # 2. 获取参考分数
        high_ref_val = high_ref_vina[key]
        base_ref_val = base_ref_vina[key]

        current_pred_scores = sorted(pred_scores)[:topk]  # 只考虑前 topk 个分数
        total_num = len(current_pred_scores)
        if total_num == 0:
            continue

        # 3. 计算 High Ref 的 MPBG
        # 防止分母为0 (虽然Vina分一般不为0)
        if high_ref_val != 0:
            mpbg_list_high = [
                (high_ref_val - score) / high_ref_val * 100 
                for score in current_pred_scores
            ]
            mpbg_to_high[key] = sum(mpbg_list_high) / total_num
        else:
            mpbg_to_high[key] = 0.0

        # 4. 计算 Base Ref 的 MPBG
        if base_ref_val != 0:
            mpbg_list_base = [
                (base_ref_val - score) / base_ref_val * 100 
                for score in current_pred_scores
            ]
            mbpg_to_ref[key] = sum(mpbg_list_base) / total_num
        else:
            mbpg_to_ref[key] = 0.0

    # 5. 计算所有样本的宏平均 (Macro Average)
    def get_mean(d):
        return sum(d.values()) / len(d) if len(d) > 0 else 0.0

    print(mpbg_to_high)

    mean_mpbg_high = get_mean(mpbg_to_high)
    mean_mpbg_base = get_mean(mbpg_to_ref)

    # 6. 打印结果
    print(f"{'MPBG High/Base':<25} | {mean_mpbg_high:.2f}% / {mean_mpbg_base:.2f}%")

def compute_vina_metrics(gt_gen_csv_path, results_pred_path, results_ref_path):
    # Compute reference vina scores
    print("="*50)
    print("-" * 50)
    print(f"{'Reference Molecule Vina Scores':^50}") # 居中标题
    print("-" * 50)
    vina_metric_dict_high, vina_metric_dict_low = cal_gt_mol_vina(gt_gen_csv_path = gt_gen_csv_path, results_ref_path = results_ref_path)
    print("-" * 50)
    
    # Compute predicted vina scores
    pred_mol_vina = cal_pred_mol_vina_docking(gt_gen_csv_path = gt_gen_csv_path, results_path = results_pred_path)
    # for topk in [1,3,5,10]:
    #     print("-" * 50)
    #     print(f"{f'Metrics For Top{topk} Generated Molecules':^50}") # 居中标题
    #     print("-" * 50)
    #     cal_pred_mol_average_metric(pred_mol_vina, topk=topk)
    #     cal_high_aff(pred_mol_vina=pred_mol_vina, high_ref_vina=vina_metric_dict_high, base_ref_vina=vina_metric_dict_low, topk = topk)
    #     cal_mpbg(pred_mol_vina=pred_mol_vina, high_ref_vina=vina_metric_dict_high, base_ref_vina=vina_metric_dict_low, topk = topk)
    #     print("-" * 50)

def add_vina_score_to_csv(results_pred_path, csv_path):
    # Load docking results
    results = torch.load(results_pred_path, weights_only=False)
    df = pd.read_csv(csv_path)
    index_to_affinity = {}

    for res in results:
        index = res['index']
        vina_score = res['vina'][0]['affinity'] if res['vina'] and len(res['vina']) > 0 else None
        index_to_affinity[int(index)] = vina_score
    
    df['affinity'] = df.index.map(index_to_affinity)
    df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--formatted', action='store', type=str, required=True)
    parser.add_argument('--results_pred_path', action='store', type=str, required=True)
    parser.add_argument('--results_ref_path', action='store', type=str, required=True)
    args = parser.parse_args()

    gt_gen_csv_path = args.formatted
    results_pred_path = args.results_pred_path
    results_ref_path = args.results_ref_path

    compute_vina_metrics(gt_gen_csv_path, results_pred_path, results_ref_path)

    add_vina_score_to_csv(results_pred_path, gt_gen_csv_path)
    
    tmp_folder = os.path.join('tmp')
    try:
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
            print('Deleted tmp files')
    except:
        print('Please delete by hand')
