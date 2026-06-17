import os
import numpy as np
import pandas as pd
import pickle
import torch
import glob
import json

from rdkit import Chem
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from src import const

def read_sdf(sdf_path):
    if os.path.getsize(sdf_path) == 0:
        return Chem.Mol() # 返回空 Mol 对象，因为此时不需要生成任何东西
    supplier = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=True)
    if not supplier:
        return None
    mol = supplier[0]
    return mol

def get_one_hot(atom, atoms_dict):
    one_hot = np.zeros(len(atoms_dict))
    one_hot[atoms_dict[atom]] = 1
    return one_hot

def parse_core(mol):
    one_hot = []
    charges = []
    atom2idx = const.ATOM2IDX
    charges_dict = const.CHARGES
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
        charges.append(charges_dict[atom.GetSymbol()])
        if atom.GetAtomMapNum() == 1: #Atom Map Number=1 means anchor
            anchor_index = atom.GetIdx()
    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot), np.array(charges), anchor_index

def parse_molecule(mol):
    one_hot = []
    charges = []
    atom2idx = const.ATOM2IDX
    charges_dict = const.CHARGES
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
        charges.append(charges_dict[atom.GetSymbol()])
    positions = mol.GetConformer().GetPositions()
    return positions, np.array(one_hot), np.array(charges)

# for multi given size
def parse_rgroup_multian(mol):
    mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    one_hot_list = []
    charges_list = []
    positions_list = []
    rgroup_atom_num_list = []
    atom2idx = const.ATOM2IDX
    charges_dict = const.CHARGES
    for i in range(len(mol_frags)):
        one_hot = []
        charges = []
        for atom in mol_frags[i].GetAtoms():
            one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
            charges.append(charges_dict[atom.GetSymbol()])
        positions = mol_frags[i].GetConformer().GetPositions()
        positions = positions.tolist()
        one_hot_list.append(one_hot)
        charges_list.append(charges)
        positions_list.append(positions)
        rgroup_atom_num_list.append(mol_frags[i].GetNumAtoms())
        # rgroup_atom_num_list.append(10)
    one_hot = sum(one_hot_list, [])
    charges = sum(charges_list, [])
    positions = sum(positions_list, [])
    return np.array(positions), np.array(one_hot), np.array(charges), rgroup_atom_num_list

# for multi w/o anchors, with fake atom
def parse_rgroup_com_scaf(mol, fake_pos):
    mol_frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    one_hot_list = []
    charges_list = []
    positions_list = []
    atom2idx = const.ATOM2IDX
    charges_dict = const.CHARGES
    for i in range(len(mol_frags)):
        one_hot = []
        charges = []
        for atom in mol_frags[i].GetAtoms():
            one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
            charges.append(charges_dict[atom.GetSymbol()])
        positions = mol_frags[i].GetConformer().GetPositions()
        one_hot.extend(get_one_hot('#', atom2idx) for _ in range(10 - mol_frags[i].GetNumAtoms()))
        charges.extend(charges_dict['#'] for _ in range(10 - mol_frags[i].GetNumAtoms()))
        positions = positions.tolist()
        positions.extend(fake_pos for _ in range(10 - mol_frags[i].GetNumAtoms()))
        one_hot_list.append(one_hot)
        charges_list.append(charges)
        positions_list.append(positions)
    one_hot = sum(one_hot_list, [])
    charges = sum(charges_list, [])
    positions = sum(positions_list, [])
    return np.array(positions), np.array(one_hot), np.array(charges)

def parse_rgroup(mol, fake_pos, rchain_size = 10):
    fake_pos = list(fake_pos)
    one_hot = []
    charges = []
    atom2idx = const.ATOM2IDX
    charges_dict = const.CHARGES
    if mol is None or mol.GetNumAtoms() == 0 or mol.GetNumConformers() == 0: #When rchian is only one H atom, we do not need gen any atom
        positions = []
        one_hot.extend(get_one_hot('#', atom2idx) for _ in range(rchain_size))
        charges.extend(charges_dict['#'] for _ in range(rchain_size))
        positions.extend(fake_pos for _ in range(rchain_size))
        return np.array(positions), np.array(one_hot), np.array(charges)
    
    for atom in mol.GetAtoms():
        one_hot.append(get_one_hot(atom.GetSymbol(), atom2idx))
        charges.append(charges_dict[atom.GetSymbol()])
    positions = mol.GetConformer().GetPositions()
    one_hot.extend(get_one_hot('#', atom2idx) for _ in range(rchain_size - mol.GetNumAtoms()))
    charges.extend(charges_dict['#'] for _ in range(rchain_size - mol.GetNumAtoms()))
    positions = positions.tolist()
    positions.extend(fake_pos for _ in range(rchain_size - mol.GetNumAtoms()))
    return np.array(positions), np.array(one_hot), np.array(charges)

def collate(batch):
    out = {}

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')

    atom_mask = (out['core_pocket_mask'].bool() | out['rgroup_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.size()

    if 'pocket_mask' in batch[0].keys():
        batch_mask = torch.cat([
            torch.ones(n_nodes, dtype=const.TORCH_INT) * i
            for i in range(batch_size)
        ]).to(atom_mask.device)
        out['edge_mask'] = batch_mask
    else:
        edge_mask = atom_mask[:, None, :] * atom_mask[:, :, None]
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=atom_mask.device).unsqueeze(0)
        edge_mask *= diag_mask
        out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out

def collate_with_scaffold_edges(batch):
    out = {}

    for i, data in enumerate(batch):
        for key, value in data.items():
            out.setdefault(key, []).append(value)

    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)
            continue
        raise Exception(f'Unknown batch key: {key}')

    scaf_mask = out['core_pocket_mask']
    # scaf_mask = out['scaffold_only_mask']
    edge_mask = scaf_mask[:, None, :] * scaf_mask[:, :, None]
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=const.TORCH_INT, device=scaf_mask.device).unsqueeze(0)
    edge_mask *= diag_mask

    batch_size, n_nodes = scaf_mask.size()
    out['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    # Building edges and covalent bond values
    rows, cols, bonds = [], [], []
    for batch_idx in range(batch_size):
        for i in range(n_nodes):
            for j in range(n_nodes):
                rows.append(i + batch_idx * n_nodes)
                cols.append(j + batch_idx * n_nodes)

    edges = [torch.LongTensor(rows).to(scaf_mask.device), torch.LongTensor(cols).to(scaf_mask.device)]
    out['edges'] = edges

    atom_mask = (out['core_pocket_mask'].bool() | out['rgroup_mask'].bool()).to(const.TORCH_INT)
    # atom_mask = (out['scaffold_only_mask'].bool() | out['rgroup_mask'].bool()).to(const.TORCH_INT)
    out['atom_mask'] = atom_mask[:, :, None]

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out

def get_dataloader(dataset, batch_size, collate_fn=collate, persistent_workers = True, num_workers = 8, shuffle=False):
    return DataLoader(dataset, batch_size, collate_fn=collate_fn, persistent_workers = persistent_workers, num_workers = num_workers, shuffle=shuffle)

def create_template(tensor, scaffold_size, rgroup_size, fill=0):
    values_to_keep = tensor[:scaffold_size]
    values_to_add = torch.ones(rgroup_size, tensor.shape[1], dtype=values_to_keep.dtype, device=values_to_keep.device)
    values_to_add = values_to_add * fill
    return torch.cat([values_to_keep, values_to_add], dim=0)

def create_templates_for_rgroup_generation_single(data, rgroup_sizes):#解释
    decoupled_data = []
    for i, rgroup_size in enumerate(rgroup_sizes):
        data_dict = {}
        core_pocket_mask = data['core_pocket_mask'][i].squeeze()
        scaffold_size = core_pocket_mask.sum().int()
        for k, v in data.items():
            if k == 'num_atoms':
                data_dict[k] = scaffold_size + rgroup_size
                continue
            if k in const.DATA_LIST_ATTRS:
                data_dict[k] = v[i]
                continue
            if k in const.DATA_ATTRS_TO_PAD:
                fill_value = 1 if k == 'rgroup_mask' else 0
                template = create_template(v[i], scaffold_size, rgroup_size, fill=fill_value)
                if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                    template = template.squeeze(-1)
                data_dict[k] = template

        decoupled_data.append(data_dict)

    return collate(decoupled_data) # for single

def create_templates_for_rgroup_generation_multi(data, rgroup_sizes):
    decoupled_data = []
    for i, rgroup_size in enumerate(rgroup_sizes):
        data_dict = {}
        core_pocket_mask = data['core_pocket_mask'][i].squeeze()
        scaffold_size = core_pocket_mask.sum().int()
        for k, v in data.items():
            if k == 'num_atoms':
                data_dict[k] = scaffold_size + rgroup_size 
                continue
            if k in const.DATA_LIST_ATTRS:
                data_dict[k] = v[i]
                continue
            if k in const.DATA_ATTRS_TO_PAD:
                if k == "affinity":
                    data_dict[k] = v[i]
                    continue
                # 检查 v[i] 的维度
                if len(v[i].shape) == 1:
                    print(f"Warning: Key '{k}' has a 1D array for molecule {i}: {v[i]}")
                fill_value = 1 if k == 'rgroup_mask' else 0
                try:
                    template = create_template(v[i], scaffold_size, rgroup_size, fill=fill_value)
                    if k in const.DATA_ATTRS_TO_ADD_LAST_DIM:
                        template = template.squeeze(-1)
                    data_dict[k] = template
                except IndexError as e:
                    print(f"Error processing key '{k}' with value '{v[i]}' for molecule {i}")
                    raise e

        decoupled_data.append(data_dict)

    return collate_mr(decoupled_data) # for multi

# single
# class BingdingNetDataset(Dataset):
#     def __init__(self, data_path, prefix, device):
#         if '.' in prefix:
#             prefix, pocket_mode = prefix.split('.')
#         else:
#             parts = prefix.split('_')
#             prefix = '_'.join(parts[:-1])
#             pocket_mode = parts[-1]

#         dataset_path = os.path.join(data_path, f'{prefix}_{pocket_mode}.pt')
#         if os.path.exists(dataset_path):
#             self.data = torch.load(dataset_path, map_location=device)
#         else:
#             print(f'Preprocessing dataset with prefix {prefix}')
#             self.data = BingdingNetDataset.preprocess(data_path, prefix, pocket_mode, device)
#             torch.save(self.data, dataset_path)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         return self.data[item]

#     @staticmethod
#     def preprocess(data_path, prefix, pocket_mode, device):#pocket_mode是什么
#         data = []
#         table_path = os.path.join(data_path, f'{prefix}_table.csv')
#         scaffold_path = os.path.join(data_path, f'{prefix}_scaf.sdf')
#         rgroups_path = os.path.join(data_path, f'{prefix}_rgroup.sdf')
#         pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')

#         with open(pockets_path, 'rb') as f:
#             pockets = pickle.load(f)

#         table = pd.read_csv(table_path)
#         generator = tqdm(
#             zip(table.iterrows(), read_sdf(scaffold_path), read_sdf(rgroups_path), pockets),
#             total=len(table)
#         )
#         for (_, row), scaffold, rgroup, pocket_data in generator:
#             if type(scaffold) != Chem.rdchem.Mol or type(rgroup) != Chem.rdchem.Mol:
#                 continue
#             uuid = row['uuid']
#             # cat = row['cat']
#             name = row['molecule']
#             anchor_id = row['anchor']
#             protein_filename = row['protein_filename']
#             scaf_pos, scaf_one_hot, scaf_charges = parse_molecule(scaffold)
            
#             # fake_pos = np.mean(scaf_pos, axis = 0) # fake atom of scaf
#             fake_pos = scaf_pos[anchor_id] # fake atom of anchor
            
#             rgroup_pos, rgroup_one_hot, rgroup_charges = parse_rgroup(rgroup, fake_pos)#假原子
#             # rgroup_pos, rgroup_one_hot, rgroup_charges = parse_molecule(rgroup)

#             pocket_pos = []#靶点的
#             pocket_one_hot = []
#             pocket_charges = []
#             for i in range(len(pocket_data[f'{pocket_mode}_types'])):
#                 atom_type = pocket_data[f'{pocket_mode}_types'][i]
#                 pos = pocket_data[f'{pocket_mode}_coord'][i]
#                 if atom_type == 'H':
#                     continue
#                 pocket_pos.append(pos)
#                 pocket_one_hot.append(get_one_hot(atom_type, const.ATOM2IDX))#类型转换成数组
#                 pocket_charges.append(const.CHARGES[atom_type])
#             pocket_one_hot = np.array(pocket_one_hot)
#             pocket_charges = np.array(pocket_charges)
#             pocket_pos = np.array(pocket_pos)

#             positions = np.concatenate([scaf_pos, pocket_pos, rgroup_pos], axis=0)
#             one_hot = np.concatenate([scaf_one_hot, pocket_one_hot, rgroup_one_hot], axis=0)
#             charges = np.concatenate([scaf_charges, pocket_charges, rgroup_charges], axis=0)
#             anchors = np.zeros_like(charges)

#             anchors[row['anchor']] = 1

#             scaf_only_mask = np.concatenate([
#                 np.ones_like(scaf_charges),
#                 np.zeros_like(pocket_charges),
#                 np.zeros_like(rgroup_charges)
#             ])
#             pocket_mask = np.concatenate([
#                 np.zeros_like(scaf_charges),
#                 np.ones_like(pocket_charges),
#                 np.zeros_like(rgroup_charges)
#             ])
#             rgroup_mask = np.concatenate([
#                 np.zeros_like(scaf_charges),
#                 np.zeros_like(pocket_charges),
#                 np.ones_like(rgroup_charges)
#             ])
#             scaf_mask = np.concatenate([
#                 np.ones_like(scaf_charges),
#                 np.ones_like(pocket_charges),
#                 np.zeros_like(rgroup_charges)
#             ])

#             data.append({
#                 'uuid': uuid,
#                 'name': name,
#                 'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
#                 'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
#                 'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
#                 'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
#                 'scaffold_only_mask': torch.tensor(scaf_only_mask, dtype=const.TORCH_FLOAT, device=device),
#                 'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device),
#                 'core_pocket_mask': torch.tensor(scaf_mask, dtype=const.TORCH_FLOAT, device=device),
#                 'rgroup_mask': torch.tensor(rgroup_mask, dtype=const.TORCH_FLOAT, device=device),
#                 'num_atoms': len(positions),
#                 # 'cat': cat,
#             })

#         return data

#     @staticmethod
#     def create_edges(positions, scaffold_mask_only, rgroup_mask_only):
#         ligand_mask = scaffold_mask_only.astype(bool) | rgroup_mask_only.astype(bool)
#         ligand_adj = ligand_mask[:, None] & ligand_mask[None, :]
#         proximity_adj = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) <= 6
#         full_adj = ligand_adj | proximity_adj
#         full_adj &= ~np.eye(len(positions)).astype(bool)

#         curr_rows, curr_cols = np.where(full_adj)
#         return [curr_rows, curr_cols]

# multi w/o anchor
# class MultiRDataset(Dataset):
#     def __init__(self, data_path, prefix, device):
#         if '.' in prefix:
#             prefix, pocket_mode = prefix.split('.')
#         else:
#             parts = prefix.split('_')
#             prefix = '_'.join(parts[:-1])
#             pocket_mode = parts[-1]

#         dataset_path = os.path.join(data_path, f'{prefix}_{pocket_mode}.pt')
#         if os.path.exists(dataset_path):
#             self.data = torch.load(dataset_path, map_location=device)
#         else:
#             print(f'Preprocessing dataset with prefix {prefix}')
#             self.data = MultiRDataset.preprocess(data_path, prefix, pocket_mode, device)
#             torch.save(self.data, dataset_path)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, item):
#         return self.data[item]

#     @staticmethod
#     def preprocess(data_path, prefix, pocket_mode, device):
#         data = []
#         table_path = os.path.join(data_path, f'{prefix}_table.csv')
#         scaffold_path = os.path.join(data_path, f'{prefix}_scaf.sdf')
#         rgroups_path = os.path.join(data_path, f'{prefix}_rgroup.sdf')
#         pockets_path = os.path.join(data_path, f'{prefix}_pockets.pkl')

#         with open(pockets_path, 'rb') as f:
#             pockets = pickle.load(f)

#         table = pd.read_csv(table_path)
#         generator = tqdm(
#             zip(table.iterrows(), read_sdf(scaffold_path), read_sdf(rgroups_path), pockets),
#             total=len(table)
#         )
#         for (_, row), scaffold, rgroup, pocket_data in generator:
#             if type(scaffold) != Chem.rdchem.Mol or type(rgroup) != Chem.rdchem.Mol:
#                 continue
#             uuid = row['uuid']
#             name = row['molecule']
#             anchor_id = row['anchor']
#             scaf_pos, scaf_one_hot, scaf_charges = parse_molecule(scaffold)
#             fake_pos = np.mean(scaf_pos, axis = 0) # fake atom of scaf
#             fake_pos = scaf_pos[anchor_id]
#             rgroup_pos, rgroup_one_hot, rgroup_charges = parse_rgroup_com_scaf(rgroup, fake_pos)
#             # rgroup_pos, rgroup_one_hot, rgroup_charges = parse_molecule(rgroup)

#             pocket_pos = []
#             pocket_one_hot = []
#             pocket_charges = []
#             for i in range(len(pocket_data[f'{pocket_mode}_types'])):
#                 atom_type = pocket_data[f'{pocket_mode}_types'][i]
#                 pos = pocket_data[f'{pocket_mode}_coord'][i]
#                 if atom_type == 'H':
#                     continue
#                 pocket_pos.append(pos)
#                 pocket_one_hot.append(get_one_hot(atom_type, const.ATOM2IDX))
#                 pocket_charges.append(const.CHARGES[atom_type])
#             pocket_one_hot = np.array(pocket_one_hot)
#             pocket_charges = np.array(pocket_charges)
#             pocket_pos = np.array(pocket_pos)

#             positions = np.concatenate([scaf_pos, pocket_pos, rgroup_pos], axis=0)
#             one_hot = np.concatenate([scaf_one_hot, pocket_one_hot, rgroup_one_hot], axis=0)
#             charges = np.concatenate([scaf_charges, pocket_charges, rgroup_charges], axis=0)
#             anchors = np.zeros_like(charges)

#             for anchor_idx in map(int, row['anchor'].split('|')):
#                 anchors[anchor_idx] = 1

#             scaf_only_mask = np.concatenate([
#                 np.ones_like(scaf_charges),
#                 np.zeros_like(pocket_charges),
#                 np.zeros_like(rgroup_charges)
#             ])
#             pocket_mask = np.concatenate([
#                 np.zeros_like(scaf_charges),
#                 np.ones_like(pocket_charges),
#                 np.zeros_like(rgroup_charges)
#             ])
#             rgroup_mask = np.concatenate([
#                 np.zeros_like(scaf_charges),
#                 np.zeros_like(pocket_charges),
#                 np.ones_like(rgroup_charges)
#             ])
#             scaf_mask = np.concatenate([
#                 np.ones_like(scaf_charges),
#                 np.ones_like(pocket_charges),
#                 np.zeros_like(rgroup_charges)
#             ])

#             data.append({
#                 'uuid': uuid,
#                 'name': name,
#                 'positions': torch.tensor(positions, dtype=const.TORCH_FLOAT, device=device),
#                 'one_hot': torch.tensor(one_hot, dtype=const.TORCH_FLOAT, device=device),
#                 'charges': torch.tensor(charges, dtype=const.TORCH_FLOAT, device=device),
#                 'anchors': torch.tensor(anchors, dtype=const.TORCH_FLOAT, device=device),
#                 'scaffold_only_mask': torch.tensor(scaf_only_mask, dtype=const.TORCH_FLOAT, device=device),
#                 'pocket_mask': torch.tensor(pocket_mask, dtype=const.TORCH_FLOAT, device=device),
#                 'core_pocket_mask': torch.tensor(scaf_mask, dtype=const.TORCH_FLOAT, device=device),
#                 'rgroup_mask': torch.tensor(rgroup_mask, dtype=const.TORCH_FLOAT, device=device),
#                 'num_atoms': len(positions),
#             })

#         return data

#     @staticmethod
#     def create_edges(positions, scaffold_mask_only, rgroup_mask_only):
#         ligand_mask = scaffold_mask_only.astype(bool) | rgroup_mask_only.astype(bool)
#         ligand_adj = ligand_mask[:, None] & ligand_mask[None, :]
#         proximity_adj = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1) <= 6
#         full_adj = ligand_adj | proximity_adj
#         full_adj &= ~np.eye(len(positions)).astype(bool)

#         curr_rows, curr_cols = np.where(full_adj)
#         return [curr_rows, curr_cols]


def collate_mr(batch):
    out = {}
    for i, data in enumerate(batch):#enumerate得到迭代索引和值
        for key, value in data.items():#数据批次。
            out.setdefault(key, []).append(value)#将值传递给out数组，果字典中不存在该键，则返回这个默认值
    for key, value in out.items():
        if key in const.DATA_LIST_ATTRS:
            continue
        if key in const.DATA_ATTRS_TO_PAD:
            out[key] = torch.nn.utils.rnn.pad_sequence(value, batch_first=True, padding_value=0)#将长度不一致的序列补成一致的。batch_first=True的形状会是(batch_size, max_sequence_length)
            continue
        raise Exception(f'Unknown batch key: {key}')

    # 新增Atom Mask用于标记谁是padding的
    atom_mask = (out['core_pocket_mask'].bool() | out['rgroup_mask'].bool()).to(const.TORCH_INT)#找出他们都是1的部分，把padding的都标记成0，所有原子的掩码。
    out['atom_mask'] = atom_mask[:, :, None]

    batch_size, n_nodes = atom_mask.size()

    # Edges will be constructed by distance and batch_idx
    batch_mask = torch.cat([
        torch.ones(n_nodes, dtype=const.TORCH_INT) * i
        for i in range(batch_size)
    ]).to(atom_mask.device)
    out['batch_mask'] = batch_mask

    for key in const.DATA_ATTRS_TO_ADD_LAST_DIM:
        if key in out.keys():
            out[key] = out[key][:, :, None]

    return out

class SARDRG(Dataset):
    def __init__(self, prefix):
        prefix, pocket_mode = prefix.split('.')
        self.prefix = prefix
        self.processed_dir = 'SAR-DRG/pt'

        # If raw data are not processed to pt data file
        if not os.path.isdir(self.processed_dir):
            print(f'Processed data directory not found.')

        self.file_paths = sorted(glob.glob(os.path.join(self.processed_dir, '*.pt')))
        
        if not self.file_paths:
             raise RuntimeError(f"No '.pt' files found in {self.processed_dir}. Preprocessing might have failed.")
        
        print(f"Dataset ready. Foundn all {len(self.file_paths)} samples.")

        #Load split json file and construct dataset path
        with open("SAR-DRG/dataset_split.json", 'r', encoding='utf-8') as f:
            data_split = json.load(f)

        if prefix == 'train':
            data_id_list = data_split.get('train_data', [])
        elif prefix == 'test':
            data_id_list = data_split.get('test_data', [])

        if not data_id_list:
            print("Warning: Can not find {} data.".format(prefix))
            return

        self.split_file_paths = []
        for data_id in data_id_list:
            path = os.path.join(self.processed_dir, '{}.pt'.format(data_id))
            self.split_file_paths.append(path)

        print(f"Success load {len(self.split_file_paths)} {prefix} files。")

    def __len__(self):
        return len(self.split_file_paths)

    def __getitem__(self, item):
        file_path = self.split_file_paths[item]
        data = torch.load(file_path, map_location='cpu', weights_only = False)
        return data