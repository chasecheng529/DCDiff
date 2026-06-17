import torch

from rdkit import Chem

TORCH_FLOAT = torch.float32
TORCH_INT = torch.int8

# Atom idx for one-hot encoding '#'-fake atom
ATOM2IDX = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4, 'Mg': 5,
            'Si': 6, 'P': 7, 'S': 8, 'Cl': 9, 'Ca': 10,
            'Mn': 11, 'Fe': 12, 'Co': 13, 'Ni': 14, 'Zn': 15,
            'Se': 16, 'Br': 17, 'I': 18, '#': 19}

IDX2ATOM =  {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F', 5: 'Mg',
             6: 'Si', 7: 'P', 8: 'S', 9: 'Cl', 10: 'Ca',
             11: 'Mn', 12: 'Fe', 13: 'Co', 14: 'Ni', 15: 'Zn',
             16: 'Se', 17: 'Br', 18: 'I', 19: '#'}

# Atoms appearing in protein pockets and ligands
POCKET_ATOMS = {'H', 'C', 'N', 'O', 'Mg', 'S', 'Ca', 'Mn', 'Fe', 'Co', 'Ni', 'Zn'}

LIGAND_ATOMS = {'H', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Se', 'Br', 'I'}

ATOMS = {'H', 'C', 'N', 'O', 'F',
        'Mg', 'Si', 'P', 'S', 'Cl',
        'Ca', 'Mn', 'Fe', 'Co', 'Ni',
        'Zn', 'Se', 'Br', 'I'}

# Element name in PDB to standard element name
POCKET_ELEMENT_MAP = {'MG': 'Mg', 'CA': 'Ca', 'MN': 'Mn', 'FE': 'Fe', 'CO': 'Co', 'NI': 'Ni','ZN': 'Zn'}

# Atomic numbers
CHARGES = {'#': 0, 'H': 1, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Mg': 12, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ca': 20, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Zn': 30, 'Se': 34, 'Br': 35, 'I': 53}

CHARGES_LIST = [0, 1, 6, 7, 8, 9, 12, 14, 15, 16, 17, 20, 25, 26, 27, 28, 30, 34, 35, 53]

# One-hot atom types count
NUMBER_OF_ATOM_TYPES = len(ATOM2IDX)

# Dataset keys
DATA_LIST_ATTRS = {
    'data_id', 'name', 'scaffold_smi', 'rgroup_smi', 'num_atoms', 'cat', 'rgroup_size', 'anchors_str', 'edge_index', 'group_id'
}
DATA_ATTRS_TO_PAD = {
    'positions', 'one_hot', 'charges', 'anchors', 'core_mask', 'rgroup_mask', 'pocket_mask', 'core_pocket_mask', 'affinity', 'rgoup_true_atom_mask','rgroup_inner_mask'
}

DATA_ATTRS_TO_ADD_LAST_DIM = {
    'charges', 'anchors', 'core_mask', 'rgroup_mask', 'pocket_mask', 'core_pocket_mask','rgoup_true_atom_mask','rgroup_inner_mask'
}

MARGINS_EDM = [10, 5, 2]