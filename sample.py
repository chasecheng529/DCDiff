import argparse
import os
import sys
import math
import torch
import pandas as pd
import json

from src.model import DDPM
from src.visualizer import save_xyz_file_fa
from src.datasets import collate_mr
from tqdm import tqdm
import subprocess
import time
from rdkit import Chem

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', action='store', type=str, required=True)
parser.add_argument('--samples', action='store', type=str, required=True)
parser.add_argument('--data', action='store', type=str, required=False, default=None)
parser.add_argument('--prefix', action='store', type=str, required=True)
parser.add_argument('--n_samples', action='store', type=int, required=True)
parser.add_argument('--n_steps', action='store', type=int, required=False, default=None)
parser.add_argument('--device', action='store', type=str, required=True)
parser.add_argument('--rgroup_size_model', action='store', type=str, required=False, default=None)
parser.add_argument('--affinity', action='store', type=float, required=True, default=None)
parser.add_argument('--guidance_scale', action='store', type=float, required=True, default=None)
parser.add_argument('--batch_size', action='store', type=int, required=True, default=None)
args = parser.parse_args()

experiment_name = args.checkpoint.split('/')[-1].replace('.ckpt', '')

output_dir = os.path.join(args.samples, experiment_name)
output_dir = output_dir + "/A{}_W{}".format(args.affinity, args.guidance_scale)

os.makedirs(output_dir, exist_ok=True)

def check_if_generated(_output_dir, _index, n_samples):
    generated = True
    starting_points = []
    uuid_dir = os.path.join(_output_dir, _index)
    numbers = []
    for fname in os.listdir(uuid_dir):
        try:
            num = int(fname.split('_')[0])
            numbers.append(num)
        except:
            continue
    if len(numbers) == 0 or max(numbers) != n_samples - 1:
        generated = False
        if len(numbers) == 0:
            starting_points.append(0)
        else:
            starting_points.append(max(numbers) - 1)

    if len(starting_points) > 0:
        starting = min(starting_points)
    else:
        starting = None

    return generated, starting

def expand_data_func(data: dict, batch_size: int) -> dict:
    # Verify that the input batch size is 1.
    if 'positions' in data and data['positions'].shape[0] != 1:
        raise ValueError(
            f"expand_data: input batch size (B) must be 1. "
            f"B={data['positions'].shape[0]}. "
            f"Please ensure dataloader batch_size=1."
        )
        
    expanded_data = {}

    # Expand every value in the data dictionary.
    for key, value in data.items():
        
        if isinstance(value, torch.Tensor):
            # Tensor values with a leading batch dimension are repeated.
            
            if value.shape[0] == 1:
                # Batched tensor, e.g. [1, N, 3], [1, N, F], or [1].
                
                if value.dim() == 1:
                    # 1D tensor, e.g. batch_new_len_tensor [1].
                    # repeat(n_samples) -> [100]
                    repeat_dims = [batch_size]
                else:
                    # Higher-dimensional tensor, e.g. positions [1, N, 3].
                    # repeat_dims = [100, 1, 1]
                    repeat_dims = [batch_size] + [1] * (value.dim() - 1)
                
                # Use repeat() to create the expanded tensor.
                expanded_data[key] = value.repeat(*repeat_dims)
            
            else:
                # Non-batched tensors, e.g. edge_index [2, E], are shared.
                expanded_data[key] = value
                
        elif isinstance(value, list) or isinstance(value, tuple):
            # Lists and tuples are expanded when they contain one element.
            
            if len(value) == 1:
                # Expand [element] to [element, element, ..., element].
                expanded_data[key] = [value[0]] * batch_size
            else:
                # Leave non-singleton sequences unchanged.
                expanded_data[key] = value
                
        else:
            # Copy scalar and other non-container values unchanged.
            expanded_data[key] = value

    return expanded_data

collate_fn = collate_mr
sample_fn = None

# Loading model form checkpoint (all hparams will be automatically set)
model = DDPM.load_from_checkpoint(args.checkpoint, map_location=args.device)

# Possibility to evaluate on different datasets (e.g., on CASF instead of ZINC)
model.val_data_prefix = args.prefix

# In case <Anonymous> will run my model or vice versa
if args.data is not None:
    model.data_path = args.data

# Less sampling steps
if args.n_steps is not None:
    model.edm.T = args.n_steps

# Setting up the model
model = model.eval().to(args.device)
model.setup(stage='val')

model.batch_size = 1
# Getting the dataloader
dataloader = model.val_dataloader(collate_fn=collate_fn)
print(f'Dataloader contains {len(dataloader)} molecules')

center_of_mass_list = []

time_start = time.time()
core_pocket_data = {}
# Collect one sample for each group in the validation split.
for _, data in enumerate(dataloader):
    group_id = data['group_id'][0] #group_id is a list contains 1 element
    if group_id not in core_pocket_data:
        core_pocket_data[group_id] = data

for sample_index, (group_id, group_data) in enumerate(core_pocket_data.items(), start = 0):
    data = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in group_data.items()}
    scaf_name = (f'{sample_index}/core')
    pock_name = (f'{sample_index}/pock')
    os.makedirs(os.path.join(output_dir, str(sample_index)), exist_ok=True)

    generated, starting_point = check_if_generated(output_dir, str(sample_index), args.n_samples)
    if generated:
        print(f'Already generated {sample_index} groups, group_id = {group_id}')
        continue
    if starting_point > 0:
        print(f'Generating {args.n_samples - starting_point} for {sample_index}th group: {group_id}')
    
    samples_need_to_gen = args.n_samples - starting_point
    total_generated_so_far = starting_point
    num_iter_gen = math.ceil(samples_need_to_gen / args.batch_size)

    h, x, node_mask = data['one_hot'], data['positions'], data['atom_mask']

    node_mask = data['atom_mask'] - data['pocket_mask']
    core_mask = data['core_mask']
    pock_mask = data['pocket_mask']
    save_xyz_file_fa(output_dir, h, x, pock_mask, [pock_name]) # Wrap the pocket name in a list for the xyz writer.
    out_xyz_pock = f'{output_dir}/{pock_name}_.xyz'
    out_pdb_pock = f'{output_dir}/{pock_name}_.pdb'
    subprocess.run(f'obabel {out_xyz_pock} -O {out_pdb_pock} 2> /dev/null', shell=True)

    # Save scaffold as .xyz and convert to .sdf
    save_xyz_file_fa(output_dir, h, x, core_mask, [scaf_name])
    out_xyz_scaf = f'{output_dir}/{scaf_name}_.xyz'
    out_sdf_scaf = f'{output_dir}/{scaf_name}_.sdf'
    subprocess.run(f'obabel {out_xyz_scaf} -O {out_sdf_scaf} 2> /dev/null', shell=True)

    # Save gt ref mol as sdf
    df = pd.read_csv('SAR-DRG/metadata.csv')
    group_df = df[df['group_id'] == group_id].copy().reset_index(drop=True)
    for index, row in group_df.iterrows():
        raw_mol_path = row['mol_path']
        mol_file_name = os.path.basename(raw_mol_path)
        mol_path = f'SAR-DRG/data/{group_id}/{mol_file_name}'
        suppl = Chem.SDMolSupplier(mol_path)
        mol = suppl[0]
        xyz_content = Chem.MolToXYZBlock(mol)
        out_xyz_gt = f'{output_dir}/{sample_index}/gt_{index}.xyz'
        with open(out_xyz_gt, 'w') as f:
            f.write(xyz_content)
        out_sdf_gt = f'{output_dir}/{sample_index}/gt_{index}.sdf'
        subprocess.run(f'obabel {out_xyz_gt} -O {out_sdf_gt} 2> /dev/null', shell=True)

    target_affinity = data["affinity"]
    target_affinity = target_affinity + args.affinity
    target_affinity = torch.clamp(target_affinity, max=12.22) # 12.22 is the max affinity in the training set, we do not want to exceed this value to avoid out-of-distribution generation

    # Sampling and saving generated molecules
    for i in tqdm(range(num_iter_gen), desc="Process {} Group: {}".format(sample_index, group_id)):
        #Compute the mols need to be gen, and adjust batch_size
        samples_remaining_in_total = args.n_samples - total_generated_so_far
        current_iter_batch_size = min(args.batch_size, samples_remaining_in_total)

        expanded_data = expand_data_func(data, current_iter_batch_size)
        target_affinity_expanded = target_affinity.repeat(current_iter_batch_size, 1)
        chain, node_mask, mean = model.sample_chain(expanded_data, sample_fn=sample_fn, keep_frames=1, target_affinity = target_affinity_expanded, guidance_scale = args.guidance_scale) # Start sampling.

        x_batch = chain[-1][:, :, :model.n_dims] # [B, N, 3]
        h_batch = chain[-1][:, :, model.n_dims:] # [B, N, F]
        x_batch += mean
        
        x_rgroup_batch = x_batch * expanded_data['rgroup_mask'] # Predicted rgroup positions.
        x_core_pocket_batch = expanded_data['positions'] * expanded_data['core_pocket_mask'] # Original core and pocket positions.
        x_final_batch = x_core_pocket_batch + x_rgroup_batch # Merge into full molecule positions.

        h_rgroup_batch = h_batch * expanded_data['rgroup_mask']
        h_core_pocket_batch = expanded_data['one_hot'] * expanded_data['core_pocket_mask']
        h_final_batch = h_core_pocket_batch + h_rgroup_batch 
        
        x = x_final_batch 
        h = h_final_batch 
        node_mask = expanded_data['atom_mask'] - expanded_data['pocket_mask'] # Keep only the molecular atoms.

        pred_names_batch = []
        for k in range(current_iter_batch_size): # Iterate over B, e.g. 8 or 4.
            # Globally unique sample index, e.g. 50 + 0, 50 + 1, ...
            cur_sample_index = total_generated_so_far + k 
            pred_names_batch.append(f'{sample_index}/{cur_sample_index}')

        save_xyz_file_fa(output_dir, h, x, node_mask, pred_names_batch)
        
        for pred_name in pred_names_batch: # This remains serial and is acceptable here.
            out_xyz = f'{output_dir}/{pred_name}_.xyz'
            out_sdf = f'{output_dir}/{pred_name}_.sdf'
            subprocess.run(f'obabel {out_xyz} -O {out_sdf} 2> /dev/null', shell=True)
            
        # Update the total counter.
        total_generated_so_far += current_iter_batch_size
time_end = time.time()
print('sample time:', time_end - time_start, 's')
