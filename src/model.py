import numpy as np
import os
import pytorch_lightning as pl
import torch
import wandb

from src import utils
from src.egnn import SimpleDynamics
from src.edm import EDM
from src.datasets import (
    create_templates_for_rgroup_generation_multi, get_dataloader,
    SARDRG, collate_mr
)
from typing import Dict, List, Optional

def get_activation(activation):
    if activation == 'silu':#激活函数
        return torch.nn.SiLU()
    else:
        raise Exception("activation fn not supported yet. Add it here.")

import torch.nn as nn

import torch

class DDPM(pl.LightningModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None
    starting_epoch = None
    metrics: Dict[str, List[float]] = {}

    FRAMES = 100

    def __init__(
        self,
        in_node_nf, n_dims, context_node_nf, hidden_nf, activation, tanh, n_layers, attention, norm_constant,
        inv_sublayers, sin_embedding, normalization_factor, aggregation_method,
        diffusion_steps, diffusion_noise_schedule, diffusion_noise_precision, diffusion_loss_type,
        normalize_factors, include_charges, model,
        data_path, train_data_prefix, val_data_prefix, batch_size, lr, torch_device, test_epochs, n_stability_samples,
        normalization=None, log_iterations=None, data_augmentation=False,
        center_of_mass='scaffold', inpainting=False, anchors_context=True,  
    ):
        super(DDPM, self).__init__()
        
        
        self.save_hyperparameters()#用于自动保存超参数
        self.data_path = data_path
        self.train_data_prefix = train_data_prefix
        self.val_data_prefix = val_data_prefix
        self.batch_size = batch_size
        self.lr = lr
        self.torch_device = torch_device
        self.include_charges = include_charges
        self.test_epochs = test_epochs
        self.n_stability_samples = n_stability_samples
        self.log_iterations = log_iterations
        self.data_augmentation = data_augmentation
        self.center_of_mass = center_of_mass
        self.inpainting = inpainting#图像保存技术
        self.loss_type = diffusion_loss_type

        self.n_dims = n_dims
        self.num_classes = in_node_nf - include_charges
        self.include_charges = include_charges
        self.anchors_context = anchors_context

        self.is_geom = True

        if type(activation) is str:
            activation = get_activation(activation)

        dynamics = SimpleDynamics(
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            context_node_nf=context_node_nf,
            device=torch_device,
            hidden_nf=hidden_nf,
            activation=activation,
            n_layers=n_layers,
            attention=attention,
            tanh=tanh,
            norm_constant=norm_constant,
            inv_sublayers=inv_sublayers,
            sin_embedding=sin_embedding,
            normalization_factor=normalization_factor,
            aggregation_method=aggregation_method,
            model=model,
            normalization=normalization,
            centering=inpainting,
        )
        self.edm = EDM(
            dynamics=dynamics,
            in_node_nf=in_node_nf,
            n_dims=n_dims,
            timesteps=diffusion_steps,
            noise_schedule=diffusion_noise_schedule,
            noise_precision=diffusion_noise_precision,
            loss_type=diffusion_loss_type,
            norm_values=normalize_factors,
        )

    def setup(self, stage: Optional[str] = None):
        dataset_type = SARDRG # anchors
        if stage == 'fit':
            self.is_geom = True
            self.train_dataset = dataset_type(
                prefix=self.train_data_prefix
            )
            self.val_dataset = dataset_type(
                prefix=self.val_data_prefix
            )
        elif stage == 'val':
            self.is_geom = True
            self.val_dataset = dataset_type(
                prefix=self.val_data_prefix
            )
            
        else:
            raise NotImplementedError
        
    
    def train_dataloader(self, collate_fn=collate_mr):
        return get_dataloader(self.train_dataset, self.batch_size, collate_fn=collate_fn, num_workers = 4, persistent_workers=True, shuffle=True)
    
    def val_dataloader(self, collate_fn=collate_mr):
        return get_dataloader(self.val_dataset, self.batch_size, collate_fn=collate_fn, persistent_workers=True, num_workers = 4)

    def test_dataloader(self, collate_fn=collate_mr):
        return get_dataloader(self.test_dataset, self.batch_size, collate_fn=collate_fn, persistent_workers=True, num_workers = 4)#导入数据
        
    # 采样时将分类条件加入上下文
    def forward(self, data, training):
        #set p to 0 when validation
        if training:
            p_uncond = 0.2
        else:
            p_uncond = 0
        
        x = data['positions']
        h = data['one_hot']
        atom_mask = data['atom_mask']
        batch_mask = data['batch_mask']
        anchors = data['anchors']
        core_pocket_mask = data['core_pocket_mask']
        rgroup_mask = data['rgroup_mask']
        pocket_mask = data['pocket_mask']
        core_mask = data['core_mask']
        affinity = data['affinity']

        # Anchors, scaffolds labels, and pocket labels are used as context
        context = torch.cat([anchors, core_mask, pocket_mask], dim=-1)

        # 选定去中心化的中心
        if self.center_of_mass == 'core':
            center_of_mass_mask = core_mask
        elif self.center_of_mass == 'core_pocket':
            center_of_mass_mask = core_pocket_mask
        elif self.center_of_mass == 'anchors':
            center_of_mass_mask = anchors
        else:
            raise NotImplementedError(self.center_of_mass)
        
        # 去中心化
        x = utils.remove_partial_mean_with_mask(x, atom_mask, center_of_mass_mask)
        utils.assert_partial_mean_zero_with_mask(x, atom_mask, center_of_mass_mask)

        # Applying random rotation
        # if training and self.data_augmentation:
        #     x = utils.random_rotation(x)

        return self.edm.forward(
            x = x,
            h = h,
            node_mask = atom_mask,
            core_mask  = core_mask,
            rgroup_mask = rgroup_mask,
            core_pocket_mask = core_pocket_mask,
            batch_mask = batch_mask,
            context = context,
            center_of_mass_mask = center_of_mass_mask, 
            affinity_condition = affinity,
            p_uncond = p_uncond,
        )
    

    def common_step(self, data, stage: str):
        """
        一个通用的步骤，处理 train/val/test 的重复逻辑。
        :param data: 输入的数据批次。
        :param stage: 'train', 'val', 或 'test'。
        """
        # 1. 模型前向传播，获取所有指标
        delta_log_px, kl_prior, loss_term_t, loss_term_0, l2_loss, noise_t, noise_0 = self.forward(
            data, training=(stage == 'train')
        )
        batch_size = data['positions'].shape[0]
        vlb_loss = kl_prior + loss_term_t + loss_term_0 - delta_log_px

        # 2. 根据配置选择最终的 loss
        if self.loss_type == 'l2':
            loss = l2_loss
        elif self.loss_type == 'vlb':
            loss = vlb_loss
        else:
            raise NotImplementedError(self.loss_type)

        # 3. 将所有指标打包进一个字典
        metrics = {
            'loss': loss,
            'delta_log_px': delta_log_px,
            'kl_prior': kl_prior,
            'loss_term_t': loss_term_t,
            'loss_term_0': loss_term_0,
            'l2_loss': l2_loss,
            'vlb_loss': vlb_loss,
            'noise_t': noise_t,
            'noise_0': noise_0
        }

        # 4. 精简日志记录
        # 只在进度条上显示我们关心的核心指标
        prog_bar_metrics = ['loss']

        for name, value in metrics.items():
            # 判断当前指标是否需要在进度条上显示
            on_prog_bar = name in prog_bar_metrics
            
            # 使用 f-string 构造日志名称，例如 'loss/val'
            log_name = f'{name}-{stage}'
            
            # on_step 只在训练时开启，on_epoch 始终开启以获得平滑曲线
            self.log(log_name, value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=on_prog_bar,
                    batch_size=batch_size,
                    sync_dist=True) # val/test 需要跨设备同步

        return loss

    def training_step(self, data, *args):
        return self.common_step(data, 'train')

    def validation_step(self, data, *args):
        return self.common_step(data, 'val')

    def test_step(self, data, *args):
        return self.common_step(data, 'test')


    # Using multiple anchors on the same graph as context to diff each rgroup
    def sample_chain(self, data, sample_fn=None, keep_frames=None, target_affinity = None, guidance_scale = 3.0):
        if sample_fn is None:
            rgroup_sizes = data['rgroup_mask'].sum(1).view(-1).int()
        else:
            rgroup_sizes = sample_fn(data)

        if self.inpainting:
            template_data = data
        else:
            template_data = create_templates_for_rgroup_generation_multi(data, rgroup_sizes)

        x = template_data['positions']
        h = template_data['one_hot']
        atom_mask = template_data['atom_mask']
        batch_mask = template_data['batch_mask']
        anchors = template_data['anchors']
        core_pocket_mask = template_data['core_pocket_mask']
        rgroup_mask = template_data['rgroup_mask']
        pocket_mask = template_data['pocket_mask']
        core_mask = template_data['core_mask']

        # Anchors, scaffolds labels, and pocket labels are used as context
        context = torch.cat([anchors, core_mask, pocket_mask], dim=-1)

        # 选定去中心化的中心
        if self.center_of_mass == 'core':
            center_of_mass_mask = core_mask
        elif self.center_of_mass == 'core_pocket':
            center_of_mass_mask = core_pocket_mask
        elif self.center_of_mass == 'anchors':
            center_of_mass_mask = anchors
        else:
            raise NotImplementedError(self.center_of_mass)
        
        x_masked = x * center_of_mass_mask
        N = center_of_mass_mask.sum(1, keepdims=True)
        mean = torch.sum(x_masked, dim=1, keepdim=True) / N

        x = utils.remove_partial_mean_with_mask(x, atom_mask, center_of_mass_mask)
        utils.assert_partial_mean_zero_with_mask(x, atom_mask, center_of_mass_mask)

        chain = self.edm.sample_chain(
            x=x,
            h=h,
            node_mask=atom_mask,
            batch_mask=batch_mask,
            core_pocket_mask=core_pocket_mask,
            rgroup_mask=rgroup_mask,
            context=context,
            affinity_condition=target_affinity,
            keep_frames=keep_frames,
            guidance_scale = guidance_scale
        )#在这采样

        return chain, atom_mask, mean

    def configure_optimizers(self):
        return torch.optim.AdamW(self.edm.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12)

    def compute_best_validation_metrics(self):
        loss = self.metrics[f'validity_and_connectivity/val']
        best_epoch = np.argmax(loss)
        best_metrics = {
            metric_name: metric_values[best_epoch]
            for metric_name, metric_values in self.metrics.items()
            if metric_name.endswith('/val')
        }
        return best_metrics, best_epoch

    @staticmethod
    def aggregate_metric(step_outputs, metric):
        return torch.tensor([out[metric] for out in step_outputs]).mean()