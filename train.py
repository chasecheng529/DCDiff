import argparse
import os
import pwd
import sys
import yaml
import random
import numpy as np

from pytorch_lightning.callbacks import Callback

class SetupCallback(Callback):
    def __init__(self, run_name, checkpoints_dir, log_dir):
        super().__init__()
        self.run_name = run_name
        self.checkpoints_dir = checkpoints_dir
        self.log_dir = log_dir

    def setup(self, trainer, pl_module, stage: str):
        """Run after DDP setup finishes and before training starts."""
        if trainer.is_global_zero:  # Use trainer.is_global_zero for robust main-process checks.
            print(f"====== Rank {trainer.global_rank} is setting up directories ======")
            # Keep all filesystem setup on the main process.
            os.makedirs(self.log_dir, exist_ok=True)
            os.makedirs(self.checkpoints_dir, exist_ok=True)

            # Redirect output only on the main process.
            sys.stdout = Logger(logpath=os.path.join(self.log_dir, 'log.log'), syspart=sys.stdout)
            sys.stderr = Logger(logpath=os.path.join(self.log_dir, 'log.log'), syspart=sys.stderr)

            print(f"Directories and logging are set up by the main process.")
        else:
            # Other ranks can still print a short debugging message.
            print(f"====== Rank {trainer.global_rank} is skipping directory setup ======")


def set_random_seed(seed=42):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)

    import torch  # Import torch here so it is seeded together with the other RNGs.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.set_float32_matmul_precision('high')

set_random_seed(42)

from datetime import datetime
from pytorch_lightning import Trainer, callbacks, loggers
from pytorch_lightning.strategies import DDPStrategy

from src.const import NUMBER_OF_ATOM_TYPES
from src.model import DDPM
from src.utils import disable_rdkit_logging, Logger

class CustomModelCheckpoint(callbacks.ModelCheckpoint):
    def _save_model(self, trainer, pl_module):
        current_loss = trainer.callback_metrics.get("loss-val", None)
        if current_loss is not None:
            self.filename = f"{trainer.logger.name}{trainer.current_epoch:02d}-loss_val={current_loss:.8f}.ckpt"
        super()._save_model(trainer, pl_module)
       
def find_last_checkpoint(checkpoints_dir):
    epoch2fname = [
        (int(fname.split('=')[1].split('.')[0]), fname)
        for fname in os.listdir(checkpoints_dir)
        if fname.endswith('.ckpt')
    ]
    latest_fname = max(epoch2fname, key=lambda t: t[0])[1]
    return os.path.join(checkpoints_dir, latest_fname)


def main(args):
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S')
    run_name = f'{args.version}_bs{args.batch_size}_nf{args.nf}_{start_time}'
    experiment = run_name if args.resume is None else args.resume
    checkpointname = f"{args.version}_bs={args.batch_size}_numf={args.nf}"
    checkpoints_dir = os.path.join(args.checkpoints, experiment)
    log_dir = os.path.join(args.logs, experiment)

    torch_device = 'cuda:0' if args.device == 'gpu' else 'cpu'

    number_of_atoms = NUMBER_OF_ATOM_TYPES
    in_node_nf = number_of_atoms + args.include_charges
    anchors_context = not args.remove_anchors_context
    #context_node_nf = 5 if anchors_context else 4
    context_node_nf = 2 if anchors_context else 1
    if '.' in args.train_data_prefix:
        context_node_nf += 1

    ddpm = DDPM(#model
        data_path=args.data,
        train_data_prefix=args.train_data_prefix,
        val_data_prefix=args.val_data_prefix,
        in_node_nf=in_node_nf,
        n_dims=3,
        context_node_nf=context_node_nf,
        hidden_nf=args.nf,
        activation=args.activation,
        n_layers=args.n_layers,
        attention=args.attention,
        tanh=args.tanh,
        norm_constant=args.norm_constant,
        inv_sublayers=args.inv_sublayers,
        sin_embedding=args.sin_embedding,
        normalization_factor=args.normalization_factor,
        aggregation_method=args.aggregation_method,
        diffusion_steps=args.diffusion_steps,
        diffusion_noise_schedule=args.diffusion_noise_schedule,
        diffusion_noise_precision=args.diffusion_noise_precision,
        diffusion_loss_type=args.diffusion_loss_type,
        normalize_factors=args.normalize_factors,
        include_charges=args.include_charges,
        lr=args.lr,
        batch_size=args.batch_size,
        torch_device=torch_device,
        model=args.model,
        test_epochs=args.test_epochs,
        n_stability_samples=args.n_stability_samples,
        normalization=args.normalization,
        log_iterations=args.log_iterations,
        data_augmentation=args.data_augmentation,
        center_of_mass=args.center_of_mass,
        inpainting=args.inpainting,
        anchors_context=anchors_context,
    )
    early_stopping_callback = callbacks.EarlyStopping(
        monitor='loss-val',   
        patience=200,         
        min_delta=0.00001,
        mode='min',           
        verbose=True          
    )
    best_checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=checkpointname + '_best_{epoch:02d}_{loss-val:.4f}',
        monitor='loss-val',
        save_top_k=1,
        mode='min',
        verbose=True
    )
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=checkpoints_dir,
        filename=checkpointname + '_{epoch:02d}',
        every_n_epochs=20,
        save_top_k=-1, 
        verbose=True
    )
    setup_callback = SetupCallback(
        run_name=experiment,
        checkpoints_dir=checkpoints_dir,
        log_dir=log_dir
    )

    # Select the training strategy based on the number of devices.
    if args.ddp_devices > 1:
        # Use DDPStrategy only when more than one device is requested.
        ddp_strategy = DDPStrategy(find_unused_parameters=False)
    else:
        # Let Lightning choose the single-device strategy.
        ddp_strategy = "auto"
    
    trainer = Trainer(
        max_epochs=args.n_epochs,
        strategy=ddp_strategy,
        callbacks=[checkpoint_callback, best_checkpoint_callback, early_stopping_callback, setup_callback],
        accelerator=args.device,
        devices=args.ddp_devices,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        logger=False
    )

    if args.resume is None:
        last_checkpoint = None
    else:
        last_checkpoint = find_last_checkpoint(checkpoints_dir)
        print(f'Training will be resumed from the latest checkpoint {last_checkpoint}')

    print('Start training')
    trainer.fit(model=ddpm, ckpt_path=last_checkpoint)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='E3Diffusion')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs/DCDiff.yml')
    p.add_argument('--data', action='store', type=str,  default="datasets")
    p.add_argument('--train_data_prefix', action='store', type=str, default='train')
    p.add_argument('--val_data_prefix', action='store', type=str,  default='val')
    p.add_argument('--checkpoints', action='store', type=str, default='checkpoints')
    p.add_argument('--logs', action='store', type=str, default='logs')
    p.add_argument('--device', action='store', type=str, default='cpu')
    p.add_argument('--trainer_params', type=dict, help='parameters with keywords of the lightning trainer')
    p.add_argument('--log_iterations', action='store', type=str, default=20)

    p.add_argument('--model', type=str, default='egnn_dynamics',help='our_dynamics | schnet | simple_dynamics | kernel_dynamics | egnn_dynamics |gnn_dynamics')
    p.add_argument('--probabilistic_model', type=str, default='diffusion', help='diffusion')

    p.add_argument('--diffusion_steps', type=int, default=500)
    p.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2', help='learned, cosine')
    p.add_argument('--diffusion_noise_precision', type=float, default=1e-5, )
    p.add_argument('--diffusion_loss_type', type=str, default='l2', help='vlb, l2')

    p.add_argument('--n_epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--brute_force', type=eval, default=False,help='True | False')
    p.add_argument('--actnorm', type=eval, default=True,help='True | False')
    p.add_argument('--break_train_epoch', type=eval, default=False,help='True | False')
    p.add_argument('--dp', type=eval, default=True,help='True | False')
    p.add_argument('--condition_time', type=eval, default=True,help='True | False')
    p.add_argument('--clip_grad', type=eval, default=True,help='True | False')
    p.add_argument('--trace', type=str, default='hutch',help='hutch | exact')
    # EGNN args -->
    p.add_argument('--n_layers', type=int, default=6,   help='number of layers')
    p.add_argument('--inv_sublayers', type=int, default=1, help='number of layers')
    p.add_argument('--nf', type=int, default=128,  help='number of layers')
    p.add_argument('--tanh', type=eval, default=True, help='use tanh in the coord_mlp')
    p.add_argument('--attention', type=eval, default=True, help='use attention in the EGNN')
    p.add_argument('--norm_constant', type=float, default=1,help='diff/(|diff| + norm_constant)')
    p.add_argument('--sin_embedding', type=eval, default=False, help='whether using or not the sin embedding')
    p.add_argument('--ode_regularization', type=float, default=1e-3)
    p.add_argument('--dataset', type=str, default='bingdingnet',  help='v')
    p.add_argument('--datadir', type=str, default='/bingdingnet/',  help='bingdingnet directory')
    p.add_argument('--filter_n_atoms', type=int, default=None, help='')
    p.add_argument('--dequantization', type=str, default='argmax_variational',  help='uniform | variational | argmax_variational | deterministic')
    p.add_argument('--n_report_steps', type=int, default=1)
    p.add_argument('--wandb_usr', type=str)
    p.add_argument('--no_wandb', action='store_true', help='Disable wandb')
    p.add_argument('--enable_progress_bar', action='store_true', help='Disable wandb')
    p.add_argument('--online', type=bool, default=True, help='True = wandb online -- False = wandb offline')
    p.add_argument('--no-cuda', action='store_true', default=False,  help='enables CUDA training')
    p.add_argument('--save_model', type=eval, default=True, help='save model')
    p.add_argument('--generate_epochs', type=int, default=1,help='save model')
    p.add_argument('--num_workers', type=int, default=16, help='Number of worker for the dataloader')
    p.add_argument('--test_epochs', type=int, default=1)
    p.add_argument('--data_augmentation', type=eval, default=False, help='use attention in the EGNN')
    p.add_argument("--conditioning", nargs='+', default=[], help='arguments : homo | lumo | alpha | gap | mu | Cv')
    p.add_argument('--resume', type=str, default=None, help='')
    p.add_argument('--start_epoch', type=int, default=0, help='')
    p.add_argument('--ema_decay', type=float, default=0.999, help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
    p.add_argument('--augment_noise', type=float, default=0)
    p.add_argument('--n_stability_samples', type=int, default=500,help='Number of samples to compute the stability')
    p.add_argument('--normalize_factors', type=eval, default=[1, 4, 1], help='normalize factors for [x, categorical, integer]')
    p.add_argument('--remove_h', action='store_true')
    p.add_argument('--include_charges', type=eval, default=True, help='include atom charge or not')
    p.add_argument('--visualize_every_batch', type=int, default=1e8,help="Can be used to visualize multiple times per epoch")
    p.add_argument('--normalization_factor', type=float, default=1,help="Normalize the sum aggregation of EGNN")
    p.add_argument('--aggregation_method', type=str, default='sum',help='"sum" or "mean"')
    p.add_argument('--normalization', type=str, default='batch_norm', help='batch_norm')
    p.add_argument('--center_of_mass', type=str, default='scaffold', help='Where to center the data: scaffold | anchors')
    p.add_argument('--inpainting', action='store_true', default=False, help='Inpainting mode (full generation)')
    p.add_argument('--remove_anchors_context', action='store_true', default=False, help='Remove anchors context')

    disable_rdkit_logging()

    args = p.parse_args()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list) and key != 'normalize_factors':
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}
    set_random_seed(getattr(args, 'seed', 42))
    main(args=args)
