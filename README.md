# DCDiff

This is the anonymous repository for the BIBM 2026 paper **DCDiff: Dual-Control Diffusion for Affinity-Guided R-chain Generation in Lead Optimization**.

## Data Preparation

Download `data.tar.gz` from the SAR-DRG dataset page:

<https://huggingface.co/datasets/ChaseCheng/SAR-DRG>

You can also download it with:

```bash
wget https://huggingface.co/datasets/ChaseCheng/SAR-DRG/resolve/main/data.tar.gz -O SAR-DRG/data.tar.gz
```

Place `data.tar.gz` under `SAR-DRG/`, then extract it inside `SAR-DRG/` so that the raw data directory is `SAR-DRG/data`:

```bash
cd SAR-DRG
tar -xzf data.tar.gz
cd ..
```

After extraction, the expected structure is:

```text
SAR-DRG/
  data/
  metadata.csv
  dataset_split.json
  utils/
```

Run the data preprocessing script in `SAR-DRG/utils` to create the processed `pt` directory:

```bash
cd SAR-DRG/utils
python data_preprocessor.py
cd ../..
```

The generated files should be placed under:

```text
SAR-DRG/
  pt/
    0.pt
    1.pt
    ...
```

Note: the preprocessing script in this repository is named `data_preprocessor.py`. It uses relative paths such as `../data` and `../metadata.csv`, so it should be run from `SAR-DRG/utils`.

## Training

Run training with the provided configuration file:

```bash
python train.py --config configs/DCDiff.yml
```

DDP can be adjusted in `configs/DCDiff.yml` by changing:

```yaml
ddp_devices: 1
```

For example, increase `ddp_devices` when using more GPUs with DDP.

## Sampling

Run:

```bash
bash sample.sh
```

Before sampling, check the editable variables at the top of `sample.sh`, especially `CHECKPOINT_PATH`, `GPU_ID`, `N_SAMPLES`, `AFFINITY`, and `GUIDENCE`.

## Evaluation

Evaluation scripts are provided under the `benchmark/` directory. To evaluate sampled molecules, edit `benchmark/evaluate_pipline.sh` and replace `SAMPLES_PATH` with the path to your sampling output directory.

Then run:

```bash
cd benchmark
bash evaluate_pipline.sh
cd ..
```
