# Mini Transformer Benchmark

This project implements a mini Transformer encoder **from scratch** for the synthetic sequence classification assignment.

## Task
Predict whether the **first non-padding token** appears again in the **second half** of the valid sequence.

## Expected dataset files
Put these files in the project folder:
- `train.csv`
- `validation.csv`
- `test.csv`

The code reads these columns directly:
- `seq_len`
- `label`
- `token_01` to `token_20`
- `mask_01` to `mask_20`

## Project files
- `data.py` - dataset + dataloaders
- `model.py` - Transformer implementation
- `train.py` - single training run
- `benchmark.py` - required benchmark experiments
- `utils.py` - helper functions

## Setup
```bash
python -m venv .venv
```

### Activate environment
Windows:
```bash
.venv\Scripts\activate
```

Mac/Linux:
```bash
source .venv/bin/activate
```

### Install packages
```bash
pip install torch pandas numpy matplotlib
```

## Run one model
```bash
python train.py --train_csv train.csv --val_csv validation.csv --test_csv test.csv --run_name model_A --use_positional_encoding --num_heads 4 --num_layers 1 --epochs 10
```

## Run benchmark
```bash
python benchmark.py --train_csv train.csv --val_csv validation.csv --test_csv test.csv --epochs 10
```

## Notes
- By default, outputs are saved in the `outputs/` folder.
- Use `--verify_labels` once if you want to check dataset labels.
- Use `--cpu` if you do not want CUDA.
