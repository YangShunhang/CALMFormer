# CALMFormer Minimal Reproducible Code

This folder contains a minimal runnable version of CALMFormer for few-shot
specific emitter identification.

## Files

- `run_calmformer.py`: short entry point for the manuscript mainline.
- `train_eval.py`: configurable training and few-shot evaluation script.
- `rf_metaformer.py`: CALMFormer / MetaFormer model definitions.
- `dataset.py`: numpy dataset loader.
- `fewshot_sampler.py`: base-to-novel support/query episode sampler.

## Dataset Layout

Datasets are not included. Put numpy files under:

```text
CALMFormer/
  Datasets/
    ORACLE/
      X_train_16Class.npy
      Y_train_16Class.npy
      X_test_16Class.npy
      Y_test_16Class.npy
```

Each `X` file should have shape `[N, 2, L]` or `[N, L, 2]`; labels are integer
class IDs. The loader applies per-sample, per-I/Q-channel temporal z-score
normalization.

## Minimal Run

After placing the ORACLE numpy files under `Datasets/ORACLE/`, run the
default ORACLE 16-class, 5-way 1-shot CALMFormer setting:

```bash
python run_calmformer.py
```

Run 5-shot:

```bash
python run_calmformer.py --k_shot 5
```

Quick smoke test:

```bash
python run_calmformer.py --epochs 2 --n_episodes 20
```

The short runner only exposes the most common knobs. The full minimal training
entry also works:

```bash
python train_eval.py
```

## Default Mainline Settings

`run_calmformer.py` fixes the manuscript mainline defaults:

- dataset: `ORACLE`
- classes: `16`
- mixer: `rf_ilcm_anchor`
- RF-Aug recipe: `full`
- initial RF residual release: `eta0 = 0.7`
- split seed: `2027`
- train seed: `2027`
- episode seed: `9001`
- epochs: `150`
- final episodes: `300`

Checkpoints are saved to `checkpoints_oracle_calmformer/` by default.

## Environment

The code uses Python, NumPy and PyTorch. The manuscript experiments used
Python 3.8 and PyTorch 1.11.0 with CUDA 11.3.

Install the minimal dependencies with:

```bash
pip install -r requirements.txt
```
