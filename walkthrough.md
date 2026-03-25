# ChiPFormer — Setup & Run Walkthrough

## Setup Summary

| Item | Details |
|------|---------|
| **Repo** | [github.com/laiyao1/ChiPFormer](https://github.com/laiyao1/ChiPFormer) |
| **Location** | `z:\CHipFormer mini project\ChiPFormer` |
| **Python** | 3.11.9 (pyenv) |
| **PyTorch** | 2.5.1+cu121 (CUDA enabled) |
| **Extra deps** | networkx 2.6.3, scipy, tqdm, dgl |
| **Benchmark** | adaptec1 (included in repo) |
| **Pretrained model** | `save_models/trained_model.pkl` |

## Code Fixes Applied

Two compatibility fixes were needed for modern numpy/PyTorch:

1. **`create_dataset.py` line 209** — `int(sum(array))` → `int(array.sum())` (numpy 1.25+ scalar conversion)
2. **`mingpt/trainer_placement.py` line 231** — Added `if benchmark not in placedb_g_lib: continue` (skip missing benchmark data in eval loop)

## Execution

### Eval-only (10 rollouts on adaptec1)
```bash
python run_dt_place.py --is_eval_only --cuda 0 --context_length 256 --epochs 1 --batch_size 1
```

### Inference with full coordinates
```bash
python run_inference.py
```

## Results — adaptec1 (63 macros)

| Metric | Value |
|--------|-------|
| **HPWL** | 930,726.86 |
| **Steiner Tree Cost** | 1,012,778.14 |
| **Wirelength Reward** | -7,059.00 |
| **Placement Score** | 0.9916 |

### Sample Placements (first 10 of 63)

| Node | Grid_X | Grid_Y | Width | Height |
|------|--------|--------|-------|--------|
| o211437 | 82 | 43 | 164 | 2136 |
| o210917 | 26 | 26 | 1206 | 2856 |
| o210916 | 37 | 27 | 1206 | 2856 |
| o210915 | 15 | 26 | 1206 | 2856 |
| o210918 | 49 | 27 | 1206 | 2856 |
| o210927 | 60 | 27 | 701 | 2856 |
| o211423 | 43 | 54 | 511 | 2136 |
| o211424 | 48 | 54 | 511 | 2136 |
| o211425 | 37 | 54 | 511 | 2136 |
| o211426 | 53 | 51 | 511 | 2136 |

Full output: [inference_output.txt](file:///z:/CHipFormer%20mini%20project/ChiPFormer/inference_output.txt)

## Validation

- ✅ Model loaded (6.2M params)
- ✅ Dataset extracted and processed (2560 samples from `adaptec1_small.pkl`)
- ✅ 10 evaluation rollouts completed (`run_dt_place.py --is_eval_only`)
- ✅ Full placement with coordinates and metrics (`run_inference.py`, exit code 0)
- ✅ HPWL and Steiner cost computed successfully
