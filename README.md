## ChiPFormer: Transferable Chip Placement via Offline Decision Transformer

ChiPFormer is an offline RL-based placement method that significantly improves design quality and efficiency.

### Publication
Yao Lai, Jinxin Liu, Zhentao Tang, Bin Wang, Jianye Hao, Ping Luo. "ChiPFormer: Transferable Chip Placement via Offline Decision Transformer." International Conference on Machine Learning (ICML 2023): 18346-18364.

[paper](https://arxiv.org/pdf/2306.14744.pdf) | [dataset](https://drive.google.com/drive/folders/1F7075SvjccYk97i2UWhahN_9krBvDCmr) | [website](https://sites.google.com/view/chipformer/home) | [video(English)](https://www.youtube.com/watch?v=9-EQmDjRLHQ) | [video(Mandarin)](https://www.bilibili.com/video/BV1ym4y177CC/)

### Environment Setup (Updated 2026)

This repository has been tested with Python 3.11.9 and PyTorch 2.5.1+cu121.

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.5/cu121/repo.html
pip install networkx==2.6.3 scipy tqdm
```

### Usage

#### Download the offline placement dataset

Download the offline placement dataset from [Google Drive](https://drive.google.com/drive/folders/1F7075SvjccYk97i2UWhahN_9krBvDCmr).
We provide placement data for 12 benchmarks: adaptec1-4, bigblue1-4, ibm01-04. You may download only the benchmark you need. (Updated: 2024-03-02)

For a quick start, you can extract the *adaptec1_small.pkl* archive included in this repo:

```bash
tar -zxvf adaptec1_small.pkl.tar.gz
```

#### Download placement benchmark files

Our test set includes the ISPD05 and ICCAD04 benchmarks. We include *adaptec1* for a quick start.
All benchmarks can also be downloaded from *placement_bench.zip* in [Google Drive](https://drive.google.com/drive/folders/1F7075SvjccYk97i2UWhahN_9krBvDCmr).

#### Quick start

ChiPFormer consists of two stages: pretraining and fine-tuning.

- For pretraining:

```bash
python run_dt_place.py
```

You can change the training dataset file/path in *create_dataset.py*.
Saved models will be in *save_models/*.

- For fine-tuning:

```bash
python odt.py --benchmark=adaptec1
```

The model checkpoint path for fine-tuning can be set in *odt.py*. Typically, you fine-tune using the best checkpoint from pretraining.

#### VGAE-based circuit feature extraction

(Updated: 2024/05/25)

To use VGAE, you need to first train the model and then run evaluation on the target benchmarks.

```bash
python graph_train.py
```

The trained model will be saved to `save_graph_models/`.

```bash
python graph_eval.py
```

Circuit embeddings are saved to `circuit_g_token*.pkl`.

Then, in `mingpt/place_db.py`, change:

```py
def __init__(self, benchmark = None, offset = 0, is_graph = False):
```

to:

```py
def __init__(self, benchmark = None, offset = 0, is_graph = True):
```

We also provide example models and embedding files that you can use directly.

### Parameters

For *run_dt_place.py*:

- **seed** Random seed.
- **context_length** Maximum sequence length for the Decision Transformer.
- **epochs** Maximum training epochs.
- **batch_size** Batch size.
- **cuda** GPU id.
- **is_eval_only** Whether to run evaluation only. (In evaluation, it will place all macros rather than only up to `context_length`.)
- **test_all_macro** Whether to place all macros.

For *odt.py*:
- **replay_size** Replay buffer size for fine-tuning.
- **traj_len** Maximum sequence length for the Decision Transformer.
- **batch_size** Batch size.
- **benchmark** Circuit benchmark for fine-tuning.
- **max_online_iters** Maximum number of fine-tuning iterations.
- **eval_interval** Evaluate every N iterations.
- **exploration_rtg** Return-to-go value for exploration.
- **is_fifo** Whether to use a FIFO buffer or a priority buffer.
- **cuda** GPU id.

### Dependencies
- [Python](https://www.python.org/) >= 3.9
- [PyTorch](https://pytorch.org/) >= 1.10  
  - Other versions may also work, but have not been tested.
- [tqdm](https://tqdm.github.io/)

### Reference code
This code is based on the following open-source repos:
- [decision-transformer](https://github.com/kzl/decision-transformer)
- [online-dt](https://github.com/facebookresearch/online-dt)

### Citation
If you find our paper/code useful in your research, please cite:

```bibtex
@inproceedings{lai2023chipformer,
  author       = {Lai, Yao and Liu, Jinxin and Tang, Zhentao and Wang, Bin and Hao, Jianye and Luo, Ping},
  title        = {ChiPFormer: Transferable Chip Placement via Offline Decision Transformer},
  booktitle    = {International Conference on Machine Learning, {ICML} 2023, 23-29 July
                  2023, Honolulu, Hawaii, {USA}},
  series       = {Proceedings of Machine Learning Research},
  volume       = {202},
  pages        = {18346--18364},
  publisher    = {{PMLR}},
  year         = {2023},
  url          = {https://proceedings.mlr.press/v202/lai23c.html},
}
```
