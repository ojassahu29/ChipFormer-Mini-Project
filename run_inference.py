"""
Quick script to run ChiPFormer inference and print placement coordinates.
Uses the pretrained model on adaptec1 benchmark.
"""
import os
import sys
import numpy as np
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from mingpt.utils import set_seed
from mingpt.model_placement import GPT, GPTConfig
from mingpt.trainer_placement import Trainer, TrainerConfig

set_seed(123)
grid = 84

# Build model
mconf = GPTConfig(grid**2, 256*3, n_layer=6, n_head=8, n_embd=128,
                  model_type="reward_conditioned", max_timestep=256)
model = GPT(mconf)

# Load pretrained weights
state_dict = torch.load("save_models/trained_model.pkl", weights_only=False)
clean_dict = {}
for k, v in state_dict.items():
    key = k.split('.', 1)[1] if "module." in k else k
    clean_dict[key] = v
model.load_state_dict(clean_dict, strict=True)
model.eval()
print("Model loaded successfully.")

# Setup trainer for inference
tconf = TrainerConfig(max_epochs=1, batch_size=1, learning_rate=1e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2e10,
                      num_workers=0, seed=123, model_type="reward_conditioned",
                      max_timestep=256, draw_placement=False,
                      is_eval_only=True, test_all_macro=True)
trainer = Trainer(model, None, None, tconf)

# Run single placement on adaptec1 with all macros
print("\n" + "="*60)
print("Running ChiPFormer placement on adaptec1 (all macros)...")
print("="*60)

outputs = trainer.get_returns(1.1, is_single=True, benchmark="adaptec1", is_all_macro=True)

print(f"\nTotal wirelength reward: {outputs['reward']:.2f}")
print(f"Placement score: {outputs['score']:.4f}")

# Now replay the actions to get coordinates
from mingpt.trainer_placement import Args, Env
args = Args(123)
env = Env(args, "adaptec1", is_all_macro=True)
state, reward_sum, done, meta_state = env.reset()

actions = outputs['actions'].squeeze().tolist()
step = 0
placements = []
while not done and step < len(actions):
    action = actions[step]
    x = action // grid
    y = action % grid
    node_name = env.placedb.node_id_to_name[step]
    node_w = env.placedb.node_info[node_name]['x']
    node_h = env.placedb.node_info[node_name]['y']
    placements.append((node_name, x, y, node_w, node_h))
    state, reward, done, meta_state = env.step(action)
    step += 1

# Print placement results
print(f"\n{'='*60}")
print(f"PLACEMENT RESULTS - adaptec1 ({len(placements)} macros placed)")
print(f"{'='*60}")
print(f"{'Node':<15} {'Grid_X':>7} {'Grid_Y':>7} {'Width':>8} {'Height':>8}")
print("-" * 60)
for name, gx, gy, w, h in placements:
    print(f"{name:<15} {gx:>7} {gy:>7} {w:>8.0f} {h:>8.0f}")

# Compute HPWL
hpwl, cost = env.comp_res()
print(f"\n{'='*60}")
print(f"FINAL METRICS")
print(f"{'='*60}")
print(f"HPWL (Half-Perimeter Wirelength): {hpwl:.2f}")
print(f"Steiner Tree Cost:                {cost:.2f}")
print(f"Total Reward:                     {outputs['reward']:.2f}")
print(f"Placement Score:                  {outputs['score']:.4f}")
print(f"\nExecution completed successfully!")
