import os
import sys
import sys, time, os, pdb, argparse, pickle, subprocess, glob
import torch

checkpint_path = 'checkpoints/my_vocoder'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_dict = torch.load(checkpoint_path, map_location="cpu")
if '.yaml' in config_path:
    config = set_hparams(config_path, global_hparams=False)
    # state = ckpt_dict["state_dict"]["model_gen"]

    # 0519
    new_state_dict = {}
    prefix = "net_g."
    for key, value in ckpt_dict["state_dict"].items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]  # 去除前缀
            new_state_dict[new_key] = value


elif '.json' in config_path:
    config = json.load(open(config_path, 'r'))
    state = ckpt_dict["generator"]

model = HifiGanGenerator(config)
# model.load_state_dict(state, strict=True)
model.load_state_dict(new_state_dict, strict=True)