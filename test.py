import os
import sys
import torch
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from engine import MonoconEngine
from utils.engine_utils import load_cfg


# Arguments
parser = argparse.ArgumentParser('MonoCon Tester')
parser.add_argument('--config_file',
                    type=str,
                    help="Path of the config file (.yaml)")
parser.add_argument('--checkpoint_file', 
                    type=str,
                    help="Path of the checkpoint file (.pth)")
parser.add_argument('--gpu_id', type=int, default=0, help="Index of GPU to use for testing")
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--save_dir', 
                    type=str,
                    help="Path of the directory to save the visualized results")

args = parser.parse_args()


# Some Torch Settings
torch_version = int(torch.__version__.split('.')[1])
if torch_version >= 7:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# Initialize Engine
cfg = load_cfg(args.config_file)
cfg.GPU_ID = args.gpu_id

engine = MonoconEngine(cfg, auto_resume=False)
engine.load_checkpoint(args.checkpoint_file, check_version=True, verbose=True)


# Evaluate
if args.evaluate:
    engine.evaluate()


# Visualize
# Unavailable Now
if args.visualize:
    raise NotImplementedError