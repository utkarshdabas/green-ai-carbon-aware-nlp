import torch
import os

# Limit CPU threads to prevent overheating
torch.set_num_threads(4)

print("Using CPU threads:", torch.get_num_threads())
