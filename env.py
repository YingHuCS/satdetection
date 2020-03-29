import os
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp



def set_env(env_config):
    set_dist(**env_config.dist)
    set_random_seed(env_config.random_seed)
    set_cuddnn(**env_config.cudnn)



def set_dist(backend='nccl', start_method='spwan'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method(start_method)

    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend)



def set_random_seed(seed=42):
    if seed is not None:
        random.seed(seed) # Python random module.
        np.random.seed(seed)  # Numpy module.
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.



def set_cuddnn(enabled=False, benchmark=False, deterministic=True):
    torch.backends.cudnn.enabled = enabled
    torch.backends.cudnn.benchmark = benchmark
    torch.backends.cudnn.deterministic = deterministic

