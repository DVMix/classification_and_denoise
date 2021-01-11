# import numpy as np
# import os
# import json
# import math
# import matplotlib.pyplot as plt
# import time 
# from tqdm.notebook import tqdm

import torch
import torch.nn as nn

from utils.dataset import Denoising_DataSet
from utils.formatter import formatter_denoise
from utils.DnCNN import DnCNN
from utils.average_meter import denoising_stat_counter
from utils.Experiment import Experimet

size         = 320 #1374
epochs       = 11


base = True
if base:
     batch_size= 40
     model     = DnCNN(channels=1, num_of_layers=6)
     arch_name = 'DnCNN6'
else:
     batch_size= 20
     num_of_layers = 3
     model     = DnCNN(channels=1, num_of_layers=num_of_layers)
     arch_name = 'DnCNN{}'.format(num_of_layers)

ds_train     = Denoising_DataSet(folder = 'dataset',split='train')
ds_val       = Denoising_DataSet(folder = 'dataset',split='val')
optimizer    = torch.optim.Adam(params=model.parameters(), lr = 1e-3)
loss_fn      = torch.nn.MSELoss()
output_dir   = 'checkpoints'
task_name    = 'Denoise'
stat_counter = denoising_stat_counter(batch_size = batch_size)

ex = Experimet(task_name = task_name,
               arch_name = arch_name, 
               net=model, 
               epochs=epochs,
               train_set=ds_train, 
               val_set=ds_val,
               formatter=formatter_denoise, 
               optimizer=optimizer,
               loss_fn=loss_fn,
               stat_counters = stat_counter, 
               output_dir=output_dir,
               img_size=size,
               batch_size=batch_size)
ex.run()