# import numpy as np
# import os
# import json
# import math
# import matplotlib.pyplot as plt
# import time 
# from tqdm.notebook import tqdm

import torch
import torch.nn as nn

from utils.dataset import Classification_DataSet
from utils.formatter import formatter_classification
from utils.resnet import resnet50, resnet50_half_conv

from utils.average_meter import classification_stat_counter
from utils.Experiment import Experimet

size         = 320
epochs       = 11


full = True
if full:
     batch_size= 15
     model     = resnet50(num_classes = 2)
     arch_name = 'ResNet50'
else:
     batch_size= 30
     model     = resnet50_half_conv(num_classes = 2)
     arch_name = 'ResNetShort'

ds_train     = Classification_DataSet(folder = 'dataset',split='train')
ds_val       = Classification_DataSet(folder = 'dataset',split='val')
optimizer    = torch.optim.Adam(params=model.parameters(), lr = 1e-3)
loss_fn      = torch.nn.CrossEntropyLoss(reduction='mean')
output_dir   = 'checkpoints'
task_name    = 'Classification'
stat_counter = classification_stat_counter()

ex = Experimet(task_name = task_name,
               arch_name = arch_name, 
               net=model, 
               epochs=epochs,
               train_set=ds_train, 
               val_set=ds_val,
               formatter=formatter_classification, 
               optimizer=optimizer,
               loss_fn=loss_fn,
               stat_counters = stat_counter, 
               output_dir=output_dir,
               img_size=size,
               batch_size=batch_size)
ex.run()