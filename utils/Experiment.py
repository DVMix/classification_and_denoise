import numpy as np
import os
import json
import torch 

from utils.progress_bar import toolbar_width, pr_bar
from utils.dataset import sound2npy
class Experimet(object):
    def __init__(self, task_name, arch_name, net, epochs, train_set, val_set, formatter, optimizer, 
                 loss_fn, stat_counters, output_dir='checkpoints', img_size=320, batch_size = 8):
        
        self.Metric_names = {
            'Classification': 'Accuracy',
            'Denoise'       : 'MSE'
        }
        
        assert task_name in self.Metric_names.keys(), '[!] Check task_name!'
        
        self.arch_name = arch_name
        self.task_name = task_name
        self.Metric_name = self.Metric_names[self.task_name]
        self.epochs = epochs
        self.img_size = img_size
        self.stat_counters = stat_counters
        self.formatter = formatter(img_size)
        self.DL_train = torch.utils.data.DataLoader(dataset = train_set, batch_size = batch_size, 
                                               shuffle = True,      collate_fn = train_set.collate_fn)
        self.DL_valid = torch.utils.data.DataLoader(dataset = val_set,   batch_size = batch_size, 
                                               shuffle = True,      collate_fn = val_set.collate_fn)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = net
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.output_dir = output_dir
        self.init_model()
        
    def save_model_weights_and_stats(self, name, stats):
        model_folder = os.path.join(self.output_dir, self.task_name, self.arch_name, str(self.img_size))
        os.makedirs(model_folder, exist_ok=True)
        with open(os.path.join(model_folder,'checkpoint.txt'), 'w') as file:
            file.write(name)
        with open(os.path.join(model_folder,'metrics.json'), 'w') as fp:
            json.dump(stats, fp)
        full_path = os.path.join(model_folder, name)
        torch.save(self.model.state_dict(), full_path)
        
    def load_model_weights(self):
        try:
            model_folder = os.path.join(self.output_dir, self.task_name, self.arch_name, str(self.img_size))
            with open(os.path.join(model_folder, 'checkpoint.txt'), 'r') as file:
                data = file.read()

            with open(os.path.join(model_folder,'metrics.json'), 'r') as fp:
                stats = json.load(fp)

            metric = 'accuracy' if self.task_name=='Classification' else 'mse'
            self.stat_counters.history = np.array([stats['loss'],stats[metric]]).T
            
            path2checkpoint = os.path.join(model_folder, data)
            self.start_from = int(data.split('_')[3])+1
            state_dict = torch.load(path2checkpoint)
            self.model.load_state_dict(state_dict)
            print('Weights loaded.')
        except:
            print('No avalible checkpoint for current net arch.')
            self.start_from = 0
    
    def reduce_lr(self, coefficient):
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] * coefficient
    
    def init_model(self):
        model_folder = os.path.join(self.output_dir, self.task_name, self.arch_name, str(self.img_size))
        print(model_folder)
        self.load_model_weights()
        self.model = self.model.to(self.device)
        
    def run(self):
        print('Model prepared.')
        self.model.train()
        for epoch in range(self.start_from, self.epochs):
            self.reduce_lr(0.9)
            for i, data in enumerate(self.DL_train):
                data, anno, _ = self.formatter(data)
                data, anno= data.to(self.device), anno.to(self.device)
                
                self.optimizer.zero_grad()
                result = self.model(data)

                loss = self.loss_fn(result, anno)
                loss.backward()
                self.optimizer.step()
                
                loss, metric = self.stat_counters.update(loss.cpu().detach(), 
                                                         anno.cpu().detach(), 
                                                         result.cpu().detach())
                additional_info = {'Loss': loss, self.Metric_name: metric}
                pr_bar(epoch, i, self.DL_train.__len__(), toolbar_width, additional_info)
                    
            converted_metric = str(round(loss.item(),4)).replace('.','_')
            name = 'Task_{}_Epoch_{}_{}_{}'.format(self.task_name, 
                                                        epoch,
                                                        self.Metric_name, 
                                                        converted_metric +'.pth')
            stats = self.stat_counters.summarize()
            self.save_model_weights_and_stats(name, stats)

    def validate(self):
        self.model.eval()
        target = 0
        for data in tqdm(self.DL_valid):
            data, anno,  _ = self.formatter(data)
            data, anno  = data.to(self.device), anno.to(self.device)
            with torch.no_grad():
                result = self.model(data)
            if self.task_name == 'Classification':
                _, predictions = torch.max(result, dim=1)
                correct_ = predictions.eq(anno).sum().float()#.item()
                target += correct_
            else:
                target += self.loss_fn(result, anno).item() 
                
        rez = target/self.DL_valid.__len__()
        print("{} = {}".format(self.Metric_names[self.task_name], rez))

    def infer(self, mel_spectrogramm, annotation = None):
        try:
            spectrogramm = torch.tensor(mel_spectrogramm, dtype = torch.float32, device = self.device).unsqueeze(0)
            if self.task_name == 'Denoise':
                annotation = torch.tensor(annotation, dtype = torch.float32, device = self.device).unsqueeze(0)
        except:
            spectrogramm = torch.tensor(sound2npy(mel_spectrogramm), dtype = torch.float32, device = self.device).unsqueeze(0)
            if self.task_name == 'Denoise':
                annotation = torch.tensor(sound2npy(annotation), dtype = torch.float32, device = self.device).unsqueeze(0)

        data = (spectrogramm, annotation)
        data, anno, _ = self.formatter(data)
        data = data.to(self.device)
        if anno != None:
            anno = anno.to(self.device)
        
        with torch.no_grad():
            result = self.model(data)
        if self.task_name == 'Classification':
            _, predictions = torch.max(result, dim=1)
            
            if predictions.cpu().item() == 0:
                print('Clean spectrogram')
            else:
                print('Noisy spectrogram')

            if anno != None: 
                if predictions == anno:
                    print('Right prediction.')
                else:
                    print('Wrong prediction.')
        else:
            if anno != None:
                mse = self.loss_fn(result, anno).item() 
                print('Mean Squared Error between clean and denoised spectrograms equals {}'.format(mse))                



            