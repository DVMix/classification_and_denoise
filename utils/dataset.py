import os
import numpy as np

import torch

import librosa # for mel-spectrogram estimation
from librosa.feature import melspectrogram
import soundfile # for opening .flac audio
from matplotlib import pyplot as plt
import numpy as np

def sound2npy(path_to_file):
    audio, _ = soundfile.read(path_to_file)
    # normalized log-mel-spectrogram of clean and noisy audios
    #librosa.feature.
    audio_mel = 1 + np.log(1.e-12 + melspectrogram(audio, sr=16000, n_fft=1024, 
                                                   hop_length=256, fmin=20, fmax=8000, 
                                                   n_mels=80)).T / 10.
    return audio_mel

class Classification_DataSet(torch.utils.data.DataLoader):
    def __init__(self, 
                 folder  = '', # './dataset' 
                 split = 'train',
                 is_npy = True):
        assert split in ['train', 'val', 'test']
        
        self.is_npy = is_npy
        if self.is_npy:
            tmp = os.path.join(folder,split)
            rez_path = tmp if os.listdir(tmp) == ['clean', 'noisy'] else os.path.join(tmp, split)
            self.paths2files = list()
            for path, folders, files in os.walk(rez_path):
                if folders == []:
                    for file in files:
                        path2file = os.path.join(path, file)
                        self.paths2files.append(path2file)
        else:
            rez_path = os.path.join(folder,split)
            self.paths2files = list()
            for path, folders, files in os.walk(rez_path):
                if folders == []:
                    for file in files:
                        path2file = os.path.join(path, file)
                        self.paths2files.append(path2file)
            
    def __getitem__(self, idx):
        name = self.paths2files[idx]
        if self.is_npy:
            data = np.load(name)
            label = 0 if 'clean' in name else 1
        else:
            data = sound2npy(name)
            label = 0 if '.flac' in name else 1
        return torch.tensor(data, dtype = torch.float32), label
    
    def __len__(self):
        return len(self.paths2files)

    def collate_fn(self, batch):
        return tuple(zip(*batch))
    
    
class Denoising_DataSet(torch.utils.data.Dataset):
    def __init__(self, folder = '', split = 'train', is_npy = True):
        
        self.is_npy = is_npy
        if self.is_npy:
            tmp = os.path.join(folder,split)
            self.rez_path = tmp if os.listdir(tmp) == ['clean', 'noisy'] else os.path.join(tmp, split)

            self.clean_folder = os.path.join(self.rez_path, 'clean')
            self.noisy_folder = os.path.join(self.rez_path, 'noisy')

            self.names = list()
            for p, fold, files in os.walk(self.clean_folder):
                if fold ==[]:
                    self.names.extend(files)
        else:
            self.rez_path = os.path.join(folder,split)
            self.names = [
                os.path.join(self.rez_path, file.split('.')[0]) 
                for file in os.listdir(self.rez_path)  if '.flac' in file]
                
    def __getitem__(self, index):
        if self.is_npy:
            folder_id = self.names[index].split('_')[0]
            path2clean = os.path.join(self.clean_folder, folder_id, self.names[index])
            path2noisy = os.path.join(self.noisy_folder, folder_id, self.names[index])

            clean = torch.tensor(np.load(path2clean), dtype = torch.float32)
            noisy = torch.tensor(np.load(path2noisy), dtype = torch.float32)
        else:
            path2clean = self.names[index]+'.flac'
            path2noisy = self.names[index]+'_noisy.wav'
            
            clean = torch.tensor(sound2npy(path2clean), dtype = torch.float32)
            noisy = torch.tensor(sound2npy(path2noisy), dtype = torch.float32)
        return noisy, clean 
    
    def __len__(self):
        return len(self.names)
    
    def collate_fn(self, batch):
        return tuple(zip(*batch))