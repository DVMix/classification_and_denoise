import torch
import math

class formatter_classification:
    def __init__(self, nts):
        self.nts = nts

    def format_it(self, tensor: torch.Tensor) -> torch.Tensor:
        n_times = math.floor(1/(tensor.shape[0]/self.nts))+1
        formatted_tensor = tensor.repeat(n_times,1)
        return formatted_tensor[:self.nts,:]

    def formatter(self, tuple_of_tensors):
        tensors, annotations = tuple_of_tensors
        original_t_sizes = [tensor.shape[0] for tensor in tensors]
        list_of_formatted_tensors = [self.format_it(tensor) for tensor in tensors]
        tensors = torch.stack(list_of_formatted_tensors).unsqueeze(1)
        if annotations != None:
            annotations = torch.tensor(annotations, dtype = torch.int64)
        return tensors, annotations, original_t_sizes
        
    def compress(self, x):
        return self.formatter(x)

    def __call__(self, x):
        return self.compress(x)
    
class formatter_denoise:
    def __init__(self, nts):
        self.nts = nts

    def format_it(self, tensor: torch.Tensor) -> torch.Tensor:
        n_times = math.floor(1/(tensor.shape[0]/self.nts))+1
        formatted_tensor = tensor.repeat(n_times,1)
        return formatted_tensor[:self.nts,:]

    def formatter(self, tuple_of_tensors: tuple) -> tuple:
        noisy, clean = tuple_of_tensors
        original_t_sizes = [tensor.shape[0] for tensor in noisy]
        
        if clean != None:
            list_of_formatted_tensors_clean = [self.format_it(tensor) for tensor in clean]
            clean = torch.stack(list_of_formatted_tensors_clean).unsqueeze(1)
        
        list_of_formatted_tensors_noisy = [self.format_it(tensor) for tensor in noisy]
        noisy = torch.stack(list_of_formatted_tensors_noisy).unsqueeze(1)
        return noisy, clean, original_t_sizes
    
    def compress(self, x):
        return self.formatter(x)
   
    def __call__(self, x):
        return self.compress(x)