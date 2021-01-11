import numpy as np
import torch 

class classification_stat_counter:
    def __init__(self, num_tracking_metrics = 2):
        self.num_tracking_metrics = num_tracking_metrics
        self.reset()

    def reset(self):
        self.count = 0
        self.loss = 0
        self.running_accuracy = 0
        self.history = np.ndarray((0, self.num_tracking_metrics))

    def update(self, loss, annotations, results):
        #print(loss, annotations, results)
        loss_    = loss.item()
        
        _, predictions = torch.max(results, dim=1)
        correct_ = predictions.eq(annotations).sum().float()#.item()
        num = results.size(0)
        
        self.running_accuracy += (correct_/num)
        self.loss += loss
        self.count += 1
        
        loss = self.loss / self.count
        accuracy = self.running_accuracy / self.count  
        return loss, accuracy
        

    def summarize(self):
        loss = np.round((self.loss / self.count).cpu().detach().numpy(), 3).item()
        accuracy = np.round((self.running_accuracy / self.count).cpu().detach().numpy(), 3).item()
        
        self.history = np.append(self.history, np.array([[loss,accuracy]]),0)
        LOSS,ACCURACY = self.history[:,0].tolist(), self.history[:,1].tolist()
        return {'loss': LOSS, 'accuracy': ACCURACY}
    
class denoising_stat_counter:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.count = 0
        self.loss = 0
        self.running_mse = 0
        self.history = np.ndarray((0,2))
        
    def update(self, loss, clean, denoised):
        mse = torch.pow(clean-denoised,2) # mse = mse*mse
        mse = torch.sum(mse).float() / mse.numel()
        self.loss += loss
        self.running_mse  += mse
        self.count += 1
        loss = self.loss / self.count
        mse  = self.running_mse / self.count
        return loss, mse
        
    def summarize(self):
        loss = np.round((self.loss / self.count).cpu().detach().numpy(), 3).item()
        mse  = np.round((self.running_mse  / self.count).cpu().detach().numpy(), 3).item()
        self.history = np.append(self.history, np.array([[loss, mse]]),0)
        LOSS,MSE = self.history[:,0].tolist(), self.history[:,1].tolist()
        return {'loss': LOSS, 'mse': MSE}