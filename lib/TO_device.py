import torch

class TO_device():
    def __init__(self):
        self.device = None
    
    def get_default_device(self):
        """Pick GPU if available"""
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    @staticmethod
    def to_device(data, device):
        """move tensor to chosen device"""
        if isinstance(data, (list,tuple)):
            return [TO_device.to_device(x, device) for x in data]
        return data.to(device, non_blocking = True)


class DeviceDataLoader():
    """wrap a DataLoader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """yield a batch of data after moving it to device"""
        for b in self.dl:
            yield TO_device.to_device(b, self.device)
        
    def __len__(self):
        """number of batches"""
        return len(self.dl)


# test code
import time 
import numpy as np
from lib.data_transforms import Data

X1 = np.random.rand(100,2)
X2 = (np.random.rand(100,2) + 1.5)
X = np.concatenate( (X1, X2) )
Y1 = np.zeros( (100,1) )
Y2 = np.ones( (100,1) )
Y = np.concatenate( (Y1, Y2) )
train_batch_size = 20

def main():

    cnt = 0
    
    Data1 = Data(X,Y)
    Data1.data_randomize()
    Data1.data_spilt( 100, 25, 25 )
    Data1.to_torch()
    Data1.features_normal_db()
    print(
        "means of train set:", Data1.mean,
        "variances of train set:", Data1.var
    )
    DV = TO_device()
    DV.get_default_device()

    train_set_tensor = torch.utils.data.TensorDataset( Data1.train[:,0:2], Data1.train[:,2] )
    train_loader = torch.utils.data.DataLoader(train_set_tensor, batch_size=train_batch_size, shuffle=True)
    train_loader = DeviceDataLoader(train_loader, DV.device)
    for batch in train_loader:
        cnt = cnt + 1
        print(cnt)
    


if "__main__" == __name__:
    start_time = time.time()

    main()

    stop_time = time.time()

    print('Duration: {}'.format(time.strftime('%H:%M:%S', time.gmtime(stop_time - start_time))))

