from torch.utils.data import Dataset

class mydataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        data = self.data[index]
        label = self.labels[index]
        return index, data, label