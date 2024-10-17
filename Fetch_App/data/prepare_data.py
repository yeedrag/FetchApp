import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class TimeSeriesDataset(Dataset):
    '''
        A simple implementation of a Time Series Dataset.
    '''
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length
    def __len__(self):
        return len(self.data) - self.seq_length 
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + 1: idx + self.seq_length + 1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), \
            torch.tensor(idx, dtype=torch.float32)
    def getall(self):
        return torch.tensor(self.data, dtype=torch.float32)
    
def PrepareDataset(path, seq_length=60, batch_size=30):
    '''
        Prepares the dataloader given a path.
        Args:
            path (str): The path of the dataset
            seq_length (int): How many days are used to predict the next one
            batch_size (int): The batch size of the data
        Returns:
            train_loader (DataLoader): A DataLoader object for the dataset.
            scaled_data (Tensor): The scaled data
    '''
    data = pd.read_csv(path) 
    # Parse date as index
    data['# Date'] = pd.to_datetime(data['# Date'])
    data.set_index('# Date', inplace=True)
    X = data['Receipt_Count'].values.reshape(-1, 1)
    # Normalize with min-max scaling
    X_std = (data - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    scaled_data = X_std.to_numpy()
    dataset = TimeSeriesDataset(scaled_data, seq_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return train_loader, torch.tensor(scaled_data, dtype=torch.float32), [X.max(axis=0), X.min(axis=0)]
