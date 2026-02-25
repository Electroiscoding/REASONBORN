import torch
from torch.utils.data import Dataset, DataLoader

class PretrainingDataset(Dataset):
    def __init__(self, data_dir, seq_len):
        self.data_dir = data_dir
        self.seq_len = seq_len
        # Placeholder for real data loading logic
        self.data = [torch.randint(0, 50000, (seq_len,)) for _ in range(100)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids = self.data[idx]
        return {
            'input_ids': input_ids,
            'labels': input_ids.clone()
        }

def PretrainingDataLoader(data_dir, batch_size, seq_len):
    dataset = PretrainingDataset(data_dir, seq_len)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
