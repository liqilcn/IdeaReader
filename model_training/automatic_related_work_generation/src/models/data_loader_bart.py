import torch
from torch.utils.data import Dataset

class SurveyGenDataset(Dataset):
    def __init__(self, pt_file):
        self.dataset = torch.load(pt_file)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]['src'], self.dataset[index]['ref']