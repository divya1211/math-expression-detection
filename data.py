import torch
from torch.utils.data.dataset import Dataset
from pathlib import Path


class MathExpressionsDataset(Dataset):
    def __init__(self, path="./data/train"):
        root = Path(path)
        self.questions = []
        for name in root.glob("*.txt"):
            f = open(name)
            for file in f:
                if file[-2] in ['0', '1']:
                    self.questions.append([file.strip()[:-1].strip(), int(file.strip()[-1])])
        
    def __getitem__(self, index):
        return self.questions[index]

    def __len__(self):
        return len(self.questions)

