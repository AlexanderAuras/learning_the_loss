import typing

import torch

class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset: torch.utils.data.Dataset, transform: typing.Any = None, target_transform: typing.Any = None)->None:
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform
        
    def __getitem__(self, index: int) -> typing.Tuple[torch.tensor, torch.tensor]:
        x, y = self.dataset[index]
        if self.transform:
            x = self.transform(x)
        if self.target_transform:
            y = self.target_transform(y)
        return x, y
        
    def __len__(self) -> int:
        return len(self.dataset)