import datasets
from torch.utils.data import Dataset


class AlpacaInstructDataset(Dataset):
    def __init__(self):
        self.data = datasets.load_dataset("iamtarun/code_instructions_120k_alpaca")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        res_dict = {
            'content': sample['prompt'],
            'mask': [[0, len(sample['prompt']) - len(sample['output'])]],
            'source': 'AlpacaInstruct'
        }
        return res_dict
