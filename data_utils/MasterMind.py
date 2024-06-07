import datasets
from torch.utils.data import Dataset


class MasterMindDouDataset(Dataset):
    def __init__(self, split: str):
        self.data = datasets.load_dataset("OpenDILabCommunity/MasterMind", "doudizhu", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        res_dict = {
            'content': item['question'] + item['sentence'],
            'mask': [[0, len(item['question'])]],
            'source': 'MasterMindDou'
        }
        return res_dict
