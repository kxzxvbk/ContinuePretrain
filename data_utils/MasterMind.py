import datasets
from torch.utils.data import Dataset


class MasterMindDouDataset(Dataset):
    def __init__(self, split: str, cache_dir: str = None):
        if cache_dir is not None:
            from data_utils.utils import load_parquet_dataset
            self.data = load_parquet_dataset(f'{cache_dir}/{split}_dataset.parquet')
        else:
            self.data = datasets.load_dataset("OpenDILabCommunity/MasterMind", "doudizhu", split=split)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        res_dict = {
            'content': item['sentence'] + item['answer'],
            'mask': [[0, len(item['sentence'])]],
            'source': 'MasterMindDou'
        }
        return res_dict
