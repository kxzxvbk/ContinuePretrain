from torch.utils.data import Dataset
import datasets


class ArxiverDataset(Dataset):
    """
    Overview:
        This is the dataset class for loading Arxiver. Link: https://huggingface.co/datasets/neuralwork/arxiver.
    """

    def __init__(self, split: str, cache_dir: str = None):
        assert split == 'train', "This dataset only support train split."
        if cache_dir is not None:
            self.data = datasets.load_dataset(cache_dir, streaming=True)['train']
        else:
            self.data = datasets.load_dataset("neuralwork/arxiver", streaming=True)['train']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        res_dict = {
            'content': item['markdown'],
            'mask': [],
            'source': 'Arxiver'
        }
        return res_dict

