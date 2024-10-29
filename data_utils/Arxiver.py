from torch.utils.data import Dataset
import datasets


class ArxiverDataset(Dataset):
    """
    Overview:
        This is the dataset class for loading Arxiver. Link: https://huggingface.co/datasets/neuralwork/arxiver.
    """

    def __init__(self, split: str = 'train', data_dir: str = None):
        assert split == 'train', "This dataset only support train split."
        if data_dir is not None:
            self.data = datasets.load_dataset(data_dir)['train']
        else:
            self.data = datasets.load_dataset("neuralwork/arxiver")['train']

    def __len__(self):
        return int(2e11)

    def __getitem__(self, index):
        item = self.data[index]
        res_dict = {
            'content': item['markdown'],
            'mask': [],
            'source': 'Arxiver'
        }
        return res_dict

