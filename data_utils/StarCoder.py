import os
from data_utils.utils import load_parquet_dataset
from torch.utils.data import Dataset


class StarCoderDataset(Dataset):
    """
    Overview:
        This is the dataset class for loading StarCoder.
        Link: https://huggingface.co/datasets/jon-tow/starcoderdata-python-edu.
    """
    def __init__(self, data_dir, chunk_size=1000):
        """
        Overview:
            Initialize the dataset.
        Arguments:
            - data_dir: The root directory for downloaded extracted files, which should contain several jsonl files.
            - chunk_size: How many data samples to load for one time.
        """
        self.data_dir = os.path.join(data_dir, 'data')
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
        self.chunk_size = chunk_size
        self.par_idx = 0
        self.current_file = load_parquet_dataset(self.files[self.par_idx])
        self.current_file_idx = 0

        self.current_chunk = []
        self.current_line = 0
        self._load_next_chunk()

    def _load_next_chunk(self):
        self.current_chunk = []
        while len(self.current_chunk) < self.chunk_size and self.current_file is not None:
            self.current_chunk.append(self.current_file[self.current_file_idx])
            self.current_file_idx += 1
            if self.current_file_idx > len(self.current_file):
                self.par_idx += 1
                if self.par_idx >= len(self.files):
                    self.current_file = None
                else:
                    self.current_file = load_parquet_dataset(self.files[self.par_idx])
                    self.current_file_idx = 0
        self.current_line = 0

    def __len__(self):
        # Estimate length
        return int(2e11)

    def __getitem__(self, idx):
        if self.current_line >= len(self.current_chunk):
            self._load_next_chunk()
        sample = self.current_chunk[self.current_line]
        self.current_line += 1
        return {'content': sample['content'], 'mask': [], 'source': 'StarCoder'}
