import os
import json
from torch.utils.data import Dataset


class WanJuanCCDataset(Dataset):
    """
    Overview:
        This is the dataset class for loading WanJuanCC. Link: https://opendatalab.com/OpenDataLab/WanJuanCC.
    """
    def __init__(self, data_dir, chunk_size=1000):
        """
        Overview:
            Initialize the dataset.
        Arguments:
            - data_dir: The root directory for downloaded extracted files, which should contain several jsonl files.
            - chunk_size: How many data samples to load for one time.
        """
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jsonl')]
        self.chunk_size = chunk_size
        self.current_file_idx = 0
        self.current_chunk = []
        self.current_line = 0
        self.file_handles = [open(file, 'r') for file in self.files]
        self._load_next_chunk()

    def _load_next_chunk(self):
        self.current_chunk = []
        while len(self.current_chunk) < self.chunk_size and self.current_file_idx < len(self.files):
            file_handle = self.file_handles[self.current_file_idx]
            line = file_handle.readline()
            if not line:
                self.current_file_idx += 1
                if self.current_file_idx < len(self.file_handles):
                    file_handle.close()
                continue
            self.current_chunk.append(json.loads(line))
        self.current_line = 0

    def __len__(self):
        # Estimate length
        return 100000000

    def __getitem__(self, idx):
        if self.current_line >= len(self.current_chunk):
            self._load_next_chunk()
        sample = self.current_chunk[self.current_line]
        self.current_line += 1
        return {'content': sample['content'], 'mask': [], 'source': 'WanJuanCC'}

    def __del__(self):
        for file_handle in self.file_handles:
            file_handle.close()
