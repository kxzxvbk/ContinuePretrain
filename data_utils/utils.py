import warnings
import random
from typing import Dict, List, Tuple

import torch
import numpy as np
import tiktoken
import pandas as pd
from torch.utils.data import Dataset

from .WanJuanCC import WanJuanCCDataset
from .MasterMind import MasterMindDouDataset
from .Alpaca import AlpacaInstructDataset
from .Arxiver import ArxiverDataset
from .StarCoder import StarCoderDataset
from .AlgebraicStack import AlgebraicStackDataset

name2dataset = {
    'WanJuanCC': WanJuanCCDataset,
    'MasterMindDou': MasterMindDouDataset,
    'AlpacaInstruct': AlpacaInstructDataset,
    'Arxiver': ArxiverDataset,
    'StarCoder': StarCoderDataset,
    'AlgebraicStack': AlgebraicStackDataset
}


def get_dataset(config: Dict) -> Dataset:
    """
    Overview:
        Return the target dataset given the description in diction form.
    Arguments:
        - config: A diction containing the description of target dataset. At least contains a key "name" referring the
            dataset name.
    Returns:
        - dataset: The constructed dataset.
    """
    dataset_name = config.pop('name')
    dataset_func = name2dataset[dataset_name]
    return dataset_func(**config)


def mix_datasets(datasets_config: List[Dict], target_dir: str) -> None:
    # Initialize each dataset object.
    datasets = [(get_dataset(datasets_config[i]['dataset']), datasets_config[i]['token_num'])
                for i in range(len(datasets_config))]

    # The tokenizer for calculating the number of tokens is ``cl100k_base``.
    encoding = tiktoken.get_encoding("cl100k_base")

    total_dataset = []
    for dataset, max_token_num in datasets:
        idx, token_num = 0, 0
        while token_num < max_token_num and idx < len(dataset):
            sample = dataset[idx]
            total_dataset.append(sample)
            token_num += len(encoding.encode(sample['content'], disallowed_special=()))
            idx += 1
        if token_num < max_token_num:
            warnings.warn(f"No enough tokens. Target: {max_token_num}, Exact: {token_num}")

    random.shuffle(total_dataset)
    df = pd.DataFrame(total_dataset)
    df.to_parquet(target_dir)


def load_parquet_dataset(dataset_path):
    df = pd.read_parquet(dataset_path)
    return df.to_dict('records')


class MaskedSFTDataset(Dataset):
    def __init__(self, data, tokenizer, train: bool, max_token=2048):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_token = max_token
        self.train = train

    def __len__(self):
        return len(self.data)

    def create_inputs_and_labels(self, content: str, mask: np.array, source: str) -> \
            Tuple[torch.Tensor, torch.Tensor, str]:
        # Slice the input content.
        mask = np.sort(mask, axis=0)
        begin_idx = 0
        slices = []
        for m in mask:
            if m[0] > begin_idx:
                slices.append([content[begin_idx: m[0]], -100])
            slices.append([content[m[0]: m[1]], 0])
            begin_idx = m[1]
        if begin_idx < len(content):
            slices.append([content[begin_idx:], 0])

        # Tokenize each slice of the original content.
        inputs = self.tokenizer.encode(slices[0][0], add_special_tokens=True)
        labels = inputs if slices[0][1] == 0 else [-100] * len(inputs)
        for sli in slices[1:]:
            new_inputs = self.tokenizer.encode(sli[0], add_special_tokens=False)
            inputs += new_inputs
            labels += new_inputs if sli[1] == 0 else [-100] * len(new_inputs)

        # Add [eop] token and convert into tensor.
        eop = self.tokenizer.eos_token_id
        inputs = torch.tensor(inputs[:self.max_token] + [eop])
        labels = torch.tensor(labels[:self.max_token] + [eop])

        return inputs, labels, source

    def __getitem__(self, index):
        item_data = self.data[index]
        input_ids, labels, source = self.create_inputs_and_labels(**item_data)
        if self.train:
            return {"input_ids": input_ids, "labels": labels, "source": source}
        else:
            return {"input_ids": input_ids, "labels": labels}


def collate_fn(batch):
    if 'source' not in batch[0]:
        return collate_single_fn(batch)
    dict_data = {}
    for i in range(len(batch)):
        sample = batch[i]
        if sample['source'] in dict_data:
            dict_data[sample['source']].append(sample)
        else:
            dict_data[sample['source']] = [sample]
    for source in dict_data.keys():
        dict_data[source] = collate_single_fn(dict_data[source])
    return dict_data


def collate_single_fn(batch):
    # Sort the batch in the descending order
    sorted_batch = sorted(batch, key=lambda x: x['input_ids'].shape[0], reverse=True)
    # Get each sequence and pad it
    sequences = [x['input_ids'] for x in sorted_batch]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    # Don't forget to grab the labels of the *sorted* batch
    labels = [x['labels'] for x in sorted_batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return {"input_ids": sequences_padded, "labels": labels_padded}
