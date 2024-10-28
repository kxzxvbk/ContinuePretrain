import random
import os.path
import argparse

import torch
from accelerate import Accelerator
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM

from data_utils import MaskedSFTDataset, mix_datasets, load_parquet_dataset, collate_fn
from utils import KLRegTrainer, EvaluateFirstStepCallback, preprocess_logits_for_metrics


def prepare_dataset(tokenizer, data_dir, train):
    train_data_cfg = [{"token_num": int(1e8),
                       "dataset": {"name": "WanJuanCC",
                                   "data_dir": "/data/share/OpenDataLab___WanJuanCC/extracted/jsonl"}},
                      {"token_num": int(2e8),
                       "dataset": {"name": "StarCoder",
                                   "data_dir": "/data/share/starcoderdata-python-edu"}},
                      {"token_num": int(1e8),
                       "dataset": {"name": "AlgebraicStack",
                                   "data_dir": "/data/share/algebraic-stack"}},
                      {"token_num": int(1e8),
                       "dataset": {"name": "Arxiver",
                                   "data_dir": "/data/share/arxiver"}},
                      ]
    if not os.path.exists(data_dir):
        mix_datasets(train_data_cfg, data_dir)
    train_data = load_parquet_dataset(data_dir)
    return MaskedSFTDataset(train_data, tokenizer, train=train)


if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--kl_weight', type=float, default=0)
    parser.add_argument('--train_dataset', type=str, default="./mixed_datasets/train_dataset.parquet")
    parser.add_argument('--test_num', type=int, default=2000)
    parser.add_argument('--out_dir', type=str, default="./default_output")
    parser.add_argument('--base_model', type=str, default="PATH-TO-MODEL")
    args = parser.parse_args()
    kl_weight = args.kl_weight

    # Initialize base model.
    device_index = Accelerator().process_index
    device_map = {"": device_index}
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, revision='main',
                                                 device_map=device_map, torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2")
    if kl_weight > 0:
        orig_model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, revision='main',
                                                          device_map=device_map, torch_dtype=torch.bfloat16,
                                                          attn_implementation="flash_attention_2")
        orig_model.eval()
    else:
        orig_model = None

    # Prepare dataset.
    train_dataset = prepare_dataset(tokenizer, data_dir=args.train_dataset, train=True)
    test_dataset = train_dataset[-args.eval_num:]
    train_dataset = train_dataset[:-args.eval_num]

    # Training arguments.
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=1,
        evaluation_strategy="steps",
        eval_steps=5000,
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=1,
        report_to='wandb',
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        group_by_length=False,
        dataloader_pin_memory=False,
        warmup_steps=5000,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        gradient_accumulation_steps=5,
        ddp_find_unused_parameters=False,
    )
    trainer = KLRegTrainer(
        kl_weight=kl_weight,
        orig_model=orig_model,
        model=model,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Resume from the checkpoint
    trainer.add_callback(EvaluateFirstStepCallback())
    trainer.train()
