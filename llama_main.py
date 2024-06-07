import re
import random
import os.path
import argparse

import torch
import evaluate
from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM

from data_utils import MaskedSFTDataset, mix_datasets, load_parquet_dataset, collate_fn
from utils import KLRegTrainer, EvaluateFirstStepCallback, preprocess_logits_for_metrics


def _compute_doudizhu_metrics(preds, labels):
    pattern = r'(?:Therefore, I will finally play )(.*)'

    # Extract from pred results.
    pred_val = []
    for string in preds:
        out = re.findall(pattern, string)
        if len(out) == 0:
            pred_val.append(None)
        else:
            pred_val.append(out[0])

    # Extract from label.
    label_val = []
    for string in labels:
        out = re.findall(pattern, string)
        if len(out) == 0:
            label_val.append(None)
        else:
            label_val.append(out[0])

    # Calculate ACC
    res = 0
    for i in range(len(pred_val)):
        if pred_val[i] is None or label_val[i] is None or pred_val[i] != label_val[i]:
            continue
        res += 1
    return {'Final acc': res / len(pred_val)}


def compute_metrics(pred):
    rouge = evaluate.load('rouge')
    labels_ids = pred.label_ids[..., 1:]
    pred_ids = pred.predictions[0][..., :-1]
    for id, pred in enumerate(pred_ids):
        pred_ids[id][labels_ids[id] == -100] = 2
        pred_ids[id][pred_ids[id] == -100] = 2
        labels_ids[id][labels_ids[id] == -100] = 2

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    rouge_output = rouge.compute(predictions=pred_str, references=label_str,
                                 rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"])

    acc_count = 0
    for pred, label in zip(pred_str, label_str):
        if pred == label:
            acc_count += 1

    res_dict = {
        "R1": round(rouge_output["rouge1"], 4),
        "R2": round(rouge_output["rouge2"], 4),
        "RL": round(rouge_output["rougeL"], 4),
        "RLsum": round(rouge_output["rougeLsum"], 4),
        "acc": round(acc_count / len(label_str), 4)
    }

    # Compute doudizhu metrics.
    res_dict.update(_compute_doudizhu_metrics(pred_str, label_str))
    return res_dict


def prepare_dataset(tokenizer, train_data_dir, test_data_dir):
    train_data_cfg = [{"token_num": int(1e9),
                       "dataset": {"name": "WanJuanCC", "data_dir": "/data1/share/OpenDataLab___WanJuanCC"}},
                      {"token_num": int(1e9),
                       "dataset": {"name": "MasterMindDou", "split": "train"}},
                      {"token_num": int(1e8),
                       "dataset": {"name": "AlpacaInstruct"}},
                      ]
    test_data_cfg = [{"token_num": int(1e9),
                       "dataset": {"name": "MasterMindDou", "split": "test"}}
                      ]
    if not os.path.exists(train_data_dir):
        mix_datasets(train_data_cfg, train_data_dir)
    train_data = load_parquet_dataset(train_data_dir)
    if not os.path.exists(test_data_dir):
        mix_datasets(test_data_cfg, test_data_dir)
    test_data = load_parquet_dataset(test_data_dir)

    return MaskedSFTDataset(train_data, tokenizer), MaskedSFTDataset(test_data, tokenizer)


if __name__ == '__main__':
    random.seed(0)
    parser = argparse.ArgumentParser()
    parser.add_argument('--kl_weight', type=float, default=0.01)
    parser.add_argument('--train_dataset', type=str, default="./mixed_datasets/train_dataset.parquet")
    parser.add_argument('--test_dataset', type=str, default="./mixed_datasets/test_dataset.parquet")
    parser.add_argument('--out_dir', type=str, default="./llama_mastermind_dou")
    parser.add_argument('--base_model', type=str, default="/mnt/nfs/whl/LLM/llama-2-7b-hf")
    args = parser.parse_args()
    kl_weight = args.kl_weight

    # Initialize base model.
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, revision='main',
                                                 device_map='auto', torch_dtype=torch.bfloat16,
                                                 attn_implementation="flash_attention_2")
    if kl_weight > 0:
        orig_model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True, revision='main',
                                                  device_map={"": 0}, torch_dtype=torch.bfloat16,
                                                  attn_implementation="flash_attention_2")
        orig_model.eval()
    else:
        orig_model = None

    # Prepare dataset.
    train_dataset, test_dataset = prepare_dataset(tokenizer, train_data_dir=args.train_dataset,
                                                  test_data_dir=args.test_dataset)

    # Training arguments.
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        num_train_epochs=1,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=1,
        report_to=None,
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=2,
        group_by_length=False,
        dataloader_pin_memory=False,
        warmup_steps=3000,
        weight_decay=0.01,
        bf16=True,
        tf32=True,
        gradient_accumulation_steps=5
    )
    trainer = KLRegTrainer(
        kl_weight=kl_weight,
        orig_model=orig_model,
        model=model,
        data_collator=collate_fn,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        args=training_args,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    # Resume from the checkpoint
    trainer.add_callback(EvaluateFirstStepCallback())
    trainer.train()
