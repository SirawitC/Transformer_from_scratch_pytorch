import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split 

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path

def get_all_sentences(dataset, lang):
    for item in dataset:
        yield item["translation"][lang]

def get_or_guild_tokenizer(config, dataset, lang):
    # config["tokenizer_file"] = "../tokenizer_{}.json"
    tokenizer_path = Path(config["tokenizer_file"].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(dataset, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    dataset_row = load_dataset(f'{config["hf_dataset"]}', f'{config["lang_src"]-{config["lang_tgt"]}}', split="train")

    # build tokenizer
    tokenizer_src = get_or_guild_tokenizer(config, dataset_row, config["lang_src"])
    tokenizer_tgt = get_or_guild_tokenizer(config, dataset_row, config["lang_tgt"])

    # train test split 
    train_split_len = int(float(config["train_split_ratio"]) * len(dataset_row))
    test_split_len = len(dataset_row) - train_split_len

    train_split, test_split = random_split(dataset_row, [train_split_len, test_split_len])
