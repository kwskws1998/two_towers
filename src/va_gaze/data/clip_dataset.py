import csv
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


os.environ["TOKENIZERS_PARALLELISM"] = "false"


class VADTextDataset(Dataset):
    def __init__(self, filename, checkpoint, maxlen, tokenizer=None):
        self.filename = filename
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(checkpoint)
        self.maxlen = maxlen

        df = pd.read_csv(
            filename,
            sep="\t",
            quotechar='"',
            engine="python",
            quoting=csv.QUOTE_NONE,
            escapechar="\\",
            keep_default_na=False,
            dtype={
                "index": np.int64,
                "text": str,
                "dataset_of_origin": str,
                "valence": np.float64,
                "arousal": np.float64,
            },
        )
        self.df = df.reset_index(drop=True)
        self.index = self.df["index"].to_list()
        self.texts = self.df["text"].to_list()
        self.valence = self.df["valence"].to_list()
        self.arousal = self.df["arousal"].to_list()

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            max_length=self.maxlen,
            truncation=True,
            padding=False,
        )
        return {
            "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "labels": torch.tensor([self.valence[idx], self.arousal[idx]], dtype=torch.float32),
        }

    def __len__(self):
        return len(self.texts)
