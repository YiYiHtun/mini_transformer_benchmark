import os
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


TOKEN_COLUMNS = [f"token_{i:02d}" for i in range(1, 21)]
MASK_COLUMNS = [f"mask_{i:02d}" for i in range(1, 21)]
VOCAB = {
    "PAD": 0,
    "A": 1,
    "B": 2,
    "C": 3,
    "D": 4,
}
ID2TOKEN = {v: k for k, v in VOCAB.items()}
MAX_LEN = 20
VOCAB_SIZE = 5


class SequenceClassificationDataset(Dataset):
    """Dataset for the assignment CSV format.

    Expected columns:
    - seq_len
    - label
    - token_01 ... token_20
    - mask_01 ... mask_20
    """

    def __init__(self, csv_path: str, verify_labels: bool = False):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self.df = pd.read_csv(csv_path)
        self._validate_columns()

        self.input_ids = torch.tensor(self.df[TOKEN_COLUMNS].values, dtype=torch.long)
        self.attention_mask = torch.tensor(self.df[MASK_COLUMNS].values, dtype=torch.float32)
        self.labels = torch.tensor(self.df["label"].values, dtype=torch.long)
        self.seq_lens = torch.tensor(self.df["seq_len"].values, dtype=torch.long)

        if verify_labels:
            self._verify_labels()

    def _validate_columns(self) -> None:
        required = {"seq_len", "label", *TOKEN_COLUMNS, *MASK_COLUMNS}
        missing = sorted(list(required - set(self.df.columns)))
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    @staticmethod
    def compute_label_from_tokens(tokens, mask) -> int:
        valid_tokens = [int(t) for t, m in zip(tokens, mask) if int(m) == 1]
        if len(valid_tokens) == 0:
            raise ValueError("Sequence has no valid tokens.")
        mid = len(valid_tokens) // 2
        first_token = valid_tokens[0]
        second_half = valid_tokens[mid:]
        return 1 if first_token in second_half else 0

    def _verify_labels(self) -> None:
        mismatches = []
        for idx in range(len(self.df)):
            tokens = self.input_ids[idx].tolist()
            mask = self.attention_mask[idx].tolist()
            expected = self.compute_label_from_tokens(tokens, mask)
            actual = int(self.labels[idx].item())
            if expected != actual:
                mismatches.append((idx, expected, actual))
                if len(mismatches) >= 5:
                    break
        if mismatches:
            raise ValueError(f"Found label mismatches. Example rows: {mismatches}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
            "seq_len": self.seq_lens[idx],
        }


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    batch_size: int = 32,
    num_workers: int = 0,
    verify_labels: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = SequenceClassificationDataset(train_csv, verify_labels=verify_labels)
    val_dataset = SequenceClassificationDataset(val_csv, verify_labels=verify_labels)
    test_dataset = SequenceClassificationDataset(test_csv, verify_labels=verify_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader
