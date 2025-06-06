from pathlib import Path

import pandas as pd
from tqdm import tqdm


def load_data(cfg):
    """
    Loads and merges training metadata with full essay texts.

    Reads train.csv and corresponding essay .txt files,
    attaches full essay text to each discourse row,
    and maps effectiveness labels to integers.

    Returns:
        pd.DataFrame: Combined dataframe with discourse info, full text, and numeric labels.
    """
    train_csv_path = Path(cfg.raw_train_csv)
    essay_folder = Path(cfg.raw_essay_folder)

    train_df = pd.read_csv(train_csv_path)

    essay_texts = {
        path.stem: path.read_text(encoding="utf-8")
        for path in tqdm(essay_folder.glob("*.txt"), desc="Reading essays")
    }
    essay_df = pd.DataFrame(list(essay_texts.items()), columns=["essay_id", "text"])

    train_df = train_df.merge(essay_df, on="essay_id", how="left")

    label_map = {'Ineffective': 0, 'Adequate': 1, 'Effective': 2}
    train_df["label"] = train_df["discourse_effectiveness"].map(label_map)

    return train_df
