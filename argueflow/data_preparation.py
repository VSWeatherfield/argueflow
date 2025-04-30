from pathlib import Path

import pandas as pd
from tqdm import tqdm


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

TRAIN_CSV_PATH = RAW_DIR / "train.csv"
ESSAY_FOLDER = RAW_DIR / "train"

CLS_TOKEN = "[FP2]"


def load_data():
    """
    Loads and merges training metadata with full essay texts.

    Reads train.csv and corresponding essay .txt files,
    attaches full essay text to each discourse row,
    and maps effectiveness labels to integers.

    Returns:
        pd.DataFrame: Combined dataframe with discourse info, full text, and numeric labels.
    """
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    essay_texts = {
        path.stem: path.read_text(encoding="utf-8")
        for path in tqdm(ESSAY_FOLDER.glob("*.txt"), desc="Reading essays")
    }
    essay_df = pd.DataFrame(list(essay_texts.items()), columns=["essay_id", "text"])

    train_df = train_df.merge(essay_df, on="essay_id", how="left")

    label_map = {'Ineffective': 0, 'Adequate': 1, 'Effective': 2}
    train_df["label"] = train_df["discourse_effectiveness"].map(label_map)

    return train_df


def format_discourses(df):
    """
    Aggregates all discourse units in a single essay into one formatted string.

    Concatenates discourse texts prefixed by a custom [FP2] token,
    and stores associated metadata and labels for the essay.

    Args:
        df (pd.DataFrame): DataFrame containing all discourse rows for one essay.

    Returns:
        pd.Series: A single-row series with formatted text, labels, and metadata.
    """
    discourse_ids = "|".join(df["discourse_id"].tolist())
    discourse_type_str = "|".join(df["discourse_type"].tolist())
    labels = "|".join(df["discourse_effectiveness"].tolist())
    discourses = "".join(
        [
            f"{CLS_TOKEN}{row['discourse_type']}. {row['discourse_text']} "
            for _, row in df.iterrows()
        ]
    ).strip()

    return pd.Series(
        {
            "discourse_ids": discourse_ids,
            "discourse_type": discourse_type_str,
            "discourses": discourses,
            "label_list": labels,
            "text": df["text"].iloc[0],
        }
    )
