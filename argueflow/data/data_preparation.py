import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from argueflow.utils.dvc_utils import download_data


log = logging.getLogger(__name__)


def read_and_merge_essays(cfg):
    """
    Reads and merges training metadata with full essay texts.

    Reads train.csv and corresponding essay.txt files,
    attaches full essay text to each discourse row,
    and maps effectiveness labels to integers.

    Returns:
        pd.DataFrame: Combined dataframe with discourse info, full text, and numeric labels.
    """
    train_csv_path = Path(cfg.data.raw_train_csv)
    essay_folder = Path(cfg.data.raw_essay_folder)

    if not train_csv_path.exists() or not essay_folder.exists():
        download_data(cfg)

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


def format_discourses(df, cls_token):
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
            f"{cls_token}{row['discourse_type']}. {row['discourse_text']} "
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


def prepare_data(cfg):
    """
    Prepares and saves the training dataset for model input.

    This function checks if the processed data file already exists.
    If not, it:
    - Loads raw training data and essay texts.
    - Formats discourse segments by inserting classification tokens.
    - Saves the processed and formatted data as a CSV.

    Args:
        cfg (DictConfig): Configuration object containing paths and model settings.
    """
    output_path = Path(cfg.data.processed_data_path)
    if output_path.exists():
        log.info(f"Processed file already exists at {output_path}. Skipping preparation.")
        return

    log.info("Loading raw data...")
    df_raw = read_and_merge_essays(cfg)

    log.info("Formatting discourses...")
    df_grouped = (
        df_raw.groupby("essay_id", group_keys=False)
        .apply(
            lambda df: format_discourses(df, cfg.model.cls_token), include_groups=False
        )
        .reset_index(drop=True)
    )

    output_path = Path(cfg.data.processed_data_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving preprocessed data to: {output_path}")
    df_grouped.to_csv(output_path, index=False)
