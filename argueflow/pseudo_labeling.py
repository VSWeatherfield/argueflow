from pathlib import Path

import pandas as pd


def load_pseudo_labeled_data(
    train_df: pd.DataFrame, fold: int, model_name: str, pl_dir: Path, label_cols: list
) -> pd.DataFrame:
    """
    Load and append pseudo-labeled data to the training dataframe.

    Args:
        train_df (pd.DataFrame): Original labeled training data.
        fold (int): Fold number (used in PL file naming).
        model_name (str): Name of the model used to generate pseudo-labels.
        pl_dir (Path): Directory where pseudo-labeled files are stored.
        label_cols (list): Columns containing soft labels (e.g., ['Ineffective', 'Adequate', 'Effective']).

    Returns:
        pd.DataFrame: Combined dataframe with pseudo-labeled rows appended.
    """
    pl_path = pl_dir / f"train_pl_{fold}_{model_name}.csv"
    pl_df = pd.read_csv(pl_path)

    pl_df["fold"] = -1
    pl_df["essay_id"] = pl_df["essay_id"].apply(lambda x: x.split("_")[0])
    pl_df = pl_df.set_index("essay_id")

    existing_ids = train_df["essay_id"].unique()
    new_pl_df = pl_df.loc[~pl_df.index.isin(existing_ids)].reset_index()

    # Retain only columns that exist in train_df (plus soft labels)
    pl_columns = train_df.columns.tolist()
    for col in label_cols:
        if col not in pl_columns:
            pl_columns.append(col)
    new_pl_df = new_pl_df[pl_columns]

    return pd.concat([train_df, new_pl_df], ignore_index=True)
