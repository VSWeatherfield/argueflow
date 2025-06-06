import pandas as pd


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
