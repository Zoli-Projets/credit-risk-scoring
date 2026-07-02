from pathlib import Path

PROCESSED = Path("data/processed")


def save_processed_data(df, filename):

    PROCESSED.mkdir(exist_ok=True)

    df.to_csv(
        PROCESSED / filename,
        index=False
    )