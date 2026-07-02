import pandas as pd


class DataPreprocessor:

    def __init__(self):
        pass

    def clean_missing_values(self, df):
        return df

    def remove_duplicates(self, df):
        return df.drop_duplicates()

    def remove_invalid_processing_fee(self, df):
        return df[df.processing_fee / df.loan_amount < 0.03]

    def correct_categories(self, df):

        df["loan_purpose"] = (
            df["loan_purpose"]
            .replace("Personaal", "Personal")
        )

        return df

    def transform(self, df):

        df = self.clean_missing_values(df)

        df = self.remove_duplicates(df)

        df = self.remove_invalid_processing_fee(df)

        df = self.correct_categories(df)

        return df