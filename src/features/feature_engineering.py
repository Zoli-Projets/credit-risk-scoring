import numpy as np


class FeatureEngineer:

    def create_loan_to_income(self, df):

        df["loan_to_income"] = (
            df["loan_amount"] /
            df["income"]
        ).round(2)

        return df

    def create_delinquency_ratio(self, df):

        df["delinquency_ratio"] = (
            100
            * df["delinquent_months"]
            / df["total_loan_months"]
        ).round(1)

        return df

    def create_average_dpd(self, df):

        df["avg_dpd_per_delinquency"] = np.where(

            df["delinquent_months"] != 0,

            (
                df["total_dpd"]
                /
                df["delinquent_months"]
            ).round(1),

            0

        )

        return df

    def transform(self, df):

        df = self.create_loan_to_income(df)

        df = self.create_delinquency_ratio(df)

        df = self.create_average_dpd(df)

        return df