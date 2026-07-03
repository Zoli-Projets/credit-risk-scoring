import pandas as pd


def merge_data(

    customers,

    loans,

    bureau

):

    df = customers.merge(

        loans,

        on="cust_id"

    )

    df = df.merge(

        bureau,

        on="cust_id"

    )

    return df