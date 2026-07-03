def convert_currency(

    customers,

    loans,

    rate=0.011

):

    customers = customers.copy()

    loans = loans.copy()

    customers["income"] *= rate

    monetary_columns = [

        "sanction_amount",

        "loan_amount",

        "processing_fee",

        "gst",

        "net_disbursement",

        "principal_outstanding",

        "bank_balance_at_application"

    ]

    for col in monetary_columns:

        loans[col] *= rate

    return customers, loans