from pathlib import Path

import duckdb
import pandas as pd

RAW_INPUT = Path("data/raw/Alectra.csv")
UTF8_INPUT = Path("data/raw/Alectra_utf8.csv")

# Ensure the CSV is UTF-8 encoded for DuckDB ingestion
df = pd.read_csv(RAW_INPUT, encoding="Windows-1252")
df.to_csv(UTF8_INPUT, index=False, encoding="utf-8")

con = duckdb.connect("trial_balance.duckdb")

# Build staging table from the UTF-8 CSV
con.execute(
    """
    CREATE OR REPLACE TABLE staging_alectra AS
    SELECT
        Year::INTEGER                       AS fiscal_year,
        USoA_Account::INTEGER               AS account_id,
        Account_Description                 AS account_name,
        TRY_CAST(REPLACE(Control_Account_in_Dollars, ',', '') AS DECIMAL(18, 2)) AS control_amount,
        TRY_CAST(REPLACE("Sub-Account_in_Dollars", ',', '') AS DECIMAL(18, 2))   AS sub_amount
    FROM read_csv_auto('data/raw/Alectra_utf8.csv', header = true);
    """
)

# Build accounts dimension
con.execute(
    """
    CREATE OR REPLACE TABLE accounts (
        account_id INTEGER PRIMARY KEY,
        account_name VARCHAR,
        is_control BOOLEAN,
        parent_account_id INTEGER
    );
    """
)

con.execute(
    """
    INSERT INTO accounts
    SELECT
        account_id,
        account_name,
        control_amount IS NOT NULL AS is_control,
        NULL::INTEGER              AS parent_account_id
    FROM staging_alectra
    ON CONFLICT (account_id) DO UPDATE
    SET
        account_name = EXCLUDED.account_name,
        is_control   = EXCLUDED.is_control;
    """
)

# Build account_balances fact table
con.execute(
    """
    CREATE OR REPLACE TABLE account_balances (
        year INTEGER,
        account_id INTEGER,
        sub_num INTEGER,
        amount DECIMAL(18, 2)
    );
    """
)

# control balances (use sub_num = 0)
con.execute(
    """
    INSERT INTO account_balances
    SELECT fiscal_year, account_id, 0, control_amount
    FROM staging_alectra
    WHERE control_amount IS NOT NULL;
    """
)

# sub-account balances (use sub_num = 1)
con.execute(
    """
    INSERT INTO account_balances
    SELECT fiscal_year, account_id, 1, sub_amount
    FROM staging_alectra
    WHERE sub_amount IS NOT NULL;
    """
)

con.execute("DROP TABLE staging_alectra;")

con.close()

