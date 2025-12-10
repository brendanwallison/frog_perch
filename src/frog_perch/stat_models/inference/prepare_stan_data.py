from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = {
    "prob",
    "day_index",
    "time_of_day_hours",
}

def prepare_stan_data(df: pd.DataFrame) -> dict:
    """
    Convert a tidy detector DataFrame into a Stan data dictionary
    for the call-intensity model.

    Removes any rows containing NaN in any required column and prints
    how many rows were removed.
    """

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"DataFrame is missing required columns: {sorted(missing)}"
        )

    # Work on a copy
    df = df.copy()

    # Ensure correct dtypes
    df["day_index"] = df["day_index"].astype(int)
    df["time_of_day_hours"] = df["time_of_day_hours"].astype(float)
    df["prob"] = df["prob"].astype(float)

    # ------------------------------------------------------------
    # Remove any rows with NaN in ANY column
    # ------------------------------------------------------------
    before = len(df)
    df = df.dropna(how="any")
    after = len(df)
    removed = before - after

    if removed > 0:
        print(f"[prepare_stan_data] Removed {removed} rows containing NaN values.")

    # ------------------------------------------------------------
    # Build Stan data dictionary
    # ------------------------------------------------------------
    stan_data = {
        "N": len(df),
        "D": int(df["day_index"].max()),
        "day": df["day_index"].tolist(),
        "t": df["time_of_day_hours"].tolist(),
        "p": df["prob"].tolist(),
    }

    return stan_data