import pandas as pd


def add_next_day_return(data: pd.DataFrame) -> pd.DataFrame:
    labelled_data = data.copy()

    labelled_data["target_next_day_return"] = (
        labelled_data["Close"].shift(-1) / labelled_data["Close"]
    ) - 1

    return labelled_data


def add_target_up_next_day(data: pd.DataFrame) -> pd.DataFrame:
    labelled_data = data.copy()

    if "target_next_day_return" not in labelled_data.columns:
        labelled_data = add_next_day_return(labelled_data)

    target_return = labelled_data["target_next_day_return"]

    labelled_data["target_up_next_day"] = (target_return > 0).astype("Int64")
    labelled_data.loc[target_return.isna(), "target_up_next_day"] = pd.NA

    return labelled_data


def create_price_labels(data: pd.DataFrame) -> pd.DataFrame:
    labelled_data = data.copy()

    labelled_data = add_next_day_return(labelled_data)
    labelled_data = add_target_up_next_day(labelled_data)

    return labelled_data
