import numpy as np
import pandas as pd


def add_derived_personal_features(df):
    df = df.copy()
    df["AGE_YEARS"] = np.abs(df["DAYS_BIRTH"]) / 365
    df["EMPLOYED_YEARS"] = np.abs(df["DAYS_EMPLOYED"]) / 365
    df = df.drop(columns=["DAYS_BIRTH", "DAYS_EMPLOYED"])

    df.loc[df["CNT_CHILDREN"] > 10, "CNT_CHILDREN"] = np.nan
    df.loc[df["EMPLOYED_YEARS"] > 100, "EMPLOYED_YEARS"] = np.nan

    return df


def feature_on_current_data(application_df):
    df = application_df.copy()
    # finantial feature
    df["DEBT_BURDEN"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["PAYMENT_BURDEN"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].replace(0, np.nan)
    df["CREDIT_TO_AGE_RATIO"] = df["AMT_CREDIT"] / df["AGE_YEARS"].replace(0, np.nan)
    # most related features
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCE_MIN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(axis=1)
    df["EXT_SOURCE_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["EXT_SOURCE_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCE_AGE"] = df["EXT_SOURCE_MEAN"] * df["AGE_YEARS"]
    df["EXT_SOURCE_EMPLOYED"] = df["EXT_SOURCE_MEAN"] * df["EMPLOYED_YEARS"]
    # additional feature (personal info)
    df["EMPLOYED_TO_AGE_RATIO"] = df["EMPLOYED_YEARS"] / df["AGE_YEARS"].replace(0, np.nan)

    return df


def feature_on_bureau(bureau_df, bureau_balance_df):
    # this tell us the active credit
    active_bureau = bureau_df[bureau_df["CREDIT_ACTIVE"] == "Active"]
    # this tell us the debt to credit ratio per customer for active
    debt_to_credit_ratio_bureau = (
        active_bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM_DEBT"].sum()
        / active_bureau.groupby("SK_ID_CURR")["AMT_CREDIT_SUM"].sum().replace(0, np.nan)
    )
    # this tell us the maximum days the customer delay
    max_overdue_bureau = bureau_df.groupby("SK_ID_CURR")["CREDIT_DAY_OVERDUE"].max()
    ### bureau_balance ##############
    # Modify the status column
    bureau_balance_df["STATUS_MOD"] = (
        pd.to_numeric(bureau_balance_df["STATUS"], errors="coerce").fillna(0).astype(int)
    )
    # get for each SK_ID_CURR the worst status
    worst_status_curr = (
        bureau_balance_df.groupby("SK_ID_BUREAU")["STATUS_MOD"]
        .max()
        .reset_index()
        .merge(bureau_df[["SK_ID_BUREAU", "SK_ID_CURR"]], on="SK_ID_BUREAU")
        .groupby("SK_ID_CURR")["STATUS_MOD"]
        .max()
        .rename("WORST_STATUS")
    )
    ### rename fueature to fetch then into train and test data set
    debt_to_credit = debt_to_credit_ratio_bureau.rename("DEBT_TO_CREDIT_RATIO").reset_index()
    max_overdue = max_overdue_bureau.rename("MAX_OVERDUE_DAYS").reset_index()
    worst_status = worst_status_curr.rename("WORST_STATUS").reset_index()
    # merge bureau balance features into bureau
    bureau_features = debt_to_credit.merge(max_overdue, on="SK_ID_CURR", how="outer").merge(
        worst_status, on="SK_ID_CURR", how="outer"
    )
    return bureau_features


def feature_on_prev_home_credit(previous_df, pos_df, cc_df, inst_df):
    ## get the previous approval rate
    previous_df["IS_APPROVED"] = (previous_df["NAME_CONTRACT_STATUS"] == "Approved").astype(int)

    previous_info = previous_df.groupby("SK_ID_CURR").agg(
        PREV_COUNT=("SK_ID_PREV", "count"),
        PREV_APPROVED_COUNT=("IS_APPROVED", "sum"),
    )

    previous_info["PREV_APPROVAL_RATE"] = (
        previous_info["PREV_APPROVED_COUNT"] / previous_info["PREV_COUNT"]
    )

    #### POS_CASH_balance ###########

    pos_df["IS_BAD_MONTH"] = (pos_df["SK_DPD"] > 0).astype(int)

    pos_features = pos_df.groupby("SK_ID_CURR").agg(
        POS_MONTHS_COUNT=("MONTHS_BALANCE", "count"),
        POS_MAX_DPD=("SK_DPD", "max"),
        POS_BAD_MONTHS_COUNT=("IS_BAD_MONTH", "sum"),
    ).reset_index()

    #### Credit_card_balance ###########

    cc_df["UTILITY"] = cc_df["AMT_BALANCE"] / cc_df["AMT_CREDIT_LIMIT_ACTUAL"].replace(0, np.nan)

    cc_features = cc_df.groupby("SK_ID_CURR").agg(
        CC_UTILITY_MEAN=("UTILITY", "mean"),
        CC_UTILITY_MAX=("UTILITY", "max"),
    ).reset_index()

    ####### Installment_balance ###########

    inst_df["DAYS_LATE"] = inst_df["DAYS_ENTRY_PAYMENT"] - inst_df["DAYS_INSTALMENT"]
    inst_df["PAYMENT_RATIO"] = inst_df["AMT_PAYMENT"] / inst_df["AMT_INSTALMENT"].replace(0, np.nan)
    inst_features = inst_df.groupby("SK_ID_CURR").agg(
        INST_MAX_DAYS_LATE=("DAYS_LATE", "max"),
        INST_MEAN_DAYS_LATE=("DAYS_LATE", "mean"),
        INST_PAYMENT_RATIO=("PAYMENT_RATIO", "mean"),
    ).reset_index()

    # merge previous balances features into previous info
    prev_all = (
        previous_info.merge(pos_features, on="SK_ID_CURR", how="left")
        .merge(cc_features, on="SK_ID_CURR", how="left")
        .merge(inst_features, on="SK_ID_CURR", how="left")
    )
    return prev_all
