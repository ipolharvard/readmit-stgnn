import argparse
import copy
import os
import pickle
import sys

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

sys.path.append("../")
from constants import DEMO_COLS, LAB_COLS

COLS_IRRELEVANT = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "splits",
    "date",
    "node_name",
    "target"
]
CAT_COLUMNS = LAB_COLS + ["gender"]

SUBGOUPRS_EXCLUDED = [
    "Z00-Z13",
    "Z14-Z15",
    "Z16-Z16",
    "Z17-Z17",
    "Z18-Z18",
    "Z19-Z19",
    "Z20-Z29",
    "Z30-Z39",
    "Z40-Z53",
    "Z55-Z65",
    "Z66-Z66",
    "Z67-Z67",
    "Z68-Z68",
    "Z69-Z76",
    "Z77-Z99",
]


def preproc_ehr_cat_embedding(X):
    # encode categorical variables
    categorical_columns = []
    categorical_dims = {}
    for col in tqdm(X.columns):
        if col in COLS_IRRELEVANT:
            continue
        if col in CAT_COLUMNS:
            l_enc = LabelEncoder()
            print(col, X[col].unique())
            X[col] = l_enc.fit_transform(X[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)

    feature_cols = [col for col in X.columns if col not in COLS_IRRELEVANT]
    cat_idxs = [i for i, f in enumerate(feature_cols) if f in categorical_columns]
    cat_dims = [categorical_dims[f] for i, f in enumerate(feature_cols) if f in categorical_columns]

    return {
        "X": X,
        "feature_cols": feature_cols,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
    }


def preproc_ehr(X):
    """
    Args:
        X: pandas dataframe
    Returns:
        X_enc: pandas dataframe, with one-hot encoded columns for categorical variables
    """
    train_indices = X[X["splits"] == "train"].index

    # encode categorical variables
    X_enc = []
    num_cols = 0
    categorical_columns = []
    categorical_dims = {}
    for col in tqdm(X.columns):
        if col in COLS_IRRELEVANT:
            X_enc.append(X[col])
            num_cols += 1
        elif col in CAT_COLUMNS:
            print(col, X[col].unique())
            curr_enc = pd.get_dummies(
                X[col], prefix=col
            )  # this will transform into one-hot encoder
            X_enc.append(curr_enc)
            num_cols += curr_enc.shape[-1]
            categorical_columns.append(col)
            categorical_dims[col] = curr_enc.shape[-1]
        else:
            X.fillna(X.loc[train_indices, col].mean(), inplace=True)
            curr_enc = X[col]
            X_enc.append(curr_enc)
            num_cols += 1

    X_enc = pd.concat(X_enc, axis=1)
    assert num_cols == X_enc.shape[-1]

    feature_cols = [col for col in X_enc.columns if  (col not in COLS_IRRELEVANT)]
    cat_idxs = [i for i, f in enumerate(feature_cols) if f in categorical_columns]
    cat_dims = [
        categorical_dims[f]
        for _, f in enumerate(feature_cols)
        if f in categorical_columns
    ]

    return {
        "X": X_enc,
        "feature_cols": feature_cols,
        "cat_idxs": cat_idxs,
        "cat_dims": cat_dims,
    }


def ehr2sequence(preproc_dict, df_demo, by="day"):
    """
    Arrange EHR into sequences for temporal models
    """

    X = preproc_dict["X"]
    df = copy.deepcopy(X)
    feature_cols = preproc_dict["feature_cols"]

    print("Rearranging to sequences by {}...".format(by))

    feat_dict = {key: group.values for key, group in X.groupby("node_name")[feature_cols]}

    if "cat_idxs" in preproc_dict:
        cat_idxs = preproc_dict["cat_idxs"]
        cat_dims = preproc_dict["cat_dims"]
        return {
            "feat_dict": feat_dict,
            "feature_cols": feature_cols,
            "cat_idxs": cat_idxs,
            "cat_dims": cat_dims,
        }
    else:
        return {"feat_dict": feat_dict, "feature_cols": feature_cols}


def date_ranges_to_days(row):
    dt_range = pd.date_range(start=row.admittime.date(), end=row.dischtime.date())
    return list(enumerate(dt_range, 1))


def main(args):
    pandarallel.initialize(progress_bar=True, nb_workers=20)
    # read csv files
    df_demo = pd.read_csv(
        args.demo_file, dtype={k: str for k in CAT_COLUMNS}, low_memory=False,
        parse_dates=["admittime", "dischtime"]

    )

    # TODO, remove this, for debugging only!
    # df_demo = df_demo.iloc[:300].copy()

    df_demo.rename(columns={"readmitted_within_30days": "target"}, inplace=True)
    df_demo["node_name"] = df_demo["subject_id"].astype(str) + "_" + df_demo["hadm_id"].astype(str)
    df_demo = df_demo.loc[:, df_demo.columns.isin(COLS_IRRELEVANT)]

    days_series = df_demo[["admittime", "dischtime"]].apply(date_ranges_to_days, axis=1).explode()
    days_df = pd.DataFrame(days_series.tolist(), columns=["Day_Number", "date"],
                           index=days_series.index)
    df_demo = df_demo.join(days_df, how="right")

    # # icd
    df_icd = pd.read_csv(args.icd_file, low_memory=False)
    df_icd_count = (
        df_icd.groupby("hadm_id").SUBGROUP
        .value_counts()
        .unstack("SUBGROUP")
    )
    df_icd_count = df_icd_count.loc[:, ~df_icd_count.columns.isin(SUBGOUPRS_EXCLUDED)]
    print("Finished processing ICD, rows:", len(df_icd_count))

    # lab
    df_lab = pd.read_csv(args.lab_file, dtype={"flag": str}, low_memory=False)
    df_lab = df_lab.loc[df_lab.label_fluid.isin(LAB_COLS)]
    df_lab_onehot = (
        df_lab.groupby(["hadm_id", "Day_Number"])
        .parallel_apply(lambda _df: _df.groupby("label_fluid").flag.apply(lambda s: (s == "abnormal").any()))
    ).unstack("label_fluid").reset_index()
    print("Finished processing lab, rows:", len(df_lab_onehot))

    # medication
    df_med = pd.read_csv(args.med_file, low_memory=False)
    df_med_count = (
        df_med.groupby(["hadm_id", "Day_Number"]).MED_THERAPEUTIC_CLASS_DESCRIPTION
        .value_counts()
        .unstack("MED_THERAPEUTIC_CLASS_DESCRIPTION")
    ).reset_index()
    print("Finished processing medication, rows:", len(df_med_count))

    # combine
    df_combined = (
        df_demo
        .merge(df_icd_count, on="hadm_id", validate="m:1", how="left")
        .merge(df_lab_onehot, on=["hadm_id", "Day_Number"], validate="1:1", how="left")
        .merge(df_med_count, on=["hadm_id", "Day_Number"], validate="1:1", how="left")
    ).fillna(0)
    df_combined.to_csv(os.path.join(args.save_dir, "ehr_combined.csv"), index=False)

    for format in ["cat_embedding", "one_hot"]:
        if format == "cat_embedding":
            preproc_dict = preproc_ehr_cat_embedding(df_combined)
        else:
            preproc_dict = preproc_ehr(df_combined)

        feature_cols = preproc_dict["feature_cols"]
        demo_cols = [
            col for col in feature_cols if any([s for s in DEMO_COLS if s in col])
        ]
        icd_cols = [
            col for col in feature_cols if col in list(set(df_icd["SUBGROUP"].tolist()))
        ]
        lab_cols = [
            col for col in feature_cols if any([s for s in LAB_COLS if s in col])
        ]
        med_cols = [
            col
            for col in feature_cols
            if col in list(set(df_med["MED_THERAPEUTIC_CLASS_DESCRIPTION"].tolist()))
        ]

        preproc_dict["demo_cols"] = demo_cols
        preproc_dict["icd_cols"] = icd_cols
        preproc_dict["lab_cols"] = lab_cols
        preproc_dict["med_cols"] = med_cols

        # save
        with open(
                os.path.join(args.save_dir, "ehr_preprocessed_all_{}.pkl".format(format)),
                "wb",
        ) as pf:
            pickle.dump(preproc_dict, pf)
        print(
            "Saved to {}".format(
                os.path.join(
                    args.save_dir, "ehr_preprocessed_all_{}.pkl".format(format)
                )
            )
        )

        # also save it into sequences for temporal models
        seq_dict = ehr2sequence(preproc_dict, df_demo, by="day")

        seq_dict["demo_cols"] = demo_cols
        seq_dict["icd_cols"] = icd_cols
        seq_dict["lab_cols"] = lab_cols
        seq_dict["med_cols"] = med_cols
        with open(
                os.path.join(
                    args.save_dir, "ehr_preprocessed_seq_by_day_{}.pkl".format(format)
                ),
                "wb",
        ) as pf:
            pickle.dump(seq_dict, pf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing EHR.")

    parser.add_argument(
        "--demo_file",
        type=str,
        default=None,
        help="Dir to filtered cohort demographics file.",
    )
    parser.add_argument(
        "--icd_file",
        type=str,
        default=None,
        help="Dir to filtered cohort ICD-10 file.",
    )
    parser.add_argument(
        "--lab_file",
        type=str,
        default=None,
        help="Dir to filtered cohort lab file.",
    )
    parser.add_argument(
        "--med_file",
        type=str,
        default=None,
        help="Dir to filtered cohort medication file.",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Dir to save preprocessed files."
    )

    args = parser.parse_args()
    main(args)
