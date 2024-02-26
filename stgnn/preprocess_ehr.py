import argparse
import os
import pickle

import pandas as pd
from pandarallel import pandarallel

COLS_IRRELEVANT = [
    "subject_id",
    "hadm_id",
    "admittime",
    "dischtime",
    "gender",
    "age",
    "splits",
    "date",
    "node_name",
    "readmitted",
    "died"
]

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


def date_ranges_to_days(row):
    dt_range = pd.date_range(start=row.admittime.date(), end=row.dischtime.date())
    return list(enumerate(dt_range, 1))


def main(args):
    pandarallel.initialize(progress_bar=True, nb_workers=10)
    # read csv files
    demo_file = os.path.join(args.cohort_dir, "mimic_admission_demo.csv")
    df_demo = pd.read_csv(demo_file, low_memory=False, parse_dates=["admittime", "dischtime"])

    # TODO, remove this, for debugging only!
    # df_demo = df_demo.iloc[:300].copy()

    df_demo["node_name"] = df_demo["subject_id"].astype(str) + "_" + df_demo["hadm_id"].astype(str)
    df_demo = df_demo.loc[:, df_demo.columns.isin(COLS_IRRELEVANT)].copy()

    days_series = df_demo[["admittime", "dischtime"]].apply(date_ranges_to_days, axis=1).explode()
    days_df = pd.DataFrame(days_series.tolist(), columns=["Day_Number", "date"],
                           index=days_series.index)
    df_demo = df_demo.join(days_df, how="right")
    df_demo.gender = df_demo.gender.replace({"M": 0, "F": 1})
    print(f"df_demo shape: [{df_demo.shape[0]:,}x{df_demo.shape[1]:,}]")
    demo_cols = ["gender", "age"]

    # icd
    icd_file = os.path.join(args.cohort_dir, "mimic_hosp_icd_subgroups.csv")
    df_icd = pd.read_csv(icd_file, usecols=["hadm_id", "SUBGROUP"])
    df_icd = df_icd.groupby("hadm_id", sort=False).parallel_apply(
        lambda s: s.SUBGROUP.value_counts(sort=False).to_dict())
    df_icd = pd.DataFrame(df_icd.tolist(), index=df_icd.index)
    df_icd = df_icd.loc[:, ~df_icd.columns.isin(SUBGOUPRS_EXCLUDED)].copy()
    print(f"df_icd shape: [{df_icd.shape[0]:,}x{df_icd.shape[1]:,}]")
    icd_cols = df_icd.columns.tolist()

    def process_lab(_df):
        return (
            _df.groupby("label_fluid").flag.apply(lambda s: int((s == "abnormal").any())).to_dict()
        )

    # lab
    index_cols = ["hadm_id", "Day_Number"]
    lab_file = os.path.join(args.cohort_dir, "mimic_hosp_lab_filtered.csv")
    df_lab = pd.read_csv(lab_file, usecols=index_cols + ["label_fluid", "flag"])
    df_lab = df_lab.groupby(index_cols, sort=False).parallel_apply(process_lab)
    df_lab = pd.DataFrame(df_lab.tolist(), index=df_lab.index)
    print(f"df_lab shape: [{df_lab.shape[0]:,}x{df_lab.shape[1]:,}]")
    lab_cols = df_lab.columns.tolist()

    # medication
    index_cols = ["hadm_id", "Day_Number"]
    med_file = os.path.join(args.cohort_dir, "mimic_hosp_med_filtered.csv")
    df_med = pd.read_csv(med_file, usecols=index_cols + ["MED_THERAPEUTIC_CLASS_DESCRIPTION"])
    df_med = df_med.groupby(index_cols, sort=False).parallel_apply(
        lambda s: s.MED_THERAPEUTIC_CLASS_DESCRIPTION.value_counts(sort=False).to_dict()
    )
    df_med = pd.DataFrame(df_med.tolist(), index=df_med.index)
    print(f"df_med shape: [{df_med.shape[0]:,}x{df_med.shape[1]:,}]")
    med_cols = df_med.columns.tolist()

    # combine
    df_combined = (
        df_demo
        .join(df_icd, on="hadm_id", validate="m:1")
        .join(df_lab, on=["hadm_id", "Day_Number"], validate="1:1")
        .join(df_med, on=["hadm_id", "Day_Number"], validate="1:1")
    ).fillna(0)
    print(f"df_combined shape: [{df_combined.shape[0]:,}x{df_combined.shape[1]:,}]")
    df_combined.to_csv(os.path.join(args.save_dir, "ehr_combined.csv"), index=False)

    preproc_dict = {
        "X": df_combined,
        "feature_cols": {
            "demo_cols": demo_cols,
            "icd_cols": icd_cols,
            "lab_cols": lab_cols,
            "med_cols": med_cols,
        }
    }

    # also save it into sequences for temporal models
    feature_cols = [col for col_group in preproc_dict["feature_cols"].values() for col in col_group]
    feat_dict = {key: group.values for key, group in df_combined.groupby("node_name")[feature_cols]}

    with open(os.path.join(args.save_dir, "ehr_preprocessed_seq_by_day.pkl"), "wb") as pf:
        seq_dict = {
            "feat_dict": feat_dict,
            "feature_cols": feature_cols,
            "demo_cols": demo_cols,
            "icd_cols": icd_cols,
            "lab_cols": lab_cols,
            "med_cols": med_cols,
        }
        pickle.dump(seq_dict, pf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing EHR.")

    parser.add_argument(
        "--cohort_dir",
        type=str,
        default=None,
        help="Dir to filtered cohort demographics file.",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="Dir to save preprocessed files."
    )

    args = parser.parse_args()
    main(args)
