import argparse
import os
import sys

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()

sys.path.append("../")
from constants import LAB_COLS

ADMISSION_TYPES_EXCLUDED = [
    "AMBULATORY OBSERVATION",
    "EU OBSERVATION",
    "DIRECT OBSERVATION",
]
DISCHARGE_LOCATION_EXCLUDED = [
    "ACUTE HOSPITAL",
    "HEALTHCARE FACILITY",
    # "SKILLED NURSING FACILITY",
    "AGAINST ADVICE",
]


def main(args):
    pandarallel.initialize(progress_bar=True, nb_workers=20)

    MIMIC_HOSP_DIR = os.path.join(args.raw_data_dir, "hosp")

    df_patients = pd.read_csv(os.path.join(MIMIC_HOSP_DIR, "patients.csv.gz"),
                              parse_dates=["anchor_year", "dod"])
    df_patients.dod += pd.Timedelta(hours=23, minutes=59)
    df_admission = pd.read_csv(os.path.join(MIMIC_HOSP_DIR, "admissions.csv.gz"),
                               parse_dates=["admittime", "dischtime", "deathtime"])

    ### Get readmission info
    print("Getting readmission information...")

    def process_readmission_info(_df: pd.DataFrame):
        df = _df.sort_values(by="admittime")
        admit_times = pd.Series(df.admittime[1:].tolist() + [df.dod.iloc[0]], index=df.index)
        readmit_df = (admit_times - df.dischtime.values).dt.days.to_frame("readmission_gap_in_days")
        readmit_df.loc[readmit_df.readmission_gap_in_days < 0] = 0
        readmit_df["readmitted_within_30days"] = readmit_df.readmission_gap_in_days < 30
        hadm_ids = pd.Series(df.hadm_id[1:].tolist() + [np.nan], index=df.index)
        readmit_df["readmission_id"] = hadm_ids.where(readmit_df.readmitted_within_30days)
        return readmit_df

    df_gb = df_admission.merge(df_patients[["subject_id", "dod"]], on="subject_id").groupby(
        "subject_id", group_keys=False)
    readmission_info_df = df_gb.parallel_apply(process_readmission_info)
    df_admission = df_admission.join(readmission_info_df)
    # drop admissions where a patient died
    df_admission = df_admission.loc[df_admission.deathtime.isna()]

    df_admission.to_csv(
        os.path.join(
            args.save_dir, "admission_48hr_discharge_cxr_filtered_labeled.csv"
        ),
        index=False,
    )

    ### Split patients into train/val/test
    test_patients = pd.read_csv("../data/mimic_our_test_cohort.csv").subject_id
    train_val_patients = list(set(df_admission["subject_id"]).difference(test_patients))
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.1, random_state=12
    )
    df_train = df_admission[df_admission["subject_id"].isin(train_patients)]
    df_val = df_admission[df_admission["subject_id"].isin(val_patients)]
    df_test = df_admission[df_admission["subject_id"].isin(test_patients)]

    print("Train pos ratio:", df_train["readmitted_within_30days"].sum() / len(df_train))
    print("Val pos ratio:", df_val["readmitted_within_30days"].sum() / len(df_val))
    print("Test pos ratio:", df_test["readmitted_within_30days"].sum() / len(df_test))

    def assign_split(id):
        if id in train_patients:
            return "train"
        elif id in val_patients:
            return "val"
        else:
            return "test"

    df_admission["splits"] = df_admission["subject_id"].parallel_apply(assign_split)

    df_admission.to_csv(
        os.path.join(
            args.save_dir, "admission_48hr_discharge_cxr_filtered_w_splits.csv"
        ),
        index=False,
    )

    ### Get demographics
    print("Getting age, gender, splits...")

    df_patients.set_index("subject_id", inplace=True)

    df_patients["year_of_birth"] = df_patients.anchor_year.dt.year - df_patients.anchor_age

    df_admission["gender"] = df_admission.subject_id.map(df_patients.gender)
    df_admission["age"] = df_admission.admittime.dt.year - df_patients.loc[
        df_admission.subject_id].year_of_birth.values

    df_admission.to_csv(
        os.path.join(args.save_dir, "mimic_admission_demo.csv"), index=False
    )
    print("Admission basic information saved...")

    ### Medication
    df_prescriptions = pd.read_csv(os.path.join(MIMIC_HOSP_DIR, "prescriptions.csv.gz"),
                                   parse_dates=["starttime"], low_memory=False)

    df_prescriptions = df_prescriptions.loc[df_prescriptions.hadm_id.isin(df_admission.hadm_id)]
    df_prescriptions = df_prescriptions.loc[df_prescriptions.starttime.notnull()].copy()

    ## Map NDC to therapeutic classes
    df_med_map = pd.read_csv("../data/ndc2therapeutic.csv").dropna()
    df_med_map = df_med_map.groupby(
        "NDC_MEDICATION_CODE").MED_THERAPEUTIC_CLASS_DESCRIPTION.first().to_dict()

    print("Mapping NDC to therapeutic classes...")
    df_prescriptions["MED_THERAPEUTIC_CLASS_DESCRIPTION"] = df_prescriptions.ndc.map(
        df_med_map)
    df_prescriptions.dropna(subset=["MED_THERAPEUTIC_CLASS_DESCRIPTION"], inplace=True)

    df_day_num = df_prescriptions[["subject_id", "hadm_id", "starttime"]].merge(
        df_admission[["hadm_id", "admittime", "dischtime"]], on="hadm_id", validate="m:1")
    day_num = (df_day_num.starttime - df_day_num.admittime).dt.days + 1
    day_num.where(df_day_num.starttime > df_day_num.admittime, inplace=True)
    day_num.where(df_day_num.starttime < df_day_num.dischtime, inplace=True)

    df_prescriptions["Day_Number"] = day_num.values
    df_prescriptions.dropna(subset=["Day_Number"], inplace=True)

    df_prescriptions.to_csv(
        os.path.join(args.save_dir, "mimic_hosp_med_filtered.csv"), index=False
    )

    ### ICD-10
    df_diag = pd.read_csv(os.path.join(MIMIC_HOSP_DIR, "diagnoses_icd.csv.gz"))
    df_diag = df_diag.loc[df_diag.icd_version == 10].drop("icd_version", axis=1)
    df_diag = df_diag[df_diag.hadm_id.isin(df_admission.hadm_id)]
    df_diag.icd_code = df_diag.icd_code.str.replace(".", "").str[:3]

    df_icd = pd.read_csv("../data/ICD10_Groups.csv")
    # convert df_icd to a reasonable mapping
    icd2group = {
        (letter, f"{i:02}"): subgroup
        for letter, start_idx, end_idx, subgroup
        in df_icd[["LETTER", "START_IDX", "END_IDX", "SUBGROUP"]].itertuples(index=False)
        if start_idx.isnumeric() and end_idx.isnumeric()
        for i in range(int(start_idx), int(end_idx) + 1)
    }
    ## add problematic mappings manually
    icd2group.update(
        {
            ("C", "7A"): "C7A-C7A",
            ("C", "7B"): "C7B-C7B",
            ("D", "3A"): "D3A-D3A",
            ("O", "94"): "O94-O9A",
            ("O", "98"): "O94-O9A",
            ("O", "99"): "O94-O9A",
            ("O", "9A"): "O94-O9A",
        }
    )

    print("Mapping ICD-10 code to subgroups...")

    df_diag["SUBGROUP"] = df_diag.icd_code.progress_apply(
        lambda icd: icd2group.get((icd[0], icd[1:]), np.nan)
    )
    df_diag.dropna(subset=["SUBGROUP"], inplace=True)
    df_diag.to_csv(
        os.path.join(args.save_dir, "mimic_hosp_icd_subgroups.csv"), index=False
    )

    ### Get labs
    df_lab_item = pd.read_csv(os.path.join(MIMIC_HOSP_DIR, "d_labitems.csv.gz"))
    df_lab_item["label_fluid"] = df_lab_item.label + " " + df_lab_item.fluid
    mask_lab_that_we_consider = df_lab_item.label_fluid.isin(LAB_COLS)
    df_lab_item_filtered = df_lab_item.loc[mask_lab_that_we_consider]

    ## labevents file is big, process using chunks
    print("Reading lab events...")
    df_lab_filtered = pd.read_csv(os.path.join(MIMIC_HOSP_DIR, "labevents.csv.gz"),
                                  parse_dates=["charttime"])
    df_lab_filtered = df_lab_filtered[df_lab_filtered.hadm_id.isin(df_admission.hadm_id)]
    df_lab_filtered = df_lab_filtered.merge(df_lab_item_filtered, on="itemid", how="inner")

    ## Add lab name by label + fluids, and day number
    print("Getting lab information...")
    df_day_num = df_lab_filtered[["hadm_id", "charttime"]].merge(
        df_admission[["hadm_id", "admittime", "dischtime"]], on="hadm_id")
    day_num = (df_day_num.charttime - df_day_num.admittime).dt.days + 1
    day_num.where(df_day_num.charttime > df_day_num.admittime, inplace=True)
    day_num.where(df_day_num.charttime < df_day_num.dischtime, inplace=True)

    df_lab_filtered["Day_Number"] = day_num.values
    df_lab_filtered.dropna(subset=["Day_Number"], inplace=True)
    df_lab_filtered.to_csv(
        os.path.join(args.save_dir, "mimic_hosp_lab_filtered.csv"), index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtering admission info from MIMIC-IV.")
    parser.add_argument(
        "--raw_data_dir", type=str, help="Dir to downloaded MIMIC-IV data."
    )
    parser.add_argument(
        "--save_dir", type=str, help="Dir to save filtered cohort files."
    )
    args = parser.parse_args()
    main(args)
