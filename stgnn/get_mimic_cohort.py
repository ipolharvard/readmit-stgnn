import argparse
import os

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from constants import LAB_COLS

tqdm.pandas()


def main(args):
    pandarallel.initialize(progress_bar=True, nb_workers=20)

    mimic_hosp_dir = os.path.join(args.raw_data_dir, "hosp")

    df_patients = pd.read_csv(os.path.join(mimic_hosp_dir, "patients.csv.gz"),
                              parse_dates=["anchor_year", "dod"])
    df_patients.dod += pd.Timedelta(hours=23, minutes=59)
    df_admission = pd.read_csv(os.path.join(mimic_hosp_dir, "admissions.csv.gz"),
                               parse_dates=["admittime", "dischtime", "deathtime"])

    # Get readmission info
    print("Getting readmission information...")

    def process_readmission_info(_df: pd.DataFrame):
        df = _df.sort_values(by="admittime")
        admit_times = pd.Series(df.admittime[1:].tolist() + [df.dod.iloc[0]], index=df.index)
        readmit_df = (admit_times - df.dischtime.values).dt.days.to_frame("readmission_gap_in_days")
        readmit_df.loc[readmit_df.readmission_gap_in_days < 0] = 0
        readmit_df["readmitted"] = readmit_df.readmission_gap_in_days < 30
        hadm_ids = pd.Series(df.hadm_id[1:].tolist() + [np.nan], index=df.index)
        readmit_df["readmission_id"] = hadm_ids.where(readmit_df.readmitted)
        return readmit_df

    df_gb = df_admission.merge(df_patients[["subject_id", "dod"]], on="subject_id").groupby(
        "subject_id", group_keys=False)
    readmission_info_df = df_gb.parallel_apply(process_readmission_info)
    df_admission = df_admission.join(readmission_info_df)
    # drop admissions where a patient died
    df_admission["died"] = df_admission.deathtime.notna()
    df_admission.loc[df_admission.died, "dischtime"] = df_admission.loc[
        df_admission.died, "deathtime"]

    # Split patients into train/val/test
    test_patients = pd.read_csv("../data/mimic_our_test_cohort.csv").subject_id
    train_val_patients = list(set(df_admission["subject_id"]).difference(test_patients))
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.1, random_state=12
    )

    def assign_split(id):
        if id in train_patients:
            return "train"
        elif id in val_patients:
            return "val"
        else:
            return "test"

    df_admission["splits"] = df_admission["subject_id"].parallel_apply(assign_split)

    df_readmission_test = df_admission.loc[(df_admission.splits == "test") & ~df_admission.died]
    df_test = df_admission.loc[df_admission.splits == "test"]
    print(
        f"Test readmission prevalence: {df_readmission_test['readmitted'].mean():.2%}"
    )
    print(f"Test admission mortality prevalence: {df_test['died'].mean():.2%}")

    # Get demographics
    print("Getting age, gender, splits...")

    df_patients.set_index("subject_id", inplace=True)

    df_patients["year_of_birth"] = df_patients.anchor_year.dt.year - df_patients.anchor_age

    df_admission["gender"] = df_admission.subject_id.map(df_patients.gender)
    df_admission["age"] = df_admission.admittime.dt.year - df_patients.loc[
        df_admission.subject_id].year_of_birth.values

    print(f"df_admission shape: [{df_admission.shape[0]:,}x{df_admission.shape[1]:,}]")
    df_admission.to_csv(
        os.path.join(args.save_dir, "mimic_admission_demo.csv"), index=False
    )
    print("Admission basic information saved...")

    # Medication
    df_prescriptions = pd.read_csv(os.path.join(mimic_hosp_dir, "prescriptions.csv.gz"),
                                   parse_dates=["starttime"], low_memory=False,
                                   usecols=["hadm_id", "ndc", "starttime"])

    df_prescriptions = df_prescriptions.loc[df_prescriptions.hadm_id.isin(df_admission.hadm_id)]
    df_prescriptions = df_prescriptions.loc[df_prescriptions.starttime.notnull()].copy()

    # Map NDC to therapeutic classes
    df_med_map = pd.read_csv("../data/ndc2therapeutic.csv").dropna()
    df_med_map = df_med_map.groupby(
        "NDC_MEDICATION_CODE").MED_THERAPEUTIC_CLASS_DESCRIPTION.first().to_dict()

    print("Mapping NDC to therapeutic classes...")
    df_prescriptions["MED_THERAPEUTIC_CLASS_DESCRIPTION"] = df_prescriptions.ndc.map(
        df_med_map)
    df_prescriptions.dropna(subset=["MED_THERAPEUTIC_CLASS_DESCRIPTION"], inplace=True)

    df_day_num = df_prescriptions[["hadm_id", "starttime"]].merge(
        df_admission[["hadm_id", "admittime", "dischtime"]], on="hadm_id", validate="m:1",
        how="left")
    day_num = (df_day_num.starttime - df_day_num.admittime).dt.days + 1
    day_num.where(df_day_num.starttime > df_day_num.admittime, inplace=True)
    day_num.where(df_day_num.starttime < df_day_num.dischtime, inplace=True)

    df_prescriptions["Day_Number"] = day_num
    df_prescriptions.dropna(subset=["Day_Number"], inplace=True)

    print(f"df_prescriptions shape: [{df_prescriptions.shape[0]:,}x{df_prescriptions.shape[1]:,}]")
    df_prescriptions.to_csv(
        os.path.join(args.save_dir, "mimic_hosp_med_filtered.csv"), index=False
    )

    # ICD-10
    df_diag = pd.read_csv(os.path.join(mimic_hosp_dir, "diagnoses_icd.csv.gz"),
                          usecols=["hadm_id", "icd_code", "icd_version"], low_memory=False)
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
    # add problematic mappings manually
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

    df_diag["SUBGROUP"] = df_diag.icd_code.apply(
        lambda icd: icd2group.get((icd[0], icd[1:]), np.nan))
    df_diag.dropna(subset=["SUBGROUP"], inplace=True)
    print(f"df_diag shape: [{df_diag.shape[0]:,}x{df_diag.shape[1]:,}]")
    df_diag.to_csv(
        os.path.join(args.save_dir, "mimic_hosp_icd_subgroups.csv"), index=False
    )

    # Get labs
    df_lab_item = pd.read_csv(os.path.join(mimic_hosp_dir, "d_labitems.csv.gz"),
                              usecols=["itemid", "label", "fluid"], low_memory=False)
    df_lab_item["label_fluid"] = df_lab_item.label + " " + df_lab_item.fluid
    mask_lab_that_we_consider = df_lab_item.label_fluid.isin(LAB_COLS)
    df_lab_item_filtered = df_lab_item.loc[mask_lab_that_we_consider]

    # labevents file is big, process using chunks
    print("Reading lab events...")
    df_lab_filtered = pd.read_csv(os.path.join(mimic_hosp_dir, "labevents.csv.gz"),
                                  parse_dates=["charttime"],
                                  usecols=["hadm_id", "itemid", "charttime", "flag"])
    df_lab_filtered = df_lab_filtered[df_lab_filtered.hadm_id.isin(df_admission.hadm_id)]
    df_lab_filtered = df_lab_filtered.merge(df_lab_item_filtered, on="itemid")

    # Add lab name by label + fluids, and day number
    print("Getting lab information...")
    df_day_num = df_lab_filtered[["hadm_id", "charttime"]].merge(
        df_admission[["hadm_id", "admittime", "dischtime"]], on="hadm_id", validate="m:1",
        how="left")
    day_num = (df_day_num.charttime - df_day_num.admittime).dt.days + 1
    day_num.where(df_day_num.charttime > df_day_num.admittime, inplace=True)
    day_num.where(df_day_num.charttime < df_day_num.dischtime, inplace=True)

    df_lab_filtered["Day_Number"] = day_num
    df_lab_filtered.dropna(subset=["Day_Number"], inplace=True)

    print(f"df_lab_filtered shape: [{df_lab_filtered.shape[0]:,}x{df_lab_filtered.shape[1]:,}]")
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
