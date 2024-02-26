import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from pandarallel import pandarallel

tqdm.pandas()

pandarallel.initialize(progress_bar=True, nb_workers=20)

# %%
df = pd.read_csv("data/mimic_processed/ehr_combined.csv")

# %%
df.drop(columns=["subject_id", "admittime", "dischtime"], inplace=True)
df = df.loc[~df.died].copy()
# %%
cols = df.select_dtypes("float").columns
df[cols] = df[cols].astype("uint16")
# %%
med_cols = df.columns[-41:].tolist()
lab_cols = df.columns[-87:-41].tolist()
diag_cols = df.columns[12:-87].tolist()
dem_cols = ['gender', 'age']

df_gb = df.groupby("hadm_id")
# %%


take_first = lambda x: x.iloc[0]


def extract(_df):
    return _df.agg({
        "readmitted": take_first,
        **{col: take_first for col in dem_cols},
        "Day_Number": "max",
        **{col: take_first for col in diag_cols},
    })


df_processed = df_gb[["readmitted", "Day_Number"] + dem_cols + diag_cols].parallel_apply(extract)

# %%

feat_extr = [
    ("MIN", "min"),
    ("Q1", lambda x: np.quantile(x, 0.25)),
    ("MEDIAN", "median"),
    ("Q3", lambda x: np.quantile(x, 0.75)),
    ("MAX", "max")
]

more_processed = []
for func_name, func in feat_extr:
    another_one = df_gb[lab_cols + med_cols].parallel_apply(lambda x: x.agg(func))
    another_one = another_one.loc[:, another_one.nunique() > 1].copy()
    another_one.columns = [f"{col}_{func_name}" for col in another_one.columns]
    more_processed.append(another_one)

df = pd.concat([df_processed, *more_processed], axis=1)
df.to_csv("data/mimic_processed/ehr_combined_flat.csv")

print(df.shape)
