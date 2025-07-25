import pandas as pd
import numpy as np

def load_tacs(tacs_tsv_path):
    df = pd.read_csv(tacs_tsv_path, sep='\t')
    tac_times = (df.iloc[:, 0] + df.iloc[:, 1]) / 2  # First and second columns
    roi_names = df.columns[2:]
    tac_values = df.iloc[:, 2:].values.T  # Shape: regions x frames
    return tac_times.values, roi_names.tolist(), tac_values

def load_blood(blood_tsv_path):
    df = pd.read_csv(blood_tsv_path, sep='\t')
    plasma_times = df['time'].values
    plasma_values = df['AIF'].values
    blood_values = df["whole_blood_radioactivity"].values
    return plasma_times, plasma_values, blood_values

def load_morphology(morph_tsv_path):
    return pd.read_csv(morph_tsv_path, sep='\t')
