"""Helper functions for analyising/plotting results in notebook"""

import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

def summarise_datasets(all_data, names):
    means = []
    stds = []
    indexes = []

    for name in names:
        means.append(all_data[name].mean())
        stds.append(all_data[name].std())
        indexes.append(name)

    df_mean = pd.DataFrame(means, index = indexes)
    df_std = pd.DataFrame(stds, index = indexes)

    return df_mean, df_std

def do_ttest(all_data, label1, label2):
    """ Wrapper around scipy ttest_ind, using Welch's t-test"""
    label1_data = all_data[label1]
    label2_data = all_data[label2]

    cat1 = label1_data["Reproj."]
    cat2 = label2_data["Reproj."]
    reproj = ttest_ind(cat1, cat2, equal_var=False)
          
    cat1 = label1_data["Recon."]
    cat2 = label2_data["Recon."]
    recon = ttest_ind(cat1, cat2, equal_var=False)
    
    cat1 = label1_data["Tracked Reproj."]
    cat2 = label2_data["Tracked Reproj."]
    tracked_reproj = ttest_ind(cat1, cat2, equal_var=False)
    
    cat1 = label1_data["Tracked Recon."]
    cat2 = label2_data["Tracked Recon."]
    tracked_recon = ttest_ind(cat1, cat2, equal_var=False)
    
    print(f"Reprojection p-value: {reproj.pvalue}")
    print(f"Reconstruction p-value: {recon.pvalue}")
    print(f"Tracked Reprojection p-value: {tracked_reproj.pvalue}")
    print(f"Tracked Reconstruction p-value: {tracked_recon.pvalue}")

    return reproj, recon, tracked_reproj, tracked_recon

def collate_results(folder: str, threshold: int = 100) -> pd.DataFrame :
    """ Combine results from all csv files in a folder into a single dataframe
    Remove data that is above a certain threshold, which is likely to be due to an error in calbration"""
    print(f"Processing {folder}")
    csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
    data = []

    for csv in csv_files:
        file = os.path.join(folder, csv)
        data.append(pd.read_csv(file, index_col=0))
    
    all_data = data[0]
    for i in range(1, len(csv_files)):
        all_data = all_data.append(data[i])
    
    # Clean the data
    good_reproj_idx = all_data["Reproj."] < threshold
    all_data = all_data[good_reproj_idx]

    good_track_reproj_idx = all_data["Tracked Reproj."] < threshold
    all_data = all_data[good_track_reproj_idx]

    good_recon_idx = all_data["Recon."] < threshold
    all_data = all_data[good_recon_idx]

    good_track_recon_idx = all_data["Tracked Recon."] < threshold
    all_data = all_data[good_track_recon_idx]

    return all_data
    