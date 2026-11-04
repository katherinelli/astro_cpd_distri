import numpy as np
import pandas as pd

#### Modify these variables ####
raw_df = pd.read_csv("data/qso_filtered0_24.csv")
bin_perc = 0.1
modified_df = "~/kat_spectrum/data/qso_filtered0_24_dist.csv"
################################

def binDividerbynum(binnums, z_np):
    """
    Returns a list of binnums redshifts, where each bin has the same percentage of redshifts
    """
    z_bins = []
    binpercent = 100 / binnums
    sum_pt = 0
    while sum_pt < 100:
        sum_pt += binpercent
        if sum_pt <= 100:
            z_bins.append(np.percentile(z_np, sum_pt))
    return z_bins


def binDividerbyperc(binperc, z_np):
    """
    Returns a list of redshifts at the desired percentiles (each bin has binperc redshifts)
    """
    z_bins = []
    sum_pt = 0
    while sum_pt < 100:
        sum_pt += binperc
        if sum_pt <= 100:
            percentile_value = np.percentile(z_np, sum_pt)
            z_bins.append(percentile_value)
    return z_bins


def binDividerbypercind(binperc, z_np):
    """
    Returns a list of indices where the redshifts corresponding to those indices are at the desired percentiles (each bin has binperc redshifts)
    """
    z_bins = []
    sum_pt = 0
    while sum_pt < 100:
        sum_pt += binperc
        if sum_pt <= 100:
            percentile_value = np.percentile(z_np, sum_pt)
            index_at_percentile = np.abs(z_np - percentile_value).argmin()
            z_bins.append(index_at_percentile)
    return z_bins

# 1. Get redshift array
z = np.array(raw_df["Redshift"])
# 2. Get indices of desired redshifts
bp = binDividerbypercind(bin_perc, z)
# 3. Get the median (or mean) redshift within each bin
z_idx = []
for t in range(len(bp)):
    if t == 0:
        z_min = 0
    z_max = bp[t]
    z_mean = np.median(z[z_min:z_max])
    z_idx.append(z_mean)
    z_min = z_max
# 4. Duplicate the median values so that the length matches with the length of the original redshift column
redshift_med = []
i = 0
for t in range(len(bp)):
    end_bp = bp[t]
    z_val = z_idx[t]
    while i < end_bp:
        redshift_med.append(z_val)
        i += 1
redshift_med.append(z_idx[len(bp)-1])
# 5. Insert the column from part 4 into the original data
df_array = raw_df
df_array.insert(0, "Assigned_Redshift", redshift_med)
# df_array["Assigned_Redshift"] = redshift_med
# 6. Save the modified data to a CSV file
df_array.to_csv(modified_df, index = False)