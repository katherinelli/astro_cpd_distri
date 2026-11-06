import numpy as np
import pandas as pd
import skfda
from skfda.preprocessing.dim_reduction import FPCA
from scipy.signal import find_peaks

#### Change variables ####
# constructed_file = "~/kat_spectrum/data/some_file.csv"
df_array = pd.read_csv("~/kat_spectrum/data/flux_array_adj_sine_dist.csv") # this allocates .1% redshifts per bin (1K bins)
df_array_fluxes = df_array.iloc[:, 2:]
##########################

# FPCA components
num_components = 2
def fpca(data, num_components):
    """
    Outputs num_components number of FPCA scores
    """
    grid_points = data.columns.astype(float) # only flux data
    fd = skfda.FDataGrid(
        data_matrix=df_array_fluxes,  # or data.values
        grid_points=grid_points,
    )
    fpca = FPCA(n_components=num_components)
    fpca.fit(fd)
    scores = fpca.transform(fd)
    components = fpca.components_
    return scores, components

scores, components = fpca(df_array_fluxes, num_components)
# scores_df = pd.DataFrame(scores)
# df_all = pd.concat([df_array[['Assigned_Redshift']], scores_df], axis=1)
# df_all.to_csv(constructed_file, index = False)

# Wavelet basis
def matrix_empirical_wavelet_transform_fixed(matrix_data, num_modes=3):
    """
    Vectorized Empirical Wavelet Transform for an N x D matrix.
    Guaranteed to return exactly a (num_modes, D) basis matrix.
    """
    N_samples, D_features = matrix_data.shape
    
    # 1. Compute FFT across features
    matrix_fft = np.fft.fft(matrix_data, axis=-1)
    
    # 2. Get global average spectrum
    mean_magnitude_spectrum = np.mean(np.abs(matrix_fft[:, :D_features // 2]), axis=0)
    
    # 3. Find natural valleys between peaks
    peaks, _ = find_peaks(mean_magnitude_spectrum, distance=5, prominence=0.01 * np.max(mean_magnitude_spectrum))
    peaks = np.sort(peaks)
    
    valleys = []
    for i in range(len(peaks) - 1):
        segment = mean_magnitude_spectrum[peaks[i]:peaks[i+1]]
        valley_idx = peaks[i] + np.argmin(segment)
        valleys.append(valley_idx)

    # To get exactly 'num_modes' bands, we need exactly 'num_modes - 1' internal cuts.
    if len(valleys) == num_modes - 1:
        # Perfect match: use the naturally found valleys
        internal_cuts = valleys
    elif len(valleys) > num_modes - 1:
        # Too many valleys: take the largest/most prominent ones
        internal_cuts = sorted(valleys[:num_modes - 1])
    else:
        # Not enough valleys: ignore the failed peak detection and force an even split
        # This guarantees exactly num_modes - 1 internal cutting points
        internal_cuts = [int(x) for x in np.linspace(0, D_features // 2, num_modes + 1)][1:-1]
        
    # Build the full boundary array (Length will always be exactly num_modes + 1)
    boundaries = [0] + list(internal_cuts) + [D_features // 2]
    
    # Storage arrays
    empirical_basis_vectors = []
    empirical_modes = []
    
    # 4. Extract basis vectors and project matrix data
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        
        mask = np.zeros(D_features)
        mask[start:end] = 1.0
        mask[-end:-start] = 1.0 if start > 0 else 0.0
        if start == 0:
            mask[-end:] = 1.0
            
        # A. Get the physical Basis function
        basis_fn = np.fft.ifft(mask)
        basis_fn_centered = np.fft.fftshift(np.real(basis_fn))
        empirical_basis_vectors.append(basis_fn_centered)
            
        # B. Get the Data Projections (Modes)
        filtered_fft = matrix_fft * mask[np.newaxis, :] 
        mode_time_domain = np.fft.ifft(filtered_fft, axis=-1)
        empirical_modes.append(np.real(mode_time_domain))
        
    return np.array(empirical_basis_vectors), np.array(empirical_modes), boundaries

def calculate_mode_energy_capture(original_matrix, empirical_modes):
    """
    Computes the percentage of total dataset energy/variance 
    captured by taking the ratio of the squared L2 norms between each mode
    and the original matrix.
    
    Parameters:
    - original_matrix: The raw input dataset of shape (N, D)
    - empirical_modes: The decomposed modes array of shape (num_modes, N, D)
    """
    # 1. Calculate the total energy (sum of squares) of the original matrix
    total_dataset_energy = np.sum(original_matrix ** 2)
    
    print("--- EMPIRICAL MODE ENERGY CAPTURE ---")
    cumulative_energy = 0
    
    # 2. Iterate through each mode to calculate its percentage share
    for idx, mode in enumerate(empirical_modes):
        mode_energy = np.sum(mode ** 2)
        percentage_capture = (mode_energy / total_dataset_energy) * 100
        cumulative_energy += percentage_capture
        
        print(f"Mode {idx + 1} Capture Share: {percentage_capture:.2f}%")
        
    print(f"Total Cumulative Capture:   {cumulative_energy:.2f}% (Lossless)")
    
    return [np.sum(m**2)/total_dataset_energy * 100 for m in empirical_modes]

basis, modes, bounds = matrix_empirical_wavelet_transform_fixed(df_array_fluxes, num_modes=3)
capture_ratios = calculate_mode_energy_capture(df_array_fluxes, modes) # 1st mode capture 95% of the original matrix
wv_score = modes[0]
wv_mode_df = pd.DataFrame(wv_score)
# df_all = pd.concat([df_array[['Assigned_Redshift']], wv_mode_df], axis=1)