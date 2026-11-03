import numpy as np
import pandas as pd
import math
import statistics
import pandas as pd

def omega_hat_dist(data):
    # data_adj = data.iloc[:, 1:]
    data_adj_pre = data.iloc[:, :-1]
    data_adj_post = data.iloc[:, 1:]
    omega_cands = []
    for j in range(data_adj_pre.shape[1]):
        data_adj_diff = abs(data_adj_post.iloc[:, j] - data_adj_pre.iloc[:, j])
        data_adj_diff_med = statistics.median(data_adj_diff)
        omega_cands.append((data_adj_diff_med/(math.sqrt(2)*0.6745))**2)
    return statistics.median(omega_cands)

def swd_vectorized(matrix1, matrix2, n_projections=100, p=1, seed=None):
    """
    Computes the Sliced Wasserstein Distance (SWD) between two point-clouds
    represented by matrix1 (N x D) and matrix2 (M x D).
    
    Parameters:
    -----------
    matrix1 : ndarray of shape (n_samples_1, n_features)
    matrix2 : ndarray of shape (n_samples_2, n_features)
    n_projections : int, number of random linear projections
    p : int or float, the power/order of the Wasserstein distance (usually 1 or 2)
    """
    if seed is not None:
        np.random.seed(seed)
        
    n_samples1, n_features = matrix1.shape
    n_samples2, n_features2 = matrix2.shape
    
    if n_features != n_features2:
        raise ValueError("Both matrices must have the same number of features (dimensions).")
        
    # 1. Generate random projection vectors on the unit sphere
    projections = np.random.normal(0, 1, (n_projections, n_features))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    
    # 2. Project both datasets to 1D: shapes will be (n_samples, n_projections)
    proj1 = np.dot(matrix1, projections.T)
    proj2 = np.dot(matrix2, projections.T)
    
    # 3. CRITICAL: Sort each projection independently (along the sample axis)
    proj1_sorted = np.sort(proj1, axis=0)
    proj2_sorted = np.sort(proj2, axis=0)
   
    # 4. Handle unequal sample sizes (N != M) via quantile matching if necessary
    if n_samples1 != n_samples2:
        # Interpolate to compare empirical distributions of different sizes
        u = np.linspace(0, 1, max(n_samples1, n_samples2))
        proj1_resampled = np.percentile(proj1_sorted, u * 100, axis=0)
        proj2_resampled = np.percentile(proj2_sorted, u * 100, axis=0)
     
        # Compute L_p distance between matched quantiles
        distances = np.abs(proj1_resampled - proj2_resampled) ** p
    else:
        # If sizes are equal, directly compare sorted elements
        distances = np.abs(proj1_sorted - proj2_sorted) ** p
        
    # 5. Take the average across projections, then the p-th root
    swd = np.mean(distances) ** (1.0 / p)
    
    return swd

def swd_vectorized_rows(segment_matrix, mean_seg, n_projections=100, seed=None):
    """
    Computes the sum of squared SWD between each individual row in segment_matrix 
    and the mean vector mean_seg, completely bypassing row-by-row iteration.
    """
    if seed is not None:
        np.random.seed(seed)
        
    n_samples, n_features = segment_matrix.shape
    
    # 1. Generate projections ONCE for the entire segment calculation
    projections = np.random.normal(0, 1, (n_projections, n_features))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    
    # 2. Project everything to 1D simultaneously using matrix multiplication
    # segment_projections shape: (n_samples, n_projections)
    # mean_projections shape: (n_projections,)
    segment_projections = np.dot(segment_matrix, projections.T)
    mean_projections = np.dot(mean_seg, projections.T)
    
    # 3. Compute 1D Wasserstein distances efficiently
    # For a 1D single-point distribution vs. another 1D single point, 
    # the Wasserstein distance simplifies to the absolute difference: |x - y|
    # swd_per_row will hold the SWD value for each row in the segment
    swd_per_row = np.zeros(n_samples)
    
    for j in range(n_projections):
        v_mean = mean_projections[j]  # Scalar value of the mean projection
        
        # Calculate 1D Wasserstein distance for each row's projection against the mean projection
        # Because x_sliced is a single scalar point for each sample, stats.wasserstein_distance 
        # reduces down to an element-wise absolute difference.
        swd_per_row += np.abs(segment_projections[:, j] - v_mean)
        
    # Average across all projection slices
    swd_per_row /= n_projections
    
    return swd_per_row

class MeanShiftChangePointDetectorDist:
    """
    Multiple Change Point Detector for High-Dimensional Mean Shift Models
    y_i = mu_i + epsilon_i, i = 1, ..., n, y_i in R^p.
    """

    def __init__(self, window_size=100, step_size=20, min_segment_len=10, omega=None):
        """
        Parameters:
        -----------
        window_size : int (h)
            Length of the rolling/moving window.
        step_size : int (l)
            Shift size for the rolling window.
        min_segment_len : int
            Minimum distance between change points or window boundaries.
        omega : float or None
            Penalty term omega_n for the Information Criterion.
            If None, default heuristic omega_n = p * log(n) is used.
        """
        self.h = window_size
        self.l = step_size
        self.min_len = min_segment_len
        self.omega = omega

    def _single_cp_exhaustive_search_dist(self, Y_window):
        """
        Step 1 Helper: Finds the best single change point inside a window.
        """
        w_len, p = Y_window.shape
        unique_z = Y_window['Assigned_Redshift'].unique().tolist()
        num_z = len(unique_z)
        Y_data = Y_window.iloc[:, 1:]

        if num_z < 2 * self.min_len:
            return None, float('inf')

        # Compute cumulative sum for rapid SSE calculation
        cum_sum = np.vstack([np.zeros((1, p-1)), np.cumsum(Y_data, axis=0)])

        best_tau = None
        max_d = -float('inf')
   
        for tau in range(self.min_len, num_z - self.min_len + 1):
            tau_mat = unique_z[:tau]
            matched_rows = Y_window['Assigned_Redshift'].isin(tau_mat).sum()
            # Segment 1 mean & SSE
            sum_left = cum_sum[matched_rows]
            mean_left = sum_left / matched_rows

            # Segment 2 mean & SSE
            sum_right = cum_sum[w_len] - cum_sum[matched_rows]
            mean_right = sum_right / (w_len - matched_rows)

            d = ((matched_rows * (w_len - matched_rows)) / w_len) * (swd_vectorized(pd.DataFrame(mean_left), pd.DataFrame(mean_right))**2)

            if d > max_d:
                max_d = d
                best_tau = tau

        return best_tau, max_d
    
    def _get_candidates_dist(self, Y):
        """
        Step 1: Extract candidate change points across rolling windows.
        """
        unique_z = Y['Assigned_Redshift'].unique().tolist()
        num_z = len(unique_z)
        candidates = set()

        b = 0
        while b + self.h < num_z:
            e = b + self.h
            start_z = unique_z[b]
            end_z = unique_z[e]
            first_match_index = (Y['Assigned_Redshift'] == start_z).idxmax()
            last_match_index = (Y['Assigned_Redshift'] == end_z)[::-1].idxmax()
            Y_win = Y.iloc[first_match_index:last_match_index, :]
            rel_tau, _ = self._single_cp_exhaustive_search_dist(Y_win)
            if rel_tau is not None:
                candidates.add(b + rel_tau)
            b += self.l

        return sorted(list(candidates))

    
    def _compute_segment_cost_dist_fast(self, Y_data, precomputed_means, first_matches, last_matches, start, end, w=1, n_projections=100, seed=None):
        """
        Truly optimized segment cost using precomputed lookups and avoiding Pandas filters.
        """
        num_z = len(precomputed_means)
        unique_z = Y_data.iloc[:, 0].unique().tolist()
        
        # Get boundary indices using precomputed lookups
        first_match_index = first_matches[start]
        last_match_index = last_matches[end]
        start_z = Y_data.iloc[first_match_index, 0]
        end_z = Y_data.iloc[last_match_index, 0]
        
        # Bound slices and compute the target segment mean
        segment_matrix = Y_data.iloc[first_match_index:last_match_index + 1, 1:].to_numpy() # Ensured last row is included if needed
        mean_seg_np = np.mean(segment_matrix, axis=0).reshape(1, 2)

        swd_distances_all = 0
        for i in range(num_z):
            if unique_z[i] >= start_z and unique_z[i] <= end_z:
                # Pull precomputed mean for this specific redshift directly out of a numpy array
                select_mean_seg_np = precomputed_means[i].reshape(1, 2)
                
                # Vectorized SWD calculation
                swd_distances = swd_vectorized(select_mean_seg_np, mean_seg_np, n_projections, p=1, seed=seed)
                swd_distances_all += (w * (swd_distances ** 2))
        return np.sum(swd_distances_all)

    def _ic_screening_dist(self, Y, candidates):
        """
        Step 2: Information Criterion Screening with Precomputed Static Data.
        """
        n, p = Y.shape
        
        # 1. Extract values to pure NumPy immediately
        Y_data_np = Y.iloc[:, 1:].to_numpy()
        z_series = Y['Assigned_Redshift']
        unique_z = z_series.unique().tolist()
        num_z = len(unique_z)
        
        # 2. PRECOMPUTE LOOKUPS: Find match indices once
        first_matches = {i: (z_series == unique_z[i]).idxmax() for i in range(num_z)}
        # Reverse lookup for last match mapping
        rev_z_series = z_series[::-1]
        last_matches = {i: (rev_z_series == unique_z[i]).idxmax() for i in range(num_z)}
        
        # 3. PRECOMPUTE MEANS: Compute every unique redshift mean group exactly ONCE
        precomputed_means = []
        for select_z in unique_z:
            # Extract rows matching this redshift efficiently
            mask = (z_series == select_z).to_numpy()
            precomputed_means.append(np.mean(Y_data_np[mask], axis=0))
        precomputed_means = np.array(precomputed_means)

        # Form grid of boundaries
        grid = [0] + candidates + [num_z - 1]
        K = len(grid)

        # Precompute segment cost matrix
        cost_matrix = np.full((K, K), float('inf'))
        for i in range(K):
            for j in range(i + 1, K):
                if grid[j] - grid[i] >= self.min_len:
                    # Pass precomputed NumPy arrays and lookups directly
                    cost_matrix[i, j] = self._compute_segment_cost_dist_fast(
                        Y, precomputed_means, first_matches, last_matches, grid[i], grid[j]
                    )

        # Dynamic Programming array for minimum Information Criterion
        dp = np.full(K, float('inf'))
        parent = np.full(K, -1, dtype=int)
        dp[0] = -self.omega  

        for j in range(1, K):
            for i in range(j):
                if cost_matrix[i, j] != float('inf'):
                    val = dp[i] + cost_matrix[i, j] + self.omega
                    if val < dp[j]:
                        dp[j] = val
                        parent[j] = i
    
        # Backtrack to reconstruct chosen change points
        curr = K - 1
        selected_grid_indices = []
        while parent[curr] != -1:
            prev = parent[curr]
            if prev != 0:
                selected_grid_indices.append(prev)
            curr = prev

        return [grid[idx] for idx in reversed(selected_grid_indices)]

    def fit(self, Y):
        """
        Fit the model to dataset Y of shape (n, p).
        Returns estimated change point locations.
        """
        n, p = Y.shape
        # Step 1: Rolling window candidate collection
        candidates = self._get_candidates_dist(pd.DataFrame(Y))

        if not candidates:
            return []
        
        # # Step 2: Information Criterion screening
        final_change_points = self._ic_screening_dist(Y, candidates)
        return final_change_points