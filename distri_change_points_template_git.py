import numpy as np
import pandas as pd
import math
import statistics
import pandas as pd

def omega_hat_dist(data):
    """
    Penalty term for initial change point screening
    """
    data_adj_pre = data.iloc[:, :-1]
    data_adj_post = data.iloc[:, 1:]
    omega_cands = []
    for j in range(data_adj_pre.shape[1]):
        data_adj_diff = abs(data_adj_post.iloc[:, j] - data_adj_pre.iloc[:, j])
        data_adj_diff_med = statistics.median(data_adj_diff)
        omega_cands.append((data_adj_diff_med/(math.sqrt(2)*0.6745))**2)
    return statistics.median(omega_cands)

def make_swd_fn(n_samples1, n_samples2, n_features, n_projections=100, p=1, seed=None):
    """
    Pre-computes projections and interpolation weights once, returning a fast
    standalone function `swd_fn(matrix1, matrix2)` for repeated calls.
    """
    # 1. Pre-compute unit projection vectors: shape (n_features, n_projections)
    if n_features == 2:
        # Uniform angles on [0, pi) are exact & faster for 2D
        angles = np.linspace(0, np.pi, n_projections, endpoint=False, dtype=np.float32)
        projections = np.vstack([np.cos(angles), np.sin(angles)])
    else:
        rng = np.random.default_rng(seed)
        projections = rng.standard_normal((n_features, n_projections), dtype=np.float32)
        projections /= np.linalg.norm(projections, axis=0, keepdims=True)

    # 2. Pre-compute midpoint quantile interpolation indices & weights ONCE
    needs_interp = (n_samples1 != n_samples2)
    if needs_interp:
        src_len = min(n_samples1, n_samples2)
        tgt_len = max(n_samples1, n_samples2)
        
        q_small = (np.arange(src_len, dtype=np.float32) + 0.5) / src_len
        q_large = (np.arange(tgt_len, dtype=np.float32) + 0.5) / tgt_len
        
        idx_high = np.clip(np.searchsorted(q_small, q_large), 1, src_len - 1)
        idx_low = idx_high - 1
        
        w_high = np.clip((q_large - q_small[idx_low]) / (q_small[idx_high] - q_small[idx_low]), 0.0, 1.0)
        w_low = 1.0 - w_high

    # 3. Return the fast inner function to call inside your loop
    def compute_swd(matrix1, matrix2):
        m1 = np.ascontiguousarray(matrix1, dtype=np.float32)
        m2 = np.ascontiguousarray(matrix2, dtype=np.float32)

        # Project & sort along contiguous rows: shape (n_projections, n_samples)
        p1 = np.sort((m1 @ projections).T, axis=1)
        p2 = np.sort((m2 @ projections).T, axis=1)

        # Resample ONLY the smaller array using pre-computed weights
        if needs_interp:
            if n_samples1 < n_samples2:
                p1 = p1[:, idx_low] * w_low + p1[:, idx_high] * w_high
            else:
                p2 = p2[:, idx_low] * w_low + p2[:, idx_high] * w_high

        diff = np.abs(p1 - p2)
        if p == 1:
            return float(np.mean(diff))
        elif p == 2:
            return float(np.sqrt(np.mean(diff * diff)))
        return float(np.mean(diff ** p) ** (1.0 / p))

    return compute_swd

def swd_cached(matrix1, matrix2, swd_dict, n_projections=100, p=1, seed=None):
    n1, d1 = matrix1.shape
    n2, d2 = matrix2.shape
    if d1 != d2:
        raise ValueError(f"Feature dimensions must match: {d1} vs {d2}")

    # Key uniquely identifies the shapes and hyperparameters
    key = (n1, n2, d1, n_projections, p, seed)

    # Create and store the function only if this shape combo hasn't been seen yet
    if key not in swd_dict:
        swd_dict[key] = make_swd_fn(n1, n2, d1, n_projections=n_projections, p=p, seed=seed)

    return swd_dict[key](matrix1, matrix2)

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
        swd_cache_init = {}

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
            # sum_left = cum_sum[matched_rows]
            # mean_left = sum_left / matched_rows

            # Segment 1
            matrix_left = Y_window.iloc[:matched_rows, :]

            # Segment 2 mean & SSE
            # sum_right = cum_sum[w_len] - cum_sum[matched_rows]
            # mean_right = sum_right / (w_len - matched_rows)

            # Segment 2
            matrix_right = Y_window.iloc[matched_rows:w_len, :]

            # d = ((matched_rows * (w_len - matched_rows)) / w_len) * (swd_vectorized(pd.DataFrame(matrix_left), pd.DataFrame(matrix_right))**2)
            d = ((matched_rows * (w_len - matched_rows)) / w_len) * (swd_cached(pd.DataFrame(matrix_left), pd.DataFrame(matrix_right), swd_cache_init, n_projections = 100, p=1)**2)
            # print(d)
            if d > max_d:
                max_d = d
                best_tau = tau
            print(tau)
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

    
    def _compute_segment_cost_dist_fast(self, Y_data, w=1, n_projections=100, seed=None):
        """
        Truly optimized segment cost using precomputed lookups and avoiding Pandas filters.
        """
        unique_z = Y_data.iloc[:, 0].unique().tolist()
        num_z = len(unique_z)
        seg_Y_np = Y_data.iloc[:, 1:].to_numpy()

        swd_cache = {}
        swd_distances_all = 0
        for i in range(num_z):
                # Pull precomputed mean for this specific redshift directly out of a numpy array
            select_ind_seg_np = Y_data[Y_data.iloc[:, 0] == unique_z[i]].iloc[:, 1:]
            # select_mean_seg_np = precomputed_means[i].reshape(1, segment_matrix_np.shape[1])
            
            # Vectorized SWD calculation
            # swd_distances = swd_vectorized(select_ind_seg_np, Y_data.iloc[:, 1:].to_numpy(), n_projections, p=1, seed=seed)
            swd_distances = swd_cached(select_ind_seg_np, seg_Y_np, swd_cache, n_projections, p=1)
            swd_distances_all += (w * (swd_distances ** 2))
        return np.sum(swd_distances_all)

    def _ic_screening_dist(self, Y, candidates):
        """
        Step 2: Information Criterion Screening with Precomputed Static Data.
        """
        # n, p = Y.shape 
        
        # 1. Extract values to pure NumPy immediately
        # Y_data_np = Y.iloc[:, 1:].to_numpy()
        z_series = Y['Assigned_Redshift']
        unique_z = z_series.unique().tolist()
        num_z = len(unique_z)
        
        # 2. PRECOMPUTE LOOKUPS: Find match indices once
        first_matches = {i: (z_series == unique_z[i]).idxmax() for i in range(num_z)}
        # Reverse lookup for last match mapping
        rev_z_series = z_series[::-1]
        last_matches = {i: (rev_z_series == unique_z[i]).idxmax() for i in range(num_z)}

        # Form grid of boundaries
        grid = [0] + candidates + [num_z - 1]
        K = len(grid)
        print(K)
        # Precompute segment cost matrix
        cost_matrix = np.full((K, K), float('inf'))
        for i in range(K):
            print(i)
            for j in range(i + 1, K):
                print(j)
                if grid[j] - grid[i] >= self.min_len:
                    start = grid[i]
                    end = grid[j]
                    first_match_index = first_matches[start]
                    last_match_index = last_matches[end]
                    seg_Y = Y.iloc[first_match_index:last_match_index + 1, :]
                    cost_matrix[i, j] = self._compute_segment_cost_dist_fast(seg_Y, n_projections = 100)

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