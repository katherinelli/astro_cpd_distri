import numpy as np
import pandas as pd
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from typing import List

# Our custom libraries and functions
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import cmath
import math
import statistics
import pandas as pd
import scipy.stats as stats

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

# def swd(X, Y, n_projections=100, seed=None):
#     """
#     Computes the Sliced Wasserstein Distance between two distributions X and Y.
    
#     Parameters:
#     -----------
#     X : ndarray of shape (n_samples_X, n_features)
#     Y : ndarray of shape (n_samples_Y, n_features)
#     n_projections : int, number of random linear projections
#     """
#     if seed is not None:
#         np.random.seed(seed)
        
#     n_features = X.shape[1]
    
#     # 1. Generate random directions on a unit d-dimensional sphere
#     projections = np.random.normal(0, 1, (n_projections, n_features))
#     projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    
#     swd_sum = 0.0
    
#     # 2. Slice the data by projecting along each random direction
#     for i in range(n_projections):
#         v = projections[i]
        
#         # Project high-dimensional matrices down to 1D vectors
#         x_sliced = np.dot(X, v)
#         y_sliced = np.dot(Y, v)
        
#         # 3. Compute the exact 1D Wasserstein distance (Sorting step)
#         wd_1d = stats.wasserstein_distance(x_sliced, y_sliced)
#         swd_sum += wd_1d
        
#     # 4. Return the average distance across all slices
#     return swd_sum / n_projections
# import numpy as np

# def swd(X, Y, n_projections=100, p=1, seed=None):
#     """
#     Computes the Sliced Wasserstein Distance between X and Y using only NumPy.
    
#     Parameters:
#     X, Y: ndarrays of shape (n_samples, n_features)
#     n_projections: Number of random 1D lines to project onto
#     p: Power of the metric (usually 1 or 2)
#     """
#     if seed is not None:
#         np.random.seed(seed)
        
#     n_features = X.shape[1]
#     assert Y.shape[1] == n_features, "Matrices must have the same number of features (columns)."
    
#     # 1. Generate random directions on the unit hypersphere
#     # Standard normal distribution gives uniform directions, then normalize to unit length
#     projections = np.random.randn(n_projections, n_features)
#     projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    
#     # 2. Project data onto the random vectors -> shape (n_projections, n_samples)
#     X_proj = np.dot(projections, X.T)
#     Y_proj = np.dot(projections, Y.T)
    
#     # If matrix row sizes match, we can use simple element-wise operations after sorting
#     # If row sizes differ, we must interpolate the empirical CDFs
#     if X.shape[0] == Y.shape[0]:
#         X_proj_sorted = np.sort(X_proj, axis=1)
#         Y_proj_sorted = np.sort(Y_proj, axis=1)
#         # Compute the L_p distance along the projections
#         distances = np.mean(np.abs(X_proj_sorted - Y_proj_sorted) ** p, axis=1)
#     else:
#         # Handle unequal sample sizes using quantile interpolation
#         distances = []
#         for x_p, y_p in zip(X_proj, Y_proj):
#             x_sorted = np.sort(x_p)
#             y_sorted = np.sort(y_p)
            
#             # Create a shared grid of quantiles [0, 1]
#             quantiles = np.linspace(0, 1, max(len(x_p), len(y_p)))
#             x_quantiles = np.quantile(x_sorted, quantiles)
#             y_quantiles = np.quantile(y_sorted, quantiles)
            
#             distances.append(np.mean(np.abs(x_quantiles - y_quantiles) ** p))
            
#     # 3. Average the distances and take the p-th root
#     return np.mean(distances) ** (1 / p)

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

    def __init__(self, window_size=100, step_size=20, min_segment_len=10, omega=None, num_z=None):
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
        self.num_z = num_z

    def _single_cp_exhaustive_search_dist(self, Y_window):
        """
        Step 1 Helper: Finds the best single change point inside a window.
        """
        w_len, p = Y_window.shape
        unique_z = Y_window['Assigned_Redshift'].unique().list()
        Y_data = Y_window.iloc[:, 1:]

        if self.num_z < 2 * self.min_len:
            return None, float('inf')

        # Compute cumulative sum for rapid SSE calculation
        cum_sum = np.vstack([np.zeros((1, p)), np.cumsum(Y_data, axis=0)])
        # total_sq = np.sum(Y_window**2)

        best_tau = None
        max_d = -float('inf')
        # min_sse = float('inf')

        for tau in range(self.min_len, self.num_z - self.min_len + 1):
            tau_mat = unique_z[:tau]
            matched_rows = Y_window['Assigned_Redshift'].isin(tau_mat).sum()

            # Segment 1 mean & SSE
            sum_left = cum_sum[matched_rows]
            mean_left = sum_left / matched_rows

            # Segment 2 mean & SSE
            sum_right = cum_sum[w_len] - cum_sum[matched_rows]
            mean_right = sum_right / (w_len - matched_rows)

            d = ((matched_rows * (w_len - matched_rows)) / w_len) * (swd_vectorized(pd.DataFrame(mean_left), pd.DataFrame(mean_right))**2)
            # Sum of squared errors
            # sse_left = np.sum((Y_window[:tau] - mean_left)**2)
            # sse_right = np.sum((Y_window[tau:] - mean_right)**2)
            # sse = sse_left + sse_right

            # if sse < min_sse:
            #     min_sse = sse
            #     best_tau = tau
            if d > max_d:
                max_d = d
                best_tau = tau

        return best_tau, max_d

    def _get_candidates_dist(self, Y):
        """
        Step 1: Extract candidate change points across rolling windows.
        """
        Y_data = Y.iloc[:, 1:]
        unique_z = Y['Assigned_Redshift'].unique().list()
        n, p = Y_data.shape
        candidates = set()

        b = 0
        while b + self.h <= self.num_z:
            e = b + self.h
            start_z = unique_z[b]
            end_z = unique_z[e]
            first_match_index = (Y['Assigned_Redshift'] == start_z).idxmax()
            last_match_index = (Y['Assigned_Redshift'] == end_z)[::-1].idxmax()
            Y_win = Y.iloc[first_match_index:last_match_index, :]
            rel_tau, _ = self._single_cp_exhaustive_search_dist(pd.DataFrame(Y_win))
            if rel_tau is not None:
                candidates.add(b + rel_tau)
            b += self.l

        return sorted(list(candidates))

    # def _compute_segment_cost_dist(self, Y, start, end, w = 1):
    #     """
    #     Calculates SSE for segment Y[start:end].
    #     """
    #     # segment = Y[start:end]
    #     # mean = np.mean(segment, axis=0)
    #     segment = pd.DataFrame(Y.iloc[start:end, :])
    #     mean_seg =  pd.DataFrame(np.mean(segment, axis=0))
    #     l_tot = 0
    #     for i in range(segment.shape[0]):
    #         segment_ind = pd.DataFrame(Y.iloc[i, :])
    #         l_tot = l_tot + (w * swd_vectorized(segment_ind, mean_seg)**2)
    #     return l_tot
    
    def _compute_segment_cost_dist_fast(self, Y, start, end, w=1, n_projections=100, seed=None):
        """
        Highly optimized segment cost computation without Pandas loops.
        """
        # Convert to pure numpy matrices up front
        segment_matrix = Y.iloc[start:end, :].to_numpy()
        mean_seg = np.mean(segment_matrix, axis=0)
        
        # Get the SWD distance array for all rows in a vectorized manner
        swd_distances = swd_vectorized_rows(segment_matrix, mean_seg, n_projections, seed)
        
        # Square the distances, apply weight, and sum
        l_tot = np.sum(w * (swd_distances ** 2))
        return l_tot

    def _ic_screening_dist(self, Y, candidates):
        """
        Step 2: Information Criterion Screening via Dynamic Programming.
        """
        n, p = Y.shape
        omega_n = self.omega if self.omega is not None else p * np.log(n)

        # Form grid of boundaries: [0, c_1, c_2, ..., c_m, n]
        grid = [0] + candidates + [n]
        K = len(grid)

        # Precompute segment cost matrix
        cost_matrix = np.full((K, K), float('inf'))
        for i in range(K):
            for j in range(i + 1, K):
                if grid[j] - grid[i] >= self.min_len:
                    cost_matrix[i, j] = self._compute_segment_cost_dist_fast(Y, grid[i], grid[j])

        # Dynamic Programming array for minimum Information Criterion
        # dp[j] = min penalized cost from 0 to grid[j]
        dp = np.full(K, float('inf'))
        parent = np.full(K, -1, dtype=int)

        dp[0] = -omega_n  # Compensate adding omega_n for the first segment

        for j in range(1, K):
            for i in range(j):
                if cost_matrix[i, j] != float('inf'):
                    val = dp[i] + cost_matrix[i, j] + omega_n
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

        selected_change_points = [grid[idx] for idx in reversed(selected_grid_indices)]
        return selected_change_points

    def fit(self, Y):
        """
        Fit the model to dataset Y of shape (n, p).
        Returns estimated change point locations.
        """
        n, p = Y.shape
        # Step 1: Rolling window candidate collection
        print(Y["Assigned_Redshift"])
        candidates = self._get_candidates_dist(pd.DataFrame(Y))

        if not candidates:
            return []
        return candidates
        # # Step 2: Information Criterion screening
        # final_change_points = self._ic_screening_dist(Y, candidates)
        # return final_change_points