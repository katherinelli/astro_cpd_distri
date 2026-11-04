import numpy as np
import pandas as pd
import skfda
from skfda.preprocessing.dim_reduction import FPCA

# FPCA components
num_components = 2
constructed_file = "~/kat_spectrum/data/some_file.csv"
df_array = pd.read_csv("~/kat_spectrum/data/flux_array_adj_sine_dist.csv") # this allocates .1% redshifts per bin (1K bins)
df_array_fluxes = df_array.iloc[:, 2:]

grid_points = df_array_fluxes.columns.astype(float)
fd = skfda.FDataGrid(
    data_matrix=df_array_fluxes,  # or data.values
    grid_points=grid_points,
)
fpca = FPCA(n_components=num_components)
fpca.fit(fd)
scores = fpca.transform(fd)
components = fpca.components_
scores_df = pd.DataFrame(scores)
df_all = pd.concat([df_array[['Assigned_Redshift']], scores_df], axis=1)
# df_all.to_csv(constructed_file, index = False)