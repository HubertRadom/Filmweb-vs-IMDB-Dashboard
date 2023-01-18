from cleanin import down_sample_both_dfs
import pandas as pd
import numpy as np
import sys
import os


def create_datasets(path_filmweb, path_imdb, path_splits):

    df_filmweb = pd.read_csv(path_filmweb)
    df_imdb_all = pd.read_csv(path_imdb)
    splits = pd.read_csv(path_splits)

    for split in splits.values:
        n_min = split[0]
        n_max = split[1]
        
        if np.isnan(n_min):
            n_min = 0

        if np.isnan(n_max):
            n_max = np.inf
        
        if n_max != np.inf :
            n_max = int(n_max)
        n_min = int(n_min)
        
        ending_convention = ""
        if n_min and n_max != np.inf:
            if n_min == n_max:
                ending_convention = str(n_min) + "_eq"
            else:
                ending_convention = "beetween_" + str(n_min) + "_and_" + str(n_max)
        elif n_min:
            ending_convention = str(n_min) + "_up"
        elif n_max != np.inf:
            ending_convention = str(n_max) + "_down"
        else:
            ending_convention = "all"

        
        # df_filmweb_sampled, df_imdb_all_sampled = down_sample_both_dfs(df_filmweb, df_imdb_all, n_min=n_min, n_max=n_max)
        for i, df_sample in enumerate(down_sample_both_dfs(df_filmweb, df_imdb_all, n_min=n_min, n_max=n_max)):
            path_name = "datasets/ready_data_" + ["filmweb", "imdb"][i] + "_" + ending_convention + ".csv"
            if not os.path.exists(path_name):
                df_sample.sample(frac=1.).to_csv(path_name, index=False)
    
if __name__ == "__main__":
    create_datasets(sys.argv[1], sys.argv[2], sys.argv[3])