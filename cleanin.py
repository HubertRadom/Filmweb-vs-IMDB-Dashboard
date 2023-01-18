import pandas as pd
import numpy as np

def clean_titles(ser: pd.Series) -> pd.Series:
    ser = ser.copy()


    ser = (ser
    .str
    .replace(pat=r"(\([a-zA-Z0-9./?]+\))|([\'\"])", repl="", regex=True)
    .str
    .strip()
    .str
    .replace("ł", "l")
    .str
    .normalize("NFKD")
    .str
    .encode("ascii", errors="ignore")
    .str
    .decode("utf-8")
    .str
    .lower()
    )
    return ser


def first_genre_filmweb(df_: pd.DataFrame, genres: set) -> pd.DataFrame:
    ## Słowniczek
    mapping = {"dramat": "drama", "komedia": "comedy", "dokumentalny": "documentary", "krótkometrażowy": "short",
            "akcja": "action", "melodramat": "drama", "familijny": "family", "przygodowy": "adventure",
            "muzyczny": "music", "romans": "romance",  "sensacyjny": "thriller", "biograficzny": "biography",
            "animacja": "animation", "dramat obyczajowy": "drama", "dramat historyczny": "drama",
            "dramat sądowy": "drama", "wojenny": "war", "kryminał": "crime", "psychologiczny": "thriller",
            "komedia rom.": "comedy", "komedia kryminalna": "comedy", "komedia obycz.": "comedy"}
    
    df_ = df_.copy()

    df_ = (df_
            .assign(genre = lambda df_d: df_d
                ["genre"]
                .str
                .split(" / ")
                .apply(lambda v: v[0])
                .str
                .lower()
                .replace(mapping))
            .loc[lambda v: v.genre.isin(genres)])
    return df_


## Not sure if what's after -> is legal
def down_sample_both_dfs(df1: pd.DataFrame, df2:pd.DataFrame, n_min:int=1, n_max:int=np.inf) -> tuple:
    same_genres = set(df1.genre) & set(df2.genre)
    # print(f"Genres without the threshold: {same_genres}\n")
    groupped_df1 = df1.groupby("genre")
    groupped_df2 = df2.groupby("genre")
    
    ## TODO: Some implementation that will exclude the movies with less than n number of samples
    ## in one of the dataset
    
    groupped_df1_sizes = groupped_df1.size()
    groupped_df2_sizes = groupped_df2.size()

    genres_df1_n = groupped_df1_sizes[groupped_df1_sizes >= n_min].index
    genres_df2_n = groupped_df2_sizes[groupped_df2_sizes >= n_min].index

    same_genres_n = set(genres_df1_n) & set(genres_df2_n)
    # print(f"Genres with the inclusion threshold of {n}: {same_genres_n}\n")
    
    mins_for_every_genre = {key: min(groupped_df1_sizes[key], groupped_df2_sizes[key]) for key in same_genres_n}

    # print("Minimal values for each genre:")
    # print(mins_for_every_genre)
    
    df1_new_beggining = pd.DataFrame()
    df2_new_beggining = pd.DataFrame()
    
    for key, value in mins_for_every_genre.items():
#         print(groupped_df1[key].sample(n=value))
        df1_new_beggining = pd.concat([df1_new_beggining, df1[df1["genre"] == key].sample(n=min(value, n_max), random_state=42)], axis=0)
        df2_new_beggining = pd.concat([df2_new_beggining, df2[df2["genre"] == key].sample(n=min(value, n_max), random_state=42)], axis=0)
    return df1_new_beggining.reset_index(drop=True), df2_new_beggining.reset_index(drop=True)