import unittest
import pandas as pd
from cleanin import clean_titles, first_genre_filmweb, down_sample_both_dfs

class TitleCleaningTest(unittest.TestCase):
    def test_upper(self):
        test_series_upper = pd.Series(data=["W pustyni i w lesie", "Człowiek który jeździł koleją",
        "Dwóch i jedna czwarta", "Przemineło z wiatrem słonecznym"])

        test_series_upper_result = pd.Series(data=["w pustyni i w lesie", "czlowiek ktory jezdzil koleja",
        "dwoch i jedna czwarta", "przeminelo z wiatrem slonecznym"])

        assert (clean_titles(test_series_upper) == test_series_upper_result).all()
    def test_proper(self):
        test_series_proper = pd.Series(data=["w pustyni i w lesie", "czlowiek ktory jezdzil kolejom",
        "dwoch i jedna czwarta", "przeminelo z wiatrem slonecznym", "hello i love you"])
        test_series_proper_result = pd.Series(data=["w pustyni i w lesie", "czlowiek ktory jezdzil kolejom",
        "dwoch i jedna czwarta", "przeminelo z wiatrem slonecznym", "hello i love you"])
        
        assert (clean_titles(test_series_proper) == test_series_proper_result).all()

    def test_all(self):
        test_series_all = pd.Series(data=["Hello I Love you (1967)", "Won't you tell me your name", "Her arms are wicked, and her legs are...",
        "Dawno temu w Polsce", "My Eyes Have Seen You(1967)", "Nie ma to jak u obcych", "Bo do tanga trzeba jednej osoby", "Człowiek który jeździł koleją"])
        
        test_series_all_result = pd.Series(data=["hello i love you", "wont you tell me your name", "her arms are wicked, and her legs are...",
        "dawno temu w polsce", "my eyes have seen you", "nie ma to jak u obcych", "bo do tanga trzeba jednej osoby", "czlowiek ktory jezdzil koleja"])
        
        assert (clean_titles(test_series_all) == test_series_all_result).all()
        


class GenreTakerTest(unittest.TestCase):
    def test_single(self):
        test_series_single = pd.Series(data=["dramat", "komedia", "thriller", "akcja"], name="genre").to_frame()
        
        test_series_signle_result = pd.Series(data=["drama", "comedy", "thriller", "action"], name="genre").to_frame()
        
        genres = {"drama", "comedy", "thriller", "action"}
        
        assert (first_genre_filmweb(test_series_single, genres) == test_series_signle_result).all()[0]
    
    def test_single_larger(self):
        test_df_single = pd.DataFrame(data={"genre": ["dramat", "komedia", "thriller", "akcja"],
        "title": ["Hello I Love you (1967)", "Won't you tell me your name", "Her arms are wicked, and her legs are...", "Dawno temu w Polsce"]})
        
        test_df_single_result = pd.DataFrame(data={"genre": ["drama", "comedy", "thriller", "action"],
        "title": ["Hello I Love you (1967)", "Won't you tell me your name", "Her arms are wicked, and her legs are...", "Dawno temu w Polsce"]})
        
        genres = {"drama", "comedy", "thriller", "action"}
        
        assert (first_genre_filmweb(test_df_single, genres) == test_df_single_result).all()[0]
        

    def test_multiple(self):
        test_df_multiple = pd.DataFrame(data={"genre": ["venice beach romance / romance", "komedia / dramat", "thriller / Lizard King Harlequin", "satyra / dramat"],
        "title": ["Hello I Love you (1967)", "Won't you tell me your name", "Her arms are wicked, and her legs are...", "Dawno temu w Polsce"]})
        
        test_df_multiple_result = pd.DataFrame(data={"genre": ["comedy", "thriller"],
        "title": ["Won't you tell me your name", "Her arms are wicked, and her legs are..."]})

        genres = {"comedy", "thriller"}

        assert (first_genre_filmweb(test_df_multiple, genres).values == test_df_multiple_result.values).all()

        
    def test_single_multiple(self):
        test_df_multiple = pd.DataFrame(data={"genre": ["romans", "komedia / dramat", "thriller / romans", "satyra"],
        "title": ["Hello I Love you (1967)", "Won't you tell me your name", "Her arms are wicked, and her legs are...", "Dawno temu w Polsce"]})
        
        test_df_multiple_result = pd.DataFrame(data={"genre": ["romance", "comedy", "thriller"],
        "title": ["Hello I Love you (1967)", "Won't you tell me your name", "Her arms are wicked, and her legs are..."]})
        
        genres = {"comedy", "thriller", "romance", "drama"}

        assert (first_genre_filmweb(test_df_multiple, genres).values == test_df_multiple_result.values).all()



class SamplingTest(unittest.TestCase):
    def test_equal(self):
        test_df1 = pd.DataFrame(data={"genre": ["romance", "comedy", "thriller", "action", "action", "action"],
                                "title": ["Hello I Love you (1967)", "Won't you tell me your name", "Her arms are wicked, and her legs are...", "Dawno temu w Polsce",
                                "Dwóch i jenda czwarta", "Człowiek który jeździł koleją"]})

        test_df2 = pd.DataFrame(data={"genre": ["romance", "comedy", "thriller", "action", "action", "action"],
                                "title": ["Break on Through (1967)", "Take It As It Comes", "Her arms are wicked, and her legs are...", "Dawno temu w Polsce",
                                "Dwóch i jenda czwarta", "Człowiek który jeździł koleją"]})

        test_df1_result = pd.DataFrame(data={"genre": ["romance", "comedy", "thriller", "action", "action", "action"],
                                "title": ["Hello I Love you (1967)", "Won't you tell me your name", "Her arms are wicked, and her legs are...", "Dawno temu w Polsce",
                                "Dwóch i jenda czwarta", "Człowiek który jeździł koleją"]})
        test_df2_result = pd.DataFrame(data={"genre": ["romance", "comedy", "thriller", "action", "action", "action"],
                                "title": ["Break on Through (1967)", "Take It As It Comes", "Her arms are wicked, and her legs are...", "Dawno temu w Polsce",
                                "Dwóch i jenda czwarta", "Człowiek który jeździł koleją"]})
        
        result1, result2 = down_sample_both_dfs(test_df1, test_df2)

        result1_set = set(tuple(result) for result in result1.values)
        result2_set = set(tuple(result) for result in result2.values)

        test_df1_result_set = set(tuple(result) for result in test_df1_result.values)
        test_df2_result_set = set(tuple(result) for result in test_df2_result.values)
        
        
        assert (test_df1_result_set == result1_set) &  (test_df2_result_set == result2_set) 

    def test_not_equal(self):
        test_df1 = pd.DataFrame(data={"genre": ["romance", "comedy", "thriller", "action", "action", "action"],
                                "title": ["Hello I Love you (1967)", "Won't you tell me your name", "Her arms are wicked, and her legs are...", "Dawno temu w Polsce",
                                "Dwóch i jenda czwarta", "Człowiek który jeździł koleją"]})

        test_df2 = pd.DataFrame(data={"genre": ["romance", "comedy", "thriller", "comedy", "satyre", "harlequin"],
                                "title": ["Break on Through (1967)", "Take It As It Comes", "Her arms are wicked, and her legs are...", "Dawno temu w Polsce",
                                "Dwóch i jenda czwarta", "Człowiek który jeździł koleją"]})

        test_df1_result = pd.DataFrame(data={"genre": ["romance", "comedy", "thriller"], "title": ["Hello I Love you (1967)", "Won't you tell me your name", "Her arms are wicked, and her legs are..."]})
        test_df2_result = pd.DataFrame(data={"genre": ["romance", "comedy", "thriller"], "title": ["Break on Through (1967)", "Take It As It Comes", "Her arms are wicked, and her legs are..."]})
        test_df2_result_v2 = pd.DataFrame(data={"genre": ["romance", "comedy", "thriller"], "title": ["Break on Through (1967)", "Dawno temu w Polsce", "Her arms are wicked, and her legs are..."]})
        ## Here there can be two results because the function is non-deterministic in nature
        
        result1, result2 = down_sample_both_dfs(test_df1, test_df2)

        result1_set = set(tuple(result) for result in result1.values)
        result2_set = set(tuple(result) for result in result2.values)

        test_df1_result_set = set(tuple(result) for result in test_df1_result.values)
        test_df2_result_set = set(tuple(result) for result in test_df2_result.values)
        test_df2_result_v2_set = set(tuple(result) for result in test_df2_result_v2.values)
        
        assert (test_df1_result_set == result1_set) &  ( (test_df2_result_set == result2_set) |  (test_df2_result_v2_set == result2_set) )



if __name__ == "__main__":
    unittest.main()

