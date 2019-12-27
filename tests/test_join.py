import unittest
from flood_forecast.preprocessing.buil_dataset import join_data
import pandas as pd
import os
import json
class JoinTest(unittest.TestCase):
    def setUp(self):
        self.test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"test_data")
    def test_join_function(self):
        df = pd.read_csv(os.path.join(self.test_data_path, "big_black_test_small.csv"), sep="\t")
        asos_df = pd.read_csv(os.path.join(self.test_data_path, "asos-12N.csv"))
        with open(os.path.join(self.test_data_path, "big_black_md.json")) as a:
            meta_data = json.load(a)
        join_data(asos_df, meta_data, df )
        # TODO create dummy joined data
        self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main()
