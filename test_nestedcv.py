import unittest
import pandas as pd
from nestedcv import NestedCV
from types import GeneratorType
import time

class TestNestedCV(unittest.TestCase):
    
    def load_test_data(self, file_path):
        # Load the DataFrame from the CSV file
        return pd.read_csv(file_path)
    
    def test_basic_case(self):
        # Load the testing DataFrame from a file
        file_path = r'data/train.csv'  
        data = self.load_test_data(file_path)
        
        data['date'] = pd.to_datetime(data['date'])
        
        # Assume 'Date' column exists in DataFrame
        k=3
        nested_cv = NestedCV(k)
        splits = nested_cv.split(data, 'date')

        count=0
        # Assert the correctness of each split
        for idx, (train, val) in enumerate(splits):
            with self.subTest(f"Subtest {idx}"):
                # Check if train_set and val_set are DataFrames
                self.assertIsInstance(train, pd.DataFrame)
                self.assertIsInstance(val, pd.DataFrame)
                
                # Check the shape of train_set and val_set
                self.assertEqual(train.shape[1], val.shape[1])
                
                self.assertTrue(train["date"].max() <= val["date"].min())
                
                count += 1
        
        # Assert the count of splits after the loop completes
        self.assertEqual(count,k)
                
        # check return type
        self.assertIsInstance(splits, GeneratorType)
        
    def test_large_dataset(self):
        # Create a large dataset
        num_rows = 20000  # Define the number of rows for the large dataset
        data = pd.DataFrame({
            'date': pd.date_range(start='2021-01-01', periods=num_rows, freq='D'),
        })
        
        k = 5  # Number of splits for NestedCV
        nested_cv = NestedCV(k)
        splits = list(nested_cv.split(data, 'date'))
        
        # Check if splits are of type GeneratorType
        self.assertIsInstance(splits, list)
        
        # Check if the number of splits matches the expected number of splits
        expected_splits = k
        self.assertEqual(len(splits), expected_splits, "Incorrect number of splits")

        count = 0
        # Iterate through the splits and perform assertions
        for idx, (train, val) in enumerate(splits):
            with self.subTest(f"Subtest {idx}"):
                # Check if train_set and val_set are DataFrames
                self.assertIsInstance(train, pd.DataFrame)
                self.assertIsInstance(val, pd.DataFrame)
                
                # Check the shape of train_set and val_set
                self.assertEqual(train.shape[1], val.shape[1])
                
                # Check if the maximum date in training set is less than or equal to the minimum date in validation set
                self.assertTrue(train["date"].max() <= val["date"].min())
                
                count += 1
        
        # Check if the count of splits matches the expected number of splits after the loop completes
        self.assertEqual(count, expected_splits)
                
    def test_k_equals_1(self):
        # Test with k = 1
        k = 1
        # Create a small dataset
        data = pd.DataFrame({
            'date': pd.date_range(start='2022-01-01', periods=100, freq='D'),
        })
        
        nested_cv = NestedCV(k)
        splits = list(nested_cv.split(data, 'date'))
        
        # Check if splits are of type GeneratorType
        self.assertIsInstance(splits, list)
        
        # Check if the number of splits matches the expected number of splits
        expected_splits = k
        self.assertEqual(len(splits), expected_splits, "Incorrect number of splits")
        
        # Check the correctness of the splits
        count = 0
        for idx, (train, val) in enumerate(splits):
            with self.subTest(f"Subtest {idx}"):
                # Check if train_set and val_set are DataFrames
                self.assertIsInstance(train, pd.DataFrame)
                self.assertIsInstance(val, pd.DataFrame)
                
                # Check the shape of train_set and val_set
                self.assertEqual(train.shape[1], val.shape[1])
                
                # Check if the maximum date in training set is less than or equal to the minimum date in validation set
                self.assertTrue(train["date"].max() <= val["date"].min())
                
                count += 1
        
        # Check if the count of splits matches the expected number of splits after the loop completes
        self.assertEqual(count, expected_splits)     

if __name__ == '__main__':
    unittest.main()