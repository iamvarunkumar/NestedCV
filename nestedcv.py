import pandas as pd

class NestedCV:
    """Nested Cross-Validation class for splitting data into training and validation sets."""

    def __init__(self, k:int):
        """
        Initialize NestedCV class with the number of folds (k).

        Parameters:
        - k (int): Number of folds for cross-validation.
        """
        self.k = k

    def split(self, data: pd.DataFrame, date_column: str):
        """
        Split the data into k folds for nested cross-validation.

        Parameters:
        - data (DataFrame): Input data to be split.
        - date_column (str): Column containing date information.

        Yields:
        - training_set (DataFrame): Subset of the data for training.
        - val_set (DataFrame): Subset of the data for validation.
        """

        # Error handling for empty DataFrame
        if data.empty:
            raise ValueError("Input data is empty. Please provide a non-empty DataFrame.")

        try:
            # Sort data based on date_column and create a 'M_Y' column for month-year
            data = data.sort_values(by=date_column, ascending=True)
            data['M_Y'] = data[date_column].dt.strftime('%m-%Y')
        except Exception as e:
            raise ValueError(f"Error processing date column: {e}. Please ensure proper date values.")

        n_month_year = data['M_Y'].value_counts().shape[0]
        months_for_validation = n_month_year // (self.k + 1)

        unique_my_values = data['M_Y'].unique()
        n_month_year = len(unique_my_values)

        for i in range(1, self.k + 1):
            training_set = i * months_for_validation
            validation_start_index = training_set
            validation_end_index = validation_start_index + months_for_validation

            if validation_end_index >= n_month_year:
                validation_range = unique_my_values[validation_start_index:]
            else:
                validation_range = unique_my_values[validation_start_index:validation_end_index]

            training_range = unique_my_values[:validation_start_index]

            # Error handling for boundary scenario
            if i == self.k:
                validation_range = unique_my_values[validation_start_index:]

            try:
                # Get training and validation sets based on the computed ranges
                training_set = data[data['M_Y'].isin(training_range)]
                val_set = data[data['M_Y'].isin(validation_range)]
            except Exception as e:
                raise ValueError(f"Error in split operation: {e}. Please check the splitting logic.")

            yield training_set, val_set
