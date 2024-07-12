import pandas as pd
import pytest
from src.data import preprocess_data


def test_preprocess_data_with_incorrect_structure():
    # Create a DataFrame that does not match the expected structure
    # For example, missing required columns or having incorrect data types
    incorrect_df = pd.DataFrame(
        {"SomeColumn": [1, 2, 3], "AnotherColumn": ["a", "b", "c"]}
    )

    # Expecting the function to raise an error due to incorrect structure
    with pytest.raises(Exception):
        preprocess_data(incorrect_df)
