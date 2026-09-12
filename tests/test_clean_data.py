import pandas as pd
from src.data.clean_data import remove_duplicates, check_missing

def test_remove_duplicates():
    df = pd.DataFrame({'A': [1,2,2,3], 'B': [4,5,5,6]})
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 3

def test_remove_duplicates_no_duplicates():
    df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
    cleaned = remove_duplicates(df)
    assert len(cleaned) == 3

def test_check_missing_no_missing(capsys):
    df = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})
    check_missing(df)
    captured = capsys.readouterr()
    assert "No missing values" in captured.out

def test_check_missing_with_missing(capsys):
    df = pd.DataFrame({'A': [1, None, 3], 'B': [4,5,6]})
    check_missing(df)
    captured = capsys.readouterr()
    assert "missing values found" in captured.out