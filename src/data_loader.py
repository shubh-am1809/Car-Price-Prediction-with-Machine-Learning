import pandas as pd

def load_data():
    df = pd.read_csv("data/car data.csv")
    return df
