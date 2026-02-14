import pandas as pd
import os

base = "/Users/leo/HRV-Complexity/Multicenter/public_release/data"

def count(f):
    path = os.path.join(base, f)
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"{f}: {len(df)}")
    else:
        print(f"{f}: Not Found")

count("chile_demographics.csv")
count("spain_demographics.csv")
count("japan_metadata.csv")
