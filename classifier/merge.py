import pandas as pd

import os


OUTPUT_PATH = "./output/"

csvs = [OUTPUT_PATH + n for n in os.listdir(OUTPUT_PATH) if n != "test.csv"]
df = pd.concat(map(pd.read_csv, csvs))
df.to_csv("dataset.csv", index=False)
