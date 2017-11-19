import pandas as pd
import json


df = pd.read_json(path_or_buf='pandas.json',orient='columns')
print(df)