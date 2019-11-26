import numpy as np
import pandas as pd
import os

batch_feature_dir = '.'
batch_files = os.listdir(batch_feature_dir)
all_features = []
for name in batch_files:
    if not name.endswith(".pkl"):
        continue
    file_path = batch_feature_dir+'/'+name
    print(f"Reading feature from {file_path}")
    df = pd.read_pickle(file_path)
    all_features.append(df)

all_features_df = pd.concat(all_features, ignore_index=True)
print(f"Writing feature ")
all_features_df.to_pickle("features_class.pkl")
print(f"Feature generated")
