import pandas as pd
import numpy as np

from utils.baseline_config import (
    BASELINE_INPUT_FEATURES,
    BASELINE_OUTPUT_FEATURES,
    FEATURE_FORMAT,
)

feature_path = "features/forecasting_features_val.pkl"
dataframe = pd.read_pickle(feature_path)

input_features = ["TIMESTAMP","X","Y","MIN_DISTANCE_FRONT","MIN_DISTANCE_BACK"]
input_feature_idx = [FEATURE_FORMAT[feature] for feature in input_features]
print(input_feature_idx)