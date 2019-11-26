import numpy as np
import pandas as pd
import pickle as pkl
import sys
import copy

from typing import Dict
from argoverse.map_representation.map_api import ArgoverseMap
from argoverse.utils.centerline_utils import (
    get_nt_distance,
    get_oracle_from_candidate_centerlines
)

import os
import time
from joblib import Parallel, delayed

from baseline_config import (
        RAW_DATA_FORMAT,
        _MAX_SEARCH_RADIUS_CENTERLINES,)

from our_social_features import SocialFeaturesUtils
from our_class_features import compute_class_features
from compute_classifier_feature import compute_best_candidates

import torch
from lstm_classification import ClassRNN

obs_len = 20
pred_len = 30

# foldername = ['001',  '002', '003',  '004',  '005']
foldername = ['013',  '014',  '015',  '016',  '017',  '018',  '019',  '020',  '021']
# following must be changed according to situation !!!!!!!!!
data_dir = "../data/train/data"
batch_feature_dir = "../data/train/data"
# feature_dir = "../data/train/data/data_pickle"
classifier_model_dir = "models/Classifier.pth.tar"

batch_size = 100
# rnn model parameters
input_dim = 8
hidden_dim = 32
layer_dim = 3
output_dim = 2
mode = "test"
###################

def find_oracle_centerline(
        class_features: np.ndarray,
        rnn
        ):
    class_features = class_features.reshape((1,class_features.shape[0],class_features.shape[1]))
    predictions = rnn(torch.from_numpy(class_features).float())
    _, pred_label = torch.max(predictions.data, 1)
    
    return pred_label
    
def compute_map_features(
            agent_track: np.ndarray,
            oracle_centerline: np.ndarray,
            obs_len: int,
            pred_len: int,
            raw_data_format: Dict[str, int],
    ):
    
    agent_obs = agent_track[:obs_len]
    agent_xy_obs = agent_obs[:, [
        raw_data_format["X"], raw_data_format["Y"]
    ]].astype("float")
    
    oracle_nt_dist = get_nt_distance(agent_xy_obs,
                                     oracle_centerline,
                                     viz=False)
    
    oracle_nt_dist_norm = oracle_nt_dist - oracle_nt_dist[0,:]
    
    delta_ref = copy.deepcopy(oracle_nt_dist[0,:])
    for i in range(agent_xy_obs.shape[0]-1,0,-1):
        oracle_nt_dist[i,:] = oracle_nt_dist[i,:]-oracle_nt_dist[i-1,:]
    oracle_nt_dist[0,:] = 0
    
    angle_w_cl = np.zeros((agent_xy_obs.shape[0],1))
    angle_w_cl[1:,0] = np.arctan2(oracle_nt_dist[1:,1],oracle_nt_dist[1:,0])
    angle_w_cl[0,:] = angle_w_cl[1,:]
#    angle_w_cl[np.isnan(angle_w_cl)] = np.pi/2
    
    map_features = np.concatenate((oracle_nt_dist_norm,angle_w_cl), axis=1)

    return map_features, delta_ref

def load_compute_save (idx, file_names, social_instance, rnn, data_subdir):
    data = []
    for name in file_names[idx:(idx+batch_size)]:
        if not name.endswith(".csv"):
            continue
        file_path = data_subdir+'/'+name
        df = pd.read_csv(file_path, dtype={"TIMESTAMP": str})
        agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values

        class_features = compute_class_features(df,agent_track,obs_len,obs_len+pred_len,RAW_DATA_FORMAT)
        
        best_str, best_turn = compute_best_candidates(
            agent_track,
            obs_len,
            pred_len,
            RAW_DATA_FORMAT
        )
        
        oracle_centerline_label = find_oracle_centerline(
                class_features,
                rnn
                )
        
        if oracle_centerline_label == 0:
            oracle_centerline = best_turn
            second_centerline = best_str
        else:
            oracle_centerline = best_str
            second_centerline = best_turn
        
        map_features, delta_ref = compute_map_features(
            agent_track,
            oracle_centerline,
            obs_len,
            pred_len,
            RAW_DATA_FORMAT,
        )
        
        social_features = social_instance.compute_social_features(
            df, agent_track, obs_len, obs_len + pred_len,
            RAW_DATA_FORMAT)
        #print(agent_track.shape,social_features.shape,map_features.shape,class_features.shape)
        if agent_track.shape[0] == obs_len:
            features = np.concatenate((agent_track,social_features,map_features,class_features[:,(0,1,3,4,5,6,7)]),axis=1)
        else:
            features = np.concatenate((map_features,class_features[:,(0,1,3,4,5,6,7)]),axis=1)
            features = np.concatenate((features,np.full((pred_len,features.shape[1]),None)),axis=0)
            features = np.concatenate((agent_track,social_features,features),axis=1)
        #print(agent_track.shape,social_features.shape,map_features.shape,class_features.shape)
        
        name_id = int(name.split(".")[0])

        data.append([name_id, features, oracle_centerline, second_centerline, delta_ref])
        
    data_df = pd.DataFrame(data, 
    columns=[
        "ID",
        "FEATURES",
        "ORACLE_CENTERLINE",
        "SECOND_CENTERLINE",
        "DELTA_REFERENCE"
        ]
    )
    data_df.to_pickle(f"{batch_feature_dir}/{mode}_{idx}_{idx+batch_size-1}.pkl")
    print(f"finished computing index {idx} to {idx+batch_size-1}")
    sys.stdout.flush()
    
def merge_all_features(data_subdir):
    batch_files = os.listdir(batch_feature_dir)
    all_features = []
    for name in batch_files:
        if not name.endswith(".pkl") or mode not in name:
            continue
        file_path = batch_feature_dir+'/'+name
        df = pd.read_pickle(file_path)
        all_features.append(df)

        os.remove(file_path)

    all_features_df = pd.concat(all_features, ignore_index=True)
    print(f"Writing feature {data_subdir}")
    all_features_df.to_pickle(f"{data_subdir}/features_{mode}_{data_subdir[-2:]}.pkl")
    print(f"Feature generated")

    

if __name__ == "__main__":
    
    rnn_model = torch.load(classifier_model_dir,map_location=torch.device('cpu'))
    rnn = ClassRNN(input_dim, hidden_dim, layer_dim, output_dim)
    rnn.load_state_dict(rnn_model["rnn_state_dict"])
    rnn.eval()
    
    for idx in range(len(foldername)):
        file_names = os.listdir(data_dir+'/'+foldername[idx])
        data_subdir = data_dir+'/'+foldername[idx]
        n_file = len(file_names)
        social_instance = SocialFeaturesUtils()
        
        start = time.time()
        
        Parallel(n_jobs=-9)(delayed(load_compute_save)(i,file_names,social_instance,rnn,data_subdir) 
        for i in range(0, n_file, batch_size))
        
        merge_all_features(data_subdir)
        end = time.time()
        print(end-start)

#    file_names = os.listdir(data_dir)
#    np.random.shuffle(file_names)
#    file_names = file_names[:10]
#    n_file = len(file_names)
#    social_instance = SocialFeaturesUtils()
#    
#    start = time.time()
#    
#    Parallel(n_jobs=-9)(delayed(load_compute_save)(i,file_names,social_instance,rnn,data_dir) 
#    for i in range(0, n_file, batch_size))
#    
#    merge_all_features(data_dir)
#    end = time.time()
#    print(end-start)
#
#    df = pd.read_pickle('../test_obs/data/features_test_ta.pkl')
#    feat = np.stack(df["FEATURES"].values)
#    ang = feat[0,:,17]
#    
#    print(ang)
    
    
#    file_path = '../forecasting_sample/data/3828.csv'
#    df = pd.read_csv(file_path, dtype={"TIMESTAMP": str})
#    agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
#    
#    map_features, oracle_centerline, delta_ref = compute_map_features(
#        agent_track,
#        obs_len,
#        pred_len,
#        RAW_DATA_FORMAT,
#        mode
#    )