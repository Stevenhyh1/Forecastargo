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
from our_social_features import SocialFeaturesUtils
import os
import time
from joblib import Parallel, delayed

from baseline_config import (
        RAW_DATA_FORMAT,
        _MAX_SEARCH_RADIUS_CENTERLINES)

obs_len = 20
pred_len = 30

foldername = ['007',  '008', '009',  '010',  '011', '012',  '013',  '014',  '015',  '016',  '017',  '018',  '019',  '020',  '021']
# following must be changed according to situation !!!!!!!!!
data_dir = "../data/train/data"
batch_feature_dir = "../data/train/data"
feature_dir = "../data/train/data/data_pickle"

batch_size = 100

mode = "train"
###################

def compute_map_features(
            agent_track: np.ndarray,
            obs_len: int,
            pred_len: int,
            raw_data_format: Dict[str, int],
            mode: str
    ):
    
    agent_xy = agent_track[:, [raw_data_format["X"], raw_data_format["Y"]
                               ]].astype("float")
    agent_obs = agent_track[:obs_len]
    agent_xy_obs = agent_obs[:, [
        raw_data_format["X"], raw_data_format["Y"]
    ]].astype("float")

    if mode=="test":
        xy = agent_xy_obs
    else:
        xy = agent_xy
    
    avm = ArgoverseMap()

    city_name = agent_track[0, raw_data_format["CITY_NAME"]]
    
    candidate_centerlines = avm.get_candidate_centerlines_for_traj(
        xy,
        city_name,
        viz=False,
        max_search_radius=_MAX_SEARCH_RADIUS_CENTERLINES,
    )
    oracle_centerline = get_oracle_from_candidate_centerlines(
            candidate_centerlines, xy)

    oracle_nt_dist = get_nt_distance(xy,
                                     oracle_centerline,
                                     viz=False)
    delta_ref = copy.deepcopy(oracle_nt_dist[0,:])
    for i in range(xy.shape[0]-1,0,-1):
        oracle_nt_dist[i,:] = oracle_nt_dist[i,:]-oracle_nt_dist[i-1,:]
    oracle_nt_dist[0,:] = 0
    
    angle_w_cl = np.zeros((xy.shape[0],1))
    angle_w_cl[1:,0] = np.arctan2(oracle_nt_dist[1:,1],oracle_nt_dist[1:,0])
    angle_w_cl[0,:] = angle_w_cl[1,:]
#    angle_w_cl[np.isnan(angle_w_cl)] = np.pi/2
    
    map_features = np.concatenate((oracle_nt_dist,angle_w_cl), axis=1)
    
    if mode=="test":
        map_features = np.concatenate(
                (map_features, np.full([pred_len,3], None)), axis=0)

    return map_features, oracle_centerline, delta_ref

def load_compute_save (idx, file_names, social_instance, data_subdir):
    data = []
    for name in file_names[idx:(idx+batch_size)]:
        if not name.endswith(".csv"):
            continue
        file_path = data_subdir+'/'+name
        df = pd.read_csv(file_path, dtype={"TIMESTAMP": str})
        agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
        
        map_features, oracle_centerline, delta_ref = compute_map_features(
            agent_track,
            obs_len,
            pred_len,
            RAW_DATA_FORMAT,
            mode
        )
        
        social_features = social_instance.compute_social_features(
            df, agent_track, obs_len, obs_len + pred_len,
            RAW_DATA_FORMAT)
        
        if mode=="test":
            agent_track = np.concatenate(
                (agent_track[:obs_len], np.full([pred_len,6], None)), axis=0)
        
        features = np.concatenate((agent_track,social_features,map_features),axis=1)
        
        name_id = int(name.split(".")[0])

        data.append([name_id, features, oracle_centerline, delta_ref])
        
    data_df = pd.DataFrame(data, 
    columns=[
        "ID",
        "FEATURES",
        "ORACLE_CENTERLINES",
        "DELTA_REFERENCE"
        ]
    )
    data_df.to_pickle(f"{batch_feature_dir}/{mode}_{idx}_{idx+batch_size-1}.pkl")
    print(f"finished computing index {idx} to {idx+batch_size-1}")
    sys.stdout.flush()
    
def merge_all_features(idx):
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
    print(f"Writing feature {foldername[idx]}")
    all_features_df.to_pickle(f"{feature_dir}/features_{mode}_{foldername[idx]}.pkl")
    print(f"Feature generated")

    

if __name__ == "__main__":
    
    for idx in range(len(foldername)):
        file_names = os.listdir(data_dir+'/'+foldername[idx])
        data_subdir = data_dir+'/'+foldername[idx]
        n_file = len(file_names)
        social_instance = SocialFeaturesUtils()
        
        start = time.time()
        
        Parallel(n_jobs=-9)(delayed(load_compute_save)(i,file_names,social_instance,data_subdir) 
        for i in range(0, n_file, batch_size))
        
        merge_all_features(idx)
        end = time.time()
        print(end-start)
    

#    df = pd.read_pickle('../forecasting_sample/all_feature/features_'+mode+'.pkl')
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
    