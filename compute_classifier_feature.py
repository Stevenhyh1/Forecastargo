import numpy as np
import pandas as pd
import pickle as pkl
import sys
import copy

from typing import Dict
from argoverse.map_representation.map_api import ArgoverseMap
#from argoverse.utils.centerline_utils import (
#    get_nt_distance,
#    get_oracle_from_candidate_centerlines
#)
from our_class_features import compute_class_features
import os
import time
from joblib import Parallel, delayed

from baseline_config import (
        RAW_DATA_FORMAT,
        _MAX_SEARCH_RADIUS_CENTERLINES)

from shapely.geometry import Point, Polygon, LineString, LinearRing
from shapely.affinity import affine_transform, rotate
import math
import matplotlib.pyplot as plt

obs_len = 20
pred_len = 30

# following must be changed according to situation !!!!!!!!!
data_dir = "../forecasting_sample/data20"
batch_feature_dir = "../forecasting_sample/feature"
feature_dir = "../forecasting_sample/all_feature"

batch_size = 3

#mode = "test"
###################

def normalize(line, end_idx):
    traj = LineString(line)
    start = line[0] #shape [2,]
    #Translation normalization: start with (0.0, 0.0)
     #[a, b, c, d, x, y] is the planner transformation matrix[a, b, x; c, d, y; 0, 0, 1]
    g = [1, 0, 0, 1, -start[0], -start[1]]
    traj_trans = affine_transform(traj, g)
    
     #Rotation normalization: 
    end = traj_trans.coords[end_idx]
    if end[0]==0 and end[1]==0:
        angle = 0.0
    elif end[0]==0:
        angle = 90.0 if end[1]<0 else -90.0
    elif end[1]==0:
        angle = 0.0 if end[0]>0 else 180.0
    else:
        angle = math.degrees(math.atan(end[1]/end[0]))
        if (end[0] > 0 and end[1] > 0) or (end[0] > 0 and end[1] < 0):
            angle = -angle
        else:
            angle = 180.0 - angle
     #Rotate normalization: end with y=0
    traj_rotate = rotate(traj_trans, angle, origin=(0,0)).coords[:]
    
     #Transform to numpy
    traj_norm = np.array(traj_rotate)
    return traj_norm

def is_straight(centerline):
    traj_norm = normalize(centerline,10)
    
    f_delta = traj_norm[1:11,:]-traj_norm[:10,:]
    b_delta = traj_norm[-10:,:]-traj_norm[-11:-1,:]
    front = np.mean(np.arctan2(f_delta[:,1],f_delta[:,0]))
    back = np.mean(np.arctan2(b_delta[:,1],b_delta[:,0]))
    
    thres = 20
    if np.abs(front-back)<thres/180*np.pi:
        straight = 1
    else:
        straight = 0
    
    return straight

# gt=1 is straight, gt=0 is turn
def compute_gt(
            agent_track: np.ndarray,
            raw_data_format: Dict[str, int]
    ):
    agent_xy = agent_track[:, [raw_data_format["X"], raw_data_format["Y"]
                               ]].astype("float")
#    traj_norm = normalize(agent_xy, agent_xy.shape[0]-1)
#    p_pos = np.sum(traj_norm[:,1]>=0)/traj_norm.shape[0]
    
    traj_norm = normalize(agent_xy, 5)
    p_end = np.abs(traj_norm[-1,1]/traj_norm[-1,0])
    
#    plt.figure()
#    plt.plot(traj_norm[:,0],traj_norm[:,1])
#    plt.axis("equal")
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.show()
    
    thres = 0.2
    if p_end>thres:
        return 0
    else:
        return 1
    

def compute_best_candidates(
            agent_track: np.ndarray,
            obs_len: int,
            pred_len: int,
            raw_data_format: Dict[str, int]
    ):
    
    agent_obs = agent_track[:obs_len]
    agent_xy_obs = agent_obs[:, [
        raw_data_format["X"], raw_data_format["Y"]
    ]].astype("float")

    xy = agent_xy_obs
    
    avm = ArgoverseMap()

    city_name = agent_track[0, raw_data_format["CITY_NAME"]]
    
    candidate_centerlines = avm.get_candidate_centerlines_for_traj(
        xy,
        city_name,
        viz=False,
        max_search_radius=_MAX_SEARCH_RADIUS_CENTERLINES,
    )
    
    cl_str = []
    cl_turn = []
    for cl in candidate_centerlines:
        straight = is_straight(cl)
        if straight==1:
            cl_str.append(cl)
        else:
            cl_turn.append(cl)
    
    if len(cl_str)!=0:
        best_str = get_oracle_from_candidate_centerlines(cl_str, xy)
    
    if len(cl_turn)!=0:
        best_turn = get_oracle_from_candidate_centerlines(cl_turn, xy)
    
    if len(cl_str)==0:
        best_str = best_turn
    
    if len(cl_turn)==0:
        best_turn = best_str
    
    return best_str, best_turn



def load_compute_save (idx, file_names):
    data = []
    for name in file_names[idx:(idx+batch_size)]:
        if not name.endswith(".csv"):
            continue
        file_path = data_dir+'/'+name
        df = pd.read_csv(file_path, dtype={"TIMESTAMP": str})
        agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
        
        features = compute_class_features(df,agent_track,obs_len,obs_len+pred_len,RAW_DATA_FORMAT)
        
        best_str, best_turn = compute_best_candidates(
            agent_track,
            obs_len,
            pred_len,
            RAW_DATA_FORMAT
        )
        
        gt = compute_gt(agent_track, RAW_DATA_FORMAT)
        
        name_id = int(name.split(".")[0])

        data.append([name_id, features, best_str, best_turn, gt])
        
    data_df = pd.DataFrame(data, 
    columns=[
        "ID",
        "FEATURES",
        "BEST_STR_CL",
        "BEST_TURN_CL",
        "GT"
        ]
    )
    data_df.to_pickle(f"{batch_feature_dir}/_{idx}_{idx+batch_size-1}.pkl")
    print(f"finished computing index {idx} to {idx+batch_size-1}")
    sys.stdout.flush()
    
def merge_all_features():
    batch_files = os.listdir(batch_feature_dir)
    all_features = []
    for name in batch_files:
        if not name.endswith(".pkl"): #or mode not in name:
            continue
        file_path = batch_feature_dir+'/'+name
        df = pd.read_pickle(file_path)
        all_features.append(df)

        os.remove(file_path)

    all_features_df = pd.concat(all_features, ignore_index=True)

    all_features_df.to_pickle(f"{feature_dir}/classifier_feat.pkl")

    

if __name__ == "__main__":
    file_names = os.listdir(data_dir)
    n_file = len(file_names)

    
#    social_instance = SocialFeaturesUtils()
   
    start = time.time()
    
    Parallel(n_jobs=-2)(delayed(load_compute_save)(i,file_names) 
    for i in range(0, n_file, batch_size))
    
    merge_all_features()
    end = time.time()
    print(end-start)
    
#    df = pd.read_pickle('../forecasting_sample/all_feature/classifier_feat.pkl')
   
#    for i in range(5):
#        file_path = '../forecasting_sample/data100/'+file_names[i+90]
#        df = pd.read_csv(file_path, dtype={"TIMESTAMP": str})
#        agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
#        
#        gt = compute_gt(agent_track, RAW_DATA_FORMAT)
#        print(gt)
#        break
#    
    
#    file_path = '../forecasting_sample/data100/'+file_names[0]
#    df = pd.read_csv(file_path, dtype={"TIMESTAMP": str})
#    agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values

#    best_str, best_turn = compute_best_candidates(
#        agent_track,
#        obs_len,
#        pred_len,
#        RAW_DATA_FORMAT
#    )
#    
#    plt.figure()
#    plt.plot(best_str[:,0],best_str[:,1])
#    plt.axis("equal")
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.show()
#    
#    plt.figure()
#    plt.plot(best_turn[:,0],best_turn[:,1])
#    plt.axis("equal")
#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.show()
        


    