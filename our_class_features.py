#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 09:08:40 2019

@author: x
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
#from baseline_config import RAW_DATA_FORMAT

#import os

def compute_rotation_angle(coords: np.ndarray):
    x = coords[:,0]
    y = coords[:,1]
    dx = np.zeros(x.shape)
    dy = np.zeros(y.shape)
    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]
    angle = np.arctan2(dy,dx)
#    abs_angle = angle
    delta_angle = np.zeros(len(angle))
    delta_angle[1:] = angle[1:] - angle[:-1]
    
    ind0 = np.nonzero(angle)[0][0]
    angle[ind0:] = angle[ind0:] - angle[ind0]
    scale = np.sign(angle)*(np.abs(angle)//np.pi)
    angle = angle - scale*np.pi*2
    
    return angle.reshape((len(x),1)), delta_angle.reshape((len(x),1))

def compute_histogram(angle: np.ndarray,d_coords: np.ndarray):
    
    thres1 = np.pi/20
    thres2 = np.pi/40
    hist = np.zeros((len(angle),5))
    for i in range(len(angle)):
        hist[i,:] = hist[i-1,:]
        if angle[i] < -thres1:
            hist[i,0] += np.linalg.norm(d_coords[i,:])
        elif angle[i] < -thres2:
            hist[i,1] += np.linalg.norm(d_coords[i,:])
        elif angle[i] < thres2:
            hist[i,2] += np.linalg.norm(d_coords[i,:])
        elif angle[i] < thres1:
            hist[i,3] += np.linalg.norm(d_coords[i,:])
        else:
            hist[i,4] += np.linalg.norm(d_coords[i,:])
            
    return hist

def compute_class_features(
        df: pd.DataFrame,
        agent_track: np.ndarray,
        obs_len: int,
        total_len: int,
        raw_data_format: Dict[str, int],
        ):
    """
    Args:
        df (pandas Dataframe): Dataframe containing all the tracks in the sequence
        agent_track (numpy array): Data for the agent track
        obs_len (int): Length of observed trajectory
        seq_len (int): Length of the sequence
        raw_data_format (Dict): Format of the sequence
    Returns:
        features (numpy array): 20 x 8

    """
    agent_t = np.unique(np.sort(df['TIMESTAMP'].values))
    
    if agent_t.shape[0] == obs_len:
        agent_track_obs = agent_track
    else:
        agent_track_obs = agent_track[:obs_len,:]
    
    agent_coords_obs = agent_track_obs[:,(raw_data_format["X"],raw_data_format["Y"])]
    angle, delta_angle = compute_rotation_angle(agent_coords_obs)
    
    delta_xy = np.zeros((agent_coords_obs.shape))
    delta_xy[1:,:] = agent_coords_obs[1:,:] - agent_coords_obs[:-1,:]
    
    hist = compute_histogram(delta_angle,delta_xy)
    
    features = np.concatenate((delta_xy,angle,hist),axis=1)
        
    return features

#if __name__ == "__main__":
#    file_names = os.listdir("../train/data")
#    np.random.shuffle(file_names)
#    file_names = file_names[:10]
##    file_names = os.listdir('../forecasting_sample/data')
#    n_file = len(file_names)
#    
##    start = time.time()
#    xy_data = np.empty((n_file,50,2))
#    for i in range(n_file):
#        file_path = "../train/data/"+file_names[i]
##        file_path = file_names
#        df = pd.read_csv(file_path, dtype={"TIMESTAMP": str})
#        agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
#        features = compute_class_features(df,agent_track,20,50,RAW_DATA_FORMAT)
#        xy_data[i,:,:] = agent_track[:,(RAW_DATA_FORMAT["X"],RAW_DATA_FORMAT["Y"])]
#        plt.figure()
#        plt.plot(xy_data[i,:,0],xy_data[i,:,1])
#        plt.plot(xy_data[i,:20,0],xy_data[i,:20,1])
#        plt.axis('equal')
#        print(i)
##        print('hist',features[-1,3:])
#        print('Angle',features[-1,2])
#
#    
##    end = time.time()
##    print(end-start)
