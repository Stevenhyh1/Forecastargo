#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module is used for computing social features for motion forecasting baselines."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import time
import os

from baseline_config import (
    PADDING_TYPE,
    STATIONARY_THRESHOLD,
    VELOCITY_THRESHOLD,
    EXIST_THRESHOLD,
    DEFAULT_MIN_DIST_FRONT_AND_BACK,
    NEARBY_DISTANCE_THRESHOLD,
    FRONT_OR_BACK_OFFSET_THRESHOLD,
    RAW_DATA_FORMAT
)

class SocialFeaturesUtils:
    """Utils class for computation of social features."""
    def __init__(self):
        """Initialize class."""
        self.PADDING_TYPE = PADDING_TYPE
        self.STATIONARY_THRESHOLD = STATIONARY_THRESHOLD
        self.VELOCITY_THRESHOLD = VELOCITY_THRESHOLD
        self.EXIST_THRESHOLD = EXIST_THRESHOLD
        self.DEFAULT_MIN_DIST_FRONT_AND_BACK = DEFAULT_MIN_DIST_FRONT_AND_BACK
        self.NEARBY_DISTANCE_THRESHOLD = NEARBY_DISTANCE_THRESHOLD

    def compute_velocity(self, track_coords: np.ndarray, time: np.ndarray):
        """Compute velocities for the given track.

        Args:
            track_df (pandas Dataframe): Data for the track
        Returns:
            vel (list of float): Velocity at each timestep

        """
        x = track_coords[:,0].astype('float')
        y = track_coords[:,1].astype('float')
        time = time.astype('float')
        velocity = np.empty((len(time)))
        
        dt = time[1:]-time[:-1]
        vx = (x[1:]-x[:-1])/dt
        vy = (y[1:]-y[:-1])/dt
        velocity[1:] = np.sqrt(np.square(vx)+np.square(vy))
        velocity[0] = velocity[1]

        return velocity.reshape((len(time),1))

    def get_is_track_stationary(self, track_df: pd.DataFrame):
        """Check if the track is stationary.

        Args:
            track_df (pandas Dataframe): Data for the track
        Return:
            _ (bool): True if track is stationary, else False 

        """
        track_coords_x = track_df["X"].values
        track_coords_y = track_df["Y"].values
        length = len(track_coords_x)
        track_coords = np.empty((length,2))
        track_coords[:,0] = track_coords_x
        track_coords[:,1] = track_coords_y
        time = track_df["TIMESTAMP"].values.astype('float')
        velocity = self.compute_velocity(track_coords, time)
        sort_velocity = np.sort(velocity)
        velocity_thres = sort_velocity[self.STATIONARY_THRESHOLD]
        
        if velocity_thres < self.VELOCITY_THRESHOLD:
            return True
        else:
            return False


    def fill_track_lost_in_middle(
            self,
            track_array: np.ndarray,
            timestamps: np.ndarray,
            raw_data_format: Dict[str, int],
    ):
        """Handle the case where the object exited and then entered the frame but still retains the same track id. It'll be a rare case.

        Args:
            track_array (numpy array): Padded data for the track
            timestamps (numpy array): All timestamps in the sequence
            raw_data_format (Dict): Format of the sequence
        Returns:
            filled_track (numpy array): Track data filled with missing timestamps

        """
        ind = 0
        
        filled_track = np.empty((0, track_array.shape[1]))
        
        for timestamp in timestamps:
            filled_track = np.vstack((filled_track, track_array[ind]))
            if timestamp in track_array[:, raw_data_format["TIMESTAMP"]]:
                ind += 1
        
        return filled_track
        

    def pad_track(
            self,
            track_df: pd.DataFrame,
            timestamps: np.ndarray,
            obs_len: int,
            raw_data_format: Dict[str, int],
    ):
        """Pad incomplete tracks.

        Args:
            track_df (Dataframe): Dataframe for the track
            timestamps (numpy array): All timestamps in the sequence
            obs_len (int): Length of observed trajectory
            raw_data_format (Dict): Format of the sequence
        Returns:
                padded_track_array (numpy array): Track data padded in front and back

        """
        track_data = track_df.values
        track_t = track_df["TIMESTAMP"].values

        # start and index of the track in the sequence
        ind_start = np.where(timestamps == track_t[0])[0][0]
        ind_end = np.where(timestamps == track_t[-1])[0][0]

        # Edge padding in front and rear, i.e., repeat the first and last coordinates
        if self.PADDING_TYPE == "REPEAT":
            padded_track_data = np.pad(track_data,
                                        ((ind_start, obs_len - ind_end - 1),
                                         (0, 0)), "edge")
            if padded_track_data.shape[0] < obs_len:
                padded_track_data = self.fill_track_lost_in_middle(
                    padded_track_data, timestamps, raw_data_format)

        # Overwrite the timestamps in padded part
        padded_track_data[:, 0] = timestamps
        
        return padded_track_data

    def track_filter(self, df: pd.DataFrame, obs_len: int,
                      raw_data_format: Dict[str, int]):
        """Pad tracks which don't last throughout the sequence. Also, filter out non-relevant tracks.

        Args:
            seq_df (pandas Dataframe): Dataframe containing all the tracks in the sequence
            obs_len (int): Length of observed trajectory
            raw_data_format (Dict): Format of the sequence
        Returns:
            social_tracks (numpy array): Array of relevant tracks

        """
        social_track = np.empty((0, obs_len, len(raw_data_format)))

        # Timestamps in the sequence
        timestamps = np.unique(df["TIMESTAMP"].values)

        # Track groups
        df_group = df.groupby("TRACK_ID")
        for name, data in df_group:

            # Check if the track is long enough
            if len(data) < self.EXIST_THRESHOLD:
                continue

            # Skip if agent track
            if data["OBJECT_TYPE"].iloc[0] == "AGENT":
                continue

            # Check if the track is stationary
            if self.get_is_track_stationary(data):
                continue

            padded_track_data = self.pad_track(data, timestamps, obs_len,
                                                raw_data_format).reshape((1, obs_len, -1))
            
            social_track = np.vstack((social_track, padded_track_data))

        return social_track

    def get_is_front_or_back(
            self,
            agent_track: np.ndarray,
            neigh_x: float,
            neigh_y: float,
            raw_data_format: Dict[str, int],
    ):
        """Check if the neighbor is in front or back of the track.

        Args:
            track (numpy array): Track data
            neigh_x (float): Neighbor x coordinate
            neigh_y (float): Neighbor y coordinate
        Returns:
            _ (str): 'front' if in front, 'back' if in back

        """
        # We don't have heading information. So we need at least 2 coordinates to determine that.
        # Here, front and back is determined wrt to last 2 coordinates of the track
        x2 = agent_track[-1, raw_data_format["X"]]
        y2 = agent_track[-1, raw_data_format["Y"]]

        # Keep taking previous coordinate until first distinct coordinate is found.
        ind = agent_track.shape[0] - 2
        while ind > -1:
            x1 = agent_track[ind, raw_data_format["X"]]
            y1 = agent_track[ind, raw_data_format["Y"]]
            if x1 != x2 or y1 != y2:
                break
            ind -= 1

        # If all the coordinates in the track are the same, there's no way to find front/back
        if ind < 0:
            return None

        p1 = np.array([x1, y1])
        p2 = np.array([x2, y2])
        p3 = np.array([neigh_x, neigh_y])
        norm_distance = np.abs(np.cross(p2 - p1,p1 - p3)) / np.linalg.norm(p2 - p1)
        
        x0 = agent_track[0, raw_data_format["X"]]
        y0 = agent_track[0, raw_data_format["Y"]]
        p0 = np.array([x0, y0])
        
        # Interested in only those neighbors who are not far away from the direction of travel
        if norm_distance < FRONT_OR_BACK_OFFSET_THRESHOLD:

            distance_from_end = np.linalg.norm(p2 - p3)
            distance_from_start = np.linalg.norm(p0 - p3)
            distance_total = np.linalg.norm(p2 - p0)

            if distance_from_end < distance_from_start and distance_from_start > distance_total:
                return "front"
            
            else:
                return "back"

        else:
            return None

    def distance_and_num_neighbour(
            self,
            agent_track: np.ndarray,
            timestamps: np.ndarray,
            social_tracks: np.ndarray,
            obs_len: int,
            raw_data_format: Dict[str, int],
    ):
        """Get minimum distance of the tracks in front and in back.

        Args:
            agent_track (numpy array): Data for the agent track
            social_tracks (numpy array): Array of relevant tracks
            obs_len (int): Length of the observed trajectory
            raw_data_format (Dict): Format of the sequence
            viz (bool): Visualize tracks
        Returns:
            features (obs_len x 8):
                min_distance_front
                min_distance_back
                MIN_DISTANCE_FRONT_VEL
                MIN_DISTANCE_BACK_VEL
                num_neighbors
                NEIGHBORS_MEAN_VEL
                NEIGHBORS_MAX_VEL
                NEIGHBORS_MIN_VEL

        """
        min_distance = np.full((obs_len, 2), self.DEFAULT_MIN_DIST_FRONT_AND_BACK)
        min_distance_vel = np.full((obs_len, 2), 0)
        num_neighbor = np.full((obs_len, 1), 0)
        neighbor_vel = np.empty((obs_len, 3))
        
        social_coords = social_tracks[:,:,(raw_data_format["X"],raw_data_format["Y"])]
        social_vel = np.empty((social_coords.shape[0],social_coords.shape[1],1))
        
        for i in range(social_tracks.shape[0]):
            social_vel[i,:,:] = self.compute_velocity(social_coords[i,:,:],timestamps)
        
        agent_coords = agent_track[:,(raw_data_format["X"],raw_data_format["Y"])]
        agent_vel = self.compute_velocity(agent_coords,timestamps)
        
        # Compute distances for each timestep in the sequence
        for i in range(obs_len):

            # Agent coordinates
            agent_x = agent_track[i, raw_data_format["X"]]
            agent_y = agent_track[i, raw_data_format["Y"]]
            
            mean_vel = 0
            max_vel = 0
            min_vel = np.inf
            front_min_ind = -1
            back_min_ind = -1
            total_weight = 0
            # Compute distances for all the social tracks
            for j in range(social_tracks.shape[0]):

                social_track = social_tracks[j,i,:]
                neigh_x = social_track[raw_data_format["X"]]
                neigh_y = social_track[raw_data_format["Y"]]

                # Distance between agent and social
                distance = np.sqrt((agent_x - neigh_x)**2 +
                                           (agent_y - neigh_y)**2)

                # If not a neighbor, continue
                if distance <= self.NEARBY_DISTANCE_THRESHOLD:
                    # Add neighbour numbers
                    num_neighbor[i, 0] += 1
                    neigh_vel = social_vel[j,i,0]
                    mean_vel += neigh_vel/distance
                    total_weight += 1/distance
                    max_vel = max(max_vel,neigh_vel)
                    min_vel = min(min_vel,neigh_vel)
                
                # Check if the social track is in front or back
                position_state = self.get_is_front_or_back(
                    agent_track[:2, :] if i == 0 else agent_track[:i + 1, :],
                    neigh_x,
                    neigh_y,
                    raw_data_format,
                )
                if position_state == "front":
                    if distance < min_distance[i,0]:
                        min_distance[i, 0] = distance
                        front_min_ind = j

                elif position_state == "back":
                    if distance < min_distance[i,1]:
                        min_distance[i, 1] = distance
                        back_min_ind = j
                    
            if num_neighbor[i,0] == 0:
                neighbor_vel[i,:] = np.asarray([agent_vel[i],agent_vel[i],agent_vel[i]]).flatten()
            else:
                neighbor_vel[i,0] = mean_vel/total_weight
                neighbor_vel[i,1] = max_vel
                neighbor_vel[i,2] = min_vel
                
            if front_min_ind > -1:
                min_distance_vel[i,0] = agent_vel[i] - social_vel[front_min_ind,i,0]
            if back_min_ind > -1:
                min_distance_vel[i,1] = agent_vel[i] - social_vel[back_min_ind,i,0]

        
        features = np.concatenate((min_distance,min_distance_vel,num_neighbor,neighbor_vel),axis=1)
        return features

    def compute_rotation_angle(self,coords: np.ndarray):
        x = coords[:,0]
        y = coords[:,1]
        dx = np.zeros(x.shape)
        dy = np.zeros(y.shape)
        dx[1:] = x[1:] - x[:-1]
        dy[1:] = y[1:] - y[:-1]
        angle = np.arctan2(dy,dx)
        ind0 = np.nonzero(angle)[0][0]
        angle[ind0:] = angle[ind0:] - angle[ind0]
        scale = np.sign(angle)*(np.abs(angle)//np.pi)
        angle = angle - scale*np.pi*2
        
        return angle.reshape((len(x),1))
    
    def compute_social_features(
            self,
            df: pd.DataFrame,
            agent_track: np.ndarray,
            obs_len: int,
            total_len: int,
            raw_data_format: Dict[str, int],
    ):
        """Compute social features for the given sequence.

        Social features are meant to capture social context. 
        Here we use minimum distance to the vehicle in front, to the vehicle in back, 
        and number of neighbors as social features.

        Args:
            df (pandas Dataframe): Dataframe containing all the tracks in the sequence
            agent_track (numpy array): Data for the agent track
            obs_len (int): Length of observed trajectory
            seq_len (int): Length of the sequence
            raw_data_format (Dict): Format of the sequence
        Returns:
            social_features (numpy array): Social features for the agent track

        """
        agent_t = np.unique(np.sort(df['TIMESTAMP'].values))
        
        if agent_t.shape[0] == obs_len:
            df_obs = df
            agent_track_obs = agent_track
            obs_timestamps = agent_t
        else:
            df_obs = df[df['TIMESTAMP']<agent_t[obs_len]]
            agent_track_obs = agent_track[:obs_len,:]
            obs_timestamps = agent_t[:obs_len]
            
        neigh_track = self.track_filter(df_obs, obs_len,raw_data_format)
        obs_timestamps = obs_timestamps.astype('float')
        features = self.distance_and_num_neighbour(agent_track_obs,
                                                   obs_timestamps,
                                                   neigh_track,
                                                   obs_len,
                                                   raw_data_format)
        
        angle = self.compute_rotation_angle(agent_track_obs[:,(raw_data_format["X"],raw_data_format["Y"])])
        
        if agent_track.shape[0] == obs_len:
            social_features = np.concatenate((features,angle),axis=1)
        
        else:
            social_features = np.full((total_len,9),None)
            social_features[:obs_len,:] = np.concatenate((features,angle),axis=1)
            
        return social_features

#if __name__ == "__main__":
##    file_names = os.listdir("../train/data")
##    np.random.shuffle(file_names)
##    file_names = file_names[:100]
##    file_names = os.listdir('../forecasting_sample/data')
#    n_file = 1#len(file_names)
#    social_instance = SocialFeaturesUtils()
#    
##    start = time.time()
#    
#    for i in range(n_file):
#        file_path = "../train/data/"+'97269.csv'#file_names[i]
##        file_path = file_names
#        df = pd.read_csv(file_path, dtype={"TIMESTAMP": str})
#        agent_track = df[df["OBJECT_TYPE"] == "AGENT"].values
#        social_instance = SocialFeaturesUtils()
#        features = social_instance.compute_social_features(df,agent_track,20,50,RAW_DATA_FORMAT)
#        print(i)
#        print(features[:20,-1])
#    
##    end = time.time()
##    print(end-start)