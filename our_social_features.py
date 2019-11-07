#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module is used for computing social features for motion forecasting baselines."""

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from baseline_config import (
    PADDING_TYPE,
    STATIONARY_THRESHOLD,
    VELOCITY_THRESHOLD,
    EXIST_THRESHOLD,
    DEFAULT_MIN_DIST_FRONT_AND_BACK,
    NEARBY_DISTANCE_THRESHOLD,
    FRONT_OR_BACK_OFFSET_THRESHOLD,
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

    def compute_velocity(self, track_df: pd.DataFrame) -> List[float]:
        """Compute velocities for the given track.

        Args:
            track_df (pandas Dataframe): Data for the track
        Returns:
            vel (list of float): Velocity at each timestep

        """
        x = track_df["X"].values
        y = track_df["Y"].values
        t = track_df["TIMESTAMP"].values.astype('float')
        
        dt = t[1:]-t[:-1]
        vx = (x[1:]-x[:-1])/dt
        vy = (y[1:]-y[:-1])/dt
        velocity = np.sqrt(np.square(vx)+np.square(vy))

        return velocity

    def get_is_track_stationary(self, track_df: pd.DataFrame) -> bool:
        """Check if the track is stationary.

        Args:
            track_df (pandas Dataframe): Data for the track
        Return:
            _ (bool): True if track is stationary, else False 

        """
        velocity = self.compute_velocity(track_df)
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
    ) -> np.ndarray:
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
    ) -> np.ndarray:
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
                      raw_data_format: Dict[str, int]) -> np.ndarray:
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
    ) -> Optional[str]:
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
            social_tracks: np.ndarray,
            obs_len: int,
            raw_data_format: Dict[str, int],
    ) -> np.ndarray:
        """Get minimum distance of the tracks in front and in back.

        Args:
            agent_track (numpy array): Data for the agent track
            social_tracks (numpy array): Array of relevant tracks
            obs_len (int): Length of the observed trajectory
            raw_data_format (Dict): Format of the sequence
            viz (bool): Visualize tracks
        Returns:
            min_distance_front_and_back (numpy array): obs_len x 2, minimum front and back distances
            num_neighbors (numpy array): Number of neighbors at each timestep

        """
        min_distance = np.full((obs_len, 2), self.DEFAULT_MIN_DIST_FRONT_AND_BACK)
        num_neighbor = np.full((obs_len, 1), 0)
        
        # Compute distances for each timestep in the sequence
        for i in range(obs_len):

            # Agent coordinates
            agent_x = agent_track[i, raw_data_format["X"]]
            agent_y = agent_track[i, raw_data_format["Y"]]

            # Compute distances for all the social tracks
            for social_track in social_tracks[:, i, :]:

                neigh_x = social_track[raw_data_format["X"]]
                neigh_y = social_track[raw_data_format["Y"]]

                # Distance between agent and social
                distance = np.sqrt((agent_x - neigh_x)**2 +
                                           (agent_y - neigh_y)**2)

                # If not a neighbor, continue
                if distance > self.NEARBY_DISTANCE_THRESHOLD:
                    continue
                
                # Add neighbour numbers
                num_neighbor[i, 0] += 1
                
                # Check if the social track is in front or back
                position_state = self.get_is_front_or_back(
                    agent_track[:2, :] if i == 0 else agent_track[:i + 1, :],
                    neigh_x,
                    neigh_y,
                    raw_data_format,
                )
                if position_state == "front":
                    min_distance[i, 0] = min(min_distance[i, 0], distance)

                elif position_state == "back":
                    min_distance[i, 1] = min(min_distance[i, 1], distance)

        return min_distance,num_neighbor

    def compute_social_features(
            self,
            df: pd.DataFrame,
            agent_track: np.ndarray,
            obs_len: int,
            total_len: int,
            raw_data_format: Dict[str, int],
    ) -> np.ndarray:
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
        else:
            df_obs = df[df['TIMESTAMP']<agent_t[obs_len]]
            agent_track_obs = agent_track[:obs_len,:]
            
        neigh_track = self.track_filter(df_obs, obs_len,raw_data_format)
        
        distance_obs,num_neigh_obs = self.distance_and_num_neighbour(agent_track_obs,
                                                                     neigh_track,
                                                                     obs_len,
                                                                     raw_data_format)
        
        if agent_track.shape[0] == obs_len:
            social_features = np.concatenate((distance_obs,num_neigh_obs),axis=1)
        
        else:
            social_features = np.full((total_len,3),None)
            social_features[:obs_len,:] = np.concatenate((distance_obs,num_neigh_obs),axis=1)
            
        return social_features
