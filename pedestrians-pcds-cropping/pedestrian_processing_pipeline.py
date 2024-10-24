import os
import pandas as pd
import numpy as np
from glob import glob
from itertools import groupby
from operator import itemgetter
import threading

import open3d as o3d

from dataset.loki_dataset import LOKIDatasetHandler
from processors.pointcloud_processor import PointCloudProcessor
from processors.pedestrian_processor import PedestrianProcessor
from visualization.visualizer import Visualizer


class PedestrianProcessingPipeline:
    """
    Encapsulates the workflow for processing and visualizing pedestrian point clouds.
    """

    def __init__(self, root_dir, csv_path, save_dir="saved_pedestrians", threshold_multiplier=0.5):
        """
        Initializes the processing pipeline.

        Args:
            root_dir (str): Root directory of the dataset.
            csv_path (str): Path to the CSV file containing pedestrian data.
            threshold_multiplier (float, optional): Multiplier for setting the minimum point threshold. Defaults to 0.5.
            save_dir (str, optional): Directory to save pedestrian point clouds. Defaults to "saved_pedestrians".
        """
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.threshold_multiplier = threshold_multiplier
        self.save_dir = save_dir

        # Initialize Handlers and Processors
        self.dataset_handler = LOKIDatasetHandler(root_dir=self.root_dir, keys=["pointcloud", "labels_3d"])
        self.pointcloud_processor = PointCloudProcessor()
        self.pedestrian_processor = PedestrianProcessor(points_threshold_multiplier=self.threshold_multiplier)
        self.visualizer = Visualizer()

        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            print(f"Created directory for saving pedestrian PCDs: {self.save_dir}")

    def load_scenario_frame_ids(self):
        """
        Loads unique scenario and frame IDs from the CSV file.

        Returns:
            tuple: Tuple containing arrays of scenario_ids and frame_ids.
        """
        print(f"Retrieving scenario & frame IDs with pedestrians from file {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        target_columns = df[['video_name', 'frame_name']]

        # Extract scenario IDs by removing the 'scenario_' prefix
        scenario_ids = target_columns['video_name'].apply(lambda x: x.split('_', 1)[1]).unique()
        # Extract frame IDs by removing the 'image_' prefix
        frame_ids = target_columns['frame_name'].apply(lambda x: x.split('_', 1)[1]).unique()

        print(f"Found {len(scenario_ids)} unique scenarios and {len(frame_ids)} unique frames.")
        return scenario_ids, frame_ids

    def verify_scenarios(self, all_scenario_ids):
        """
        Verifies the existence of scenario directories.

        Args:
            all_scenario_ids (array-like): Array of scenario IDs to verify.

        Returns:
            list: List of valid scenario IDs that exist.
        """
        missing_scenarios = []
        valid_scenarios = []

        for scenario_id in all_scenario_ids:
            if self._scenario_exists(scenario_id):
                valid_scenarios.append(scenario_id)
                # print(f"Scenario {scenario_id} exists.")
            else:
                missing_scenarios.append(int(scenario_id))  # Convert to int for processing ranges
                # print(f"Scenario {scenario_id} is missing.")

        if missing_scenarios:
            missing_ranges = self._group_consecutive_ids(sorted(missing_scenarios))
            self._print_missing_scenarios(missing_ranges)
        else:
            print("All scenarios exist.")

        return valid_scenarios

    def verify_frames(self, valid_scenario_ids, all_frame_ids):
        """
        Verifies the existence of frame files within valid scenarios.

        Args:
            valid_scenario_ids (list): List of valid scenario IDs.
            all_frame_ids (list): List of all frame IDs to verify.

        Returns:
            list: List of valid frame IDs that exist across all valid scenarios.
        """
        missing_frames = {}
        valid_frame_ids = set(all_frame_ids)  # Initialize with all frames

        for scenario_id in valid_scenario_ids:
            scenario_dir = os.path.join(self.root_dir, f'scenario_{scenario_id}')
            existing_frames = self._get_existing_frames(scenario_dir, all_frame_ids)
            missing = set(all_frame_ids) - existing_frames

            if missing:
                missing_frames[scenario_id] = sorted(missing)
                valid_frame_ids &= existing_frames  # Intersection to ensure frame exists across all scenarios
                print(f"Scenario {scenario_id}: Missing frames {sorted(missing)}")
            else:
                valid_frame_ids &= existing_frames
                print(f"Scenario {scenario_id}: All frames exist.")

        if missing_frames:
            self._print_missing_frames(missing_frames)
        else:
            print("All frames exist for the valid scenarios.")

        return sorted(valid_frame_ids)

    def process_all_frames_and_crop_pedestrians(self, valid_scenario_ids, valid_frame_ids):
        """
        Processes all valid scenarios and frames, crops pedestrian point clouds,
        and saves them asynchronously.

        Args:
            valid_scenario_ids (list): List of valid scenario IDs.
            valid_frame_ids (list): List of valid frame IDs.
        """
        for scenario_id in valid_scenario_ids:
            for frame_id in valid_frame_ids:
                print(f"\nProcessing Scenario: {scenario_id}, Frame: {frame_id}")

                # Retrieve sample
                raw_pcd, labels3d_ndarray = self.get_pcd_and_labels(scenario_id, frame_id)
                if raw_pcd is None or labels3d_ndarray is None:
                    continue

                # Extract Pedestrian DataFrame
                df_pedestrians = self.pedestrian_processor.extract_pedestrian_df(labels3d_ndarray)
                print(f"Extracted {len(df_pedestrians)} pedestrians in Scenario: {scenario_id}, Frame: {frame_id}")

                if df_pedestrians.empty:
                    print(f"No pedestrians found in Scenario: {scenario_id}, Frame: {frame_id}.")
                    continue

                # Preprocess Point Cloud
                cleaned_pcd = self.pointcloud_processor.preprocess_pcd(raw_pcd)
                print(f"Preprocessed point cloud for Scenario: {scenario_id}, Frame: {frame_id}")

                # Extract Pedestrian Point Clouds
                pedestrian_pcds = self.visualizer.extract_pedestrian_pcds(cleaned_pcd, df_pedestrians)
                print(f"Extracted {len(pedestrian_pcds)} pedestrian point clouds.")

                if not pedestrian_pcds:
                    print("No pedestrian point clouds extracted.")
                    continue

                # Calculate Average Number of Points and filter low-count pedestrians
                avg_points = self.pedestrian_processor.calculate_average_points(pedestrian_pcds)

                # Set Minimum Point Threshold
                min_threshold = self.pedestrian_processor.set_min_point_threshold(avg_points)

                # Filter Pedestrians Based on Threshold
                df_pedestrians_filtered, pedestrian_pcds_filtered = self.pedestrian_processor.filter_pedestrians(
                    df_pedestrians, pedestrian_pcds, min_threshold
                )

                # Recenter and prepare pedestrian PCDs
                pedestrian_pcd_dict = self._prepare_pedestrian_pcds(
                    scenario_id, frame_id, df_pedestrians_filtered, pedestrian_pcds_filtered
                )
                print(f"Prepared pedestrian PCD dictionary for Scenario: {scenario_id}, Frame: {frame_id}")

                # Save pedestrian PCDs asynchronously
                save_thread = threading.Thread(target=self.save_pedestrian_pcds, args=(pedestrian_pcd_dict,))
                save_thread.start()
                print(f"Started saving pedestrian PCDs for Scenario: {scenario_id}, Frame: {frame_id} asynchronously.")

        print("\nProcessing completed.")

    def get_pcd_and_labels(self, scenario_id, frame_id):
        # Retrieve sample
        try:
            sample = self.dataset_handler.get_sample_by_id(scenario_id, frame_id)
            print(f"Retrieved sample for Scenario: {scenario_id}, Frame: {frame_id}")
        except Exception as e:
            print(f"Error retrieving sample for Scenario: {scenario_id}, Frame: {frame_id}: {e}")
            return None, None

        if any(not v for v in sample.values()):
            print(f"Skipping Scenario: {scenario_id}, Frame: {frame_id} as no values were found.")
            return None, None

        # Extract Point Cloud and Labels
        raw_pcd = sample.get("pointcloud", [])[0]
        labels3d_ndarray = sample.get("labels_3d", [])[0]

        return raw_pcd, labels3d_ndarray

    def save_pedestrian_pcds(self, pcd_dict):
        """
        Saves the pedestrian point clouds in the provided dictionary as .ply files asynchronously.

        Args:
            pcd_dict (dict): Dictionary where keys are a combination of scenario_id, frame_id, and pedestrian_id,
                            and values are the cropped pedestrian point clouds.
        """
        for key, pcd in pcd_dict.items():
            # Construct the filename
            filename = f"{key}.ply"
            filepath = os.path.join(self.save_dir, filename)

            # Save the point cloud to a .ply file
            try:
                o3d.io.write_point_cloud(filepath, pcd)
                print(f"Saved pedestrian PCD: {filepath}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")

        print("All pedestrian point clouds for the current frame have been saved.")

    # ----------------------- Helper Methods -----------------------

    def _scenario_exists(self, scenario_id):
        """
        Checks if a scenario directory exists.

        Args:
            scenario_id (str): Scenario ID to check.

        Returns:
            bool: True if scenario exists, False otherwise.
        """
        scenario_dir = os.path.join(self.root_dir, f'scenario_{scenario_id}')
        return os.path.isdir(scenario_dir)

    @staticmethod
    def _group_consecutive_ids(id_list):
        """
        Groups consecutive IDs into ranges.

        Args:
            id_list (list): Sorted list of integer IDs.

        Returns:
            list: List of grouped ID ranges as strings.
        """
        ranges = []
        for k, g in groupby(enumerate(id_list), lambda x: x[0] - x[1]):
            group = list(map(itemgetter(1), g))
            if len(group) > 1:
                ranges.append(f"{str(group[0]).zfill(3)}-{str(group[-1]).zfill(3)}")
            else:
                ranges.append(str(group[0]).zfill(3))
        return ranges

    def _print_missing_scenarios(self, missing_ranges):
        """
        Prints the missing scenarios in grouped ranges.

        Args:
            missing_ranges (list): List of missing scenario ID ranges.
        """
        if len(missing_ranges) == 1 and '-' in missing_ranges[0]:
            print(f"Skipping the following scenarios because they do not exist in {self.root_dir}:")
            print(f"{missing_ranges[0]}")
        else:
            print(f"Skipping the following scenarios because they do not exist in {self.root_dir}:")
            print(", ".join(missing_ranges))

    @staticmethod
    def _get_existing_frames(scenario_dir, frame_ids):
        """
        Retrieves existing frame IDs within a scenario directory.

        Args:
            scenario_dir (str): Path to the scenario directory.
            frame_ids (list): List of frame IDs to check.

        Returns:
            set: Set of existing frame IDs.
        """
        existing_frames = set()
        for frame_id in frame_ids:
            # Check if any file with the pattern *_XXXX (frame_id) exists
            frame_files = glob(os.path.join(scenario_dir, f"*_{frame_id}.*"))
            if frame_files:
                existing_frames.add(frame_id)
        return existing_frames

    def _print_missing_frames(self, missing_frames):
        """
        Prints the missing frames for each scenario.

        Args:
            missing_frames (dict): Dictionary mapping scenario IDs to lists of missing frame IDs.
        """
        for scenario_id, frames in missing_frames.items():
            grouped_frames = self._group_consecutive_ids([int(f) for f in frames])
            print(f"Skipping the following frames in scenario {scenario_id} because they do not exist:")
            print(", ".join(grouped_frames))

    @staticmethod
    def _prepare_pedestrian_pcds(scenario_id, frame_id, df_pedestrians_filtered, pedestrian_pcds_filtered):
        """
        Prepares the pedestrian PCDs by recentering and creating a dictionary.

        Args:
            scenario_id (str): Scenario ID.
            frame_id (str): Frame ID.
            df_pedestrians_filtered (pd.DataFrame): Filtered pedestrian DataFrame.
            pedestrian_pcds_filtered (list): List of filtered pedestrian point clouds.

        Returns:
            dict: Dictionary with unique keys and pedestrian PCDs.
        """
        pedestrian_pcd_dict = {}
        for idx, pcd in enumerate(pedestrian_pcds_filtered):
            pedestrian_id = df_pedestrians_filtered.iloc[idx].get("Ped_ID", idx)

            # Recenter the point cloud (optional based on requirements)
            points = np.asarray(pcd.points)
            centroid = np.mean(points, axis=0)
            points_recentered = points - centroid
            pcd.points = o3d.utility.Vector3dVector(points_recentered)

            # Create a unique key for each pedestrian (scenario_id_frame_id_pedestrian_id)
            pedestrian_key = f"{scenario_id}_{frame_id}_ped_{pedestrian_id}"
            pedestrian_pcd_dict[pedestrian_key] = pcd

        return pedestrian_pcd_dict
