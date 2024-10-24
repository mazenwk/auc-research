import pandas as pd
import numpy as np


class PedestrianProcessor:
    """
    Processes pedestrian data, including extraction, averaging, thresholding, and filtering.
    """

    def __init__(self, minimum_point_threshold=10, points_threshold_multiplier=0.5):
        """
        Initializes the PedestrianProcessor with a threshold multiplier.

        Args: points_threshold_multiplier (float, optional): Multiplier to set the minimum point threshold based on
        the average Defaults to 0.5.
        """
        self.minimum_point_threshold = minimum_point_threshold
        self.points_threshold_multiplier = points_threshold_multiplier
        self.column_names = [
            "labels", "track_id", "stationary", "pos_x", "pos_y", "pos_z",
            "dim_x", "dim_y", "dim_z", "yaw", "vehicle_state",
            "intended_actions", "potential_destination", "additional_info"
        ]
        self.numerical_columns = ["pos_x", "pos_y", "pos_z", "dim_x", "dim_y", "dim_z", "yaw"]

    def extract_pedestrian_df(self, labels3d_ndarray):
        """
        Extracts pedestrian information from the labels ndarray and returns a DataFrame.

        Args:
            labels3d_ndarray (numpy.ndarray): The labels' data.

        Returns:
            pd.DataFrame: DataFrame containing pedestrian information.

        Raises:
            SystemExit: If the labels ndarray has an unexpected shape or no pedestrians are found.
        """
        expected_num_columns = len(self.column_names)
        if labels3d_ndarray.ndim == 2 and labels3d_ndarray.shape[1] >= expected_num_columns:
            df_labels3d = pd.DataFrame(labels3d_ndarray[:, :expected_num_columns], columns=self.column_names)
        else:
            print(f"labels3d_ndarray has an unexpected shape: {labels3d_ndarray.shape}")
            exit(1)

        # Ensure numerical columns are correctly typed
        for col in self.numerical_columns:
            df_labels3d[col] = pd.to_numeric(df_labels3d[col], errors='coerce')

        # Filter to include only pedestrians
        df_pedestrians = df_labels3d[df_labels3d['labels'] == 'Pedestrian'].reset_index(drop=True)

        if df_pedestrians.empty:
            print("No pedestrian data found in this sample.")
            exit(1)

        return df_pedestrians

    def calculate_average_points(self, pedestrian_pcds):
        """
        Calculates the average number of points per pedestrian after removing those with very low point counts.

        Args:
            pedestrian_pcds (list of o3d.geometry.PointCloud): List of pedestrian point clouds.

        Returns:
            float: Average number of points per pedestrian after filtering.
        """
        # Filter out pedestrians with points less than min_point_count
        filtered_pcds = [pcd for pcd in pedestrian_pcds if len(np.asarray(pcd.points)) >= self.minimum_point_threshold]
        removed = len(pedestrian_pcds) - len(filtered_pcds)

        if removed > 0:
            print(f"Removed {removed} pedestrians with fewer than {self.minimum_point_threshold} points.")

        # Calculate average number of points from filtered pedestrians
        counts = [len(np.asarray(pcd.points)) for pcd in filtered_pcds]
        avg = sum(counts) / len(counts) if counts else 0

        print(f"Calculated average points per pedestrian: {avg:.2f}")

        return avg

    def set_min_point_threshold(self, average_points):
        """
        Sets the minimum point threshold based on the average points and multiplier.

        Args:
            average_points (float): The average number of points per pedestrian.

        Returns:
            float: The calculated minimum point threshold.
        """
        min_threshold = average_points * self.points_threshold_multiplier
        print(f"Minimum point threshold set to: {min_threshold:.2f} (Multiplier: {self.points_threshold_multiplier})")
        return min_threshold

    def filter_pedestrians(self, df_pedestrians, pedestrian_pcds, min_threshold):
        """
        Filters pedestrians based on the minimum point threshold.

        Args:
            df_pedestrians (pd.DataFrame): DataFrame containing pedestrian data.
            pedestrian_pcds (list of o3d.geometry.PointCloud): List of pedestrian point clouds.
            min_threshold (float): The minimum number of points required to keep a pedestrian.

        Returns:
            tuple:
                pd.DataFrame: Filtered DataFrame of pedestrians.
                list of o3d.geometry.PointCloud: Filtered list of pedestrian point clouds.
        """
        print(f"Filtering pedestrians with fewer than {min_threshold:.2f} points.")

        # Identify indices of pedestrians to keep
        indices_to_keep = [idx for idx, pcd in enumerate(pedestrian_pcds) if len(np.asarray(pcd.points)) >= min_threshold]

        # Filter DataFrame and point clouds
        df_filtered = df_pedestrians.iloc[indices_to_keep].reset_index(drop=True)
        pcds_filtered = [pedestrian_pcds[idx] for idx in indices_to_keep]

        print(f"Filtered down to {len(pcds_filtered)} pedestrians after applying threshold.")

        return df_filtered, pcds_filtered
