import pandas as pd


class PedestrianProcessor:
    """
    Processes pedestrian data, including extraction, averaging, thresholding, and filtering.
    """

    def __init__(self, threshold_multiplier=0.5):
        """
        Initializes the PedestrianProcessor with a threshold multiplier.

        Args:
            threshold_multiplier (float, optional): Multiplier to set the minimum point threshold based on the average
            Defaults to 0.5.
        """
        self.threshold_multiplier = threshold_multiplier
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

    @staticmethod
    def calculate_average_points(pedestrian_pcds):
        """
        Calculates the average number of points across all pedestrian point clouds.

        Args:
            pedestrian_pcds (list of o3d.geometry.PointCloud): List of pedestrian PCDs.

        Returns:
            float: The average number of points per pedestrian PCD.
        """
        total_points = sum(len(pcd_ped.points) for pcd_ped in pedestrian_pcds)
        avg_points = total_points / len(pedestrian_pcds) if pedestrian_pcds else 0
        print(f"Average number of points per pedestrian PCD: {avg_points:.2f}")
        return avg_points

    def set_min_point_threshold(self, avg_points):
        """
        Sets the minimum point threshold based on the average number of points.

        Args:
            avg_points (float): The average number of points per pedestrian PCD.

        Returns:
            float: The minimum point threshold.
        """
        min_threshold = avg_points * self.threshold_multiplier
        print(f"Minimum point threshold set to: {min_threshold:.2f}")
        return min_threshold

    @staticmethod
    def filter_pedestrians(df_pedestrians, pedestrian_pcds, min_threshold):
        """
        Filters out pedestrians with point counts below the minimum threshold.

        Args:
            df_pedestrians (pd.DataFrame): DataFrame containing pedestrian information.
            pedestrian_pcds (list of o3d.geometry.PointCloud): List of pedestrian PCDs.
            min_threshold (float): The minimum number of points required.

        Returns:
            tuple:
                pd.DataFrame: Filtered pedestrian DataFrame.
                list of o3d.geometry.PointCloud: Filtered list of pedestrian PCDs.

        Raises:
            SystemExit: If no pedestrians meet the threshold.
        """
        pedestrian_pcds_filtered = []
        df_pedestrians_filtered = pd.DataFrame(columns=df_pedestrians.columns)

        for (idx, ped_data), pcd_ped in zip(df_pedestrians.iterrows(), pedestrian_pcds):
            point_count = len(pcd_ped.points)
            if point_count >= min_threshold:
                pedestrian_pcds_filtered.append(pcd_ped)
                df_pedestrians_filtered = pd.concat([df_pedestrians_filtered, ped_data.to_frame().T], ignore_index=True)
            else:
                print(f"Removing pedestrian {ped_data['track_id']} with only {point_count} points.")

        if not pedestrian_pcds_filtered:
            print("No pedestrians meet the minimum point threshold.")
            exit(1)

        print(f"Number of pedestrians after filtering: {len(pedestrian_pcds_filtered)}")
        return df_pedestrians_filtered, pedestrian_pcds_filtered
