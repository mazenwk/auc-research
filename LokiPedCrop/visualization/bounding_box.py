import numpy as np
import open3d as o3d
import pandas as pd


class BoundingBox:
    """
    Represents a bounding box in 3D space.
    """

    def __init__(self, row):
        """
        Initializes the BoundingBox object from a DataFrame row.

        Args:
            row (pd.Series): A row from the DataFrame containing bounding box info.
        """
        self.row = row
        self.obb = self.create_oriented_bounding_box()

    def create_oriented_bounding_box(self):
        """
        Creates an Oriented Bounding Box (OBB) for the pedestrian.

        Returns:
            o3d.geometry.OrientedBoundingBox or None: The oriented bounding box or None if data is invalid.
        """
        required_fields = ['pos_x', 'pos_y', 'pos_z', 'dim_x', 'dim_y', 'dim_z', 'yaw']
        for field in required_fields:
            if pd.isnull(self.row[field]):
                print(f"Missing field '{field}' in row {self.row.name}. Skipping this bounding box.")
                return None

        # Center position
        center = np.array([self.row['pos_x'], self.row['pos_y'], self.row['pos_z']])

        # Dimensions
        extent = np.array([self.row['dim_x'], self.row['dim_y'], self.row['dim_z']])

        # Yaw angle (rotation around Z-axis)
        yaw = self.row['yaw']

        # Create rotation matrix around Z-axis
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        R = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])

        # Create the Oriented Bounding Box
        obb = o3d.geometry.OrientedBoundingBox(center, R, extent)

        return obb
