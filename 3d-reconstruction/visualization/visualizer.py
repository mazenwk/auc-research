# visualization/visualizer.py

import open3d as o3d
from .bounding_box import BoundingBox
import pandas as pd


class Visualizer:
    """
    Handles visualization of point clouds and bounding boxes using Open3D.
    """

    def __init__(self):
        """
        Initializes the Visualizer.
        """
        self.geometries = []
        self.pedestrian_pcds = []  # To store cropped pedestrian point clouds

    def add_point_cloud(self, pcd):
        """
        Adds a point cloud to the visualization.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud object.
        """
        self.geometries.append(pcd)

    def add_bounding_boxes(self, df_pedestrians, color=[0, 0, 1]):
        """
        Adds pedestrian bounding boxes to the visualization.

        Args:
            df_pedestrians (pd.DataFrame): DataFrame containing pedestrian bounding box information.
            color (list, optional): RGB color for the bounding boxes. Defaults to Blue ([0, 0, 1]).
        """
        for idx, row in df_pedestrians.iterrows():
            bbox_obj = BoundingBox(row)
            obb = bbox_obj.obb
            if obb is not None:
                obb.color = color
                self.geometries.append(obb)

    def add_coordinate_axes(self, size=5.0, origin=[0, 0, 0]):
        """
        Adds coordinate axes to the visualization.

        Args:
            size (float, optional): Size of the coordinate frame. Defaults to 5.0.
            origin (list, optional): Origin point of the coordinate frame. Defaults to [0, 0, 0].
        """
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        self.geometries.append(axes)

    def extract_pedestrian_pcds(self, pcd, df_pedestrians):
        """
        Extracts pedestrian-specific point clouds by cropping the original point cloud
        based on the bounding boxes.

        Args:
            pcd (o3d.geometry.PointCloud): The original point cloud.
            df_pedestrians (pd.DataFrame): DataFrame containing pedestrian bounding box information.

        Returns:
            list of o3d.geometry.PointCloud: List of pedestrian-specific point clouds.
        """
        pedestrian_pcds = []
        for idx, row in df_pedestrians.iterrows():
            bbox = BoundingBox(row).obb
            if bbox is not None:
                # Use the 'crop' method of PointCloud
                cropped_pcd = pcd.crop(bbox)
                pedestrian_pcds.append(cropped_pcd)
        self.pedestrian_pcds = pedestrian_pcds
        return pedestrian_pcds

    def visualize(self, window_name='Visualization', width=1280, height=720):
        """
        Launches the Open3D visualization window with all geometries.

        Args:
            window_name (str, optional): Title of the visualization window. Defaults to 'Visualization'.
            width (int, optional): Width of the window. Defaults to 1280.
            height (int, optional): Height of the window. Defaults to 720.
        """
        o3d.visualization.draw_geometries(
            self.geometries,
            window_name=window_name,
            width=width,
            height=height,
            left=50,
            top=50,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=False
        )

    def visualize_pedestrians_only(self, window_name='Pedestrians Only', width=1280, height=720, color=[1, 0, 0]):
        """
        Visualizes only the pedestrian point clouds.

        Args:
            window_name (str, optional): Title of the visualization window. Defaults to 'Pedestrians Only'.
            width (int, optional): Width of the window. Defaults to 1280.
            height (int, optional): Height of the window. Defaults to 720.
            color (list, optional): RGB color to assign to all pedestrian point clouds. Defaults to Red ([1, 0, 0]).
        """
        pedestrian_geometries = []
        for pcd in self.pedestrian_pcds:
            # Optionally, assign a color to the pedestrian pcd
            colored_pcd = pcd.paint_uniform_color(color)
            pedestrian_geometries.append(colored_pcd)

        # Add coordinate axes
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])
        pedestrian_geometries.append(axes)

        # Visualize
        o3d.visualization.draw_geometries(
            pedestrian_geometries,
            window_name=window_name,
            width=width,
            height=height,
            left=50,
            top=50,
            point_show_normal=False,
            mesh_show_wireframe=False,
            mesh_show_back_face=False
        )
