import open3d as o3d
import pandas as pd
from .bounding_box import BoundingBox


class Visualizer:
    """
    Handles visualization of point clouds and bounding boxes using Open3D.
    """

    def __init__(self):
        """
        Initializes the Visualizer.
        """
        self.geometries = []

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
            bbox = BoundingBox(row)
            if bbox.obb is not None:
                bbox.obb.color = color
                self.geometries.append(bbox.obb)

    def add_coordinate_axes(self, size=5.0, origin=[0, 0, 0]):
        """
        Adds coordinate axes to the visualization.

        Args:
            size (float, optional): Size of the coordinate frame. Defaults to 5.0.
            origin (list, optional): Origin point of the coordinate frame. Defaults to [0, 0, 0].
        """
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
        self.geometries.append(axes)

    def visualize(self, window_name='Visualization', width=1280, height=720):
        """
        Launches the Open3D visualization window.

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
