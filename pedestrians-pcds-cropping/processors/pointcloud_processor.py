from utilities.convert import convert_from_vertex_to_open3d_pcd
import open3d as o3d


class PointCloudProcessor:
    """
    Handles preprocessing of point clouds, including conversion, downsampling, and outlier removal.
    """

    def __init__(self, voxel_size=0.02, nb_neighbors=20, std_ratio=2.0):
        """
        Initializes the PointCloudProcessor with specified parameters.

        Args:
            voxel_size (float, optional): Voxel size for downsampling. Defaults to 0.02.
            nb_neighbors (int, optional): Number of neighbors for statistical outlier removal. Defaults to 20.
            std_ratio (float, optional): Standard deviation ratio for statistical outlier removal. Defaults to 2.0.
        """
        self.voxel_size = voxel_size
        self.nb_neighbors = nb_neighbors
        self.std_ratio = std_ratio

    def preprocess_pcd(self, raw_pcd):
        """
        Preprocesses the raw point cloud by converting, downsampling, and removing outliers.

        Args:
            raw_pcd (Any): The raw point cloud data.

        Returns:
            o3d.geometry.PointCloud: The cleaned and downsampled point cloud.

        Raises:
            SystemExit: If the point cloud cannot be converted.
        """
        # Convert pointcloud to Open3D PointCloud object
        try:
            pcd = convert_from_vertex_to_open3d_pcd(raw_pcd)
        except ValueError as ve:
            print(f"Error converting point cloud: {ve}")
            exit(1)

        print("Downsampling the point cloud...")
        pcd_down = pcd.voxel_down_sample(voxel_size=self.voxel_size)

        print("Removing statistical outliers...")
        cl, ind = pcd_down.remove_statistical_outlier(nb_neighbors=self.nb_neighbors, std_ratio=self.std_ratio)
        pcd_clean = pcd_down.select_by_index(ind)

        return pcd_clean
