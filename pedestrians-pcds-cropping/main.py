from dataset.dataset_manager import DatasetManager
from processors.pointcloud_processor import PointCloudProcessor
from processors.pedestrian_processor import PedestrianProcessor
from visualization.visualizer import Visualizer

# Configuration Parameters
ROOT_DIR = "sample/"
SAMPLE_IDX = 0
THRESHOLD_MULTIPLIER = 0.5


def main():
    """
    Executes the data processing and visualization workflow.
    """

    # Initialize Managers and Processors
    dataset_manager = DatasetManager(root_dir=ROOT_DIR)
    pointcloud_processor = PointCloudProcessor()
    pedestrian_processor = PedestrianProcessor(threshold_multiplier=THRESHOLD_MULTIPLIER)
    visualizer = Visualizer()

    # Retrieve Sample
    sample = dataset_manager.retrieve_sample(SAMPLE_IDX)

    # Extract Point Cloud and Labels
    raw_pcd = sample.get("pointcloud", [])[0]
    labels3d_ndarray = sample.get("labels_3d", [])[0]

    # Preprocess Point Cloud
    cleaned_pcd = pointcloud_processor.preprocess_pcd(raw_pcd)

    # Extract Pedestrian DataFrame
    df_pedestrians = pedestrian_processor.extract_pedestrian_df(labels3d_ndarray)

    # Initialize Visualizer and Add Initial Geometries
    visualizer.add_point_cloud(cleaned_pcd)
    visualizer.add_bounding_boxes(df_pedestrians, color=[0, 0, 1])
    visualizer.add_coordinate_axes(size=5.0, origin=[0, 0, 0])

    # Extract Pedestrian Point Clouds
    pedestrian_pcds = visualizer.extract_pedestrian_pcds(cleaned_pcd, df_pedestrians)

    if not pedestrian_pcds:
        print("No pedestrian point clouds extracted.")
        exit(1)

    # Calculate Average Number of Points
    avg_points = pedestrian_processor.calculate_average_points(pedestrian_pcds)

    # Set Minimum Point Threshold
    min_threshold = pedestrian_processor.set_min_point_threshold(avg_points)

    # Filter Pedestrians Based on Threshold
    df_pedestrians_filtered, pedestrian_pcds_filtered = pedestrian_processor.filter_pedestrians(
        df_pedestrians, pedestrian_pcds, min_threshold
    )

    # Update Visualizer with Filtered Data
    visualizer.clear_bounding_boxes()  # Remove existing bounding boxes
    print("Updating visualizer...")
    visualizer.add_bounding_boxes(df_pedestrians_filtered, color=[0, 0, 1])  # Add filtered bounding boxes
    visualizer.pedestrian_pcds = pedestrian_pcds_filtered  # Update pedestrian PCDs

    # Visualize Filtered Pedestrians
    print("Visualizing...")
    visualizer.visualize_pedestrians_only(window_name='Cropped Pedestrians', color=[1, 0, 0])


if __name__ == "__main__":
    main()
