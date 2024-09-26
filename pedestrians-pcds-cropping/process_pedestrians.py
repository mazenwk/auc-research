# scripts/process_pedestrians.py

import sys
import argparse
from dataset.dataset_manager import DatasetManager
from processors.pointcloud_processor import PointCloudProcessor
from processors.pedestrian_processor import PedestrianProcessor
from visualization.visualizer import Visualizer


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process and visualize pedestrian point clouds.")
    parser.add_argument(
        '--root_dir',
        type=str,
        default='sample/',
        help='Root directory of the dataset. Defaults to "sample/".'
    )
    parser.add_argument(
        '--sample_idx',
        type=int,
        default=0,
        help='Index of the sample to process. Defaults to 0.'
    )
    parser.add_argument(
        '--threshold_multiplier',
        type=float,
        default=0.5,
        help='Multiplier for setting the minimum point threshold based on the average. Defaults to 0.5.'
    )
    return parser.parse_args()


def process_pedestrians(root_dir, sample_idx, threshold_multiplier):
    """
    Executes the pedestrian processing and visualization workflow.

    Args:
        root_dir (str): Root directory of the dataset.
        sample_idx (int): Index of the sample to process.
        threshold_multiplier (float): Multiplier for setting the minimum point threshold.
    """
    # Initialize Managers and Processors
    dataset_manager = DatasetManager(root_dir=root_dir)
    pointcloud_processor = PointCloudProcessor()
    pedestrian_processor = PedestrianProcessor(threshold_multiplier=threshold_multiplier)
    visualizer = Visualizer()

    # Retrieve Sample
    sample = dataset_manager.retrieve_sample(sample_idx)

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
        sys.exit(1)

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
    args = parse_arguments()
    process_pedestrians(args.root_dir, args.sample_idx, args.threshold_multiplier)
