from dataset.loki_dataset import LOKIDatasetHandler
from utilities.convert import convert_from_vertex_to_open3d_pcd
from visualization.visualizer import Visualizer

import pandas as pd


def main():
    # Hard-coded parameters
    root_dir = "sample/"
    sample_idx = 0

    # Initialize the dataset handler
    try:
        dataset_handler = LOKIDatasetHandler(root_dir=root_dir, keys=["pointcloud", "labels_3d"])
    except ValueError as ve:
        print(f"Error initializing dataset: {ve}")
        exit(1)

    # Check dataset length
    if len(dataset_handler) == 0:
        print("The dataset is empty.")
        exit(1)

    # Retrieve the specified sample
    try:
        sample = dataset_handler.get_sample(sample_idx)
    except IndexError as ie:
        print(f"Error retrieving sample: {ie}")
        exit(1)

    # Extract pointcloud and labels3d
    pointcloud = sample.get("pointcloud", [])[0]
    labels3d_ndarray = sample.get("labels_3d", [])[0]

    # Convert pointcloud to Open3D PointCloud object
    try:
        pcd = convert_from_vertex_to_open3d_pcd(pointcloud)
    except ValueError as ve:
        print(f"Error converting point cloud: {ve}")
        exit(1)

    # Define column names based on your data structure
    column_names = [
        "labels", "track_id", "stationary", "pos_x", "pos_y", "pos_z",
        "dim_x", "dim_y", "dim_z", "yaw", "vehicle_state",
        "intended_actions", "potential_destination", "additional_info"
    ]

    # Check if labels3d_ndarray has the correct number of columns
    expected_num_columns = len(column_names)
    if labels3d_ndarray.ndim == 2 and labels3d_ndarray.shape[1] >= expected_num_columns:
        # Slice the ndarray to match the number of columns if it has extra columns
        df_labels3d = pd.DataFrame(labels3d_ndarray[:, :expected_num_columns], columns=column_names)
    else:
        print(f"labels3d_ndarray has an unexpected shape: {labels3d_ndarray.shape}")
        exit(1)

    # Ensure numerical columns are correctly typed
    numerical_columns = ["pos_x", "pos_y", "pos_z", "dim_x", "dim_y", "dim_z", "yaw"]
    for col in numerical_columns:
        df_labels3d[col] = pd.to_numeric(df_labels3d[col], errors='coerce')

    # Filter to include only pedestrians
    df_pedestrians = df_labels3d[df_labels3d['labels'] == 'Pedestrian'].reset_index(drop=True)

    if df_pedestrians.empty:
        print("No pedestrian data found in this sample.")
        exit(1)

    visualizer = Visualizer()
    visualizer.add_point_cloud(pcd)
    visualizer.add_bounding_boxes(df_pedestrians, color=[0, 0, 1])
    visualizer.add_coordinate_axes(size=5.0, origin=[0, 0, 0])
    pedestrian_pcds = visualizer.extract_pedestrian_pcds(pcd, df_pedestrians)
    visualizer.visualize_pedestrians_only(window_name='Cropped Pedestrians', color=[1, 0, 0])


if __name__ == "__main__":
    main()
