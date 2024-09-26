import os
import json
import torch
import plyfile
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from torch.utils.data import Dataset


class LOKIDatasetHandler:
    """
    Handles loading and accessing samples from the LOKI dataset.
    """

    def __init__(self, root_dir, keys=["pointcloud", "labels_3d"]):
        """
        Initializes the dataset handler.

        Args:
            root_dir (str): Root directory of the LOKI dataset.
            keys (list, optional): Keys to load from the dataset. Defaults to ["pointcloud", "labels_3d"].
        """
        self.root_dir = root_dir
        self.keys = keys
        self.dataset = self._initialize_dataset()

    def _initialize_dataset(self):
        """
        Initializes the LOKIDataset.

        Returns:
            LOKIDataset: Initialized dataset object.
        """
        if not os.path.isdir(self.root_dir):
            raise ValueError(f"Provided root_dir '{self.root_dir}' is not a valid directory.")
        return LOKIDataset(root_dir=self.root_dir, keys=self.keys)

    def get_sample(self, index):
        """
        Retrieves a sample from the dataset by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            dict: Sample containing pointcloud and labels_3d data.
        """
        if index < 0 or index >= len(self.dataset):
            raise IndexError(f"Sample index {index} out of range. Dataset size: {len(self.dataset)}.")
        return self.dataset[index]

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset)


class LOKIDataset(Dataset):
    def __init__(self, root_dir, keys=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the scenarios.
            keys (list of strings): List of keys to specify which data to load.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.keys = (
            keys
            if keys is not None
            else ["odometry", "labels_2d", "labels_3d", "pointcloud", "images", "map"]
        )
        self.transform = transform
        self.scenarios = [
            os.path.join(root_dir, scenario) for scenario in os.listdir(root_dir)
        ]

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        scenario_path = self.scenarios[idx]
        sample = {}

        if "odometry" in self.keys:
            sample["odometry"] = self.load_odometry(scenario_path)
        if "labels_2d" in self.keys:
            sample["labels_2d"] = self.load_labels_2d(scenario_path)
        if "labels_3d" in self.keys:
            sample["labels_3d"] = self.load_labels_3d(scenario_path)
        if "pointcloud" in self.keys:
            sample["pointcloud"] = self.load_pointcloud(scenario_path)
        if "images" in self.keys:
            sample["images"] = self.load_images(scenario_path)
        if "map" in self.keys:
            sample["map"] = self.load_map(scenario_path)

        if self.transform and "images" in sample:
            sample["images"] = [self.transform(image) for image in sample["images"]]

        return sample

    def load_odometry(self, scenario_path):
        odometry_files = sorted(glob(os.path.join(scenario_path, "odom_*.txt")))
        odometry_data = [
            pd.read_csv(f, header=None, dtype=float).values for f in odometry_files
        ]
        return odometry_data

    def load_labels_2d(self, scenario_path):
        label2d_files = sorted(glob(os.path.join(scenario_path, "label2d_*.json")))
        labels_2d = [json.load(open(f)) for f in label2d_files]
        return labels_2d

    def load_labels_3d(self, scenario_path):
        label3d_files = sorted(glob(os.path.join(scenario_path, "label3d_*.txt")))
        labels_3d = [pd.read_csv(f).values for f in label3d_files]
        return labels_3d

    def load_pointcloud(self, scenario_path):
        pointcloud_files = sorted(glob(os.path.join(scenario_path, "pc_*.ply")))
        pointcloud_data = [self.load_ply(f) for f in pointcloud_files]
        return pointcloud_data

    def load_images(self, scenario_path):
        image_files = sorted(glob(os.path.join(scenario_path, "image_*.png")))
        images = [Image.open(f).convert("RGB") for f in image_files]
        return images

    def load_map(self, scenario_path):
        map_file = os.path.join(scenario_path, "map.ply")
        map_data = self.load_ply(map_file)
        return map_data

    def load_ply(self, file_path):
        plydata = plyfile.PlyData.read(file_path)
        return np.array([list(vertex) for vertex in plydata.elements[0]])

# Example usage of the custom dataset and dataloader
# root_dir = "../Datasets/loki_data/"
# keys = ["odometry", "images"]
# loki_dataset = LOKIDataset(root_dir=root_dir, keys=keys, transform=None)
# sample = loki_dataset.__getitem__(0)
# print("Loaded Successfully")
