import os
import json
import torch
import plyfile
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from enum import Enum

class Keys(Enum):
    odometry = 1
    labels2d = 2
    labels3d = 3
    pointcloud = 4
    images = 5
    map = 6

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
            else [key.name for key in Keys]
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

        if Keys.odometry in self.keys:
            sample[Keys.odometry.name] = self.load_odometry(scenario_path)
        if Keys.labels2d in self.keys:
            sample[Keys.labels2d.name] = self.load_labels_2d(scenario_path)
        if Keys.labels3d in self.keys:
            sample[Keys.labels3d.name] = self.load_labels_3d(scenario_path)
        if Keys.pointcloud in self.keys:
            sample[Keys.pointcloud.name] = self.load_pointcloud(scenario_path)
        if Keys.images in self.keys:
            sample[Keys.images.name] = self.load_images(scenario_path)
        if Keys.map in self.keys:
            sample[Keys.map.name] = self.load_map(scenario_path)

        if self.transform and Keys.images.name in sample:
            sample[Keys.images.name] = [self.transform(image) for image in sample[Keys.images.name]]

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
# root_dir = "../../LOKI/"
# keys = ["odometry", "images"]

# loki_dataset = LOKIDataset(root_dir=root_dir, keys=keys, transform=None)
# # sample = loki_dataset.__getitem__(0)
# sample = loki_dataset[0]
# print(len(sample))

# print("Loaded Successfully")
