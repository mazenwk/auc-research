from .loki_dataset import LOKIDatasetHandler


class DatasetManager:
    """
    Manages dataset loading and sample retrieval.
    """

    def __init__(self, root_dir):
        """
        Initializes the DatasetManager with the given root directory.

        Args:
            root_dir (str): The root directory of the dataset.
        """
        self.root_dir = root_dir
        self.dataset_handler = self.load_dataset_handler()

    def load_dataset_handler(self):
        """
        Initializes and loads the dataset handler.

        Returns:
            LOKIDatasetHandler: Initialized dataset handler.

        Raises:
            SystemExit: If the dataset cannot be initialized or is empty.
        """
        try:
            dataset_handler = LOKIDatasetHandler(root_dir=self.root_dir, keys=["pointcloud", "labels_3d"])
        except ValueError as ve:
            print(f"Error initializing dataset: {ve}")
            exit(1)

        if len(dataset_handler) == 0:
            print("The dataset is empty.")
            exit(1)

        return dataset_handler

    def retrieve_sample(self, sample_idx):
        """
        Retrieves a specific sample from the dataset.

        Args:
            sample_idx (int): Index of the sample to retrieve.

        Returns:
            dict: The retrieved sample containing pointcloud and labels_3d.

        Raises:
            SystemExit: If the sample cannot be retrieved.
        """
        try:
            sample = self.dataset_handler.get_sample(sample_idx)
        except IndexError as ie:
            print(f"Error retrieving sample: {ie}")
            exit(1)
        return sample

    def dataset_length(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.dataset_handler)
