from dataset.loki_dataset import LOKIDataset, Keys

def main():
    """ Main program """
    root_dir = "LOKI/"
    keys = [Keys.odometry, Keys.images]

    loki_dataset = LOKIDataset(root_dir=root_dir, keys=keys, transform=None)
    sample = loki_dataset[0]
    print(len(sample))

    print("Loaded Successfully")
    return 0

if __name__ == "__main__":
    main()