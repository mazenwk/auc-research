import argparse
import sys

from pedestrian_processing_pipeline import PedestrianProcessingPipeline


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
        help='Root directory of the dataset. Defaults to "data/".'
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default='loki.csv',
        help='Path to the CSV file containing pedestrian data. Defaults to "pedestrians.csv".'
    )
    parser.add_argument(
        '--threshold_multiplier',
        type=float,
        default=0.5,
        help='Multiplier for setting the minimum point threshold based on the average. Defaults to 0.5.'
    )
    return parser.parse_args()


def main():
    """
    Entry point for the application. Parses command-line arguments and invokes processing functions.
    """
    args = parse_arguments()

    pipeline = PedestrianProcessingPipeline(
        root_dir=args.root_dir,
        csv_path=args.csv_path,
        threshold_multiplier=args.threshold_multiplier
    )

    try:
        # Load scenario and frame IDs
        scenario_ids, frame_ids = pipeline.load_scenario_frame_ids()

        # Verify scenarios
        valid_scenario_ids = pipeline.verify_scenarios(scenario_ids)

        # Verify frames
        valid_frame_ids = pipeline.verify_frames(valid_scenario_ids, frame_ids)

        # Process all valid frames
        pipeline.process_all_frames(valid_scenario_ids, valid_frame_ids)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting gracefully...")
        sys.exit(0)


if __name__ == "__main__":
    # TODO: SAVE CROPPED PEDESTRIAN PCD
    # TODO: THREADING & PARALLEL PROCESSING
    main()
