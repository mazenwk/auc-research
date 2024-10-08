# main.py

import subprocess
import sys
import os

"""
!!!!!!!!!!!!!!CURRENTLY NOT WORKING - RUN FILE DIRECTLY FROM process_pedestrians.py!!!!!!!!!!!!!!
"""


def main():
    """
    Entry point for the application. Parses command-line arguments and delegates to the processing script.
    """
    # Determine the path to the processing script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processing_script = os.path.join(script_dir, 'pedestrians-pcds-cropping', 'process_pedestrians.py')

    if not os.path.exists(processing_script):
        print(f"Processing script not found at {processing_script}")
        sys.exit(1)

    # Pass all command-line arguments to the processing script
    try:
        result = subprocess.run([sys.executable, processing_script] + sys.argv[1:], check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the processing script: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
