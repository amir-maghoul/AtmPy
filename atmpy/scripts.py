""" Module for run scripts and parser"""

import argparse


def parse_arguments():
    """Parses command-line arguments for the Atmpy simulation."""
    parser = argparse.ArgumentParser(description="Atmpy Simulation")
    parser.add_argument(
        "--restart", type=str, help="Path to checkpoint file to restart from."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "visualize_only", "run_and_visualize"],
        default="run_and_visualize",
        help="Simulation mode: 'run', 'visualize_only', or 'run_and_visualize'.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to NetCDF file to visualize (required if mode is 'visualize_only').",
    )

    args = parser.parse_args()

    # You can also add some basic validation here if desired
    if args.mode == "visualize_only" and not args.input_file:
        parser.error("--input_file is required when --mode is 'visualize_only'")

    return args
