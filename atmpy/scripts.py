""" Module for run scripts and parser"""

import argparse

DEFAULT_PROFILE_STEPS = 20  # Define a default


def parse_arguments():
    """Parses command-line arguments for the Atmpy simulation."""
    parser = argparse.ArgumentParser(description="Atmpy Simulation")

    # Existing arguments
    parser.add_argument(
        "--restart", type=str, help="Path to checkpoint file to restart from."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["run", "visualize_only", "run_and_visualize"],
        default="run_and_visualize",  # Consider changing default to 'run' if visualization is often separate
        help="Simulation mode: 'run', 'visualize_only', or 'run_and_visualize'.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to NetCDF file to visualize (required if mode is 'visualize_only').",
    )
    parser.add_argument(
        "--initial_value",  # Corrected typo: inital -> initial
        type=str,
        help="Initial value to visualize.",
    )

    # New arguments for test case selection and profiling
    parser.add_argument(
        "--case",
        type=str,
        choices=["TravelingVortex", "RisingBubble"],  # Add more cases as needed
        default="RisingBubble",  # Or your preferred default
        help="Test case to run.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",  # Makes this a flag; args.profile will be True if --profile is present
        help="Enable profiling mode. Runs for --profile_steps, adjusts output.",
    )
    parser.add_argument(
        "--profile_steps",
        type=int,
        default=DEFAULT_PROFILE_STEPS,
        help=f"Number of simulation steps to run when --profile is enabled (default: {DEFAULT_PROFILE_STEPS}).",
    )

    args = parser.parse_args()

    # Validation
    if args.mode == "visualize_only" and not args.input_file:
        parser.error("--input_file is required when --mode is 'visualize_only'")
    if args.initial_value and args.mode not in ["visualize_only", "run_and_visualize"]:
        parser.error("--initial_value can only be passed when visualizing.")
    if args.profile and args.mode == "visualize_only":
        # It's not an error, but good to inform the user if they combine them.
        # Or you could make it an error if it's truly meaningless.
        print(
            "Warning: --profile flag is used with --mode visualize_only. Profiling typically applies to 'run' modes."
        )
    if args.profile and not (args.mode == "run" or args.mode == "run_and_visualize"):
        parser.error(
            "--profile flag is intended for 'run' or 'run_and_visualize' modes."
        )

    return args
