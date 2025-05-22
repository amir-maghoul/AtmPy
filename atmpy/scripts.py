""" Module for run scripts and parser"""

import argparse
import logging

DEFAULT_PROFILE_STEPS = 20
DEFAULT_VIS_VAR = "rho"
DEFAULT_RUN_CASE = "TravelingVortex"
CASE_CHOICES = ["TravelingVortex", "RisingBubble", "SineWaveAdvection1D"]


def parse_arguments():
    """Parses command-line arguments for the Atmpy simulation."""
    parser = argparse.ArgumentParser(description="Atmpy Simulation Framework")

    parser.add_argument(
        "mode_or_case",
        nargs="?",
        help="Execution mode ('run', 'visualize') or test case name (if mode is 'run' and omitted).",
    )

    parser.add_argument(
        "--case",
        type=str,
        choices=CASE_CHOICES,
        help=f"Test case to run or visualize. Defaults to '{DEFAULT_RUN_CASE}' for run mode if not specified.",
    )

    # --- Run specific arguments ---
    run_group = parser.add_argument_group("Run Mode Options")
    run_group.add_argument(
        "--restart", type=str, help="Path to checkpoint file to restart from."
    )
    run_group.add_argument(
        "--config-pickle-path",
        type=str,
        default=None,
        help="Path to a pickled SimulationConfig object to override default test case configuration for running.",
    )
    run_group.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling mode. Runs for --profile_steps, adjusts output.",
    )
    run_group.add_argument(
        "--profile_steps",
        type=int,
        default=DEFAULT_PROFILE_STEPS,
        help=f"Number of simulation steps to run when --profile is enabled (default: {DEFAULT_PROFILE_STEPS}).",
    )

    # --- Visualize specific arguments ---
    vis_group = parser.add_argument_group("Visualize Mode Options")
    # --case is already a top-level arg, but crucial for visualize
    vis_group.add_argument(  # Note: --case is already defined globally, this just mentally groups it for help
        "--vis_case_info",
        action="help",  # Does nothing, just for grouping in help message. Relies on global --case.
        help="The --case option (defined globally) is required for visualize mode to identify the simulation.",
    )
    vis_group.add_argument(
        "--nx",
        type=int,
        default=None,  # Default to None, meaning use case's default from config
        help="Number of grid points in X dimension for the visualized output. (Visualize mode only)",
    )
    vis_group.add_argument(
        "--ny",
        type=int,
        default=None,
        help="Number of grid points in Y dimension for the visualized output. (Visualize mode only, for 2D/3D cases)",
    )
    vis_group.add_argument(
        "--nz",
        type=int,
        default=None,
        help="Number of grid points in Z dimension for the visualized output. (Visualize mode only, for 3D cases)",
    )
    vis_group.add_argument(
        "--variable",
        type=str,
        default=DEFAULT_VIS_VAR,
        help=f"Name of the variable to plot (default: {DEFAULT_VIS_VAR}). (Visualize mode only)",
    )
    vis_group.add_argument(
        "--plot_type",
        type=str,
        choices=["static", "animate"],
        default="static",
        help="Type of plot: 'static' or 'animate'. (Visualize mode only)",
    )
    vis_group.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=[-1],
        help="Time indices to plot. For static: one index. For animate: one or two for range. (Visualize mode only)",
    )

    args = parser.parse_args()

    # Determine actual mode and case
    potential_mode = args.mode_or_case

    if potential_mode is None:
        args.mode = "run"
        if args.case is None:
            args.case = DEFAULT_RUN_CASE
    elif potential_mode.lower() == "run":
        args.mode = "run"
        if args.case is None:
            args.case = DEFAULT_RUN_CASE
    elif potential_mode.lower() == "visualize":
        args.mode = "visualize"
        if args.case is None:
            parser.error("For visualize mode, --case must be specified.")
    elif potential_mode in CASE_CHOICES:
        args.mode = "run"
        if args.case is not None and args.case != potential_mode:
            parser.error(
                f"Conflicting case names specified: '{potential_mode}' and '--case {args.case}'."
            )
        args.case = potential_mode
    else:
        parser.error(
            f"Unrecognized command or case: '{potential_mode}'. Valid modes are 'run', 'visualize'."
        )

    if args.mode == "run" and args.case is None:
        args.case = DEFAULT_RUN_CASE

    # Validation for visualize mode
    if args.mode == "visualize":
        if args.case is None:
            parser.error("--case is required for visualize mode.")
        # Ensure --ny is not used for 1D cases, --nz for 1D/2D if we want strictness (can be handled in main.py too)
        # For now, parser allows them, main.py logic might ignore irrelevant ones.

        if args.plot_type == "static" and args.steps and len(args.steps) > 1:
            logging.warning(
                f"Static plot: Using first time index {args.steps[0]} from {args.steps}."
            )
            args.steps = [args.steps[0]]
        elif args.plot_type == "animate" and args.steps and len(args.steps) > 2:
            logging.warning(
                f"Animate plot: Using range {args.steps[0]}-{args.steps[-1]} from {args.steps}."
            )
            args.steps = [args.steps[0], args.steps[-1]]

    return args
