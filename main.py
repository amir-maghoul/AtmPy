# main.py
import logging
import os
import pickle
import sys
from argparse import ArgumentTypeError
from typing import Optional

import numpy as np

# Atmpy imports
from atmpy.boundary_conditions.boundary_manager import BoundaryManager
from atmpy.configuration.simulation_configuration import SimulationConfig
from atmpy.flux.flux import Flux
from atmpy.infrastructure.enums import (
    PressureSolvers,
    DiscreteOperators,
    LinearSolvers,
    TimeIntegrators,
    Preconditioners,
)
from atmpy.physics.eos import ExnerBasedEOS
from atmpy.physics.thermodynamics import Thermodynamics
from atmpy.pressure_solver.classical_pressure_solvers import ClassicalPressureSolver
from atmpy.pressure_solver.contexts import (
    DiscreteOperatorsContext,
    PressureContext,
)
from atmpy.scripts import parse_arguments
from atmpy.solver.solver import Solver
from atmpy.test_cases.sine_advection_1d import SineWaveAdvection1D
from atmpy.test_cases.traveling_vortex import TravelingVortex
from atmpy.test_cases.rising_bubble import RisingBubble
from atmpy.time_integrators.contexts import TimeIntegratorContext
from atmpy.variables.multiple_pressure_variables import MPV
from atmpy.variables.variables import Variables
from atmpy.time_integrators.imex_operator_splitting import IMEXTimeIntegrator

# Import the new filename utility
from atmpy.io.filename_utils import (
    generate_output_filepath,
)  # Assuming it's in atmpy.io

import atmpy.solver.visualize as visualizer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

np.set_printoptions(linewidth=300, suppress=True, precision=7)


def get_base_config_for_case(case_name: str) -> SimulationConfig:
    if case_name == "TravelingVortex":
        case_instance = TravelingVortex()
    elif case_name == "RisingBubble":
        case_instance = RisingBubble()
    elif case_name == "SineWaveAdvection1D":
        case_instance = SineWaveAdvection1D()
    else:
        raise ValueError(f"Unknown test case for config retrieval: {case_name}")
    return case_instance.config


args = parse_arguments()

if args.mode == "debug":
    if args.debugger == "pycharm":
        try:
            import pydevd_pycharm

            # NOTE: Remember to use your actual local IP if 'localhost' fails.
            host_ip = 'localhost'
            print("--- DEBUG MODE ---")
            print(f"Attempting to connect to PyCharm debugger at {host_ip}:{args.debug_port}...")
            print("Start the Python Debug Server in your IDE now.")
            pydevd_pycharm.settrace(host_ip, port=args.debug_port, stdoutToServer=True, stderrToServer=True,
                                    suspend=True)
            print("Debugger connected successfully.")
        except ImportError:
            print("ERROR: pydevd_pycharm module not found. Please install it.")
            pass
        except ConnectionRefusedError:
            print(f"ERROR: Connection refused. Is the PyCharm Debug Server listening on port {args.debug_port}?")
            pass
        except Exception as e:
            print(f"ERROR: Could not connect to PyCharm debugger: {e}")
            pass

if args.mode in ["run", "debug"]:
    logger.info(f"Starting in RUN mode for case: {args.case}")
    loaded_config_override: Optional[SimulationConfig] = None
    if args.config_pickle_path:
        try:
            with open(args.config_pickle_path, "rb") as f:
                loaded_config_override = pickle.load(f)
            logger.info(f"Loaded SimulationConfig from {args.config_pickle_path}")
        except Exception as e:
            logger.error(
                f"Failed to load config from {args.config_pickle_path}: {e}."
            )  # Continue with case default

    ################################ CHOICE OF TEST CASE ###################################################################
    if args.case == "TravelingVortex":
        case = TravelingVortex(config_override=loaded_config_override)
    elif args.case == "RisingBubble":
        case = RisingBubble(config_override=loaded_config_override)
    elif args.case == "SineWaveAdvection1D":
        case = SineWaveAdvection1D(config_override=loaded_config_override)
    else:
        logger.error(f"Unknown test case specified for run: {args.case}")
        sys.exit(1)
    # case = RisingBubble()
    config = case.config  # This 'config' has the definitive grid for THIS run.

    # Set defaults if not present in config
    config.temporal.tmax = 1
    config.temporal.stepmax = 10000
    config.outputs.output_frequency_steps = 3
    config.temporal.tout = [0.0]

    if args.profile:  # Profiler config adjustments
        # ... (profiler config logic remains same) ...
        logger.info(f"PROFILING ENABLED: Running for {args.profile_steps} steps.")
        config.temporal.tmax = 1e9
        config.temporal.stepmax = args.profile_steps
        config.outputs.output_frequency_steps = args.profile_steps + 10
        config.temporal.tout = []
        config.diagnostics.analysis = False

    # --- Generate output filename for RUN mode using the utility ---
    # config.grid contains the actual grid parameters for this run.
    config.outputs.output_filename = generate_output_filepath(
        config_outputs=config.outputs,  # Pass the outputs part of the config
        case_name=args.case,
        nx=config.spatial_grid.nx,
        ny=config.spatial_grid.ny if config.spatial_grid.ndim >= 2 else None,
        nz=config.spatial_grid.nz if config.spatial_grid.ndim >= 3 else None,
        ndim=config.spatial_grid.ndim,
    )

    output_dir = os.path.dirname(config.outputs.output_filename)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output (run) will be saved to: {config.outputs.output_filename}")

    # ... (Rest of simulation component setup: grid, mpv, variables, case.initialize_solution, etc.) ...
    # ... (This part uses the 'config' object which now has its 'output_filename' set) ...
    grid = config.grid
    Msq = config.model_regimes.Msq
    th = Thermodynamics()
    th.update(config.global_constants.gamma)
    mpv = MPV(grid)
    from atmpy.infrastructure.enums import VariableIndices as VI

    num_vars_cell = max(VI.RHO, VI.RHOU, VI.RHOV, VI.RHOW, VI.RHOX, VI.RHOY) + 1
    variables = Variables(grid, num_vars_cell=num_vars_cell, num_vars_node=1)
    case.initialize_solution(variables, mpv)
    bm_config = config.get_boundary_manager_config(mpv)
    manager = BoundaryManager(bm_config)
    manager.apply_boundary_on_all_sides(variables.cell_vars)
    eos = ExnerBasedEOS()
    flux = Flux(grid, variables, eos, config.numerics.riemann_solver, config.numerics.reconstruction, config.numerics.limiter)
    op_context = DiscreteOperatorsContext(DiscreteOperators.CLASSIC_OPERATOR, grid=grid)
    ps_context: PressureContext[ClassicalPressureSolver] = PressureContext(
        solver_type=PressureSolvers.CLASSIC_PRESSURE_SOLVER,
        op_context=op_context,
        linear_solver_type=config.numerics.linear_solver,
        precondition_type=config.numerics.preconditioner,
        extra_dependencies={
            "grid": grid,
            "variables": variables,
            "mpv": mpv,
            "boundary_manager": manager,
            "coriolis": config.physics.coriolis,
            "Msq": Msq,
            "thermodynamics": th,
        },
    )
    pressure_solver = ps_context.instantiate()
    ti_context: TimeIntegratorContext[IMEXTimeIntegrator] = TimeIntegratorContext(
        integrator_type=TimeIntegrators.IMEX,
        grid=grid,
        variables=variables,
        flux=flux,
        boundary_manager=manager,
        first_order_advection_routine=config.numerics.first_order_advection_routine,
        second_order_advection_routine=config.numerics.second_order_advection_routine,
        dt=config.temporal.dtfixed,
        extra_dependencies={
            "mpv": mpv,
            "pressure_solver": pressure_solver,
            "wind_speed": config.physics.wind_speed,
            "is_nonhydrostatic": config.model_regimes.is_nonhydrostatic,
            "is_nongeostrophic": config.model_regimes.is_nongeostrophic,
            "is_compressible": config.model_regimes.is_compressible,
        },
    )
    time_integrator = ti_context.instantiate()

    if (
        hasattr(config.numerics, "initial_projection")
        and config.numerics.initial_projection
    ):
        logger.info("Performing initial projection...")
        pass

    initial_t = 0.0
    initial_step = 0
    if args.restart:
        try:
            loaded_data = Solver.load_checkpoint_data(args.restart)
            initial_t = loaded_data["current_t"]
            initial_step = loaded_data["current_step"] + 1
            variables.cell_vars[...] = loaded_data["cell_vars"]
            mpv.p2_nodes[...] = loaded_data["p2_nodes"]
            logger.info(
                f"Restarting from checkpoint: t={initial_t:.4f}, next step={initial_step}"
            )
        except Exception as e:
            logger.error(f"Failed to restart from {args.restart}: {e}. Exiting.")
            sys.exit(1)

    # Solver uses config.outputs.output_filename which is now correctly set.
    solver = Solver(
        config=config,
        grid=grid,
        variables=variables,
        mpv=mpv,
        time_integrator=time_integrator,
        initial_t=initial_t,
        initial_step=initial_step,
    )
    solver.run()
    logger.info(f"Run for case {args.case} completed.")


elif args.mode == "visualize":
    logger.info(f"Starting in VISUALIZE mode for case: {args.case}")
    if args.file is None:
        try:
            base_vis_config = get_base_config_for_case(args.case)  # Gets default config
        except ValueError as e:
            logger.error(str(e))
            sys.exit(1)

    # Determine effective grid dimensions for path construction
    if args.file is not None and (
        args.nx is not None or args.ny is not None or args.nz is not None
    ):
        raise ArgumentTypeError(
            "Cannot specify both --file and --nx or --ny or --nz at the same time."
        )

    elif args.file is not None:
        input_file_to_visualize = args.file[0]
    else:
        nx_to_use = args.nx
        ny_to_use = None
        nz_to_use = None
        ndim_to_use = base_vis_config.spatial_grid.ndim

        if ndim_to_use >= 2:
            ny_to_use = (
                args.ny if args.ny is not None else base_vis_config.spatial_grid.ny
            )
        elif args.ny is not None:
            logger.warning(f"Ignoring --ny for {ndim_to_use}D case '{args.case}'.")
        if ndim_to_use >= 3:
            nz_to_use = (
                args.nz if args.nz is not None else base_vis_config.spatial_grid.nz
            )
        elif args.nz is not None:
            logger.warning(f"Ignoring --nz for {ndim_to_use}D case '{args.case}'.")

        logger.info(
            f"Visualizing for effective grid: Nx={nx_to_use}, Ny={ny_to_use}, Nz={nz_to_use} (Ndim={ndim_to_use})"
        )


        # --- Generate output filename for VISUALIZATION using the utility ---
        try:
            input_file_to_visualize = generate_output_filepath(
                config_outputs=base_vis_config.outputs,  # Use output settings from default config
                case_name=args.case,
                nx=nx_to_use,
                ny=ny_to_use,
                nz=nz_to_use,
                ndim=ndim_to_use,
            )
        except (
            ValueError
        ) as e:  # Handles missing ny/nz for 2D/3D in generate_output_filepath
            logger.error(f"Error generating filepath for visualization: {e}")
            sys.exit(1)

        logger.info(f"Attempting to visualize file: {input_file_to_visualize}")

    if not os.path.exists(input_file_to_visualize):
        error_msg = f"Visualization input file not found: {input_file_to_visualize}\n"

    visualizer.visualize_data(
        input_file=input_file_to_visualize,
        variable_name=args.variable,
        plot_type=args.plot_type,
        time_indices=args.steps,
    )
    logger.info("Visualization process finished.")

else:
    logger.error(f"Invalid mode: {args.mode}. This should not happen.")
    sys.exit(1)
