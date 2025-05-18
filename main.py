# main.py

import logging
import os
from line_profiler import profile

import numpy as np

from atmpy.boundary_conditions.boundary_manager import BoundaryManager
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
from atmpy.test_cases.traveling_vortex import TravelingVortex
from atmpy.test_cases.rising_bubble import RisingBubble
from atmpy.time_integrators.contexts import TimeIntegratorContext
from atmpy.variables.multiple_pressure_variables import MPV
from atmpy.variables.variables import Variables

np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=10)

################################### Parser #########################################################################
args = parse_arguments()

#################################### Instantiate Test Case and Get Config ##########################################
if args.case == "TravelingVortex":
    case = TravelingVortex()
    logging.info("Selected Test Case: TravelingVortex")
elif args.case == "RisingBubble":
    case = RisingBubble()
    logging.info("Selected Test Case: RisingBubble")
else:
    # This should ideally be caught by argparse choices, but as a fallback:
    raise ValueError(f"Unknown test case specified: {args.case}")

config = case.config  # The SimulationConfig object is now held by the case

# Modify config if needed (e.g., simulation time)
# These are defaults if not profiling
config.temporal.tmax = 1
config.temporal.stepmax = 101000
config.outputs.output_frequency_steps = 3
config.temporal.tout = [0]

#################################### Profiler Configuration ############################################################
# Apply profiling-specific configurations if --profile flag is set
if args.profile and args.mode in ["run", "run_and_visualize"]:
    logging.info(f"PROFILING ENABLED: Running for {args.profile_steps} steps.")
    config.temporal.tmax = 1e9  # Effectively disable tmax, rely on stepmax
    config.temporal.stepmax = args.profile_steps
    # Prevent output during profiling runs to not skew timing, unless explicitly very short
    config.outputs.output_frequency_steps = (
        args.profile_steps + 10
    )  # Make sure no output happens
    config.temporal.tout = []  # Disable specific time outputs during profiling
    # Potentially disable other verbose logging or non-essential computations for cleaner profiles
    # Example: config.outputs.enable_console_output = False (if you have such a flag)

#################################### Parser Fill #######################################################################
# --- Determine output filename for visualization ---
# This needs to be robust whether we run or just visualize
if args.mode == "visualize_only" or args.mode == "run_and_visualize":
    output_data_file = args.input_file
else:  # 'run' or 'run_and_visualize'
    # Ensure output filename in config is up-to-date if changed by case.setup()
    if not config.outputs.output_filename:  # If not explicitly set by user/case
        config.outputs.output_filename = os.path.join(
            config.outputs.output_path,
            config.outputs.output_folder,
            f"{config.outputs.output_base_name}{config.outputs.output_suffix}{config.outputs.output_extension}",
        )
        # Ensure the directory exists if we are running
        if args.mode in ["run", "run_and_visualize"]:
            os.makedirs(os.path.dirname(config.outputs.output_filename), exist_ok=True)

if args.mode in ["run", "run_and_visualize"]:
    ##################################### Create Simulation Components based on Config #################################
    grid = config.grid
    gravity_vec = config.physics.gravity_strength
    Msq = config.model_regimes.Msq
    th = Thermodynamics()
    th.update(config.global_constants.gamma)

    ##################################### MPV ##########################################################################
    mpv = MPV(grid)

    ##################################### Variables ####################################################################
    num_vars = 6
    variables = Variables(
        grid, num_vars_cell=num_vars, num_vars_node=1
    )  # Adjust num_vars if needed

    # --- Initialize Solution using Test Case
    case.initialize_solution(variables, mpv)  # This now uses the case object

    ##################################### Boundary Manager #############################################################
    bm_config = config.get_boundary_manager_config()
    manager = BoundaryManager(bm_config)
    # Apply initial BCs (though initialize_solution might handle inner domain, ghosts need update)
    manager.apply_boundary_on_all_sides(variables.cell_vars)
    # nodal_bc_contexts = ([manager.get_context_for_side(i, is_nodal=True) for i in range(grid.ndim * 2)]) # Get contexts for p2
    # manager.apply_boundary_on_single_var_all_sides(mpv.p2_nodes, nodal_bc_contexts)

    ###################################### Flux ########################################################################
    eos = ExnerBasedEOS()
    flux = Flux(grid, variables, eos)

    ###################################### Pressure Solver #############################################################
    op_context = DiscreteOperatorsContext(
        operator_type=DiscreteOperators.CLASSIC_OPERATOR, grid=grid
    )
    linear_solver = LinearSolvers.BICGSTAB

    ps_context: PressureContext[ClassicalPressureSolver] = PressureContext(
        solver_type=PressureSolvers.CLASSIC_PRESSURE_SOLVER,
        op_context=op_context,
        linear_solver_type=linear_solver,
        precondition_type=Preconditioners.DIAGONAL,
        extra_dependencies={
            "grid": grid,
            "variables": variables,
            "mpv": mpv,
            "boundary_manager": manager,
            "coriolis": config.physics.coriolis,  # Get from config
            "Msq": Msq,
            "thermodynamics": th,
        },
    )
    pressure_solver = ps_context.instantiate()

    ######################################### Time Integrator ##########################################################
    from atmpy.time_integrators.imex_operator_splitting import IMEXTimeIntegrator

    ti_context: TimeIntegratorContext[IMEXTimeIntegrator] = TimeIntegratorContext(
        integrator_type=TimeIntegrators.IMEX,
        grid=grid,
        variables=variables,
        flux=flux,
        boundary_manager=manager,
        first_order_advection_routine=config.numerics.first_order_advection_routine,  # Get from config
        second_order_advection_routine=config.numerics.second_order_advection_routine,  # Get from config
        dt=config.temporal.dtfixed,  # Get from config
        extra_dependencies={
            "mpv": mpv,
            "pressure_solver": pressure_solver,
            "wind_speed": config.physics.wind_speed,  # Get from config
            "is_nonhydrostatic": config.model_regimes.is_nonhydrostatic,
            "is_nongeostrophic": config.model_regimes.is_nongeostrophic,
            "is_compressible": config.model_regimes.is_compressible,
        },
    )
    time_integrator = ti_context.instantiate()

    #     #################################### Optional(Initial Projection) ##################################################
    #     # 5. Perform Initial Projection (Optional)
    #     if config.numerics.initial_projection:
    #         print("Performing initial projection...")
    #         # This step needs careful implementation. It's essentially applying the
    #         # pressure correction mechanism once at t=0.
    #         # Option A: Add a dedicated method to the PressureSolver
    #         # pressure_solver.perform_initial_projection(variables, mpv, config.temporal.dtfixed0) # Pass initial dt if needed
    #         # Option B: Add a dedicated method to the TimeIntegrator
    #         # time_integrator.perform_initial_projection()
    #         # Option C: Implement the logic directly here (less ideal, duplicates solver logic)
    #
    #         # Let's assume Option A exists:
    #         try:
    #             # We need a way to call the implicit pressure update logic *once*
    #             # In Atmpy's IMEX integrator, this logic is inside `backward_update_implicit`.
    #             # We might need to refactor slightly or add a dedicated method.
    #             # A potential approach:
    #             logging.info(
    #                 "Applying initial projection via backward_update_implicit (check if dt=0 or small dt is appropriate)")
    #             # Temporarily modify state if needed (like PyBella subtracting background wind)
    #             # variables.adjust_background_wind(config.physics.wind_speed, scale=-1.0, in_place=True) # Example if needed
    #
    #             # Call the core implicit update. The dt might need consideration (e.g., dt=0 or a small initial dt?)
    #             # Using the initial dt (dtfixed0) seems plausible based on PyBella.
    #             time_integrator.backward_update_implicit(config.temporal.dtfixed0)  # Pass initial dt
    #
    #             # Restore state if modified
    #             # variables.adjust_background_wind(config.physics.wind_speed, scale=1.0, in_place=True) # Example if needed
    #
    #             # Re-apply BCs after projection modifies the state
    #             print("Re-applying boundary conditions after initial projection...")
    #             manager.apply_boundary_on_all_sides(variables.cell_vars)
    #             manager.apply_pressure_boundary_on_all_sides(mpv.p2_nodes, nodal_bc_contexts)
    #
    #         except AttributeError:
    #             print(
    #                 "Warning: Initial projection requested but corresponding method not found/implemented in pressure solver/time integrator.")
    #         except Exception as e:
    #             print(f"Error during initial projection: {e}")

    ################################## Restarting From Checkpoint ######################################################
    initial_t = 0.0
    initial_step = 0

    if args.restart:
        try:
            # --- Load state from checkpoint ---
            loaded_data = Solver.load_checkpoint_data(args.restart)
            initial_t = loaded_data["current_t"]
            # IMPORTANT: Increment step number to start *after* the loaded step
            initial_step = loaded_data["current_step"] + 1  # Start next step

            # --- Populate variables with loaded data ---
            if variables.cell_vars.shape == loaded_data["cell_vars"].shape:
                variables.cell_vars[...] = loaded_data["cell_vars"]
            else:
                raise ValueError("Checkpoint cell_vars shape mismatch!")
            if mpv.p2_nodes.shape == loaded_data["p2_nodes"].shape:
                mpv.p2_nodes[...] = loaded_data["p2_nodes"]
            else:
                raise ValueError("Checkpoint p2_nodes shape mismatch!")
            # ... load other state like mpv.p2_cells if saved ...

            logging.info(
                f"Restarting simulation from checkpoint state at t={initial_t:.4f}, step={initial_step}"
            )
            # No need to call case.initialize_solution or apply initial BCs/projection

        except (FileNotFoundError, IOError, KeyError, ValueError) as e:
            logging.error(f"Failed to restart from {args.restart}: {e}")
            logging.error("Exiting.")
            exit(1)
    else:
        # --- Standard Initialization ---
        logging.info("Starting new simulation.")
        case.initialize_solution(variables, mpv)
        manager.apply_boundary_on_all_sides(variables.cell_vars)
        # ... apply MPV BCs ...
        if config.numerics.initial_projection:
            # ... perform initial projection ...
            # ... reapply BCs ...
            pass

    ##################################### Create and Run the Solver ####################################################
    solver = Solver(
        config=config,
        grid=grid,
        variables=variables,
        mpv=mpv,
        time_integrator=time_integrator,
        initial_t=initial_t,  # Pass loaded/initial time
        initial_step=initial_step,  # Pass loaded/initial step
    )
    solver.run()

############################################ Visualization #############################################################
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

if args.mode in ["visualize_only", "run_and_visualize"]:
    if args.mode == "visualize_only":
        ds = xr.open_dataset(output_data_file)
    elif args.mode == "run_and_visualize":
        ds = xr.open_dataset(config.outputs.output_filename)
        print(config.outputs.output_filename)
    logging.info(
        f"Starting visualization in '{args.mode}' mode for file: {output_data_file}"
    )
    try:
        print("\n--- Dataset for Visualization ---")
        print(ds)

        # Example: Animate 'rho' over time
        fig, ax = plt.subplots()

        # Determine data to plot (e.g., rho)
        data_var = "Y"  # or 'u', 'v', etc.
        if data_var not in ds:
            logging.error(
                f"Variable '{data_var}' not found in {output_data_file}. Cannot animate."
            )
        else:
            times = ds["time"].values
            x_coords = ds["x"].values
            y_coords = ds["y"].values

            # Initial plot setup
            # For the first frame, plot the data at the first timestep
            # The contourf object will be updated in the animation function

            # Get min/max for consistent color scaling across all frames
            data_min = ds[data_var].min().item()
            data_max = ds[data_var].max().item()

            cmap = "viridis"

            # Initial plot for the first time step
            # contour = ax.contourf(x_coords, y_coords, ds[data_var].isel(time=0).values.T, cmap=cmap, levels=np.linspace(data_min, data_max, 15))
            # Using .T because xarray usually has (time, y, x) and contourf expects (X, Y, Z) where Z aligns with X,Y
            # For your data, it seems output is (time, x, y) based on your original visualization
            contour = ax.contourf(
                x_coords,
                y_coords,
                ds[data_var].isel(time=0).data.T,
                cmap=cmap,
                levels=np.linspace(data_min, data_max, 15),
            )

            cbar = fig.colorbar(contour, ax=ax)
            cbar.set_label(f"{data_var} ({ds[data_var].attrs.get('units', '')})")
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Y (m)")

            def animate(i):
                ax.clear()  # Clear previous frame's contours
                current_data = ds[data_var].isel(time=i).data

                # If X_coords (1D for x), Y_coords (1D for y), then Z data should be (len(Y_coords), len(X_coords))
                # Your original plot: plt.contourf(ds['x'], ds['y'], rho[i, :, :])
                # This implies rho[i] has shape (len(ds['x']), len(ds['y'])) if ds['x'] and ds['y'] are 1D.
                # If ds[data_var].isel(time=i) gives (nx, ny), then it's fine.
                # If it gives (ny, nx), you might need .T
                # Let's assume ds[data_var].isel(time=i) is (nx, ny) as per your original code.
                cont = ax.contourf(
                    x_coords,
                    y_coords,
                    current_data.T,
                    cmap=cmap,
                    levels=np.linspace(data_min, data_max, 15),
                )
                ax.set_title(f"{data_var} at t={times[i]}s")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Y (m)")
                # fig.colorbar(cont, ax=ax) # Re-adding colorbar can be slow/messy, update existing one if possible
                return (cont,)  # Comma is important for blitting

            # Create animation
            # frames = len(times)
            # interval is delay between frames in ms
            # blit=True means only re-draw the parts that have changed for efficiency
            ani = FuncAnimation(
                fig, animate, frames=len(times), interval=2, blit=False
            )

            plt.tight_layout()
            plt.show()

            # To save the animation (optional, requires ffmpeg or imagemagick):
            # output_animation_file = os.path.join(config.outputs.output_path, config.outputs.output_folder, "animation.mp4")
            # ani.save(output_animation_file, writer='ffmpeg', fps=5)
            # logging.info(f"Animation saved to {output_animation_file}")

    except FileNotFoundError:
        logging.error(f"Output file {output_data_file} not found for visualization.")
    except Exception as e:
        logging.error(f"An error occurred during visualization: {e}", exc_info=True)

elif args.mode == "run":
    logging.info(
        f"Simulation run complete. No visualization requested in '{args.mode}' mode."
    )
