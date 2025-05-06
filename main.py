# main.py

import numpy as np
import logging
import argparse

from atmpy.test_cases.rising_bubble import RisingBubble
from atmpy.test_cases.traveling_vortex import TravelingVortex
from atmpy.physics.thermodynamics import Thermodynamics
from atmpy.grid.utility import DimensionSpec, create_grid
from atmpy.variables.variables import Variables
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    HydrostateIndices as HI,
    BoundaryConditions as BdryType,
    BoundarySide,
    PressureSolvers,
    DiscreteOperators,
    LinearSolvers,
    AdvectionRoutines,
    TimeIntegrators,
    Preconditioners,
)
from atmpy.boundary_conditions.bc_extra_operations import (
    WallAdjustment,
    PeriodicAdjustment,
)
from atmpy.variables.multiple_pressure_variables import MPV
from atmpy.boundary_conditions.boundary_manager import BoundaryManager
from atmpy.physics.eos import ExnerBasedEOS
from atmpy.flux.flux import Flux
from atmpy.pressure_solver.contexts import (
    DiscreteOperatorsContext,
    PressureContext,
)
from atmpy.time_integrators.contexts import TimeIntegratorContext
from atmpy.pressure_solver.classical_pressure_solvers import ClassicalPressureSolver
from atmpy.solver.solver import Solver

np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(precision=10)

################################### Parser #########################################################################
parser = argparse.ArgumentParser(description="Atmpy Simulation")
parser.add_argument(
    "--restart", type=str, help="Path to checkpoint file to restart from."
)
args = parser.parse_args()

#################################### Instantiate Test Case and Get Config ##########################################
case = TravelingVortex()
# case = RisingBubble()
config = case.config  # The SimulationConfig object is now held by the case

# Modify config if needed (e.g., simulation time)
config.temporal.tmax = 0.05  # Run for a short time for testing
config.temporal.stepmax = 101  # Limit steps
config.outputs.output_frequency_steps = 2  # Output every 2 steps
config.temporal.tout = [0.025]  # Also output at a specific time

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
    advection_routine=config.numerics.advection_routine,  # Get from config
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

# --- Post-processing / Analysis (Optional) ---
print("\n --- Final State (Inner Domain Rho Example) ---")
inner_slice = grid.get_inner_slice()
print(variables.cell_vars[inner_slice + (VI.RHO,)])

print("\n --- Final State (Inner Domain p2 Example) ---")
print(mpv.p2_nodes[inner_slice])

############################################ Visualization #############################################################
import xarray as xr
import matplotlib.pyplot as plt

path = "/home/amir/Projects/Python/Atmpy/atmpy/output_data/traveling_vortex/"
ds = xr.open_dataset(config.outputs.output_filename)
print(ds)
# rho_at_time_0 = ds['rho'].sel(time=0.0, method='nearest')
rho = ds["rho"].values
u = ds["u"].values
print(u.shape)

i = 2
plt.figure()
plt.contourf(ds["x"], ds["y"], u[i, :, :])  # Example for a 2D contour plot
plt.colorbar(label="Density (kg/m^3)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid()
# plt.title(f"Density at t={rho[3, :, :].time.item():.2f}s")
plt.show()
