""" This module handles different saving routines for different variables"""

from __future__ import annotations

import logging
import os
import glob
from typing import TYPE_CHECKING, Dict, Any, Tuple

import netCDF4
import numpy as np

# Assuming output_writer.py is in atmpy.io
from atmpy.io.output_writers import NetCDFWriter
from atmpy.infrastructure.enums import (
    VariableIndices as VI,
    PrimitiveVariableIndices as PVI,
)
from atmpy.physics.eos import ExnerBasedEOS

if TYPE_CHECKING:
    from atmpy.solver.solver import Solver  # Forward reference for type hinting

DIM_MAP_CELLS = {0: "x", 1: "y", 2: "z"}
DIM_MAP_NODES = {0: "x_node", 1: "y_node", 2: "z_node"}


def _get_checkpoint_filename(solver_instance: "Solver") -> str:
    """Generates the filename for the current checkpoint."""
    return f"{solver_instance.checkpoint_filename}_step_{solver_instance.current_step:09d}.nc"


def _manage_old_checkpoints(solver_instance: "Solver") -> None:
    """Deletes older checkpoint files."""
    if solver_instance.checkpoint_keep_n <= 0:
        return
    try:
        pattern = f"{solver_instance.checkpoint_filename}_step_*.nc"
        checkpoint_files = sorted(glob.glob(pattern))
        files_to_delete = checkpoint_files[: -solver_instance.checkpoint_keep_n]
        for f_path in files_to_delete:
            try:
                os.remove(f_path)
                logging.info(f"Removed old checkpoint file: {f_path}")
            except OSError as e:
                logging.warning(f"Could not remove old checkpoint file {f_path}: {e}")
    except Exception as e:
        logging.error(f"Error during old checkpoint management: {e}", exc_info=True)


def save_checkpoint(solver_instance: "Solver", force_save: bool = False) -> None:
    """Saves the complete simulation state for restarting."""
    if not solver_instance.enable_checkpointing and not force_save:
        return

    chk_filename = _get_checkpoint_filename(solver_instance)
    reason = "error recovery" if force_save else "scheduled"
    logging.info(
        f"Saving checkpoint to {chk_filename} (Reason: {reason}) at t={solver_instance.current_t:.4f}, step={solver_instance.current_step}"
    )

    try:
        with NetCDFWriter(
            chk_filename,
            solver_instance.grid,
            solver_instance.config,
            mode="w",
            is_checkpoint=True,
        ) as writer:
            full_cell_dims = tuple(
                DIM_MAP_CELLS[i] for i in range(solver_instance.grid.ndim)
            )
            full_node_dims = tuple(
                DIM_MAP_NODES[i] for i in range(solver_instance.grid.ndim)
            )

            writer.define_variable(
                "cell_vars",
                full_cell_dims + ("var_cell",),
                dtype=solver_instance.variables.cell_vars.dtype,
                attributes={"long_name": "Full cell-centered state variables"},
            )
            writer.define_variable(
                "p2_nodes",
                full_node_dims,
                dtype=solver_instance.mpv.p2_nodes.dtype,
                attributes={"long_name": "Full p2_nodes state"},
            )
            # Add other MPV state if needed (e.g., p2_cells, dp2_nodes)

            data_to_write = {
                "cell_vars": solver_instance.variables.cell_vars,
                "p2_nodes": solver_instance.mpv.p2_nodes,
            }

            global_attrs = {
                "checkpoint_time": solver_instance.current_t,
                "checkpoint_step": solver_instance.current_step,
                "description": f"Atmpy Checkpoint for step {solver_instance.current_step}",
            }
            writer.write_snapshot_data(data_to_write, global_attrs)

        logging.debug(f"Successfully wrote checkpoint file: {chk_filename}")
        if not force_save:
            _manage_old_checkpoints(solver_instance)

    except Exception as e:
        logging.error(
            f"Failed to save checkpoint to {chk_filename}: {e}", exc_info=True
        )


def check_and_save_checkpoint(solver_instance: "Solver") -> None:
    """Handles saving of checkpoint files based on frequency."""
    if not solver_instance.enable_checkpointing:
        return
    if solver_instance.current_step % solver_instance.checkpoint_frequency_steps == 0:
        save_checkpoint(solver_instance)


def save_analysis_output(solver_instance: "Solver", reason: str) -> None:
    """Saves standard analysis output using NetCDFWriter."""
    logging.info(
        f"Saving analysis output to {solver_instance.output_filename} (Reason: {reason}) at t={solver_instance.current_t:.4f}, step={solver_instance.current_step}"
    )

    file_exists = os.path.exists(solver_instance.output_filename)
    mode = (
        "a"
        if file_exists
        and (solver_instance.current_t > 1e-9 or solver_instance.current_step > 0)
        else "w"
    )

    try:
        with NetCDFWriter(
            solver_instance.output_filename,
            solver_instance.grid,
            solver_instance.config,
            mode=mode,
            is_checkpoint=False,
        ) as writer:
            inner_cell_dims = tuple(
                DIM_MAP_CELLS[i] for i in range(solver_instance.grid.ndim)
            )
            time_inner_cell_dims = ("time",) + inner_cell_dims
            inner_node_dims = tuple(
                DIM_MAP_NODES[i] for i in range(solver_instance.grid.ndim)
            )
            time_inner_node_dims = ("time",) + inner_node_dims

            eos = solver_instance.time_integrator.flux.eos

            attrs_conservative = {
                "rho": {"units": "kg m-3", "long_name": "Density (conservative)"},
                "rhou": {
                    "units": "kg m-2 s-1",
                    "long_name": "X-momentum density (rho*u)",
                },
                "rhov": {
                    "units": "kg m-2 s-1",
                    "long_name": "Y-momentum density (rho*v)",
                },
                "rhow": {
                    "units": "kg m-2 s-1",
                    "long_name": "Z-momentum density (rho*w)",
                },
                "rhoY": {
                    "units": "kg m-3 K",
                    "long_name": "Potential temperature density (rho*Theta_nd or rho*Y)",
                },
                "rhoX": {"units": "kg m-3", "long_name": "Tracer density (rho*X)"},
            }
            attrs_primitive = {
                "u": {
                    "units": "m s-1",
                    "long_name": "X-velocity component (primitive)",
                },
                "v": {
                    "units": "m s-1",
                    "long_name": "Y-velocity component (primitive)",
                },
                "w": {
                    "units": "m s-1",
                    "long_name": "Z-velocity component (primitive)",
                },
                "Y": {
                    "units": "K",
                    "long_name": "Potential temperature (Theta_nd or Y)",
                },
                "X": {"units": "kg/kg", "long_name": "Tracer mixing ratio (X)"},
            }
            attrs_nodal = {
                "p2_nodes": {
                    "units": "Pa K m3 kg-1",
                    "long_name": "Exner Pressure Perturbation (Nodes)",
                },
            }

            vars_to_define = {}
            vars_to_define["rho"] = (time_inner_cell_dims, attrs_conservative["rho"])
            vars_to_define["rhou"] = (time_inner_cell_dims, attrs_conservative["rhou"])
            if (
                solver_instance.grid.ndim >= 2
                and VI.RHOV < solver_instance.variables.num_vars_cell
            ):
                vars_to_define["rhov"] = (
                    time_inner_cell_dims,
                    attrs_conservative["rhov"],
                )
            if (
                solver_instance.grid.ndim >= 3
                and VI.RHOW < solver_instance.variables.num_vars_cell
            ):
                vars_to_define["rhow"] = (
                    time_inner_cell_dims,
                    attrs_conservative["rhow"],
                )
            vars_to_define["rhoY"] = (time_inner_cell_dims, attrs_conservative["rhoY"])
            if VI.RHOX < solver_instance.variables.num_vars_cell:
                vars_to_define["rhoX"] = (
                    time_inner_cell_dims,
                    attrs_conservative["rhoX"],
                )

            vars_to_define["u"] = (time_inner_cell_dims, attrs_primitive["u"])
            if (
                solver_instance.grid.ndim >= 2
                and VI.RHOV < solver_instance.variables.num_vars_cell
            ):
                vars_to_define["v"] = (time_inner_cell_dims, attrs_primitive["v"])
            if (
                solver_instance.grid.ndim >= 3
                and VI.RHOW < solver_instance.variables.num_vars_cell
            ):
                vars_to_define["w"] = (time_inner_cell_dims, attrs_primitive["w"])
            vars_to_define["Y"] = (time_inner_cell_dims, attrs_primitive["Y"])
            if VI.RHOX < solver_instance.variables.num_vars_cell:
                vars_to_define["X"] = (time_inner_cell_dims, attrs_primitive["X"])
            vars_to_define["p2_nodes"] = (time_inner_node_dims, attrs_nodal["p2_nodes"])

            for name, (dims, attributes) in vars_to_define.items():
                all_dims_exist = all(d_name in writer.ds.dimensions for d_name in dims)
                if all_dims_exist:
                    writer.define_variable(
                        name, dims, dtype=np.float64, attributes=attributes
                    )
                else:
                    logging.warning(
                        f"Variable '{name}' not defined due to missing dimensions in {dims}. Available: {list(writer.ds.dimensions.keys())}"
                    )

            inner_slice = solver_instance.grid.get_inner_slice()
            data_to_write = {}

            rho_data = solver_instance.variables.cell_vars[inner_slice + (VI.RHO,)]
            data_to_write["rho"] = rho_data
            data_to_write["rhou"] = solver_instance.variables.cell_vars[
                inner_slice + (VI.RHOU,)
            ]
            if "rhov" in vars_to_define:
                data_to_write["rhov"] = solver_instance.variables.cell_vars[
                    inner_slice + (VI.RHOV,)
                ]
            if "rhow" in vars_to_define:
                data_to_write["rhow"] = solver_instance.variables.cell_vars[
                    inner_slice + (VI.RHOW,)
                ]
            data_to_write["rhoY"] = solver_instance.variables.cell_vars[
                inner_slice + (VI.RHOY,)
            ]
            if "rhoX" in vars_to_define:
                data_to_write["rhoX"] = solver_instance.variables.cell_vars[
                    inner_slice + (VI.RHOX,)
                ]

            # Primitive velocities and tracers (derived)
            # Ensure rho_data is not zero to avoid division by zero errors
            safe_rho_data = np.where(
                np.abs(rho_data) < np.finfo(float).eps, np.finfo(float).eps, rho_data
            )

            data_to_write["u"] = (
                solver_instance.variables.cell_vars[inner_slice + (VI.RHOU,)]
                / safe_rho_data
            )
            if "v" in vars_to_define:
                data_to_write["v"] = (
                    solver_instance.variables.cell_vars[inner_slice + (VI.RHOV,)]
                    / safe_rho_data
                )
            if "w" in vars_to_define:
                data_to_write["w"] = (
                    solver_instance.variables.cell_vars[inner_slice + (VI.RHOW,)]
                    / safe_rho_data
                )
            data_to_write["Y"] = (
                solver_instance.variables.cell_vars[inner_slice + (VI.RHOY,)]
                / safe_rho_data
            )
            if "X" in vars_to_define:
                data_to_write["X"] = (
                    solver_instance.variables.cell_vars[inner_slice + (VI.RHOX,)]
                    / safe_rho_data
                )

            if "p2_nodes" in vars_to_define:
                data_to_write["p2_nodes"] = solver_instance.mpv.p2_nodes[inner_slice]

            writer.write_timestep_data(solver_instance.current_t, data_to_write)
        logging.debug("Successfully wrote analysis output step.")
    except Exception as e:
        logging.error(
            f"Failed to save analysis output to {solver_instance.output_filename}: {e}",
            exc_info=True,
        )


def check_and_save_analysis_output(
    solver_instance: "Solver", analysis_active: bool
) -> None:
    """Handles saving of standard analysis output based on analysis_active flag."""

    # Handle initial state (t=0, step=0) separately and unconditionally for analysis output
    if solver_instance.current_step == 0:
        save_analysis_output(solver_instance, "initial state (t=0, step=0)")
        # Mark t=0 as saved if it's in tout
        for i, t_val in enumerate(solver_instance.tout):
            # Using a very small tolerance for t=0 comparison
            if abs(t_val - 0.0) < np.finfo(float).eps * 10:
                if not solver_instance._output_times_saved[i]:
                    solver_instance._output_times_saved[i] = True
        return

    # For subsequent steps (current_step > 0)
    reason_for_saving = None

    # Check specific output times (tout)
    for i, t_out in enumerate(solver_instance.tout):
        if (
            not solver_instance._output_times_saved[i]
            and abs(solver_instance.current_t - t_out)
            <= solver_instance._output_time_tolerance
        ):
            reason_for_saving = f"specific time t={t_out:.4f}"
            solver_instance._output_times_saved[i] = True
            break  # Prioritize saving for specific time

    if analysis_active:
        # If analysis_active, only save if a 'tout' condition was met (and not step 0)
        if reason_for_saving:
            save_analysis_output(solver_instance, reason_for_saving)
    else:
        # If not analysis_active, also check step-based frequency
        if not reason_for_saving:  # If not already saving due to 'tout'
            current_step_is_output_freq = (
                solver_instance.output_frequency_steps > 0
                and solver_instance.current_step
                % solver_instance.output_frequency_steps
                == 0
            )
            if current_step_is_output_freq:
                reason_for_saving = (
                    f"step frequency ({solver_instance.output_frequency_steps})"
                )

        if reason_for_saving:  # If any condition (tout or frequency) met
            save_analysis_output(solver_instance, reason_for_saving)


def handle_saving(solver_instance: "Solver") -> None:
    """
    Checks and performs saving for both analysis output and checkpoints.
    The 'analysis_active' flag is sourced from solver_instance.config.
    """
    # Safely get the analysis flag
    analysis_active = False
    if hasattr(solver_instance.config, "diagnostics") and hasattr(
        solver_instance.config.diagnostics, "analysis"
    ):
        analysis_active = solver_instance.config.diagnostics.analysis
    else:
        logging.warning(
            "config.diagnostics.analysis not found, defaulting to False for saving behavior."
        )

    check_and_save_analysis_output(solver_instance, analysis_active)
    check_and_save_checkpoint(solver_instance)


def load_checkpoint_data(filename: str) -> Dict[str, Any]:
    """Loads simulation state data from a checkpoint file."""
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")

    logging.info(f"Loading state from checkpoint: {filename}")
    loaded_state = {}
    try:
        with netCDF4.Dataset(filename, "r") as ds:
            if (
                "checkpoint_time" not in ds.ncattrs()
                or "checkpoint_step" not in ds.ncattrs()
            ):
                raise KeyError("Checkpoint missing time/step attributes.")
            loaded_state["current_t"] = ds.getncattr("checkpoint_time")
            loaded_state["current_step"] = ds.getncattr("checkpoint_step")

            required_vars = [
                "cell_vars",
                "p2_nodes",
            ]  # Add others if they are saved in save_checkpoint
            for var_name in required_vars:
                if var_name not in ds.variables:
                    raise KeyError(
                        f"Checkpoint missing required variable: '{var_name}'"
                    )
                loaded_state[var_name] = ds.variables[var_name][:]

            logging.info(
                f"Successfully loaded state for step {loaded_state['current_step']}, t={loaded_state['current_t']:.4f}"
            )
    except Exception as e:
        logging.error(f"Failed to load checkpoint file {filename}: {e}", exc_info=True)
        raise IOError(f"Error reading checkpoint file {filename}: {e}") from e
    return loaded_state
