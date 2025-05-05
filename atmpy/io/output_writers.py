from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Dict, Tuple, Optional, Any

import netCDF4  # type: ignore
import numpy as np

if TYPE_CHECKING:
    from atmpy.grid.kgrid import Grid
    from atmpy.configuration.simulation_configuration import SimulationConfig

# Helper dictionaries outside class for clarity
DIM_MAP_CELLS = {0: "x", 1: "y", 2: "z"}
DIM_MAP_NODES = {0: "x_node", 1: "y_node", 2: "z_node"}
COORD_ATTRS = {
    "x": {"units": "m", "long_name": "X-coordinate (cell centers)"},
    "y": {"units": "m", "long_name": "Y-coordinate (cell centers)"},
    "z": {"units": "m", "long_name": "Z-coordinate (cell centers)"},
    "x_node": {"units": "m", "long_name": "X-coordinate (nodes)"},
    "y_node": {"units": "m", "long_name": "Y-coordinate (nodes)"},
    "z_node": {"units": "m", "long_name": "Z-coordinate (nodes)"},
}


class NetCDFWriter:
    """
    Handles writing simulation data to NetCDF4 files, respecting grid dimensionality.
    Supports analysis output (inner grid, time dim) & checkpoints (full grid, no time dim).
    """

    def __init__(
        self,
        filename: str,
        grid: "Grid",
        config: "SimulationConfig",
        mode: str = "w",
        is_checkpoint: bool = False,
    ):
        if is_checkpoint and mode != "w":
            raise ValueError("Checkpoints must be written in mode 'w' (overwrite).")

        self.filename = filename
        self.grid = grid
        self.config = config
        self.mode = mode
        self.is_checkpoint = is_checkpoint
        self.add_time_dim = not is_checkpoint
        self.ds: Optional[netCDF4.Dataset] = None
        self._time_index: int = 0

        # Directory creation logic... (same as before)
        try:
            directory = os.path.dirname(self.filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory)
                logging.info(f"Created output directory: {directory}")
        except OSError as e:
            logging.error(f"Failed to create directory {directory}: {e}")
            raise

        self._open_file()

    def _open_file(self):
        """Opens the NetCDF file and sets up structure."""
        try:
            self.ds = netCDF4.Dataset(self.filename, self.mode, format="NETCDF4")
            logging.info(
                f"Opened NetCDF file: {self.filename} (Mode: '{self.mode}', Checkpoint: {self.is_checkpoint})"
            )

            needs_setup = False
            if self.mode == "w":
                needs_setup = True
                self._time_index = 0
            elif self.mode == "a" and self.add_time_dim:
                if "time" in self.ds.dimensions:
                    self._time_index = len(self.ds.dimensions["time"])
                    logging.debug(
                        f"Appending analysis output. Next time index: {self._time_index}"
                    )
                else:
                    logging.warning(
                        "Append analysis mode, but 'time' dim missing. Re-initializing."
                    )
                    needs_setup = True
                    self._time_index = 0

            if needs_setup:
                self._define_dimensions()
                self._define_coordinate_variables()
                self._write_global_attributes()

        except Exception as e:
            logging.error(f"Failed to open/initialize NetCDF file {self.filename}: {e}")
            self.ds = None
            raise

    def _define_dimensions(self):
        """Defines spatial and time dimensions based on grid.ndim and writer type."""
        if self.ds is None:
            return
        logging.debug(
            f"Defining NetCDF dimensions (NDim={self.grid.ndim}, Checkpoint={self.is_checkpoint})..."
        )

        if self.add_time_dim and "time" not in self.ds.dimensions:
            self.ds.createDimension("time", None)
            logging.debug("  Created dimension: time (unlimited)")

        # Define dimensions ONLY for existing grid dimensions (0 to ndim-1)
        for i in range(self.grid.ndim):
            dim_name_cell = DIM_MAP_CELLS[i]
            dim_name_node = DIM_MAP_NODES[i]

            # Determine length based on checkpoint status
            cell_len = (
                self.grid.cshape[i] if self.is_checkpoint else self.grid.icshape[i]
            )
            node_len = (
                self.grid.nshape[i] if self.is_checkpoint else self.grid.inshape[i]
            )

            if cell_len > 0 and dim_name_cell not in self.ds.dimensions:
                self.ds.createDimension(dim_name_cell, cell_len)
                logging.debug(
                    f"  Created dimension: {dim_name_cell} (length: {cell_len})"
                )

            if node_len > 0 and dim_name_node not in self.ds.dimensions:
                self.ds.createDimension(dim_name_node, node_len)
                logging.debug(
                    f"  Created dimension: {dim_name_node} (length: {node_len})"
                )

        # --- Checkpoint Specific: Define dimension for last axis of cell_vars ---
        # This dimension doesn't have a coordinate variable.
        if self.is_checkpoint and "var_cell" not in self.ds.dimensions:
            # Assuming num_vars_cell is accessible, otherwise needs to be passed
            num_vars = (
                self.config.numerics.num_vars_cell
            )  # Example: Get from config if available
            # Or determine from a passed variable shape during write_snapshot_data? Less ideal.
            # Let's assume config holds it for now.
            if num_vars > 0:
                self.ds.createDimension("var_cell", num_vars)
                logging.debug(f"  Created dimension: var_cell (length: {num_vars})")

    def _define_coordinate_variables(self):
        """Defines and writes coordinate variables for existing dimensions."""
        if self.ds is None:
            return
        logging.debug(
            f"Defining NetCDF coordinate variables (NDim={self.grid.ndim}, Checkpoint={self.is_checkpoint})..."
        )

        if (
            self.add_time_dim
            and "time" in self.ds.dimensions
            and "time" not in self.ds.variables
        ):
            time_var = self.ds.createVariable("time", "f8", ("time",))
            time_var.units = "seconds"
            time_var.long_name = "Simulation time"
            logging.debug("  Created coordinate variable: time")

        inner_slice = self.grid.get_inner_slice()

        # Define coordinates ONLY for existing grid dimensions
        for i in range(self.grid.ndim):
            dim_name_cell = DIM_MAP_CELLS[i]
            dim_name_node = DIM_MAP_NODES[i]

            # Cell coordinate variable
            if (
                dim_name_cell in self.ds.dimensions
                and dim_name_cell not in self.ds.variables
            ):
                var = self.ds.createVariable(dim_name_cell, "f8", (dim_name_cell,))
                coord_data = self.grid.get_cell_coordinates(i)  # Use getter method
                attrs = COORD_ATTRS[dim_name_cell]
                var.setncatts(attrs)
                var[:] = (
                    coord_data if self.is_checkpoint else coord_data[inner_slice[i]]
                )
                logging.debug(f"  Created coordinate variable: {dim_name_cell}")

            # Node coordinate variable
            if (
                dim_name_node in self.ds.dimensions
                and dim_name_node not in self.ds.variables
            ):
                node_var = self.ds.createVariable(dim_name_node, "f8", (dim_name_node,))
                node_coord_data = self.grid.get_node_coordinates(i)  # Use getter method
                node_attrs = COORD_ATTRS[dim_name_node]
                node_var.setncatts(node_attrs)
                node_var[:] = (
                    node_coord_data
                    if self.is_checkpoint
                    else node_coord_data[inner_slice[i]]
                )
                logging.debug(f"  Created coordinate variable: {dim_name_node}")

    def _write_global_attributes(self):
        """Writes global metadata from the SimulationConfig."""
        # (No change needed here, logic is independent of ndim)
        if self.ds is None:
            return
        logging.debug("Writing NetCDF global attributes...")
        self.ds.description = f"Atmpy simulation output (NDim={self.grid.ndim}, Checkpoint={self.is_checkpoint})"
        self.ds.source = "Atmpy Simulation Framework"
        self.ds.Conventions = "CF-1.6"
        try:
            self.ds.setncattr("config_dt", self.config.temporal.dtfixed)
            self.ds.setncattr("config_grid_ndim", self.grid.ndim)
            self.ds.setncattr("config_grid_nx", self.grid.nx)
            if self.grid.ndim >= 2:
                self.ds.setncattr("config_grid_ny", self.grid.ny)
            if self.grid.ndim >= 3:
                self.ds.setncattr("config_grid_nz", self.grid.nz)
            # ... add other config params ...
        except Exception as e:
            logging.warning(f"Could not write some global attributes: {e}")

    def define_variable(
        self,
        var_name: str,
        dimensions: Tuple[str, ...],  # Tuple of dimension NAMES
        dtype: type = np.float64,
        attributes: Optional[Dict[str, Any]] = None,
        zlib: bool = True,
        complevel: int = 4,
    ) -> Optional[netCDF4.Variable]:
        """Defines a variable in the NetCDF dataset if it doesn't exist."""
        if self.ds is None:
            return None
        # Check if all specified dimensions exist in the dataset
        for dim_name in dimensions:
            if dim_name not in self.ds.dimensions:
                logging.error(
                    f"Cannot define variable '{var_name}'. Dimension '{dim_name}' not found in dataset."
                )
                return None  # Or raise error

        if var_name in self.ds.variables:
            # Optional: Check if existing variable matches dimensions/dtype?
            return self.ds.variables[var_name]

        logging.debug(
            f"Defining NetCDF variable: {var_name} with dimensions {dimensions}"
        )
        try:
            nc_dtype = np.dtype(dtype).str.replace("<", "").replace(">", "")
            var = self.ds.createVariable(
                var_name, nc_dtype, dimensions, zlib=zlib, complevel=complevel
            )
            if attributes:
                var.setncatts(attributes)
            return var
        except Exception as e:
            logging.error(f"Failed to define variable '{var_name}': {e}")
            return None

    def write_timestep_data(
        self,
        sim_time: float,
        data_dict: Dict[str, np.ndarray],
    ):
        """Writes data for multiple variables at a single time step (for analysis output)."""
        if self.is_checkpoint:
            raise TypeError("Cannot use write_timestep_data on a checkpoint writer.")
        if self.ds is None:
            raise IOError("Dataset not open.")
        if "time" not in self.ds.variables:
            raise ValueError("Time variable not defined.")

        time_var = self.ds.variables["time"]
        time_var[self._time_index] = sim_time
        logging.debug(
            f"Writing time-series data for time={sim_time:.4f} at index={self._time_index}"
        )

        for var_name, data_array in data_dict.items():
            if var_name in self.ds.variables:
                var = self.ds.variables[var_name]
                try:
                    write_slice = [slice(None)] * var.ndim
                    write_slice[0] = self._time_index  # Assumes time is first dim
                    expected_shape = var.shape[1:]
                    if data_array.shape == expected_shape:
                        var[tuple(write_slice)] = data_array
                    else:
                        logging.error(
                            f"Shape mismatch var '{var_name}'. Expected {expected_shape}, got {data_array.shape}."
                        )
                except Exception as e:
                    logging.error(
                        f"Failed write var '{var_name}' at index {self._time_index}: {e}"
                    )
            else:
                logging.warning(f"Var '{var_name}' not defined. Cannot write.")
        self._time_index += 1
        # self.ds.sync()

    def write_snapshot_data(
        self,
        data_dict: Dict[str, np.ndarray],
        global_attributes: Optional[Dict[str, Any]] = None,
    ):
        """Writes data for multiple variables for a single snapshot (for checkpoints)."""
        if not self.is_checkpoint:
            logging.warning("Using write_snapshot_data on non-checkpoint writer.")
        if self.ds is None:
            raise IOError("Dataset not open.")

        logging.debug("Writing snapshot data...")
        for var_name, data_array in data_dict.items():
            if var_name in self.ds.variables:
                var = self.ds.variables[var_name]
                try:
                    expected_shape = var.shape
                    if data_array.shape == expected_shape:
                        var[:] = data_array
                    else:
                        logging.error(
                            f"Shape mismatch var '{var_name}'. Expected {expected_shape}, got {data_array.shape}."
                        )
                except Exception as e:
                    logging.error(f"Failed write snapshot var '{var_name}': {e}")
            else:
                logging.warning(f"Var '{var_name}' not defined. Cannot write snapshot.")

        if global_attributes:
            logging.debug("Updating global attributes for snapshot...")
            for key, value in global_attributes.items():
                try:
                    self.ds.setncattr(key, value)
                except Exception as e:
                    logging.warning(f"Could not set global attr '{key}': {e}")
        self.ds.sync()

    def close(self):
        """Closes the NetCDF file."""
        if self.ds:
            try:
                self.ds.close()
                logging.info(f"Closed NetCDF file: {self.filename}")
                self.ds = None
            except Exception as e:
                logging.error(f"Failed to close NetCDF file {self.filename}: {e}")

    def __enter__(self):
        if self.ds is None:
            self._open_file()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
