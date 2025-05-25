""" Module to handle the visualization functionality."""

import logging
import os
from typing import List, Optional, Union

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

logger = logging.getLogger(__name__)  # Use a dedicated logger for this module


def _get_time_slice(
    ds: xr.Dataset, time_indices: Optional[List[int]]
):
    """
    Helper function to determine the time slice for plotting.
    For static plots, expects one index (or defaults to last).
    For animation, expects two indices for start and end (or defaults to all).
    """
    num_times = len(ds["time"])

    if time_indices is None or not time_indices:  # Default behavior
        return (
            -1
        )  # Default to last time step for static, or all for animation (handled later)

    # Ensure indices are within bounds
    processed_indices = []
    for ti in time_indices:
        if ti < 0:
            processed_indices.append(num_times + ti)  # Convert negative index
        else:
            processed_indices.append(ti)
        if not (0 <= processed_indices[-1] < num_times):
            raise IndexError(
                f"Time index {ti} (resolved to {processed_indices[-1]}) is out of bounds for dataset with {num_times} time steps."
            )

    return processed_indices


def plot_1d_static(
    ds: xr.Dataset, variable_name: str, time_index: int, ax: Optional[plt.Axes] = None
) -> None:
    """Plots a 1D variable at a specific time index."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.figure

    if variable_name not in ds:
        logger.error(f"Variable '{variable_name}' not found in dataset.")
        return
    if "x" not in ds.coords:
        logger.error("Coordinate 'x' not found for 1D plot.")
        return

    data_var = ds[variable_name]
    time_value = ds["time"].isel(time=time_index).item()
    if variable_name == "p2_nodes":
        x_coords = ds["x_nodes"].values
    else:
        x_coords = ds["x"].values

    ax.plot(
        x_coords,
        data_var.isel(time=time_index).data,
        marker="o",
        linestyle="-",
        label=f"{variable_name} (t={time_value:.2f}s)",
    )
    ax.set_xlabel("X coordinate")
    ax.set_ylabel(f"{variable_name} ({data_var.attrs.get('units', '')})")
    ax.set_title(f"1D Static Plot: {variable_name} at t={time_value:.2f}s")
    ax.legend()
    ax.grid(True)
    if ax is None:  # Only show if we created the figure here
        plt.show()


def animate_1d(
    ds: xr.Dataset,
    variable_name: str,
    time_indices_range: List[int],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> FuncAnimation:
    """Animates a 1D variable over a range of time indices."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if variable_name not in ds:
        logger.error(f"Variable '{variable_name}' not found in dataset.")
        return None
    if "x" not in ds.coords:
        logger.error("Coordinate 'x' not found for 1D animation.")
        return None

    data_var = ds[variable_name]
    x_coords = data_var["x"].values

    start_index, end_index = min(time_indices_range), max(time_indices_range)
    if start_index == end_index:  # Animate a single frame if range is just one step
        end_index += 1
    if end_index > len(ds["time"]):
        end_index = len(ds["time"])

    times_to_animate = ds["time"].isel(time=slice(start_index, end_index))
    frames_indices = list(range(start_index, end_index))

    (line,) = ax.plot([], [], marker="o", linestyle="-")

    data_min = data_var.isel(time=slice(start_index, end_index)).min().item()
    data_max = data_var.isel(time=slice(start_index, end_index)).max().item()
    ax.set_xlim(x_coords.min(), x_coords.max())
    ax.set_ylim(
        data_min - 0.1 * abs(data_max - data_min),
        data_max + 0.1 * abs(data_max - data_min),
    )

    def init():
        line.set_data([], [])
        ax.set_xlabel("X coordinate")
        ax.set_ylabel(f"{variable_name} ({data_var.attrs.get('units', '')})")
        ax.grid(True)
        return (line,)

    def update(frame_idx_in_animation):
        actual_time_index = frames_indices[frame_idx_in_animation]
        current_data = data_var.isel(time=actual_time_index).data
        time_value = times_to_animate.isel(time=frame_idx_in_animation).item()

        line.set_data(x_coords, current_data)
        ax.set_title(f"1D Animation: {variable_name} at t={time_value:.2f}s")
        return (line,)

    ani = FuncAnimation(
        fig, update, frames=len(frames_indices), init_func=init, blit=True, interval=100
    )
    return ani


def plot_2d_static(
    ds: xr.Dataset, variable_name: str, time_index: int, ax: Optional[plt.Axes] = None
) -> None:
    """Plots a 2D variable at a specific time index."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    if variable_name not in ds:
        logger.error(f"Variable '{variable_name}' not found in dataset.")
        return
    if not ("x" in ds.coords and "y" in ds.coords):
        logger.error("Coordinates 'x' and 'y' not found for 2D plot.")
        return

    data_var = ds[variable_name]
    time_value = ds["time"].isel(time=time_index).item()
    if variable_name == "p2_nodes":
        x_coords = ds["x_node"].values
        y_coords = ds["y_node"].values
    else:
        x_coords = ds["x"].values
        y_coords = ds["y"].values

    plot_data = data_var.isel(time=time_index).data
    # contourf expects Z with dimensions (Y, X)
    # If data_var is (time, x, y), then .data.T might be needed if x is first dim after time
    # If data_var is (time, y, x), then .data is fine.
    # Assuming xarray loads it as (time, y, x) or (time, x, y) consistently based on NetCDF.
    # If ds[variable_name] is (time, dim0, dim1) and x_coords correspond to dim0, y_coords to dim1,
    # then plot_data needs to be (len(y_coords), len(x_coords)).
    # Let's assume ds[variable_name].isel(time=time_index) gives (nx, ny) based on original code.
    # Matplotlib contourf expects Z to be (ny, nx) if X and Y are 1D arrays of len nx and ny respectively.
    # So, if data is (nx, ny), we need data.T
    if plot_data.shape[0] == len(x_coords) and plot_data.shape[1] == len(y_coords):
        plot_data_for_contour = plot_data.T  # Transpose if (nx, ny)
    elif plot_data.shape[0] == len(y_coords) and plot_data.shape[1] == len(x_coords):
        plot_data_for_contour = plot_data  # Already (ny, nx)
    else:
        logger.error(
            f"Data shape {plot_data.shape} for variable {variable_name} does not match x ({len(x_coords)}) and y ({len(y_coords)}) coordinate lengths."
        )
        return

    cmap = "viridis"
    data_min = np.nanmin(plot_data_for_contour)
    data_max = np.nanmax(plot_data_for_contour)
    levels = np.linspace(data_min, data_max, 15) if data_min != data_max else 15

    contour = ax.contourf(
        x_coords, y_coords, plot_data_for_contour, cmap=cmap, levels=levels
    )

    cbar = fig.colorbar(contour, ax=ax, orientation="vertical")
    cbar.set_label(f"{variable_name} ({data_var.attrs.get('units', '')})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(f"2D Static Plot: {variable_name} at t={time_value:.2f}s")
    ax.set_aspect("equal", adjustable="box")
    if ax is None:
        plt.show()


def animate_2d(
    ds: xr.Dataset,
    variable_name: str,
    time_indices_range: List[int],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> FuncAnimation:
    """Animates a 2D variable over a range of time indices."""
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))

    if variable_name not in ds:
        logger.error(f"Variable '{variable_name}' not found in dataset.")
        return None
    if not ("x" in ds.coords and "y" in ds.coords):
        logger.error("Coordinates 'x' and 'y' not found for 2D animation.")
        return None

    data_var = ds[variable_name]
    if variable_name == "p2_nodes":
        x_coords = ds["x_node"].values
        y_coords = ds["y_node"].values
    else:
        x_coords = ds["x"].values
        y_coords = ds["y"].values

    start_index, end_index = min(time_indices_range), max(time_indices_range)
    if start_index == end_index:  # Animate a single frame if range is just one step
        end_index += 1
    if end_index > len(ds["time"]):
        end_index = len(ds["time"])

    times_to_animate = ds["time"].isel(time=slice(start_index, end_index))
    frames_indices = list(range(start_index, end_index))

    # Determine consistent color limits for the animation
    anim_data_slice = data_var.isel(time=slice(start_index, end_index))
    data_min = np.nanmin(anim_data_slice.data)
    data_max = np.nanmax(anim_data_slice.data)
    cmap = "viridis"
    levels = np.linspace(data_min, data_max, 15) if data_min != data_max else 15

    # Initial plot for structure
    # Determine transpose need based on first frame
    first_frame_data = data_var.isel(time=start_index).data
    if first_frame_data.shape[0] == len(x_coords) and first_frame_data.shape[1] == len(
        y_coords
    ):
        initial_plot_data = first_frame_data.T
    else:  # Assuming (ny, nx)
        initial_plot_data = first_frame_data

    cont = ax.contourf(x_coords, y_coords, initial_plot_data, cmap=cmap, levels=levels)
    cbar = fig.colorbar(cont, ax=ax)
    cbar.set_label(f"{variable_name} ({data_var.attrs.get('units', '')})")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")

    def update(frame_idx_in_animation):
        nonlocal cont  # Allow modification of cont
        ax.clear()  # Clear previous frame's contours and title

        actual_time_index = frames_indices[frame_idx_in_animation]
        current_data_raw = data_var.isel(time=actual_time_index).data
        time_value = times_to_animate.isel(time=frame_idx_in_animation).item()

        if current_data_raw.shape[0] == len(x_coords) and current_data_raw.shape[
            1
        ] == len(y_coords):
            current_data_for_contour = current_data_raw.T
        else:
            current_data_for_contour = current_data_raw

        cont = ax.contourf(
            x_coords, y_coords, current_data_for_contour, cmap=cmap, levels=levels
        )

        # Re-set labels and title as clear() removes them
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title(f"2D Animation: {variable_name} at t={time_value:.2f}s")
        ax.set_aspect("equal", adjustable="box")
        # The colorbar is associated with `fig`, not `ax`, so it might persist or need careful handling
        # For simplicity, we are not re-creating/updating colorbar in each frame here.
        # If cbar needs to update dynamically (e.g. if levels change), it's more complex.
        return cont.collections  # FuncAnimation expects an iterable of artists

    ani = FuncAnimation(
        fig, update, frames=len(frames_indices), interval=0.2, blit=False
    )  # blit=True is tricky with contourf and clear()
    return ani


def plot_3d_static(
    ds: xr.Dataset, variable_name: str, time_index: int, ax: Optional[plt.Axes] = None
) -> None:
    """Placeholder for 3D static plotting."""
    if "x" not in ds.coords or "y" not in ds.coords or "z" not in ds.coords:
        logger.error("Coordinates 'x', 'y', and 'z' not found for 3D plot.")
        return
    logger.info(
        f"3D static plot for {variable_name} at time index {time_index} - Not yet implemented."
    )
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.set_title(f"3D Static Plot: {variable_name} (Not Implemented)")
    if ax is None:
        plt.show()


def animate_3d(
    ds: xr.Dataset,
    variable_name: str,
    time_indices_range: List[int],
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Optional[FuncAnimation]:
    """Placeholder for 3D animation."""
    if "x" not in ds.coords or "y" not in ds.coords or "z" not in ds.coords:
        logger.error("Coordinates 'x', 'y', and 'z' not found for 3D animation.")
        return None
    logger.info(
        f"3D animation for {variable_name} between time indices {time_indices_range} - Not yet implemented."
    )
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")  # Ensure 3D axis for placeholder

    # Basic animation structure
    def init():
        ax.set_title(f"3D Animation: {variable_name} (Not Implemented)")
        return []

    def update(frame):
        ax.set_title(f"3D Animation: {variable_name} frame {frame} (Not Implemented)")
        return []

    ani = FuncAnimation(
        fig,
        update,
        frames=len(range(min(time_indices_range), max(time_indices_range) + 1)),
        init_func=init,
        blit=False,
        interval=200,
    )
    return ani


def visualize_data(
    input_file: str,
    variable_name: str,
    plot_type: str,  # 'static' or 'animate'
    time_indices: Optional[
        List[int]
    ] = None,  # For static: single index. For animation: [start, end]
):
    """
    Main dispatcher for visualization.
    Loads data and calls the appropriate plotting function.
    """
    logger.info(f"Visualizing data from: {input_file}")
    logger.info(
        f"Variable: {variable_name}, Plot type: {plot_type}, Time indices: {time_indices}"
    )

    if not os.path.exists(input_file):
        logger.error(f"Input file not found: {input_file}")
        return

    try:
        ds = xr.open_dataset(input_file)
    except Exception as e:
        logger.error(f"Failed to open dataset {input_file}: {e}")
        return

    # Determine dimensionality from dataset attributes or coordinates
    ndim = ds.attrs.get("config_grid_ndim", None)
    if ndim is None:  # Fallback: count 'x', 'y', 'z' coordinates
        if "z" in ds.coords:
            ndim = 3
        elif "y" in ds.coords:
            ndim = 2
        elif "x" in ds.coords:
            ndim = 1
        else:
            logger.error("Could not determine dimensionality of the dataset.")
            return

    logger.info(f"Dataset dimensionality detected: {ndim}D")

    # Resolve time indices
    try:
        resolved_time_indices = _get_time_slice(ds, time_indices)
    except IndexError as e:
        logger.error(e)
        return

    # For static plots, we need a single index.
    # For animation, we need a range (start, end).
    # _get_time_slice returns a list of resolved indices.

    fig, ax = plt.subplots(figsize=(12, 7))  # Create figure and axes once
    animation_object = None  # To store FuncAnimation object

    if plot_type == "static":
        static_time_index = (
            resolved_time_indices[0]
            if isinstance(resolved_time_indices, list) and resolved_time_indices
            else -1
        )  # Default to last if not specified or empty
        if isinstance(resolved_time_indices, list) and len(resolved_time_indices) > 1:
            logger.warning(
                f"Multiple time indices provided for static plot: {resolved_time_indices}. Using the first one: {static_time_index}"
            )
        elif isinstance(
            resolved_time_indices, int
        ):  # If _get_time_slice returned single int for default
            static_time_index = resolved_time_indices

        if ndim == 1:
            plot_1d_static(ds, variable_name, static_time_index, ax=ax)
        elif ndim == 2:
            plot_2d_static(ds, variable_name, static_time_index, ax=ax)
        elif ndim == 3:
            plot_3d_static(
                ds, variable_name, static_time_index, ax=ax
            )  # ax might need to be 3D for this
        else:
            logger.error(f"Unsupported dimensionality for static plot: {ndim}")

    elif plot_type == "animate":
        anim_time_indices_range: List[int]
        if isinstance(resolved_time_indices, list) and len(resolved_time_indices) == 1:
            # If only one index given for animation, animate that single frame (or from 0 to that frame)
            # For simplicity, let's treat it as a range from 0 to that index, or index to index+1
            logger.info(
                f"Single time index {resolved_time_indices[0]} for animation. Animating from index 0 to {resolved_time_indices[0]}."
            )
            # anim_time_indices_range = [0, resolved_time_indices[0]] # Option 1: 0 to index
            anim_time_indices_range = [
                resolved_time_indices[0],
                resolved_time_indices[0],
            ]  # Option 2: effectively one frame, but _animate functions handle start==end
        elif (
            isinstance(resolved_time_indices, list) and len(resolved_time_indices) >= 2
        ):
            anim_time_indices_range = [
                resolved_time_indices[0],
                resolved_time_indices[-1],
            ]  # Use first and last of provided list
            if len(resolved_time_indices) > 2:
                logger.warning(
                    f"More than two time indices provided for animation: {resolved_time_indices}. Using first ({resolved_time_indices[0]}) and last ({resolved_time_indices[-1]}) as range."
                )
        else:  # Default to full animation
            anim_time_indices_range = [0, len(ds["time"]) - 1]
            logger.info(
                f"No specific time range for animation. Animating all {len(ds['time'])} time steps."
            )

        if ndim == 1:
            animation_object = animate_1d(
                ds, variable_name, anim_time_indices_range, fig=fig, ax=ax
            )
        elif ndim == 2:
            if fig.axes[0].name == "2d":
                fig.clear()  # Clear figure
                ax = fig.add_subplot(111)  # Add new 2D axes
            animation_object = animate_2d(
                ds, variable_name, anim_time_indices_range, fig=fig, ax=ax
            )
        elif ndim == 3:
            if fig.axes[0].name != "3d":
                fig.clear()
                ax = fig.add_subplot(111, projection="3d")
            animation_object = animate_3d(
                ds, variable_name, anim_time_indices_range, fig=fig, ax=ax
            )
        else:
            logger.error(f"Unsupported dimensionality for animation: {ndim}")
    else:
        logger.error(f"Unknown plot type: {plot_type}")

    if animation_object:
        plt.tight_layout()
        plt.show()  # plt.show() for animations needs to be called after FuncAnimation
        # and animation_object must be kept in scope.
    elif (
        plot_type == "static"
    ):  # For static plots, show if not shown by individual plot func
        plt.tight_layout()
        plt.show()

    if ds:
        ds.close()
