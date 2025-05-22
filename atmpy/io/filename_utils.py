""" Create filenames in a central manner"""

from __future__ import annotations  # For type hinting OutputConfig if defined elsewhere
import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from atmpy.configuration.simulation_configuration import Outputs as OutputConfig


def generate_output_filepath(
    config_outputs: OutputConfig,
    case_name: str,
    nx: int,
    ny: Optional[int] = None,
    nz: Optional[int] = None,
    ndim: int = 1,  # Dimensionality of the grid (1, 2, or 3)
) -> str:
    """
    Generates the standardized full filepath for analysis output files.

    Args:
        config_outputs: An object/dict containing output configuration
                        (output_path, output_folder, output_base_name,
                         output_suffix, output_extension).
        case_name: The name of the test case (e.g., "TravelingVortex").
        nx: Number of grid points in X.
        ny: Number of grid points in Y (required if ndim >= 2).
        nz: Number of grid points in Z (required if ndim >= 3).
        ndim: The dimensionality of the simulation (1, 2, or 3).

    Returns:
        The absolute or relative filepath string.
    """

    # Base path components from config_outputs
    base_path = getattr(config_outputs, "output_path", ".")

    # My example: /home/amir/Projects/Python/Atmpy/atmpy/atmpy/output_data/traveling_vortex/
    current_output_folder = os.path.join(
        getattr(config_outputs, "output_folder", "output_data")
    )

    # Filename stem construction (e.g., _traveling_vortex_64_64)
    # Based on your example: _case_name_nx[_ny][_nz]
    filename_stem_parts = [
        config_outputs.output_base_name
    ]  # Start with underscore and case name

    dim_values = [str(nx)]
    if ndim >= 2:
        if ny is None:
            raise ValueError(
                "ny must be provided for 2D/3D cases in filename generation."
            )
        dim_values.append(str(ny))
    if ndim >= 3:
        if nz is None:
            raise ValueError("nz must be provided for 3D cases in filename generation.")
        dim_values.append(str(nz))

    filename_stem_parts.append(
        "_".join(dim_values)
    )  # e.g., ["_traveling_vortex", "64_64"]
    filename_stem = "_".join(filename_stem_parts)

    full_path = os.path.join(
        base_path,
        current_output_folder,
        f"{filename_stem}{config_outputs.output_extension}",
    )
    return os.path.normpath(full_path)
