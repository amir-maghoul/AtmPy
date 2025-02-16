""" Module for abstract base class for test cases."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from atmpy.configuration.simulation_configuration import (
    SimulationConfig,
)
from atmpy.configuration.simulation_data import BoundaryFace
from atmpy.infrastructure.enums import BoundarySide, BoundaryConditions as BdryType


class BaseTestCase(ABC):
    def __init__(self, name: str, config: SimulationConfig):
        self.name = name
        self.config = config
        self.parameters: Dict[str, Any] = {}
        self.initial_conditions: Dict[str, Any] = {}
        self.boundary_conditions: Dict[BoundarySide, BdryType] = {}

    def set_boundary_condition(
        self,
        boundary_side: BoundarySide,
        condition: BdryType,
    ):
        """Update the boundary condition for a given boundary side.

        Parameters
        ----------
        boundary_side : BoundarySide
            The boundary side to update
        condition : BdryType
            The new boundary condition
        """
        self.config.update_boundary_condition(boundary_side, condition)
        self.boundary_conditions[boundary_side] = condition

    def set_global_constants(self, global_constant_updates: Dict[str, float]):
        """Update the global constants in the configuration.

        Parameters
        ----------
        global_constant_updates : dict
            The new global constant data
        """
        self.config.update_from_kwargs(global_constants=global_constant_updates)

    def set_temporal(self, temporal_updates: Dict[str, Any]):
        """ Update the temporal configuration as a bulk. Single changes should be done manually in the child class.

        Parameters
        ---------
        temporal_updates : dict
            The new temporal data
        """
        self.config.update_from_kwargs(temporal=temporal_updates)

    def set_boundary_conditions(self, bc_updates: Dict[BoundarySide, BdryType]):
        """ Update the boundary conditions in the configuration.

        Parameters
        ----------
        bc_updates : Dict[BoundarySide, BdryType]
            The new boundary condition data.
        """
        self.config.update_boundary_condition(boundary_conditions=bc_updates)


    def set_physics(self, physics_updates: dict):
        """ Update the physics configuration as a bulk. Single changes should be done manually in the child class.

        Parameters
        ----------
        physics_updates : dict
            The new physics data
        """
        self.config.update_from_kwargs(physics=physics_updates)

    def set_model_regimes(self, model_regime_updates: dict):
        """ Update the model regime configuration as a bulk. Single changes should be done manually in the child class."""

        self.config.update_from_kwargs(model_regimes=model_regime_updates)

    def numerics(self, numerics_updates: dict):
        """ Update the numerics configuration as a bulk. Single changes should be done manually in the child class."""

        self.config.update_from_kwargs(numerics=numerics_updates)

    def set_parameters(self, parameters_updates: dict):
        """ Update the parameters configuration as a bulk. Single changes should be done manually in the child class."""

        self.config.update_from_kwargs(parameters=parameters_updates)

    def set_diagnostics(self, diagnostics_updates: dict):
        """ Update the diagnostics configuration as a bulk. Single changes should be done manually in the child class."""

        self.config.update_from_kwargs(diagnostics=diagnostics_updates)

    def set_output(self, output_updates: dict):
        """ Update the output configuration as a bulk. Single changes should be done manually in the child class."""

        self.config.update_from_kwargs(output=output_updates)

    @abstractmethod
    def setup(self):
        """Initialize the test case with specific parameters, initial conditions, and boundary conditions."""
        pass
