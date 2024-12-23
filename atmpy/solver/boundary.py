from atmpy.grid.kgrid import Grid


def mirror_boundary():
    pass


def linear_boundary():
    pass


def quadratic_boundary():
    pass


def periodic_boundary():
    pass


def relaxation_boundary():
    pass

# def apply_transmissive(grid, variable, side):
#     """
#     Apply transmissive (zero-gradient) boundary condition.
#
#     Parameters:
#         grid : :py:class:`atmpy.grid.kgrid.Grid`
#             Grid object on which the problem is based.
#         variable : :py:class:`atmpy.variable.Variable`
#             The variable container of the problem
#         side : str
#             The boundary side to handle. Values are 'low' or 'high'.
#     """
#
#     dim = grid.dimension
#
#     if side == 'low':
#         # Set ghost cell equal to the first interior cell
#         slices = [slice(None)] * (1 + dim)
#         slices[dim + 1] = 0
#         interior_slice = [slice(None)] * (1 + dim)
#         interior_slice[dim + 1] = 1
#         self.variables.U[tuple(slices)] = variables.U[tuple(interior_slice)].copy()
#     elif side == 'high':
#         # Set ghost cell equal to the last interior cell
#         slices = [slice(None)] * (1 + self.grid.ndim)
#         slices[dim + 1] = -1
#         interior_slice = [slice(None)] * (1 + self.grid.ndim)
#         interior_slice[dim + 1] = -2
#         self.variables.U[tuple(slices)] = self.variables.U[tuple(interior_slice)].copy()
#
# def apply_periodic(self, dim, side):
#     """
#     Apply periodic boundary condition.
#
#     Parameters:
#         dim (int): Dimension index.
#         side (str): 'low' or 'high'.
#     """
#     if side == 'low':
#         # Set ghost cell equal to the last interior cell of the opposite side
#         slices = [slice(None)] * (1 + self.grid.ndim)
#         slices[dim + 1] = 0
#         opposite_slice = [slice(None)] * (1 + self.grid.ndim)
#         opposite_slice[dim + 1] = -2
#         self.variables.U[tuple(slices)] = self.variables.U[tuple(opposite_slice)].copy()
#     elif side == 'high':
#         # Set ghost cell equal to the first interior cell of the opposite side
#         slices = [slice(None)] * (1 + self.grid.ndim)
#         slices[dim + 1] = -1
#         opposite_slice = [slice(None)] * (1 + self.grid.ndim)
#         opposite_slice[dim + 1] = 1
#         self.variables.U[tuple(slices)] = self.variables.U[tuple(opposite_slice)].copy()
#
# def apply_reflective(self, dim, side):
#     """
#     Apply reflective (wall) boundary condition.
#
#     Parameters:
#         dim (int): Dimension index.
#         side (str): 'low' or 'high'.
#     """
#     if side == 'low':
#         # Mirror the first interior cell into the ghost cell
#         slices = [slice(None)] * (1 + self.grid.ndim)
#         slices[dim + 1] = 0
#         interior_slice = [slice(None)] * (1 + self.grid.ndim)
#         interior_slice[dim + 1] = 1
#         self.variables.U[tuple(slices)] = self.variables.U[tuple(interior_slice)].copy()
#
#         # Reverse the normal velocity component
#         self.variables.U[1 + dim, tuple(slices)] *= -1  # Reverse velocity in this dimension
#     elif side == 'high':
#         # Mirror the last interior cell into the ghost cell
#         slices = [slice(None)] * (1 + self.grid.ndim)
#         slices[dim + 1] = -1
#         interior_slice = [slice(None)] * (1 + self.grid.ndim)
#         interior_slice[dim + 1] = -2
#         self.variables.U[tuple(slices)] = self.variables.U[tuple(interior_slice)].copy()
#
#         # Reverse the normal velocity component
#         self.variables.U[1 + dim, tuple(slices)] *= -1  # Reverse velocity in this dimension
