# """ This module tests if GPU is available, if so it sets the backend to CuPy by default. One can switch between
#  backends by using the set_backend function."""
#
# import importlib
#
# # Initialize backend variables
# xp = None
# current_backend = None
#
#
# def set_backend(name):
#     global xp, asnumpy, current_backend
#     if name == "cupy":
#         try:
#             cupy = importlib.import_module("cupy")
#             xp = cupy
#             current_backend = "cupy"
#             print("Switched to CuPy.")
#         except ImportError:
#             raise ImportError("CuPy is not available.")
#     elif name == "numpy":
#         import numpy
#
#         xp = numpy
#         current_backend = "numpy"
#         print("Switched to NumPy.")
#     else:
#         raise ValueError("Unknown backend.")
#
#
# # Attempt to set CuPy as default, fallback to NumPy
# try:
#     set_backend("cupy")
# except ImportError:
#     set_backend("numpy")
#
# # Optionally, expose backend-related utilities
# # __all__ = __all__ + ['xp', 'set_backend', 'current_backend']
