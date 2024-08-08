import numpy as np
from dataclasses import dataclass


@dataclass
class PyTestCases1D:

    success = [(np.array([[0, 3], [0, 0], [0, 0]]), [5, 0, 0], [2, 0, 0])]

    fail = [
        (np.zeros((3, 2)), np.zeros(3), np.zeros(3)),
        (np.zeros((3, 2)), [1, 0, 0], np.zeros(3)),
        (np.zeros((3, 2)), np.zeros(3), [1, 0, 0]),
        (np.array([[0, 1], [0, 0], [0, 0]]), np.zeros(3), np.zeros(3)),
        (np.array([[0, 1], [0, 0], [0, 0]]), [0, 1, 0], np.zeros(3)),
        (np.array([[0, 1], [0, 0], [0, 0]]), np.zeros(3), [0, 1, 0]),
        (np.array([[0, 1], [0, 0], [0, 0]]), [1, 0, 0], [0, 1, 0]),
        (np.array([[0, 1], [0, 0], [0, 0]]), [1, 0, 0], [0, 0, 1]),
        (np.array([[0, 1], [0, 0], [0, 0]]), [0, 1, 0], [0, 0, 1]),
        (np.array([[0, 1], [0, 0], [0, 0]]), [0, 1, 0], [0, 0, 1]),
        (np.array([[0, 0], [0, 0], [0, 1]]), [1, 0, 0], [1, 0, 0]),
        (np.array([[0, 0], [0, 1], [0, 0]]), [1, 0, 0], [1, 0, 0]),
        (np.array([[0, 0], [0, -1], [0, 0]]), [1, 0, 0], [1, 0, 0]),
        (np.array([[0, -1], [0, 0], [0, 0]]), [1, 0, 0], [1, 0, 0]),
    ]
