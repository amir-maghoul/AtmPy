""" This is an abstract class for the test cases. It brings the given test case into a
uniform class for the whole project.

The code snippet is taken from PyBella+ Library by Ray Chew https://github.com/ray-chew/pyBELLA """


class TestCaseData(object):
    """Test Case class. It brings the given test case into a uniform class for the whole project."""

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def update_test_case(self, update):
        """Update the test case with the given update.

        Parameters
        ----------
        update : dict
            the update value for the test case
        """
        for key, value in update.items():
            setattr(self, key, value)
