def roe_solver(self, left_state, right_state, direction):
    raise NotImplementedError(f"Roe solver for {direction}-direction not implemented.")


def hllc_solver(self, left_state, right_state, direction):
    raise NotImplementedError(f"HLLC solver for {direction}-direction not implemented.")


def rusanov_solver(self, left_state, right_state, direction):
    raise NotImplementedError(
        f"Rusanov solver for {direction}-direction not implemented."
    )
