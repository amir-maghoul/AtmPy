# The top-level solver knows nothing but the interface to a time integrator.
#
# class Solver:
#     def __init__(self, time_integrator: AbstractTimeIntegrator):
#         self.time_integrator = time_integrator
#         self.sim_time = 0.0
#
#     def run(self, nsteps: int):
#         for _ in range(nsteps):
#             self.time_integrator.step()
#             self.sim_time += self.time_integrator.get_dt()
