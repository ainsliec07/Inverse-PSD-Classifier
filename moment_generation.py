import numpy as np
from simulation_utils import BeamPropagationSimulation

# Original repository configuration
basis_modes = [(0, 1), (1, 4), (0, -6), (1, 8)]

sim_params = {
    "Lx": 64.0,
    "Nx": 256,       # smaller grid for initial testing
    "z": 5.0,
    "dz": 1.0 / 32.0,
    "sigma": 1.1,
    "l0": 1.5,
    "w0": 4.0,
    "total_power": 1.0e5,
}

# Choose one fixed source beam.
# "1000" selects the first basis mode: (0, 1).
code = "1000"

N = 10  # start with 10 independent realizations

M1_sum = np.zeros(
    (sim_params["Nx"], sim_params["Nx"]),
    dtype=np.complex128
)

M2_sum = np.zeros(
    (sim_params["Nx"], sim_params["Nx"]),
    dtype=np.float64
)

for j in range(N):
    simulation = BeamPropagationSimulation({
        "basis_modes": basis_modes,
        "code": code,
        "sim_params": sim_params,
    })

    simulation.run()

    u = simulation.get_final_field()

    M1_sum += u
    M2_sum += np.abs(u)**2

M1 = M1_sum / N
M2_diagonal = M2_sum / N