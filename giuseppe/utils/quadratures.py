import numpy as np


class TrapezoidalQuadrature:
    """
    See Ascher 1995 Section 5.3.1

    sum beta = 1
    sum alpha_jl = rho_j

    """
    k = 2
    alpha = np.array([
        [0.,  0.],
        [0.5, 0.5]
    ])
    beta = np.array([0.5, 0.5])
    rho = np.array([0, 1])


class MidpointQuadrature:
    """
    See Ascher 1995 Section 5.3.1

    sum beta = 1
    sum alpha_jl = rho_j

    """
    k = 1
    alpha = np.array([[0.5]])
    beta = np.array([1])
    rho = np.array([0.5])


class SimpsonQuadrature:
    """
    See Ascher 1995 Section 5.3.1

    sum beta = 1
    sum alpha_jl = rho_j

    """
    k = 3
    alpha = np.array([
        [0.,   0.,   0.],
        [5/24, 1/3, -1/24],
        [1/6,  2/3,  1/6]
    ])
    beta = np.array([1/6, 2/3, 1/6])
    rho = np.array([0, 0.5, 1])
