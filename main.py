"""Main code
"""

from functools import partial

import jax.numpy as np
from jax import random
from trajax.optimizers import ilqr
import matplotlib.pyplot as plt


def cost_test(x, u, t) -> float:
    """A cost function just for code verification"""
    C = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]) * 0.5
    return x @ C @ x


def dynamics_test(x, u, t, dt=1e-2, mass=1.0):
    """A dynamics function just for code verification

    A simple harmonic oscillator with mass = 1 around x = 0
    """
    A = np.array([
        [ 1,       dt,             0.5*dt             ],
        [-0.5*dt,  1 - 0.5*dt**2,  0.5*dt - 0.25*dt**2],
        [-1,      -dt,            -0.5*dt             ],
    ])
    B = np.array([
        [0],
        [0.5*dt / mass],
        [1 / mass],
    ])
    return A @ x + B @ u


if __name__ == '__main__':
    key = random.PRNGKey(0)
    T = 1000
    DELTA_T = 1e-2
    x0 = np.array([1.0, 0.0, 0.0])
    U0 = random.uniform(key, (T, 1), minval=-1., maxval=1.)
    X, U, obj, grad, adjoints, lqr, iteration = ilqr(
        cost_test,
        partial(dynamics_test, dt=DELTA_T),
        x0,
        U0,
        maxiter=1000,
    )
    tarray = np.arange(T+1)*DELTA_T
    plt.plot(tarray, X[:, 0], 'b-', label='optimal trajectory')
    plt.plot(tarray[:-1], U0[:, 0], 'g--', label='initial control')
    plt.plot(tarray[:-1], U[:, 0], 'r-', label='optimal control')
    plt.xlabel('Time')
    plt.ylabel('Displacement from 0')
    plt.title(f'obj = {obj}')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
