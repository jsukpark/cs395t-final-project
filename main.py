"""Main code
"""

from functools import partial

import jax.numpy as np
from jax import random
from trajax.optimizers import ilqr
import matplotlib.pyplot as plt


def example_cost(x, u, t) -> float:
    """A cost function just for code verification
    
    Spring potential energy with Hooke's constant k = 1
    times the l2 norm of force squared (hence quadratic)
    cost = (0.5 * pos**2) + ||force||^2
    """
    C = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]) * 0.5
    return x @ C @ x + u @ u


def _dyn(x, u, dt=1e-2, mass=1.0):
    pos, vel, acc = x
    newpos = pos + vel*dt + acc*(dt*dt*0.5)
    newacc = -newpos + u/mass
    newvel = vel + (acc+newacc)*(dt*0.5)
    return (newpos, newvel, newacc)


def test_dyn():
    from math import cos
    x0 = (1.0, 0.0, -1.0)
    dt = 1e-2
    tarray = [dt*t for t in range(1001)]
    X = [x0]
    for t in tarray[1:]:
        xt = _dyn(X[-1], 0, dt=dt)
        X.append(xt)
    xarray, _, _ = zip(*X)
    plt.plot(tarray, [cos(t) for t in tarray], 'r-', label='expected')
    plt.plot(tarray, xarray, 'b:', label='observed')
    plt.legend()
    plt.show()


def example_dynamics(x, u, t, dt=1e-2, mass=1.0):
    """A dynamics function just for code verification

    A simple harmonic oscillator with mass = 1 around x = 0
    """
    A = np.array([
        list(_dyn((1., 0., 0.), 0., dt=dt, mass=mass)),
        list(_dyn((0., 1., 0.), 0., dt=dt, mass=mass)),
        list(_dyn((0., 0., 1.), 0., dt=dt, mass=mass)),
    ]).T
    B = np.array([
        list(_dyn((0., 0., 0.), 1., dt=dt, mass=mass)),
    ]).T
    return A @ x + B @ u


if __name__ == '__main__':
    key = random.PRNGKey(0)
    T = 1000
    DELTA_T = 1e-1
    MASS = 1.0
    x0 = np.array([1.0, 0.0, -1.0])
    U0 = np.zeros((T, 1))
    X, U, obj, grad, adjoints, lqr, iteration = ilqr(
        example_cost,
        partial(example_dynamics, dt=DELTA_T, mass=MASS),
        x0,
        U0,
        grad_norm_threshold=1e-12,
        maxiter=1000,
    )
    tarray = np.arange(T+1)*DELTA_T
    X0 = [x0]
    for t in range(T):
        X0.append(
            example_dynamics(
                X0[-1], np.zeros(1), t*DELTA_T,
                dt=DELTA_T, mass=MASS,
            )
        )
    X0 = np.array(X0)
    plt.figure(figsize=(14., 6.))
    plt.plot(tarray, np.zeros_like(tarray), 'k-')
    plt.plot(tarray, X0[:, 0], 'r--',
             label='trajectory without control')
    plt.plot(tarray, X[:, 0], 'b-',
             label='trajectory with optimal control')
    # plt.plot(tarray[:-1], U0[:, 0], 'g:', label='initial control')
    # plt.plot(tarray[:-1], U[:, 0], 'r:', label='optimal control')
    plt.xlim(0, T*DELTA_T)
    plt.xlabel('Time')
    plt.ylabel('Displacement from 0')
    plt.title(f'obj = {obj}')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()
