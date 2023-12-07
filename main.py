"""Main code
"""

from typing import Tuple
from functools import partial

import jax.numpy as jnp
from trajax.optimizers import ilqr
import matplotlib.pyplot as plt


def verify_ilqr():
    """
    Verify that `trajax.optimizers.ilqr()` works properly.

    We consider a toy system,
    a 1D simple harmonic oscillator around x = 0
    with mass = 1 and k (spring constant) = 1,
    where control u is the external force applied along the x-axis.

    Such system has the following dynamics:
    ```
    \ddot{q} = -q + u/mass
    ```

    We define a state to be a 3-tuple `(q, \dot{q}, \ddot{q})` of
    position, velocity, and acceleration.
    Then the next state is computed using velocity Verlet algorithm
    (https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet)
    """
    def example_cost(x: jnp.ndarray, u: jnp.ndarray, t: float) -> float:
        """Cost function to minimize
        for the 1D simple harmonic oscillator
        (must also minimize the amount of force applied)
        """
        C = jnp.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]) * 0.5
        return x @ C @ x + u @ u

    def _dyn(
            x: Tuple, u: float,
            dt: float = 1e-2, mass: float = 1.0,
            ) -> Tuple:
        """The velocity Verlet algorithm"""
        pos, vel, acc = x
        newpos = pos + vel*dt + acc*(dt*dt*0.5)
        newacc = -newpos + u/mass
        newvel = vel + (acc+newacc)*(dt*0.5)
        return (newpos, newvel, newacc)

    def test_dyn():
        """Test function for _dyn()"""
        from math import cos
        x0 = (1.0, 0.0, -1.0)
        dt = 1e-2
        mass = 1.0
        tarray = [dt*t for t in range(1001)]
        X = [x0]
        for t in tarray[1:]:
            xt = _dyn(X[-1], 0, dt=dt, mass=mass)
            X.append(xt)
        observed, _, _ = zip(*X)
        expected = [cos(t) for t in tarray]
        expected, observed = jnp.array(expected), jnp.array(observed)
        plt.plot(tarray, expected, 'b-', label='expected')
        plt.plot(tarray, observed, 'r--', label='observed')
        plt.xlim(min(tarray), max(tarray))
        plt.xlabel('Time')
        plt.ylabel('Displacement from 0')
        plt.title('Verifying the velocity Verlet algorithm')
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    test_dyn()

    def example_dynamics(
            x: jnp.ndarray,
            u: jnp.ndarray,
            t: float,
            dt: float = 1e-2,
            mass: float = 1.0,
            ) -> jnp.ndarray:
        """Dynamics function
        for the 1D simple harmonic oscillator
        """
        A = jnp.array([
            list(_dyn((1., 0., 0.), 0., dt=dt, mass=mass)),
            list(_dyn((0., 1., 0.), 0., dt=dt, mass=mass)),
            list(_dyn((0., 0., 1.), 0., dt=dt, mass=mass)),
        ]).T
        B = jnp.array([
            list(_dyn((0., 0., 0.), 1., dt=dt, mass=mass)),
        ]).T
        return A @ x + B @ u

    def trajectory_without_control(
            T: int,
            x0: jnp.ndarray,
            dt: float,
            mass: float,
            ) -> jnp.ndarray:
        """Return the trajectory of system states
        as an array of shape (T+1, x0.size)
        """
        X0 = [x0]
        for t in range(T):
            X0.append(
                example_dynamics(
                    X0[-1], jnp.zeros(1), t*dt,
                    dt=dt, mass=mass,
                )
            )
        X0 = jnp.array(X0)
        return X0

    # Verification code
    T = 1000
    DELTA_T = 1e-1
    MASS = 1.0
    x0 = jnp.array([1.0, 0.0, -1.0])
    U0 = jnp.zeros((T, 1))
    X, U, obj, grad, adjoints, lqr, iteration = ilqr(
        example_cost,
        partial(example_dynamics, dt=DELTA_T, mass=MASS),
        x0,
        U0,
        grad_norm_threshold=1e-12,
        maxiter=1000,
    )
    tarray = jnp.arange(T+1)*DELTA_T
    X0 = trajectory_without_control(T, x0, dt=DELTA_T, mass=MASS)
    plt.figure(figsize=(14., 6.))
    plt.plot(tarray, jnp.zeros_like(tarray), 'k-')
    plt.plot(tarray, X0[:, 0], 'r--',
             label='trajectory without control')
    plt.plot(tarray, X[:, 0], 'b-',
             label='trajectory with optimal control')
    plt.xlim(tarray.min(), tarray.max())
    plt.xlabel('Time')
    plt.ylabel('Displacement from 0')
    plt.title(f'obj = {obj}')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    verify_ilqr()
