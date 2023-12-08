"""Main code
"""

from typing import Any, Tuple, Callable, Optional
from functools import partial
from itertools import product

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import gymnasium as gym
import jax
from jax import numpy as jnp, vmap
from trajax.optimizers import ilqr
import matplotlib.pyplot as plt


PRNG_SEED = 0
KEY = jax.random.key(PRNG_SEED)

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


def obs_v1_to_v0(state: Tuple) -> Tuple:
    """Given a tuple of v1 Acrobat state variables
    (cos1, sin1, cos2, sin2, dtheta1, dtheta2),
    return its v0 version
    (theta1, theta2, dtheta1, dtheta2)
    """
    c1, s1, c2, s2, dtheta1, dtheta2 = state
    theta1 = np.arccos(c1)
    if s1 < 0:
        theta1 = -theta1
    theta2 = np.arccos(c2)
    if s2 < 0:
        theta2 = -theta2
    return theta1, theta2, dtheta1, dtheta2


def discretize_action(a: jnp.ndarray) -> int:
    """Convert each value `a[i]` in range [-1, 1]
    to index of nearest action (in {0, 1, 2}, where
    action[i] = a[i].round() + 1)
    """
    return a.round() + 1


def jnp_arr_to_tensor(a: jnp.ndarray) -> torch.Tensor:
    return torch.tensor(a.tolist())


def tensor_to_jnp_arr(t: torch.Tensor) -> jnp.ndarray:
    return jnp.array(t.numpy())


def guided_policy_search(
        policy: torch.nn.Module,
        cost: Callable,
        dynamics: Callable,
        env: gym.Env,
        x0: Optional[Tuple] = None,
        maxiter: int = 100,
        tol: float = 1e-9,
        num_steps_pred: int = 50,
        num_epochs: int = 100,
        alpha: float = 1e-3,
        ) -> torch.nn.Module:
    """Given an untrained parametrized policy (a neural network),
    train the policy function parameters in-place
    using guided policy search
    and return the trained policy

    Args:
        policy: (torch.nn.Module) x_t -> u_t (continuous var.)

        cost: (Callable) (x_t, u_t) -> c_t

        dynamics: (Callable) (x_t, u_t) -> x_{t+1}

        env: RL environment

        num_steps_pred: (int) number of time steps to predict

        num_epochs: (int) number of epochs for SGD

        convert_to_action: (Callable) postprocess control computed with iLQR
        to an element in `env.action_space`

        alpha: (float) Lagrange multiplier learning rate
    
    Returns:
        policy: (torch.nn.Module) x_t -> u_t
    """
    if x0 is None:
        x0 = env.reset(seed=PRNG_SEED)
        if (isinstance(x0, Tuple) and
                len(x0) == 2 and
                isinstance(x0[1], dict)):
            x0 = x0[0]
    x0 = obs_v1_to_v0(x0)
    policy.eval()
    action_dim = policy(torch.tensor(x0)).numel()
    # Define Lagrangian
    def lagrangian(x: jnp.ndarray, u: jnp.ndarray,
                   t: jnp.ndarray, l: jnp.ndarray) -> float:
        """Lagrangian for DGD"""
        if x.ndim == u.ndim == 2:
            assert t.ndim == 1 and l.ndim == u.ndim and \
                x.shape[0] == u.shape[0] == t.shape[0] == l.shape[0]
            return vmap(lagrangian)(x, u, t, l).sum()
        diff: torch.Tensor = policy(jnp_arr_to_tensor(x)) - jnp_arr_to_tensor(u)
        return cost(x, u, t) + l @ tensor_to_jnp_arr(diff)  # scalar
    
    lambda_ = list(jnp.zeros((num_steps_pred, action_dim)))

    # TODO: Below doesn't work. Somehow need to use the correct lambda_ row but there's no way to index them.
    @partial(jax.jit, static_argnums=2)
    def s_cost(x, u, t):
        """Surrogate cost"""
        # assert isinstance(t, int), f"{type(t)} != int"
        return lagrangian(x, u, t, lambda_[t])
    
    # iterative LQR loop
    for idx in range(maxiter):
        # Step 1 (minimizing over trajectory):
        # Find traj that minimizes Lagrangian(traj, params, lambda_) using iLQR
        policy.eval()
        X, U, _, _, _, _, _ = ilqr(
            s_cost,
            dynamics,
            jnp.array(x0),
            jnp.zeros((num_steps_pred, action_dim)),
        )
        # Step 2 (minimizing over parameters):
        # Find params that minimizes Lagrangian(traj, params, lambda_) using SGD
        assert X.shape[0] == U.shape[0] + 1
        data = list(zip(np.array(X), np.array(U)))
        loader = DataLoader(data, batch_size=1)
        # loader = [(torch.tensor(np.array(X)), torch.tensor(np.array(U)))]
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        policy.train()
        for epoch in range(num_epochs):
            for x, u in loader:
                optimizer.zero_grad()
                uhat = policy(x)
                loss = F.cross_entropy(uhat, u)
                loss.backward()
                optimizer.step()
            scheduler.step()
        # Step 3 (maximizing over Lagrange multiplier):
        # Find lambda_ that maximizes Lagrangian(traj, params, lambda_)
        dlambda = jax.grad(
            partial(lagrangian,
                    jnp_arr_to_tensor(X[:-1]),
                    jnp_arr_to_tensor(U),
                    jnp.arange(num_steps_pred),
                    ),
        )(lambda_)
        lambda_ = lambda_ + alpha * dlambda
        if jnp.linalg.norm(alpha * dlambda) < tol:
            break
    return policy


def cost(x, u, t) -> float:
    """Cost function

    Args:
        x: [theta1, theta2, dtheta1, dtheta2]
        u: [torque] (torque \in [-1, 1])
        t: int (time index)
    """
    cosines = jnp.cos(jnp.array([[1., 0., 0., 0.],
                                 [1., 1., 0., 0.]]) @ x)
    return max(0., 1. - jnp.array([1., 1.]) @ cosines) + (u @ u)


def dynamics(env, x, u, t) -> jnp.ndarray:
    """Dynamics function"""
    return env.step(discretize_action(u))[0]


class MyPolicy:
    def __init__(self):
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(4, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
            torch.nn.Tanh(),
        )
        self.tstep = 0
    
    def __call__(self, env, obs):
        if self.tstep % 50 == 0:
            self.policy = guided_policy_search(
                self.policy,
                cost,
                dynamics,
                init_env(),
                obs,
            )
        act = self.policy(torch.tensor(obs_v1_to_v0(obs))).numpy()
        assert act.shape == (1,)
        act = discretize_action(act)[0]
        self.tstep += 1
        return act
    
select_action = MyPolicy()

if __name__ == '__main__':
    policy = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.ReLU(),
        torch.nn.Linear(4, 1),
        torch.nn.Tanh(),
    )
