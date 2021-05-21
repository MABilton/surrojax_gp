import jax.numpy as jnp

import numpy as np

import oed_gp

import random 

from math import pi

from matplotlib import pyplot as plt

from scipy.linalg import cholesky, cho_solve

random.seed(1)

def f(x):
    """The function to predict."""
    return x * jnp.sin(x)

def kernel(x_1, x_2, params):
    val = params["const"]*jnp.exp(-0.5*((x_2 - x_1)/params["length"])**2)
    return val

if __name__ == "__main__":
    #  First the noiseless case
    X = jnp.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    # Observations:
    Y = (f(X).ravel())
    Y = Y.reshape(len(Y),1)
    x =  jnp.atleast_2d(jnp.linspace(0, 10, 1000)).T
    constraints = {"const": {">": 10**(-3)}, "length": {">": 10**(-3)}} # "noise":None
    surrogate = oed_gp.GP_Surrogate(kernel, X, Y, constraints)
    y = surrogate.predict(x)
    # Plotting: 
    fig = plt.figure()
    plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.plot(X, Y, 'r.', markersize=10, label='Observations')
    plt.plot(x, y[0], 'b-', label='Prediction')
    plt.fill(jnp.concatenate([x, x[::-1]]),
         jnp.concatenate([y[0] - 1.9600 * y[1],
                        (y[0] + 1.9600 * y[1])[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.savefig('sklearn_example_1_noiseless.png', dpi=600)

    # ----------------------------------------------------------------------
    # now the noisy case
    X = jnp.linspace(0.1, 9.9, 20)
    X = jnp.atleast_2d(X).T

    # Observations and noise
    Y = f(X).ravel()
    dy = 0.5 + 1.0 * np.random.random(Y.shape)
    noise = np.random.normal(0, dy)
    Y += noise
    constraints = {"const": {">": 10**(-3)}, "length": {">": 10**(-3)}, "noise":{">":10**(-3)}} 
    surrogate = oed_gp.GP_Surrogate(kernel, X, Y, constraints)
    y = surrogate.predict(x)

    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    plt.figure()
    plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.errorbar(X.ravel(), Y, dy, fmt='r.', markersize=10, label='Observations')
    plt.plot(x, y[0], 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y[0] - 1.9600 * y[1],
                            (y[0] + 1.9600 * y[1])[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.savefig('sklearn_example_1_noise.png', dpi=600)
