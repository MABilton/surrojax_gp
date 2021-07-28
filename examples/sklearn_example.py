import sys
sys.path.append('../gp')
import numpy as np
import jax.numpy as jnp
from math import pi
from matplotlib import pyplot as plt
from gp_create import create_gp

np.random.seed(1)

# Function to predict:
def f(x):
    return x * jnp.sin(x)

# Kernel used by GP:
def kernel(x_1, x_2, params):
    val = params["const"]*jnp.exp(-0.5*((x_2 - x_1)/params["length"])**2)
    return val

# Helper function to plot training data and GP predictions:
def plot_gp(save_name, x_pred, y_pred, x_train, y_train):
    fig = plt.figure()
    plt.plot(x_pred, f(x_pred), 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
    plt.plot(x_pred, y_pred[0], 'b-', label='Prediction')
    mean, var = y_pred[0], y_pred[1]
    mean_minus_std = (mean - 1.9600 * jnp.sqrt(var)).squeeze()
    mean_plus_std = (mean + 1.9600 * jnp.sqrt(var)).squeeze()
    plt.fill_between(x_pred.squeeze(), mean_plus_std, mean_minus_std, alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.savefig(save_name+'.png', dpi=300)

def main():
    # Create noiseless dataset:
    x_train = jnp.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    y_train = (f(x_train).ravel())
    y_train = y_train.reshape(len(y_train),1)
    # Train Gaussian Process model:
    constraints = {"const": {">": 10**-2, "<": 10**2}, 
                   "length": {">": 10**-1, "<": 10**1}}
    surrogate = create_gp(kernel, x_train, y_train, constraints)
    # Plot predictions of GP model:
    x_pred =  jnp.atleast_2d(jnp.linspace(0, 10, 1000)).T
    y_pred = surrogate.predict(x_pred)
    plot_gp("sklearn_example_noiseless", x_pred, y_pred, x_train, y_train)

    # Create noisy dataset:
    x_train = jnp.linspace(0.1, 9.9, 20)
    x_train = jnp.atleast_2d(x_train).T
    y_train = f(x_train).ravel()
    noise = np.random.normal(loc=0, scale=0.5, size=y_train.size)
    y_train += noise
    # Train Gaussian Process model:
    constraints = {"const": {">": 10**-2, "<": 10**2}, 
                   "length": {">": 10**-1, "<": 10**1}, 
                   "noise": {">":10**-1, "<":10**1}} 
    surrogate = create_gp(kernel, x_train, y_train, constraints)
    # Plot predictions of GP model:
    x_pred = jnp.atleast_2d(jnp.linspace(0, 10, 1000)).T
    y_pred = surrogate.predict(x_pred)
    plot_gp("sklearn_example_noise", x_pred, y_pred, x_train, y_train)

if __name__ == "__main__":
    main()