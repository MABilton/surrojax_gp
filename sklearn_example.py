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

<<<<<<< HEAD
if __name__ == "__main__":
    #  First the noiseless case
    X = jnp.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    # Observations:
    Y = (f(X).ravel())
    Y = Y.reshape(len(Y),1)
    x =  jnp.atleast_2d(jnp.linspace(0, 10, 1000)).T
    constraints = {"const": {">": 10**(-2), "<": 10**(2)}, "length": {">": 10**(-1), "<": 10**(1)}} # "noise":None
    surrogate = oed_gp.GP_Surrogate(kernel, X, Y, constraints)
    y = surrogate.predict(x)
    # Plotting: 
=======
# Helper function to plot training data and GP predictions:
def plot_gp(save_name, x_pred, y_pred, x_train, y_train):
>>>>>>> d5204f7b97c2516fa81a7571e1b024376d918dbb
    fig = plt.figure()
    plt.plot(x_pred, f(x_pred), 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.plot(x_train, y_train, 'r.', markersize=10, label='Observations')
    plt.plot(x_pred, y_pred[0], 'b-', label='Prediction')
    mean_minus_std = y_pred[0] - 1.9600 * jnp.sqrt(y_pred[1])
    mean_plus_std = y_pred[0] + 1.9600 * jnp.sqrt(y_pred[1])
    plt.fill(jnp.concatenate([x_pred, x_pred[::-1]]),
         jnp.concatenate([mean_minus_std,
                        (mean_plus_std)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.savefig(save_name+'.png', dpi=300)

if __name__ == "__main__":
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

<<<<<<< HEAD
    # Observations and noise
    Y = f(X).ravel()
    dy = 0.5 + 1.0 * np.random.random(Y.shape)
    noise = np.random.normal(0, dy)
    Y += noise
    constraints = {"const": {">": 10**(-2), "<": 10**(2)}, 
                   "length": {">": 10**(-1), "<": 10**(1)}, 
                   "noise":{">":10**(-3), "<":10**(2)}} 
    surrogate = oed_gp.GP_Surrogate(kernel, X, Y, constraints)
    x =  jnp.atleast_2d(jnp.linspace(0, 10, 1000)).T
    y = surrogate.predict(x)
    # Plot the function, the prediction and the 95% confidence interval based on
    # the MSE
    plt.figure()
    plt.plot(x, f(x), 'r:', label=r'$f(x) = x\,\sin(x)$')
    plt.errorbar(X.ravel(), Y, dy, fmt='r.', markersize=10, label='Observations')
    plt.plot(x, y[0], 'b-', label='Prediction')
    plt.fill(np.concatenate([x, x[::-1]]),
            np.concatenate([y[0] - 1.9600 * jnp.sqrt(y[1]),
                            (y[0] + 1.9600 * jnp.sqrt(y[1]))[::-1]]),
            alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.ylim(-10, 20)
    plt.legend(loc='upper left')
    plt.savefig('sklearn_example_noise.png', dpi=600)
=======
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
>>>>>>> d5204f7b97c2516fa81a7571e1b024376d918dbb
