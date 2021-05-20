import jax.numpy as jnp

import oed_gp

import random 

from matplotlib import pyplot as plt

random.seed(1)

def f(x):
    """The function to predict."""
    return x * jnp.sin(x)

def kernel(x_1, x_2, params):
    return params["const"]*jnp.exp(-0.5*((x_2 - x_1)/params["length"])**2)

if __name__ == "__main__":
    #  First the noiseless case
    X = jnp.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    # Observations:
    y = (f(X).ravel())
    y = y.reshape(len(y),1)
    x_pred =  jnp.atleast_2d(jnp.linspace(0, 10, 1000)).T
    constraints = {"const": {">": 10**(-3)}, "length": {">": 10**(-3)}}
    surrogate = oed_gp.GP_Surrogate(kernel, X, y, constraints)
    surrogate.params = {"length":10, "const":1}
    print(jnp.linalg.cholesky(surrogate.K_fun(X, X, surrogate.params)))
    # y_pred = surrogate.predict(x_pred)
    
    # Plotting: 
    #fig = plt.figure()
    #plt.plot(x_pred, f(x_pred), 'r:', label=r'$f(x) = x\,\sin(x)$')
    #plt.plot(X, y, 'r.', markersize=10, label='Observations')
    #plt.plot(x_pred, y_pred[0], 'b-', label='Prediction')
    # plt.fill(jnp.concatenate([x_pred, x_pred[::-1]]),
    #      jnp.concatenate([y_pred[0] - 1.9600 * y_pred[1],
    #                     (y_pred[0] + 1.9600 * y_pred[1])[::-1]]),
    #      alpha=.5, fc='b', ec='None', label='95% confidence interval')
    # plt.xlabel('$x$')
    # plt.ylabel('$f(x)$')
    # plt.ylim(-10, 20)
    # plt.legend(loc='upper left')
    # plt.savefig('GP.png', dpi=600)