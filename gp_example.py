import jax
import jax.numpy as jnp

import oed_gp

# Function to generate artificial data from:
def nonlin_model(d, theta):
    return jnp.array([d**2 * theta[:,0]*theta[:,1]]).squeeze()

# Kernel function (w/o noise):
def cov_fun(x_1, x_2, params):
    theta_1, d_1 = x_1[0:-1], x_1[-1].reshape((1,1))
    theta_2, d_2 = x_2[0:-1], x_2[-1].reshape((1,1))
    # Inner product kernel for theta:
    theta_k = theta_1.T @ theta_2
    # Squared exponential kernel for d:
    M = jnp.array(params["l_0"])
    d_k = params["sigma_f"]*jnp.exp(-0.5*(d_2 - d_1) * M * (d_2 - d_1))
    # Must squeeze to ensure output is of shape (,)
    return (theta_k + d_k).squeeze()

# Kernel function (w/o noise):
def cov_fun_1(x_1, x_2, params):
    # Must squeeze to ensure output is of shape (,)
    return params["sigma_f"]*jnp.exp(-0.5*(1/params["l"])*(x_2 - x_1).T @ (x_2 - x_1)).squeeze()

if __name__ == "__main__":
    # Generate grid of theta values:
    linspace_1 = jnp.linspace(0.1,5.1,2)
    linspace_2 = jnp.linspace(2.1,7.1,2)
    grid_1, grid_2 = jnp.meshgrid(linspace_1,linspace_2)
    theta = jnp.array([grid_1.flatten(), grid_2.flatten()]).T
    # Generate range of d values:
    d = jnp.linspace(0.1,2.0,2)
    # Dimensions:
    y_dim = 1
    theta_dim = theta.shape[1]
    d_dim = 1

    # Compute y values from non-linear model:
    x_train = jnp.empty((0,theta_dim+d_dim))
    y_train = []
    for i in range(d.shape[0]):
        d_current = d[i].reshape((1,d_dim))
        y = nonlin_model(d_current, theta)
        d_stack = d_current*jnp.ones((theta.shape[0],1))
        x = jnp.hstack((theta, d_stack))
        y_train.append(y)
        x_train = jnp.vstack((x_train, x))
    y_train = jnp.array(y_train)
    y_train = y_train.flatten()
    num_samples = len(y_train)

    y_train.reshape(y_dim,num_samples)

    # Define constraints:
    #constraints = {"noise": {">":10**(-3)}, "sigma_f": {">":10**(-3)}, "l_0": {">":10**(-3)}}
    constraints = {"noise": {">":10**(-3)}, "sigma_f": {">":10**(-3)}, "l": {">":10**(-3)}}

    # Create surrogate:
    surrogate = oed_gp.GP_Surrogate(cov_fun_1, x_train, y_train, constraints)
    x_pred = jnp.array([0.1,2.1,0.1]).reshape((1,3))
    pred = surrogate.predict(x_pred)
    #print(pred)
    #print(y_train[0])