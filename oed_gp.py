import jax
import jax.numpy as jnp

import math

import random
random.seed(1)

# y_train = (num_samples) - learning a single function!
# x_train.size = (num_samples, num_features)

# Constraints of the form: {"a":None, "b":{"<":1, ">":-1}, "c":None, "d":{"<":10}}

class GP_Surrogate:
    def __init__(self, cov, x_train, y_train, constraints):
        
        self.x_train = x_train
        self.y_train = y_train
        
        # Vectorise covariance function:
        self.cov_fun = jax.vmap(jax.vmap(cov, in_axes=(0,None,None)), in_axes=(None,0,None))
        cov_grad = jax.grad(cov, argnums=2)
        #self.cov_grad_fun = jax.grad(cov, argnums=2)
        self.cov_grad_fun = jax.vmap(jax.vmap(cov_grad, in_axes=(0,None,None)), in_axes=(None,0,None))

        # Optimise hyperparameters of covariance function:    
        self.hyperparams = opt_hyperparams(x_train, y_train, self.cov_fun, self.cov_grad_fun, constraints)

        # With optimal hyperparameters, compute covariance matrix and inverse covariance matrix:
        self.cov = cov_fun(x_train, x_train, self.hyperparams)
        self.inv_cov = jnp.linalg.inv(cov)
        self.inv_cov_y = self.inv_cov @ y_train

    def predict(self, x_new):
        return (self.predict_mean(x_new), self.predict_cov(x_new))

    def predict_mean(self, x_new):
        return self.cov_fun(x_new, self.x_train).T @ self.inv_cov_y

    def predict_cov(self, x_new):
        cov_new = self.cov_fun(x_new, self.x_train)
        return self.cov_fun(x_new, x_new) - cov_new.T @ self.inv_cov @ cov_new

def opt_hyperparams(x_train, y_train, fun_cov, fun_cov_grad, constraints):

    # Randomly initialise according to specified constraints:
    num_repeats = 5
    num_steps = 1000
    best_loss_out = math.inf

    for repeat in range(num_repeats):
        # Randomly initialise hyperparameters (subject to constaints):
        params = init_params(constraints)
        # Reinitialise other varaibles:
        best_loss_in = math.inf
        grad_hist = []
        del_hist = []
        for i in range(num_steps):
            # Compute covariance-related quantities:
            cov = fun_cov(x_train, x_train, params)
            print(cov)
            cov_inv = jnp.linalg.inv(cov)
            cov_grad = fun_cov_grad(x_train, x_train, params)
            #print(cov_grad)
            # Compute loss function:
            loss = fun_loss(cov, cov_inv, y_train)
            for key, value in params.items():
                grad = fun_grad(cov_inv, cov_grad[key], y_train)
                grad_hist.append(grad)
                update, del_hist = rprop(grad_hist, del_hist)
                new_param = value + update
                # Project params to closest point which satisfies constraints:
                params[key] = param_proj(new_param, constraints[key])
            # Store parameters associated with best loss seen:
            if loss > best_loss_in:
                best_loss_in = loss
                best_params_in = params
        # Best 
        if best_loss_in > best_loss_out:
            best_loss_out = best_loss_in
            best_params_out = best_params_in
    return best_params_out

def rprop(grad_hist,del_hist):
    # Standard Rprop hyperparameter values:
    eta_p = 1.2
    eta_m = 0.5
    del_0 = 0.5
    del_min = 10**(-6)
    del_max = 50
    # If first iteration, set del_t to initial val:
    if len(grad_hist) == 1:
        del_t = del_0
    # If past first iteration, compute update:
    else:
        grad_prod = grad_hist[-2]*grad_hist[-1]
        if grad_prod > 0:
            del_t = eta_p*del_hist[-1]
        elif grad_prod < 0:
            del_t = eta_m*del_hist[-1]
        else:
            del_t = del_hist[-1]
    # Apply gradient limits:
    update = del_max if del_t > del_max else del_t
    update = del_min if del_t < del_min else del_t
    del_hist.append(update)
    return (update, del_hist)

def param_proj(val, constraint):
    if constraint is not None:
        if "<" in constraint:
            val = constraint["<"] if val > constraint["<"] else val
        if ">" in constraint:
            val = constraint[">"] if val < constraint[">"] else val
    return val

def init_params(constraints):
    max_val = 100
    min_val = -1*max_val
    params = {}
    for key, value in constraints.items():
        # Generate random number:
        rand = random.random()
        # No constraints applied:
        if value == None:
            params[key] = min_val + (max_val - min_val)*rand
        # Constraints applied:
        else:
            con_max = value["<"] if "<" in value else max_val
            con_min = value[">"] if ">" in value else min_val
            params[key] = con_min + (con_max - con_min)*rand
    return params

def fun_loss(cov, cov_inv, y_train):
    N = cov.shape[0]
    loss = -0.5*(jnp.log(jnp.linalg.det(cov)) + y_train.T @ cov_inv @ y_train + N*jnp.log(2*math.pi))
    return loss

def fun_grad(cov_inv, cov_grad, y_train):    
    inv_times_grad = cov_inv @ cov_grad
    grad = -0.5*(jnp.trace(inv_times_grad) - ((y_train.T @ inv_times_grad) @ cov_inv) @ y_train)
    return grad