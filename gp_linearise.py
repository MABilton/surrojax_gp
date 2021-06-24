from gp_create import create_gp
from gp_utilities import chol_decomp

from math import factorial

import jax.scipy.linalg as jlinalg
import jax.numpy as jnp

class LinearisedModel:
    def __init__(self, x, y, dim_2_lin, kernel, constraints, ln_diag_constraints, intercept_flag=True):
        self.dim_2_lin = dim_2_lin
        # Linearise data:
        self.training_d, self.training_w, self.diag_list, off_diag_list = linearise_data(x, y, dim_2_lin, intercept_flag)
        # Create GP surrogate with training data:
        self.w_surrogate = create_gp(kernel, self.training_d, self.training_w[:,0], constraints)
        self.b_surrogate = create_gp(kernel, self.training_d, self.training_w[:,1], constraints)
        self.ln_diag_surrogate = create_gp(kernel, self.training_d, jnp.log(self.diag_list), ln_diag_constraints) # Assumed only one entry

    def predict_weights(self, d):
        w = self.w_surrogate.predict(d)
        b = self.b_surrogate.predict(d)
        return (w, b)

    def predict_ln_diag(self, d):
        ln_diag = self.ln_diag_surrogate.predict(d)[0]
        return ln_diag

    def predict_pt(self, theta, d):
        w = self.w_surrogate.predict(d)
        b = self.b_surrogate.predict(d)
        return w @ theta + b

def linearise_data(x, y, dim_2_lin, intercept_flag, epsilon = 10**(-3)):
    # Ensure x.shape = (num_samples, num_features) and y.shape = (num_samples, num_targets)
    x, y = jnp.atleast_2d(x), jnp.atleast_2d(y)
    num_samples = x.shape[0]
    num_features, num_labels = int(x.size/num_samples), int(y.size/num_samples)
    x, y = x.reshape(num_samples, num_features), y.reshape(num_samples, num_labels)
    # Rearrange X matrix so that dimensions we want to linearise over appear last:
    x, nl_end = preprocess_input(x, dim_2_lin)
    # Sort x by row values:
    sort_idx = jnp.argsort(x[:,0:nl_end], axis=0).squeeze()
    x, y = x[sort_idx, :], y[sort_idx, :]
    # Append column of 1's to x if we want an intercept term:
    if intercept_flag:
        ones_col = jnp.ones((x.shape[0],1))
        x = jnp.hstack((x, ones_col))
        num_features += 1
    #
    nl_diff  = jnp.diff(x[:,0:nl_end],n=1,axis=0)
    nl_change_idx = jnp.sum(nl_diff > epsilon, axis=1)
    nl_change_idx = jnp.nonzero(nl_change_idx)[0] + 1
    num_d = len(nl_change_idx) + 1
    #
    nl_list, w_list, diag_list, off_diag_list = [], [], [], []
    for i in range(num_d):
        if i == 0:
            fit_idx = [0, nl_change_idx[i]]
        elif i == (num_d-1):
            fit_idx = [nl_change_idx[i-1], x.shape[0]]
        else:
            fit_idx = [nl_change_idx[i-1], nl_change_idx[i]]
        current_lin = x[fit_idx[0]:fit_idx[1],nl_end:]
        current_w, current_cov = bayesian_lr(current_lin, y[fit_idx[0]:fit_idx[1],:])
        nl_list.append(x[fit_idx[0],0:nl_end])
        w_list.append(current_w)
        chol = chol_decomp(current_cov)
        diag_list.append(jnp.diag(chol)), off_diag_list.append(jnp.tril(chol,k=-1))
    # Reshape arrays to output:
    nl_list = jnp.array(nl_list).reshape(num_d,nl_end)
    w_list = jnp.array(w_list).reshape(num_d,(num_features-nl_end))
    diag_list = jnp.array(diag_list).reshape(num_d,num_labels)
    off_diag_list = jnp.array(off_diag_list).reshape(num_d,factorial(num_labels-1))
    return (nl_list, w_list, diag_list, off_diag_list)

def bayesian_lr(x, y):
    w = []
    for y_col in y.T:
        w.append(jlinalg.solve((x.T @ x), x.T @ y_col))
    w = jnp.array(w).reshape(x.shape[1], y.shape[1])
    dy = x @ w - y
    cov = jnp.mean(dy[:,:,None] @ dy[:,None,:], axis=0)
    return (w, cov)

def preprocess_input(x, dim_2_lin):
    x = jnp.atleast_2d(x)
    new_order = [i for i in range(x.shape[1]) if i not in dim_2_lin]
    new_order.append(*dim_2_lin)
    x = x[:, new_order]
    nl_end = x.shape[1] - len(dim_2_lin)
    return (x, nl_end)