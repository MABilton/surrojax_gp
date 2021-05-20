import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from math import pi

from scipy.optimize import dual_annealing, shgo

# y_train = (num_samples) - learning a single function!
# x_train.size = (num_samples, num_features)

# Constraints of the form: {"a":None, "b":{"<":1, ">":-1}, "c":None, "d":{"<":10}}

class GP_Surrogate:
    def __init__(self, cov, x_train, y_train, constraints):
        
        # TODO: Check dimensions of x_train and y_train
        self.x_train = x_train
        self.y_train = y_train

        # Vectorise covariance function:
        cov_fun = jax.vmap(jax.vmap(cov, in_axes=(0,None,None), out_axes=0), in_axes=(None,0,None), out_axes=1)
        self.cov_fun = lambda x_1, x_2, params : jnp.atleast_1d(cov_fun(x_1, x_2, params).squeeze())
        
        # Create function which returns 'noisy' kernel if requested by user:
        if "noise" in constraints:
            K_fun = lambda x_1, x_2, params : noisy_K(self.cov_fun, x_1, x_2, params)
        # Otherwise, add jitter to diagonal of K for numerical stability:
        else:
            K_fun = lambda x_1, x_2, params : jitter_K(self.cov_fun, x_1, x_2, params)
            self.K_fun = K_fun

        # Optimise hyperparameters of covariance function:    
        self.params = opt_hyperparams(x_train, y_train, K_fun, constraints)

        # With optimal hyperparameters, compute covariance matrix and inverse covariance matrix:
        self.cov = K_fun(x_train, x_train, self.params)
        self.L = jnp.linalg.cholesky(self.cov)
        self.alpha = jnp.linalg.lstsq((self.L).T, jnp.linalg.lstsq(self.L,y_train)[0])[0]

        # Store vectorised prediction function:
        self.pred_fun = jax.vmap(pred_fun_temp, in_axes=(None, 0))

    def predict(self, x_new):
        return self.pred_fun(self, x_new)

def pred_fun_temp(obj, x_new):
    k = obj.cov_fun(x_new, obj.x_train, obj.params)
    mean = jnp.dot(k,obj.alpha)
    v = jnp.linalg.lstsq(obj.L, k)[0]
    var = obj.cov_fun(x_new, x_new, obj.params) - jnp.dot(v.T, v)
    return (mean, var)

# Calls Scipy Optimise function to tune hyperparameters:
def opt_hyperparams(x_train, y_train, K_fun, constraints):
    bounds, idx_2_key = create_bounds(constraints)
    loss_fun = lambda params : loss(params, x_train, y_train, K_fun, idx_2_key)
    opt_params = dual_annealing(loss_fun, bounds, no_local_search=True, maxfun=100.0) #shgo(loss_fun, bounds)
    opt_params = create_param_dict(opt_params["x"], idx_2_key)
    return opt_params

def loss(params, x_train, y_train, K_fun, idx_2_key):
    # Convert array of param values into dictionary:
    params = create_param_dict(params, idx_2_key)
    # Compute kernel matrix:
    K = K_fun(x_train, x_train, params)
    # Cholesky decomposition:
    L = jnp.linalg.cholesky(K)
    alpha = jnp.linalg.lstsq(L.T, jnp.linalg.lstsq(L,y_train)[0])[0]
    n = y_train.shape[0]
    loss  = 0.5*jnp.dot(y_train.T, alpha) + jnp.sum(jnp.log(jnp.diag(L))) + 0.5*n*jnp.log(2*pi)
    return loss.ravel()

def create_param_dict(params, idx_2_key):
    param_dict = {}
    for i, value in enumerate(params):
        param_dict[idx_2_key[i]] = value
    return param_dict

def create_bounds(constraints):
    idx_2_key = []
    lb = []
    ub = []
    max_val = 10^5
    min_val =  -1*max_val
    for i, (key, value) in enumerate(constraints.items()):
        idx_2_key.append(key)
        if value is None:
            lb.append(min_val)
            ub.append(max_val)
        else:
            if ">" in value:
                lb.append(value[">"])  
            else: 
                lb.append(min_val)
            if "<" in value:
                ub.append(value["<"])  
            else:
                ub.append(max_val)
    bounds = list(zip(lb, ub))
    return (bounds, idx_2_key)

def noisy_K(cov_fun, x_1, x_2, params):
    jitter = 10**(-8)
    K = cov_fun(x_1, x_2, params)
    return K + (params["noise"] + jitter)*jnp.identity(K.shape[0])

def jitter_K(cov_fun, x_1, x_2, params):
    jitter = 10**(-8)
    K = cov_fun(x_1, x_2, params)
    return K + jitter*jnp.identity(K.shape[0])

#def noisy_K_grad(cov_grad_fun, x_1, x_2, params):
    #cov_grad = cov_grad_fun(x_1, x_2, params)
    #num_samples = min(x_1.shape)
    #cov_grad["noise"] = jnp.identity(num_samples)
    #return cov_grad

    # Create function to return derivative of kernel wrt hyperparameters:
    #cov_grad = jax.grad(cov_fun, argnums=2)
    #cov_grad_fun = jax.vmap(jax.vmap(cov_grad, in_axes=(0,None,None)), in_axes=(None,0,None))
    #if "noise" in constraints:
    #    self.K_grad_fun = lambda x_1, x_2, params : noisy_K_grad(cov_grad_fun, x_1, x_2, params)
    #else:
    #    self.K_grad_fun = cov_grad_fun