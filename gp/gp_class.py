import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from gp_utilities import create_K, create_cov_diag, create_noisy_K, compute_L_and_alpha
from gp_optimise import fit_hyperparameters

# y_train = (num_samples) - learning a single function!
# x_train.size = (num_samples, num_features)
# Constraints of the form: {"a":None, "b":{"<":1, ">":-1}, "c":None, "d":{"<":10}}

class GP_Surrogate:
    def __init__(self, create_dict):
        # Attributes to store:
        self.kernel_file = create_dict["kernel_file"]
        self.kernel_name = create_dict["kernel_name"]
        self.kernel = create_dict["kernel"]
        self.x_train = create_dict["x_train"]
        self.y_train = create_dict["y_train"]
        self.constraints = create_dict["constraints"]
    
        # Create functions to compute covariance matrix:
        self.K = create_K(self.kernel)
        self.cov_diag = create_cov_diag(self.kernel)
        noise_flag = True if "noise" in self.constraints else False
        # If hyperparameters have not been fit:
        if "params" not in create_dict: 
            # Create function which adds noise or jitter to covariance matrix:
            noisy_K = create_noisy_K(self.K, noise_flag)
            # Optimise hyperparameters of covariance function:    
            self.params = fit_hyperparameters(self.x_train, self.y_train, noisy_K, self.constraints)
        # If user provides hyperparameters
        else:
            noisy_K = None
            self.params = create_dict["params"]

        # If user doesn't provide L and alpha:
        if any(key not in create_dict for key in ("L", "alpha")):
            if noisy_K is None:
                noisy_K = create_noisy_K(self.K, noise_flag)
            self.L, self.alpha = compute_L_and_alpha(noisy_K, self.x_train, self.y_train, self.params)
        # If user provides L and alpha:
        else:
            self.L, self.alpha = create_dict["L"], create_dict["alpha"]

    def predict(self, x_new, cov="diag"):
        k = self.compute_k(x_new)
        mean = self.predict_mean(x_new, k=k)
        if cov.lower()=="diag":
            cov_out = self.predict_cov_diag(x_new, k=k)
        elif cov.lower()=="full":
            cov_out = self.predict_cov(x_new, k=k)
        else:
            cov_out = None
        return (mean, cov_out)

    def predict_mean(self, x_new, k=None):
        x_new = jnp.atleast_2d(x_new)
        k = self.compute_k(x_new) if k is None else k
        mean = jnp.einsum("ji,j->i", k, self.alpha)
        return mean.reshape((mean.size,1))

    def predict_cov_diag(self, x_new, k=None, min_var=10**-9):
        x_new = jnp.atleast_2d(x_new)
        k = self.compute_k(x_new) if k is None else k
        v = cho_solve((self.L, True), k)
        var = self.cov_diag(x_new, self.params) - jnp.einsum("ij,ij->j", k, v) # jnp.sum(v*v, axis=0, keepdims=True)
        var = jax.ops.index_update(var, var<min_var, min_var)
        return var.reshape((var.size,1))

    def predict_cov(self, x_new, k=None):
        if k is None:
            k = self.compute_k(x_new)
        K_full = self.K(x_new, x_new)
        cov = K_full - k.T @ self.L
        return cov

    def compute_k(self, x_new):
        return self.K(self.x_train, x_new, self.params)