import jax
import jax.numpy as jnp
from .gp_utilities import create_K, create_cov_diag, create_noisy_K, compute_L_and_alpha
from .gp_optimise import fit_hyperparameters
from .prediction import create_predict_method, create_grad_key

# y_train = (num_samples) - learning a single function!
# x_train.size = (num_samples, num_features)
# Constraints of the form: {"a":None, "b":{"<":1, ">":-1}, "c":None, "d":{"<":10}}

class GP_Surrogate:
    def __init__(self, create_dict):

        # Attributes to store:
        self.kernel = create_dict["kernel"]
        self.x_train = create_dict["x_train"]
        self.y_train = create_dict["y_train"]
        self.constraints = create_dict["constraints"]
        self.x_dim = self.x_train.shape[1]
        self.y_dim = 1
        self.train_size = self.x_train.shape[0]
    
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

        # Initialise dictionary to store prediction functions:
        self.predict_functions = {}

    def predict(self, x_new, return_var=True, return_cov=False, grad=None):

        # Create key associated with requested gradient:
        grad_key = create_grad_key(grad)

        # Check if we have the requested gradient function - if not, create it:
        if grad_key not in self.predict_functions:
            self.predict_functions[grad_key] = create_predict_method(self, grad=grad)

        # Perform prediction:
        output_dict = self.predict_functions[grad_key](x_new, return_var, return_cov)

        return output_dict

    def compute_k(self, x_new):
        return self.K(self.x_train, x_new, self.params).reshape(self.train_size, -1)