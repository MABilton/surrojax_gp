from gp_create import create_gp, save_gp

def kernel(x_1, x_2, params):
    # This function accepts three inputs: two feature vectors (which are jax.numpy arrays) and
    # a dictionary of hyperparameter values. This function should return the SINGLE value
    # of the kernel functions for these two input feature vectors. Note that only jax.numpy 
    # functions should be used in this function definition, since this function will be differentiated.
    pass

def load_data(data_dir):
    # This function should accept a string which specifies the directory of the training data
    # and should return two Numpy arrays: the first constains the feature values (i.e. x_train)
    # and the second contains the label values (i.e. y_train)
    return (x_train, y_train)

if __name__ == "__main__":
    # Load training data:
    data_dir = "specify_data_dir_here"
    x_train, y_train = load_data(data_dir)

    # Specify constraints on parameters:
    constraints = {"param_1": {">": low_lim_1, "<": high_lim_1},
                   "param_2": {">": low_lim_2, "<": high_lim_2},
                   "param_3": {">": low_lim_3, "<": high_lim_3}}

    # Create Gaussian process:
    gp = create_gp(kernel, x_train, y_train, constraints)

    # Save Gaussian process:
    save_name = "my_gp"
    save_gp(save_name)