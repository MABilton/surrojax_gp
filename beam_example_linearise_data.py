import numpy as np
import jax.numpy as jnp

from gp_linearise import LinearisedModel

from matplotlib import pyplot as plt

import pickle

def kernel(x_1, x_2, params):
    return params["const"]*jnp.exp(-0.5*((x_1 - x_2)/params["length"])**2)

if __name__ == "__main__":
    # Import data from text file:
    training_data = np.loadtxt("beam_data_one_pt.txt")
    # Note that columns of beam data arranged as: [angle, c10, disp]

    # Convert to Jax.numpy array:
    training_x = jnp.array(training_data[:,0:-1])
    training_y = training_data[:,-1]

    # Fit GP to w vs d data:
    dim_2_linearise = [1]
    constraints  = {"const":{">": 10**(-2), "<": 10**3}, "length":{">": 10**0, "<": 10**2}}
    LM = LinearisedModel(training_x, training_y, dim_2_linearise, kernel, constraints)

    # Plot predicted displacement vs theta and d:
    plot_pts = 1000
    d_plot = jnp.linspace(jnp.min(training_x[:,0]), jnp.max(training_x[:,0]), plot_pts)
    w_plot, w_var = LM.predict_weights(d_plot)[0]
    b_plot, b_var = LM.predict_weights(d_plot)[1]
    theta_plot = jnp.linspace(jnp.min(training_x[:,1]), jnp.max(training_x[:,1]), plot_pts)
    theta_grid, d_grid = jnp.meshgrid(theta_plot, d_plot)
    y_grid = jnp.multiply(theta_grid, w_plot[:, np.newaxis]) + b_plot[:,np.newaxis]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(theta_grid, d_grid, y_grid,cmap='viridis', edgecolor='none')
    ax.set_xlabel("c10")
    ax.set_ylabel("Angle in Degrees")
    ax.set_zlabel("Predicted Displacement")
    plt.savefig('beam_example_linearise_data_disp.png', dpi=600)

    # Save Gaussian process model to Pickle file:
    with open("beam_gp.obj", 'wb') as f: 
        # Can't save entire object since vmap'd functions can't be saved:
        dict_2_save = {"w_params": LM.w_surrogate.params, "w_alpha": LM.w_surrogate.alpha, "w_L": LM.w_surrogate.L, \
        "b_params": LM.b_surrogate.params, "b_alpha": LM.b_surrogate.alpha, "b_L": LM.b_surrogate.L, \
        "kernel": LM.w_surrogate.kernel, "x_train": LM.w_surrogate.x_train, }
        pickle.dump(dict_2_save, f)