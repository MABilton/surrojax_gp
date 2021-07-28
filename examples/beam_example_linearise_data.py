import numpy as np
import jax.numpy as jnp
from matplotlib import pyplot as plt
from gp_oed_surrogate.gp_linearise import LinearisedModel

def kernel(x_1, x_2, params):
    return params["const"]*jnp.exp(-0.5*((x_1 - x_2)/params["length"])**2)

def load_data(data_dir):
    # Import data from text file; columns of data arranged as [angle, c10, disp]:
    training_data = np.loadtxt("beam_data_one_pt.txt")
    # Convert to Jax.numpy array:
    x_train = jnp.array(training_data[:,0:-1])
    y_train = training_data[:,-1]
    return (x_train, y_train)

if __name__ == "__main__":
    # Load training data:
    data_dir = "specify_data_dir_here"
    x_train, y_train = load_data(data_dir)

    # Specify constraints on parameters:
    constraints = {"const": {">": 10**(-2), "<": 10**5},
                   "length": {">": 10**(-1), "<": 10**2}}

    # Fit GP to w vs d data:
    dim_2_linearise = [1] # List of array indices corresponding to 1st dimension of x_train
    ln_diag_constraints = {"const":{">": 10**(-2), "<": 10**3}, "length":{">": 10**(0), "<": 10**2}}
    LM = LinearisedModel(x_train, y_train, dim_2_linearise, kernel, constraints, ln_diag_constraints)

    # Plot cov vs d:
    plot_pts = 1000
    d_plot = jnp.linspace(jnp.min(training_x[:,0]), jnp.max(training_x[:,0]), plot_pts)
    ln_diag_plot = LM.predict_ln_diag(d_plot)
    cov_plot = jnp.exp(ln_diag_plot)**2
    plt.figure()
    plt.plot(d_plot, cov_plot)
    plt.plot(LM.training_d, LM.diag_list**2, 'r.', markersize=10, label='Observations')
    plt.xlabel('d')
    plt.ylabel('y cov')
    plt.savefig('beam_example_linearise_data_cov.png', dpi=600)

    # Plot w vs d:
    plot_pts = 1000
    d_plot = jnp.linspace(jnp.min(training_x[:,0]), jnp.max(training_x[:,0]), plot_pts)
    w_plot, w_var = LM.predict_weights(d_plot)[0]
    plt.figure()
    plt.plot(d_plot, w_plot)
    plt.plot(LM.training_d, LM.training_w[:,0], 'r.', markersize=10, label='Observations')
    plt.xlabel('d')
    plt.ylabel('w')
    plt.savefig('beam_example_linearise_data_w.png', dpi=600)

    # Plot cov/w**2:
    plt.figure()
    cov_div_w_plot = (cov_plot).squeeze()/(w_plot**2).squeeze()
    plt.plot(d_plot, cov_div_w_plot)
    plt.xlabel('d')
    plt.ylabel('y cov divided by w squared')
    plt.savefig('beam_example_linearise_data_cov_div_w.png', dpi=600)

    # Plot predicted displacement vs theta and d:
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
    with open("beam_gp.pkl", 'wb') as f: 
        # Can't save entire object since vmap'd functions can't be saved:
        dict_2_save = {"w_params": LM.w_surrogate.params, "w_alpha": LM.w_surrogate.alpha, "w_L": LM.w_surrogate.L, \
        "b_params": LM.b_surrogate.params, "b_alpha": LM.b_surrogate.alpha, "b_L": LM.b_surrogate.L, \
        "ln_diag_params": LM.ln_diag_surrogate.params, "ln_diag_alpha": LM.ln_diag_surrogate.alpha, "ln_diag_L": LM.ln_diag_surrogate.L, \
        "w_train": LM.w_surrogate.x_train, "b_train": LM.b_surrogate.x_train, "ln_diag_train": LM.ln_diag_surrogate.x_train}
        pickle.dump(dict_2_save, f)