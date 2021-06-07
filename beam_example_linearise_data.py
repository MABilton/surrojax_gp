
import numpy as np
import jax.numpy as jnp

from gp_linearise import LinearisedModel

from matplotlib import pyplot as plt

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
    constraints  = {"const": {">": 10**(-2), "<": 10**3}, "length":{">": 10**(-2), "<": 10**3}}
    surrogate = LinearisedModel(training_x, training_y, dim_2_linearise, kernel, constraints)

    plot_pts = 1000

    # Plot predicted w vs d:
    d_plot = jnp.linspace(jnp.min(training_x[:,0]), jnp.max(training_x[:,0]), plot_pts)
    w_plot, w_var = surrogate.predict_w(d_plot)[0]
    fig = plt.figure()
    plt.plot(d_plot, w_plot, 'b-', label='Prediction')
    mean_minus_std = w_plot - 1.9600 * jnp.sqrt(w_var)
    mean_plus_std = w_plot + 1.9600 * jnp.sqrt(w_var)
    plt.fill(jnp.concatenate([w_plot, w_plot[::-1]]),
         jnp.concatenate([mean_minus_std,
                        (mean_plus_std)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$d$')
    plt.ylabel('$w(d)$')
    plt.legend(loc='upper left')
    plt.savefig('beam_example_linearise_data_w.png', dpi=600)

    # Plot predicted b vs d:
    b_plot, b_var = surrogate.predict_w(d_plot)[1]
    fig = plt.figure()
    plt.plot(d_plot, b_plot, 'b-', label='Prediction')
    mean_minus_std = b_plot - 1.9600 * jnp.sqrt(b_var)
    mean_plus_std = b_plot + 1.9600 * jnp.sqrt(b_var)
    plt.fill(jnp.concatenate([b_plot, b_plot[::-1]]),
         jnp.concatenate([mean_minus_std,
                        (mean_plus_std)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.xlabel('$d$')
    plt.ylabel('$b(d)$')
    plt.legend(loc='upper left')
    plt.savefig('beam_example_linearise_data_b.png', dpi=600)

    # Plot predicted displacement vs theta and d: [angle, c10, disp]
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