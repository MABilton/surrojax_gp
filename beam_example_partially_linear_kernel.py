import numpy as np
import jax.numpy as jnp

import gp_create

import matplotlib.pyplot as plt

# Kernel if we include x, y and z:
# def kernel(x_1, x_2, params):
#     # Dot product kernel for theta (i.e. linearise in terms of theta = c10);
#     # Squared exponential kernel for d (i.e. non-linear in terms of angle, x, y and z):
#     k_theta = params["const_1"]*x_1[0]*x_2[0]
#     lengths = jnp.array([params[f"length_{i}"] for i in range(3)])
#     inv_lengths = jnp.diag(lengths**(-1))
#     ln_k_d = -0.5*(x_1[1:] - x_2[1:]).T @ inv_lengths @ (x_1[1:] - x_2[1:])
#     k_d = params["const_2"]*jnp.exp(ln_k_d)
#     return k_theta + k_d

# Kernel if we only have data for one point:
def kernel(x_1, x_2, params):
    # Dot product kernel for theta (i.e. linearise in terms of theta = c10);
    # Squared exponential kernel for d (i.e. non-linear in terms of angle, x, y and z):
    k_theta = params["const_0"]*(x_1[0]-params["const_1"])*(x_2[0]-params["const_1"]) + params["const_2"]
    k_d = params["const_3"]*jnp.exp(-0.5*((x_1[1] - x_2[1])/params["length"])**2)
    return k_theta + k_d

if __name__ == "__main__":
    # Import data from text file:
    training_data = np.loadtxt("beam_data_one_pt.txt")
    # Note that columns of beam data arranged as: [angle, c10, disp]
    # Rearrange to form: [c10, angle, x, y, z, disp]:
    training_data.T[[0,1]] = training_data.T[[1,0]]

    # Convert to Jax.numpy array:
    training_x = jnp.array(training_data[:,0:-1])
    training_y = training_data[:,-1]

    # # Plot training data:
    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot_surface(training_x[:,0].reshape(10,10), training_x[:,1].reshape(10,10), training_y.reshape(10,10),cmap='viridis', edgecolor='none')
    # ax.set_xlabel("c10")
    # ax.set_ylabel("Angle in Degrees")
    # ax.set_zlabel("Mean Displacement")
    # plt.savefig('beam_example_training.png', dpi=600)

    #constraints = {f"length_{i}": {">": 10**(-1)} for i in range(3)}
    constraints  = {f"const_{i}": {">": 10**(-3), "<": 10**2} for i in range(4)} # if i!=1 else {">": -10**2, "<": 10**3} for i in range(4)
    constraints["length"] = {">": 10**(-3), "<": 10**3}
    constraints["noise"] = {">": 10**(-3), "<": 10**2}

    surrogate = gp_create.GP_Surrogate(kernel, training_x, training_y, constraints)
    
    # Make predictions:
    d_pts = 1000
    theta_pts = 1000
    x_d = jnp.linspace(0, 180, d_pts)
    x_theta = jnp.linspace(0.1, 5, d_pts)
    x_1, x_2 = jnp.meshgrid(x_theta, x_d)
    x = jnp.vstack((x_1.flatten(), x_2.flatten())).T
    y = surrogate.predict(x)

    # Plot 2D slice along c10 axis:
    plt_idx = 5
    plt.figure()
    mean_plot = jnp.atleast_1d(y[0].reshape(d_pts,theta_pts)[plt_idx,:])
    # std_plot = jnp.atleast_2d(jnp.sqrt(y[1].reshape(d_pts,theta_pts)[:,plt_idx]))
    x_plot = jnp.atleast_1d(x_1[plt_idx,:])
    plt.plot(x_plot, mean_plot, 'b-', label='Prediction')
    # plt.fill(np.concatenate([x_plot, x_plot[::-1]]), \
    #         np.concatenate([mean_plot - 1.9600 * std_plot, \
    #                         mean_plot + 1.9600 * std_plot[::-1]]), \
    #         alpha=.5, fc='b', ec='None', label='95% confidence interval')
    plt.legend(loc='upper left')
    plt.savefig('beam_example_mean_slice.png', dpi=600)

    # Plot mean predictions:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x_1, x_2, (y[0]).reshape(d_pts,theta_pts),cmap='viridis', edgecolor='none')
    ax.set_xlabel("c10")
    ax.set_ylabel("Angle in Degrees")
    ax.set_zlabel("Mean Displacement")
    plt.savefig('beam_example_mean.png', dpi=600)

    # Plot resulting variance predictions:
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x_1, x_2, y[1].reshape(d_pts,theta_pts),cmap='viridis', edgecolor='none')
    ax.set_xlabel("c10")
    ax.set_ylabel("Angle in Degrees")
    ax.set_zlabel("Variance in Displacement")
    plt.savefig('beam_example_variance.png', dpi=600)
