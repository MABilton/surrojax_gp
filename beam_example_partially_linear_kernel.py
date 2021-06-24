import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from gp_create import create_gp, save_gp, load_gp

def kernel(x_1, x_2, params):
    # Dot product kernel for theta (i.e. linearise in terms of theta = c10);
    # Squared exponential kernel for d (i.e. non-linear in terms of angle, x, y and z):
    k_theta = params["const_0"]*(x_1[0]-params["const_1"])*(x_2[0]-params["const_1"]) + params["const_2"]
    k_d = params["const_3"]*jnp.exp(-0.5*((x_1[1] - x_2[1])/params["length"])**2)
    return k_theta + k_d

def load_data(data_dir):
    # Import data from text file:
    training_data = np.loadtxt(data_dir)
    # Note that columns of beam data arranged as: [angle, c10, disp]
    # Rearrange to form: [c10, angle, x, y, z, disp]:
    training_data.T[[0,1]] = training_data.T[[1,0]]
    # Convert to Jax.numpy array:
    x_train = jnp.array(training_data[:,0:-1])
    y_train = jnp.array(training_data[:,-1])
    return (x_train, y_train)

def plot_gp_surface(d_pts, theta_pts, y_pts, save_name="plot.png"):
    # Check if supplied save name has .png extension:
    if save_name[-4:] != ".png":
        save_name += ".png"
    # Create surface plot:
    fig, ax = plt.subplots()
    contour_fig = ax.contourf(theta_pts, d_pts, y_pts, cmap='viridis')
    cbar = fig.colorbar(contour_fig)
    ax.set_xlabel('Beam stiffness c10')
    ax.set_ylabel('Angle of Beam in Degrees')
    cbar.set_label('Displacement', rotation=270, labelpad=15)
    # Save image of plot:
    fig.savefig(save_name, dpi=300)

if __name__ == "__main__":
    # Load training data:
    data_dir = "beam_data_one_pt.txt"
    x_train, y_train =  load_data(data_dir)
    constraints  = {f"const_{i}": {">": 10**(-3), "<": 10**3} for i in range(4)} 
    constraints["length"] = {">": 10**(-1), "<": 10**2}
    constraints["noise"] = {">": 10**(-1), "<": 10**3}
    surrogate = create_gp(kernel, x_train, y_train, constraints)

    # Plot surrogate model surface:
    d_pts, theta_pts = jnp.linspace(0, 5, 1000), jnp.linspace(0, 180, 1000)
    d_grid, theta_grid = jnp.meshgrid(theta_pts, d_pts)
    x = jnp.vstack((theta_grid.flatten(), d_grid.flatten())).T
    y_pts = surrogate.predict_mean(x)
    y_grid = y_pts.reshape(d_pts.size,theta_pts.size)
    plot_gp_surface(d_grid, theta_grid, y_grid, save_name="partially_linear_kernel")

    # Save Gaussian process model:
    save_gp(surrogate, "partially_linear_kernel_gp")