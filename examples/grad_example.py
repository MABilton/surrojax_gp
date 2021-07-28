import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from gp_oed_surrogate.gp_create import create_gp
from gp_oed_surrogate.gp_grad_2 import create_derivative_gp

def kernel(x_1, x_2, params):
    lengths = jnp.array([params[f"length_{i}"] for i in range(2)])
    inv_lengths = jnp.diag(lengths**(-1))
    ln_k_d = -0.5*(x_1 - x_2).T @ inv_lengths @ (x_1 - x_2) 
    return params["const"]*jnp.exp(ln_k_d)

# Peak function (see: https://mathworks.com/help/matlab/ref/peaks.html)
def generate_data(x, y):
    first_term = 3*(1-x)**2*jnp.exp(-x**2-(y+1)**2)
    second_term = -10*(x/5 - x**3 - y**5)*jnp.exp(-x**2-y**2)
    third_term = -1/3*jnp.exp(-(x+1)**2-y**2)
    return first_term + second_term + third_term

def plot_contours(x, y, z, num_pts, save_name="plot.png"):
    # Check if supplied save name has .png extension:
    if save_name[-4:] != ".png":
        save_name += ".png"
    # Reshape inputs:
    x, y, z = x.reshape(num_pts,num_pts), y.reshape(num_pts,num_pts), z.reshape(num_pts,num_pts)
    # Create surface plot:
    fig, ax = plt.subplots()
    contour_fig = ax.contourf(x, y, z, cmap='viridis')
    fig.colorbar(contour_fig)
    # Save image of plot:
    fig.savefig(save_name, dpi=300)

if __name__ == "__main__":
    # Generate test data:
    generate_data_vmap = jax.vmap(generate_data, in_axes=[0,0])
    train_pts = 10
    x_train, y_train = jnp.linspace(-3, 3, train_pts), jnp.linspace(-3, 3, train_pts)
    y_train, x_train = jnp.meshgrid(x_train, y_train)
    z_train = generate_data_vmap(x_train.flatten(), y_train.flatten())
    xy_train = jnp.vstack((x_train.flatten(),y_train.flatten())).T

    # Define constraints:
    constraints = {"length_0": {">": 10**-1, "<": 10**1}, 
                   "length_1": {">": 10**-1, "<": 10**1}, 
                   "const": {">": 10**-1, "<": 10**2}}
    # Create GP surrogate:
    gp = create_gp(kernel, xy_train, z_train, constraints)
    # Make predictions with GP:
    pred_pts = 100
    x_pred, y_pred = jnp.linspace(-3, 3, pred_pts), jnp.linspace(-3, 3, pred_pts)
    y_pred, x_pred = jnp.meshgrid(x_pred, y_pred)
    xy_pred = jnp.vstack((x_pred.flatten(),y_pred.flatten())).T
    z_pred = gp.predict_mean(xy_pred)
    # Plot GP surrogate surface and original data surface:
    z_train = generate_data_vmap(x_train, y_train)
    z_true = generate_data_vmap(xy_pred[:,0], xy_pred[:,1])
    plot_contours(x_train, y_train, z_train, train_pts, save_name="peaks_train.png")
    plot_contours(xy_pred[:,0], xy_pred[:,1], z_true, pred_pts, save_name="peaks_true.png")
    plot_contours(xy_pred[:,0], xy_pred[:,1], z_pred, pred_pts, save_name="peaks_pred.png")

    # Compute Jacobian of Gaussian process:
    idx_2_diff = [([0,1], 1)]
    gp_jac = create_derivative_gp(gp, idx_2_diff)
    
    # Make predictions with Jacobian GP:
    z_jac_pred = gp_jac.predict_grad_mean(xy_pred)
    z_delx_pred, z_dely_pred = z_jac_pred[:,0], z_jac_pred[:,1]
    plot_contours(xy_pred[:,0], xy_pred[:,1], z_delx_pred, pred_pts, save_name="peaks_delx_pred.png")
    plot_contours(xy_pred[:,0], xy_pred[:,1], z_dely_pred, pred_pts, save_name="peaks_dely_pred.png")
    # Compare with actual Jacobian:
    generate_data_delx_vmap = jax.vmap(jax.grad(generate_data, argnums=0), in_axes=[0,0])
    z_delx_true = generate_data_delx_vmap(xy_pred[:,0], xy_pred[:,1])
    plot_contours(xy_pred[:,0], xy_pred[:,1], z_delx_true, pred_pts, save_name="peaks_delx_true.png")
    generate_data_dely_vmap = jax.vmap(jax.grad(generate_data, argnums=1), in_axes=[0,0])
    z_dely_true = generate_data_dely_vmap(xy_pred[:,0], xy_pred[:,1])
    plot_contours(xy_pred[:,0], xy_pred[:,1], z_dely_true, pred_pts, save_name="peaks_dely_true.png")

    # Plot variance of predicted Jacobian:
    z_grad_var = gp_jac.predict_grad_var(xy_pred)
    z_delx_var, z_dely_var = z_grad_var[:,0], z_grad_var[:,1]
    plot_contours(xy_pred[:,0], xy_pred[:,1], z_delx_var, pred_pts, save_name="peaks_delx_var.png")
    plot_contours(xy_pred[:,0], xy_pred[:,1], z_dely_var, pred_pts, save_name="peaks_dely_var.png")