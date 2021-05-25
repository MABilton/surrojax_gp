import matplotlib.pyplot as plt

def f(x):
    return x[:,0] * (x[:,1])**2

def kernel(x_1, x_2, params):
    val = params["const_1"]*x_1[0]*x_2[0] \
         + params["const_2"]*jnp.exp(-0.5*((x_2[1] - x_1[1])/params["length"])**2)
    return val

if __name__ == "__main__":
    # Noiseless case:
    X = jnp.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    # Observations:
    Y = (f(X).ravel())
    Y = Y.reshape(len(Y),1)
    x =  jnp.atleast_2d(jnp.linspace(0, 10, 1000)).T
    constraints = {"const_1": {">": 10**(-3)}, "const_2": {">": 10**(-3)}, "length": {">": 10**(-1)}} 
    surrogate = oed_gp.GP_Surrogate(kernel, X, Y, constraints)
    y = surrogate.predict(x)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(x, x, y, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)