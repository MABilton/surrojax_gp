import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky

def nearestPD(A):
    B = (A + A.T)/2
    w, v = jnp.linalg.eigh(B)
    w = jnp.where(w<0., 0., w) # jax.ops.index_update(w, , 0.)
    A_PD = v @ jnp.diag(w) @ v.T
    i = 0
    while not isPD(A_PD):
        epsilon = 10**(-12+i)
        w = jnp.where(w<epsilon, epsilon, w) # jax.ops.index_update(w, , )
        A_PD = v @ jnp.diag(w) @ v.T
        i += 1
    return A_PD

def isPD(A):
    B = cholesky(A, lower=True, check_finite=True)
    nan_flag = jnp.isnan(jnp.sum(B))
    return not nan_flag