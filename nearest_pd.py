
import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky

def nearestPD(A):
    B = (A + A.T)/2
    w, v = jnp.linalg.eigh(B)
    w = jax.ops.index_update(w, w<0., 0.)
    A_PD = v @ jnp.diag(w) @ v.T
    i = 0
    while not isPD(A_PD):
        epsilon = 10**(-12+i)
        w = jax.ops.index_update(w, w<epsilon, epsilon)
        A_PD = v @ jnp.diag(w) @ v.T
        i += 1
    return A_PD

def isPD(A):
    try:
        B = cholesky(A, lower=True, check_finite=True)
        if jnp.isnan(jnp.sum(B)): raise Exception
    except:
        return False
    return True