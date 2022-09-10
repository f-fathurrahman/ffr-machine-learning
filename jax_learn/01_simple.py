import jax
import jax.numpy as jnp

def sum_of_squares(x):
    return jnp.sum(x**2)

# Apply jax.grad to sum_of_squares to get a different function, namely
# the gradient of sum_of_squares w.r.t its first parameter x

D_sum_of_squares = jax.grad(sum_of_squares)

x = jnp.asarray([1.1, 1.2, 1.3, 1.4])

print("x = ", x)

print(sum_of_squares(x))

print(D_sum_of_squares(x))

