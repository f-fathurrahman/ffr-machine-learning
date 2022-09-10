import numpy as np

a = np.random.rand(2,2,2,2)

print( a[..., 0].flatten() )
print( a[:,:,:,0].flatten() )
