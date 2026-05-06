import numpy as np
if not hasattr(np.dtypes, 'StringDType'):
    np.dtypes.StringDType = type('StringDType', (object,), {}) # Dummy class

import jax
print("JAX imported successfully!")
