import numpy as np
print(np.__version__)
try:
    print(np.dtypes.StringDType)
except Exception as e:
    print(e)
