!!! info "Documentation under construction"

An example of PolyAny usage

```py title="example.py"
from polyany import Polynomial

exponents = [[0, 0, 0], [1, 0, 0], [0, 1, 0]] # (1)!
coefficients = [1, 2, 3]

poly = Polynomial(exponents, coefficients)
```

1. The exponents matrix can be a Numpy 2D-Array
