!!! info "Documentation under construction"

## :pencil: Creating polynomials

In {{ polyany }} a polynomial can be created in many forms. The most flexible way is defining a matrix of `exponents` and a vector of `coefficients`.

### :one: From exponents and coefficients

The `exponents` matrix contains the exponents of each monomial of the polynomial, and the `coefficients` vector contains the corresponding scalars multipliers.

???+ tip "Basic example"

    Consider the monomial $M(\mathbf{x}) = 5\,x_1^2\,x_2\,x_3^4\,x_5$, its exponents and coefficient are given by:

    ```py
    exponents = [[2, 1, 4, 0, 1]] #(1)!
    coefficient = [5]
    ```

    1. When declaring the exponents matrix, variables are ordered increasingly, i.e., $[x_1,\,x_2,\,x_3,\,\dots,\,x_n]$.

    Note that the exponent matrix represents the **complete monomial**, meaning that it must **include all variables**, even those raised to the power of $0$.

???+ tip "Example"

    If we want to represent a polynomial, we declare the exponents and coefficients of each monomial. Consider:

    $$
    P(\mathbf{x}) = 5\,x_1^2\,x_2\,x_3^4\,x_5 + 3\,x_1\,x_2 + 4\,x_4^4\,x_5^3
    $$

    The monomials of $P(\mathbf{x})$ are:

    * $M_1(\mathbf{x}) = 3\,x_1\,x_2$
    * $M_2(\mathbf{x}) = 4\,x_4^4\,x_5^3$
    * $M_3(\mathbf{x}) = 5\,x_1^2\,x_2\,x_3^4\,x_5$

    To represent $P(\mathbf{x})$, we define:

    ```py
    exponents = [[1, 1, 0, 0, 0], [0, 0, 0, 4, 3], [2, 1, 4, 0, 1]]
    coefficients = [3, 4, 5]
    ```

    It is very important that the coefficients appear in the **same order** as their corresponding rows in the exponents matrix.

To create a polynomial from an exponents matrix and a coefficients vector in {{ polyany }}, use the following syntax:

```py
from polyany import Polynomial

exponents = [[0, 0, 0], [1, 0, 0], [0, 1, 0]] #(1)!
coefficients = [1, 2, 3] #(2)!

poly = Polynomial(exponents, coefficients)
```

1. The exponents matrix can also be a nested tuples or a NumPy 2D array
2. The coefficients vector can also be a tuple or a NumPy 1D array

The code above creates the following multivariate polynomial:

$$
P(\mathbf{x}) = 1 + 2\,x_1 + 3\,x_2
$$

### :two: Univariate Polynomials

??? definition "Univariate Polynomials"

    A univariate polynomial $P(x_1)$ is a polynomial that depends on a single variable $x_1$. An example of univariate polynomial is:

    $$
    P(x_1) = 2 x_1^2 + 3 x_1 + 4
    $$

To create univariate polynomials, a simpler syntax can be used, it only requires a `coefficients` vector.

```py
from polyany import Polynomial

coefficients = [1, 5, 8, 9]
univar_poly = Polynomial.univariate(coefficients)
```

The coefficients are related with power of the variable $x_1$ in increasing degree. The code above creates the polynomial:

$$
P(x_1) = 1 + 5\,x_1 + 8\,x_1^2 + 9\,x_1^3
$$
