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

### :three: Quadratic forms

??? definition "Quadratic forms"

    A quadratic form is a second-degree homogeneous multivariate polynomial. It can be represented using a symmetric matrix $A \in \mathbb{R}^{n \times n}$ as:

    $$
    P(\mathbf{x}) = \sum_{i=1}^{n}\sum_{j=1}^{n} a_{ij} x_i x_j = \mathbf{x}^{\top}A\mathbf{x}
    $$

    For further reading, see [*Linear Algebra Done Right* by Sheldon Axler](https://linear.axler.net/)

To create a quadratic form a **square matrix** must be provided.

```py
from polyany import Polynomial

matrix = [[1, 2], [2, 3]] #(1)!
quadratic_form_poly = Polynomial.quadratic_form(matrix)
```

1. The input matrix can also be nested tuples or a NumPy 2D array

The code above creates the polynomial:

$$
P(\mathbf{x}) = x_1^2 + 4\,x_1\,x_2 + 3\,x_2^2
$$

!!! warning "Non-symmetric matrices"

    If the input matrix $A$ is not symmetric, a warning is raised and it's **symmetric part** $A_{\mathrm{sym}}$ is used instead, where

    $$
    A_{\mathrm{sym}} = \frac{1}{2} \left( A + A^{\top} \right)
    $$

    ??? example "Example"

        Suppose the input matrix is:

        $$
        A =
        \begin{bmatrix}
            1 & 6 \\
            0 & 2
        \end{bmatrix}
        $$

        Its symmetric part is:

        $$
        A_{\mathrm{sym}} =
        \begin{bmatrix}
            1 & 3 \\
            3 & 2
        \end{bmatrix}
        $$

        which corresponds to the polynomial:

        $$
        P(\mathbf{x}) = x_1^2 + 6\,x_1\,x_2 + 2\,x_2^2
        $$
