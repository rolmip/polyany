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

## :input_numbers: Evaluating polynomials

Polynomials can be easily evaluated in {{ polyany }} by treating the polynomial object as a callable function.
The input argument (`point`) must be a vector with `n_vars` components.

```pycon
>>> poly = Polynomial.univariate([1, 2, 3])
>>> poly([2]) #(1)!
np.float64(17.0)
```

1. Even for univariate polynomials, `point` needs to be a list, tuple or NumPy 1D-array.

For multivariate polynomials:

```pycon
>>> matrix = [[1, 2, 3],
...           [2, 4, 5],
...           [3, 5, 6]]
>>> poly = Polynomial.quadratic_form(matrix)
>>> poly([0, 0, 0])
np.float64(0.0)
>>> poly([1, 1, 1])
np.float64(31.0)
```

## :heavy_equals_sign: Comparing polynomials

In {{ polyany }}, Polynomial objects support **equality comparisons** (`==`) with other polynomials, but do not support **ordering comparisons** (`<`, `<=`, `>`, `>=`), which raises a `TypeError`.

Two polynomials are considered equal **if and only if** they have:

- [x] The same number of variables (`n_vars` attribute)
- [x] The same total degree (`degree` attribute)
- [x] The same coefficients (`coefficients` attribute)[^1]

[^1]:
    A comparison is made by using [`np.allclose()`](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html), which checks if two arrays are equal within a tolerance.

Comparison with other types (sequences, scalars, NumPy arrays) always returns [`NotImplemented`](https://docs.python.org/3/library/constants.html#NotImplemented).

???+ tip "Example"

    Internally, the polynomial object stores the coefficients and exponents in an **ordered way**, meaning that a polynomial is created **regardless of the order** of the input coefficients and exponents.

    ```py
    >>> poly1 = Polynomial([[0, 0], [1, 0], [0, 1]], [1, 2, 3])
    >>> poly2 = Polynomial([[0, 1], [0, 0], [1, 0]], [3, 1, 2])
    >>> poly1 == poly2
    True
    ```

## :scissors: Pruning

Pruning is the process of removing **empty monomials** of a polynomial. The
[`Polynomial`][polyany.Polynomial] object stores the exponents and coefficients
provided by the user in a **ordered way**.

```numpy
>>> poly = Polynomial([[0, 0], [0, 1], [1, 0], [1, 1]], [1, 0, 2, 0])
>>> poly.exponents
array([[0, 0],
       [1, 0],
       [0, 1],
       [1, 1]])
>>> poly.coefficients
array([1., 2., 0., 0.])
```

When a polynomial is pruned, all empty monomials are removed, that is, the entries in
`exponents` whose associated coefficients are  exactly zero, which have no effect
on the polynomial behavior.

To prune a polynomial, use the [`prune`][polyany.Polynomial.prune] method:

```numpy
>>> pruned = poly.prune()
>>> pruned.exponents
array([[0, 0],
       [1, 0]])
>>> pruned.coefficients
array([1., 2.])
```

The pruned polynomial retains only the first and second term, which are the
non-empty monomials.

## :heavy_plus_sign: Addition and subtraction

In {{ polyany }}, Polynomial objects can be added or subtracted with scalars[^2] and other polynomials.

```pycon
>>> poly = Polynomial.univariate([1, -2, 3])
>>> poly
1 - 2*x_1 + 3*x_1^2
>>> poly + 5
6 - 2*x_1 + 3*x_1^2
>>> poly - 1
-2*x_1 + 3*x_1^2
```

For addition/subtraction between polynomials:

```pycon
>>> another_poly = Polynomial([[0, 0], [1, 0], [0, 1], [1, 1]], [1, -2, 3, -4])
>>> another_poly
1 - 2*x_1 + 3*x_2 - 4*x_1*x_2
>>> poly + another_poly
2 - 4*x_1 + 3*x_2 + 3*x_1^2 - 4*x_1*x_2
>>> poly - another_poly
-3*x_2 + 3*x_1^2 + 4*x_1*x_2
```

[^2]:
    Python builtins numeric types (`int`, `float`) and NumPy scalars. See [`Scalar`][polyany.types.Scalar]

## :heavy_multiplication_x: Multiplication and division

Polynomials can be multiplied with other polynomials and scalars[^2].

```pycon
>>> poly = Polynomial.univariate([10, -20, 5])
>>> poly
10 - 20*x_1 + 5*x_1^2
>>> poly * 2
20 - 40*x_1 + 10*x_1^2
```

Multiplying two polynomials:

```pycon
>>> poly1 = Polynomial.univariate([1, -2, 3])
>>> poly1
1 - 2*x_1 + 3*x_1^2
>>> poly2 = Polynomial([[0, 0], [1, 1]], [3, 3])
>>> poly2
3 + 3*x_1*x_2
>>> poly1 * poly2
3 - 6*x_1 + 9*x_1^2 + 3*x_1*x_2 - 6*x_1^2*x_2 + 9*x_1^3*x_2
```

!!! warning "Division between polynomials"
    Currently, division can only be performed **between polynomials and scalars**.
    In the future, it is possible that division between polynomials will be supported.

Dividing a polynomial by a scalar:

```pycon
>>> poly = Polynomial.univariate([10, -20, 5])
>>> poly
10 - 20*x_1 + 5*x_1^2
>>> poly / 5
2 - 4*x_1 + x_1^2
```

## :curly_loop: Partial derivatives

The partial derivatives of polynomials can be evaluated by using the
[`partial`][polyany.Polynomial.partial] method. Let's consider the polynomial:

$$
P(\mathbf{x}) = 10 + 2\,x_1^2\,x_2\,x_3 + 5\,x_1\,x_2^3\,x_3^2
$$

This can be declared in {{ polyany }} as:

```pycon
>>> poly = Polynomial([[0, 0, 0], [2, 1, 1], [1, 3, 2]], [10, 2, 5])
```

Their first partial derivatives are:

$$
\begin{cases}
\displaystyle\frac{\partial P(\mathbf{x})}{\partial x_1} = 4\,x_1\,x_2\,x_3 + 5\,x_2^3\,x_3^2 \\[.5em]
\displaystyle\frac{\partial P(\mathbf{x})}{\partial x_2} = 2\,x_1^2\,x_3 + 15\,x_1\,x_2^2\,x_3^2 \\[.5em]
\displaystyle\frac{\partial P(\mathbf{x})}{\partial x_3} = 2\,x_1^2\,x_2 + 10\,x_1\,x_2^3\,x_3
\end{cases}
$$

which can be obtained in {{ polyany }} as:

```pycon
>>> poly.partial(0) #(1)!
4*x_1*x_2*x_3 + 5*x_2^3*x_3^2
>>> poly.partial(1)
2*x_1^2*x_3 + 15*x_1*x_2^2*x_3^2
>>> poly.partial(2)
2*x_1^2*x_2 + 10*x_1*x_2^3*x_3
```

1. The method [partial][polyany.Polynomial.partial] uses a zero-based index.
