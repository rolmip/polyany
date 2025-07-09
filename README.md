<h1 align="center">
<img src="docs/assets/polyany_logo.png" alt="PolyAny Logo" height="200">
</h1><br>

![Static Badge](https://img.shields.io/badge/status-pre--alpha-orange)
[![codecov](https://codecov.io/gh/rolmip/polyany/graph/badge.svg?token=XMNXDY6AZ7)](https://codecov.io/gh/rolmip/polyany)
[![Tests](https://github.com/rolmip/polyany/actions/workflows/tests.yml/badge.svg)](https://github.com/rolmip/polyany/actions/workflows/tests.yml)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/rolmip/polyany/main.svg)](https://results.pre-commit.ci/latest/github/rolmip/polyany/main)

**A Python package for algebraic manipulation of multivariate polynomials.**

> 🚧 **This package is under active development.**
>
> It is not yet stable or ready for production use. <span style="color:red">Expect breaking changes!</span>

---

## ✨ Overview

PolyAny provides a flexible framework for representing and manipulating multivariate polynomials using structured, non-symbolic representations.

Unlike symbolic engines, PolyAny operates directly on the algebraic structure of polynomials (coefficients and exponents), enabling integration with numerical libraries and efficient structural transformations.

---

## 🔧 Features (planned)

- Polynomial creation from multiple formats (list, tuples, NumPy arrays, quadratic forms, ...)
- Support for multivariate expressions
- Algebraic operations: addition, multiplication, truncation, homogenization, ...
- Polynomial exporting into LaTeX code

---

## 📦 Installation

> ⚠️ Not yet available in PyPI.

This project uses `uv` to manage dependencies and environments.

> 🚀 To install uv, follow the instructions in the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

For local development:

```bash
git clone https://github.com/rolmip/polyany.git
cd polyany
uv sync
```

## 📄 License

This project is open-source and licensed under the BSD-3-Clause.

## 👥 Contributors

PolyAny is maintained by the **ROLMIP** developers:

* [Cristiano Agulhari](mailto:agulhari@utfpr.edu.br)
* [Esdras Battosti](mailto:esdras.2019@alunos.utfpr.edu.br)

## 🧪 Status

This repository is part of the early foundation of **RolmiPy**, a Python implementation of ROLMIP.
