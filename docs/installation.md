---
title: PolyAny
---

## :package: How to install

Using [`uv`](https://docs.astral.sh/uv/):

```bash
$ uv pip install polyany
```

!!! note "To install uv, follow the instructions in the [official documentation](https://docs.astral.sh/uv/getting-started/installation/)."

Using pip:

```bash
$ pip install polyany
```

## :computer: How to contribute

This project uses `uv` to manage dependencies and environments.

1. Clone the repository

    ```bash
    $ git clone https://github.com/rolmip/polyany.git
    $ cd polyany
    ```

2. Sync the uv environment

    ```bash
    $ uv sync --all-groups
    ```

3. Activate the virtual environment

    === "POSIX"

        ```bash
        $ source .venv/bin/activate
        ```

    === "Windows cmd"

        ```bat
        C:\> .venv\Scripts\activate.bat
        ```

    === "Windows PowerShell"

        ```pwsh-session
        PS C:\> .venv\Scripts\Activate.ps1
        ```

4. Install git hooks

    ```bash
    $ pre-commit install
    ```

## :test_tube: Running tests

To run all tests, just use:

```bash
$ uv run pytest
```

To run all tests with coverage analysis:

```bash
$ uv run pytest --cov --cov-branch
```
