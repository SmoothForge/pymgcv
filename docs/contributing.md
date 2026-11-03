## Development setup
For development, fork the repo on github, then clone your fork:
```bash
git clone https://github.com/<your-username>/pymgcv.git
cd pymgcv
```

pymgcv needs both Python and R (with ``mgcv``), so the R side should be installed with
pixi or conda rather than pip. In both options below pymgcv is installed in
**editable** mode, so changes to the source in ``pymgcv/`` are used the next time the
package is imported (restart the Python session or notebook kernel to pick them up).
Reinstalling is only needed if the dependencies in ``pyproject.toml`` change.

### Option A: pixi (recommended, matches CI)
Install [pixi](https://pixi.sh), then from the repository root run
```bash
pixi shell -e dev  # creates and activates the development env
```
The editable install is configured in ``pyproject.toml``
(``[tool.pixi.pypi-dependencies]``), so no separate ``pip install`` is needed. Run
``pixi install -e dev`` to update the environment after changing dependencies.

### Option B: conda and pip
```bash
conda create --name pymgcv-dev -c conda-forge python=3.12 r-base r-mgcv rpy2
conda activate pymgcv-dev
python -m pip install -e ".[dev]"
```

### Check the install
```bash
python -c "import pymgcv; print(pymgcv.__file__)"
```
This should print a path inside your cloned repository, not ``site-packages``.

### Using your local version from another project
To try your changes from a separate project or notebook environment, install the
local copy in editable mode there, e.g.
```bash
python -m pip install -e /path/to/pymgcv
```
or, in a pixi project, add ``pymgcv = { path = "/path/to/pymgcv", editable = true }``
under ``[pypi-dependencies]`` (or ``[tool.pixi.pypi-dependencies]``). The environment
also needs ``r-base``, ``r-mgcv`` and ``rpy2`` from conda-forge.

## Making changes
Create a branch, make your changes, and verify that the tests pass by running:
```bash
pytest  # or: make test
```
This also runs the code examples in the docstrings and markdown files. Lint and format
the code with
```bash
ruff check .
ruff format .
```
then open a pull request against ``main``.

## Documentation
The documentation includes notebook examples in docs/examples. To rerun all these prior to
building the docs, ensure that the development environment is activated and run
```bash
pixi run notebooks
```

We can locally serve the docs using
```bash
mkdocs serve
```

The docs are built with ``mkdocs build`` (output in ``site/``). The ``Makefile`` wraps
both commands, running them in the pixi dev environment:
```bash
make docs        # mkdocs build
make docs-serve  # mkdocs serve
```
If you use another activated environment (e.g. conda), run ``make docs RUN=``.

## Adding code examples in docstrings
In order for the doctest to run properly, and documentation to be generated properly:

- Use the admonition `!!! example`, containing a python fenced code block, i.e.:
```
!!! example

    ```python
    # Insert runnable example here
    ```
```
- Do not use a prompt `>>> ` as this interferes with copying code blocks and looks cluttered!
- We generally do not test the output, just that each example runs (although you could use an assert if desired)
- Use a comment at the bottom if you want to show an output example
