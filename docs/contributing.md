For development use the fork the repo on github, then run
```bash
git clone https://github.com/<your-username>/pymgcv.git
cd pymgcv
pixi shell -e dev  # activates the development env
```

Make changes, and verify that the tests pass by running:
```bash
pytest
```

### Test coverage
To run the tests with a coverage report for the ``pymgcv`` package, run
```bash
pixi run coverage
```
This prints the coverage of each module with the line numbers that are not run by any
test. Any extra arguments are passed on to ``pytest``, e.g.
```bash
pixi run coverage tests/test_gam.py      # coverage from a subset of the tests
pixi run coverage --cov-report=html      # browsable report in htmlcov/index.html
```
Note that the code examples in the docs and docstrings are run as tests too, so they
also count towards the coverage.

### Documentation
The documentation includes notebook examples in docs/examples. To rerun all these prior to
building the docs, ensure that the development environment is activated and run
```bash
pixi run notebooks
```

We can locally serve the docs using
```bash
mkdocs serve
```

### Adding code examples in docstrings
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
