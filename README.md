# pymgcv: Generalized Additive Models in Python

**pymgcv** provides a Python interface to R's powerful [mgcv](https://cran.r-project.org/web/packages/mgcv/index.html) library for fitting Generalized Additive Models (GAMs).

## 📚 Documentation
All the information and examples you need are in the **documentation**:

- [Check out the documentation here!](https://SmoothForge.github.io/pymgcv/)

If you like the project, please consider giving it a star on github!

## 🛠️ Development
To work on pymgcv locally, clone the repository and create the development environment
with [pixi](https://pixi.sh), which installs R, mgcv and an editable install of pymgcv:

```bash
git clone https://github.com/SmoothForge/pymgcv.git
cd pymgcv
pixi shell -e dev
pytest
```

See the [contributing guide](docs/contributing.md) for alternatives (e.g. conda + pip)
and more details.

