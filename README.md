# dlg_ska_jones

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An example daliuge component exposing parameters derived from a notebook code.

## Installation

There are multiple options for the installation, depending on how you are intending to run the
DALiuGE engine, directly in a virtual environment (host) or inside a docker container. You can also
install it either from PyPI (latest released version).

## Install it from PyPI

### Engine in virtual environment

```bash
pip install dlg_ska_jones --extra-index-url=https://artefact.skao.int/repository/pypi-internal/simple
```

## Rascil data

You may need to install additional RASCIL data for successful execution. Follow
the [instructions](https://ska-telescope.gitlab.io/external/rascil/RASCIL_install.html) from the
SKAO. Make sure to set the RASCIL_DATA environment variable to the installed location during
execution.
