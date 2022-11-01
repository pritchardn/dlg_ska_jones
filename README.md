# dlg_ska_jones

[![codecov](https://codecov.io/gh/pritchardn/dlg_ska_jones/branch/main/graph/badge.svg?token=dlg_ska_jones_token_here)](https://codecov.io/gh/pritchardn/dlg_ska_jones)
[![CI](https://github.com/pritchardn/dlg_ska_jones/actions/workflows/main.yml/badge.svg)](https://github.com/pritchardn/dlg_ska_jones/actions/workflows/main.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Awesome dlg_ska_jones created by pritchardn

## Installation

There are multiple options for the installation, depending on how you are intending to run the DALiuGE engine, directly in a virtual environment (host) or inside a docker container. You can also install it either from PyPI (latest released version).

## Install it from PyPI

### Engine in virtual environment
```bash
pip install dlg_ska_jones
```
This will only work after releasing the project to PyPi.
### Engine in Docker container
```bash
docker exec -t daliuge-engine bash -c 'pip install --prefix=$DLG_ROOT/code dlg_ska_jones'
```
## Usage
For example the MyComponent component will be available to the engine when you specify 
```
dlg_ska_jones.apps.MyAppDROP
```
in the AppClass field of a Python Branch component. The EAGLE palette associated with these components are also generated and can be loaded directly into EAGLE. In that case all the fields are correctly populated for the respective components.

