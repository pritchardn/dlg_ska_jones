"""Python setup.py for dlg_ska_jones package"""
import io
import os

from setuptools import find_packages, setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("dlg_ska_jones", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [
        line.strip()
        for line in read(path).split("\n")
        if not line.startswith(('"', "#", "-", "git+"))
    ]


setup(
    name="dlg_ska_jones",
    version=read("dlg_ska_jones", "VERSION"),
    python_requires=">=3.0",
    description="Awesome dlg_ska_jones created by pritchardn",
    url="https://github.com/pritchardn/dlg_ska_jones/",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    author="pritchardn",
    packages=find_packages(exclude=["tests", ".github"]),
    install_requires=read_requirements("requirements.txt"),
    entry_points={
        "console_scripts": ["dlg_ska_jones = dlg_ska_jones.__main__:main"]
    },
    extras_require={"test": read_requirements("requirements-test.txt")},
)
