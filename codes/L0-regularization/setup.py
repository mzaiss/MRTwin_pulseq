from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="l0_regularization",
    version="0.9",
    author="Kuan Lee",
    author_email="ulamaca.lee@gmail.com",
    description="A TensorFlow implementation of L0 regularization (Louizos et al 2017)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.tuebingen.mpg.de/mrolinek/L0-regularization",
    packages=find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)