from setuptools import find_packages, setup

setup(
    name="cuda-linear-attention",
    packages=find_packages(include=["cula", "cula.*"]),
)
