from setuptools import setup, find_packages

setup(
    name="hd2d",
    version="0.1",
    packages=find_packages(include=["hd2d_src", "hd2d_src.*"]), 
)

