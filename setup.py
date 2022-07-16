from distutils.command.install_data import install_data
from gettext import install
from http.client import PROXY_AUTHENTICATION_REQUIRED
from pickle import LONG_BINGET
from platform import python_branch
from re import A
from struct import pack
from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()
    
setup(
    name ="src",
    version = "0.0.1",
    author = "Developer Ashish",
    description = "A small package for dvc ml pipeline demo",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/pydeveloperashish/dvc-ml-mlops",
    author_email = "therobomarket@gmail.com",
    packages = ["src"],
    python_requires = ">=3.7",
    install_requires = [
        "dvc", "pandas", "scikit-learn", "joblib"
                       ]
    )    