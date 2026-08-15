"""Compatibility shim for older pip/setuptools (e.g. Ubuntu 18.04's system
Python) that don't support PEP 660 editable installs from pyproject.toml
alone. All actual package metadata lives in pyproject.toml; this just gives
old pip a setup.py to fall back to for `pip install -e .`.
"""
from setuptools import setup

setup()
