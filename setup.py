"""Compatibility shim for older pip/setuptools (e.g. Ubuntu 18.04's system
Python) that don't understand PEP 621 `[project]` metadata in
pyproject.toml. Without this, an old setuptools calling bare `setup()`
can't find a name/version and installs a package literally called
"UNKNOWN".

This file can't just parse pyproject.toml itself either: reading TOML
needs either Python 3.11+'s stdlib `tomllib` or a third-party parser, and
neither is guaranteed to be available in the exact old/minimal environment
this shim exists for, before any dependencies are installed. So the
metadata below is a plain-Python mirror of pyproject.toml's `[project]`
table instead. pyproject.toml is still the source of truth for anyone on
a modern pip (PEP 517/660 installs never execute this file's `setup()`
call at all) -- if you change dependencies, version, entry points, etc.
there, update them here too.
"""
from pathlib import Path

from setuptools import find_packages, setup

long_description = (Path(__file__).parent / ".github" / "README.md").read_text()

setup(
    name="clearview",
    version="1.0.0",
    description="Deep learning framework for single image deraining",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Saumya Saksena",
    author_email="kumaar324@gmail.com",
    maintainer="Saumya Saksena",
    maintainer_email="kumaar324@gmail.com",
    url="https://github.com/dronefreak/clearview",
    project_urls={
        "Homepage": "https://github.com/dronefreak/clearview",
        "Repository": "https://github.com/dronefreak/clearview",
        "Bug Tracker": "https://github.com/dronefreak/clearview/issues",
    },
    license="Apache-2.0",
    python_requires=">=3.6",
    keywords=[
        "deep-learning",
        "computer-vision",
        "image-deraining",
        "rain-removal",
        "pytorch",
        "image-restoration",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    packages=find_packages(where=".", include=["clearview*"]),
    package_data={"clearview": ["py.typed"]},
    install_requires=[
        "torch",
        "torchvision",
        "numpy",
        "pillow",
        "matplotlib",
        "tqdm",
        "pyyaml",
        "lark",
        "opencv-python",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "ruff",
            "mypy",
            "pre-commit",
        ],
        "metrics": [
            "lpips",
            "piq",
            "scipy",
        ],
    },
    entry_points={
        "console_scripts": [
            "clearview-train = clearview.scripts.train:main",
            "clearview-evaluate = clearview.scripts.evaluate:main",
            "clearview-inference = clearview.scripts.inference:main",
        ],
    },
)
