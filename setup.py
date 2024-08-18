#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="GRASSEG",
    version="0.0.1",
    description="遥感草地语义分割,测量草地覆盖度",
    author="",
    author_email="",
    url="https://github.com/caixiaoshun/grasseg.git",
    install_requires=["lightning", "hydra-core"],
    packages=find_packages(),
    # use this to customize global commands available in the terminal after installing the package
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
