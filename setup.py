#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

# Read the contents of README.md
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt") as f:
    requirements = f.read().strip().splitlines()

# Get version from package
about = {}
with open(os.path.join("naganlp", "__init__.py"), "r", encoding="utf-8") as f:
    exec(f.read(), about)

setup(
    name="naganlp",
    version=about["__version__"],
    author="Agniva Maiti",
    author_email="maitiagniva@gmail.com",
    maintainer="Agniva Maiti",
    maintainer_email="maitiagniva@gmail.com",
    description="A Natural Language Processing toolkit for the Nagamese creole language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AgnivaMaiti/naga-nlp",
    project_urls={
        "Bug Reports": "https://github.com/AgnivaMaiti/naga-nlp/issues",
        "Source": "https://github.com/AgnivaMaiti/naga-nlp",
        "Documentation": "https://github.com/AgnivaMaiti/naga-nlp#readme",
        "Changelog": "https://github.com/AgnivaMaiti/naga-nlp/releases",
    },
    packages=find_packages(exclude=["tests*"]),
    package_data={
        "naganlp": [
            "py.typed",
            "*.model",
            "*.pkl",
            "*.conll",
            "data/*"
        ],
    },
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing",
        "Topic :: Text Processing :: Linguistic",
        "Typing :: Typed",
    ],
    keywords=[
        "nlp",
        "natural-language-processing",
        "computational-linguistics",
        "nagamese",
        "machine-learning",
        "artificial-intelligence",
        "deep-learning"
    ],
    license="MIT",
    zip_safe=False,

    # Enable setuptools_scm for versioning
    use_scm_version={
        "write_to": "naganlp/_version.py",
        "write_to_template": "__version__ = '{version}'",
    },
    setup_requires=["setuptools_scm"],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-mock>=3.10.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "isort>=5.12.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "naganlp=naganlp.cli:main",
        ],
    },
    zip_safe=False,
)
