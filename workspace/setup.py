#!/usr/bin/env python3
"""
Setup script for UFRa Python package
"""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension
import pybind11
import cv2
import numpy as np

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "pyufra",
        [
            "python_bindings/src/python_bindings.cpp",
        ],
        include_dirs=[
            pybind11.get_cmake_dir(),
            "core/include",
            cv2.includes(),
            np.get_include(),
        ],
        libraries=["ufra_core", "opencv_core", "opencv_imgproc", "opencv_imgcodecs"],
        library_dirs=["build/lib"],
        language='c++'
    ),
]

setup(
    name="pyufra",
    version="1.0.0",
    author="MetaGPT Team",
    author_email="team@metagpt.com",
    description="Python bindings for Universal Face Re-Aging (UFRa)",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: C++",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-org/ufra/issues",
        "Source": "https://github.com/your-org/ufra",
        "Documentation": "https://docs.ufra.ai",
    },
)